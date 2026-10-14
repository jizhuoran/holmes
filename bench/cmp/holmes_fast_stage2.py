"""E8: fused encode stage 2 (``HOLMES_FAST_STAGE2=1``): kernels/holmes_encode.cu replaces EasyFHE's ``cv_encode``."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "kernels", "holmes_encode.cu")
_lib = None
_installed = False
_orig_cv_encode = None
STATS = {"calls": 0, "fallbacks": 0, "rows": 0, "checked": 0}


def _cache_dir() -> str:
    return os.environ.get("HOLMES_MAC_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "holmes_mac"))


def load():
    global _lib
    if _lib is not None:
        return _lib
    with open(_SRC, "rb") as f:
        tag = hashlib.sha1(f.read()).hexdigest()[:12]
    arch = os.environ.get("SYLPH_CUDA_ARCH", "sm_80")
    os.makedirs(_cache_dir(), exist_ok=True)
    so = os.path.join(_cache_dir(), f"libholmes_encode_{arch}_{tag}.so")
    if not os.path.exists(so):
        nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.run([nvcc, "-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC", "-o", tmp, _SRC], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    p, i, ll, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_double
    lib.holmes_encode_stage2.argtypes = [p, p, ll, i, d, i, p, p, p, p, p, p, p, i, i, i]
    lib.holmes_encode_stage2.restype = i
    _lib = lib
    return lib


# HOLMES_FAST_STAGE2_BATCH_MB: rows per kernel A / kernel B launch pair, sized so the intermediate of one pair
# (rows x limbs x 512 KB) stays in L2 between the two kernels (0: one launch pair for the whole tensor).
_BATCH_MB = float(os.environ.get("HOLMES_FAST_STAGE2_BATCH_MB", "0") or 0)
# HOLMES_FAST_STAGE2_PIPE=1: with batching, kernel A of the next batch runs on a side stream during kernel B
_PARTS = 3 | (4 if os.environ.get("HOLMES_FAST_STAGE2_PIPE", "0") == "1" else 0)
# HOLMES_FAST_STAGE2_LPB: limbs per kernel A block (0 = all limbs in one block; 1 = one block per limb, for small batches)
_LPB = int(os.environ.get("HOLMES_FAST_STAGE2_LPB", "0") or 0)


def rows_per_launch(limbs: int, batch_mb: float | None = None) -> int:
    mb = _BATCH_MB if batch_mb is None else float(batch_mb)
    if mb <= 0:
        return 0
    return max(1, int(mb * (1 << 20) // (int(limbs) * (1 << 16) * 8)))


def encode_stage2_tensor(encoded_values, cur_limbs, scaling_factor, context, batch_mb=None, parts=None, lpb=None):
    """[rows][cur_limbs][N] uint64 NTT plaintext of ``encoded_values`` ([rows][N] float64, device); None if unsupported."""
    import easyfhe as torch

    n = int(context.N)
    if n != (1 << 16) or not encoded_values.is_cuda or encoded_values.dtype != torch.float64:
        return None
    if encoded_values.dim() != 2 or int(encoded_values.shape[1]) != n:
        return None
    if encoded_values.stride(1) != 1:
        encoded_values = encoded_values.contiguous()
    rows = int(encoded_values.shape[0])
    limbs = int(cur_limbs)
    lib = load()
    out = torch.empty((rows, limbs, n), dtype=torch.uint64, device=encoded_values.device)
    primes = context.QplusP_map[limbs]
    diffs = context.QmaxdiffplusPmaxdiff_map[limbs]
    ratios = context.QbarretRatioplusPbarretRatio_map[limbs]
    ks = context.QbarretKplusPbarretK_map[limbs]
    rc = lib.holmes_encode_stage2(
        out.data_ptr(), encoded_values.data_ptr(), int(encoded_values.stride(0)), rows, float(scaling_factor), limbs,
        primes.data_ptr(), diffs.data_ptr(), ratios.data_ptr(), ks.data_ptr(),
        context.power_of_roots.data_ptr(), context.power_of_roots_shoup.data_ptr(),
        torch.cuda.current_stream().cuda_stream, rows_per_launch(limbs, batch_mb), int(_PARTS if parts is None else parts),
        int(_LPB if lpb is None else lpb))
    if rc != 0:
        raise RuntimeError(f"holmes_encode_stage2 failed: cuda error {rc}")
    return out


def count_diff(a, b) -> int:
    """Number of differing uint64 words (EasyFHE disables tensor comparison operators; compare on the host)."""
    import numpy as np

    return int(np.count_nonzero(a.detach().cpu().numpy() != b.detach().cpu().numpy()))


def install() -> dict:
    global _installed, _orig_cv_encode
    if _installed:
        return {"fast_stage2": True}
    import easyfhe as torch
    from easyfhe.fhe.ops import kernels

    orig = kernels.cv_encode
    _orig_cv_encode = orig
    check = os.environ.get("HOLMES_FAST_STAGE2_CHECK", "0") == "1"

    def cv_encode(input, N, cur_limbs, slots, scaling_factor, is_ext, context):
        if bool(is_ext) or int(slots) * 2 != int(N):
            STATS["fallbacks"] += 1
            return orig(input, N, cur_limbs, slots, scaling_factor, is_ext, context)
        values = input if input.dim() == 2 else input.reshape(1, -1)
        out = encode_stage2_tensor(values, cur_limbs, scaling_factor, context)
        if out is None:
            STATS["fallbacks"] += 1
            return orig(input, N, cur_limbs, slots, scaling_factor, is_ext, context)
        STATS["calls"] += 1
        STATS["rows"] += int(values.shape[0])
        if check:
            ref = orig(input, N, cur_limbs, slots, scaling_factor, is_ext, context)
            bad = count_diff(ref, out) if ref.shape == out.shape else -1
            if bad != 0:
                raise RuntimeError(f"HOLMES_FAST_STAGE2 mismatch: shape {tuple(ref.shape)} vs {tuple(out.shape)}, {bad} words differ")
            STATS["checked"] += 1
        return out

    kernels.cv_encode = cv_encode
    _installed = True
    load()
    return {"fast_stage2": True, "check": check}
