"""E6: multi-input BSGS MAC for Holmes's LinearOperator (kernels/holmes_mac.cu).



Env: ``HOLMES_MULTI_MAC=1`` installs the patch (bench/t2_model.py worker), ``HOLMES_MULTI_MAC_TI`` = inputs per
thread (4 default, 2 or 1; 14 / 18 / 116 = one ciphertext component per block with 4 / 8 / 16 inputs per thread), ``HOLMES_MULTI_MAC_CHECK=1`` compares every chunk against the original path and raises
on a mismatch (slow; correctness runs only).  Applies to the ``normal`` hoist strategy with one baby block (the
production linear config: baby 64 / anchor 64 / 512 rows); anything else falls back to the original ``_mac_chunk``.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "kernels", "holmes_mac.cu")
_lib = None
_installed = False
_orig_mac_chunk = None
STATS = {"calls": 0, "fallbacks": 0, "inputs": 0, "checked": 0, "batched_giants": 0, "hoisted_giants": 0}
# HOLMES_MULTI_MAC_PROFILE=1: device time (CUDA events, one sync per chunk) of the fused MAC and of the batched giant
# step, keyed by "<limbs>l_<groups>g_<inputs>i" -> [calls, mac_ms, giant_ms]
PROFILE: dict = {}
_PROFILE = os.environ.get("HOLMES_MULTI_MAC_PROFILE", "0") == "1"


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
    so = os.path.join(_cache_dir(), f"libholmes_mac_{arch}_{tag}.so")
    if not os.path.exists(so):
        nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.run([nvcc, "-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC", "-o", tmp, _SRC], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    p, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    lib.holmes_mac_multi.argtypes = [p, p, p, ll, i, i, p, ll, i, i, i, p, p, p, p]
    lib.holmes_mac_multi.restype = i
    lib.holmes_mac_gather.argtypes = [p, p, p, ll, p, p, p, i, i, p, ll, i, i, p, p, p, p]
    lib.holmes_mac_gather.restype = i
    _lib = lib
    return lib


def _ti() -> int:
    return int(os.environ.get("HOLMES_MULTI_MAC_TI", "10108"))


def _layout(block, limbs, n):
    """(c0, c1, bstride) of a non-ext baby block [nb][>= limbs][N], or None when the layout is unexpected."""
    if len(block.cv) != 2 or bool(block.is_ext):
        return None
    c0, c1 = block.cv
    if c0.dim() != 3 or c1.dim() != 3 or c0.shape != c1.shape:
        return None
    if c0.size(0) != int(block.batch_size) or c0.size(1) < limbs or c0.size(2) != n:
        return None
    for c in (c0, c1):
        if c.stride(2) != 1 or c.stride(1) != n:
            return None
    if c0.stride(0) != c1.stride(0):
        return None
    return c0, c1, int(c0.stride(0))


def partial_sums(blocks, plaintexts, groups, ctx, *, ti=None):
    """Baby-step partial sums of every input: list of cipher batches [groups] (one per block in ``blocks``),
    equal to ``grouped_pairwise_mac(block, plaintexts, groups)`` of each block.  ``blocks`` are the inputs'
    (single) baby blocks at one level; ``plaintexts`` the chunk rows [groups * nb]."""
    raw = partial_sums_raw(blocks, plaintexts, groups, ctx, ti=ti)
    if raw is None:
        return None
    res, state = raw
    # res[c, :, index] is a strided [groups][limbs][N] view; EasyFHE's batch ops assume contiguous items
    return [block.cipher_like([res[0, :, index].contiguous(), res[1, :, index].contiguous()], state=state,
                              slots=plaintexts.slots, is_ext=False, batch_size=int(groups))
            for index, block in enumerate(blocks)]


def partial_sums_raw(blocks, plaintexts, groups, ctx, *, ti=None):
    """``partial_sums`` as one tensor: (res [2][groups][inputs][limbs][N], state) or None.  ``res[c, g]`` is a
    contiguous [inputs][limbs][N] batch (the giant step's batched key switch reads it without strides); at most
    32 inputs (kMaxInputs) per call."""
    import easyfhe as torch
    from easyfhe.fhe.ciphertext import CipherState

    blocks = tuple(blocks)
    n = int(ctx.N)
    limbs = int(blocks[0].state.cur_limbs)
    nb = int(blocks[0].batch_size)
    groups = int(groups)
    layouts = []
    for block in blocks:
        if int(block.state.cur_limbs) != limbs or int(block.batch_size) != nb or block.state != blocks[0].state:
            return None
        lay = _layout(block, limbs, n)
        if lay is None:
            return None
        layouts.append(lay)
    pv = plaintexts.cv[0]
    if int(plaintexts.state.cur_limbs) != limbs or bool(plaintexts.is_ext):
        return None
    if pv.dim() != 3 or pv.size(0) != groups * nb or pv.size(1) < limbs or pv.size(2) != n \
            or pv.stride(1) != n or pv.stride(2) != 1:
        return None
    if groups not in (1, 2, 4, 8, 16) or len(blocks) > 32:
        return None
    ti = _ti() if ti is None else int(ti)
    if groups == 16:
        ti = 10116   # the only tile that holds 16 giants per thread
    if ti not in (1, 2, 4, 404, 408, 802, 804, 1601, 1602, 10108, 10116, 10204, 10208, 10402, 10404):
        raise ValueError("HOLMES_MULTI_MAC_TI must be 1, 2, 4 (TI inputs per thread, all groups), TI * 100 + TG "
                         "(a TI x TG input x group tile): 404, 408, 802, 804, 1601, 1602, or 10000 + that for "
                         "lazy 128-bit accumulation: 10108, 10204, 10208, 10402, 10404")
    lib = load()
    stream = torch.cuda.current_stream().cuda_stream
    res = torch.empty((2, groups, len(blocks), limbs, n), dtype=torch.uint64, device=pv.device)
    ptr_t = ctypes.c_void_p * 32
    state = CipherState(limbs, blocks[0].state.scale_degree + plaintexts.state.scale_degree,
                        blocks[0].state.scaling_factor * plaintexts.state.scaling_factor)
    c0s = ptr_t(*[c0.data_ptr() for c0, _, _ in layouts])
    c1s = ptr_t(*[c1.data_ptr() for _, c1, _ in layouts])
    bstride = layouts[0][2]
    if any(bs != bstride for _, _, bs in layouts):
        return None
    code = lib.holmes_mac_multi(res.data_ptr(), c0s, c1s, bstride, nb, len(blocks), pv.data_ptr(), int(pv.stride(0)),
                              groups, ti, limbs, ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(),
                              ctx.barret_k.data_ptr(), stream)
    if code:
        raise RuntimeError(f"holmes_mac_multi failed with code {code}")
    STATS["calls"] += 1
    STATS["inputs"] += len(blocks)
    return res, state


def _giant_step(partial, giant_offset, giant_count, ctx, strategy):
    from easyfhe.fhe.ops import rotation
    from easyfhe.fhe.ops.layout import cipher_batch_item

    if int(giant_count) == 1:
        result = cipher_batch_item(partial, 0)
        return rotation.moddown_from_ext(result, ctx) if result.is_ext else result
    if int(giant_offset) == 0:
        return rotation._sum_batch_items_without_rotation(partial, ctx)
    return rotation.giant_rotate_sum(partial, giant_offset, ctx, strategy=strategy)


def _sylph_keyswitch():
    """Sylph's fused key-switch module when it is installed (HOLMES_SYLPH_KERNELS=1), else None."""
    if os.environ.get("HOLMES_MULTI_GIANT", "1") == "0":
        return None
    try:
        from sylph_easy.easyfhe_ext import keyswitch
    except ImportError:
        return None
    return keyswitch if keyswitch.enabled() else None


def _rotate_batch(ks, c0, c1, offset, limbs, ctx, add0=None, add1=None):
    """rot((c0, c1), offset) [+ (add0, add1)] for a batch [B, limbs, N] (one key, Sylph ``rotate_pair``)."""
    if ks.P1():
        digits = ks.modup_p1(c1, limbs, ctx)
        return ks.rotate_pair(c0, digits, int(offset), limbs, ctx, add0, add1, own=c1)
    digits = ks.modup(c1, limbs, ctx)
    return ks.rotate_pair(c0, digits, int(offset), limbs, ctx, add0, add1)


def giant_step_batched(res, giant_offset, giant_count, ctx, ks, *, base_offset=0, add0=None, add1=None):
    """Horner giant step of all inputs at once: ``res`` [2][G][inputs][limbs][N] (``partial_sums_raw``) ->
    (c0, c1) [inputs][limbs][N] = rot(sum_g rot^g(partial[g]), base_offset) + (add0, add1).  Every Horner step
    rotates the 16 inputs' accumulators by the same key in one batched key switch instead of 16 single-cipher
    rotations (``giant_rotate_sum``'s normal strategy); same message, residues differ by ModDown rounding.
    ``base_offset`` is the
    chunk's giant_base rotation (0 = none); the addend (the running accumulators) is folded into that rotation."""
    _, groups, inputs, limbs, _ = res.shape
    limbs = int(limbs)
    cur0, cur1 = res[0, groups - 1], res[1, groups - 1]
    for g in range(groups - 2, -1, -1):
        cur0, cur1 = _rotate_batch(ks, cur0, cur1, giant_offset, limbs, ctx, res[0, g], res[1, g])
    if int(base_offset):
        cur0, cur1 = _rotate_batch(ks, cur0, cur1, base_offset, limbs, ctx, add0, add1)
    return cur0, cur1


def hoist_enabled() -> bool:
    """HOLMES_MULTI_GIANT_HOIST=1: double-hoisted giant step (needs the keys g * giant_offset, g < chunk giants)."""
    return os.environ.get("HOLMES_MULTI_GIANT_HOIST", "0") == "1"


def hoist_key_offsets(giant_offset: int, chunk_giants: int) -> tuple[int, ...]:
    """Rotation keys :func:`giant_step_hoisted` needs beyond the Horner chain's (``giant_offset`` and the chunk
    giant-base multiples): ``g * giant_offset`` for ``2 <= g < chunk_giants``."""
    return tuple(g * int(giant_offset) for g in range(2, int(chunk_giants)))


_zero_cache: dict = {}


def _zeros(shape, device):
    import easyfhe as torch

    key = (tuple(int(v) for v in shape), str(device))
    z = _zero_cache.get(key)
    if z is None:
        if len(_zero_cache) >= 4:
            _zero_cache.clear()
        z = _zero_cache[key] = torch.zeros(key[0], dtype=torch.uint64, device=device)
    return z


def giant_step_hoisted(res, giant_offset, giant_count, ctx, ks, *, base_offset=0, add0=None, add1=None,
                       max_keys=None):
    """Double-hoisted giant step of all inputs at once (Sylph ``giant_sum_ext``): the ``G - 1`` rotated partial sums
    sum_{g>=1} rot(partial[g], g * giant_offset) are ModUp'ed and multiplied by their own key in one launch and
    ModDown'ed once per input (the Horner chain of :func:`giant_step_batched` ModDowns ``G - 1`` times); partial[0]
    is added in that ModDown.  Then the chunk giant-base rotation (``base_offset``, one more key switch) with the
    accumulators ``(add0, add1)`` folded in, exactly as the Horner path.  ``max_keys`` bounds the ModUp digits held
    at once (``(G - 1) * inputs * beta * (limbs + K) * N`` words for all keys: 2.2 GB at 7 x 16 x 11 limbs); the
    remaining keys are summed into the same extended accumulator in a second launch.  Returns (c0, c1)
    [inputs][limbs][N].  Same message as the Horner chain; residues differ by ModDown rounding."""
    import easyfhe as torch

    from sylph_easy.easyfhe_ext import bsgs

    _, groups, inputs, limbs, n = res.shape
    groups, inputs, limbs, n = int(groups), int(inputs), int(limbs), int(n)
    k, L, alpha = int(ctx.K), int(ctx.L), int(ctx.alpha)
    active = limbs + k
    beta = (limbs + alpha - 1) // alpha
    if max_keys is None:
        max_keys = int(os.environ.get("HOLMES_GIANT_HOIST_KEYS", "7"))
    lib = bsgs._lib()
    stream = torch.cuda.current_stream().cuda_stream
    acc = _zeros((2, inputs, active, n), res.device)
    out = None
    g = 1
    while g < groups:
        g_end = min(groups, g + max(1, int(max_keys)))
        nk = g_end - g
        c0 = res[0, g:g_end].reshape(nk * inputs, limbs, n)
        c1 = res[1, g:g_end].reshape(nk * inputs, limbs, n)
        if not (c0.is_contiguous() and c1.is_contiguous()):
            raise ValueError("partial sums must be contiguous")
        digits = ks.modup(c1, limbs, ctx)
        offsets = tuple(gg * int(giant_offset) for gg in range(g, g_end))
        table = ks._fast_table(ctx, offsets, limbs)
        nxt = torch.empty((2, inputs, active, n), dtype=torch.uint64, device=res.device)
        from sylph_easy.easyfhe_ext import cudalib
        cudalib.check(lib.sylph_giant_sum_ext(
            nxt[0].data_ptr(), nxt[1].data_ptr(), acc[0].data_ptr(), acc[1].data_ptr(), digits.data_ptr(),
            beta * active * n, c0.data_ptr(), limbs * n, table.data_ptr(), nk, inputs, limbs, k, L, beta,
            ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(), ctx.barret_k.data_ptr(), ctx.PModq.data_ptr(),
            stream), "giant_sum_ext")
        del digits
        acc = nxt
        g = g_end
    cur0, cur1 = ks.moddown_pair(acc[0], acc[1], limbs, ctx, res[0, 0], res[1, 0])
    del acc
    if int(base_offset):
        cur0, cur1 = _rotate_batch(ks, cur0, cur1, base_offset, limbs, ctx, add0, add1)
    return cur0, cur1


def _item2d(t):
    return t if t.dim() == 2 else t[0]


def mac_chunk_multi(self, chunk, plaintexts, input_ciphers, accumulators, cached_babies, *, plan, strategy,
                    baby_anchor_step):
    """Drop-in for ``LinearOperator._mac_chunk``: one fused MAC for every input, then the giant step (batched
    over the inputs with Sylph's key switch when available, else per input)."""
    import easyfhe as torch
    from easyfhe import fhe

    from llama3fhe.backend import release_if_supported

    ctx = self.crypto_context.context
    usable = (
        str(strategy) == "normal"
        and bool(self.config.reuse_baby_rotations)
        and all(b is not None and len(b) == 1 for b in cached_babies)
        and len(plan.baby_blocks) == 1
    )
    raw = None
    if usable:
        if _PROFILE:
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True); ev0.record()
        raw = partial_sums_raw([b[0] for b in cached_babies], plaintexts, int(chunk.giant_count), ctx)
        if _PROFILE and raw is not None:
            ev1.record()
    if raw is None:
        STATS["fallbacks"] += 1
        return _orig_mac_chunk(self, chunk, plaintexts, input_ciphers, accumulators, cached_babies, plan=plan,
                               strategy=strategy, baby_anchor_step=baby_anchor_step)
    res, state = raw
    groups = int(chunk.giant_count)
    check = os.environ.get("HOLMES_MULTI_MAC_CHECK", "0") == "1"
    template = cached_babies[0][0]

    def partial(index):
        return template.cipher_like([res[0, :, index].contiguous(), res[1, :, index].contiguous()], state=state,
                                    slots=plaintexts.slots, is_ext=False, batch_size=groups)

    ks = _sylph_keyswitch()
    if ks is not None and groups > 1 and int(plan.giant_offset) != 0:
        base = int(chunk.giant_base) * int(plan.giant_offset)
        fused_add = base != 0 and all(a is not None for a in accumulators)
        add0 = add1 = None
        if fused_add:
            add0 = torch.stack([_item2d(a.cv[0]) for a in accumulators], 0)
            add1 = torch.stack([_item2d(a.cv[1]) for a in accumulators], 0)
        if hoist_enabled():
            out0, out1 = giant_step_hoisted(res, int(plan.giant_offset), groups, ctx, ks, base_offset=base,
                                            add0=add0, add1=add1)
            STATS["hoisted_giants"] += 1
        else:
            out0, out1 = giant_step_batched(res, int(plan.giant_offset), groups, ctx, ks, base_offset=base,
                                            add0=add0, add1=add1)
        STATS["batched_giants"] += 1
        if _PROFILE:
            ev2 = torch.cuda.Event(enable_timing=True); ev2.record(); ev2.synchronize()
            key = f"{int(res.shape[3])}l_{groups}g_{len(input_ciphers)}i" + ("_h" if hoist_enabled() else "")
            entry = PROFILE.setdefault(key, [0, 0.0, 0.0])
            entry[0] += 1; entry[1] += ev0.elapsed_time(ev1); entry[2] += ev1.elapsed_time(ev2)
        for index, cipher in enumerate(input_ciphers):
            result = template.cipher_like([out0[index], out1[index]], state=state, slots=plaintexts.slots,
                                          is_ext=False, batch_size=1)
            if check:
                # the partial sums are bit-exact; the batched key switch rounds its ModDown differently from
                # EasyFHE's single-cipher ``cv_hrot`` (same message, ~1e-12 apart after decoding: debug_giant.py),
                # so the giant step is validated at the decoded level (micro_multi_mac.py --sylph) and by the
                # layer's final_hidden_error, not residue by residue.
                _check_partial(partial(index), cipher, plan, plaintexts, chunk, ctx, strategy, baby_anchor_step,
                               cached_babies[index])
            current = accumulators[index]
            if current is None or fused_add:
                if current is not None:
                    release_if_supported(current)
                accumulators[index] = result
            else:
                combined = fhe.homo_add(current, result, ctx)
                release_if_supported(current)
                accumulators[index] = combined
        return

    for index, cipher in enumerate(input_ciphers):
        result = None
        try:
            if check:
                _check_partial(partial(index), cipher, plan, plaintexts, chunk, ctx, strategy, baby_anchor_step,
                               cached_babies[index])
            result = _giant_step(partial(index), plan.giant_offset, groups, ctx, strategy)
            if int(chunk.giant_base):
                shifted = fhe.homo_rotate(result, int(chunk.giant_base) * plan.giant_offset, ctx)
                release_if_supported(result)
                result = shifted
            current = accumulators[index]
            if current is None:
                accumulators[index] = result
                result = None
            else:
                combined = fhe.homo_add(current, result, ctx)
                release_if_supported(current)
                release_if_supported(result)
                accumulators[index] = combined
                result = None
        finally:
            release_if_supported(result)


def count_diff(a, b) -> int:
    """Number of differing uint64 residues (EasyFHE disables tensor comparison operators; compare on the host)."""
    import numpy as np

    x = a.detach().cpu().numpy() if hasattr(a, "numpy") else np.asarray(a.cpu().tolist(), dtype=np.uint64)
    y = b.detach().cpu().numpy() if hasattr(b, "numpy") else np.asarray(b.cpu().tolist(), dtype=np.uint64)
    return int(np.count_nonzero(x != y))


def _check_partial(partial, cipher, plan, plaintexts, chunk, ctx, strategy, baby_anchor_step, baby_rotations):
    import easyfhe as torch
    from easyfhe.fhe.ops import rotation

    baby_offsets, specs = rotation._baby_block_specs(plan.baby_offsets, baby_anchor_step)
    ref = rotation._grouped_mac_baby_blocks(baby_rotations, plaintexts, int(chunk.giant_count), ctx,
                                            baby_count=len(baby_offsets), specs=specs)
    for c in range(2):
        a, b = partial.cv[c], ref.cv[c]
        if a.shape != b.shape:
            raise RuntimeError(f"multi_mac check: shape {tuple(a.shape)} != {tuple(b.shape)}")
        bad = count_diff(a, b)
        if bad:
            raise RuntimeError(f"multi_mac check: component {c} differs in {bad} residues")
    STATS["checked"] += 1


def install() -> dict:
    global _installed, _orig_mac_chunk
    from llama3fhe.operators.linear.operator import LinearOperator

    if not _installed:
        _orig_mac_chunk = LinearOperator._mac_chunk
        LinearOperator._mac_chunk = mac_chunk_multi
        _installed = True
    return {"multi_mac": True, "ti": _ti(), "check": os.environ.get("HOLMES_MULTI_MAC_CHECK", "0") == "1",
            "batched_giant": _sylph_keyswitch() is not None, "hoisted_giant": hoist_enabled(),
            "lib": load() is not None}


def enabled() -> bool:
    return _installed
