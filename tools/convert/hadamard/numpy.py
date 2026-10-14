from __future__ import annotations

import numpy as np

from .had_k28_data import HAD_K28


def _is_pow2(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0


def _load_had_k28() -> np.ndarray:
    """Return the vendored 28×28 Hadamard block (14336 = 28 × 512)."""

    matrix = np.asarray(HAD_K28, dtype=np.float32)
    if matrix.shape != (28, 28):
        raise ValueError(f"had_k28 must be shaped (28, 28), got {matrix.shape}.")
    return np.asarray(matrix, dtype=np.float32)


def get_had_k(n: int, *, transpose: bool = False) -> tuple[np.ndarray | None, int]:
    """Return ``(hadK, K)`` matching QuaRot ``hadamard_utils.get_hadK`` for Llama-relevant sizes.

    For ``n = K * 2^k`` with nontrivial ``K``, ``hadK`` is a fixed ``K×K`` matrix and the
    remaining factor is handled by a Walsh FWHT. When ``K == 1``, ``hadK`` is ``None``.
    """

    n = int(n)
    if n <= 0:
        raise ValueError(f"Hadamard size must be positive, got {n}.")

    # Llama-3-8B intermediate (14336 = 28 * 512).
    if n % 28 == 0 and _is_pow2(n // 28):
        had_k = _load_had_k28()
        if transpose:
            had_k = had_k.T
        return had_k, 28

    # Other QuaRot-supported composite sizes (not needed for Llama3-8B layer0, kept for parity).
    composite_sizes = (172, 156, 140, 108, 60, 52, 36, 40, 20, 12)
    for block in composite_sizes:
        if n % block == 0 and _is_pow2(n // block):
            raise NotImplementedError(
                f"Hadamard size n={n} requires K={block}, but only K=28 is bundled. "
                "Add the corresponding hadK block or clone QuaRot into tools/third_party."
            )

    if _is_pow2(n):
        return None, 1

    raise ValueError(f"Unsupported Hadamard dimension n={n} for get_had_k.")


def fwht_last_dim(values: np.ndarray, *, scale: float) -> np.ndarray:
    """Fast Walsh–Hadamard on the last dimension (last dim must be a power of two)."""

    output = np.array(values, dtype=np.float64, copy=True)
    last = int(output.shape[-1])
    if not _is_pow2(last):
        raise ValueError(f"FWHT last dimension must be a power of two, got {last}.")

    stride = 1
    while stride < last:
        pairs = last // (2 * stride)
        view = output.reshape(*output.shape[:-1], pairs, 2, stride)
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        stride *= 2

    output *= float(scale)
    return output


def matmul_had_u(values: np.ndarray, *, transpose: bool = False) -> np.ndarray:
    """Apply exact Hadamard on the last dimension (QuaRot ``matmul_hadU_cuda`` NumPy twin)."""

    values = np.asarray(values, dtype=np.float32)
    n = int(values.shape[-1])
    had_k, k = get_had_k(n, transpose=transpose)
    original_shape = values.shape

    if k == 1:
        return fwht_last_dim(values, scale=1.0 / np.sqrt(n)).astype(np.float32, copy=False)

    # Reshape to [..., K, n//K], FWHT on the Walsh block, then multiply by hadK on the K axis.
    reshaped = values.reshape(*original_shape[:-1], k, n // k).astype(np.float64, copy=False)
    reshaped = fwht_last_dim(reshaped, scale=1.0 / np.sqrt(n))
    transformed = np.einsum("ij,...jk->...ik", had_k.astype(np.float64), reshaped)
    return transformed.reshape(original_shape).astype(np.float32, copy=False)


def online_full_had_down(
    values: np.ndarray,
    *,
    intermediate_size: int | None = None,
) -> np.ndarray:
    """Online full Hadamard on FFN activations before ``down_proj`` (QuaRot ``online_full_had``)."""

    values = np.asarray(values, dtype=np.float32)
    if intermediate_size is not None and int(values.shape[-1]) != int(intermediate_size):
        raise ValueError(
            f"online_full_had_down expects width={intermediate_size}, got {values.shape[-1]}."
        )
    return matmul_had_u(values)


def online_partial_had_o(
    values: np.ndarray,
    *,
    num_heads: int = 32,
    had_dim: int = 128,
) -> np.ndarray:
    """Online partial Hadamard on attention output before ``o_proj`` (QuaRot ``online_partial_had``).

    For Llama3-8B: hidden=4096, num_heads=32, had_dim=128 → apply H on the head axis (32).
    """

    values = np.asarray(values, dtype=np.float32)
    init_shape = values.shape
    if init_shape[-1] != int(num_heads) * int(had_dim):
        raise ValueError(
            f"online_partial_had_o expects last dim {num_heads * had_dim}, got {init_shape[-1]}."
        )

    had_k, k = get_had_k(int(num_heads))
    head_count = int(init_shape[-1] // int(had_dim))
    scale = 1.0 / np.sqrt(head_count)

    if k == 1:
        # Match ActQuantWrapper: reshape (batch, heads, had_dim) → (batch, had_dim, heads), FWHT on heads.
        reshaped = values.reshape(-1, head_count, int(had_dim)).astype(np.float64, copy=False)
        transformed = fwht_last_dim(reshaped.transpose(0, 2, 1), scale=scale).transpose(0, 2, 1)
        return transformed.reshape(init_shape).astype(np.float32, copy=False)

    # K != 1 path (not used for Llama3-8B o_proj, kept for parity with official code).
    reshaped = values.reshape(-1, head_count, int(had_dim)).astype(np.float64, copy=False)
    transformed = (had_k.astype(np.float64) @ reshaped) / scale
    return transformed.reshape(init_shape).astype(np.float32, copy=False)
