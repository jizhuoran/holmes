"""Fuse online QuaRot Hadamard into exported o_proj/down_proj weights."""

from __future__ import annotations

import numpy as np

from .numpy import (
    fwht_last_dim,
    matmul_had_u,
    online_full_had_down,
    online_partial_had_o,
)

__all__ = [
    "apply_full_fuse_to_state_dict",
    "fuse_online_hadamard_into_down_proj",
    "fuse_online_hadamard_into_o_proj",
    "verify_down_proj_fuse_numpy",
    "verify_o_proj_fuse_numpy",
]


def fuse_online_hadamard_into_o_proj(
    weight_o: np.ndarray,
    *,
    num_heads: int = 32,
    had_dim: int = 128,
) -> np.ndarray:
    """Fuse ``(H_{n_h} ⊗ I)`` into ``o_proj`` weights (PyTorch ``[out, in]`` layout)."""

    d = int(num_heads) * int(had_dim)
    weight_o = np.asarray(weight_o, dtype=np.float32).reshape(d, d)
    weight_reshaped = weight_o.reshape(d, int(num_heads), int(had_dim))
    weight_reshaped = weight_reshaped.transpose(0, 2, 1)
    weight_reshaped = fwht_last_dim(weight_reshaped, scale=1.0 / np.sqrt(int(num_heads)))
    weight_reshaped = weight_reshaped.transpose(0, 2, 1)
    return weight_reshaped.reshape(d, d).astype(np.float32, copy=False)


def fuse_online_hadamard_into_down_proj(
    weight_down: np.ndarray,
    *,
    intermediate_size: int = 14336,
) -> np.ndarray:
    """Fuse full ``H_ffn`` into ``down_proj`` weights."""

    weight_down = np.asarray(weight_down, dtype=np.float32)
    if int(weight_down.shape[-1]) != int(intermediate_size):
        raise ValueError(
            f"down_proj expects intermediate_size={intermediate_size}, got {weight_down.shape[-1]}."
        )
    return matmul_had_u(weight_down).astype(np.float32, copy=False)


def apply_full_fuse_to_state_dict(
    state: dict[str, np.ndarray],
    *,
    num_layers: int = 32,
    num_heads: int = 32,
    had_dim: int = 128,
    intermediate_size: int = 14336,
) -> dict[str, np.ndarray]:
    """Return a new state dict with online Hadamard fused into every layer's o/down proj."""

    fused = dict(state)
    for layer_idx in range(int(num_layers)):
        o_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        if o_key in fused:
            fused[o_key] = fuse_online_hadamard_into_o_proj(
                fused[o_key],
                num_heads=int(num_heads),
                had_dim=int(had_dim),
            )
        if down_key in fused:
            fused[down_key] = fuse_online_hadamard_into_down_proj(
                fused[down_key],
                intermediate_size=int(intermediate_size),
            )
    return fused


def verify_o_proj_fuse_numpy(
    weight_o: np.ndarray,
    activation: np.ndarray,
    *,
    num_heads: int = 32,
    had_dim: int = 128,
    atol: float = 5e-4,
) -> dict[str, float | bool]:
    weight_o = np.asarray(weight_o, dtype=np.float32)
    activation = np.asarray(activation, dtype=np.float32)
    weight_fused = fuse_online_hadamard_into_o_proj(weight_o, num_heads=num_heads, had_dim=had_dim)
    z_rot = online_partial_had_o(activation, num_heads=num_heads, had_dim=had_dim)
    diff = np.abs((z_rot @ weight_o.T).astype(np.float64) - (activation @ weight_fused.T).astype(np.float64))
    max_abs = float(diff.max())
    return {"max_abs_diff": max_abs, "passed": bool(max_abs <= atol), "atol": float(atol)}


def verify_down_proj_fuse_numpy(
    weight_down: np.ndarray,
    activation: np.ndarray,
    *,
    intermediate_size: int = 14336,
    atol: float = 5e-4,
) -> dict[str, float | bool]:
    weight_down = np.asarray(weight_down, dtype=np.float32)
    activation = np.asarray(activation, dtype=np.float32)
    weight_fused = fuse_online_hadamard_into_down_proj(
        weight_down,
        intermediate_size=intermediate_size,
    )
    f_rot = online_full_had_down(activation, intermediate_size=intermediate_size)
    diff = np.abs((f_rot @ weight_down.T).astype(np.float64) - (activation @ weight_fused.T).astype(np.float64))
    max_abs = float(diff.max())
    return {"max_abs_diff": max_abs, "passed": bool(max_abs <= atol), "atol": float(atol)}
