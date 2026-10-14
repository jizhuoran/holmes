"""Offline QuaRot Hadamard transforms used by weight conversion."""

from .fuse import (
    apply_full_fuse_to_state_dict,
    fuse_online_hadamard_into_down_proj,
    fuse_online_hadamard_into_o_proj,
)

__all__ = [
    "apply_full_fuse_to_state_dict",
    "fuse_online_hadamard_into_down_proj",
    "fuse_online_hadamard_into_o_proj",
]
