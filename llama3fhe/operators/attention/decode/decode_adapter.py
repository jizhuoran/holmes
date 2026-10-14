"""Compatibility import for the archived legacy C-form Decode adapter.

The adapter is no longer part of the active Decode design.  It remains
importable only so historical scripts that explicitly request ``legacy_c`` can
be replayed; new code must use PairFold128 routing.
"""

from archive.decode_legacy.decode_adapter import (
    DecodeAdapterFHEResult,
    decode_adapter_rotations,
    decode_prepare_qkv_fhe,
    decode_qkv_carrier_to_c_numpy,
    unpack_decode_qkv_carrier,
)

__all__ = [
    "DecodeAdapterFHEResult",
    "decode_adapter_rotations",
    "decode_prepare_qkv_fhe",
    "decode_qkv_carrier_to_c_numpy",
    "unpack_decode_qkv_carrier",
]

