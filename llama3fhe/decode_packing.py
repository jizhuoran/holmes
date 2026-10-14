"""Public packing selector shared by Llama Prefill and Decode planning.

PairFold128 is the active design.  ``legacy_c`` remains accepted solely for
replaying archived C-form experiments through the compatibility shims.
"""

from __future__ import annotations

from typing import Literal, cast

DecodePacking = Literal["legacy_c", "pairfold128"]
DEFAULT_DECODE_PACKING: DecodePacking = "pairfold128"

_DECODE_PACKINGS = frozenset(("legacy_c", "pairfold128"))


def validate_decode_packing(value: object) -> DecodePacking:
    """Return a supported Decode packing or reject it at the API boundary."""

    if not isinstance(value, str) or value not in _DECODE_PACKINGS:
        choices = ", ".join(repr(choice) for choice in sorted(_DECODE_PACKINGS))
        raise ValueError(f"decode_packing must be one of {choices}, got {value!r}.")
    return cast(DecodePacking, value)


__all__ = [
    "DEFAULT_DECODE_PACKING",
    "DecodePacking",
    "validate_decode_packing",
]
