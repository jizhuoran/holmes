"""Compatibility import for archived legacy C-form KV helpers.

PairFold128 owns the active persistent KV representation.  This module stays
as a narrow import bridge for explicit ``legacy_c`` replay only.
"""

from archive.decode_legacy.kv_cache import (
    add_c_pages,
    copy_c_pages,
    drop_c_ciphers_to_limbs,
    extend_c_pages,
    extend_kv_c_pages,
    flatten_c_pages,
    split_c_pages,
)

__all__ = [
    "add_c_pages",
    "copy_c_pages",
    "drop_c_ciphers_to_limbs",
    "extend_c_pages",
    "extend_kv_c_pages",
    "flatten_c_pages",
    "split_c_pages",
]

