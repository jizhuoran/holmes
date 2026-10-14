"""Absolute positions for a fixed-width encrypted Prefill page."""

from __future__ import annotations

import numpy as np


def content_rope_start_pos(
    *,
    prefix_k: int,
    token_valid_mask: np.ndarray | None,
    seq_len: int,
    content_offset: int = 0,
) -> int:
    """Place the first valid content token after Prefix and earlier pages.

    A left-padded page still occupies physical slots 0..127.  Subtracting its
    leading padding keeps the first real token at the intended RoPE position.
    """

    start = int(prefix_k) + int(content_offset)
    if token_valid_mask is None:
        return start
    valid = np.asarray(token_valid_mask, dtype=bool).reshape(-1)
    if valid.shape != (int(seq_len),):
        raise ValueError(
            f"token_valid_mask must have shape ({int(seq_len)},), got {valid.shape}."
        )
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        raise ValueError("Prefill page must contain at least one valid token.")
    return start - int(indices[0])


def validate_prefill_cache_page(
    *,
    occupied: int | None,
    token_valid_mask: np.ndarray | None,
    seq_len: int,
) -> int:
    """Require a dense prefix of valid columns for a Decode KV cache page."""

    seq_len = int(seq_len)
    count = seq_len if occupied is None else int(occupied)
    if not 1 <= count <= seq_len:
        raise ValueError(f"prefill_occupied must be in [1, {seq_len}], got {count}.")
    if token_valid_mask is None:
        if count != seq_len:
            raise ValueError(
                "partial PairFold Prefill requires a right-padded token_valid_mask."
            )
        return count
    valid = np.asarray(token_valid_mask, dtype=bool).reshape(-1)
    if valid.shape != (seq_len,):
        raise ValueError(
            f"attention_token_valid_mask must have shape ({seq_len},), got {valid.shape}."
        )
    if not (np.all(valid[:count]) and not np.any(valid[count:])):
        raise ValueError(
            "PairFold Prefill cache requires valid tokens in columns "
            "0..prefill_occupied-1 and padding only at the tail."
        )
    return count
