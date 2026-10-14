from __future__ import annotations

"""Decode attention on C-form KV: NumPy oracles and the 1-query kernels.

Prefill still uses Delta QK / Alg2 / PV.  Decode consumes ``K_C`` / ``V_C``
pages and a single query token.  Slot contracts:

```text
K_C / V_C     AttentionPairLayout.pack_key_c / pack_value_c
Q compact     pack_compact_query_c    (pair p on token column p, 1 CT)
scores / P    pack_compact_decode_scores (pair p on feature row p, 1 CT)
o_new         pack_decode_output      (Q_C, token column 0, still 16 CT)
              → output_c_to_decode_fm (one DecodeFM ciphertext)
              → decode_w_o_fhe        (same 1-CT linear, skip PairFM)
```

Softmax keeps Alg2 mathematics but not the Prefill Delta graph: prefix
occupied mask, ``1/√d``, and a stride-2 token fold (full sum on column 0).
QK isolates each compact-Q column, broadcasts, multiplies ``K_C``, then
places the feature-reduced row onto compact score feature ``p``.
PV isolates each compact-P feature row back onto row 0, broadcasts, then
token-reduces.
"""

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from llama3fhe.approx.attention import apply_rope_heads, rope_cos_sin
from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.layouts.kv_cache import LayerKvCache

from ...primitives import (
    accumulate_owned,
    encode_mask,
    mul_cipher_rescale,
    mul_plain_rescale,
)
from ..prefill.input import _apply_sparse_routes_selected, choose_sparse_diagonal_plan
from ..sparse_bsgs_scheduler import canonical_rotation


def prefill_kv_cache_numpy(
    key: np.ndarray,
    value: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> LayerKvCache:
    """Pack a full Prefill window of RoPE'd K/V into one C-form page."""

    return LayerKvCache.from_prefill(key, value, layout=layout)


def _kv_heads(values: np.ndarray, *, layout: AttentionPairLayout, name: str) -> np.ndarray:
    values = np.asarray(values)
    packed_shape = (int(layout.key_value_pair_count), int(layout.slots))
    if values.ndim == 2 and values.shape == packed_shape:
        unpack = layout.unpack_key_c if name == "key" else layout.unpack_value_c
        values = unpack(values)
    return layout._logical_heads(
        values, heads=int(layout.key_value_heads), name=name
    )


def _query_heads(query: np.ndarray, *, layout: AttentionPairLayout) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    heads = int(layout.query_heads)
    dim = int(layout.head_dim)
    if query.ndim == 2 and query.shape == (heads, dim):
        return query[None, :, :]
    if query.ndim == 3 and query.shape[1:] == (heads, dim):
        return query
    if query.ndim == 2 and query.shape[1] == heads * dim:
        return query.reshape(query.shape[0], heads, dim)
    if query.ndim == 1 and query.shape == (heads * dim,):
        return query.reshape(1, heads, dim)
    raise ValueError(
        "decode query must be one or more tokens of "
        f"{(heads, dim)}, got {query.shape}."
    )


@dataclass(frozen=True)
class DecodePreparedQKV:
    """RoPE'd one-token Q/K/V plus C packs for the Decode adapter contract."""

    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    compact_query_c: np.ndarray
    key_token_c: np.ndarray
    value_token_c: np.ndarray


def decode_prepare_qkv_numpy(
    hidden: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    layout: AttentionPairLayout,
    position: int = 0,
    token_index: int = 0,
    rope_theta: float = 500000.0,
) -> DecodePreparedQKV:
    """RMSN1 row → RoPE(Q/K) → compact Q_C and one-column K/V C.

    This is the NumPy oracle for the FHE QKV→C adapter.  ``position``
    is the RoPE index; ``token_index`` is the cache page column to write.
    """

    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2 or int(hidden.shape[0]) != 1:
        raise ValueError(
            f"Decode QKV hidden must have shape (1, hidden), got {hidden.shape}."
        )
    query = hidden @ np.asarray(q_weight, dtype=np.float32).T
    key = hidden @ np.asarray(k_weight, dtype=np.float32).T
    value = hidden @ np.asarray(v_weight, dtype=np.float32).T
    cos, sin = rope_cos_sin(
        seq_len=1,
        head_dim=int(layout.head_dim),
        theta=float(rope_theta),
        start_pos=int(position),
    )
    query = apply_rope_heads(
        query,
        cos,
        sin,
        num_heads=int(layout.query_heads),
        head_dim=int(layout.head_dim),
    )
    key = apply_rope_heads(
        key,
        cos,
        sin,
        num_heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
    )
    return DecodePreparedQKV(
        query=query,
        key=key,
        value=value,
        compact_query_c=layout.pack_compact_query_c(query),
        key_token_c=layout.pack_token_c(
            key, token_index=int(token_index), kv=True, name="key"
        ),
        value_token_c=layout.pack_token_c(
            value, token_index=int(token_index), kv=True, name="value"
        ),
    )


def qk_c_numpy(
    query: np.ndarray,
    key: np.ndarray,
    *,
    layout: AttentionPairLayout,
    occupied: int | None = None,
) -> np.ndarray:
    """``Q[1 or N] x K_C -> [query_heads, seq]`` scores, unscaled.

    ``key`` may be packed C or logical ``[seq, kv_heads, d]``.
    ``query`` is logical tokens (decode: one row).
    """

    key_heads = _kv_heads(key, layout=layout, name="key")
    query = _query_heads(query, layout=layout)
    occupied = int(key_heads.shape[0] if occupied is None else occupied)
    key_heads = key_heads[:occupied]
    ratio = layout.gqa_ratio
    key_for_q = key_heads[:, np.arange(int(layout.query_heads)) // ratio, :]
    scores = np.einsum("qhd,khd->hqk", query, key_for_q, optimize=True)
    if int(query.shape[0]) == 1:
        return np.asarray(scores[:, 0, :], dtype=np.float32)
    return np.asarray(scores, dtype=np.float32)


def softmax_row_numpy(
    scores: np.ndarray,
    *,
    layout: AttentionPairLayout,
    occupied: int | None = None,
    scale: bool = True,
) -> np.ndarray:
    """Exact causal-as-prefix Softmax on Decode score rows ``[heads, seq]``."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 3:
        if scores.shape[1] != 1:
            raise ValueError("decode softmax expects one query row or [H,S].")
        scores = scores[:, 0, :]
    expected = (int(layout.query_heads), int(layout.seq_len))
    if scores.shape[0] != expected[0]:
        raise ValueError(f"decode softmax heads {scores.shape} vs {expected}.")
    seq = int(scores.shape[1])
    occupied = seq if occupied is None else int(occupied)
    if occupied < 0 or occupied > seq:
        raise ValueError(f"occupied={occupied} outside [0, {seq}].")
    values = np.array(scores, dtype=np.float64, copy=True)
    if scale:
        values /= math.sqrt(float(layout.head_dim))
    valid = np.zeros((seq,), dtype=bool)
    valid[:occupied] = True
    masked = np.where(valid[None, :], values, -np.inf)
    row_max = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.where(valid[None, :], np.exp(masked - row_max), 0.0)
    denominators = np.sum(exponentials, axis=-1, keepdims=True)
    probabilities = exponentials / np.maximum(denominators, np.finfo(np.float64).tiny)
    if seq < int(layout.seq_len):
        padded = np.zeros((int(layout.query_heads), int(layout.seq_len)), dtype=np.float32)
        padded[:, :seq] = probabilities
        return padded
    return np.asarray(probabilities, dtype=np.float32)


def softmax_alg2_row_numpy(
    scores: np.ndarray,
    *,
    layout: AttentionPairLayout,
    occupied: int | None = None,
    k: int = 2,
    scale: bool = True,
) -> np.ndarray:
    """Exact-exp Alg2 on a Decode score row. Matches ``softmax_row_numpy``.

    Same mathematics as Prefill Alg2 (``y=exp(x/2^k)``, then ``k`` rounds of
    ``y←(y λ)^2``), but the mask is a prefix on the C token axis rather than
    a Delta causal grid.  FHE reuses this packing and the stride-2 token
    fold; it must not call ``softmax_alg2_fhe_delta``.
    """

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 3:
        if scores.shape[1] != 1:
            raise ValueError("decode softmax expects one query row or [H,S].")
        scores = scores[:, 0, :]
    expected = (int(layout.query_heads), int(layout.seq_len))
    if scores.shape[0] != expected[0]:
        raise ValueError(f"decode softmax heads {scores.shape} vs {expected}.")
    seq = int(scores.shape[1])
    occupied = seq if occupied is None else int(occupied)
    if occupied < 0 or occupied > seq:
        raise ValueError(f"occupied={occupied} outside [0, {seq}].")
    k = int(k)
    if k < 1:
        raise ValueError(f"Alg2 k must be >= 1, got {k}.")
    values = np.array(scores, dtype=np.float64, copy=True)
    if scale:
        values /= math.sqrt(float(layout.head_dim))
    valid = np.zeros((seq,), dtype=bool)
    valid[:occupied] = True
    scaled = np.where(valid[None, :], values / float(2**k), -np.inf)
    y = np.where(valid[None, :], np.exp(scaled), 0.0)
    tiny = np.finfo(np.float64).tiny
    for _ in range(k):
        row_sum = np.sum(y * y, axis=-1, keepdims=True)
        lam = 1.0 / np.sqrt(np.maximum(row_sum, tiny))
        y = np.square(y * lam)
        y = np.where(valid[None, :], y, 0.0)
    if seq < int(layout.seq_len):
        padded = np.zeros(
            (int(layout.query_heads), int(layout.seq_len)), dtype=np.float32
        )
        padded[:, :seq] = y
        return padded
    return np.asarray(y, dtype=np.float32)


def pv_c_numpy(
    probabilities: np.ndarray,
    value: np.ndarray,
    *,
    layout: AttentionPairLayout,
    occupied: int | None = None,
) -> np.ndarray:
    """``P[heads, seq] x V_C -> [1, query_heads, d]`` for a single query."""

    if probabilities.ndim == 3:
        probabilities = probabilities[:, 0, :]
    probabilities = np.asarray(probabilities, dtype=np.float64)
    value_heads = _kv_heads(value, layout=layout, name="value")
    occupied = int(value_heads.shape[0] if occupied is None else occupied)
    value_heads = value_heads[:occupied]
    probabilities = probabilities[:, :occupied]
    ratio = layout.gqa_ratio
    value_for_q = value_heads[:, np.arange(int(layout.query_heads)) // ratio, :]
    output = np.einsum("hk,khd->hd", probabilities, value_for_q, optimize=True)
    return np.asarray(output[None, ...], dtype=np.float32)


def _pad_decode_score_rows(
    scores: np.ndarray, *, layout: AttentionPairLayout
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    heads = int(layout.query_heads)
    seq = int(layout.seq_len)
    if scores.shape == (heads, seq):
        return scores
    if scores.ndim == 2 and scores.shape[0] == heads and scores.shape[1] <= seq:
        padded = np.zeros((heads, seq), dtype=np.float32)
        padded[:, : scores.shape[1]] = scores
        return padded
    raise ValueError(
        f"decode scores must be [{heads}, ≤{seq}], got {scores.shape}."
    )


def decode_attention_numpy(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    layout: AttentionPairLayout,
    occupied: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-query GQA on C-form KV. Returns scores, P, output token."""

    packed_scores = _pad_decode_score_rows(
        qk_c_numpy(query, key, layout=layout, occupied=occupied), layout=layout
    )
    probabilities = softmax_row_numpy(
        packed_scores, layout=layout, occupied=occupied, scale=True
    )
    output = pv_c_numpy(
        probabilities, value, layout=layout, occupied=occupied
    )
    return packed_scores, probabilities, output


def decode_attention_alg2_numpy(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    layout: AttentionPairLayout,
    occupied: int | None = None,
    k: int = 2,
    config=None,
    layer_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as ``decode_attention_numpy`` with Alg2 Softmax (exact-exp or poly)."""

    packed_scores = _pad_decode_score_rows(
        qk_c_numpy(query, key, layout=layout, occupied=occupied), layout=layout
    )
    if config is None:
        probabilities = softmax_alg2_row_numpy(
            packed_scores, layout=layout, occupied=occupied, k=k, scale=True
        )
    else:
        from .decode_softmax import softmax_alg2_poly_numpy

        probabilities, _ = softmax_alg2_poly_numpy(
            packed_scores,
            layout=layout,
            config=config,
            layer_idx=layer_idx,
            occupied=occupied,
        )
    output = pv_c_numpy(
        probabilities, value, layout=layout, occupied=occupied
    )
    return packed_scores, probabilities, output


def _softmax_alg2_poly_logical_numpy(
    scores: np.ndarray,
    *,
    config,
    head_dim: int,
    layer_idx: int = 0,
) -> np.ndarray:
    """Alg2 over a dense ``[heads, T]`` score row (all columns valid)."""

    from llama3fhe.approx.chebyshev import eval_chebyshev_series_direct

    scores = np.asarray(scores, dtype=np.float64)
    k = int(config.k_for(layer_idx))
    scale = 1.0 / math.sqrt(float(head_dim))
    mapped = scores * (scale / float(2**k)) - float(
        config.U_hat_for(layer_idx)
    ) / float(2**k)
    exp_entry = config.layer_exp[int(layer_idx)]
    exp_lo, exp_hi = (float(value) for value in exp_entry["fit_interval"])
    y = eval_chebyshev_series_direct(
        mapped,
        np.asarray(exp_entry["coefficients"], dtype=np.float64),
        exp_lo,
        exp_hi,
    ).astype(np.float64)
    rough = config.invsqrt_alg2["j1_rough"]
    precise = config.invsqrt_alg2["j2_precise"]
    for round_index in range(1, k + 1):
        poly = config.poly_for_layer(
            rough if round_index == 1 else precise, int(layer_idx)
        )
        cheb = poly.get("chebyshev_fhe") or poly
        lo, hi = (float(value) for value in cheb["fit_interval"])
        coeffs = np.asarray(cheb["coefficients"], dtype=np.float64)
        row_sum = np.sum(y * y, axis=-1, keepdims=True)
        lam = eval_chebyshev_series_direct(row_sum, coeffs, lo, hi).astype(
            np.float64
        )
        y = np.square(y * lam)
    return np.asarray(y, dtype=np.float32)


def decode_attention_alg2_pages_numpy(
    query: np.ndarray,
    key_pages: tuple[np.ndarray, ...] | list[np.ndarray],
    value_pages: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    layout: AttentionPairLayout,
    occupied_pages: tuple[int, ...] | list[int],
    config,
    layer_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-query Alg2 attention over concatenated C-form KV pages."""

    if len(key_pages) != len(value_pages) or len(key_pages) != len(occupied_pages):
        raise ValueError("key/value/occupied page counts must match.")
    logical = [
        np.asarray(
            qk_c_numpy(query, key, layout=layout, occupied=int(occupied)),
            dtype=np.float32,
        )[:, : int(occupied)]
        for key, occupied in zip(key_pages, occupied_pages, strict=True)
    ]
    concat = np.concatenate(logical, axis=-1)
    probabilities = _softmax_alg2_poly_logical_numpy(
        concat,
        config=config,
        head_dim=int(layout.head_dim),
        layer_idx=int(layer_idx),
    )
    output = None
    offset = 0
    for value, occupied in zip(value_pages, occupied_pages, strict=True):
        occupied = int(occupied)
        padded = np.zeros(
            (int(layout.query_heads), int(layout.seq_len)), dtype=np.float32
        )
        padded[:, :occupied] = probabilities[:, offset : offset + occupied]
        offset += occupied
        contrib = pv_c_numpy(
            padded,
            value,
            layout=layout,
            occupied=occupied,
        )
        output = contrib if output is None else output + contrib
    return concat, probabilities, output


def decode_attention_rotations(layout: AttentionPairLayout) -> tuple[int, ...]:
    """Rotation union for Decode compact QK, Softmax token fold, and PV."""

    score_rows = layout.compact_score_row_rotations()
    return tuple(
        dict.fromkeys(
            layout.decode_qk_feature_rotations()
            + layout.decode_softmax_token_rotations()
            + layout.decode_softmax_token_broadcast_rotations()
            + layout.decode_feature_broadcast_rotations()
            + layout.compact_q_column_rotations()
            + score_rows
            + tuple(-step for step in score_rows)
            + layout.decode_page_merge_rotations()
        )
    )


def _rotate_add_tree(
    cipher,
    *,
    stride: int,
    count: int,
    crypto_context,
):
    """Butterfly-add ``count`` groups along ``stride``. Does not consume ``cipher``."""

    if int(count) <= 1:
        return cipher.deep_copy()

    acc = cipher
    owned = False
    bit = 1
    try:
        while bit < int(count):
            offset = int(stride) * bit
            rotate_add = getattr(crypto_context.fhe, "homo_rotate_add", None)
            if callable(rotate_add) and os.environ.get("LLAMA_FHE_ROTATE_ADD", "1") != "0":
                # EasyFHE's fused kernel computes rotate(acc, offset)+acc
                # without materializing the rotated temporary.  The fallback
                # remains for emulation and older backends.
                summed = rotate_add(
                    acc,
                    offset,
                    crypto_context.context,
                    addend=acc,
                )
            else:
                rotated = crypto_context.fhe.homo_rotate(
                    acc, offset, crypto_context.context
                )
                try:
                    summed = crypto_context.fhe.homo_add(
                        acc, rotated, crypto_context.context
                    )
                finally:
                    release_if_supported(rotated)
            if owned:
                release_if_supported(acc)
            acc = summed
            owned = True
            bit <<= 1
        result = acc
        acc = None
        return result
    finally:
        if owned:
            release_if_supported(acc)


def _feature_reduce_cipher(cipher, *, layout: AttentionPairLayout, crypto_context):
    """Sum C feature rows onto row 0 (token axis preserved)."""

    return _rotate_add_tree(
        cipher,
        stride=2 * int(layout.seq_len),
        count=int(layout.head_dim),
        crypto_context=crypto_context,
    )


def _token_reduce_cipher(cipher, *, layout: AttentionPairLayout, crypto_context):
    """Sum C token columns; the full row sum lands on token column 0."""

    return _rotate_add_tree(
        cipher,
        stride=2,
        count=int(layout.seq_len),
        crypto_context=crypto_context,
    )


def _feature_broadcast_cipher(cipher, *, layout: AttentionPairLayout, crypto_context):
    """Copy feature row 0 onto every feature row (P for PV)."""

    return _rotate_add_tree(
        cipher,
        stride=-(2 * int(layout.seq_len)),
        count=int(layout.head_dim),
        crypto_context=crypto_context,
    )


def _token_broadcast_cipher(cipher, *, layout: AttentionPairLayout, crypto_context):
    """Copy token column 0 onto every C token column."""

    return _rotate_add_tree(
        cipher,
        stride=-2,
        count=int(layout.seq_len),
        crypto_context=crypto_context,
    )


def _homo_rotate(cipher, amount: int, *, crypto_context):
    """Rotate ``cipher`` by ``amount``.  Does not consume the input."""

    amount = int(amount)
    if amount == 0:
        return cipher.deep_copy()
    return crypto_context.fhe.homo_rotate(
        cipher, amount, crypto_context.context
    )


def _require_compact_query(query_ciphers, *, layout: AttentionPairLayout):
    query_ciphers = tuple(query_ciphers)
    if len(query_ciphers) != 1:
        raise ValueError(
            f"Decode compact Q expects 1 ciphertext, got {len(query_ciphers)}."
        )
    layout._require_compact_q()
    return query_ciphers


def _require_compact_scores(score_ciphers, *, layout: AttentionPairLayout, name: str):
    score_ciphers = tuple(score_ciphers)
    if len(score_ciphers) != 1:
        raise ValueError(
            f"{name} expects 1 compact ciphertext, got {len(score_ciphers)}."
        )
    layout._require_compact_q()
    return score_ciphers


def _pad_mask_row(row: np.ndarray, *, slots: int) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size == int(slots):
        return row
    if row.size > int(slots):
        raise ValueError(f"mask row length {row.size} exceeds slots={slots}.")
    padded = np.zeros((int(slots),), dtype=np.float64)
    padded[: row.size] = row
    return padded


def _mul_packed_rows(
    ciphers: tuple[object, ...],
    packed_rows: np.ndarray,
    *,
    name: str,
    crypto_context,
) -> tuple[object, ...]:
    outputs: list[object] = []
    try:
        for index, cipher in enumerate(ciphers):
            slots = int(getattr(cipher, "slots", packed_rows.shape[-1]))
            row = _pad_mask_row(packed_rows[index], slots=slots)
            level = int(crypto_context.level_for_cipher(cipher))
            plain = encode_mask(
                row,
                name=f"{name}.{index}",
                level=level,
                crypto_context=crypto_context,
            )
            try:
                outputs.append(
                    mul_plain_rescale(cipher, plain, crypto_context=crypto_context)
                )
            finally:
                release_if_supported(plain)
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in outputs:
            release_if_supported(cipher)


def prefill_key_c_emit_correction_mask(
    layout: AttentionPairLayout, *, dtype=np.float64
) -> np.ndarray:
    """Undo Prefill K_C expand+unfold: ``2 (K_lo, -K_hi)`` → ``(K_lo, K_hi)``."""

    seq_len = int(layout.seq_len)
    heads = int(layout.key_value_heads)
    dim = int(layout.head_dim)
    scale = np.ones((seq_len, heads, dim), dtype=np.float32)
    scale[:, :, dim // 2 :] = -1.0
    return 0.5 * layout.pack_key_c(
        scale.reshape(seq_len, heads * dim), dtype=dtype
    )


def correct_prefill_emitted_key_c_numpy(
    packed: np.ndarray, *, layout: AttentionPairLayout
) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.float64)
    return packed * prefill_key_c_emit_correction_mask(layout)


def correct_prefill_emitted_key_c_fhe(
    ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Consume unfolded Prefill ``K_C`` and return Decode-ready ``pack_key_c``."""

    ciphers = tuple(ciphers)
    repaired = _mul_packed_rows(
        ciphers,
        prefill_key_c_emit_correction_mask(layout),
        name="prefill.key_c.correct",
        crypto_context=crypto_context,
    )
    for cipher in ciphers:
        release_if_supported(cipher)
    return repaired


def _add_packed_rows(
    ciphers: tuple[object, ...],
    packed_rows: np.ndarray,
    *,
    name: str,
    crypto_context,
    max_abs_chunk: float = 1.0,
) -> tuple[object, ...]:
    """Add a packed row without consuming ``ciphers``.

    Prefill splits the Alg2 invalid fill into unit-abs chunks so a
    ``stats_min/2^k`` of several units still encodes at scale ``2^59``.
    Compact Decode hits the same bound on later layers (``k=2``).
    """

    packed = np.asarray(packed_rows, dtype=np.float64)
    if packed.ndim == 1:
        packed = packed.reshape(1, -1)
    peak = float(np.max(np.abs(packed))) if packed.size else 0.0
    chunk_count = (
        1
        if peak == 0.0
        else max(1, int(math.ceil(peak / float(max_abs_chunk))))
    )
    chunk = packed / float(chunk_count)
    current = tuple(ciphers)
    owns = False
    try:
        for chunk_index in range(chunk_count):
            nxt: list[object] = []
            try:
                for index, cipher in enumerate(current):
                    slots = int(getattr(cipher, "slots", chunk.shape[-1]))
                    row = _pad_mask_row(chunk[index], slots=slots)
                    level = int(crypto_context.level_for_cipher(cipher))
                    plain = encode_mask(
                        row,
                        name=f"{name}.{chunk_index}.{index}",
                        level=level,
                        crypto_context=crypto_context,
                    )
                    try:
                        nxt.append(
                            crypto_context.fhe.homo_add_pt(
                                cipher, plain, crypto_context.context
                            )
                        )
                    finally:
                        release_if_supported(plain)
            except Exception:
                for cipher in nxt:
                    release_if_supported(cipher)
                raise
            if owns:
                for cipher in current:
                    release_if_supported(cipher)
            current = tuple(nxt)
            owns = True
        result = current
        owns = False
        return result
    finally:
        if owns:
            for cipher in current:
                release_if_supported(cipher)


def _add_scalar_ciphers(
    ciphers: tuple[object, ...], value: float, *, crypto_context
) -> tuple[object, ...]:
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            encoded = crypto_context.fhe.encode_scalar(
                float(value),
                cur_limbs=int(cipher.state.cur_limbs),
                scale_degree=int(cipher.state.scale_degree),
                scaling_factor=cipher.state.scaling_factor,
                context=crypto_context.context,
            )
            outputs.append(
                crypto_context.fhe.homo_add_scalar(
                    cipher, encoded, crypto_context.context
                )
            )
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in outputs:
            release_if_supported(cipher)


def apply_decode_softmax_public_map_fhe(
    score_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    occupied: int | None = None,
    scale: bool = True,
) -> tuple[object, ...]:
    """Prefix-mask (and optional ``1/√d``) in compact Decode score slots."""

    score_ciphers = _require_compact_scores(
        score_ciphers, layout=layout, name="Decode Softmax"
    )
    mask = layout.compact_decode_prefix_mask(occupied)
    if scale:
        mask = np.asarray(mask, dtype=np.float64) * (
            1.0 / math.sqrt(float(layout.head_dim))
        )
    return _mul_packed_rows(
        score_ciphers,
        mask,
        name="decode.softmax.prefix",
        crypto_context=crypto_context,
    )


def decode_reduce_tokens_fhe(
    ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Token-axis fold used by Decode Alg2 ``λ`` and by PV."""

    ciphers = tuple(ciphers)
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            outputs.append(
                _token_reduce_cipher(
                    cipher, layout=layout, crypto_context=crypto_context
                )
            )
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in outputs:
            release_if_supported(cipher)


def decode_broadcast_token0_numpy(
    packed: np.ndarray,
    *,
    layout: AttentionPairLayout,
    occupied: int | None = None,
) -> np.ndarray:
    """Copy token column 0 across the C token axis, then apply the prefix mask."""

    scores = layout.unpack_compact_decode_scores(packed)
    occupied = int(layout.seq_len if occupied is None else occupied)
    broadcast = np.repeat(scores[:, :1], int(layout.seq_len), axis=1)
    broadcast[:, occupied:] = 0.0
    return layout.pack_compact_decode_scores(broadcast)


def decode_broadcast_tokens_fhe(
    ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    occupied: int | None = None,
) -> tuple[object, ...]:
    """Isolate token column 0, broadcast, remask to ``occupied``.

    After ``decode_reduce_tokens_fhe`` the row sum lives on column 0; other
    columns hold partial folds.  Alg2 ``λ`` must use this copy, not the
    unmasked reduce.
    """

    ciphers = _require_compact_scores(
        ciphers, layout=layout, name="Decode token broadcast"
    )
    isolated = decode_broadcast = remasked = None
    broadcast: list[object] = []
    try:
        isolated = _mul_packed_rows(
            ciphers,
            layout.compact_decode_token0_mask(),
            name="decode.softmax.token0",
            crypto_context=crypto_context,
        )
        broadcast = []
        for cipher in isolated:
            broadcast.append(
                _token_broadcast_cipher(
                    cipher, layout=layout, crypto_context=crypto_context
                )
            )
        decode_broadcast = tuple(broadcast)
        broadcast = []
        remasked = _mul_packed_rows(
            decode_broadcast,
            layout.compact_decode_prefix_mask(occupied),
            name="decode.softmax.lambda_prefix",
            crypto_context=crypto_context,
        )
        result = remasked
        remasked = None
        return result
    finally:
        for cipher in tuple(isolated or ()) + tuple(decode_broadcast or ()):
            release_if_supported(cipher)
        for cipher in remasked or ():
            release_if_supported(cipher)
        for cipher in broadcast:
            release_if_supported(cipher)


def folded_key_c_to_real_c_fhe(
    ciphers: tuple[object, ...] | list[object],
    *,
    crypto_context,
) -> tuple[object, ...]:
    """Turn Prefill folded ``K_C`` into Decode real ``pack_key_c``.

    Folded slots hold ``0.5 (K + i K_paired)``.  Adding the conjugate
    recovers ``K``.  Consumes the folded inputs.
    """

    inputs = tuple(ciphers)
    outputs: list[object] = []
    # The operation consumes inputs one at a time.  Keep only the not-yet
    # consumed suffix so a failure in a rotation/addition still releases every
    # source exactly once; callers can then treat this helper as an ownership
    # transfer at invocation time.
    remaining = list(inputs)
    try:
        for cipher in inputs:
            conjugated = crypto_context.fhe.homo_rotate(
                cipher,
                int(crypto_context.context.M) - 1,
                crypto_context.context,
            )
            try:
                outputs.append(
                    crypto_context.fhe.homo_add(
                        cipher, conjugated, crypto_context.context
                    )
                )
            finally:
                release_if_supported(conjugated)
            release_if_supported(cipher)
            for index, candidate in enumerate(remaining):
                if candidate is cipher:
                    remaining.pop(index)
                    break
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in outputs:
            release_if_supported(cipher)
        for cipher in remaining:
            release_if_supported(cipher)


def prefill_key_c_emit_correction_mask(
    layout: AttentionPairLayout, *, dtype=np.float64
) -> np.ndarray:
    """Undo Prefill K_C expand+unfold: ``2 (K_lo, -K_hi)`` → ``(K_lo, K_hi)``."""

    seq_len = int(layout.seq_len)
    heads = int(layout.key_value_heads)
    dim = int(layout.head_dim)
    scale = np.ones((seq_len, heads, dim), dtype=np.float32)
    scale[:, :, dim // 2 :] = -1.0
    return 0.5 * layout.pack_key_c(
        scale.reshape(seq_len, heads * dim), dtype=dtype
    )


def correct_prefill_emitted_key_c_numpy(
    packed: np.ndarray, *, layout: AttentionPairLayout
) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.float64)
    return packed * prefill_key_c_emit_correction_mask(layout)


def correct_prefill_emitted_key_c_fhe(
    ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Consume unfolded Prefill ``K_C`` and return Decode-ready ``pack_key_c``."""

    ciphers = tuple(ciphers)
    repaired = _mul_packed_rows(
        ciphers,
        prefill_key_c_emit_correction_mask(layout),
        name="prefill.key_c.correct",
        crypto_context=crypto_context,
    )
    for cipher in ciphers:
        release_if_supported(cipher)
    return repaired


def qk_c_fhe(
    query_ciphers: tuple[object, ...] | list[object],
    key_c_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Compact Q_C × K_C: isolate column p, broadcast, reduce, place on feature p."""

    query_ciphers = _require_compact_query(query_ciphers, layout=layout)
    key_c_ciphers = tuple(key_c_ciphers)
    if len(key_c_ciphers) != int(layout.key_value_pair_count):
        raise ValueError(
            f"Decode QK expects {layout.key_value_pair_count} K_C ciphers, "
            f"got {len(key_c_ciphers)}."
        )
    query = query_ciphers[0]
    ratio = int(layout.gqa_ratio)
    stride = 2 * int(layout.seq_len)
    accumulator = None
    try:
        for pair_index in range(int(layout.query_pair_count)):
            isolated = shifted = broadcast = product = reduced = isolated_row = (
                placed
            ) = None
            try:
                isolated = _mul_packed_rows(
                    (query,),
                    layout.compact_token_column_mask(pair_index),
                    name=f"decode.qk.qcol{pair_index}",
                    crypto_context=crypto_context,
                )[0]
                shifted = _homo_rotate(
                    isolated, 2 * int(pair_index), crypto_context=crypto_context
                )
                broadcast = _token_broadcast_cipher(
                    shifted, layout=layout, crypto_context=crypto_context
                )
                product = mul_cipher_rescale(
                    broadcast,
                    key_c_ciphers[int(pair_index) // ratio],
                    crypto_context=crypto_context,
                )
                reduced = _feature_reduce_cipher(
                    product, layout=layout, crypto_context=crypto_context
                )
                isolated_row = _mul_packed_rows(
                    (reduced,),
                    layout.compact_feature_row_mask(0),
                    name=f"decode.qk.srow0.{pair_index}",
                    crypto_context=crypto_context,
                )[0]
                placed = _homo_rotate(
                    isolated_row,
                    -stride * int(pair_index),
                    crypto_context=crypto_context,
                )
            finally:
                release_if_supported(isolated)
                release_if_supported(shifted)
                release_if_supported(broadcast)
                release_if_supported(product)
                release_if_supported(reduced)
                release_if_supported(isolated_row)
            accumulator = accumulate_owned(
                accumulator, placed, crypto_context=crypto_context
            )
        if accumulator is None:
            raise RuntimeError("compact QK produced no score ciphertext.")
        result = accumulator
        accumulator = None
        return (result,)
    finally:
        release_if_supported(accumulator)


def _prefix_page_payload(
    values: np.ndarray,
    *,
    layout: AttentionPairLayout,
    name: str,
    prefix_k: int,
) -> np.ndarray:
    """Return a zero-padded C-page payload for a plaintext Prefix KV tensor."""

    values = np.asarray(values)
    prefix_k = int(prefix_k)
    if prefix_k < 1 or prefix_k > int(layout.seq_len):
        raise ValueError(
            f"{name} prefix_k={prefix_k} must fit one C page of "
            f"seq_len={layout.seq_len}."
        )
    packed_shape = (int(layout.key_value_pair_count), int(layout.slots))
    if values.shape == packed_shape:
        return np.asarray(values, dtype=np.float64)
    expected_flat = (int(layout.key_value_heads) * int(layout.head_dim),)
    expected_heads = (int(layout.key_value_heads), int(layout.head_dim))
    if values.ndim == 2 and values.shape[1:] == expected_flat:
        logical = values.reshape(values.shape[0], *expected_heads).astype(
            np.float32, copy=False
        )
    elif values.ndim == 3 and values.shape[1:] == expected_heads:
        logical = values.astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"{name} must have shape [tokens, {expected_flat[0]}] or "
            f"[tokens, {expected_heads[0]}, {expected_heads[1]}], got {values.shape}."
        )
    if logical.shape[0] < prefix_k:
        raise ValueError(
            f"{name} has {logical.shape[0]} tokens but prefix_k={prefix_k}."
        )
    padded = np.zeros(
        (int(layout.seq_len), int(layout.key_value_heads), int(layout.head_dim)),
        dtype=np.float32,
    )
    padded[:prefix_k] = logical[:prefix_k]
    pack = layout.pack_key_c if name == "key" else layout.pack_value_c
    return np.asarray(
        pack(padded.reshape(int(layout.seq_len), -1), dtype=np.float64),
        dtype=np.float64,
    )


def _encode_plain_page(
    page: np.ndarray,
    cipher,
    *,
    name: str,
    crypto_context,
):
    level = int(crypto_context.level_for_cipher(cipher))
    row = _pad_mask_row(np.asarray(page, dtype=np.float64), slots=int(cipher.slots))
    return encode_mask(row, name=name, level=level, crypto_context=crypto_context)


def qk_prefix_pc_compact_fhe(
    query_ciphers: tuple[object, ...] | list[object],
    key_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    prefix_k: int,
) -> tuple[object, ...]:
    """Compact Decode Q×Prefix-K (CT×PT) in the C-page contract.

    The Prefix page is encoded once per KV pair and reused by all GQA query
    pairs.  The result is one compact score ciphertext, so it can participate
    in the existing multi-page Decode Softmax without a Prefill-sized Q_C
    expansion.
    """

    query_ciphers = _require_compact_query(query_ciphers, layout=layout)
    prefix_k = int(prefix_k)
    payload = _prefix_page_payload(
        key_prefix,
        layout=layout,
        name="key",
        prefix_k=prefix_k,
    )
    query = query_ciphers[0]
    ratio = int(layout.gqa_ratio)
    stride = 2 * int(layout.seq_len)
    accumulator = None
    plaintext_cache: dict[tuple[int, int], object] = {}
    try:
        for pair_index in range(int(layout.query_pair_count)):
            isolated = shifted = broadcast = product = reduced = row0 = placed = None
            try:
                isolated = _mul_packed_rows(
                    (query,),
                    layout.compact_token_column_mask(pair_index),
                    name=f"decode.prefix.qk.qcol{pair_index}",
                    crypto_context=crypto_context,
                )[0]
                shifted = _homo_rotate(
                    isolated, 2 * int(pair_index), crypto_context=crypto_context
                )
                broadcast = _token_broadcast_cipher(
                    shifted, layout=layout, crypto_context=crypto_context
                )
                kv_pair = int(pair_index) // ratio
                level = int(crypto_context.level_for_cipher(broadcast))
                cache_key = (kv_pair, level)
                plain = plaintext_cache.get(cache_key)
                if plain is None:
                    plain = _encode_plain_page(
                        payload[kv_pair],
                        broadcast,
                        name=f"decode.prefix.qk.k{kv_pair}",
                        crypto_context=crypto_context,
                    )
                    plaintext_cache[cache_key] = plain
                product = mul_plain_rescale(
                    broadcast, plain, crypto_context=crypto_context
                )
                reduced = _feature_reduce_cipher(
                    product, layout=layout, crypto_context=crypto_context
                )
                row0 = _mul_packed_rows(
                    (reduced,),
                    layout.compact_feature_row_mask(0),
                    name=f"decode.prefix.qk.srow0.{pair_index}",
                    crypto_context=crypto_context,
                )[0]
                placed = _homo_rotate(
                    row0, -stride * int(pair_index), crypto_context=crypto_context
                )
                accumulator = accumulate_owned(
                    accumulator, placed, crypto_context=crypto_context
                )
                placed = None
            finally:
                for cipher in (isolated, shifted, broadcast, product, reduced, row0, placed):
                    release_if_supported(cipher)
        if accumulator is None:
            raise RuntimeError("compact Prefix QK produced no scores.")
        masked = _mul_packed_rows(
            (accumulator,),
            layout.compact_decode_prefix_mask(prefix_k),
            name="decode.prefix.qk.mask",
            crypto_context=crypto_context,
        )[0]
        release_if_supported(accumulator)
        accumulator = None
        return (masked,)
    finally:
        release_if_supported(accumulator)
        for plain in plaintext_cache.values():
            release_if_supported(plain)


def pv_prefix_pc_compact_fhe(
    probability_ciphers: tuple[object, ...] | list[object],
    value_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    prefix_k: int,
) -> tuple[object, ...]:
    """Compact Decode P×Prefix-V (CT×PT), returning normal O_C pair ciphers."""

    probability_ciphers = _require_compact_scores(
        probability_ciphers, layout=layout, name="Decode Prefix PV"
    )
    prefix_k = int(prefix_k)
    payload = _prefix_page_payload(
        value_prefix,
        layout=layout,
        name="value",
        prefix_k=prefix_k,
    )
    probability = probability_ciphers[0]
    ratio = int(layout.gqa_ratio)
    stride = 2 * int(layout.seq_len)
    outputs: list[object] = []
    plaintext_cache: dict[tuple[int, int], object] = {}
    try:
        for pair_index in range(int(layout.query_pair_count)):
            isolated = shifted = broadcast = product = reduced = None
            try:
                isolated = _mul_packed_rows(
                    (probability,),
                    layout.compact_feature_row_mask(pair_index),
                    name=f"decode.prefix.pv.prow{pair_index}",
                    crypto_context=crypto_context,
                )[0]
                shifted = _homo_rotate(
                    isolated, stride * int(pair_index), crypto_context=crypto_context
                )
                broadcast = _feature_broadcast_cipher(
                    shifted, layout=layout, crypto_context=crypto_context
                )
                kv_pair = int(pair_index) // ratio
                level = int(crypto_context.level_for_cipher(broadcast))
                cache_key = (kv_pair, level)
                plain = plaintext_cache.get(cache_key)
                if plain is None:
                    plain = _encode_plain_page(
                        payload[kv_pair],
                        broadcast,
                        name=f"decode.prefix.pv.v{kv_pair}",
                        crypto_context=crypto_context,
                    )
                    plaintext_cache[cache_key] = plain
                product = mul_plain_rescale(
                    broadcast, plain, crypto_context=crypto_context
                )
                reduced = _token_reduce_cipher(
                    product, layout=layout, crypto_context=crypto_context
                )
                outputs.append(reduced)
                reduced = None
            finally:
                for cipher in (isolated, shifted, broadcast, product, reduced):
                    release_if_supported(cipher)
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in outputs:
            release_if_supported(cipher)
        for plain in plaintext_cache.values():
            release_if_supported(plain)


def pv_c_fhe(
    probability_ciphers: tuple[object, ...] | list[object],
    value_c_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Isolate compact-P feature p, broadcast along features, ``P ⊙ V_C``, token-reduce."""

    probability_ciphers = _require_compact_scores(
        probability_ciphers, layout=layout, name="Decode PV"
    )
    value_c_ciphers = tuple(value_c_ciphers)
    if len(value_c_ciphers) != int(layout.key_value_pair_count):
        raise ValueError(
            f"Decode PV expects {layout.key_value_pair_count} V_C ciphers, "
            f"got {len(value_c_ciphers)}."
        )
    probability = probability_ciphers[0]
    ratio = int(layout.gqa_ratio)
    stride = 2 * int(layout.seq_len)
    outputs: list[object] = []
    try:
        for pair_index in range(int(layout.query_pair_count)):
            isolated = shifted = broadcast = product = None
            try:
                isolated = _mul_packed_rows(
                    (probability,),
                    layout.compact_feature_row_mask(pair_index),
                    name=f"decode.pv.prow{pair_index}",
                    crypto_context=crypto_context,
                )[0]
                shifted = _homo_rotate(
                    isolated,
                    stride * int(pair_index),
                    crypto_context=crypto_context,
                )
                broadcast = _feature_broadcast_cipher(
                    shifted, layout=layout, crypto_context=crypto_context
                )
                product = mul_cipher_rescale(
                    broadcast,
                    value_c_ciphers[int(pair_index) // ratio],
                    crypto_context=crypto_context,
                )
                outputs.append(
                    _token_reduce_cipher(
                        product, layout=layout, crypto_context=crypto_context
                    )
                )
            finally:
                release_if_supported(isolated)
                release_if_supported(shifted)
                release_if_supported(broadcast)
                release_if_supported(product)
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in outputs:
            release_if_supported(cipher)


def _validate_decode_fm(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> None:
    if int(feature_layout.seq_len) != 1:
        raise ValueError("Decode FM must have seq_len=1.")
    if int(feature_layout.cipher_count) != 1:
        raise ValueError("Decode FM must occupy one ciphertext.")
    if int(feature_layout.slots) != int(attention_layout.slots):
        raise ValueError("Decode FM and attention slot counts must match.")
    hidden = int(attention_layout.query_heads) * int(attention_layout.head_dim)
    if int(feature_layout.hidden_dim) != hidden:
        raise ValueError(
            "Decode FM hidden_dim must equal query_heads*head_dim, "
            f"got {feature_layout.hidden_dim} vs {hidden}."
        )


def _decode_output_vector(
    output: np.ndarray, *, layout: AttentionPairLayout
) -> np.ndarray:
    output = np.asarray(output, dtype=np.float32)
    heads = int(layout.query_heads)
    dim = int(layout.head_dim)
    packed_shape = (int(layout.query_pair_count), int(layout.slots))
    if output.ndim == 2 and output.shape == packed_shape:
        return layout.unpack_output_c(output)[0]
    if output.shape == (1, heads, dim):
        return output.reshape(-1)
    if output.shape == (heads, dim):
        return output.reshape(-1)
    if output.shape == (1, heads * dim):
        return output.reshape(-1)
    if output.shape == (heads * dim,):
        return output.reshape(-1)
    raise ValueError(
        "decode output must be packed O_C or one token of "
        f"{(heads, dim)}, got {output.shape}."
    )


def output_c_to_decode_fm_numpy(
    output: np.ndarray,
    *,
    attention_layout: AttentionPairLayout,
    feature_layout: FeatureMajorPrefillLayout,
) -> np.ndarray:
    """Scatter token-0 of O_C / ``o_new`` onto one DecodeFM ciphertext."""

    _validate_decode_fm(feature_layout, attention_layout)
    hidden = _decode_output_vector(output, layout=attention_layout)
    return feature_layout.pack(hidden.reshape(1, -1))


def decode_output_c_routes(
    pair_index: int,
    *,
    attention_layout: AttentionPairLayout,
    feature_layout: FeatureMajorPrefillLayout,
) -> dict[int, np.ndarray]:
    """Sparse ``src → FM.slot(logical, 0)`` diagonals for one O_C pair cipher."""

    _validate_decode_fm(feature_layout, attention_layout)
    pairs = attention_layout.query_head_pairs()
    if pair_index < 0 or pair_index >= len(pairs):
        raise ValueError(
            f"pair_index={pair_index} outside [0, {len(pairs)})."
        )
    slots = int(attention_layout.slots)
    destinations: dict[int, list[int]] = {}
    head_dim = int(attention_layout.head_dim)
    for feature in range(head_dim):
        for lane, head in enumerate(pairs[pair_index]):
            source = attention_layout.c_slot(feature, 0, lane)
            logical = int(head) * head_dim + feature
            destination = feature_layout.slot(logical, 0)
            offset = canonical_rotation(source - destination, slots)
            destinations.setdefault(offset, []).append(destination)
    return {
        int(offset): np.asarray(indices, dtype=np.int64)
        for offset, indices in destinations.items()
    }


def decode_output_c_rotations(
    attention_layout: AttentionPairLayout,
    feature_layout: FeatureMajorPrefillLayout,
    *,
    max_baby_offsets: int = 32,
) -> tuple[int, ...]:
    """Rotation union for the O_C → DecodeFM scatter."""

    rotations: set[int] = set()
    for pair_index in range(int(attention_layout.query_pair_count)):
        routes = decode_output_c_routes(
            pair_index,
            attention_layout=attention_layout,
            feature_layout=feature_layout,
        )
        plan = choose_sparse_diagonal_plan(
            tuple(routes),
            slots=int(attention_layout.slots),
            max_baby_offsets=int(max_baby_offsets),
        )
        rotations.update(int(value) for value in plan.baby_offsets)
        rotations.update(int(value) for value in plan.giant_offsets)
    return tuple(sorted(value for value in rotations if value))


def output_c_to_decode_fm_fhe(
    output_ciphers: tuple[object, ...] | list[object],
    *,
    attention_layout: AttentionPairLayout,
    feature_layout: FeatureMajorPrefillLayout,
    crypto_context,
    max_baby_offsets: int = 32,
) -> tuple[object, ...]:
    """Skip Prefill PairFM: gather token-0 O_C into one DecodeFM ciphertext.

    Uses the existing sparse-route BSGS (rotate / ``mul_pt`` / add).  No
    EasyFHE kernel change.
    """

    _validate_decode_fm(feature_layout, attention_layout)
    output_ciphers = tuple(output_ciphers)
    if len(output_ciphers) != int(attention_layout.query_pair_count):
        raise ValueError(
            f"Decode PairFM-skip expects {attention_layout.query_pair_count} "
            f"O_C ciphers, got {len(output_ciphers)}."
        )
    slots = int(attention_layout.slots)
    accumulator = None
    try:
        for pair_index, cipher in enumerate(output_ciphers):
            routes = decode_output_c_routes(
                pair_index,
                attention_layout=attention_layout,
                feature_layout=feature_layout,
            )
            plan = choose_sparse_diagonal_plan(
                tuple(routes),
                slots=slots,
                max_baby_offsets=int(max_baby_offsets),
            )
            converted = _apply_sparse_routes_selected(
                (cipher,),
                routes=routes,
                plan=plan,
                slots=slots,
                crypto_context=crypto_context,
                function_prefix=f"decode.output_c_to_fm.pair{pair_index}",
                padded_plaintext_rows=None,
                use_batched=False,
                batch_cap=32,
                fallback_on_error=True,
            )
            if len(converted) != 1:
                raise RuntimeError("O_C scatter returned more than one ciphertext.")
            accumulator = accumulate_owned(
                accumulator, converted[0], crypto_context=crypto_context
            )
        if accumulator is None:
            raise RuntimeError("O_C scatter produced no DecodeFM ciphertext.")
        result = accumulator
        accumulator = None
        return (result,)
    finally:
        release_if_supported(accumulator)


@dataclass
class DecodeAttentionFHEResult:
    """Owned ``O_C`` (Q_C token-0) from the Decode QK → Alg2 → PV graph."""

    output_ciphers: tuple[object, ...]
    wall_seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def release(self) -> None:
        for cipher in self.output_ciphers:
            release_if_supported(cipher)
        self.output_ciphers = ()


def decode_attention_fhe(
    query_ciphers: tuple[object, ...] | list[object],
    key_c_ciphers: tuple[object, ...] | list[object],
    value_c_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    softmax_config,
    occupied: int | None = None,
    layer_idx: int = 0,
    rounds: int | None = None,
    score_scale: float | None = None,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
) -> DecodeAttentionFHEResult:
    """1-query C-form attention.  Does not consume Q / K_C / V_C.

    Prefill ``attention_fhe_delta_alg2`` stays on Delta.  This graph is
    ``qk_c_fhe`` → ``softmax_alg2_row_fhe`` → ``pv_c_fhe``.
    """

    from .decode_softmax import softmax_alg2_row_fhe

    query_ciphers = tuple(query_ciphers)
    key_c_ciphers = tuple(key_c_ciphers)
    value_c_ciphers = tuple(value_c_ciphers)
    scores = probabilities = outputs = None
    stage_seconds: dict[str, float] = {}
    wall_start = time.perf_counter()
    try:
        start = time.perf_counter()
        scores = qk_c_fhe(
            query_ciphers,
            key_c_ciphers,
            layout=layout,
            crypto_context=crypto_context,
        )
        stage_seconds["qk"] = time.perf_counter() - start
        start = time.perf_counter()
        probabilities = softmax_alg2_row_fhe(
            scores,
            layout=layout,
            crypto_context=crypto_context,
            config=softmax_config,
            layer_idx=layer_idx,
            occupied=occupied,
            score_scale=score_scale,
            rounds=rounds,
            bootstrap_operator=bootstrap_operator,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_precision_bits=bootstrap_precision_bits,
        )
        stage_seconds["softmax"] = time.perf_counter() - start
        from .decode_softmax import _refresh_row_ciphers_if_short

        # PV is isolate(1)+mul-V(1).  Conversion rescale needs another.
        # Layer 31 k=5 can leave P at 2 and O_C at 1, which cannot Meta-BTS.
        probabilities = _refresh_row_ciphers_if_short(
            probabilities,
            bootstrap_operator=bootstrap_operator,
            need_limbs=6,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.probability",
        )
        start = time.perf_counter()
        outputs = pv_c_fhe(
            probabilities,
            value_c_ciphers,
            layout=layout,
            crypto_context=crypto_context,
        )
        stage_seconds["pv"] = time.perf_counter() - start
        result = DecodeAttentionFHEResult(
            output_ciphers=tuple(outputs),
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds=stage_seconds,
        )
        outputs = None
        return result
    finally:
        for cipher in tuple(scores or ()) + tuple(probabilities or ()) + tuple(
            outputs or ()
        ):
            release_if_supported(cipher)


def _add_output_ciphers(left, right, *, crypto_context):
    from .decode_softmax import _align_cipher_pair

    outputs = []
    try:
        for first, second in zip(left, right, strict=True):
            aligned_left, aligned_right, owned_left, owned_right = _align_cipher_pair(
                first, second, crypto_context=crypto_context
            )
            try:
                outputs.append(
                    crypto_context.fhe.homo_add(
                        aligned_left, aligned_right, crypto_context.context
                    )
                )
            finally:
                if owned_left:
                    release_if_supported(aligned_left)
                if owned_right:
                    release_if_supported(aligned_right)
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in outputs:
            release_if_supported(cipher)


def decode_attention_pages_fhe(
    query_ciphers: tuple[object, ...] | list[object],
    key_pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]],
    value_pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    softmax_config,
    occupied_pages: tuple[int, ...] | list[int],
    layer_idx: int = 0,
    rounds: int | None = None,
    score_scale: float | None = None,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    prefix_bundle=None,
) -> DecodeAttentionFHEResult:
    """1-query attention over CT cache pages and an optional PT Prefix page.

    The Prefix page is never copied into the ciphertext cache.  Its QK and PV
    contributions use the same compact score layout as a normal page, so the
    existing page Softmax can normalize ``[Prefix PT page, content CT pages]``
    with one shared denominator.
    """

    from .decode_softmax import softmax_alg2_row_pages_fhe

    key_pages = tuple(tuple(page) for page in key_pages)
    value_pages = tuple(tuple(page) for page in value_pages)
    occupied_pages = tuple(int(value) for value in occupied_pages)
    prefix_k = int(getattr(prefix_bundle, "prefix_k", 0) or 0)
    if prefix_bundle is not None and prefix_k < 1:
        raise ValueError("prefix_bundle must contain at least one Prefix token.")
    if prefix_bundle is None and len(key_pages) == 1 and len(value_pages) == 1:
        return decode_attention_fhe(
            query_ciphers,
            key_pages[0],
            value_pages[0],
            layout=layout,
            crypto_context=crypto_context,
            softmax_config=softmax_config,
            occupied=occupied_pages[0],
            layer_idx=layer_idx,
            rounds=rounds,
            score_scale=score_scale,
            bootstrap_operator=bootstrap_operator,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_precision_bits=bootstrap_precision_bits,
        )
    if not (
        len(key_pages) == len(value_pages) == len(occupied_pages)
    ):
        raise ValueError("key/value/occupied page counts must match.")
    score_pages: list[tuple[object, ...]] = []
    probability_pages = acc = None
    stage_seconds: dict[str, float] = {}
    wall_start = time.perf_counter()
    try:
        start = time.perf_counter()
        if prefix_bundle is not None:
            prefix_key_payload = getattr(
                prefix_bundle, "packed_key_c", prefix_bundle.key
            )
            score_pages.append(
                qk_prefix_pc_compact_fhe(
                    query_ciphers,
                    prefix_key_payload,
                    layout=layout,
                    crypto_context=crypto_context,
                    prefix_k=prefix_k,
                )
            )
            stage_seconds["qk_prefix"] = time.perf_counter() - start
            start = time.perf_counter()
        for key_ciphers in key_pages:
            score_pages.append(
                qk_c_fhe(
                    query_ciphers,
                    key_ciphers,
                    layout=layout,
                    crypto_context=crypto_context,
                )
            )
        stage_seconds["qk"] = time.perf_counter() - start
        score_occupied = ((prefix_k,) if prefix_bundle is not None else ()) + occupied_pages
        start = time.perf_counter()
        probability_pages = softmax_alg2_row_pages_fhe(
            score_pages,
            layout=layout,
            crypto_context=crypto_context,
            config=softmax_config,
            occupied_pages=score_occupied,
            layer_idx=layer_idx,
            score_scale=score_scale,
            rounds=rounds,
            bootstrap_operator=bootstrap_operator,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_precision_bits=bootstrap_precision_bits,
        )
        stage_seconds["softmax"] = time.perf_counter() - start
        from .decode_softmax import _refresh_row_ciphers_if_short

        probability_pages = [
            _refresh_row_ciphers_if_short(
                page,
                bootstrap_operator=bootstrap_operator,
                need_limbs=6,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag="softmax.probability",
            )
            for page in probability_pages
        ]
        start = time.perf_counter()
        content_offset = 1 if prefix_bundle is not None else 0
        prefix_pv_start = time.perf_counter()
        if prefix_bundle is not None:
            prefix_probabilities = probability_pages[0]
            prefix_value_payload = getattr(
                prefix_bundle, "packed_value_c", prefix_bundle.value
            )
            prefix_outputs = pv_prefix_pc_compact_fhe(
                prefix_probabilities,
                prefix_value_payload,
                layout=layout,
                crypto_context=crypto_context,
                prefix_k=prefix_k,
            )
            probability_pages[0] = ()
            acc = tuple(prefix_outputs)
            stage_seconds["pv_prefix"] = time.perf_counter() - prefix_pv_start
        else:
            stage_seconds["pv_prefix"] = 0.0
        content_pv_start = time.perf_counter()
        for probabilities, value_ciphers in zip(
            probability_pages[content_offset:], value_pages, strict=True
        ):
            contrib = pv_c_fhe(
                probabilities,
                value_ciphers,
                layout=layout,
                crypto_context=crypto_context,
            )
            if acc is None:
                acc = contrib
            else:
                summed = _add_output_ciphers(
                    acc, contrib, crypto_context=crypto_context
                )
                for cipher in acc + contrib:
                    release_if_supported(cipher)
                acc = summed
        stage_seconds["pv_content"] = time.perf_counter() - content_pv_start
        stage_seconds["pv"] = time.perf_counter() - start
        result = DecodeAttentionFHEResult(
            output_ciphers=tuple(acc),
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds=stage_seconds,
        )
        acc = None
        return result
    finally:
        for page in score_pages:
            for cipher in page:
                release_if_supported(cipher)
        if probability_pages is not None:
            for page in probability_pages:
                for cipher in page:
                    release_if_supported(cipher)
        if acc is not None:
            for cipher in acc:
                release_if_supported(cipher)


__all__ = [
    "apply_decode_softmax_public_map_fhe",
    "DecodeAttentionFHEResult",
    "decode_attention_alg2_numpy",
    "decode_attention_alg2_pages_numpy",
    "decode_attention_fhe",
    "decode_attention_pages_fhe",
    "DecodePreparedQKV",
    "decode_prepare_qkv_numpy",
    "decode_attention_numpy",
    "decode_attention_rotations",
    "decode_broadcast_token0_numpy",
    "decode_broadcast_tokens_fhe",
    "decode_reduce_tokens_fhe",
    "prefill_kv_cache_numpy",
    "correct_prefill_emitted_key_c_fhe",
    "correct_prefill_emitted_key_c_numpy",
    "prefill_key_c_emit_correction_mask",
    "pv_c_fhe",
    "pv_c_numpy",
    "qk_c_fhe",
    "qk_prefix_pc_compact_fhe",
    "pv_prefix_pc_compact_fhe",
    "folded_key_c_to_real_c_fhe",
    "qk_c_numpy",
    "softmax_alg2_row_numpy",
    "softmax_row_numpy",
    "output_c_to_decode_fm_fhe",
    "output_c_to_decode_fm_numpy",
    "decode_output_c_rotations",
]
