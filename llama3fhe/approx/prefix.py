"""Prefix attention as two partitions: 128×k (PC) + 128×128 (CC).

Logical scores are ``[S_p | S_c]`` with shape ``[H, Q, k+Q]``.  Packed
oracles never build that concat: ``S_c`` stays ``S_Δ`` and ``S_p`` occupies
C-form feature rows ``0..k-1`` (token = query).  Softmax shares one row
sum ``Z = Z_p + Z_c``; PV adds two ``O_C`` tensors of length ``Q``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.kv_cache import LayerKvCache

from .attention import (
    apply_rope_heads,
    pv_delta_numpy,
    qk_delta_numpy,
    rope_cos_sin,
)


@dataclass(frozen=True)
class PrefixAttentionResult:
    scores_p: np.ndarray
    scores_c: np.ndarray
    probabilities_p: np.ndarray
    probabilities_c: np.ndarray
    output_p: np.ndarray
    output_c: np.ndarray
    output: np.ndarray
    scores_concat: np.ndarray
    probabilities_concat: np.ndarray


def prefix_content_causal_mask(
    q_len: int,
    *,
    prefix_k: int,
    dtype=bool,
) -> np.ndarray:
    """Content query ``i`` (abs pos ``k+i``) sees prefix keys and content ``0..i``."""

    q_len = int(q_len)
    prefix_k = int(prefix_k)
    if q_len <= 0 or prefix_k < 0:
        raise ValueError(f"q_len={q_len} prefix_k={prefix_k}")
    kv_len = prefix_k + q_len
    query_abs = np.arange(q_len, dtype=np.int64)[:, None] + prefix_k
    key_abs = np.arange(kv_len, dtype=np.int64)[None, :]
    return (key_abs <= query_abs).astype(dtype, copy=False)


def _gqa_repeat(kv: np.ndarray, *, layout: AttentionPairLayout) -> np.ndarray:
    ratio = int(layout.gqa_ratio)
    heads = np.arange(int(layout.query_heads), dtype=np.int64) // ratio
    return kv[:, heads, :]


def _as_kv_heads(
    values: np.ndarray,
    *,
    layout: AttentionPairLayout,
    tokens: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    heads = int(layout.key_value_heads)
    dim = int(layout.head_dim)
    flat = (int(tokens), heads * dim)
    shaped = (int(tokens), heads, dim)
    if values.shape == flat:
        return values.reshape(shaped)
    if values.shape == shaped:
        return values
    raise ValueError(f"{name} must have shape {flat} or {shaped}, got {values.shape}.")


def _as_query_heads(
    values: np.ndarray,
    *,
    layout: AttentionPairLayout,
    name: str = "query",
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    q_len = int(layout.seq_len)
    heads = int(layout.query_heads)
    dim = int(layout.head_dim)
    flat = (q_len, heads * dim)
    shaped = (q_len, heads, dim)
    if values.shape == flat:
        return values.reshape(shaped)
    if values.shape == shaped:
        return values
    raise ValueError(f"{name} must have shape {flat} or {shaped}, got {values.shape}.")


def qk_prefix_numpy(
    query: np.ndarray,
    key_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
    scale: bool = True,
) -> np.ndarray:
    """``S_p[h, q, j] = ⟨Q[q,h], K_p[j,h_kv]⟩ / √d``."""

    query = _as_query_heads(query, layout=layout)
    prefix_k = int(key_prefix.shape[0]) if key_prefix.ndim >= 1 else 0
    key = _as_kv_heads(
        key_prefix, layout=layout, tokens=prefix_k, name="key_prefix"
    )
    key_for_q = _gqa_repeat(key, layout=layout)
    scores = np.einsum("qhd,khd->hqk", query, key_for_q, optimize=True)
    if scale:
        scores = scores / np.float32(math.sqrt(float(layout.head_dim)))
    return np.asarray(scores, dtype=np.float32)


def qk_content_numpy(
    query: np.ndarray,
    key_content: np.ndarray,
    *,
    layout: AttentionPairLayout,
    scale: bool = True,
) -> np.ndarray:
    """Square content scores ``[H, Q, Q]``, causal applied later."""

    query = _as_query_heads(query, layout=layout)
    key = _as_kv_heads(
        key_content,
        layout=layout,
        tokens=int(layout.seq_len),
        name="key_content",
    )
    key_for_q = _gqa_repeat(key, layout=layout)
    scores = np.einsum("qhd,khd->hqk", query, key_for_q, optimize=True)
    if scale:
        scores = scores / np.float32(math.sqrt(float(layout.head_dim)))
    return np.asarray(scores, dtype=np.float32)


def dual_softmax_numpy(
    scores_p: np.ndarray,
    scores_c: np.ndarray,
    *,
    prefix_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Row softmax over prefix columns (always valid) plus causal content."""

    scores_p = np.asarray(scores_p, dtype=np.float64)
    scores_c = np.asarray(scores_c, dtype=np.float64)
    if scores_p.ndim != 3 or scores_c.ndim != 3:
        raise ValueError("scores_p and scores_c must be [H, Q, ·].")
    if scores_p.shape[:2] != scores_c.shape[:2]:
        raise ValueError(
            f"query/head mismatch: S_p {scores_p.shape} vs S_c {scores_c.shape}."
        )
    if int(scores_p.shape[-1]) != int(prefix_k):
        raise ValueError(
            f"prefix_k={prefix_k} vs S_p keys {scores_p.shape[-1]}."
        )
    q_len = int(scores_c.shape[-1])
    if q_len != int(scores_c.shape[1]):
        raise ValueError(f"content scores must be square in Q, got {scores_c.shape}.")
    valid_c = np.arange(q_len, dtype=np.int64)[None, :] <= np.arange(
        q_len, dtype=np.int64
    )[:, None]
    masked_c = np.where(valid_c[None, :, :], scores_c, -np.inf)
    row_max = np.maximum(
        np.max(scores_p, axis=-1, keepdims=True),
        np.max(masked_c, axis=-1, keepdims=True),
    )
    exp_p = np.exp(scores_p - row_max)
    exp_c = np.where(valid_c[None, :, :], np.exp(masked_c - row_max), 0.0)
    denom = np.sum(exp_p, axis=-1, keepdims=True) + np.sum(
        exp_c, axis=-1, keepdims=True
    )
    tiny = np.finfo(np.float64).tiny
    probabilities_p = exp_p / np.maximum(denom, tiny)
    probabilities_c = exp_c / np.maximum(denom, tiny)
    return (
        np.asarray(probabilities_p, dtype=np.float32),
        np.asarray(probabilities_c, dtype=np.float32),
    )


def concat_softmax_numpy(
    scores_p: np.ndarray,
    scores_c: np.ndarray,
    *,
    prefix_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Golden 128×(k+128) softmax used only as a numerical reference."""

    scores_p = np.asarray(scores_p, dtype=np.float64)
    scores_c = np.asarray(scores_c, dtype=np.float64)
    scores = np.concatenate([scores_p, scores_c], axis=-1)
    q_len = int(scores_c.shape[1])
    valid = prefix_content_causal_mask(q_len, prefix_k=int(prefix_k))
    masked = np.where(valid[None, :, :], scores, -np.inf)
    row_max = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.where(valid[None, :, :], np.exp(masked - row_max), 0.0)
    probabilities = exponentials / np.maximum(
        np.sum(exponentials, axis=-1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(probabilities, dtype=np.float32),
    )


def pv_prefix_numpy(
    probabilities_p: np.ndarray,
    value_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> np.ndarray:
    """``O_p[q,h] = Σ_j P_p[h,q,j] V_p[j,h_kv]`` → ``[Q, Hq*d]``."""

    probabilities_p = np.asarray(probabilities_p, dtype=np.float32)
    prefix_k = int(probabilities_p.shape[-1])
    value = _as_kv_heads(
        value_prefix, layout=layout, tokens=prefix_k, name="value_prefix"
    )
    value_for_q = _gqa_repeat(value, layout=layout)
    output = np.einsum("hqk,khd->qhd", probabilities_p, value_for_q, optimize=True)
    return output.reshape(int(layout.seq_len), -1).astype(np.float32, copy=False)


def pv_content_numpy(
    probabilities_c: np.ndarray,
    value_content: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> np.ndarray:
    value = _as_kv_heads(
        value_content,
        layout=layout,
        tokens=int(layout.seq_len),
        name="value_content",
    )
    value_for_q = _gqa_repeat(value, layout=layout)
    output = np.einsum(
        "hqk,khd->qhd",
        np.asarray(probabilities_c, dtype=np.float32),
        value_for_q,
        optimize=True,
    )
    return output.reshape(int(layout.seq_len), -1).astype(np.float32, copy=False)


def attention_prefix_gqa_numpy(
    query: np.ndarray,
    key_prefix: np.ndarray,
    key_content: np.ndarray,
    value_prefix: np.ndarray,
    value_content: np.ndarray,
    *,
    layout: AttentionPairLayout,
    scale: bool = True,
) -> PrefixAttentionResult:
    """Split PC+CC GQA.  Output length is content ``Q``, not ``k+Q``."""

    scores_p = qk_prefix_numpy(
        query, key_prefix, layout=layout, scale=scale
    )
    scores_c = qk_content_numpy(
        query, key_content, layout=layout, scale=scale
    )
    prefix_k = int(scores_p.shape[-1])
    probabilities_p, probabilities_c = dual_softmax_numpy(
        scores_p, scores_c, prefix_k=prefix_k
    )
    output_p = pv_prefix_numpy(probabilities_p, value_prefix, layout=layout)
    output_c = pv_content_numpy(probabilities_c, value_content, layout=layout)
    scores_concat, probabilities_concat = concat_softmax_numpy(
        scores_p, scores_c, prefix_k=prefix_k
    )
    return PrefixAttentionResult(
        scores_p=scores_p,
        scores_c=scores_c,
        probabilities_p=probabilities_p,
        probabilities_c=probabilities_c,
        output_p=output_p,
        output_c=output_c,
        output=np.asarray(output_p + output_c, dtype=np.float32),
        scores_concat=scores_concat,
        probabilities_concat=probabilities_concat,
    )


def concat_attention_gqa_numpy(
    query: np.ndarray,
    key_prefix: np.ndarray,
    key_content: np.ndarray,
    value_prefix: np.ndarray,
    value_content: np.ndarray,
    *,
    layout: AttentionPairLayout,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single matmul over concatenated KV; golden for the split path."""

    query = _as_query_heads(query, layout=layout)
    prefix_k = int(np.asarray(key_prefix).shape[0])
    key_p = _as_kv_heads(
        key_prefix, layout=layout, tokens=prefix_k, name="key_prefix"
    )
    key_c = _as_kv_heads(
        key_content,
        layout=layout,
        tokens=int(layout.seq_len),
        name="key_content",
    )
    value_p = _as_kv_heads(
        value_prefix, layout=layout, tokens=prefix_k, name="value_prefix"
    )
    value_c = _as_kv_heads(
        value_content,
        layout=layout,
        tokens=int(layout.seq_len),
        name="value_content",
    )
    key = np.concatenate([key_p, key_c], axis=0)
    value = np.concatenate([value_p, value_c], axis=0)
    key_for_q = _gqa_repeat(key, layout=layout)
    value_for_q = _gqa_repeat(value, layout=layout)
    scores = np.einsum("qhd,khd->hqk", query, key_for_q, optimize=True)
    if scale:
        scores = scores / np.float32(math.sqrt(float(layout.head_dim)))
    q_len = int(layout.seq_len)
    valid = prefix_content_causal_mask(q_len, prefix_k=prefix_k)
    masked = np.where(valid[None, :, :], scores, -np.inf)
    row_max = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.where(valid[None, :, :], np.exp(masked - row_max), 0.0)
    probabilities = exponentials / np.maximum(
        np.sum(exponentials, axis=-1, keepdims=True),
        np.finfo(np.float64).tiny,
    )
    output = np.einsum("hqk,khd->qhd", probabilities, value_for_q, optimize=True)
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(probabilities, dtype=np.float32),
        output.reshape(q_len, -1).astype(np.float32, copy=False),
    )


def rope_prefix_and_content(
    query: np.ndarray,
    key_prefix: np.ndarray,
    key_content: np.ndarray,
    *,
    layout: AttentionPairLayout,
    prefix_k: int,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefix K at positions ``0..k-1``; content Q/K at ``k..k+Q-1``."""

    prefix_k = int(prefix_k)
    cos_p, sin_p = rope_cos_sin(
        seq_len=prefix_k,
        head_dim=int(layout.head_dim),
        theta=float(theta),
        start_pos=0,
    )
    cos_c, sin_c = rope_cos_sin(
        seq_len=int(layout.seq_len),
        head_dim=int(layout.head_dim),
        theta=float(theta),
        start_pos=prefix_k,
    )
    query_rope = apply_rope_heads(
        np.asarray(query, dtype=np.float32).reshape(
            int(layout.seq_len), int(layout.query_heads) * int(layout.head_dim)
        ),
        cos_c,
        sin_c,
        num_heads=int(layout.query_heads),
        head_dim=int(layout.head_dim),
    )
    key_p = _as_kv_heads(
        key_prefix, layout=layout, tokens=prefix_k, name="key_prefix"
    )
    key_c = _as_kv_heads(
        key_content,
        layout=layout,
        tokens=int(layout.seq_len),
        name="key_content",
    )
    key_prefix_rope = apply_rope_heads(
        key_p.reshape(prefix_k, -1),
        cos_p,
        sin_p,
        num_heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
    )
    key_content_rope = apply_rope_heads(
        key_c.reshape(int(layout.seq_len), -1),
        cos_c,
        sin_c,
        num_heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
    )
    return query_rope, key_prefix_rope, key_content_rope


def packed_prefix_softmax_oracle(
    scores_p: np.ndarray,
    scores_c: np.ndarray,
    *,
    layout: AttentionPairLayout,
    prefix_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """A.2: pack ``S_p`` / ``S_Δ``, unpack, dual-softmax, pack ``P`` back."""

    packed_p = layout.pack_prefix_scores_prefill(scores_p)
    packed_c = layout.pack_score_delta(scores_c)
    round_p = layout.unpack_prefix_scores_prefill(packed_p, prefix_k=prefix_k)
    round_c = layout.unpack_score_delta(packed_c)
    probs_p, probs_c = dual_softmax_numpy(
        round_p, round_c, prefix_k=int(prefix_k)
    )
    packed_pp = layout.pack_prefix_scores_prefill(probs_p)
    packed_pc = layout.pack_score_delta(probs_c)
    out_p = layout.unpack_prefix_scores_prefill(packed_pp, prefix_k=prefix_k)
    out_c = layout.unpack_score_delta(packed_pc)
    err = {
        "scores_p_roundtrip": float(np.max(np.abs(round_p - scores_p))),
        "scores_c_roundtrip": float(np.max(np.abs(round_c - scores_c))),
        "probs_p_roundtrip": float(np.max(np.abs(out_p - probs_p))),
        "probs_c_roundtrip": float(np.max(np.abs(out_c - probs_c))),
    }
    return out_p, out_c, err


def packed_content_qk_matches_logical(
    query: np.ndarray,
    key_content: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, float]:
    """Content QK through the production Δ kernel vs logical ``S_c``."""

    query = np.asarray(query, dtype=np.float32).reshape(
        int(layout.seq_len), int(layout.query_heads) * int(layout.head_dim)
    )
    key = np.asarray(key_content, dtype=np.float32).reshape(
        int(layout.seq_len), int(layout.key_value_heads) * int(layout.head_dim)
    )
    dummy_v = np.zeros_like(key)
    payload = layout.pack_qkv(query, key, dummy_v)
    score_delta = qk_delta_numpy(payload.query_c, payload.key_delta, layout=layout)
    packed_scores = layout.unpack_score_delta(score_delta) / np.float32(
        math.sqrt(float(layout.head_dim))
    )
    logical = qk_content_numpy(query, key, layout=layout, scale=True)
    return packed_scores, float(np.max(np.abs(packed_scores - logical)))


def packed_content_pv_matches_logical(
    probabilities_c: np.ndarray,
    value_content: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, float]:
    value = np.asarray(value_content, dtype=np.float32).reshape(
        int(layout.seq_len), int(layout.key_value_heads) * int(layout.head_dim)
    )
    dummy_q = np.zeros(
        (int(layout.seq_len), int(layout.query_heads) * int(layout.head_dim)),
        dtype=np.float32,
    )
    dummy_k = np.zeros_like(value)
    payload = layout.pack_qkv(dummy_q, dummy_k, value)
    packed_p = layout.pack_score_delta(probabilities_c)
    output_c = pv_delta_numpy(packed_p, payload.value_delta, layout=layout)
    unpacked = layout.unpack_output_c(output_c)
    logical = pv_content_numpy(probabilities_c, value, layout=layout)
    return unpacked, float(np.max(np.abs(unpacked - logical)))


def _c_grid(packed: np.ndarray, *, layout: AttentionPairLayout, pairs: int, name: str) -> np.ndarray:
    packed = np.asarray(packed)
    if packed.shape == (int(pairs), int(layout.payload_slots)):
        base = packed
    else:
        base = layout._base_payload(packed, pairs=int(pairs), name=name)
    return base.reshape(
        int(pairs), int(layout.head_dim), int(layout.seq_len), 2
    )


def _repeat_c_grid(view: np.ndarray, *, layout: AttentionPairLayout) -> np.ndarray:
    pairs = int(view.shape[0])
    base = np.asarray(view, dtype=np.float64).reshape(pairs, layout.payload_slots)
    return layout._repeat_payload(base)


def broadcast_prefix_token_c(
    token: np.ndarray,
    *,
    layout: AttentionPairLayout,
    dtype=np.float64,
) -> np.ndarray:
    """Replicate one KV token across every C token column (plaintext PC broadcast)."""

    heads = int(layout.key_value_heads)
    dim = int(layout.head_dim)
    token = np.asarray(token, dtype=np.float32)
    if token.shape == (heads * dim,):
        token = token.reshape(heads, dim)
    elif token.shape != (heads, dim):
        raise ValueError(
            f"prefix KV token must have shape {(heads, dim)} or {(heads * dim,)}, "
            f"got {token.shape}."
        )
    pages = np.repeat(token[None, :, :], int(layout.seq_len), axis=0)
    return layout.pack_key_c(pages.reshape(int(layout.seq_len), heads * dim), dtype=dtype)


def qk_prefix_packed_numpy(
    query: np.ndarray,
    key_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> np.ndarray:
    """Unscaled PC QK on C grids: ``Q_C ⊙ broadcast(K_p[j])``, feature-reduce onto row ``j``."""

    query = _as_query_heads(query, layout=layout)
    prefix_k = int(np.asarray(key_prefix).shape[0])
    key = _as_kv_heads(key_prefix, layout=layout, tokens=prefix_k, name="key_prefix")
    q_view = _c_grid(
        layout.pack_query_c(query.reshape(int(layout.seq_len), -1)),
        layout=layout,
        pairs=int(layout.query_pair_count),
        name="Q_C",
    )
    cache = pack_prefix_kv_c(key, np.zeros_like(key), layout=layout)
    k_view = _c_grid(
        cache.pages[0].packed_keys(),
        layout=layout,
        pairs=int(layout.key_value_pair_count),
        name="K_p",
    )
    scores = np.zeros_like(q_view, dtype=np.float64)
    ratio = int(layout.gqa_ratio)
    for kv_pair in range(int(layout.key_value_pair_count)):
        for offset in range(ratio):
            q_pair = q_view[kv_pair * ratio + offset]
            for prefix_j in range(prefix_k):
                k_vec = k_view[kv_pair, :, prefix_j, :]
                scores[kv_pair * ratio + offset, prefix_j] = np.einsum(
                    "dtl,dl->tl", q_pair, k_vec, optimize=True
                )
    return _repeat_c_grid(scores, layout=layout)


def pv_prefix_packed_numpy(
    probabilities_p: np.ndarray,
    value_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> np.ndarray:
    """PC PV on C grids: ``broadcast(P_p[j]) ⊙ broadcast(V_p[j])``, summed over ``j``."""

    probabilities_p = np.asarray(probabilities_p, dtype=np.float32)
    prefix_k = int(probabilities_p.shape[-1])
    value = _as_kv_heads(
        value_prefix, layout=layout, tokens=prefix_k, name="value_prefix"
    )
    p_view = _c_grid(
        layout.pack_prefix_scores_prefill(probabilities_p),
        layout=layout,
        pairs=int(layout.query_pair_count),
        name="P_p",
    )
    dummy_key = np.zeros_like(value)
    cache = pack_prefix_kv_c(dummy_key, value, layout=layout)
    v_view = _c_grid(
        cache.pages[0].packed_values(),
        layout=layout,
        pairs=int(layout.key_value_pair_count),
        name="V_p",
    )
    output = np.zeros_like(p_view, dtype=np.float64)
    ratio = int(layout.gqa_ratio)
    for kv_pair in range(int(layout.key_value_pair_count)):
        for offset in range(ratio):
            q_pair = kv_pair * ratio + offset
            output[q_pair] = np.einsum(
                "jtl,djl->dtl",
                p_view[q_pair, :prefix_k],
                v_view[kv_pair, :, :prefix_k, :],
                optimize=True,
            )
    return _repeat_c_grid(output, layout=layout)


def packed_prefix_qk_matches_logical(
    query: np.ndarray,
    key_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, float]:
    packed = qk_prefix_packed_numpy(query, key_prefix, layout=layout)
    prefix_k = int(np.asarray(key_prefix).shape[0])
    unpacked = layout.unpack_prefix_scores_prefill(packed, prefix_k=prefix_k)
    logical = qk_prefix_numpy(query, key_prefix, layout=layout, scale=False)
    return unpacked, float(np.max(np.abs(unpacked - logical)))


def packed_prefix_pv_matches_logical(
    probabilities_p: np.ndarray,
    value_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, float]:
    packed = pv_prefix_packed_numpy(
        probabilities_p, value_prefix, layout=layout
    )
    unpacked = layout.unpack_output_c(packed)
    logical = pv_prefix_numpy(probabilities_p, value_prefix, layout=layout)
    return unpacked, float(np.max(np.abs(unpacked - logical)))


def packed_prefix_attention_numpy(
    query: np.ndarray,
    key_prefix: np.ndarray,
    key_content: np.ndarray,
    value_prefix: np.ndarray,
    value_content: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, dict[str, float]]:
    """PC QK/PV on C grids, CC through production Δ; output is content ``O_C``."""

    scale = np.float32(math.sqrt(float(layout.head_dim)))
    prefix_k = int(np.asarray(key_prefix).shape[0])
    packed_sp = qk_prefix_packed_numpy(query, key_prefix, layout=layout)
    scores_p = layout.unpack_prefix_scores_prefill(packed_sp, prefix_k=prefix_k) / scale
    scores_c, qk_c_err = packed_content_qk_matches_logical(
        query, key_content, layout=layout
    )
    probabilities_p, probabilities_c = dual_softmax_numpy(
        scores_p, scores_c, prefix_k=prefix_k
    )
    output_p, pv_p_err = packed_prefix_pv_matches_logical(
        probabilities_p, value_prefix, layout=layout
    )
    output_c, pv_c_err = packed_content_pv_matches_logical(
        probabilities_c, value_content, layout=layout
    )
    output = np.asarray(output_p + output_c, dtype=np.float32)
    _, qk_p_err = packed_prefix_qk_matches_logical(
        query, key_prefix, layout=layout
    )
    err = {
        "qk_p": float(qk_p_err),
        "qk_c": float(qk_c_err),
        "pv_p": float(pv_p_err),
        "pv_c": float(pv_c_err),
        "output_tokens": float(output.shape[0]),
    }
    return output, err


def pack_prefix_kv_c(
    key_prefix: np.ndarray,
    value_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> LayerKvCache:
    """C-form page occupying token columns ``0..k-1`` (plaintext cache object)."""

    prefix_k = int(np.asarray(key_prefix).shape[0])
    key = _as_kv_heads(
        key_prefix, layout=layout, tokens=prefix_k, name="key_prefix"
    )
    value = _as_kv_heads(
        value_prefix, layout=layout, tokens=prefix_k, name="value_prefix"
    )
    cache = LayerKvCache(layout=layout)
    for token in range(prefix_k):
        cache.append(key[token], value[token])
    return cache


@dataclass(frozen=True)
class PrefixKVBundle:
    """Plaintext prefix cache page.  Always C-form, never Delta.

    Slot formula is ``2*(feature*S + token)+lane``.  Only token columns
    ``0..k-1`` are occupied; the rest of the page is zero.  CKKS encodes
    this as a plaintext (``homo_mul_pt``), not a ciphertext.
    """

    key: np.ndarray
    value: np.ndarray
    packed_key_c: np.ndarray
    packed_value_c: np.ndarray
    prefix_k: int

    @classmethod
    def from_logical(
        cls,
        key_prefix: np.ndarray,
        value_prefix: np.ndarray,
        *,
        layout: AttentionPairLayout,
    ) -> "PrefixKVBundle":
        cache = pack_prefix_kv_c(key_prefix, value_prefix, layout=layout)
        key = cache.pages[0].logical_keys()
        value = cache.pages[0].logical_values()
        return cls(
            key=np.asarray(key, dtype=np.float32),
            value=np.asarray(value, dtype=np.float32),
            packed_key_c=np.asarray(cache.pages[0].packed_keys()),
            packed_value_c=np.asarray(cache.pages[0].packed_values()),
            prefix_k=int(cache.occupied),
        )


def dual_softmax_alg2_numpy(
    scores_p: np.ndarray,
    scores_c: np.ndarray,
    *,
    layout: AttentionPairLayout,
    config,
    layer_idx: int = 0,
    prefix_k: int | None = None,
    token_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Chebyshev Alg2 on concat ``[S_p | S_c]``, sliced back into partitions.

    ``scores_*`` are unscaled GQA dots; the public map applies ``1/√d``.
    """

    from llama3fhe.approx.layer import softmax_alg2_poly_scores_numpy

    scores_p = np.asarray(scores_p, dtype=np.float64)
    scores_c = np.asarray(scores_c, dtype=np.float64)
    prefix_k = int(scores_p.shape[-1] if prefix_k is None else prefix_k)
    if int(scores_p.shape[-1]) != prefix_k:
        raise ValueError(f"prefix_k={prefix_k} vs S_p {scores_p.shape}")
    scores = np.concatenate([scores_p, scores_c], axis=-1)
    valid = prefix_content_causal_mask(
        int(scores_c.shape[1]), prefix_k=prefix_k
    )
    if token_valid_mask is not None:
        q_len = int(scores_c.shape[1])
        token_valid = np.asarray(token_valid_mask, dtype=bool).reshape(-1)
        if token_valid.shape != (q_len,):
            raise ValueError(
                "token_valid_mask must have shape "
                f"{(q_len,)}, got {token_valid.shape}."
            )
        content_valid = valid[:, prefix_k:]
        content_valid &= token_valid[None, :]
        content_valid &= token_valid[:, None]
        invalid_queries = np.flatnonzero(~token_valid)
        content_valid[invalid_queries, invalid_queries] = True
    probabilities, report = softmax_alg2_poly_scores_numpy(
        scores,
        config=config,
        layer_idx=int(layer_idx),
        head_dim=int(layout.head_dim),
        valid=valid,
    )
    return (
        np.asarray(probabilities[..., :prefix_k], dtype=np.float32),
        np.asarray(probabilities[..., prefix_k:], dtype=np.float32),
        report,
    )
