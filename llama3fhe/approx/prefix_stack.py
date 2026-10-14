"""Plaintext Llama-3 stack that matches Holmes FHE + Prefix.

FHE never encrypts sink tokens.  The matching clear path is the ``kv_cache``
protocol, not a 130-token concat Prefill:

* PP: k copies of ``\\n`` (id 198) run an exact transformer; each layer
  snapshots RoPE'd ``K_p`` / un-RoPE'd ``V_p`` into a C-form
  :class:`PrefixKVBundle`.
* Content: the 128 Paris tokens stay length 128.  RMSN / MLP see only those
  rows.  RoPE starts at ``k``.  Attention is dual-source ``128×k`` +
  ``128×128``.

Interval collection samples **content activations only**.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from llama3fhe.approx.attention import apply_rope_heads, rope_cos_sin
from llama3fhe.approx.prefix import (
    PrefixKVBundle,
    dual_softmax_numpy,
    prefix_content_causal_mask,
    pv_content_numpy,
    pv_prefix_numpy,
    qk_content_numpy,
    qk_prefix_numpy,
)
from llama3fhe.approx.rmsnorm import rmsnorm_numpy
from llama3fhe.layouts.attention import AttentionPairLayout

NEWLINE_TOKEN_ID = 198


def prefix_token_ids(prefix_k: int, *, token_id: int = NEWLINE_TOKEN_ID) -> np.ndarray:
    prefix_k = int(prefix_k)
    if prefix_k < 0:
        raise ValueError(f"prefix_k must be non-negative, got {prefix_k}.")
    return np.full((prefix_k,), int(token_id), dtype=np.int64)


def exact_causal_gqa(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal GQA on an arbitrary sequence length (used for prefix PP)."""

    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    seq = int(query.shape[0])
    query_heads = int(query_heads)
    key_value_heads = int(key_value_heads)
    head_dim = int(head_dim)
    if seq != int(key.shape[0]) or seq != int(value.shape[0]):
        raise ValueError("Q/K/V token counts must match for prefix self-attention.")
    if query_heads % key_value_heads:
        raise ValueError("query_heads must be divisible by key_value_heads.")

    def heads(values: np.ndarray, n_heads: int, name: str) -> np.ndarray:
        flat = (seq, n_heads * head_dim)
        shaped = (seq, n_heads, head_dim)
        if values.shape == flat:
            return values.reshape(shaped)
        if values.shape == shaped:
            return values
        raise ValueError(f"{name} must have shape {flat} or {shaped}, got {values.shape}.")

    query_h = heads(query, query_heads, "query")
    key_h = heads(key, key_value_heads, "key")
    value_h = heads(value, key_value_heads, "value")
    ratio = query_heads // key_value_heads
    key_for_q = key_h[:, np.arange(query_heads) // ratio, :]
    value_for_q = value_h[:, np.arange(query_heads) // ratio, :]
    scores = np.einsum("qhd,khd->hqk", query_h, key_for_q, optimize=True)
    scores = scores / np.float32(math.sqrt(float(head_dim)))
    queries = np.arange(seq, dtype=np.int64)[:, None]
    keys = np.arange(seq, dtype=np.int64)[None, :]
    valid = keys <= queries
    masked = np.where(valid[None, :, :], scores, -np.inf)
    row_max = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.where(valid[None, :, :], np.exp(masked - row_max), 0.0)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    output = np.einsum("hqk,khd->qhd", probabilities, value_for_q, optimize=True)
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(probabilities, dtype=np.float32),
        output.reshape(seq, -1).astype(np.float32, copy=False),
    )


def _project_qkv(hidden: np.ndarray, weights):
    hidden = np.asarray(hidden, dtype=np.float32)
    query = hidden @ np.asarray(weights.q_proj, dtype=np.float32).T
    key = hidden @ np.asarray(weights.k_proj, dtype=np.float32).T
    value = hidden @ np.asarray(weights.v_proj, dtype=np.float32).T
    return query, key, value


def _exact_mlp(residual: np.ndarray, weights, *, eps: float):
    mlp_in = rmsnorm_numpy(
        residual, weights.post_attention_layernorm, eps=float(eps)
    )
    gate = mlp_in @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = mlp_in @ np.asarray(weights.up_proj, dtype=np.float32).T
    silu = gate.astype(np.float64) / (1.0 + np.exp(-gate.astype(np.float64)))
    output = residual + (silu * up).astype(np.float32) @ np.asarray(
        weights.down_proj, dtype=np.float32
    ).T
    return np.asarray(output, dtype=np.float32), np.asarray(gate, dtype=np.float64)


def run_exact_prefix_layer(
    hidden: np.ndarray,
    *,
    weights,
    layout: AttentionPairLayout,
    rope_theta: float = 500000.0,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One exact PP layer on k sink tokens.  Returns ``(hidden, K_rope, V)``."""

    hidden = np.asarray(hidden, dtype=np.float32)
    attn_in = rmsnorm_numpy(hidden, weights.input_layernorm, eps=float(eps))
    query, key, value = _project_qkv(attn_in, weights)
    prefix_k = int(hidden.shape[0])
    cos, sin = rope_cos_sin(
        seq_len=prefix_k,
        head_dim=int(layout.head_dim),
        theta=float(rope_theta),
        start_pos=0,
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
    _, _, attn_out = exact_causal_gqa(
        query,
        key,
        value,
        query_heads=int(layout.query_heads),
        key_value_heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
    )
    residual = hidden + attn_out @ np.asarray(weights.o_proj, dtype=np.float32).T
    output, _ = _exact_mlp(residual, weights, eps=float(eps))
    return output, key, value


def build_prefix_kv_bundles(
    prefix_hidden: np.ndarray,
    layer_weights: Sequence[Any],
    *,
    layout: AttentionPairLayout,
    rope_theta: float = 500000.0,
    eps: float = 1e-5,
) -> tuple[list[PrefixKVBundle], np.ndarray]:
    """Run the k-token PP stack and snapshot each layer's C-form prefix page."""

    hidden = np.asarray(prefix_hidden, dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[0] < 1:
        raise ValueError(f"prefix_hidden must be [k, hidden], got {hidden.shape}.")
    bundles: list[PrefixKVBundle] = []
    for weights in layer_weights:
        hidden, key, value = run_exact_prefix_layer(
            hidden,
            weights=weights,
            layout=layout,
            rope_theta=float(rope_theta),
            eps=float(eps),
        )
        bundles.append(PrefixKVBundle.from_logical(key, value, layout=layout))
    return bundles, hidden


def run_exact_content_layer(
    hidden: np.ndarray,
    *,
    weights,
    layout: AttentionPairLayout,
    prefix_bundle: PrefixKVBundle | None = None,
    rope_theta: float = 500000.0,
    eps: float = 1e-5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exact content layer matching FHE geometry.  Samples are content-only."""

    hidden = np.asarray(hidden, dtype=np.float32)
    if int(hidden.shape[0]) != int(layout.seq_len):
        raise ValueError(
            "content Prefill sampling requires seq_len="
            f"{layout.seq_len}, got {hidden.shape[0]}."
        )
    var1 = np.mean(np.square(hidden, dtype=np.float64), axis=-1)
    attn_in = rmsnorm_numpy(hidden, weights.input_layernorm, eps=float(eps))
    query, key, value = _project_qkv(attn_in, weights)
    prefix_k = 0 if prefix_bundle is None else int(prefix_bundle.prefix_k)
    cos, sin = rope_cos_sin(
        seq_len=int(hidden.shape[0]),
        head_dim=int(layout.head_dim),
        theta=float(rope_theta),
        start_pos=int(prefix_k),
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
    if prefix_k:
        scores_p = qk_prefix_numpy(
            query, prefix_bundle.key, layout=layout, scale=True
        )
        scores_c = qk_content_numpy(query, key, layout=layout, scale=True)
        probabilities_p, probabilities_c = dual_softmax_numpy(
            scores_p, scores_c, prefix_k=prefix_k
        )
        attn_out = pv_prefix_numpy(
            probabilities_p, prefix_bundle.value, layout=layout
        ) + pv_content_numpy(probabilities_c, value, layout=layout)
        scores = np.concatenate([scores_p, scores_c], axis=-1)
    else:
        from llama3fhe.approx.attention import attention_gqa_numpy

        scores, _, attn_out = attention_gqa_numpy(
            query, key, value, layout=layout
        )
        scores_p = np.zeros(
            (int(layout.query_heads), int(layout.seq_len), 0), dtype=np.float32
        )
        scores_c = scores
    residual = hidden + attn_out @ np.asarray(weights.o_proj, dtype=np.float32).T
    var2 = np.mean(np.square(residual, dtype=np.float64), axis=-1)
    output, gate = _exact_mlp(residual, weights, eps=float(eps))
    return np.asarray(output, dtype=np.float32), {
        "rmsn1_var": np.asarray(var1, dtype=np.float64),
        "rmsn2_var": np.asarray(var2, dtype=np.float64),
        "silu_gate": np.asarray(gate, dtype=np.float64).reshape(-1),
        "scores": np.asarray(scores, dtype=np.float64),
        "scores_p": np.asarray(scores_p, dtype=np.float64),
        "scores_c": np.asarray(scores_c, dtype=np.float64),
        "prefix_k": np.asarray(prefix_k),
    }


def run_exact_content_stack(
    content_hidden: np.ndarray,
    layer_weights: Sequence[Any],
    *,
    layout: AttentionPairLayout,
    prefix_bundles: Sequence[PrefixKVBundle] | None = None,
    rope_theta: float = 500000.0,
    eps: float = 1e-5,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    hidden = np.asarray(content_hidden, dtype=np.float32)
    samples: list[dict[str, np.ndarray]] = []
    for index, weights in enumerate(layer_weights):
        bundle = None if prefix_bundles is None else prefix_bundles[index]
        hidden, layer_samples = run_exact_content_layer(
            hidden,
            weights=weights,
            layout=layout,
            prefix_bundle=bundle,
            rope_theta=float(rope_theta),
            eps=float(eps),
        )
        samples.append(layer_samples)
    return hidden, samples


def run_poly_content_stack(
    content_hidden: np.ndarray,
    layer_weights: Sequence[Any],
    approximations: Sequence[Any],
    *,
    layout: AttentionPairLayout,
    prefix_bundles: Sequence[PrefixKVBundle] | None = None,
    rope_theta: float = 500000.0,
) -> np.ndarray:
    """Poly RMSN/SiLU + exact dual-source softmax; same geometry as FHE."""

    from llama3fhe.approx.layer import poly_transformer_layer_numpy

    hidden = np.asarray(content_hidden, dtype=np.float32)
    for index, weights in enumerate(layer_weights):
        bundle = None if prefix_bundles is None else prefix_bundles[index]
        hidden = poly_transformer_layer_numpy(
            hidden,
            weights=weights,
            approximations=approximations[index],
            attention_layout=layout,
            rope_theta=float(rope_theta),
            prefix_bundle=bundle,
        ).hidden
    return hidden


def softmax_valid_mask(scores: np.ndarray, *, prefix_k: int) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 3:
        raise ValueError(f"scores must be [H,Q,K], got {scores.shape}.")
    q_len = int(scores.shape[1])
    prefix_k = int(prefix_k)
    if prefix_k:
        return prefix_content_causal_mask(q_len, prefix_k=prefix_k)
    seq = int(scores.shape[-1])
    queries = np.arange(q_len, dtype=np.int64)[:, None]
    keys = np.arange(seq, dtype=np.int64)[None, :]
    return keys <= queries


def summarize_samples(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "p50": float(np.median(array)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def load_layer_weight_list(loader, *, num_layers: int):
    from llama3fhe.application.checkpoint import load_layer_weights

    return [load_layer_weights(loader, index) for index in range(int(num_layers))]


def embed_token_ids(embedding: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    table = np.asarray(embedding, dtype=np.float32)
    if np.any(ids < 0) or np.any(ids >= table.shape[0]):
        raise ValueError("token ids fall outside the embedding vocabulary.")
    return np.asarray(table[ids], dtype=np.float32)


__all__ = [
    "NEWLINE_TOKEN_ID",
    "build_prefix_kv_bundles",
    "embed_token_ids",
    "exact_causal_gqa",
    "load_layer_weight_list",
    "prefix_token_ids",
    "run_exact_content_layer",
    "run_exact_content_stack",
    "run_exact_prefix_layer",
    "run_poly_content_stack",
    "softmax_valid_mask",
    "summarize_samples",
]
