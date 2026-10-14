from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from llama3fhe.approx.attention import apply_rope_heads, attention_gqa_numpy, rope_cos_sin
from llama3fhe.approx.chebyshev import eval_chebyshev_coefficients, eval_chebyshev_series_direct
from llama3fhe.approx.layer import (
    poly_prefill_layer_numpy,
    poly_transformer_layer_numpy,
)
from llama3fhe.graph import LayerWeights
from llama3fhe.layouts.attention import AttentionPairLayout


def _silu(value: float) -> float:
    return float(value / (1.0 + math.exp(-float(value))))


def test_poly_transformer_layer_matches_manual_identity_rmsnorm() -> None:
    rng = np.random.default_rng(260821)
    layout = AttentionPairLayout(
        seq_len=8,
        head_dim=8,
        query_heads=4,
        key_value_heads=2,
        slots=128,
    )
    hidden_dim = 32
    intermediate = 48
    kv_dim = 16
    weights = LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.05, size=(hidden_dim, hidden_dim)).astype(np.float32),
        k_proj=rng.normal(scale=0.05, size=(kv_dim, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.05, size=(kv_dim, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.05, size=(hidden_dim, hidden_dim)).astype(np.float32),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.05, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        up_proj=rng.normal(scale=0.05, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        down_proj=rng.normal(scale=0.05, size=(hidden_dim, intermediate)).astype(
            np.float32
        ),
    )
    silu_coeffs = eval_chebyshev_coefficients(_silu, -2.0, 2.0, 12)
    approximations = SimpleNamespace(
        input_norm={"coefficients": [2.0], "fit_interval": [0.0, 1.0]},
        post_attention_norm={"coefficients": [2.0], "fit_interval": [0.0, 1.0]},
        silu={"coefficients": silu_coeffs, "fit_interval": [-2.0, 2.0]},
    )
    hidden = rng.normal(scale=0.1, size=(8, hidden_dim)).astype(np.float32)

    result = poly_transformer_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
    )

    query = hidden @ weights.q_proj.T
    key = hidden @ weights.k_proj.T
    value = hidden @ weights.v_proj.T
    cos, sin = rope_cos_sin(seq_len=8, head_dim=8)
    query = apply_rope_heads(query, cos, sin, num_heads=4, head_dim=8)
    key = apply_rope_heads(key, cos, sin, num_heads=2, head_dim=8)
    _, _, attn_out = attention_gqa_numpy(query, key, value, layout=layout)
    residual = hidden + attn_out @ weights.o_proj.T
    gate = residual @ weights.gate_proj.T
    up = residual @ weights.up_proj.T
    activated = eval_chebyshev_series_direct(gate, silu_coeffs, -2.0, 2.0) * up
    expected = residual + activated @ weights.down_proj.T

    np.testing.assert_allclose(result.hidden, expected, rtol=1e-5, atol=1e-5)
    assert result.silu_oob_frac == 0.0
    assert result.silu_gate_max_abs is not None
    assert result.silu_gate_max_abs < 2.0


def test_poly_layer_gamma_matches_fused_linear_weights() -> None:
    """Both RMSNorm gammas match the encrypted graph's weight fusion."""

    rng = np.random.default_rng(260926)
    layout = AttentionPairLayout(
        seq_len=8,
        head_dim=8,
        query_heads=4,
        key_value_heads=2,
        slots=128,
    )
    hidden_dim = 32
    intermediate = 48
    kv_dim = 16
    input_gamma = np.linspace(0.35, 1.65, hidden_dim, dtype=np.float32)
    post_gamma = np.linspace(1.7, 0.4, hidden_dim, dtype=np.float32)

    def weight(shape):
        return rng.normal(scale=0.04, size=shape).astype(np.float32)

    explicit = LayerWeights(
        input_layernorm=input_gamma,
        q_proj=weight((hidden_dim, hidden_dim)),
        k_proj=weight((kv_dim, hidden_dim)),
        v_proj=weight((kv_dim, hidden_dim)),
        o_proj=weight((hidden_dim, hidden_dim)),
        post_attention_layernorm=post_gamma,
        gate_proj=weight((intermediate, hidden_dim)),
        up_proj=weight((intermediate, hidden_dim)),
        down_proj=weight((hidden_dim, intermediate)),
    )
    fused = replace(
        explicit,
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=explicit.q_proj * input_gamma[None, :],
        k_proj=explicit.k_proj * input_gamma[None, :],
        v_proj=explicit.v_proj * input_gamma[None, :],
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=explicit.gate_proj * post_gamma[None, :],
        up_proj=explicit.up_proj * post_gamma[None, :],
    )
    silu_coeffs = eval_chebyshev_coefficients(_silu, -2.0, 2.0, 12)
    approximations = SimpleNamespace(
        input_norm={"coefficients": [2.0], "fit_interval": [0.0, 1.0]},
        post_attention_norm={
            "coefficients": [2.0],
            "fit_interval": [0.0, 1.0],
        },
        silu={"coefficients": silu_coeffs, "fit_interval": [-2.0, 2.0]},
    )
    hidden = rng.normal(scale=0.08, size=(8, hidden_dim)).astype(np.float32)

    explicit_transformer = poly_transformer_layer_numpy(
        hidden,
        weights=explicit,
        approximations=approximations,
        attention_layout=layout,
    )
    fused_transformer = poly_transformer_layer_numpy(
        hidden,
        weights=fused,
        approximations=approximations,
        attention_layout=layout,
    )
    np.testing.assert_allclose(
        explicit_transformer.hidden,
        fused_transformer.hidden,
        rtol=2e-5,
        atol=2e-6,
    )

    explicit_prefill = poly_prefill_layer_numpy(
        hidden,
        weights=explicit,
        approximations=approximations,
        attention_layout=layout,
        use_alg2_softmax=False,
    )
    fused_prefill = poly_prefill_layer_numpy(
        hidden,
        weights=fused,
        approximations=approximations,
        attention_layout=layout,
        use_alg2_softmax=False,
    )
    np.testing.assert_allclose(
        explicit_prefill.hidden,
        fused_prefill.hidden,
        rtol=2e-5,
        atol=2e-6,
    )
