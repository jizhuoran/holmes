from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np

from llama3fhe.approx.chebyshev import eval_chebyshev_coefficients
from llama3fhe.approx.layer import (
    poly_decode_layer_numpy,
    poly_prefill_layer_numpy,
    poly_prefill_stack_numpy,
    poly_transformer_layer_numpy,
)
from llama3fhe.approx.prefix import PrefixKVBundle
from llama3fhe.graph import LayerWeights, evaluate, required_rotations
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.attention.prefill.operator import attention_fhe_delta_alg2
from llama3fhe.operators.attention.prefix.prefix_pc import prefix_pc_rotations


def _silu(value: float) -> float:
    return float(value / (1.0 + math.exp(-float(value))))


def _small_layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=8,
        head_dim=8,
        query_heads=4,
        key_value_heads=2,
        slots=128,
    )


def _identity_approx():
    silu_coeffs = eval_chebyshev_coefficients(_silu, -2.0, 2.0, 12)
    return SimpleNamespace(
        input_norm={"coefficients": [2.0], "fit_interval": [0.0, 1.0]},
        post_attention_norm={"coefficients": [2.0], "fit_interval": [0.0, 1.0]},
        silu={"coefficients": silu_coeffs, "fit_interval": [-2.0, 2.0]},
    )


def _tiny_weights(rng: np.random.Generator, *, hidden_dim: int = 32) -> LayerWeights:
    kv_dim = 16
    intermediate = 48
    return LayerWeights(
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


def test_prefix_kv_bundle_is_c_form_not_delta() -> None:
    layout = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=32,
    )
    rng = np.random.default_rng(7)
    key = rng.standard_normal((2, 8)).astype(np.float32)
    value = rng.standard_normal((2, 8)).astype(np.float32)
    bundle = PrefixKVBundle.from_logical(key, value, layout=layout)
    unpacked_k = layout.unpack_key_c(bundle.packed_key_c)
    unpacked_v = layout.unpack_value_c(bundle.packed_value_c)
    np.testing.assert_allclose(unpacked_k[:2], key, atol=1e-6)
    np.testing.assert_allclose(unpacked_v[:2], value, atol=1e-6)
    np.testing.assert_allclose(unpacked_k[2:], 0.0, atol=1e-6)
    padded = np.zeros((4, 8), dtype=np.float32)
    padded[:2] = key
    delta = layout.pack_key_delta(padded)
    assert bundle.packed_key_c.shape == delta.shape
    assert not np.allclose(bundle.packed_key_c, delta)
    assert bundle.prefix_k == 2


def test_poly_layer0_prefix_keeps_content_length() -> None:
    rng = np.random.default_rng(8)
    layout = _small_layout()
    weights = _tiny_weights(rng)
    approximations = _identity_approx()
    hidden = rng.normal(scale=0.1, size=(8, 32)).astype(np.float32)
    prefix_hidden = rng.normal(scale=0.1, size=(2, 32)).astype(np.float32)
    result = poly_transformer_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        prefix_hidden=prefix_hidden,
        rope_theta=1_000_000.0,
    )
    baseline = poly_transformer_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        rope_theta=1_000_000.0,
    )
    assert result.hidden.shape == hidden.shape
    assert result.hidden.shape[0] == 8
    assert not np.allclose(result.hidden, baseline.hidden)


def test_poly_layer0_prefix_bundle_matches_prefix_hidden() -> None:
    rng = np.random.default_rng(9)
    layout = _small_layout()
    weights = _tiny_weights(rng)
    approximations = _identity_approx()
    hidden = rng.normal(scale=0.1, size=(8, 32)).astype(np.float32)
    prefix_hidden = rng.normal(scale=0.1, size=(2, 32)).astype(np.float32)
    from_hidden = poly_transformer_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        prefix_hidden=prefix_hidden,
        rope_theta=1_000_000.0,
    )
    from llama3fhe.approx.attention import apply_rope_heads, rope_cos_sin
    from llama3fhe.approx.layer import _project_qkv, _rmsnorm_poly

    prefix_in = _rmsnorm_poly(prefix_hidden, approximations.input_norm)
    _, key_p, value_p = _project_qkv(prefix_in, weights)
    cos, sin = rope_cos_sin(seq_len=2, head_dim=8, theta=1_000_000.0, start_pos=0)
    key_p = apply_rope_heads(key_p, cos, sin, num_heads=2, head_dim=8)
    bundle = PrefixKVBundle.from_logical(key_p, value_p, layout=layout)
    from_bundle = poly_transformer_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        prefix_bundle=bundle,
        rope_theta=1_000_000.0,
    )
    np.testing.assert_allclose(from_hidden.hidden, from_bundle.hidden, atol=1e-5)


def test_poly_decode_prefix_bundle_matches_logical_prepended_cache() -> None:
    """Prefix PT + content cache is equivalent to one logical KV history."""

    rng = np.random.default_rng(10)
    layout = _small_layout()
    weights = _tiny_weights(rng)
    approximations = _identity_approx()
    hidden = rng.normal(scale=0.1, size=(1, 32)).astype(np.float32)
    prefix_key = rng.normal(scale=0.1, size=(2, 16)).astype(np.float32)
    prefix_value = rng.normal(scale=0.1, size=(2, 16)).astype(np.float32)
    content_key = rng.normal(scale=0.1, size=(3, 16)).astype(np.float32)
    content_value = rng.normal(scale=0.1, size=(3, 16)).astype(np.float32)
    bundle = PrefixKVBundle.from_logical(
        prefix_key, prefix_value, layout=layout
    )

    split = poly_decode_layer_numpy(
        hidden,
        cache_key=content_key,
        cache_value=content_value,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        position=5,
        rope_theta=1_000_000.0,
        use_alg2_softmax=False,
        prefix_bundle=bundle,
    )
    logical = poly_decode_layer_numpy(
        hidden,
        cache_key=np.concatenate([prefix_key, content_key], axis=0),
        cache_value=np.concatenate([prefix_value, content_value], axis=0),
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        position=5,
        rope_theta=1_000_000.0,
        use_alg2_softmax=False,
    )
    np.testing.assert_allclose(split.hidden, logical.hidden, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(split.key, logical.key, atol=1e-6)
    np.testing.assert_allclose(split.value, logical.value, atol=1e-6)


def test_evaluate_and_rotations_accept_prefix() -> None:
    from llama3fhe.model import Llama3Model

    assert "prefix_bundle" in inspect.signature(evaluate).parameters
    assert "prefix_bundle" in inspect.signature(attention_fhe_delta_alg2).parameters
    assert "prefix_bundles" in inspect.signature(Llama3Model.forward).parameters
    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    extra = set(prefix_pc_rotations(attention, prefix_k=2))
    extra.discard(0)
    with_prefix = set(required_rotations(feature, attention, prefix_k=2))
    assert extra <= with_prefix


def test_poly_prefill_prefix_keeps_content_length() -> None:
    rng = np.random.default_rng(11)
    layout = _small_layout()
    weights = _tiny_weights(rng)
    approximations = _identity_approx()
    hidden = rng.normal(scale=0.1, size=(8, 32)).astype(np.float32)
    prefix_hidden = rng.normal(scale=0.1, size=(2, 32)).astype(np.float32)
    result = poly_prefill_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        prefix_hidden=prefix_hidden,
        rope_theta=1_000_000.0,
        use_alg2_softmax=False,
    )
    assert result.hidden.shape == hidden.shape
    assert result.key.shape[0] == hidden.shape[0]
    assert result.value.shape[0] == hidden.shape[0]


def test_poly_prefill_stack_uses_checkpoint_layer_offset(monkeypatch) -> None:
    import llama3fhe.approx.layer as layer_module

    selected: list[int] = []
    calls: list[tuple[int, object]] = []

    class ApproxConfig:
        def for_layer(self, layer_idx: int):
            selected.append(int(layer_idx))
            return f"approx-{int(layer_idx)}"

    def fake_layer(hidden, *, approximations, layer_idx, prefix_bundle, **_kwargs):
        calls.append((int(layer_idx), prefix_bundle))
        value = np.full((1, 1), int(layer_idx), dtype=np.float32)
        return SimpleNamespace(
            hidden=np.asarray(hidden, dtype=np.float32) + 1.0,
            key=value,
            value=-value,
            approximations=approximations,
        )

    monkeypatch.setattr(layer_module, "poly_prefill_layer_numpy", fake_layer)
    prefix_bundles = [object(), object()]
    hidden, keys, values = poly_prefill_stack_numpy(
        np.zeros((1, 1), dtype=np.float32),
        [object(), object()],
        approx_config=ApproxConfig(),
        attention_layout=object(),
        start_layer=8,
        prefix_bundles=prefix_bundles,
    )

    assert selected == [8, 9]
    assert calls == [(8, prefix_bundles[0]), (9, prefix_bundles[1])]
    np.testing.assert_array_equal(hidden, [[2.0]])
    np.testing.assert_array_equal(keys[:, 0, 0], [8.0, 9.0])
    np.testing.assert_array_equal(values[:, 0, 0], [-8.0, -9.0])


def test_prefix_kv_cache_matches_concat_content_hidden() -> None:
    from llama3fhe.approx.attention import apply_rope_heads, rope_cos_sin
    from llama3fhe.approx.prefix_stack import (
        build_prefix_kv_bundles,
        exact_causal_gqa,
        run_exact_content_stack,
        run_exact_prefix_layer,
    )
    from llama3fhe.approx.rmsnorm import rmsnorm_numpy

    rng = np.random.default_rng(12)
    layout = _small_layout()
    weights = _tiny_weights(rng)
    content = rng.normal(scale=0.1, size=(8, 32)).astype(np.float32)
    prefix = rng.normal(scale=0.1, size=(2, 32)).astype(np.float32)
    concat = np.concatenate([prefix, content], axis=0)
    hidden_pp = prefix
    for _ in range(1):
        hidden_pp, _, _ = run_exact_prefix_layer(
            hidden_pp,
            weights=weights,
            layout=layout,
            rope_theta=1_000_000.0,
        )
    bundles, _ = build_prefix_kv_bundles(
        prefix, (weights,), layout=layout, rope_theta=1_000_000.0
    )
    split, _ = run_exact_content_stack(
        content,
        (weights,),
        layout=layout,
        prefix_bundles=bundles,
        rope_theta=1_000_000.0,
    )

    def exact_unconstrained(hidden):
        attn_in = rmsnorm_numpy(hidden, weights.input_layernorm, eps=1e-5)
        query = attn_in @ weights.q_proj.T
        key = attn_in @ weights.k_proj.T
        value = attn_in @ weights.v_proj.T
        cos, sin = rope_cos_sin(
            seq_len=int(hidden.shape[0]), head_dim=8, theta=1_000_000.0, start_pos=0
        )
        query = apply_rope_heads(query, cos, sin, num_heads=4, head_dim=8)
        key = apply_rope_heads(key, cos, sin, num_heads=2, head_dim=8)
        _, _, attn_out = exact_causal_gqa(
            query,
            key,
            value,
            query_heads=4,
            key_value_heads=2,
            head_dim=8,
        )
        residual = hidden + attn_out @ weights.o_proj.T
        mlp_in = rmsnorm_numpy(residual, weights.post_attention_layernorm, eps=1e-5)
        gate = mlp_in @ weights.gate_proj.T
        up = mlp_in @ weights.up_proj.T
        silu = gate / (1.0 + np.exp(-gate))
        return residual + (silu * up) @ weights.down_proj.T

    concat_out = exact_unconstrained(concat)
    np.testing.assert_allclose(split, concat_out[2:], atol=2e-5, rtol=2e-5)
