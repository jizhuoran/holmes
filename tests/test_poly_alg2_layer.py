from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import llama3fhe.approx.layer as layer_module
from llama3fhe.application.approx_config import ApproximationConfig
from llama3fhe.application.args import DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
from llama3fhe.approx.layer import (
    _content_causal_valid_mask,
    poly_prefill_layer_numpy,
    softmax_alg2_poly_scores_numpy,
    softmax_schain_poly_scores_numpy,
)
from llama3fhe.approx.chebyshev import eval_chebyshev_series_direct
from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly
from llama3fhe.graph import LayerWeights
from llama3fhe.layouts.attention import AttentionPairLayout


def test_alg2_causal_last_row_matches_decode_query() -> None:
    rng = np.random.default_rng(180918)
    layout = AttentionPairLayout()
    seq = 8
    scores = rng.normal(
        scale=2.0,
        size=(int(layout.query_heads), seq, seq),
    ).astype(np.float64)
    softmax = ApproximationConfig(
        rmsnorm_path=DEFAULT_RMS,
        silu_path=DEFAULT_SILU,
        softmax_path=DEFAULT_SOFTMAX,
    ).for_layer(0).softmax
    valid = np.tri(seq, dtype=bool)
    causal, report = softmax_alg2_poly_scores_numpy(
        scores,
        config=softmax,
        layer_idx=0,
        head_dim=int(layout.head_dim),
        valid=valid,
    )
    decode, _ = softmax_alg2_poly_scores_numpy(
        scores[:, seq - 1 : seq, :],
        config=softmax,
        layer_idx=0,
        head_dim=int(layout.head_dim),
    )
    np.testing.assert_allclose(causal[:, seq - 1, :], decode[:, 0, :], atol=1e-6)
    assert report["exp_in_range"] in (True, False)
    assert report["k"] == int(softmax.k_for(0))


def test_paged_prefill_poly_normalizes_first_row_sum_only() -> None:
    config = DecodeAlg2Poly.synthetic(
        k=1, degree=8, U_hat=1.0, invsqrt_interval=(0.01, 20.0)
    )
    scores = np.zeros((4, 16, 16), dtype=np.float64)
    valid = np.tri(16, dtype=bool)
    _probabilities, report = softmax_alg2_poly_scores_numpy(
        scores,
        config=config,
        layer_idx=0,
        head_dim=8,
        valid=valid,
        prefill_page_size=8,
    )
    exp_entry = config.layer_exp[0]
    exp_value = float(
        eval_chebyshev_series_direct(
            np.asarray([-0.5]),
            exp_entry["coefficients"],
            *exp_entry["fit_interval"],
        )[0]
    )
    first = report["invsqrt_rounds"][0]
    assert report["prefill_max_page_count"] == 2
    assert first["min"] == pytest.approx(exp_value**2)
    assert first["max"] == pytest.approx(8.0 * exp_value**2)


def test_schain_variant_matches_alg2_and_sees_same_rsqrt_inputs() -> None:
    """Algorithm 4' (S-chain) is algebraically identical to per-round Alg2:
    same output and the same rsqrt inputs s_t in every round."""

    rng = np.random.default_rng(4071)
    layout = AttentionPairLayout()
    seq = 16
    approx = ApproximationConfig(
        rmsnorm_path=DEFAULT_RMS,
        silu_path=DEFAULT_SILU,
        softmax_path=DEFAULT_SOFTMAX,
    )
    valid = np.tri(seq, dtype=bool)
    for layer_idx in (0, 10):  # k=4 and k=2 in the default table
        softmax = approx.for_layer(layer_idx).softmax
        scores = rng.normal(
            scale=2.0, size=(int(layout.query_heads), seq, seq)
        ).astype(np.float64)
        alg2, rep_a = softmax_alg2_poly_scores_numpy(
            scores, config=softmax, layer_idx=layer_idx,
            head_dim=int(layout.head_dim), valid=valid,
        )
        schain, rep_s = softmax_schain_poly_scores_numpy(
            scores, config=softmax, layer_idx=layer_idx,
            head_dim=int(layout.head_dim), valid=valid,
        )
        assert rep_s["variant"] == "schain"
        assert rep_s["k"] == rep_a["k"]
        np.testing.assert_allclose(schain, alg2, rtol=1e-5, atol=1e-7)
        assert len(rep_s["invsqrt_rounds"]) == rep_a["k"]
        for round_a, round_s in zip(rep_a["invsqrt_rounds"], rep_s["invsqrt_rounds"]):
            assert round_s["interval"] == round_a["interval"]
            np.testing.assert_allclose(round_s["min"], round_a["min"], rtol=1e-9)
            np.testing.assert_allclose(round_s["max"], round_a["max"], rtol=1e-9)
        assert np.all(schain[:, ~valid] == 0.0)


def test_poly_prefill_uses_selected_layer_alg2_and_causal_mask(
    monkeypatch,
) -> None:
    rng = np.random.default_rng(260926)
    layout = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=32,
    )
    hidden_dim = 16
    kv_dim = 8
    intermediate = 6
    softmax_config = object()
    observed: dict[str, object] = {}

    def fake_alg2(
        scores,
        *,
        config,
        layer_idx,
        head_dim,
        valid,
    ):
        observed.update(
            scores=np.asarray(scores),
            config=config,
            layer_idx=int(layer_idx),
            head_dim=int(head_dim),
            valid=np.asarray(valid, dtype=bool),
        )
        probabilities = np.broadcast_to(
            np.asarray(valid, dtype=np.float64)[None, :, :],
            np.asarray(scores).shape,
        ).copy()
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        return probabilities.astype(np.float32), {
            "exp_in_range": True,
            "invsqrt_in_range": True,
        }

    monkeypatch.setattr(
        layer_module, "softmax_alg2_poly_scores_numpy", fake_alg2
    )

    def weight(shape):
        return rng.normal(scale=0.03, size=shape).astype(np.float32)

    weights = LayerWeights(
        input_layernorm=np.ones((hidden_dim,), dtype=np.float32),
        q_proj=weight((hidden_dim, hidden_dim)),
        k_proj=weight((kv_dim, hidden_dim)),
        v_proj=weight((kv_dim, hidden_dim)),
        o_proj=weight((hidden_dim, hidden_dim)),
        post_attention_layernorm=np.ones((hidden_dim,), dtype=np.float32),
        gate_proj=np.zeros((intermediate, hidden_dim), dtype=np.float32),
        up_proj=np.zeros((intermediate, hidden_dim), dtype=np.float32),
        down_proj=np.zeros((hidden_dim, intermediate), dtype=np.float32),
    )
    approximations = SimpleNamespace(
        input_norm={"coefficients": [2.0], "fit_interval": [0.0, 1.0]},
        post_attention_norm={
            "coefficients": [2.0],
            "fit_interval": [0.0, 1.0],
        },
        silu={"coefficients": [0.0], "fit_interval": [-1.0, 1.0]},
        softmax=softmax_config,
    )
    hidden = rng.normal(scale=0.1, size=(4, hidden_dim)).astype(np.float32)
    token_valid = np.ones((4,), dtype=bool)

    result = poly_prefill_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        layer_idx=7,
        use_alg2_softmax=True,
        token_valid_mask=token_valid,
    )

    assert result.hidden.shape == hidden.shape
    assert observed["config"] is softmax_config
    assert observed["layer_idx"] == 7
    assert observed["head_dim"] == 4
    assert np.asarray(observed["scores"]).shape == (4, 4, 4)
    np.testing.assert_array_equal(
        observed["valid"], np.tri(4, dtype=bool)
    )


def test_content_causal_valid_mask_keeps_invalid_query_self_sentinel() -> None:
    token_valid = np.asarray([False, True, False, True], dtype=bool)

    actual = _content_causal_valid_mask(
        4, token_valid_mask=token_valid
    )

    expected = np.asarray(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, True, False, True],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(actual, expected)
