from __future__ import annotations

import numpy as np

from llama3fhe.application.approx_config import ApproximationConfig
from llama3fhe.application.args import DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
from llama3fhe.approx.sample_fit import fit_stack_from_plaintext
from llama3fhe.graph import LayerWeights
from llama3fhe.layouts.attention import AttentionPairLayout


def _tiny_weights(rng, hidden_dim: int, intermediate: int) -> LayerWeights:
    return LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        k_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.005, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        up_proj=rng.normal(scale=0.005, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        down_proj=rng.normal(scale=0.005, size=(hidden_dim, intermediate)).astype(
            np.float32
        ),
    )


def test_sample_fit_reuses_or_refits_prefill_polynomials() -> None:
    rng = np.random.default_rng(180918)
    hidden_dim = 4096
    prefill_len = 128
    attention = AttentionPairLayout()
    prefill = rng.normal(scale=0.03, size=(prefill_len, hidden_dim)).astype(
        np.float32
    )
    decode_h = rng.normal(scale=0.03, size=(1, hidden_dim)).astype(np.float32)
    weights = _tiny_weights(rng, hidden_dim, 14336)
    config = ApproximationConfig(
        rmsnorm_path=DEFAULT_RMS,
        silu_path=DEFAULT_SILU,
        softmax_path=DEFAULT_SOFTMAX,
    )
    production = config.for_layer(0)
    fitted, reports = fit_stack_from_plaintext(
        prefill,
        (weights, weights),
        attention_layout=attention,
        decode_hidden=decode_h,
        prefill_config=config,
    )
    assert len(fitted) == 2
    assert reports[0].reused["input_norm"] is False
    variance = np.mean(np.square(prefill.astype(np.float64)), axis=-1)
    lo, hi = (float(v) for v in fitted[0].input_norm["fit_interval"])
    assert float(np.min(variance)) >= lo
    assert float(np.max(variance)) <= hi
    assert fitted[0].input_norm["degree"] == production.input_norm["degree"]
    print(
        "sample-fit reports: "
        + ", ".join(
            f"L{row.layer} reused={row.reused} degrees={row.degrees}"
            for row in reports
        ),
        flush=True,
    )
