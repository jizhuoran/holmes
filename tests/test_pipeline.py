from __future__ import annotations

import numpy as np
import pytest

from llama3fhe import Llama3CKKSConfig
from llama3fhe.approx.attention import (
    apply_rope_heads,
    attention_gqa_numpy,
    rope_cos_sin,
)
from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.layouts.attention import (
    AttentionPairLayout,
)
from llama3fhe.layouts.feature_major import (
    FeatureMajorPrefillLayout,
)
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.operators.attention.audit import (
    delta_attention_rotations,
    pipeline_rotations,
    pipeline_static_audit,
)
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.operators.attention.prefill.input import (
    complex_token_attention_rotations,
    prepare_attention_inputs_fhe,
)
from llama3fhe.operators.attention.prefill.qk import qk_delta_fhe
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.prefill.qkv import (
    qkv_fhe,
    qkv_rotations,
)


def _production_axis_layouts():
    feature_major = FeatureMajorPrefillLayout()
    # Four Q/K/V heads retain the production token, feature and folded-head
    # axes while keeping the integration smoke substantially smaller.
    attention = AttentionPairLayout(
        seq_len=128,
        head_dim=128,
        query_heads=4,
        key_value_heads=4,
        slots=32768,
    )
    return feature_major, attention


def test_feature_major_pipeline_static_audit_and_rotation_union() -> None:
    feature_major, attention = _production_axis_layouts()
    carrier = LinearCarrierLayout(
        seq_len=feature_major.seq_len,
        dimension=feature_major.hidden_dim,
        slots=feature_major.slots,
    )
    audit = pipeline_static_audit(
        feature_major, attention, include_scores=True
    )

    assert audit.canonical_input_cipher_count == 16
    assert audit.projection_input_cipher_count == 8
    assert audit.carrier_cipher_count == 16
    assert audit.query_cipher_count == 2
    assert audit.key_cipher_count == 2
    assert audit.value_cipher_count == 2
    assert audit.score_cipher_count == 2
    assert audit.query_multiplicative_depth == 4
    assert audit.key_multiplicative_depth == 5
    assert audit.value_multiplicative_depth == 4
    assert audit.score_multiplicative_depth == 7
    assert audit.canonical_input_is_real
    assert not audit.requires_input_layout_conversion
    assert set(audit.stages) == {
        "real_feature_major_pair",
        "heterogeneous_qkv",
        "carrier_to_attention",
        "delta_qk",
    }
    assert not any("transpose" in stage for stage in audit.stages)

    rotations = set(
        pipeline_rotations(
            feature_major, attention, include_scores=True
        )
    )
    qkv_union = set(qkv_rotations(feature_major))
    adapter_rotations = set(
        complex_token_attention_rotations(carrier, attention, log_n=16)
    )
    qk_rotations = set(
        delta_attention_rotations(
            attention,
            include_qk_feature_fold=True,
            qk_key_features_pre_folded=True,
            qk_query_features_pre_folded=True,
            log_n=16,
        )
    )
    assert rotations == qkv_union | adapter_rotations | qk_rotations
    assert audit.rotation_key_count == len(rotations)


@pytest.mark.gpu
def test_feature_major_pipeline_fhe_production_axes_to_s_delta() -> None:
    rng = np.random.default_rng(26081421)
    feature_major, attention = _production_axis_layouts()
    hidden = rng.normal(
        scale=0.002,
        size=(feature_major.seq_len, feature_major.hidden_dim),
    ).astype(np.float32)
    output_dim = attention.query_heads * attention.head_dim
    q_weight = rng.normal(
        scale=0.002, size=(output_dim, feature_major.hidden_dim)
    ).astype(np.float32)
    k_weight = rng.normal(
        scale=0.002, size=(output_dim, feature_major.hidden_dim)
    ).astype(np.float32)
    v_weight = rng.normal(
        scale=0.002, size=(output_dim, feature_major.hidden_dim)
    ).astype(np.float32)

    config = Llama3CKKSConfig()
    config.simulator.logN = 16
    config.simulator.maxLevelsRemaining = 10
    config.simulator.device = "cuda"
    config.qkv_linear.input_level = 0
    config.qkv_linear.dtype = "float64"
    config.layout_transform.dtype = "float64"

    # One policy object for the planner and the kernel below.
    production_linear = LinearOperatorConfig(
        baby_steps=32, max_plaintext_rows=512
    )
    rotations = pipeline_rotations(
        feature_major,
        attention,
        qkv_operator_config=production_linear,
        include_scores=True,
    )
    context = create_sim_context(
        config.simulator,
        input_dim=feature_major.hidden_dim,
        block_out=feature_major.tokens_per_cipher,
        rotations=rotations,
        include_default_rotations=False,
    )
    inputs = tuple(
        context.encrypt(row, level=0, slots=feature_major.slots)
        for row in feature_major.pack(hidden)
    )

    qkv = adapter = None
    scores: tuple[object, ...] = ()
    try:
        qkv = qkv_fhe(
            hidden,
            q_weight,
            k_weight,
            v_weight,
            attention_layout=attention,
            layout=feature_major,
            config=config,
            operator_config=production_linear,
            extra_rotations=rotations,
            crypto_context=context,
            input_ciphers=inputs,
            verify=True,
        )
        expected_query = hidden @ q_weight.T
        expected_key = hidden @ k_weight.T
        expected_value = hidden @ v_weight.T
        adapter = prepare_attention_inputs_fhe(
            qkv.carrier_ciphers,
            carrier_layout=qkv.output_layout,
            attention_layout=attention,
            crypto_context=context,
            model_config=config,
            verify=True,
            expected_query=expected_query,
            expected_key=expected_key,
            expected_value=expected_value,
        )
        scores = qk_delta_fhe(
            adapter.query_ciphers,
            adapter.key_ciphers,
            layout=attention,
            crypto_context=context,
            operator_config=AttentionOperatorConfig(
                qk_output_level=None,
                qk_complex_feature_folding=True,
                qk_key_features_pre_folded=True,
                qk_query_features_pre_scaled=True,
                qk_query_features_pre_folded=True,
            ),
        )

        assert qkv.output_levels == (1,)
        assert {context.level_for_cipher(c) for c in adapter.query_ciphers} == {4}
        assert {context.level_for_cipher(c) for c in adapter.key_ciphers} == {5}
        assert {context.level_for_cipher(c) for c in adapter.value_ciphers} == {4}
        assert {context.level_for_cipher(c) for c in scores} == {7}
        assert adapter.max_abs_diff is not None
        assert max(adapter.max_abs_diff.values()) < 5e-2

        decrypted = np.stack([np.asarray(context.decrypt(cipher)) for cipher in scores])
        actual = attention.unpack_score_delta(decrypted)
        actual /= np.float32(np.sqrt(float(attention.head_dim)))
        cos, sin = rope_cos_sin(
            seq_len=attention.seq_len,
            head_dim=attention.head_dim,
        )
        query = apply_rope_heads(
            expected_query,
            cos,
            sin,
            num_heads=attention.query_heads,
            head_dim=attention.head_dim,
        )
        key = apply_rope_heads(
            expected_key,
            cos,
            sin,
            num_heads=attention.key_value_heads,
            head_dim=attention.head_dim,
        )
        expected, _, _ = attention_gqa_numpy(
            query, key, expected_value, layout=attention
        )
        assert float(np.max(np.abs(actual - expected))) < 5e-2
    finally:
        for cipher in scores:
            release_if_supported(cipher)
        if adapter is not None:
            adapter.release()
        if qkv is not None:
            qkv.release()
        for cipher in inputs:
            release_if_supported(cipher)
