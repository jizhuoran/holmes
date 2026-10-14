from __future__ import annotations

import numpy as np
import pytest

from llama3fhe import Llama3CKKSConfig
from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import (
    FeatureMajorPrefillLayout,
)
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.prefill.qkv import (
    qkv_fhe,
    qkv_rotations,
    token_pair_complexity,
)
from llama3fhe.operators.linear.prefill.qkv_kernel import _projection_weights


def test_feature_major_qkv_pairing_static_contract() -> None:
    layout = FeatureMajorPrefillLayout()
    cost = token_pair_complexity(layout)

    assert cost.input_cipher_count == 16
    assert cost.output_cipher_count == 8
    assert cost.additions == 8
    assert cost.imults == 8
    assert cost.multiplicative_depth == 0
    rotations = qkv_rotations(layout)
    assert 4 * layout.slots - 1 in rotations
    assert 8 in rotations


def test_qkv_input_column_scale_is_absorbed_into_plaintext_weight() -> None:
    rng = np.random.default_rng(26081502)
    seq_len = head_dim = 8
    hidden_dim = 64
    heads = 4
    hidden = rng.normal(size=(seq_len, hidden_dim)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, size=(hidden_dim,)).astype(np.float32)
    weights = [
        rng.normal(size=(heads * head_dim, hidden_dim)).astype(np.float32)
        for _ in range(3)
    ]
    attention_layout = AttentionPairLayout(
        seq_len=seq_len,
        head_dim=head_dim,
        query_heads=heads,
        key_value_heads=heads,
        slots=2 * seq_len * head_dim,
    )
    _, baseline, _ = _projection_weights(
        hidden,
        *weights,
        attention_layout=attention_layout,
    )
    _, fused, _ = _projection_weights(
        hidden / scale[None, :],
        *weights,
        attention_layout=attention_layout,
        input_column_scale=scale,
    )
    np.testing.assert_allclose(
        (hidden / scale[None, :]) @ fused.T,
        hidden @ baseline.T,
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("fuse_input_scale", [False, True])
def test_feature_major_qkv_fhe_small_matches_numpy(
    fuse_input_scale: bool,
) -> None:
    rng = np.random.default_rng(26081411)
    seq_len = head_dim = 32
    hidden_dim = 512
    heads = 4
    slots = 8192
    hidden = rng.normal(scale=0.01, size=(seq_len, hidden_dim)).astype(np.float32)
    q_weight = rng.normal(
        scale=0.01, size=(heads * head_dim, hidden_dim)
    ).astype(np.float32)
    k_weight = rng.normal(
        scale=0.01, size=(heads * head_dim, hidden_dim)
    ).astype(np.float32)
    v_weight = rng.normal(
        scale=0.01, size=(heads * head_dim, hidden_dim)
    ).astype(np.float32)
    column_scale = rng.uniform(0.75, 1.25, size=(hidden_dim,)).astype(
        np.float32
    )
    physical_hidden = (
        hidden / column_scale[None, :] if fuse_input_scale else hidden
    )
    layout = FeatureMajorPrefillLayout(
        seq_len=seq_len, hidden_dim=hidden_dim, slots=slots
    )
    attention_layout = AttentionPairLayout(
        seq_len=seq_len,
        head_dim=head_dim,
        query_heads=heads,
        key_value_heads=heads,
        slots=slots,
    )
    config = Llama3CKKSConfig()
    config.simulator.logN = 14
    config.simulator.maxLevelsRemaining = 4
    config.simulator.device = "cuda"
    config.qkv_linear.dtype = "float64"

    # One policy object, shared by the key planner and the kernel.
    small_linear = LinearOperatorConfig(
        baby_steps=8, baby_anchor_step=8, max_plaintext_rows=64
    )
    context = create_sim_context(
        config.simulator,
        input_dim=hidden_dim,
        block_out=layout.tokens_per_cipher,
        rotations=qkv_rotations(
            layout,
            operator_config=small_linear,
        ),
        include_default_rotations=False,
    )
    inputs = tuple(
        context.encrypt(row, level=0, slots=slots)
        for row in layout.pack(physical_hidden)
    )

    result = qkv_fhe(
        physical_hidden,
        q_weight,
        k_weight,
        v_weight,
        attention_layout=attention_layout,
        layout=layout,
        config=config,
        crypto_context=context,
        input_ciphers=inputs,
        verify=True,
        input_column_scale=(column_scale if fuse_input_scale else None),
        operator_config=small_linear,
    )
    try:
        assert layout.cipher_count == 2
        assert result.pair_complexity.output_cipher_count == 1
        assert len(result.carrier_ciphers) == 2
        assert result.output_layout.cipher_count == 2
        assert result.carrier_width == 192
        assert result.max_abs_diff is not None
        assert result.max_abs_diff < 2e-3
        assert result.output_levels == (1,)
        assert result.projection.complexity.multiplicative_depth == 1
        assert result.projection.complexity.pt_ct_multiplications == 1024
    finally:
        result.release()
        for cipher in inputs:
            release_if_supported(cipher)
