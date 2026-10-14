from __future__ import annotations

import numpy as np
import pytest

from llama3fhe import Llama3CKKSConfig
from llama3fhe.backend import create_sim_context, release_all
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.prefill.qkv import qkv_rotations
from llama3fhe.operators.linear.prefill.qkv_kernel import _projection_weights
from llama3fhe.operators.linear.shard.qkv_source_shard import (
    qkv_carrier_reference,
    qkv_token_shard_fhe,
    qkv_token_shard_plan,
    qkv_token_shard_resource_report,
)


def _fixture():
    rng = np.random.default_rng(26092601)
    seq_len = head_dim = 64
    hidden_dim = 512
    slots = 8192
    hidden = rng.normal(scale=0.02, size=(seq_len, hidden_dim)).astype(
        np.float32
    )
    q = rng.normal(scale=0.02, size=(4 * head_dim, hidden_dim)).astype(
        np.float32
    )
    k = rng.normal(scale=0.02, size=(2 * head_dim, hidden_dim)).astype(
        np.float32
    )
    v = rng.normal(scale=0.02, size=(2 * head_dim, hidden_dim)).astype(
        np.float32
    )
    fm = FeatureMajorPrefillLayout(
        seq_len=seq_len, hidden_dim=hidden_dim, slots=slots
    )
    attention = AttentionPairLayout(
        seq_len=seq_len,
        head_dim=head_dim,
        query_heads=4,
        key_value_heads=2,
        slots=slots,
    )
    return hidden, q, k, v, fm, attention


def test_qkv_token_plan_preserves_global_input_and_output_indices() -> None:
    _, _, _, _, layout, _ = _fixture()
    plan = qkv_token_shard_plan(layout, (3, 7))

    assert layout.tokens_per_cipher == 16
    assert layout.cipher_count == 4
    assert plan.input_cipher_count == 4
    assert plan.output_cipher_count == 4
    assert plan.for_rank(3).input_indices == (0, 1)
    assert plan.for_rank(3).output_indices == (0, 1)
    assert (plan.for_rank(3).token_start, plan.for_rank(3).token_stop) == (0, 32)
    assert plan.for_rank(7).input_indices == (2, 3)
    assert plan.for_rank(7).output_indices == (2, 3)
    assert (plan.for_rank(7).token_start, plan.for_rank(7).token_stop) == (32, 64)


def test_qkv_token_shards_are_row_slices_not_partial_sums() -> None:
    hidden, q, k, v, layout, attention = _fixture()
    plan = qkv_token_shard_plan(layout, (0, 1))
    actual = qkv_carrier_reference(
        hidden, q, k, v, attention_layout=attention
    )
    _, carrier_weight, _ = _projection_weights(
        hidden, q, k, v, attention_layout=attention
    )
    expected = hidden.astype(np.float64) @ carrier_weight.T
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

    local_rows = [
        actual[spec.token_start : spec.token_stop] for spec in plan.assignments
    ]
    np.testing.assert_array_equal(np.concatenate(local_rows), actual)
    assert all(rows.shape == (32, actual.shape[1]) for rows in local_rows)


def test_production_qkv_token_shard_report_is_explicit_about_replication() -> None:
    layout = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    operator = LinearOperatorConfig(
        baby_steps=64, baby_anchor_step=64, max_plaintext_rows=512
    )
    report = qkv_token_shard_resource_report(
        layout, attention, tuple(range(8)), operator_config=operator
    )

    assert report["axis"] == "token_shard"
    assert report["source_feature_partial_sum"] is False
    assert report["homomorphic_output_reduction"] is False
    assert report["public_qkv_fhe_accepts_local_pair"] is False
    assert report["worker_count"] == 8
    assert report["assignments"]["7"]["input_indices"] == [14, 15]
    assert report["assignments"]["7"]["output_indices"] == [14, 15]
    assert report["full"]["paired_input_cipher_count"] == 8
    assert report["full"]["evaluated_cipher_count"] == 16
    assert report["per_worker"]["real_input_cipher_count"] == 2
    assert report["per_worker"]["local_seq_len"] == 16
    assert report["per_worker"]["paired_input_cipher_count"] == 1
    assert report["per_worker"]["evaluated_cipher_count"] == 2
    assert report["per_worker"]["output_carrier_count"] == 2
    assert report["per_worker"]["encrypted_batch_fraction"] == 0.125
    assert report["per_worker"]["baby_rotation_work_fraction"] == 0.125
    assert report["per_worker"]["weight_source_fraction"] == 1.0
    assert report["per_worker"]["full_weight_source_bytes"] == 96 * 1024**2
    assert report["per_worker"]["rotation_key_union_replicated"] is True


def test_production_token_ranges_reconstruct_global_128_row_linear_output() -> None:
    """The exact production 128->8x16 token split is row-independent."""

    rng = np.random.default_rng(26092603)
    plan = qkv_token_shard_plan(FeatureMajorPrefillLayout(), tuple(range(8)))
    # This lightweight dense map checks the sequence-axis identity at the exact
    # production S=128 boundary.  The heterogeneous Q/K/V carrier map itself is
    # checked independently above, and the encrypted prepared-kernel path below.
    hidden = rng.normal(size=(128, 64)).astype(np.float64)
    projection = rng.normal(size=(48, 64)).astype(np.float64)
    global_output = hidden @ projection.T
    local_outputs = tuple(
        hidden[spec.token_start : spec.token_stop] @ projection.T
        for spec in plan.assignments
    )

    assert all(part.shape == (16, 48) for part in local_outputs)
    np.testing.assert_allclose(
        np.concatenate(local_outputs), global_output, atol=2e-14, rtol=1e-12
    )


def test_qkv_token_plan_rejects_non_pair_topologies() -> None:
    _, _, _, _, layout, _ = _fixture()
    # ranks must divide the pair count (2 pairs here): 1 rank owning both
    # pairs is a legal multi-slot rank, 3 ranks is not.
    single = qkv_token_shard_plan(layout, (0,))
    assert single.for_rank(0).input_indices == (0, 1, 2, 3)
    assert single.for_rank(0).cipher_count == 4
    with pytest.raises(ValueError, match="need a divisor of 2 ranks"):
        qkv_token_shard_plan(layout, (0, 1, 2))
    with pytest.raises(ValueError, match="unique"):
        qkv_token_shard_plan(layout, (0, 0))

    partial = FeatureMajorPrefillLayout(
        seq_len=48, hidden_dim=512, slots=8192
    )
    with pytest.raises(ValueError, match="even feature-major cipher count"):
        qkv_token_shard_plan(partial, (0, 1))


@pytest.mark.gpu
def test_qkv_token_shard_fhe_accepts_two_of_four_global_inputs() -> None:
    hidden, q, k, v, layout, attention = _fixture()
    operator = LinearOperatorConfig(
        baby_steps=8, baby_anchor_step=8, max_plaintext_rows=64
    )
    config = Llama3CKKSConfig()
    config.simulator.logN = 14
    config.simulator.maxLevelsRemaining = 4
    config.simulator.device = "cuda"
    context = create_sim_context(
        config.simulator,
        input_dim=int(layout.hidden_dim),
        block_out=int(layout.tokens_per_cipher),
        rotations=qkv_rotations(layout, operator_config=operator),
        include_default_rotations=False,
    )
    inputs = tuple(
        context.encrypt(payload, level=0, slots=int(layout.slots))
        for payload in layout.pack(hidden)
    )
    result = None
    try:
        spec = qkv_token_shard_plan(layout, (0, 1)).for_rank(0)
        result = qkv_token_shard_fhe(
            hidden,
            q,
            k,
            v,
            spec=spec,
            attention_layout=attention,
            layout=layout,
            config=config,
            operator_config=operator,
            crypto_context=context,
            input_ciphers=inputs[:2],
            verify=True,
        )
        assert result.spec.output_indices == (0, 1)
        assert len(result.carrier_ciphers) == 2
        assert result.output_layout.seq_len == 32
        assert result.complexity.input_cipher_count == 1
        assert result.complexity.evaluated_cipher_count == 2
        assert result.max_abs_diff is not None
        assert result.max_abs_diff < 2e-3
    finally:
        if result is not None:
            result.release()
        release_all(inputs)
