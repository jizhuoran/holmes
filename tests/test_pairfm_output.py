from __future__ import annotations

import numpy as np
import pytest

from llama3fhe import Llama3CKKSConfig
from llama3fhe.backend import create_sim_context
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import (
    FeatureMajorPrefillLayout,
)
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.operators.attention.prefill.output import (
    _blocked_masks,
    _pairfm_masks,
    output_c_to_feature_major_pairfm_numpy,
    output_c_to_pairfm_numpy,
    pairfm_output_complexity,
    pairfm_output_rotations,
    prepare_w_o_inputs_fhe,
)


def _small_layouts() -> tuple[FeatureMajorPrefillLayout, AttentionPairLayout]:
    attention = AttentionPairLayout(
        seq_len=8,
        head_dim=8,
        query_heads=4,
        key_value_heads=2,
        slots=128,
    )
    feature = FeatureMajorPrefillLayout(seq_len=8, hidden_dim=32, slots=128)
    return feature, attention


def test_pairfm_output_cpu_matches_existing_contract() -> None:
    rng = np.random.default_rng(26081443)
    feature, attention = _small_layouts()
    logical = rng.normal(scale=0.02, size=(8, 32)).astype(np.float32)
    output_c = attention.pack_output_c(logical)

    pairfm = output_c_to_pairfm_numpy(
        output_c,
        feature_layout=feature,
        attention_layout=attention,
    )
    expected_pairfm = (
        output_c.reshape(2, 8, 8, 2)
        .transpose(0, 1, 3, 2)
        .reshape(2, 128)
    )
    np.testing.assert_array_equal(pairfm, expected_pairfm)

    actual = output_c_to_feature_major_pairfm_numpy(
        output_c,
        feature_layout=feature,
        attention_layout=attention,
    )
    physical_channels = [
        head * attention.head_dim + dim
        for first, second in attention.query_head_pairs()
        for dim in range(attention.head_dim)
        for head in (first, second)
    ]
    expected = feature.pack(logical[:, physical_channels], dtype=actual.dtype)
    np.testing.assert_array_equal(actual, expected)


def test_pairfm_output_production_static_cost() -> None:
    cost = pairfm_output_complexity(
        FeatureMajorPrefillLayout(), AttentionPairLayout()
    )
    assert cost.query_pair_count == 16
    assert cost.token_shards == 16
    assert cost.token_lanes == 8
    assert cost.pair_feature_width == 256
    assert cost.complex_source_count == 8
    assert cost.pairfm_useful_rows == 255
    assert cost.pairfm_padded_rows == 256
    assert cost.pairfm_pt_ct == 2048
    assert cost.pairfm_rotations == 248
    assert cost.blocked_useful_rows == 511
    assert cost.blocked_padded_rows == 512
    assert cost.blocked_pt_ct == 4096
    assert cost.blocked_rotations == 376
    assert cost.owner_encoded_rows == 16
    assert cost.owner_pt_ct == 256
    assert cost.owner_rotations == 240
    assert cost.conversion_encoded_rows == 784
    assert cost.conversion_pt_ct == 6400
    assert cost.conversion_rotations == 864
    assert cost.conversion_conjugations == 8
    assert cost.conversion_imults == 24
    assert cost.conversion_depth == 3


def test_pairfm_masks_are_exact_on_small_geometry() -> None:
    feature, attention = _small_layouts()
    first, first_stride = _pairfm_masks(
        feature, attention, dtype=np.dtype(np.float64)
    )
    second, second_stride = _blocked_masks(
        feature, attention, dtype=np.dtype(np.float64)
    )
    assert first_stride == 1
    assert first.shape == (15, 128)
    assert second_stride == 4
    assert second.shape == (31, 128)

    for source in range(128):
        d, q, lane = np.unravel_index(source, (8, 8, 2))
        destination = np.ravel_multi_index((d, lane, q), (8, 2, 8))
        offset = source - destination
        assert first[offset + 7, destination] == 1.0

    for source in range(128):
        feature_index, shard, local = np.unravel_index(source, (16, 2, 4))
        destination = np.ravel_multi_index(
            (shard, feature_index, local), (2, 16, 4)
        )
        offset = source - destination
        row = offset // second_stride + 15
        assert offset % second_stride == 0
        assert second[row, destination] == 1.0


@pytest.mark.gpu
def test_pairfm_output_fhe_small() -> None:
    """CUDA smoke at the smallest full-payload ring supported by EasyFHE."""

    rng = np.random.default_rng(26081444)
    seq_len = head_dim = 64
    query_heads = 8
    dimension = query_heads * head_dim
    slots = 8192
    attention = AttentionPairLayout(
        seq_len=seq_len,
        head_dim=head_dim,
        query_heads=query_heads,
        key_value_heads=2,
        slots=slots,
    )
    feature = FeatureMajorPrefillLayout(
        seq_len=seq_len, hidden_dim=dimension, slots=slots
    )
    logical = rng.normal(scale=0.003, size=(seq_len, dimension)).astype(np.float32)
    output_c = attention.pack_output_c(logical, dtype=np.float64)

    config = Llama3CKKSConfig()
    config.simulator.logN = 14
    config.simulator.maxLevelsRemaining = 6
    config.simulator.device = "cuda"
    config.layout_transform.dtype = "float64"
    rotations = pairfm_output_rotations(
        feature,
        attention,
        first_baby_steps=16,
        block_baby_steps=16,
        log_n=14,
    )
    context = create_sim_context(
        config.simulator,
        input_dim=dimension,
        block_out=feature.tokens_per_cipher,
        rotations=rotations,
        include_default_rotations=False,
    )
    inputs: tuple[object, ...] = ()
    result = None
    try:
        inputs = tuple(context.encrypt(row, level=0, slots=slots) for row in output_c)
        sync_device(context.device)
        result = prepare_w_o_inputs_fhe(
            inputs,
            feature_layout=feature,
            attention_layout=attention,
            crypto_context=context,
            model_config=config,
            operator_config=AttentionOperatorConfig(
                pairfm_first_baby_steps=16, pairfm_block_baby_steps=16
            ),
            verify=True,
            expected_output_c=output_c,
            function_prefix="smoke.output_pairfm",
        )
        assert len(result.real_ciphers) == 4
        assert len(result.carrier_ciphers) == 2
        assert {context.level_for_cipher(c) for c in result.real_ciphers} == {3}
        assert {context.level_for_cipher(c) for c in result.carrier_ciphers} == {3}
        assert result.max_abs_diff is not None
        assert result.max_abs_diff < 1.2e-2
    finally:
        if result is not None:
            result.release()
        for cipher in inputs:
            if hasattr(cipher, "release"):
                cipher.release()




def test_pairfm_masks_are_memoised_per_process() -> None:
    from llama3fhe.operators.transform import _bsgs_plaintext_rows

    feature, attention = _small_layouts()
    first, stride = _pairfm_masks(feature, attention, dtype=np.dtype(np.float64))
    again, again_stride = _pairfm_masks(
        feature, attention, dtype=np.dtype(np.float64)
    )
    assert again is first and again_stride == stride
    assert not first.flags.writeable
    other, _ = _pairfm_masks(feature, attention, dtype=np.dtype(np.float32))
    assert other is not first and other.dtype == np.float32

    rows_a = _bsgs_plaintext_rows(first, stride=stride, baby_steps=4)
    rows_b = _bsgs_plaintext_rows(first, stride=stride, baby_steps=4)
    assert rows_a[0] is rows_b[0]
    assert rows_a[1:] == rows_b[1:]
    # memoised rows are read-only so transform_plaintext_scope can key on their identity
    assert not rows_a[0].flags.writeable

    writable = np.array(first)
    rows_c = _bsgs_plaintext_rows(writable, stride=stride, baby_steps=4)
    rows_d = _bsgs_plaintext_rows(writable, stride=stride, baby_steps=4)
    assert rows_c[0] is not rows_d[0]
    np.testing.assert_array_equal(rows_c[0], rows_a[0])
