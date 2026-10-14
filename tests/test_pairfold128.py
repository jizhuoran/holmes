from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.pairfold.pairfold128 import (
    folded_w_o_pairfold128_numpy,
    pv_pairfold128_numpy,
    qk_pairfold128_numpy,
)


def test_pairfold128_slot_contract_is_bijective() -> None:
    layout = PairFold128Layout()

    channels = layout.active_carrier_channels()
    assert len(channels) == layout.active_channel_count == 3072
    assert len(set(channels)) == 3072
    assert max(channels) == 4087
    assert layout.carrier_dimension == 4096
    assert layout.cache_cipher_count == 4
    assert layout.query_residue_count == 8

    page_slots = {
        layout.page_slot(pair, feature, token, lane)
        for pair in range(layout.pair_blocks)
        for feature in range(layout.fold_dim)
        for token in range(layout.seq_len)
        for lane in range(layout.head_lanes)
    }
    assert page_slots == set(range(layout.slots))

    score_slots = {
        layout.score_slot(group, pair, query_offset, token, lane)
        for group in range(layout.key_value_groups)
        for pair in range(layout.pair_blocks)
        for query_offset in range(layout.gqa_ratio)
        for token in range(layout.seq_len)
        for lane in range(layout.head_lanes)
    }
    assert score_slots == set(range(layout.query_heads * layout.seq_len))

    output_slots = {
        layout.output_slot(group, pair, query_offset, feature, lane)
        for group in range(layout.key_value_groups)
        for pair in range(layout.pair_blocks)
        for query_offset in range(layout.gqa_ratio)
        for feature in range(layout.fold_dim)
        for lane in range(layout.head_lanes)
    }
    assert len(output_slots) == layout.query_heads * layout.fold_dim
    assert layout.token_rotations() == (2, 4, 8, 16, 32, 64, 128)
    assert layout.feature_rotations() == (256, 512, 1024, 2048, 4096, 8192)


def test_pairfold128_route_offsets_cover_every_boundary() -> None:
    layout = PairFold128Layout()
    for residue in range(12):
        for pair in range(layout.pair_blocks):
            for feature in range(layout.fold_dim):
                for lane in range(layout.head_lanes):
                    source = layout.carrier_slot(pair, feature, residue, lane)
                    for token in range(layout.seq_len):
                        rotation = layout.carrier_to_page_rotation(
                            residue, lane, token
                        )
                        destination = (source - rotation) % layout.slots
                        assert destination == layout.page_slot(
                            pair, feature, token, lane
                        )

    for group in range(layout.key_value_groups):
        for pair in range(layout.pair_blocks):
            for query_offset in range(layout.gqa_ratio):
                score_rotation = layout.page_to_score_rotation(
                    group, pair, query_offset
                )
                for token in range(layout.seq_len):
                    for lane in range(layout.head_lanes):
                        source = layout.page_slot(pair, 0, token, lane)
                        destination = (source - score_rotation) % layout.slots
                        assert destination == layout.score_slot(
                            group, pair, query_offset, token, lane
                        )
                for feature in range(layout.fold_dim):
                    for lane in range(layout.head_lanes):
                        source = layout.page_slot(pair, feature, 0, lane)
                        output_rotation = layout.page_to_output_rotation(
                            group, query_offset, lane
                        )
                        destination = (source - output_rotation) % layout.slots
                        assert destination == layout.output_slot(
                            group,
                            pair,
                            query_offset,
                            feature,
                            lane,
                        )


def test_pairfold128_rejects_non_production_shape() -> None:
    with pytest.raises(ValueError, match="seq_len=128"):
        PairFold128Layout(seq_len=64)


def test_pairfold128_qkv_carrier_round_trip_and_direct_columns() -> None:
    rng = np.random.default_rng(26092101)
    layout = PairFold128Layout()
    query = rng.normal(size=(layout.query_heads, layout.head_dim))
    key = rng.normal(size=(layout.key_value_heads, layout.head_dim))
    value = rng.normal(size=(layout.key_value_heads, layout.head_dim))

    carrier = layout.pack_qkv_carrier(query, key, value)
    got_query, got_key, got_value = layout.unpack_qkv_carrier(carrier)
    np.testing.assert_allclose(got_query, query, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got_key, key, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got_value, value, atol=1e-12, rtol=1e-12)

    packed_query = layout.pack_query(query)
    token_index = 73
    key_page = np.zeros(
        (layout.seq_len, layout.key_value_heads, layout.head_dim),
        dtype=np.float64,
    )
    value_page = np.zeros_like(key_page)
    key_page[token_index] = key
    value_page[token_index] = value
    packed_key = layout.pack_key(key_page)
    packed_value = layout.pack_value(value_page)

    for group in range(layout.key_value_groups):
        for pair in range(layout.pair_blocks):
            for lane in range(layout.head_lanes):
                for feature in (0, 17, 63):
                    target = layout.page_slot(pair, feature, token_index, lane)
                    key_source = layout.carrier_slot(
                        pair, feature, layout.key_residue(group), lane
                    )
                    value_source = layout.carrier_slot(
                        pair, feature, layout.value_residue(group), lane
                    )
                    assert carrier[0, key_source] == packed_key[group, target]
                    assert carrier[0, value_source] == packed_value[group, target]
                    for query_offset in range(layout.gqa_ratio):
                        source = layout.carrier_slot(
                            pair,
                            feature,
                            layout.query_residue(group, query_offset),
                            lane,
                        )
                        q_target = layout.page_slot(pair, feature, 0, lane)
                        cipher = layout.query_cipher_index(group, query_offset)
                        assert carrier[0, source] == packed_query[cipher, q_target]


def test_pairfold128_query_and_kv_page_round_trip() -> None:
    rng = np.random.default_rng(26092102)
    layout = PairFold128Layout()
    occupied = 37
    query = rng.normal(size=(layout.query_heads, layout.head_dim))
    key = rng.normal(size=(occupied, layout.key_value_heads, layout.head_dim))
    value = rng.normal(size=key.shape)

    np.testing.assert_allclose(
        layout.unpack_query(layout.pack_query(query)), query, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        layout.unpack_key(layout.pack_key(key), occupied=occupied),
        key,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        layout.unpack_value(layout.pack_value(value), occupied=occupied),
        value,
        atol=1e-12,
        rtol=1e-12,
    )


def test_pairfold128_scores_are_exactly_legacy_compact_scores() -> None:
    layout = PairFold128Layout()
    legacy = AttentionPairLayout()
    scores = np.arange(
        layout.query_heads * layout.seq_len, dtype=np.float32
    ).reshape(layout.query_heads, layout.seq_len)
    packed = layout.pack_scores(scores, dtype=np.float32)
    np.testing.assert_array_equal(packed, legacy.pack_compact_decode_scores(scores))
    np.testing.assert_array_equal(layout.unpack_scores(packed), scores)

    occupied = 91
    prefix = layout.unpack_scores(layout.score_prefix_mask(occupied))
    np.testing.assert_array_equal(prefix[:, :occupied], 1.0)
    np.testing.assert_array_equal(prefix[:, occupied:], 0.0)
    token0 = layout.unpack_scores(layout.score_token0_mask())
    np.testing.assert_array_equal(token0[:, :1], 1.0)
    np.testing.assert_array_equal(token0[:, 1:], 0.0)


def test_pairfold128_qk_numpy_matches_logical_gqa_float64() -> None:
    rng = np.random.default_rng(26092104)
    layout = PairFold128Layout()
    occupied = 91
    query = rng.normal(size=(layout.query_heads, layout.head_dim))
    key = rng.normal(
        size=(layout.seq_len, layout.key_value_heads, layout.head_dim)
    )

    packed_scores = qk_pairfold128_numpy(
        layout.pack_query(query),
        layout.pack_key(key),
        layout=layout,
        occupied=occupied,
    )
    scores = layout.unpack_scores(packed_scores)
    key_for_query = key[:, np.arange(layout.query_heads) // layout.gqa_ratio, :]
    expected = np.einsum("hd,thd->ht", query, key_for_query, optimize=True)
    expected[:, occupied:] = 0.0
    np.testing.assert_allclose(scores, expected, atol=1e-11, rtol=1e-11)


def test_pairfold128_pv_numpy_matches_logical_gqa_float64() -> None:
    rng = np.random.default_rng(26092105)
    layout = PairFold128Layout()
    occupied = 91
    probabilities = rng.uniform(
        0.0, 1.0, size=(layout.query_heads, layout.seq_len)
    )
    probabilities[:, occupied:] = 0.0
    probabilities[:, :occupied] /= np.sum(
        probabilities[:, :occupied], axis=-1, keepdims=True
    )
    value = rng.normal(
        size=(layout.seq_len, layout.key_value_heads, layout.head_dim)
    )

    packed_output = pv_pairfold128_numpy(
        layout.pack_scores(probabilities),
        layout.pack_value(value),
        layout=layout,
        occupied=occupied,
    )
    output = layout.unpack_output(packed_output)
    value_for_query = value[:, np.arange(layout.query_heads) // layout.gqa_ratio, :]
    expected = np.einsum(
        "ht,thd->hd",
        probabilities[:, :occupied],
        value_for_query[:occupied],
        optimize=True,
    )
    np.testing.assert_allclose(output, expected, atol=1e-11, rtol=1e-11)


def test_pairfold128_folded_output_and_w_o_identity() -> None:
    rng = np.random.default_rng(26092106)
    layout = PairFold128Layout()
    output = rng.normal(size=(layout.query_heads, layout.head_dim)).astype(
        np.float32
    )
    packed = layout.pack_output(output, dtype=np.complex64)
    np.testing.assert_allclose(
        layout.unpack_output(packed), output, atol=1e-6, rtol=1e-6
    )

    dimension = layout.query_heads * layout.head_dim
    weight = np.eye(dimension, dtype=np.float32)
    projected = folded_w_o_pairfold128_numpy(packed, weight, layout=layout)
    np.testing.assert_allclose(
        projected, output.reshape(1, -1), atol=1e-5, rtol=1e-5
    )


def test_pairfold128_folded_w_o_permutation_uses_both_feature_halves() -> None:
    layout = PairFold128Layout()
    logical = np.arange(
        layout.query_heads * layout.head_dim, dtype=np.float32
    ).reshape(layout.query_heads, layout.head_dim)
    packed = layout.pack_output(logical, dtype=np.complex64)
    dimension = layout.query_heads * layout.head_dim
    permutation = np.arange(dimension - 1, -1, -1, dtype=np.int64)
    weight = np.zeros((dimension, dimension), dtype=np.float32)
    weight[np.arange(dimension), permutation] = 1.0
    projected = folded_w_o_pairfold128_numpy(packed, weight, layout=layout)
    expected = logical.reshape(1, -1)[:, permutation]
    np.testing.assert_allclose(projected, expected, atol=1e-5, rtol=1e-5)
