"""CPU slot and key-plan contracts for direct Delta-V to PairFold V."""

from __future__ import annotations

import numpy as np
import pytest

from llama3fhe import graph
from llama3fhe.graph.common import _prefill_delta_pairfold_v_enabled
from llama3fhe.layouts import AttentionPairLayout, FeatureMajorPrefillLayout
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.prefill.input import (
    _value_delta_pairfold_high_routes,
    _value_delta_pairfold_low_routes,
    _value_delta_pairfold_routes,
    choose_sparse_diagonal_plan,
    emit_cache_c_rotations,
)
from llama3fhe.operators.attention.sparse_bsgs_scheduler import sparse_route_arrays


def _apply_routes(rows: np.ndarray, routes: dict) -> np.ndarray:
    result = np.zeros_like(rows, dtype=np.complex128)
    for offset, route in routes.items():
        destinations, coefficients = sparse_route_arrays(route)
        sources = (destinations + offset) % rows.shape[1]
        result[:, destinations] += rows[:, sources] * coefficients
    return result


@pytest.mark.parametrize("occupied", (1, 127, 128))
def test_delta_value_routes_match_pairfold_page(occupied: int) -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    rng = np.random.default_rng(20261009)
    value = rng.normal(size=(128, 8, 128)).astype(np.float32)
    delta = attention.pack_value_delta(value)
    actual = np.zeros((pairfold.key_value_groups, pairfold.slots), dtype=np.complex128)

    for pair in range(pairfold.pair_blocks):
        for offset, route in _value_delta_pairfold_routes(pair, occupied).items():
            destinations = route.destinations
            sources = (destinations + offset) % pairfold.slots
            for group in range(pairfold.key_value_groups):
                actual[group, destinations] += (
                    route.coefficients * delta[group * pairfold.pair_blocks + pair, sources]
                )

    wanted = pairfold.pack_value(value)
    for pair in range(pairfold.pair_blocks):
        for feature in range(pairfold.fold_dim):
            for token in range(occupied, pairfold.seq_len):
                wanted[:, pairfold.page_slot(pair, feature, token, 0)] = 0
                wanted[:, pairfold.page_slot(pair, feature, token, 1)] = 0
    np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("occupied", (1, 127, 128))
def test_staged_delta_value_routes_match_pairfold_page(occupied: int) -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    rng = np.random.default_rng(20261009)
    value = rng.normal(size=(128, 8, 128)).astype(np.float32)
    delta = attention.pack_value_delta(value)
    unsheared = _apply_routes(delta, _value_delta_pairfold_low_routes())
    actual = np.zeros((pairfold.key_value_groups, pairfold.slots), dtype=np.complex128)

    for pair in range(pairfold.pair_blocks):
        routes = _value_delta_pairfold_high_routes(pair, occupied)
        actual += _apply_routes(unsheared[[pair, pair + 2]], routes)

    wanted = pairfold.pack_value(value)
    for pair in range(pairfold.pair_blocks):
        for feature in range(pairfold.fold_dim):
            for token in range(occupied, pairfold.seq_len):
                wanted[:, pairfold.page_slot(pair, feature, token, 0)] = 0
                wanted[:, pairfold.page_slot(pair, feature, token, 1)] = 0
    np.testing.assert_array_equal(actual, wanted)


def test_staged_delta_value_route_counts_and_rotation_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = AttentionPairLayout()
    carrier = LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768)
    routes = (
        _value_delta_pairfold_low_routes(),
        _value_delta_pairfold_high_routes(0, 128),
        _value_delta_pairfold_high_routes(1, 128),
    )
    assert [len(route) for route in routes] == [31, 15, 15]
    expected: set[int] = set()
    for route in routes:
        plan = choose_sparse_diagonal_plan(
            tuple(route), slots=attention.slots, max_baby_offsets=32
        )
        expected.update(plan.baby_offsets)
        expected.update(plan.giant_offsets)
    expected.discard(0)
    monkeypatch.setenv("LLAMA_FHE_PREFILL_DELTA_PAIRFOLD_V_STAGED", "1")
    assert set(emit_cache_c_rotations(carrier, attention, direct_pairfold_v=True)) == expected
    monkeypatch.setenv("LLAMA_FHE_PREFILL_DELTA_PAIRFOLD_V", "1")
    assert expected <= set(
        graph.required_rotations(
            FeatureMajorPrefillLayout(),
            attention,
            emit_kv_cache=True,
            decode_packing="pairfold128",
        )
    )


def test_delta_value_rotation_keys_cover_both_pair_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = AttentionPairLayout()
    carrier = LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768)
    direct = set(
        emit_cache_c_rotations(carrier, attention, direct_pairfold_v=True)
    )
    expected: set[int] = set()
    for pair in (0, 1):
        routes = _value_delta_pairfold_routes(pair, 128)
        plan = choose_sparse_diagonal_plan(
            tuple(routes), slots=attention.slots, max_baby_offsets=32
        )
        expected.update(plan.baby_offsets)
        expected.update(plan.giant_offsets)
    expected.discard(0)
    assert direct == expected

    monkeypatch.setenv("LLAMA_FHE_PREFILL_DELTA_PAIRFOLD_V", "1")
    assert direct <= set(
        graph.required_rotations(
            FeatureMajorPrefillLayout(),
            attention,
            emit_kv_cache=True,
            decode_packing="pairfold128",
        )
    )
    assert _prefill_delta_pairfold_v_enabled(True, "pairfold128")
    assert not _prefill_delta_pairfold_v_enabled(False, "pairfold128")
    assert not _prefill_delta_pairfold_v_enabled(True, "legacy_c")
