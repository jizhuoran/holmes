"""Periodic slim pack of the Prefill Softmax Alg2 row sums (sparse-BS enabler).

Geometry (CPU) + a small-context GPU check that pack -> broadcast through the
periodic layout returns exactly what the legacy one-row-per-pair layout does.
"""

from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.attention.audit import delta_softmax_rotations
from llama3fhe.operators.attention.prefill.softmax import (
    _pair_slice_mask,
    build_delta_softmax_mask_bundle,
    delta_broadcast_slim_rows_fhe,
    delta_pack_row_sums_slim_fhe,
    delta_slim_active_mask,
    delta_slim_broadcast_rotations,
    delta_softmax_row_rotations,
    softmax_slim_period_rows,
    softmax_slim_period_slots,
)


def _layout(pairs: int, *, seq_len: int = 128, slots: int = 32768) -> AttentionPairLayout:
    kv = 2 if pairs < 4 else 2 * max(1, pairs // 4)
    return AttentionPairLayout(seq_len=seq_len, head_dim=seq_len, query_heads=2 * pairs, key_value_heads=kv, slots=slots)


@pytest.mark.parametrize("pairs, period, log_slots", [(2, 2, 9), (4, 4, 10), (8, 8, 11), (16, 16, 12), (3, 4, 10)])
def test_period_rows_follow_the_rank_pair_count(pairs: int, period: int, log_slots: int) -> None:
    layout = _layout(pairs)
    assert softmax_slim_period_rows(layout) == period
    slots = softmax_slim_period_slots(layout)
    assert slots == period * 2 * 128
    assert slots.bit_length() - 1 == log_slots
    assert layout.slots % slots == 0


def test_periodic_pair_masks_are_periodic_disjoint_and_cover_the_rows() -> None:
    layout = _layout(2)
    period = softmax_slim_period_rows(layout)
    slots = softmax_slim_period_slots(layout, period)
    union = np.zeros(layout.slots)
    for pair in range(2):
        mask = _pair_slice_mask(layout, pair, period_rows=period)
        tiled = mask.reshape(-1, slots)
        np.testing.assert_array_equal(tiled, np.broadcast_to(tiled[0], tiled.shape))  # periodic
        assert mask.sum() == 2 * 128 * (128 // period)  # 64 rows x 256 slots
        legacy = _pair_slice_mask(layout, pair)
        assert legacy.sum() == 2 * 128  # one row
        np.testing.assert_array_equal(mask * legacy, legacy)  # the legacy row is one of the periodic rows
        union += mask
    np.testing.assert_array_equal(union, 1.0)  # P == pairs: every row live, disjoint
    np.testing.assert_array_equal(delta_slim_active_mask(layout, period_rows=period), 1.0)
    # P > pairs leaves periodic inactive residues
    odd = _layout(3)
    active = delta_slim_active_mask(odd, period_rows=4).reshape(128, 256)
    assert active[0::4].all() and active[1::4].all() and active[2::4].all() and not active[3::4].any()


def test_periodic_broadcast_rotations_are_a_subset_of_the_legacy_chain() -> None:
    layout = _layout(2)
    legacy = delta_softmax_row_rotations(layout)
    assert len(legacy) == 7
    periodic = delta_slim_broadcast_rotations(layout, period_rows=2)
    assert periodic == (256,)  # log2(P)=1 step inside each period
    assert set(periodic) <= set(legacy)
    assert delta_slim_broadcast_rotations(_layout(16), period_rows=16) == tuple(256 * (1 << b) for b in range(4))
    assert delta_slim_broadcast_rotations(layout) == legacy
    with pytest.raises(ValueError):
        delta_slim_broadcast_rotations(layout, period_rows=1)  # cannot hold 2 pairs
    with pytest.raises(ValueError):
        delta_slim_broadcast_rotations(layout, period_rows=6)  # not a power of two


def _small_layout() -> AttentionPairLayout:
    # 2 Q pairs x 1 KV pair (one W=8 rank), seq = head = 16: payload 512 slots, 16x tiled into 8192
    return AttentionPairLayout(seq_len=16, head_dim=16, query_heads=4, key_value_heads=2, slots=8192)


def _context(layout: AttentionPairLayout, *, depth: int):
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = int(depth)
    cfg.simulator.device = "cuda"
    return create_sim_context(
        cfg.simulator,
        input_dim=16,
        block_out=16,
        rotations=delta_softmax_rotations(layout),
        include_default_rotations=False,
    )


@pytest.mark.gpu
def test_periodic_pack_then_broadcast_matches_legacy_layout() -> None:
    """pack -> (identity) -> broadcast gives every pair its own row sums on all Delta rows, both layouts."""

    rng = np.random.default_rng(26100201)
    layout = _small_layout()
    pairs = int(layout.query_pair_count)
    period = softmax_slim_period_rows(layout)
    # ``reduced`` ciphers: row sums already broadcast over the Delta axis (what delta_reduce_rows_fhe returns):
    # the (query, head) matrix of pair p repeated on every Delta row and every payload repetition
    sums = [rng.uniform(1.0, 2.0, size=(layout.seq_len, 2)) for _ in range(pairs)]
    reduced_plain = [
        np.tile(np.tile(s[None, :, :], (layout.seq_len, 1, 1)).reshape(-1), layout.payload_repetitions) for s in sums
    ]
    context = _context(layout, depth=6)
    reduced = tuple(context.encrypt(v, level=0, slots=layout.slots) for v in reduced_plain)
    outputs: dict[str, list[np.ndarray]] = {}
    try:
        for name, rows in (("legacy", None), ("periodic", period)):
            masks = build_delta_softmax_mask_bundle(layout, crypto_context=context, period_rows=rows)
            slim = delta_pack_row_sums_slim_fhe(reduced, layout=layout, crypto_context=context, masks=masks)
            try:
                slim_plain = np.asarray(context.decrypt(slim), dtype=np.float64).real
                expect = sum(reduced_plain[p] * _pair_slice_mask(layout, p, period_rows=rows) for p in range(pairs))
                np.testing.assert_allclose(slim_plain, expect, atol=1e-4)
                if rows is not None:  # the slim message is periodic with the advertised period
                    tiled = slim_plain.reshape(-1, softmax_slim_period_slots(layout, rows))
                    np.testing.assert_allclose(tiled, np.broadcast_to(tiled[0], tiled.shape), atol=1e-4)
                pairs_out = delta_broadcast_slim_rows_fhe(
                    slim, layout=layout, crypto_context=context, masks=masks, period_rows=rows
                )
            finally:
                release_if_supported(slim)
            try:
                outputs[name] = [np.asarray(context.decrypt(c), dtype=np.float64).real for c in pairs_out]
            finally:
                for c in pairs_out:
                    release_if_supported(c)
    finally:
        for c in reduced:
            release_if_supported(c)
    for p in range(pairs):
        np.testing.assert_allclose(outputs["legacy"][p], reduced_plain[p], atol=1e-3)
        np.testing.assert_allclose(outputs["periodic"][p], reduced_plain[p], atol=1e-3)
