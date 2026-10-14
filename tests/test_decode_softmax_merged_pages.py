"""Merged-page Decode Softmax (all CT score pages of one query in one ciphertext).

The merged kernel must be slot-for-slot the same Alg2 as the per-page kernel
(``LLAMA_FHE_SOFTMAX_MERGE_PAGES=0``) while spending one exp / one post-exp
bootstrap / one set of round multiplications for every page.
"""

from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.decode.decode import decode_attention_rotations
from llama3fhe.operators.attention.decode.decode_softmax import (
    BOOTSTRAP_MIN_INPUT_LIMBS,
    SOFTMAX_MERGE_PAGES_ENV,
    SOFTMAX_SPARSE_BS_ENV,
    DecodeAlg2Poly,
    _merged_page_masks,
    merged_page_sparse_period,
    softmax_alg2_row_pages_fhe,
    softmax_merge_pages_enabled,
    softmax_page_merge_rotations,
)
from llama3fhe.operators.attention.pairfold.pairfold_decode import (
    pairfold_decode_rotations,
)


def _layout() -> AttentionPairLayout:
    # head_dim=4 rows, 2 query pairs -> 2 merged groups of 16 slots each.
    return AttentionPairLayout(
        seq_len=4, head_dim=4, query_heads=4, key_value_heads=2, slots=8192
    )


def _production_layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=128, head_dim=128, query_heads=32, key_value_heads=8, slots=32768
    )


def test_merge_geometry_matches_production_pages() -> None:
    layout = _production_layout()
    assert layout.decode_page_merge_stride == 4096
    assert layout.decode_page_merge_groups == 8
    rotations = layout.decode_page_merge_rotations()
    assert set(rotations) == {4096 * g for g in range(-7, 8) if g} | {4096, 8192, 16384}
    assert set(rotations) <= set(decode_attention_rotations(layout))
    # pairfold decode shares the softmax kernel, so its key union must cover it
    assert set(softmax_page_merge_rotations(layout)) == set(rotations)
    fold = PairFold128Layout()
    assert set(softmax_page_merge_rotations(fold)) == set(rotations)
    assert set(rotations) <= set(pairfold_decode_rotations(fold))


def test_merged_masks_shift_pages_into_their_group() -> None:
    layout = _layout()
    masks = _merged_page_masks(layout, (4, 2), None)
    stride = layout.decode_page_merge_stride
    period = layout.payload_slots
    valid0 = masks.page_valid[0][0, :period]
    valid1 = masks.page_valid[1][0, :period]
    merged = masks.merged_valid[0, :period]
    # page 0 stays on rows 0..1 (slots 0..15), page 1 moves to rows 2..3
    assert np.array_equal(merged[:stride], valid0[:stride])
    assert np.array_equal(merged[stride:], valid1[:stride])
    assert valid0[stride:].sum() == 0 and valid1[stride:].sum() == 0
    # the mask is periodic over the payload repetitions
    assert np.array_equal(
        masks.merged_valid.reshape(-1, period), np.tile(merged, (layout.payload_repetitions, 1))
    )
    # token-0 mask covers column 0 of both rows in both groups
    token0 = masks.merged_token0[0, :period].reshape(layout.head_dim, layout.seq_len, 2)
    assert np.array_equal(token0[:, 0, :], np.ones((4, 2)))
    assert token0[:, 1:, :].sum() == 0


def test_merge_toggle_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SOFTMAX_MERGE_PAGES_ENV, raising=False)
    assert softmax_merge_pages_enabled()
    monkeypatch.setenv(SOFTMAX_MERGE_PAGES_ENV, "0")
    assert not softmax_merge_pages_enabled()


class _CountingRefresh:
    """Bootstrap stand-in: re-encrypts at ``limbs`` and counts the calls."""

    def __init__(self, crypto_context, *, limbs: int, slots: int) -> None:
        self.crypto_context = crypto_context
        self.limbs = int(limbs)
        self.slots = int(slots)
        self.calls = 0
        self.config = type("Config", (), {"iterations": 1, "precision_bits": 0})()

    def refresh(self, cipher, *, iterations=None, precision_bits=None):
        del iterations, precision_bits
        self.calls += 1
        assert int(cipher.state.cur_limbs) >= BOOTSTRAP_MIN_INPUT_LIMBS
        values = np.asarray(self.crypto_context.decrypt(cipher), dtype=np.float64).real
        level = int(self.crypto_context.context.L) - self.limbs
        return self.crypto_context.encrypt(values, level=level, slots=self.slots)


def _context(layout: AttentionPairLayout, *, depth: int):
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = int(depth)
    cfg.simulator.device = "cuda"
    return create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=decode_attention_rotations(layout),
        include_default_rotations=True,
    )


def _decrypt(ciphers, *, layout, context) -> np.ndarray:
    rows = np.stack([np.asarray(context.decrypt(c), dtype=np.float64).real for c in ciphers])
    return layout.unpack_compact_decode_scores(rows)


def _run_pages(context, layout, pages, config, *, occupied, rounds, head_rows, bootstrap):
    probs = softmax_alg2_row_pages_fhe(
        pages,
        layout=layout,
        crypto_context=context,
        config=config,
        occupied_pages=occupied,
        layer_idx=0,
        rounds=rounds,
        bootstrap_operator=bootstrap,
        bootstrap_iterations=1,
        bootstrap_precision_bits=0,
        head_rows=head_rows,
    )
    try:
        return tuple(_decrypt(page, layout=layout, context=context) for page in probs)
    finally:
        for page in probs:
            for cipher in page:
                release_if_supported(cipher)


@pytest.mark.gpu
@pytest.mark.parametrize("head_rows", [None, (0, 2)])
@pytest.mark.parametrize("rounds", [1, 2])
def test_merged_pages_match_per_page_kernel(
    monkeypatch: pytest.MonkeyPatch, rounds: int, head_rows
) -> None:
    rng = np.random.default_rng(26100101 + rounds)
    layout = _layout()
    occupied = (4, 2)
    scores = [rng.normal(scale=0.3, size=(4, 4)).astype(np.float32) for _ in occupied]
    if head_rows is not None:
        for page in scores:
            page[[1, 3], :] = 0.0
    config = DecodeAlg2Poly.synthetic(k=rounds, degree=8)
    context = _context(layout, depth=40)
    ciphers = tuple(
        tuple(context.encrypt(row, level=0, slots=layout.slots) for row in layout.pack_compact_decode_scores(s))
        for s in scores
    )
    try:
        monkeypatch.setenv(SOFTMAX_MERGE_PAGES_ENV, "0")
        per_page = _run_pages(
            context, layout, ciphers, config, occupied=occupied, rounds=rounds, head_rows=head_rows, bootstrap=None
        )
        monkeypatch.setenv(SOFTMAX_MERGE_PAGES_ENV, "1")
        merged = _run_pages(
            context, layout, ciphers, config, occupied=occupied, rounds=rounds, head_rows=head_rows, bootstrap=None
        )
    finally:
        for page in ciphers:
            for cipher in page:
                release_if_supported(cipher)
    assert len(merged) == len(per_page) == 2
    for got, want, occ in zip(merged, per_page, occupied, strict=True):
        np.testing.assert_allclose(got, want, atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(got[:, occ:], 0.0, atol=1e-3)
        if head_rows is not None:
            np.testing.assert_allclose(got[[1, 3], :], 0.0, atol=1e-3)
    # the union of both pages is one softmax row: probabilities sum to ~1
    rows = np.concatenate([page[:, :occ] for page, occ in zip(merged, occupied, strict=True)], axis=1)
    owned = list(head_rows) if head_rows is not None else list(range(4))
    np.testing.assert_allclose(rows[owned].sum(axis=1), 1.0, atol=0.2)  # synthetic k=1 poly


@pytest.mark.gpu
def test_merged_pages_bootstrap_once_per_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-page: post_exp + (fill?) + lam per page; merged: one of each."""

    rng = np.random.default_rng(26100102)
    layout = _layout()
    occupied = (4, 2)
    scores = [rng.normal(scale=0.3, size=(4, 4)).astype(np.float32) for _ in occupied]
    config = DecodeAlg2Poly.synthetic(k=1, degree=8)
    context = _context(layout, depth=40)
    ciphers = tuple(
        tuple(context.encrypt(row, level=0, slots=layout.slots) for row in layout.pack_compact_decode_scores(s))
        for s in scores
    )
    try:
        monkeypatch.setenv(SOFTMAX_MERGE_PAGES_ENV, "0")
        counter = _CountingRefresh(context, limbs=20, slots=layout.slots)
        per_page = _run_pages(
            context, layout, ciphers, config, occupied=occupied, rounds=1, head_rows=None, bootstrap=counter
        )
        per_page_calls = counter.calls
        monkeypatch.setenv(SOFTMAX_MERGE_PAGES_ENV, "1")
        counter = _CountingRefresh(context, limbs=20, slots=layout.slots)
        merged = _run_pages(
            context, layout, ciphers, config, occupied=occupied, rounds=1, head_rows=None, bootstrap=counter
        )
        merged_calls = counter.calls
    finally:
        for page in ciphers:
            for cipher in page:
                release_if_supported(cipher)
    # per page: 2 post_exp + 1 lam = 3; merged: 1 post_exp + 1 lam = 2
    assert per_page_calls == 3
    assert merged_calls == 2
    for got, want in zip(merged, per_page, strict=True):
        np.testing.assert_allclose(got, want, atol=2e-3, rtol=2e-3)


class _PeriodicCheckingRefresh(_CountingRefresh):
    """Stand-in with one sparse program: a ``periodic_slots`` refresh asserts the
    decrypted message really has that period (what the real sparse view relies on)."""

    def __init__(self, crypto_context, *, limbs: int, slots: int, period: int) -> None:
        super().__init__(crypto_context, limbs=limbs, slots=slots)
        self.period = int(period)
        self.sparse_calls = 0
        self.dense_calls = 0

    def supports_periodic_slots(self, period: int) -> bool:
        return int(period) == self.period

    def refresh(self, cipher, *, iterations=None, precision_bits=None, periodic_slots=None):
        if periodic_slots is None:
            self.dense_calls += 1
        else:
            assert int(periodic_slots) == self.period
            self.sparse_calls += 1
            values = np.asarray(self.crypto_context.decrypt(cipher), dtype=np.float64).real
            blocks = values.reshape(-1, int(periodic_slots))
            np.testing.assert_allclose(blocks, np.broadcast_to(blocks[:1], blocks.shape), atol=2e-3, rtol=2e-3)
        return super().refresh(cipher, iterations=iterations, precision_bits=precision_bits)


def test_merged_page_sparse_period_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    layout = _production_layout()
    operator = _PeriodicCheckingRefresh.__new__(_PeriodicCheckingRefresh)
    operator.period = 4096
    monkeypatch.delenv(SOFTMAX_SPARSE_BS_ENV, raising=False)
    assert merged_page_sparse_period(layout, operator) == 4096
    assert merged_page_sparse_period(layout, None) is None
    operator.period = 2048  # no program of the right period -> dense
    assert merged_page_sparse_period(layout, operator) is None
    operator.period = 4096
    monkeypatch.setenv(SOFTMAX_SPARSE_BS_ENV, "0")
    assert merged_page_sparse_period(layout, operator) is None
    # legacy single-program operators without supports_periodic_slots stay dense
    monkeypatch.delenv(SOFTMAX_SPARSE_BS_ENV, raising=False)
    legacy = type("Legacy", (), {"sparse_slots": 4096})()
    assert merged_page_sparse_period(layout, legacy) is None


def test_merged_masks_are_stride_periodic() -> None:
    """``fill_mask = 1 - merged_token0`` must repeat every ``stride`` slots (it is added to the
    rotate-add total before the sparse refresh), for the full and the partial-world head rows."""

    for layout in (_production_layout(), _layout()):
        stride = layout.decode_page_merge_stride
        for head_rows in (None, (0, 2), (1,)):
            if head_rows is not None and max(head_rows) >= layout.query_heads:
                continue
            from llama3fhe.operators.attention.decode.decode_softmax import _head_rows_mask

            masks = _merged_page_masks(layout, (layout.seq_len, 2), _head_rows_mask(layout, head_rows))
            blocks = masks.merged_token0.reshape(-1, stride)
            np.testing.assert_array_equal(blocks, np.broadcast_to(blocks[:1], blocks.shape))


@pytest.mark.gpu
@pytest.mark.parametrize("head_rows", [None, (0, 2)])
def test_merged_pages_sparse_refresh_sees_periodic_rows(
    monkeypatch: pytest.MonkeyPatch, head_rows
) -> None:
    """With a sparse program of period ``stride`` the fill / lambda refreshes go sparse and the
    message they see is exactly ``stride``-periodic; the probabilities match the dense run."""

    rng = np.random.default_rng(26100201)
    layout = _layout()
    occupied = (4, 2)
    scores = [rng.normal(scale=0.3, size=(4, 4)).astype(np.float32) for _ in occupied]
    config = DecodeAlg2Poly.synthetic(k=2, degree=8)
    context = _context(layout, depth=40)
    stride = layout.decode_page_merge_stride
    ciphers = tuple(
        tuple(context.encrypt(row, level=0, slots=layout.slots) for row in layout.pack_compact_decode_scores(s))
        for s in scores
    )
    try:
        monkeypatch.setenv(SOFTMAX_MERGE_PAGES_ENV, "1")
        monkeypatch.setenv(SOFTMAX_SPARSE_BS_ENV, "0")
        dense_counter = _PeriodicCheckingRefresh(context, limbs=20, slots=layout.slots, period=stride)
        dense = _run_pages(
            context, layout, ciphers, config, occupied=occupied, rounds=2, head_rows=head_rows, bootstrap=dense_counter
        )
        monkeypatch.setenv(SOFTMAX_SPARSE_BS_ENV, "1")
        sparse_counter = _PeriodicCheckingRefresh(context, limbs=20, slots=layout.slots, period=stride)
        sparse = _run_pages(
            context, layout, ciphers, config, occupied=occupied, rounds=2, head_rows=head_rows, bootstrap=sparse_counter
        )
    finally:
        for page in ciphers:
            for cipher in page:
                release_if_supported(cipher)
    assert dense_counter.sparse_calls == 0
    # post_exp stays dense (page contents differ); the k=2 lambda refreshes go sparse
    assert sparse_counter.dense_calls == 1
    assert sparse_counter.sparse_calls == 2
    assert sparse_counter.calls == dense_counter.calls
    for got, want in zip(sparse, dense, strict=True):
        np.testing.assert_allclose(got, want, atol=2e-3, rtol=2e-3)
