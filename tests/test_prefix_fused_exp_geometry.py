from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.attention.prefill.softmax import delta_causal_mask


def _rotate_left(values: np.ndarray, offset: int) -> np.ndarray:
    return np.roll(values, -int(offset), axis=-1)


@pytest.mark.parametrize("seq_len", [8, 128])
def test_two_prefix_scores_share_content_exp_without_slot_collisions(seq_len: int) -> None:
    layout = AttentionPairLayout(
        seq_len=seq_len,
        head_dim=seq_len,
        query_heads=4,
        key_value_heads=2,
        slots=2 * seq_len * seq_len,
    )
    rng = np.random.default_rng(2601006 + seq_len)
    prefix_scores = rng.normal(scale=0.2, size=(4, seq_len, 2)).astype(np.float32)
    content_scores = rng.normal(scale=0.2, size=(4, seq_len, seq_len)).astype(np.float32)
    prefix_c = layout.pack_prefix_scores_prefill(prefix_scores)
    content_delta = layout.pack_score_delta(content_scores)
    content_mask = delta_causal_mask(layout).astype(bool)

    regular_source = np.zeros(layout.slots, dtype=bool)
    overflow_source = np.zeros(layout.slots, dtype=bool)
    overflow = {(0, seq_len - 1), (1, seq_len - 2), (1, seq_len - 1)}
    for prefix_index in range(2):
        for query in range(seq_len):
            mask = overflow_source if (prefix_index, query) in overflow else regular_source
            for lane in (0, 1):
                mask[layout.c_slot(prefix_index, query, lane)] = True

    # FHE positive rotations shift left. The three values that cannot enter
    # Delta rows 1/2 use invalid cells in rows 3/4 until exp is complete.
    regular_shift = -2 * seq_len
    donor_shift = -(4 * seq_len + 4)
    recovery_shift = 2 * seq_len + 4
    regular_mask = _rotate_left(regular_source, regular_shift)
    donor_mask = _rotate_left(overflow_source, donor_shift)
    assert np.count_nonzero(regular_mask) == 2 * (2 * seq_len - 3)
    assert np.count_nonzero(donor_mask) == 6
    assert not np.any(regular_mask & donor_mask)
    assert not np.any(content_mask & (regular_mask | donor_mask))
    assert set(np.flatnonzero(donor_mask)[::2]) == {
        2 * (3 * seq_len + 1),
        2 * (4 * seq_len),
        2 * (4 * seq_len + 1),
    }

    recovered_mask = _rotate_left(donor_mask, recovery_shift)
    assert set(np.flatnonzero(recovered_mask)[::2]) == {
        2 * (seq_len + seq_len - 1),
        2 * (2 * seq_len + seq_len - 2),
        2 * (2 * seq_len + seq_len - 1),
    }

    active = content_mask | regular_mask | donor_mask
    combined_scores = (
        content_delta * content_mask
        + _rotate_left(prefix_c * regular_source, regular_shift)
        + _rotate_left(prefix_c * overflow_source, donor_shift)
    )
    exp_once = np.exp(combined_scores) * active
    content_exp = exp_once * content_mask
    main_exp = exp_once * (content_mask | regular_mask)
    spill_exp = _rotate_left(exp_once * donor_mask, recovery_shift)
    prefix_exp = main_exp * regular_mask + spill_exp

    expected_content = np.exp(content_scores) * np.tri(seq_len, dtype=bool)[None, :, :]
    np.testing.assert_allclose(
        layout.unpack_score_delta(content_exp), expected_content, rtol=2e-7, atol=2e-7
    )
    # Each page squares independently; one shared Delta reduction yields the
    # same scalar for every key page of this query block.
    row_sums = (main_exp**2 + spill_exp**2).reshape(
        layout.query_pair_count, seq_len, seq_len, 2
    ).sum(axis=1)
    expected_z = np.sum(np.exp(prefix_scores) ** 2, axis=-1) + np.sum(
        expected_content**2, axis=-1
    )
    content_probability = layout.unpack_score_delta(
        content_exp**2 / np.repeat(row_sums[:, None, :, :], seq_len, axis=1).reshape(
            layout.query_pair_count, layout.slots
        )
    )
    np.testing.assert_allclose(
        content_probability,
        expected_content**2 / expected_z[:, :, None],
        rtol=2e-7,
        atol=2e-7,
    )
    for pair_index, heads in enumerate(layout.query_head_pairs()):
        for lane, head in enumerate(heads):
            np.testing.assert_allclose(row_sums[pair_index, :, lane], expected_z[head], rtol=2e-6)
            for prefix_index in range(2):
                slots = [
                    2 * ((1 + prefix_index) * seq_len + query) + lane
                    for query in range(seq_len)
                ]
                np.testing.assert_allclose(
                    prefix_exp[pair_index, slots],
                    np.exp(prefix_scores[head, :, prefix_index]),
                    rtol=2e-7,
                    atol=2e-7,
                )
                np.testing.assert_allclose(
                    prefix_exp[pair_index, slots] ** 2 / row_sums[pair_index, :, lane],
                    np.exp(prefix_scores[head, :, prefix_index]) ** 2 / expected_z[head],
                    rtol=2e-7,
                    atol=2e-7,
                )


def test_130_encrypted_content_tokens_use_a_two_token_fm_tail() -> None:
    layout = FeatureMajorPrefillLayout(seq_len=130)
    assert layout.cipher_count == 17
    assert [layout.active_tokens(i) for i in range(17)] == [8] * 16 + [2]
    rows = np.arange(130 * 4096, dtype=np.float32).reshape(130, 4096)
    packed = layout.pack(rows)
    np.testing.assert_array_equal(layout.unpack(packed), rows)
    assert packed[16, layout.slot(0, 0)] == rows[128, 0]
    assert packed[16, layout.slot(0, 1)] == rows[129, 0]
    assert np.all(packed[16].reshape(4096, 8)[:, 2:] == 0)
