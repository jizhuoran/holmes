from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.application.args import DEFAULT_SOFTMAX
from llama3fhe.approx.softmax import SoftmaxPolyConfig, load_softmax_poly_coeffs
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.attention.audit import delta_softmax_rotations
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly
from llama3fhe.operators.attention.prefill.multi_page import (
    delta_page_exp_amplitude_mask,
    delta_full_page_mask,
    delta_reduce_page_squares_fhe,
    prefill_delta_page_masks,
    softmax_alg2_fhe_delta_pages,
    softmax_alg2_pages_numpy,
)
from llama3fhe.operators.attention.prefill.softmax import softmax_alg2_fhe_delta
from llama3fhe.operators.nonlinear.bootstrap import BootstrapConfig, BootstrapOperator


@pytest.mark.parametrize("seq_len", [8, 128])
def test_full_history_mask_uses_cyclic_delta_key_and_query_validity(
    seq_len: int,
) -> None:
    layout = AttentionPairLayout(
        seq_len=seq_len,
        head_dim=seq_len,
        query_heads=4,
        key_value_heads=2,
        slots=2 * seq_len * seq_len,
    )
    query_valid = np.arange(seq_len) < seq_len - 2
    key_valid = np.arange(seq_len) < seq_len - 1
    mask = delta_full_page_mask(
        layout, query_valid_mask=query_valid, key_valid_mask=key_valid
    ).reshape(seq_len, seq_len, 2)
    for delta in range(seq_len):
        for query in range(seq_len):
            key = (query + delta) % seq_len
            assert np.all(mask[delta, query] == (query_valid[query] and key_valid[key]))


@pytest.mark.parametrize("seq_len", [8, 128])
def test_shared_alg2_matches_dense_multi_page_attention(seq_len: int) -> None:
    layout = AttentionPairLayout(
        seq_len=seq_len,
        head_dim=seq_len,
        query_heads=4,
        key_value_heads=2,
        slots=2 * seq_len * seq_len,
    )
    rng = np.random.default_rng(20261007 + seq_len)
    logical_scores = [
        rng.normal(scale=0.8, size=(4, seq_len, seq_len)).astype(np.float32)
        for _ in range(3)
    ]
    current_valid = np.arange(seq_len) < seq_len - 2
    historical_valid = [np.ones(seq_len, dtype=bool), np.arange(seq_len) < seq_len - 1]
    masks = prefill_delta_page_masks(
        layout,
        historical_key_valid_masks=historical_valid,
        current_valid_mask=current_valid,
    )
    packed_scores = [layout.pack_score_delta(scores) for scores in logical_scores]
    packed_probabilities = softmax_alg2_pages_numpy(
        packed_scores,
        masks,
        layout=layout,
        rounds=4,
        score_scale=1.0 / np.sqrt(seq_len),
        u_hat=0.3,
    )
    probabilities = [
        layout.unpack_score_delta(page) for page in packed_probabilities
    ]
    for head in range(4):
        for query in range(seq_len - 2):
            keys = [
                historical_valid[0],
                historical_valid[1],
                np.arange(seq_len) <= query,
            ]
            selected = [
                logical_scores[page][head, query, key_valid]
                for page, key_valid in enumerate(keys)
            ]
            all_scores = np.concatenate(selected).astype(np.float64) / np.sqrt(
                seq_len
            )
            expected = np.exp(all_scores - np.max(all_scores))
            expected /= expected.sum()
            actual = np.concatenate(
                [
                    probabilities[page][head, query, key_valid]
                    for page, key_valid in enumerate(keys)
                ]
            )
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
            assert sum(np.sum(page[head, query]) for page in probabilities) == pytest.approx(1.0)

    # An invalid query has no historical probability and one current-page
    # self entry, keeping the shared denominator defined.
    invalid_query = seq_len - 1
    assert all(np.count_nonzero(page[:, invalid_query]) == 0 for page in probabilities[:-1])
    np.testing.assert_allclose(
        probabilities[-1][:, invalid_query, invalid_query], 1.0, atol=1e-7
    )


def test_alg2_pages_require_one_active_score_per_query_lane() -> None:
    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=128
    )
    score = np.zeros((layout.query_pair_count, layout.slots))
    with pytest.raises(ValueError, match="active score"):
        softmax_alg2_pages_numpy(
            [score], [np.zeros(layout.slots)], layout=layout, rounds=2
        )


def test_page_exp_amplitude_preserves_padding_sentinels() -> None:
    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=128
    )
    query_valid = np.arange(8) < 3
    amplitude = delta_page_exp_amplitude_mask(
        layout, page_count=2, query_valid_mask=query_valid
    ).reshape(8, 8, 2)
    np.testing.assert_allclose(amplitude[:, :3], 1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(amplitude[:, 3:], 1.0)


def test_production_softmax_table_rejects_uncalibrated_second_page() -> None:
    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=128
    )
    config = SoftmaxPolyConfig.from_artifact(
        load_softmax_poly_coeffs(DEFAULT_SOFTMAX)
    )
    assert config.max_prefill_pages == 1
    score = (_FakeCipher(np.zeros(layout.slots)),) * 2
    with pytest.raises(ValueError, match="at most 1 Prefill page"):
        softmax_alg2_fhe_delta_pages(
            (score, score),
            prefill_delta_page_masks(layout, historical_key_valid_masks=(None,)),
            layout=layout,
            layer_idx=0,
            config=config,
            crypto_context=_FakeContext(),
        )


class _FakeCipher:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.float64)
        self.released = False

    def deep_copy(self):
        return _FakeCipher(self.values.copy())

    def release(self):
        self.released = True


class _FakeFHE:
    def __init__(self):
        self.rotations = 0

    def homo_add(self, left, right, _context):
        return _FakeCipher(left.values + right.values)

    def homo_rotate(self, cipher, rotation, _context):
        self.rotations += 1
        return _FakeCipher(np.roll(cipher.values, int(rotation)))


class _FakeContext:
    def __init__(self):
        self.fhe = _FakeFHE()
        self.context = object()


def test_fhe_page_square_sum_uses_one_row_rotation_chain() -> None:
    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=128
    )
    rng = np.random.default_rng(17)
    pages = tuple(
        tuple(_FakeCipher(rng.random(layout.slots)) for _ in range(2))
        for _ in range(3)
    )
    context = _FakeContext()
    reduced = delta_reduce_page_squares_fhe(
        pages, layout=layout, crypto_context=context
    )
    try:
        assert context.fhe.rotations == 2 * 3
        for pair, result in enumerate(reduced):
            page_sum = sum(page[pair].values for page in pages)
            expected = page_sum.copy()
            for shift in (16, 32, 64):
                expected += np.roll(expected, shift)
            np.testing.assert_allclose(result.values, expected)
        assert all(not cipher.released for page in pages for cipher in page)
    finally:
        for cipher in reduced:
            cipher.release()


def test_fhe_page_square_sum_rejects_mismatched_limbs() -> None:
    from types import SimpleNamespace

    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=128
    )
    pages = tuple(
        tuple(_FakeCipher(np.ones(layout.slots)) for _ in range(2))
        for _ in range(2)
    )
    for index, page in enumerate(pages):
        for cipher in page:
            cipher.state = SimpleNamespace(
                cur_limbs=8 - index,
                scale_degree=1,
                scaling_factor=float(2**40),
            )
    with pytest.raises(ValueError, match="matching limb count"):
        delta_reduce_page_squares_fhe(
            pages, layout=layout, crypto_context=_FakeContext()
        )
    assert all(not cipher.released for page in pages for cipher in page)


def test_multi_page_entry_rejects_mismatched_score_limbs() -> None:
    from types import SimpleNamespace

    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=128
    )
    pages = tuple(
        tuple(_FakeCipher(np.zeros(layout.slots)) for _ in range(2))
        for _ in range(2)
    )
    for index, page in enumerate(pages):
        for cipher in page:
            cipher.state = SimpleNamespace(
                cur_limbs=8 - index,
                scale_degree=1,
                scaling_factor=float(2**40),
            )
    with pytest.raises(ValueError, match="matching limb count"):
        softmax_alg2_fhe_delta_pages(
            pages,
            prefill_delta_page_masks(layout, historical_key_valid_masks=(None,)),
            layout=layout,
            layer_idx=0,
            config=DecodeAlg2Poly.synthetic(),
            crypto_context=_FakeContext(),
        )
    assert all(not cipher.released for page in pages for cipher in page)


@pytest.mark.gpu
def test_multi_page_fhe_matches_slot_oracle_and_single_page_entry() -> None:
    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=8192
    )
    rng = np.random.default_rng(20261008)
    logical_scores = [
        rng.normal(scale=0.02, size=(4, 8, 8)).astype(np.float32)
        for _ in range(2)
    ]
    packed_scores = [layout.pack_score_delta(page) for page in logical_scores]
    current_valid = np.arange(8) < 3
    masks = prefill_delta_page_masks(
        layout,
        historical_key_valid_masks=(None,),
        current_valid_mask=current_valid,
    )
    config = DecodeAlg2Poly.synthetic(
        k=1, degree=24, invsqrt_interval=(0.05, 8.0)
    )
    policy = AttentionOperatorConfig()
    simulator = Llama3CKKSConfig().simulator
    simulator.logN = 14
    simulator.maxLevelsRemaining = 32
    simulator.device = "cuda"
    context = create_sim_context(
        simulator,
        input_dim=8,
        block_out=8,
        rotations=delta_softmax_rotations(layout),
        include_default_rotations=False,
    )
    encrypted = tuple(
        tuple(context.encrypt(row, level=0, slots=layout.slots) for row in page)
        for page in packed_scores
    )
    multi = one_new = one_old = None
    try:
        multi = softmax_alg2_fhe_delta_pages(
            encrypted, masks,
            layout=layout,
            layer_idx=0,
            config=config,
            crypto_context=context,
            operator_config=policy,
            current_valid_mask=current_valid,
        )
        expected = softmax_alg2_pages_numpy(
            packed_scores,
            masks,
            layout=layout,
            rounds=1,
            u_hat=1.0,
            query_valid_mask=current_valid,
        )
        for actual_page, expected_page in zip(
            multi.probability_pages, expected, strict=True
        ):
            actual = np.stack([np.asarray(context.decrypt(ct)).real for ct in actual_page])
            np.testing.assert_allclose(actual, expected_page, atol=0.06, rtol=0.1)
        assert len(multi.output_levels) == 1

        one_new = softmax_alg2_fhe_delta_pages(
            (encrypted[-1],), (masks[-1],),
            layout=layout,
            layer_idx=0,
            config=config,
            crypto_context=context,
            operator_config=policy,
            current_valid_mask=current_valid,
        )
        one_old = softmax_alg2_fhe_delta(
            encrypted[-1],
            layout=layout,
            layer_idx=0,
            config=config,
            crypto_context=context,
            operator_config=policy,
            token_valid_mask=current_valid,
        )
        for current, previous in zip(
            one_new.probability_pages[0], one_old.probability_ciphers, strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(context.decrypt(current)).real,
                np.asarray(context.decrypt(previous)).real,
                atol=1e-3,
                rtol=1e-3,
            )
        for encrypted_page, original_page in zip(encrypted, packed_scores, strict=True):
            for cipher, original in zip(encrypted_page, original_page, strict=True):
                np.testing.assert_allclose(
                    np.asarray(context.decrypt(cipher)).real,
                    original,
                    atol=1e-4,
                )
    finally:
        if multi is not None:
            multi.release()
        if one_new is not None:
            one_new.release()
        if one_old is not None:
            one_old.release()
        for page in encrypted:
            for cipher in page:
                release_if_supported(cipher)


@pytest.mark.gpu
def test_multi_page_paired_pre_exp_bootstrap() -> None:
    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=8192
    )
    config = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.05, 8.0)
    )
    simulator = Llama3CKKSConfig().simulator
    simulator.logN = 14
    simulator.maxLevelsRemaining = 40
    simulator.device = "cuda"
    rotations = set(delta_softmax_rotations(layout))
    rotations.update(BootstrapOperator.required_rotations(log_n=14, log_slots=13))
    context = create_sim_context(
        simulator,
        input_dim=8,
        block_out=8,
        rotations=tuple(sorted(rotations)),
        include_default_rotations=False,
    )
    bootstrap = BootstrapOperator.prepare(
        context,
        BootstrapConfig(
            log_slots=13,
            level_budget=(4, 4),
            output_levels=19,
            iterations=1,
            precision_bits=14,
        ),
    )
    logical_scores = [
        np.full((4, 8, 8), 0.02 * value, dtype=np.float32)
        for value in (1, 2)
    ]
    packed = [layout.pack_score_delta(page) for page in logical_scores]
    masks = prefill_delta_page_masks(layout, historical_key_valid_masks=(None,))
    encrypted = tuple(
        tuple(context.encrypt(row, level=0, slots=layout.slots) for row in page)
        for page in packed
    )
    result = None
    try:
        result = softmax_alg2_fhe_delta_pages(
            encrypted,
            masks,
            layout=layout,
            layer_idx=0,
            config=config,
            crypto_context=context,
            bootstrap_operator=bootstrap,
            operator_config=AttentionOperatorConfig(
                refresh_softmax_exp_inputs=True,
                pair_softmax_pre_exp_bootstrap=True,
            ),
        )
        expected = softmax_alg2_pages_numpy(
            packed, masks, layout=layout, rounds=1, u_hat=1.0
        )
        for actual_page, expected_page in zip(
            result.probability_pages, expected, strict=True
        ):
            actual = np.stack([np.asarray(context.decrypt(ct)).real for ct in actual_page])
            np.testing.assert_allclose(actual, expected_page, atol=0.08, rtol=0.1)
        assert bootstrap.calls >= 2
    finally:
        if result is not None:
            result.release()
        for page in encrypted:
            for cipher in page:
                release_if_supported(cipher)
        bootstrap.release()
