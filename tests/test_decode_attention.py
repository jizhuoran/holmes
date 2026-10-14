from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llama3fhe.approx.attention import attention_gqa_numpy
from llama3fhe.approx.softmax import SoftmaxPolyConfig, load_softmax_poly_coeffs
from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import decode_feature_major_layout
from llama3fhe.approx.prefix import PrefixKVBundle
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.operators.attention.decode.decode import (
    apply_decode_softmax_public_map_fhe,
    decode_attention_alg2_numpy,
    decode_attention_alg2_pages_numpy,
    decode_attention_fhe,
    decode_attention_pages_fhe,
    decode_attention_numpy,
    decode_attention_rotations,
    decode_broadcast_token0_numpy,
    decode_broadcast_tokens_fhe,
    decode_output_c_rotations,
    decode_reduce_tokens_fhe,
    output_c_to_decode_fm_fhe,
    output_c_to_decode_fm_numpy,
    correct_prefill_emitted_key_c_numpy,
    prefill_kv_cache_numpy,
    pv_c_fhe,
    pv_c_numpy,
    qk_c_fhe,
    qk_prefix_pc_compact_fhe,
    pv_prefix_pc_compact_fhe,
    qk_c_numpy,
    softmax_alg2_row_numpy,
    softmax_row_numpy,
)
from llama3fhe.operators.attention.decode.decode_softmax import (
    ALG2_ROUND_NEED_LIMBS,
    ALG2_ROUND_Y_LIMBS,
    BOOTSTRAP_MIN_INPUT_LIMBS,
    DecodeAlg2Poly,
    apply_decode_alg2_public_map_fhe,
    apply_decode_alg2_public_map_numpy,
    softmax_alg2_poly_numpy,
    softmax_alg2_row_fhe,
    softmax_alg2_row_pages_fhe,
)
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.prefill.output import (
    decode_w_o_fhe,
    decode_w_o_numpy,
    decode_w_o_rotations,
)


def _small_layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=32,
    )


def test_decode_attention_io_contracts() -> None:
    layout = _small_layout()
    assert layout.c_slot(0, 3, 1) == 2 * 3 + 1
    query = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    packed = layout.pack_compact_query_c(query)
    np.testing.assert_allclose(layout.unpack_compact_query_c(packed), query)
    packed_broadcast = layout.pack_query_broadcast_c(query)
    unpacked = layout.unpack_query_c(packed_broadcast).reshape(4, 4, 4)
    for token in range(4):
        np.testing.assert_allclose(unpacked[token], query)
    scores = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    packed_s = layout.pack_compact_decode_scores(scores)
    np.testing.assert_allclose(layout.unpack_compact_decode_scores(packed_s), scores)
    packed_legacy = layout.pack_decode_scores(scores)
    np.testing.assert_allclose(layout.unpack_decode_scores(packed_legacy), scores)
    mask = layout.compact_decode_prefix_mask(2)
    unpacked_mask = layout.unpack_compact_decode_scores(mask)
    np.testing.assert_allclose(unpacked_mask[:, :2], 1.0)
    np.testing.assert_allclose(unpacked_mask[:, 2:], 0.0)
    token0 = layout.unpack_compact_decode_scores(layout.compact_decode_token0_mask())
    np.testing.assert_allclose(token0[:, :1], 1.0)
    np.testing.assert_allclose(token0[:, 1:], 0.0)
    assert layout.decode_softmax_token_rotations() == (2, 4)
    assert layout.decode_softmax_token_broadcast_rotations() == (-2, -4)
    assert layout.decode_qk_feature_rotations() == (8, 16)
    assert layout.decode_feature_broadcast_rotations() == (-8, -16)
    assert layout.compact_q_column_rotations() == (2,)
    assert layout.compact_score_row_rotations() == (-8,)
    output = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    packed_o = layout.pack_decode_output(output)
    unpacked_o = layout.unpack_query_c(packed_o).reshape(4, 4, 4)
    np.testing.assert_allclose(unpacked_o[0], output)
    np.testing.assert_allclose(unpacked_o[1:], 0.0)
    folded = layout.pack_key_c_feature_folded(np.ones((4, 8), dtype=np.float32))
    assert folded.shape == (layout.key_value_pair_count, layout.slots)
    assert np.iscomplexobj(folded)
    recovered = (folded + np.conjugate(folded)).real
    np.testing.assert_allclose(
        recovered, layout.pack_key_c(np.ones((4, 8), dtype=np.float32)), atol=1e-6
    )


def test_prefill_emitted_key_c_correction_undoes_expand_unfold() -> None:
    rng = np.random.default_rng(26091801)
    layout = _small_layout()
    seq_len = int(layout.seq_len)
    heads = int(layout.key_value_heads)
    dim = int(layout.head_dim)
    key = rng.normal(size=(seq_len, heads, dim)).astype(np.float32)
    lo = key[:, :, : dim // 2]
    hi = key[:, :, dim // 2 :]
    broken = np.concatenate([2.0 * lo, -2.0 * hi], axis=-1)
    packed_broken = layout.pack_key_c(broken.reshape(seq_len, heads * dim))
    repaired = correct_prefill_emitted_key_c_numpy(packed_broken, layout=layout)
    np.testing.assert_allclose(
        repaired, layout.pack_key_c(key), atol=1e-6, rtol=1e-6
    )


def test_prefill_numpy_kv_cache_matches_pack_key_c() -> None:
    rng = np.random.default_rng(26091701)
    layout = _small_layout()
    key = rng.normal(size=(4, 2, 4)).astype(np.float32)
    value = rng.normal(size=(4, 2, 4)).astype(np.float32)
    cache = prefill_kv_cache_numpy(key, value, layout=layout)
    assert cache.occupied == 4
    np.testing.assert_allclose(cache.pages[0].logical_keys(), key.reshape(4, 8))
    np.testing.assert_allclose(
        cache.pages[0].packed_keys(), layout.pack_key_c(key), atol=1e-6
    )


def test_one_query_c_matmul_and_softmax_match_gqa() -> None:
    rng = np.random.default_rng(26091702)
    layout = _small_layout()
    query = rng.normal(size=(4, 4, 4)).astype(np.float32)
    key = rng.normal(size=(4, 2, 4)).astype(np.float32)
    value = rng.normal(size=(4, 2, 4)).astype(np.float32)
    packed_k = layout.pack_key_c(key)
    packed_v = layout.pack_value_c(value)

    scores = qk_c_numpy(query[-1:], packed_k, layout=layout)
    _, _, gqa_out = attention_gqa_numpy(
        query.reshape(4, 16),
        key.reshape(4, 8),
        value.reshape(4, 8),
        layout=layout,
    )
    gqa_scores = np.einsum(
        "hd,khd->hk",
        query[-1],
        key[:, np.arange(4) // 2, :],
    )
    np.testing.assert_allclose(scores, gqa_scores, atol=1e-5, rtol=1e-5)

    padded = np.zeros((4, 4), dtype=np.float32)
    padded[:, :4] = scores
    probs = softmax_row_numpy(padded, layout=layout, occupied=4)
    _, probs_full, out = decode_attention_numpy(
        query[-1:], packed_k, packed_v, layout=layout, occupied=4
    )
    np.testing.assert_allclose(probs, probs_full, atol=1e-6)
    np.testing.assert_allclose(out[0].reshape(-1), gqa_out[-1], atol=1e-5, rtol=1e-5)

    partial = softmax_row_numpy(padded, layout=layout, occupied=2)
    alg2 = softmax_alg2_row_numpy(padded, layout=layout, occupied=2, k=2)
    np.testing.assert_allclose(alg2, partial, atol=1e-5, rtol=1e-5)
    assert float(np.sum(partial[:, 2:])) == 0.0

    _, alg2_probs, alg2_out = decode_attention_alg2_numpy(
        query[-1:], packed_k, packed_v, layout=layout, occupied=2, k=2
    )
    np.testing.assert_allclose(alg2_probs, partial, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        alg2_out,
        pv_c_numpy(partial, packed_v, layout=layout, occupied=2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_decode_alg2_pages_numpy_matches_single_page_and_uses_prefix() -> None:
    rng = np.random.default_rng(26091722)
    layout = _small_layout()
    query = rng.normal(size=(1, 4, 4)).astype(np.float32)
    key = rng.normal(size=(4, 2, 4)).astype(np.float32)
    value = rng.normal(size=(4, 2, 4)).astype(np.float32)
    packed_k = layout.pack_key_c(key)
    packed_v = layout.pack_value_c(value)
    softmax = DecodeAlg2Poly.synthetic(k=1, degree=8)
    _, _, single = decode_attention_alg2_numpy(
        query,
        packed_k,
        packed_v,
        layout=layout,
        occupied=4,
        config=softmax,
    )
    _, _, paged = decode_attention_alg2_pages_numpy(
        query,
        [packed_k],
        [packed_v],
        layout=layout,
        occupied_pages=[4],
        config=softmax,
    )
    np.testing.assert_allclose(paged, single, atol=1e-5, rtol=1e-5)

    extra_key = rng.normal(size=(2, 2, 4)).astype(np.float32)
    extra_value = rng.normal(size=(2, 2, 4)).astype(np.float32)
    page1_k = layout.pack_token_c(extra_key[0], token_index=0, kv=True, name="key")
    page1_k = page1_k + layout.pack_token_c(
        extra_key[1], token_index=1, kv=True, name="key"
    )
    page1_v = layout.pack_token_c(extra_value[0], token_index=0, kv=True, name="value")
    page1_v = page1_v + layout.pack_token_c(
        extra_value[1], token_index=1, kv=True, name="value"
    )
    _, _, with_prefix = decode_attention_alg2_pages_numpy(
        query,
        [packed_k, page1_k],
        [packed_v, page1_v],
        layout=layout,
        occupied_pages=[4, 2],
        config=softmax,
    )
    _, _, suffix_only = decode_attention_alg2_pages_numpy(
        query,
        [page1_k],
        [page1_v],
        layout=layout,
        occupied_pages=[2],
        config=softmax,
    )
    assert not np.allclose(with_prefix, suffix_only, atol=1e-5)


def test_decode_token0_broadcast_and_alg2_public_map_numpy() -> None:
    rng = np.random.default_rng(26091708)
    layout = _small_layout()
    values = rng.normal(size=(4, 4)).astype(np.float32)
    packed = layout.pack_compact_decode_scores(values)
    broadcast = decode_broadcast_token0_numpy(packed, layout=layout, occupied=2)
    actual = layout.unpack_compact_decode_scores(broadcast)
    np.testing.assert_allclose(actual[:, :2], np.repeat(values[:, :1], 2, axis=1))
    np.testing.assert_allclose(actual[:, 2:], 0.0)

    config = DecodeAlg2Poly.synthetic(k=1)
    mapped = apply_decode_alg2_public_map_numpy(
        values, layout=layout, config=config, layer_idx=0, occupied=2
    )
    scale = 1.0 / np.sqrt(float(layout.head_dim))
    shift = -float(config.U_hat) / 2.0
    invalid = float(config.stats_min) / 2.0 + shift
    expected = np.full(values.shape, invalid, dtype=np.float64)
    expected[:, :2] = np.asarray(values[:, :2], dtype=np.float64) * (scale / 2.0) + shift
    np.testing.assert_allclose(mapped, expected, atol=1e-6, rtol=1e-6)

    probabilities, report = softmax_alg2_poly_numpy(
        values, layout=layout, config=config, layer_idx=0, occupied=2
    )
    assert probabilities.shape == (4, 4)
    np.testing.assert_allclose(probabilities[:, 2:], 0.0, atol=1e-6)
    assert report["exp_in_range"]
    assert all(round_report["in_range"] for round_report in report["invsqrt_rounds"])


def test_decode_alg2_poly_numpy_probe_vs_exact_exp() -> None:
    rng = np.random.default_rng(26091709)
    layout = AttentionPairLayout()
    artifact = load_softmax_poly_coeffs(
        Path(__file__).resolve().parents[1]
        / "assets"
        / "polynomials"
        / "softmax_poly_coeffs.json"
    )
    config = SoftmaxPolyConfig.from_artifact(artifact)
    scores = rng.normal(scale=1.0, size=(layout.query_heads, layout.seq_len)).astype(
        np.float32
    )
    reports = []
    for occupied in (1, 2, 8, int(layout.seq_len)):
        _, report = softmax_alg2_poly_numpy(
            scores,
            layout=layout,
            config=config,
            layer_idx=0,
            occupied=occupied,
        )
        reports.append(report)
        assert report["occupied"] == occupied
        assert "exp_interval" in report
        assert "invsqrt_rounds" in report
        if report["exp_in_range"] and all(
            round_report["in_range"] for round_report in report["invsqrt_rounds"]
        ):
            assert report["vs_exact_max_abs"] < 5e-2
    assert [report["occupied"] for report in reports] == [1, 2, 8, 128]


def test_emit_cache_c_is_opt_in_on_attention_config() -> None:
    assert AttentionOperatorConfig().emit_cache_c is False
    assert AttentionOperatorConfig(emit_cache_c=True).emit_cache_c is True


@pytest.mark.gpu
def test_qk_c_fhe_matches_numpy_on_small_c_page() -> None:
    rng = np.random.default_rng(26091703)
    layout = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=8192,
    )
    query = rng.normal(scale=0.05, size=(1, 4, 4)).astype(np.float32)
    key = rng.normal(scale=0.05, size=(4, 2, 4)).astype(np.float32)
    q_packed = layout.pack_compact_query_c(query)
    k_packed = layout.pack_key_c(key)
    expected = qk_c_numpy(query, k_packed, layout=layout)
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 6
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=decode_attention_rotations(layout),
        include_default_rotations=True,
    )
    q_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in q_packed
    )
    k_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in k_packed
    )
    scores = None
    try:
        scores = qk_c_fhe(
            q_ciphers, k_ciphers, layout=layout, crypto_context=context
        )
        decrypted = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in scores
            ]
        )
        actual = layout.unpack_compact_decode_scores(decrypted)
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
    finally:
        for cipher in q_ciphers + k_ciphers + tuple(scores or ()):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_decode_prefix_pc_compact_matches_numpy() -> None:
    rng = np.random.default_rng(26091731)
    layout = _gpu_decode_layout()
    prefix_k = 2
    query = rng.normal(scale=0.05, size=(1, 4, 4)).astype(np.float32)
    key = rng.normal(scale=0.05, size=(prefix_k, 2, 4)).astype(np.float32)
    value = rng.normal(scale=0.05, size=(prefix_k, 2, 4)).astype(np.float32)
    q_packed = layout.pack_compact_query_c(query)
    expected_scores = np.einsum(
        "hd,khd->hk", query[0], key[:, np.arange(4) // 2, :]
    )
    probabilities = rng.uniform(0.1, 0.9, size=(4, prefix_k)).astype(np.float32)
    logical_probabilities = np.pad(
        probabilities, ((0, 0), (0, int(layout.seq_len) - prefix_k))
    )
    p_packed = layout.pack_compact_decode_scores(logical_probabilities)
    logical_probabilities = layout.unpack_compact_decode_scores(p_packed)[:, :prefix_k]
    value_for_q = value[:, np.arange(4) // 2, :]
    expected_output = np.einsum(
        "hk,khd->hd", logical_probabilities, value_for_q
    ).reshape(-1)
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 12
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=decode_attention_rotations(layout),
        include_default_rotations=True,
    )
    q_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in q_packed
    )
    p_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in p_packed
    )
    scores = outputs = None
    try:
        scores = qk_prefix_pc_compact_fhe(
            q_ciphers,
            key,
            layout=layout,
            crypto_context=context,
            prefix_k=prefix_k,
        )
        actual_scores = layout.unpack_compact_decode_scores(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in scores]
            )
        )
        np.testing.assert_allclose(
            actual_scores[:, :prefix_k], expected_scores, atol=8e-3, rtol=8e-3
        )
        outputs = pv_prefix_pc_compact_fhe(
            p_ciphers,
            value,
            layout=layout,
            crypto_context=context,
            prefix_k=prefix_k,
        )
        actual_output = layout.unpack_output_c(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in outputs]
            )
        )[0]
        np.testing.assert_allclose(actual_output, expected_output, atol=8e-3, rtol=8e-3)
    finally:
        for cipher in q_ciphers + p_ciphers + tuple(scores or ()) + tuple(outputs or ()):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_decode_pages_combines_plain_prefix_and_cipher_content() -> None:
    rng = np.random.default_rng(26091732)
    layout = _gpu_decode_layout()
    prefix_k = 2
    query = rng.normal(scale=0.05, size=(1, 4, 4)).astype(np.float32)
    key_prefix = rng.normal(scale=0.05, size=(prefix_k, 2, 4)).astype(np.float32)
    value_prefix = rng.normal(scale=0.05, size=(prefix_k, 2, 4)).astype(np.float32)
    key_content = rng.normal(scale=0.05, size=(4, 2, 4)).astype(np.float32)
    value_content = rng.normal(scale=0.05, size=(4, 2, 4)).astype(np.float32)
    bundle = PrefixKVBundle.from_logical(
        key_prefix.reshape(prefix_k, -1),
        value_prefix.reshape(prefix_k, -1),
        layout=layout,
    )
    q_packed = layout.pack_compact_query_c(query)
    key_packed = layout.pack_key_c(key_content)
    value_packed = layout.pack_value_c(value_content)
    query_heads = query[0]
    key_for_q = key_content[:, np.arange(4) // 2, :]
    key_prefix_for_q = key_prefix[:, np.arange(4) // 2, :]
    value_for_q = value_content[:, np.arange(4) // 2, :]
    value_prefix_for_q = value_prefix[:, np.arange(4) // 2, :]
    scores = np.concatenate(
        [
            np.einsum("hd,khd->hk", query_heads, key_prefix_for_q),
            np.einsum("hd,khd->hk", query_heads, key_for_q),
        ],
        axis=-1,
    )
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    expected = np.einsum(
        "hk,khd->hd", probs, np.concatenate([value_prefix_for_q, value_for_q], axis=0)
    ).reshape(-1)
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 30
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=decode_attention_rotations(layout),
        include_default_rotations=True,
    )
    q_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in q_packed
    )
    key_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in key_packed
    )
    value_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in value_packed
    )
    result = None
    try:
        result = decode_attention_pages_fhe(
            q_ciphers,
            [key_ciphers],
            [value_ciphers],
            layout=layout,
            crypto_context=context,
            softmax_config=DecodeAlg2Poly.synthetic(k=1, degree=8),
            occupied_pages=[4],
            prefix_bundle=bundle,
        )
        actual = layout.unpack_output_c(
            np.stack(
                [
                    np.asarray(context.decrypt(ct), dtype=np.float64).real
                    for ct in result.output_ciphers
                ]
            )
        )[0]
        np.testing.assert_allclose(actual, expected, atol=8e-2, rtol=8e-2)
    finally:
        if result is not None:
            result.release()
        for cipher in q_ciphers + key_ciphers + value_ciphers:
            release_if_supported(cipher)


def _gpu_decode_layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=8192,
    )


def _gpu_decode_context(layout: AttentionPairLayout, *, depth: int = 6):
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


def _decrypt_decode_scores(ciphers, *, layout, crypto_context) -> np.ndarray:
    decrypted = np.stack(
        [
            np.asarray(crypto_context.decrypt(cipher), dtype=np.float64).real
            for cipher in ciphers
        ]
    )
    return layout.unpack_compact_decode_scores(decrypted)


@pytest.mark.gpu
def test_decode_softmax_mask_and_token_fold_on_c_scores() -> None:
    rng = np.random.default_rng(26091704)
    layout = _gpu_decode_layout()
    scores = rng.normal(scale=0.2, size=(4, 4)).astype(np.float32)
    packed = layout.pack_compact_decode_scores(scores)
    context = _gpu_decode_context(layout)
    ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed
    )
    mapped = reduced = broadcast = None
    try:
        mapped = apply_decode_softmax_public_map_fhe(
            ciphers,
            layout=layout,
            crypto_context=context,
            occupied=2,
            scale=False,
        )
        masked = _decrypt_decode_scores(mapped, layout=layout, crypto_context=context)
        expected = np.array(scores, dtype=np.float64, copy=True)
        expected[:, 2:] = 0.0
        np.testing.assert_allclose(masked, expected, atol=5e-3, rtol=5e-3)

        reduced = decode_reduce_tokens_fhe(
            mapped, layout=layout, crypto_context=context
        )
        sums = _decrypt_decode_scores(reduced, layout=layout, crypto_context=context)
        expected_sum = np.sum(expected, axis=-1)
        np.testing.assert_allclose(sums[:, 0], expected_sum, atol=5e-3, rtol=5e-3)

        broadcast = decode_broadcast_tokens_fhe(
            reduced, layout=layout, crypto_context=context, occupied=2
        )
        copied = _decrypt_decode_scores(
            broadcast, layout=layout, crypto_context=context
        )
        np.testing.assert_allclose(
            copied[:, :2], np.repeat(expected_sum[:, None], 2, axis=1), atol=5e-3, rtol=5e-3
        )
        np.testing.assert_allclose(copied[:, 2:], 0.0, atol=5e-3, rtol=5e-3)
    finally:
        for cipher in (
            ciphers
            + tuple(mapped or ())
            + tuple(reduced or ())
            + tuple(broadcast or ())
        ):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_decode_alg2_public_map_and_row_fhe_match_numpy() -> None:
    rng = np.random.default_rng(26091710)
    layout = _gpu_decode_layout()
    scores = rng.normal(scale=0.2, size=(4, 4)).astype(np.float32)
    packed = layout.pack_compact_decode_scores(scores)
    config = DecodeAlg2Poly.synthetic(k=2, degree=8)
    expected_map = apply_decode_alg2_public_map_numpy(
        scores, layout=layout, config=config, layer_idx=0, occupied=2
    )
    expected_k1, _ = softmax_alg2_poly_numpy(
        scores,
        layout=layout,
        config=config,
        layer_idx=0,
        occupied=2,
        rounds=1,
    )
    expected_k2, _ = softmax_alg2_poly_numpy(
        scores,
        layout=layout,
        config=config,
        layer_idx=0,
        occupied=2,
        rounds=2,
    )
    context = _gpu_decode_context(layout, depth=32)
    ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed
    )
    mapped = probs_k1 = probs_k2 = None
    try:
        mapped = apply_decode_alg2_public_map_fhe(
            ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            layer_idx=0,
            occupied=2,
        )
        actual_map = _decrypt_decode_scores(
            mapped, layout=layout, crypto_context=context
        )
        np.testing.assert_allclose(actual_map, expected_map, atol=5e-3, rtol=5e-3)

        probs_k1 = softmax_alg2_row_fhe(
            ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            layer_idx=0,
            occupied=2,
            rounds=1,
        )
        actual_k1 = _decrypt_decode_scores(
            probs_k1, layout=layout, crypto_context=context
        )
        np.testing.assert_allclose(actual_k1, expected_k1, atol=5e-2, rtol=5e-2)
        np.testing.assert_allclose(actual_k1[:, 2:], 0.0, atol=5e-3, rtol=5e-3)

        probs_k2 = softmax_alg2_row_fhe(
            ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            layer_idx=0,
            occupied=2,
            rounds=2,
        )
        actual_k2 = _decrypt_decode_scores(
            probs_k2, layout=layout, crypto_context=context
        )
        np.testing.assert_allclose(actual_k2, expected_k2, atol=5e-2, rtol=5e-2)
        np.testing.assert_allclose(actual_k2[:, 2:], 0.0, atol=5e-3, rtol=5e-3)
    finally:
        for cipher in (
            ciphers
            + tuple(mapped or ())
            + tuple(probs_k1 or ())
            + tuple(probs_k2 or ())
        ):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_decode_alg2_row_fhe_head_rows_zero_other_rows_and_match_numpy() -> None:
    """Multi-card partial score page: only ``head_rows`` carry scores."""

    rng = np.random.default_rng(26093001)
    layout = _gpu_decode_layout()
    head_rows = (0, 2)
    scores = rng.normal(scale=0.2, size=(4, 4)).astype(np.float32)
    scores[[1, 3], :] = 0.0
    packed = layout.pack_compact_decode_scores(scores)
    config = DecodeAlg2Poly.synthetic(k=2, degree=8)
    expected_map = apply_decode_alg2_public_map_numpy(
        scores,
        layout=layout,
        config=config,
        layer_idx=0,
        occupied=3,
        head_rows=head_rows,
    )
    expected, _ = softmax_alg2_poly_numpy(
        scores,
        layout=layout,
        config=config,
        layer_idx=0,
        occupied=3,
        rounds=2,
        head_rows=head_rows,
    )
    full, _ = softmax_alg2_poly_numpy(
        scores, layout=layout, config=config, layer_idx=0, occupied=3, rounds=2
    )
    # the owned rows are the single-card rows; the others are exactly zero
    np.testing.assert_array_equal(expected[[0, 2]], full[[0, 2]])
    np.testing.assert_array_equal(expected[[1, 3]], 0.0)
    context = _gpu_decode_context(layout, depth=32)
    ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed
    )
    mapped = probs = None
    try:
        mapped = apply_decode_alg2_public_map_fhe(
            ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            layer_idx=0,
            occupied=3,
            head_rows=head_rows,
        )
        actual_map = _decrypt_decode_scores(
            mapped, layout=layout, crypto_context=context
        )
        np.testing.assert_allclose(actual_map, expected_map, atol=5e-3, rtol=5e-3)
        probs = softmax_alg2_row_fhe(
            ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            layer_idx=0,
            occupied=3,
            rounds=2,
            head_rows=head_rows,
        )
        actual = _decrypt_decode_scores(probs, layout=layout, crypto_context=context)
        np.testing.assert_allclose(actual[[0, 2]], expected[[0, 2]], atol=5e-2, rtol=5e-2)
        np.testing.assert_allclose(actual[[1, 3]], 0.0, atol=5e-3)
        np.testing.assert_allclose(actual[:, 3:], 0.0, atol=5e-3)
    finally:
        for cipher in ciphers + tuple(mapped or ()) + tuple(probs or ()):
            release_if_supported(cipher)


class _ReencryptBootstrap:
    """Test stand-in for the bootstrap operator: decrypt and re-encrypt.

    Returns the ciphertext at ``limbs`` limbs (production Meta-BTS leaves the
    Softmax ``y`` with ~16), so the Alg2 level accounting of the kernel is
    exercised without a real bootstrapping context.
    """

    def __init__(self, crypto_context, *, limbs: int, slots: int) -> None:
        self.crypto_context = crypto_context
        self.limbs = int(limbs)
        self.slots = int(slots)
        self.calls = 0
        self.input_limbs: list[int] = []
        self.config = type("Config", (), {"iterations": 1, "precision_bits": 0})()

    def refresh(self, cipher, *, iterations=None, precision_bits=None):
        del iterations, precision_bits
        self.calls += 1
        self.input_limbs.append(int(cipher.state.cur_limbs))
        assert int(cipher.state.cur_limbs) >= BOOTSTRAP_MIN_INPUT_LIMBS
        values = np.asarray(self.crypto_context.decrypt(cipher), dtype=np.float64).real
        level = int(self.crypto_context.context.L) - self.limbs
        return self.crypto_context.encrypt(values, level=level, slots=self.slots)


def _run_alg2_k5(context, layout, ciphers, config, bootstrap, *, pages: bool):
    if pages:
        (probs,) = softmax_alg2_row_pages_fhe(
            (ciphers,),
            layout=layout,
            crypto_context=context,
            config=config,
            occupied_pages=(3,),
            layer_idx=0,
            rounds=5,
            bootstrap_operator=bootstrap,
            bootstrap_iterations=1,
            bootstrap_precision_bits=0,
        )
        return probs
    return softmax_alg2_row_fhe(
        ciphers,
        layout=layout,
        crypto_context=context,
        config=config,
        layer_idx=0,
        occupied=3,
        rounds=5,
        bootstrap_operator=bootstrap,
        bootstrap_iterations=1,
        bootstrap_precision_bits=0,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("pages", [False, True])
def test_decode_alg2_k5_refreshes_y_before_a_round_that_would_run_dry(
    pages: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layer-31 schedule (``k=5``): ``y`` is refreshed once mid-chain.

    With a 16-limb bootstrap exit the rounds drain ``y`` to 1 limb before the
    post-Softmax probability refresh (the 8x4090 32-layer run died there with
    ``rescale: cur_limbs must be > 1``).  The kernel now refreshes ``y`` before
    a round once it is down to ``ALG2_ROUND_NEED_LIMBS`` limbs; every round
    then starts above that bound and the output stays bootstrappable.
    """

    import llama3fhe.operators.attention.decode.decode_softmax as decode_softmax

    round_entry_limbs: list[int] = []
    original = decode_softmax._alg2_round_pages_fhe if pages else decode_softmax._alg2_round_fhe

    def traced(y, **kwargs):
        rows = tuple(c for page in y for c in page) if pages else tuple(y)
        round_entry_limbs.append(min(int(c.state.cur_limbs) for c in rows))
        return original(y, **kwargs)

    monkeypatch.setattr(
        decode_softmax, "_alg2_round_pages_fhe" if pages else "_alg2_round_fhe", traced
    )
    rng = np.random.default_rng(26100101)
    layout = _gpu_decode_layout()
    scores = rng.normal(scale=0.2, size=(4, 4)).astype(np.float32)
    packed = layout.pack_compact_decode_scores(scores)
    config = DecodeAlg2Poly.synthetic(k=5, degree=8)
    expected, _ = softmax_alg2_poly_numpy(
        scores, layout=layout, config=config, layer_idx=0, occupied=3, rounds=5
    )
    context = _gpu_decode_context(layout, depth=32)
    bootstrap = _ReencryptBootstrap(context, limbs=16, slots=layout.slots)
    ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed
    )
    probs = None
    try:
        probs = _run_alg2_k5(context, layout, ciphers, config, bootstrap, pages=pages)
        assert len(round_entry_limbs) == 5
        # no round starts at or below the bound ...
        assert min(round_entry_limbs) > ALG2_ROUND_NEED_LIMBS
        # ... and exactly one later round starts on a freshly refreshed y
        assert round_entry_limbs[0] == 16
        assert sum(1 for limbs in round_entry_limbs[1:] if limbs == 16) == 1
        assert min(int(c.state.cur_limbs) for c in probs) >= BOOTSTRAP_MIN_INPUT_LIMBS
        actual = _decrypt_decode_scores(probs, layout=layout, crypto_context=context)
        np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=5e-2, rtol=5e-2)
        np.testing.assert_allclose(actual[:, 3:], 0.0, atol=5e-3)
    finally:
        for cipher in ciphers + tuple(probs or ()):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_decode_alg2_k5_without_mid_refresh_runs_the_chain_dry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression control: disabling the bound reproduces the layer-31 failure."""

    import llama3fhe.operators.attention.decode.decode_softmax as decode_softmax

    monkeypatch.setattr(decode_softmax, "ALG2_ROUND_NEED_LIMBS", 0)
    rng = np.random.default_rng(26100101)
    layout = _gpu_decode_layout()
    scores = rng.normal(scale=0.2, size=(4, 4)).astype(np.float32)
    packed = layout.pack_compact_decode_scores(scores)
    config = DecodeAlg2Poly.synthetic(k=5, degree=8)
    context = _gpu_decode_context(layout, depth=32)
    bootstrap = _ReencryptBootstrap(context, limbs=16, slots=layout.slots)
    ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed
    )
    try:
        # the same error the 8x4090 32-layer run hit in layer 31
        with pytest.raises(ValueError, match="cur_limbs must be > 1"):
            _run_alg2_k5(context, layout, ciphers, config, bootstrap, pages=False)
    finally:
        for cipher in ciphers:
            release_if_supported(cipher)


@pytest.mark.gpu
def test_decode_attention_fhe_qk_alg2_pv_matches_numpy() -> None:
    rng = np.random.default_rng(26091711)
    layout = _gpu_decode_layout()
    query = rng.normal(scale=0.05, size=(1, 4, 4)).astype(np.float32)
    key = np.zeros((4, 2, 4), dtype=np.float32)
    value = np.zeros((4, 2, 4), dtype=np.float32)
    key[:2] = rng.normal(scale=0.05, size=(2, 2, 4)).astype(np.float32)
    value[:2] = rng.normal(scale=0.05, size=(2, 2, 4)).astype(np.float32)
    q_packed = layout.pack_compact_query_c(query)
    k_packed = layout.pack_key_c(key)
    v_packed = layout.pack_value_c(value)
    config = DecodeAlg2Poly.synthetic(k=2, degree=8)
    _, expected_probs, expected = decode_attention_alg2_numpy(
        query,
        k_packed,
        v_packed,
        layout=layout,
        occupied=2,
        config=config,
        layer_idx=0,
    )
    context = _gpu_decode_context(layout, depth=36)
    q_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in q_packed
    )
    k_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in k_packed
    )
    v_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in v_packed
    )
    result = None
    try:
        result = decode_attention_fhe(
            q_ciphers,
            k_ciphers,
            v_ciphers,
            layout=layout,
            crypto_context=context,
            softmax_config=config,
            occupied=2,
        )
        decrypted = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in result.output_ciphers
            ]
        )
        actual = layout.unpack_query_c(decrypted)[0].reshape(4, 4)
        np.testing.assert_allclose(actual, expected[0], atol=5e-2, rtol=5e-2)
        assert set(result.stage_seconds) >= {"qk", "softmax", "pv"}
        np.testing.assert_allclose(expected_probs[:, 2:], 0.0, atol=1e-6)
    finally:
        if result is not None:
            result.release()
        for cipher in q_ciphers + k_ciphers + v_ciphers:
            release_if_supported(cipher)


@pytest.mark.gpu
def test_pv_c_fhe_matches_numpy_on_small_c_page() -> None:
    rng = np.random.default_rng(26091705)
    layout = _gpu_decode_layout()
    scores = rng.normal(scale=0.2, size=(4, 4)).astype(np.float32)
    value = rng.normal(scale=0.05, size=(4, 2, 4)).astype(np.float32)
    probs = softmax_row_numpy(scores, layout=layout, occupied=4, scale=True)
    expected = pv_c_numpy(probs, value, layout=layout, occupied=4)
    p_packed = layout.pack_compact_decode_scores(probs)
    v_packed = layout.pack_value_c(value)
    context = _gpu_decode_context(layout)
    p_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in p_packed
    )
    v_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in v_packed
    )
    outputs = None
    try:
        outputs = pv_c_fhe(
            p_ciphers, v_ciphers, layout=layout, crypto_context=context
        )
        decrypted = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in outputs
            ]
        )
        actual = layout.unpack_query_c(decrypted)[0].reshape(4, 4)
        np.testing.assert_allclose(actual, expected[0], atol=5e-3, rtol=5e-3)
    finally:
        for cipher in p_ciphers + v_ciphers + tuple(outputs or ()):
            release_if_supported(cipher)


def test_output_c_to_decode_fm_skips_pairfm_contract() -> None:
    rng = np.random.default_rng(26091706)
    attention = _small_layout()
    feature = decode_feature_major_layout(hidden_dim=16, slots=32)
    output = rng.normal(size=(1, 4, 4)).astype(np.float32)
    packed_c = attention.pack_decode_output(output)
    packed_fm = output_c_to_decode_fm_numpy(
        packed_c, attention_layout=attention, feature_layout=feature
    )
    assert packed_fm.shape == (1, 32)
    np.testing.assert_allclose(feature.unpack(packed_fm)[0], output.reshape(-1))
    for dim in range(16):
        assert packed_fm[0, feature.slot(dim, 0)] == pytest.approx(
            float(output.reshape(-1)[dim])
        )
        for lane in range(1, feature.tokens_per_cipher):
            assert packed_fm[0, feature.slot(dim, lane)] == 0.0

    weight = rng.normal(size=(16, 16)).astype(np.float32)
    projected = decode_w_o_numpy(
        feature.unpack(packed_fm), weight, feature_layout=feature
    )
    np.testing.assert_allclose(
        projected, output.reshape(1, -1) @ weight.T, atol=1e-5, rtol=1e-5
    )


@pytest.mark.gpu
def test_decode_short_w_o_matches_numpy() -> None:
    rng = np.random.default_rng(26091707)
    attention = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=8192,
    )
    feature = decode_feature_major_layout(hidden_dim=16, slots=8192)
    output = rng.normal(scale=0.05, size=(1, 4, 4)).astype(np.float32)
    weight = rng.normal(scale=0.05, size=(16, 16)).astype(np.float32)
    packed_c = attention.pack_decode_output(output)
    expected_fm = output_c_to_decode_fm_numpy(
        packed_c, attention_layout=attention, feature_layout=feature
    )
    expected = decode_w_o_numpy(
        feature.unpack(expected_fm), weight, feature_layout=feature
    )
    linear_config = LinearOperatorConfig(
        baby_steps=4, baby_anchor_step=4, max_plaintext_rows=16
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 8
    cfg.simulator.device = "cuda"
    rotations = set(decode_output_c_rotations(attention, feature))
    rotations.update(decode_w_o_rotations(feature, operator_config=linear_config))
    context = create_sim_context(
        cfg.simulator,
        input_dim=16,
        block_out=int(feature.tokens_per_cipher),
        rotations=tuple(rotations),
        include_default_rotations=True,
    )
    o_ciphers = tuple(
        context.encrypt(row, level=0, slots=attention.slots) for row in packed_c
    )
    fm_ciphers = projected = None
    try:
        fm_ciphers = output_c_to_decode_fm_fhe(
            o_ciphers,
            attention_layout=attention,
            feature_layout=feature,
            crypto_context=context,
        )
        decrypted_fm = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in fm_ciphers
            ]
        )
        np.testing.assert_allclose(
            feature.unpack(decrypted_fm),
            feature.unpack(expected_fm),
            atol=5e-3,
            rtol=5e-3,
        )
        projected = decode_w_o_fhe(
            fm_ciphers,
            weight,
            feature_layout=feature,
            crypto_context=context,
            operator_config=linear_config,
        )
        decrypted = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in projected
            ]
        )
        np.testing.assert_allclose(
            feature.unpack(decrypted), expected, atol=2e-2, rtol=2e-2
        )
    finally:
        for cipher in o_ciphers + tuple(fm_ciphers or ()) + tuple(projected or ()):
            release_if_supported(cipher)
