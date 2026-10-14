from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.approx.prefix import (
    attention_prefix_gqa_numpy,
    concat_attention_gqa_numpy,
    packed_content_pv_matches_logical,
    packed_content_qk_matches_logical,
    packed_prefix_attention_numpy,
    packed_prefix_pv_matches_logical,
    packed_prefix_qk_matches_logical,
    packed_prefix_softmax_oracle,
    pack_prefix_kv_c,
    pv_prefix_packed_numpy,
    qk_prefix_packed_numpy,
    rope_prefix_and_content,
)
from llama3fhe.layouts.attention import AttentionPairLayout


def _small_layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=32,
    )


def _draw_tensors(layout: AttentionPairLayout, *, prefix_k: int, rng: np.random.Generator):
    q_len = int(layout.seq_len)
    hq = int(layout.query_heads)
    hkv = int(layout.key_value_heads)
    dim = int(layout.head_dim)
    query = rng.standard_normal((q_len, hq * dim)).astype(np.float32)
    key_p = rng.standard_normal((prefix_k, hkv * dim)).astype(np.float32)
    key_c = rng.standard_normal((q_len, hkv * dim)).astype(np.float32)
    value_p = rng.standard_normal((prefix_k, hkv * dim)).astype(np.float32)
    value_c = rng.standard_normal((q_len, hkv * dim)).astype(np.float32)
    return query, key_p, key_c, value_p, value_c


@pytest.mark.parametrize("prefix_k", [1, 2])
def test_split_softmax_matches_concat_small(prefix_k: int) -> None:
    layout = _small_layout()
    rng = np.random.default_rng(0)
    query, key_p, key_c, value_p, value_c = _draw_tensors(
        layout, prefix_k=prefix_k, rng=rng
    )
    split = attention_prefix_gqa_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    scores, probs, output = concat_attention_gqa_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    np.testing.assert_allclose(split.scores_concat, scores, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(split.probabilities_concat, probs, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.concatenate([split.probabilities_p, split.probabilities_c], axis=-1),
        probs,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(split.output, output, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        split.output_p + split.output_c, output, atol=1e-5, rtol=1e-5
    )
    assert split.output.shape == (layout.seq_len, layout.query_heads * layout.head_dim)
    row = probs.sum(axis=-1)
    np.testing.assert_allclose(row, np.ones_like(row), atol=1e-5)


def test_prefix_columns_absorb_mass() -> None:
    layout = _small_layout()
    prefix_k = 2
    q_len = int(layout.seq_len)
    hq = int(layout.query_heads)
    hkv = int(layout.key_value_heads)
    dim = int(layout.head_dim)
    query = np.ones((q_len, hq * dim), dtype=np.float32)
    key_p = np.full((prefix_k, hkv * dim), 8.0, dtype=np.float32)
    key_c = np.zeros((q_len, hkv * dim), dtype=np.float32)
    value_p = np.ones((prefix_k, hkv * dim), dtype=np.float32)
    value_c = np.zeros((q_len, hkv * dim), dtype=np.float32)
    split = attention_prefix_gqa_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    prefix_mass = split.probabilities_p.sum(axis=-1)
    assert float(np.min(prefix_mass)) > 0.99
    np.testing.assert_allclose(split.output_c, 0.0, atol=1e-6)


def test_prefix_score_packing_roundtrip() -> None:
    layout = _small_layout()
    prefix_k = 2
    rng = np.random.default_rng(1)
    scores = rng.standard_normal(
        (layout.query_heads, layout.seq_len, prefix_k)
    ).astype(np.float32)
    packed = layout.pack_prefix_scores_prefill(scores)
    got = layout.unpack_prefix_scores_prefill(packed, prefix_k=prefix_k)
    np.testing.assert_allclose(got, scores, atol=0, rtol=0)
    assert packed.shape == (layout.query_pair_count, layout.slots)


@pytest.mark.parametrize("prefix_k", [1, 2])
def test_prefix_compact_score_packing_roundtrip(prefix_k: int) -> None:
    layout = _small_layout()
    rng = np.random.default_rng(101 + prefix_k)
    scores = rng.standard_normal(
        (layout.query_heads, layout.seq_len, prefix_k)
    ).astype(np.float32)
    packed = layout.pack_prefix_scores_compact(scores)
    got = layout.unpack_prefix_scores_compact(packed, prefix_k=prefix_k)
    np.testing.assert_allclose(got, scores, atol=0, rtol=0)
    assert packed.shape == (1, layout.slots)
    assert layout.prefix_compact_row_count(prefix_k) == (
        layout.query_pair_count * prefix_k
    )


def test_packed_softmax_oracle_small() -> None:
    layout = _small_layout()
    rng = np.random.default_rng(2)
    query, key_p, key_c, value_p, value_c = _draw_tensors(
        layout, prefix_k=2, rng=rng
    )
    split = attention_prefix_gqa_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    packed_p, packed_c, err = packed_prefix_softmax_oracle(
        split.scores_p, split.scores_c, layout=layout, prefix_k=2
    )
    assert max(err.values()) < 1e-6
    np.testing.assert_allclose(packed_p, split.probabilities_p, atol=1e-6)
    np.testing.assert_allclose(packed_c, split.probabilities_c, atol=1e-6)
    _, qk_err = packed_content_qk_matches_logical(query, key_c, layout=layout)
    assert qk_err < 1e-5
    _, pv_err = packed_content_pv_matches_logical(
        split.probabilities_c, value_c, layout=layout
    )
    assert pv_err < 1e-5
    cache = pack_prefix_kv_c(key_p, value_p, layout=layout)
    assert cache.occupied == 2
    np.testing.assert_allclose(cache.pages[0].logical_keys(), key_p, atol=1e-6)


def test_split_softmax_matches_concat_production_geometry() -> None:
    layout = AttentionPairLayout()
    rng = np.random.default_rng(3)
    query, key_p, key_c, value_p, value_c = _draw_tensors(
        layout, prefix_k=2, rng=rng
    )
    query, key_p, key_c = rope_prefix_and_content(
        query,
        key_p,
        key_c,
        layout=layout,
        prefix_k=2,
        theta=1_000_000.0,
    )
    split = attention_prefix_gqa_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    _, _, output = concat_attention_gqa_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    np.testing.assert_allclose(split.output, output, atol=2e-5, rtol=2e-5)
    _, _, err = packed_prefix_softmax_oracle(
        split.scores_p, split.scores_c, layout=layout, prefix_k=2
    )
    assert max(err.values()) < 1e-6
    _, qk_err = packed_content_qk_matches_logical(query, key_c, layout=layout)
    assert qk_err < 2e-5


def _unpack_prefix_decrypted(packed, layout: AttentionPairLayout, prefix_k: int):
    packed = np.asarray(packed, dtype=np.float64).real
    base = packed[:, : layout.payload_slots]
    tiled = np.tile(base, (1, layout.payload_repetitions))
    return layout.unpack_prefix_scores_prefill(tiled, prefix_k=prefix_k)


def _unpack_output_decrypted(packed, layout: AttentionPairLayout):
    packed = np.asarray(packed, dtype=np.float64).real
    base = packed[:, : layout.payload_slots]
    tiled = np.tile(base, (1, layout.payload_repetitions))
    return layout.unpack_output_c(tiled)


def test_dual_softmax_alg2_matches_concat_small() -> None:
    from llama3fhe.approx.prefix import dual_softmax_alg2_numpy, qk_content_numpy, qk_prefix_numpy
    from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly

    layout = _small_layout()
    rng = np.random.default_rng(4)
    query = rng.normal(scale=0.05, size=(4, 16)).astype(np.float32)
    key_p = rng.normal(scale=0.05, size=(2, 8)).astype(np.float32)
    key_c = rng.normal(scale=0.05, size=(4, 8)).astype(np.float32)
    scores_p = qk_prefix_numpy(query, key_p, layout=layout, scale=False)
    scores_c = qk_content_numpy(query, key_c, layout=layout, scale=False)
    config = DecodeAlg2Poly.synthetic(k=1, degree=8)
    p_p, p_c, report = dual_softmax_alg2_numpy(
        scores_p, scores_c, layout=layout, config=config, prefix_k=2
    )
    concat = np.concatenate([p_p, p_c], axis=-1)
    row = concat.sum(axis=-1)
    np.testing.assert_allclose(row, np.ones_like(row), atol=2e-1)
    for query in range(4):
        if query < 3:
            np.testing.assert_allclose(p_c[:, query, query + 1 :], 0.0, atol=1e-6)
    assert p_p.shape[-1] == 2
    assert p_c.shape[-1] == 4
    assert report["k"] == 1
    assert report["exp_in_range"]
    assert report["invsqrt_in_range"]


def test_c_score_masks_small() -> None:
    layout = _small_layout()
    prefix = layout.c_prefix_score_mask(2)
    causal = layout.c_content_score_causal_mask()
    unpacked_p = layout.unpack_prefix_scores_prefill(prefix, prefix_k=2)
    unpacked_c = layout.unpack_prefix_scores_prefill(causal, prefix_k=4)
    np.testing.assert_allclose(unpacked_p, 1.0)
    for query in range(4):
        np.testing.assert_allclose(unpacked_c[:, query, : query + 1], 1.0)
        if query < 3:
            np.testing.assert_allclose(unpacked_c[:, query, query + 1 :], 0.0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("alg2_k", "prefix_k"), [(1, 2), (2, 2), (2, 3)]
)
def test_prefix_alg2_fhe_matches_numpy_small(alg2_k: int, prefix_k: int) -> None:
    from llama3fhe.approx.prefix import dual_softmax_alg2_numpy, qk_content_numpy, qk_prefix_numpy
    from llama3fhe.backend import create_sim_context, release_if_supported
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly
    from llama3fhe.operators.attention.prefix.prefix_softmax import (
        prefix_softmax_rotations,
        softmax_alg2_prefix_partitions_fhe,
    )

    layout = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=8192,
    )
    rng = np.random.default_rng(26091801)
    query = rng.normal(scale=0.05, size=(4, 16)).astype(np.float32)
    key_p = rng.normal(scale=0.05, size=(prefix_k, 8)).astype(np.float32)
    key_c = rng.normal(scale=0.05, size=(4, 8)).astype(np.float32)
    scores_p = qk_prefix_numpy(query, key_p, layout=layout, scale=False)
    scores_c = qk_content_numpy(query, key_c, layout=layout, scale=False)
    config = DecodeAlg2Poly.synthetic(k=alg2_k, degree=8)
    expected_p, expected_c, _ = dual_softmax_alg2_numpy(
        scores_p, scores_c, layout=layout, config=config, prefix_k=prefix_k
    )
    packed_p = layout.pack_prefix_scores_prefill(scores_p)
    packed_c = layout.pack_prefix_scores_prefill(scores_c)
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 32
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=prefix_softmax_rotations(layout),
        include_default_rotations=True,
    )
    p_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_p
    )
    c_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_c
    )
    out_p = out_c = None
    try:
        out_p, out_c = softmax_alg2_prefix_partitions_fhe(
            p_ciphers,
            c_ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            prefix_k=prefix_k,
            rounds=alg2_k,
        )
        dec_p = np.stack(
            [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in out_p]
        )
        dec_c = np.stack(
            [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in out_c]
        )
        actual_p = _unpack_prefix_decrypted(dec_p, layout, prefix_k)
        actual_c = _unpack_prefix_decrypted(dec_c, layout, 4)
        np.testing.assert_allclose(actual_p, expected_p, atol=8e-2, rtol=8e-2)
        np.testing.assert_allclose(actual_c, expected_c, atol=8e-2, rtol=8e-2)
        for query in range(4):
            if query < 3:
                np.testing.assert_allclose(
                    actual_c[:, query, query + 1 :], 0.0, atol=5e-3
                )
    finally:
        for cipher in p_ciphers + c_ciphers + tuple(out_p or ()) + tuple(out_c or ()):
            release_if_supported(cipher)


@pytest.mark.parametrize("prefix_k", [1, 2])
def test_packed_pc_qk_pv_match_logical_small(prefix_k: int) -> None:
    layout = _small_layout()
    rng = np.random.default_rng(5)
    query, key_p, key_c, value_p, value_c = _draw_tensors(
        layout, prefix_k=prefix_k, rng=rng
    )
    _, qk_err = packed_prefix_qk_matches_logical(query, key_p, layout=layout)
    assert qk_err < 1e-5
    split = attention_prefix_gqa_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    _, pv_err = packed_prefix_pv_matches_logical(
        split.probabilities_p, value_p, layout=layout
    )
    assert pv_err < 1e-5
    packed_o, err = packed_prefix_attention_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    np.testing.assert_allclose(packed_o, split.output, atol=1e-5, rtol=1e-5)
    assert packed_o.shape[0] == layout.seq_len
    assert max(v for k, v in err.items() if k != "output_tokens") < 1e-5


def test_packed_pc_attention_matches_a1_production() -> None:
    layout = AttentionPairLayout()
    rng = np.random.default_rng(6)
    query, key_p, key_c, value_p, value_c = _draw_tensors(
        layout, prefix_k=2, rng=rng
    )
    query, key_p, key_c = rope_prefix_and_content(
        query,
        key_p,
        key_c,
        layout=layout,
        prefix_k=2,
        theta=1_000_000.0,
    )
    packed_o, err = packed_prefix_attention_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    split = attention_prefix_gqa_numpy(
        query, key_p, key_c, value_p, value_c, layout=layout
    )
    np.testing.assert_allclose(packed_o, split.output, atol=2e-5, rtol=2e-5)
    assert packed_o.shape == (128, 32 * 128)
    assert err["output_tokens"] == 128.0
    assert max(v for k, v in err.items() if k != "output_tokens") < 2e-5


def _gpu_prefix_layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=8192,
    )


@pytest.mark.gpu
def test_prefix_pc_qk_pv_fhe_matches_numpy_small() -> None:
    from llama3fhe.approx.prefix import dual_softmax_numpy, qk_content_numpy
    from llama3fhe.backend import create_sim_context, release_if_supported
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.attention.prefix.prefix_pc import (
        prefix_pc_rotations,
        pv_prefix_pc_fhe,
        qk_prefix_pc_fhe,
    )

    layout = _gpu_prefix_layout()
    rng = np.random.default_rng(26091804)
    query = rng.normal(scale=0.05, size=(4, 16)).astype(np.float32)
    key_p = rng.normal(scale=0.05, size=(2, 8)).astype(np.float32)
    key_c = rng.normal(scale=0.05, size=(4, 8)).astype(np.float32)
    value_p = rng.normal(scale=0.05, size=(2, 8)).astype(np.float32)
    expected_s = layout.unpack_prefix_scores_prefill(
        qk_prefix_packed_numpy(query, key_p, layout=layout), prefix_k=2
    )
    scores_p = expected_s / np.float32(np.sqrt(float(layout.head_dim)))
    scores_c = qk_content_numpy(query, key_c, layout=layout, scale=True)
    probs_p, _ = dual_softmax_numpy(scores_p, scores_c, prefix_k=2)
    expected_o = layout.unpack_output_c(
        pv_prefix_packed_numpy(probs_p, value_p, layout=layout)
    )
    packed_q = layout.pack_query_c(query)
    packed_p = layout.pack_prefix_scores_prefill(probs_p)
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 8
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=prefix_pc_rotations(layout, prefix_k=2),
        include_default_rotations=True,
    )
    q_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_q
    )
    p_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_p
    )
    out_s = out_o = None
    try:
        out_s = qk_prefix_pc_fhe(
            q_ciphers, key_p, layout=layout, crypto_context=context, prefix_k=2
        )
        actual_s = _unpack_prefix_decrypted(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in out_s]
            ),
            layout,
            2,
        )
        np.testing.assert_allclose(actual_s, expected_s, atol=5e-3, rtol=5e-3)

        out_o = pv_prefix_pc_fhe(
            p_ciphers, value_p, layout=layout, crypto_context=context, prefix_k=2
        )
        actual_o = _unpack_output_decrypted(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in out_o]
            ),
            layout,
        )
        np.testing.assert_allclose(actual_o, expected_o, atol=5e-3, rtol=5e-3)
        assert actual_o.shape[0] == layout.seq_len
    finally:
        for cipher in q_ciphers + p_ciphers + tuple(out_s or ()) + tuple(out_o or ()):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_prefix_pc_qk_extracts_real_for_folded_query() -> None:
    from llama3fhe.backend import create_sim_context, release_if_supported
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.attention.prefix.prefix_pc import (
        prefix_pc_rotations,
        qk_prefix_pc_fhe,
    )

    layout = _gpu_prefix_layout()
    rng = np.random.default_rng(26091806)
    query = rng.normal(scale=0.05, size=(4, 16)).astype(np.float32)
    key_p = rng.normal(scale=0.05, size=(2, 8)).astype(np.float32)
    expected = layout.unpack_prefix_scores_prefill(
        qk_prefix_packed_numpy(query, key_p, layout=layout), prefix_k=2
    )
    packed_q = layout.pack_query_c_feature_folded(query, scale=0.5)
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 8
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=prefix_pc_rotations(layout, prefix_k=2),
        include_default_rotations=True,
    )
    q_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_q
    )
    outputs = ()
    try:
        outputs = qk_prefix_pc_fhe(
            q_ciphers,
            key_p,
            layout=layout,
            crypto_context=context,
            prefix_k=2,
            complex_feature_folding=True,
        )
        actual = _unpack_prefix_decrypted(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in outputs]
            ),
            layout,
            2,
        )
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
    finally:
        for cipher in q_ciphers + tuple(outputs):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_prefix_pc_attention_fhe_e2e_small() -> None:
    from llama3fhe.approx.prefix import (
        dual_softmax_alg2_numpy,
        pv_content_numpy,
        pv_prefix_numpy,
        qk_content_numpy,
    )
    from llama3fhe.backend import create_sim_context, release_if_supported
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly
    from llama3fhe.operators.attention.prefix.prefix_pc import (
        prefix_pc_rotations,
        pv_prefix_pc_fhe,
        qk_prefix_pc_fhe,
    )
    from llama3fhe.operators.attention.prefix.prefix_softmax import (
        prefix_softmax_rotations,
        softmax_alg2_prefix_partitions_fhe,
    )

    layout = _gpu_prefix_layout()
    rng = np.random.default_rng(26091805)
    query = rng.normal(scale=0.05, size=(4, 16)).astype(np.float32)
    key_p = rng.normal(scale=0.05, size=(2, 8)).astype(np.float32)
    key_c = rng.normal(scale=0.05, size=(4, 8)).astype(np.float32)
    value_p = rng.normal(scale=0.05, size=(2, 8)).astype(np.float32)
    value_c = rng.normal(scale=0.05, size=(4, 8)).astype(np.float32)
    scores_c = qk_content_numpy(query, key_c, layout=layout, scale=False)
    packed_q = layout.pack_query_c(query)
    packed_sc = layout.pack_prefix_scores_prefill(scores_c)
    config = DecodeAlg2Poly.synthetic(k=1, degree=8)
    rotations = tuple(
        dict.fromkeys(
            prefix_pc_rotations(layout, prefix_k=2) + prefix_softmax_rotations(layout)
        )
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 36
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=rotations,
        include_default_rotations=True,
    )
    q_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_q
    )
    c_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_sc
    )
    scores_p_ct = probs_p = probs_c = out_p = None
    try:
        scores_p_ct = qk_prefix_pc_fhe(
            q_ciphers, key_p, layout=layout, crypto_context=context, prefix_k=2
        )
        packed_sp = np.stack(
            [
                np.asarray(context.decrypt(ct), dtype=np.float64).real
                for ct in scores_p_ct
            ]
        )
        scores_p = _unpack_prefix_decrypted(packed_sp, layout, 2)
        expected_p, expected_c, _ = dual_softmax_alg2_numpy(
            scores_p, scores_c, layout=layout, config=config, prefix_k=2
        )
        expected_o = pv_prefix_numpy(expected_p, value_p, layout=layout)
        expected_oc = pv_content_numpy(expected_c, value_c, layout=layout)
        expected = expected_o + expected_oc

        probs_p, probs_c = softmax_alg2_prefix_partitions_fhe(
            scores_p_ct,
            c_ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            prefix_k=2,
            rounds=1,
        )
        out_p = pv_prefix_pc_fhe(
            probs_p, value_p, layout=layout, crypto_context=context, prefix_k=2
        )
        actual_o = _unpack_output_decrypted(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in out_p]
            ),
            layout,
        )
        actual_pc = _unpack_prefix_decrypted(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in probs_c]
            ),
            layout,
            4,
        )
        actual_oc = pv_content_numpy(actual_pc, value_c, layout=layout)
        actual = actual_o + actual_oc
        np.testing.assert_allclose(actual_o, expected_o, atol=8e-2, rtol=8e-2)
        np.testing.assert_allclose(actual, expected, atol=8e-2, rtol=8e-2)
        assert actual.shape[0] == layout.seq_len
    finally:
        for cipher in (
            q_ciphers
            + c_ciphers
            + tuple(scores_p_ct or ())
            + tuple(probs_p or ())
            + tuple(probs_c or ())
            + tuple(out_p or ())
        ):
            release_if_supported(cipher)


def _unpack_delta_decrypted(packed, layout: AttentionPairLayout):
    packed = np.asarray(packed, dtype=np.float64).real
    base = packed[:, : layout.payload_slots]
    tiled = np.tile(base, (1, layout.payload_repetitions))
    return layout.unpack_score_delta(tiled)


@pytest.mark.gpu
@pytest.mark.parametrize("alg2_k", [1, 2])
def test_prefix_alg2_delta_content_form_matches_numpy(alg2_k: int) -> None:
    from llama3fhe.approx.prefix import dual_softmax_alg2_numpy, qk_content_numpy, qk_prefix_numpy
    from llama3fhe.backend import create_sim_context, release_if_supported
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly
    from llama3fhe.operators.attention.prefix.prefix_softmax import (
        prefix_softmax_rotations,
        softmax_alg2_prefix_partitions_fhe,
    )

    layout = _gpu_prefix_layout()
    rng = np.random.default_rng(26091806)
    query = rng.normal(scale=0.05, size=(4, 16)).astype(np.float32)
    key_p = rng.normal(scale=0.05, size=(2, 8)).astype(np.float32)
    key_c = rng.normal(scale=0.05, size=(4, 8)).astype(np.float32)
    scores_p = qk_prefix_numpy(query, key_p, layout=layout, scale=False)
    scores_c = qk_content_numpy(query, key_c, layout=layout, scale=False)
    config = DecodeAlg2Poly.synthetic(k=alg2_k, degree=8)
    expected_p, expected_c, _ = dual_softmax_alg2_numpy(
        scores_p, scores_c, layout=layout, config=config, prefix_k=2
    )
    packed_p = layout.pack_prefix_scores_prefill(scores_p)
    packed_c = layout.pack_score_delta(scores_c)
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 32
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=prefix_softmax_rotations(layout),
        include_default_rotations=True,
    )
    p_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_p
    )
    c_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in packed_c
    )
    out_p = out_c = None
    try:
        out_p, out_c = softmax_alg2_prefix_partitions_fhe(
            p_ciphers,
            c_ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            prefix_k=2,
            rounds=alg2_k,
            content_form="delta",
        )
        actual_p = _unpack_prefix_decrypted(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in out_p]
            ),
            layout,
            2,
        )
        actual_c = _unpack_delta_decrypted(
            np.stack(
                [np.asarray(context.decrypt(ct), dtype=np.float64).real for ct in out_c]
            ),
            layout,
        )
        np.testing.assert_allclose(actual_p, expected_p, atol=8e-2, rtol=8e-2)
        np.testing.assert_allclose(actual_c, expected_c, atol=8e-2, rtol=8e-2)
    finally:
        for cipher in p_ciphers + c_ciphers + tuple(out_p or ()) + tuple(out_c or ()):
            release_if_supported(cipher)
