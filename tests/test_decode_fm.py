from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.application.client import decrypt_hidden
from llama3fhe.application.session import _encrypt_hidden
from llama3fhe.approx.chebyshev import (
    eval_chebyshev_coefficients,
    eval_chebyshev_series_direct,
)
from llama3fhe.approx.layer import (
    poly_decode_layer_numpy,
    poly_transformer_layer_numpy,
)
from llama3fhe.approx.prefix import PrefixKVBundle
from llama3fhe.approx.rmsnorm import rmsnorm_poly_numpy
from llama3fhe.approx.sample_fit import fit_stack_from_plaintext
from llama3fhe.application.approx_config import ApproximationConfig
from llama3fhe.application.args import DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
from llama3fhe.backend import create_sim_context, release_if_supported, trim_device_allocator_cache
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.graph import (
    LayerApproximations,
    LayerWeights,
    decode_layer_rotations,
    evaluate_decode,
    inspect_decode_layer_input,
    required_rotations,
)
from llama3fhe.graph import (
    evaluate as evaluate_layer,
)
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import (
    FeatureMajorPrefillLayout,
    decode_feature_major_layout,
)
from llama3fhe.layouts.kv_cache import LayerKvCache
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.model import Llama3Model, TransformerLayer, prepare_bootstrap_operator
from llama3fhe.operators.attention.decode.decode import (
    decode_attention_alg2_numpy,
    decode_attention_alg2_pages_numpy,
    decode_prepare_qkv_numpy,
    output_c_to_decode_fm_numpy,
)
from llama3fhe.operators.attention.decode.decode_adapter import (
    decode_adapter_rotations,
    decode_prepare_qkv_fhe,
    decode_qkv_carrier_to_c_numpy,
    unpack_decode_qkv_carrier,
)
from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly
from llama3fhe.operators.kv_cache import add_c_pages
from llama3fhe.operators.linear.complex import pair_real_ciphers
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.prefill.mlp import mlp_packed_numpy
from llama3fhe.operators.linear.prefill.output import decode_w_o_numpy
from llama3fhe.operators.linear.prefill.qkv import qkv_fhe, qkv_rotations, token_pair_complexity
from llama3fhe.operators.linear.prefill.qkv_kernel import _prepare_qkv_weights
from llama3fhe.operators.norm.rmsnorm import (
    residual_add,
    rmsnorm_fhe,
    rmsnorm_rotations,
)
from llama3fhe.schedule import LayerSchedule


def _silu(value: float) -> float:
    return float(value / (1.0 + np.exp(-float(value))))


def test_decode_fm_is_one_cipher_with_empty_lanes() -> None:
    layout = decode_feature_major_layout(hidden_dim=16, slots=128)
    assert layout.seq_len == 1
    assert layout.tokens_per_cipher == 8
    assert layout.cipher_count == 1

    row = np.arange(16, dtype=np.float32)
    packed = layout.pack(row.reshape(1, 16))
    assert packed.shape == (1, 128)
    for dim in range(16):
        assert packed[0, layout.slot(dim, 0)] == row[dim]
        for lane in range(1, 8):
            assert packed[0, layout.slot(dim, lane)] == 0.0
    np.testing.assert_array_equal(layout.unpack(packed), row.reshape(1, 16))


def test_decrypt_hidden_accepts_one_cipher() -> None:
    layout = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=4, slots=8)
    logical = np.arange(4, dtype=np.float32).reshape(1, 4)
    packed = layout.pack(logical)

    class Context:
        @staticmethod
        def decrypt(cipher):
            return cipher

    decoded = decrypt_hidden(
        tuple(packed), crypto_context=Context(), layout=layout
    )
    np.testing.assert_array_equal(decoded, logical)


def test_encrypt_hidden_emits_one_cipher() -> None:
    layout = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=4, slots=8)
    hidden = np.arange(4, dtype=np.float32).reshape(1, 4)
    captured: list[np.ndarray] = []

    class Native:
        L = 10
        device = "cpu"
        scale_at = staticmethod(lambda limbs: 1.0)

    class Client:
        @staticmethod
        def encrypt(row, **kwargs):
            captured.append(np.asarray(row))
            return ("ct", len(captured))

    class Context:
        context = Native()
        client = Client()

    ciphers = _encrypt_hidden(
        hidden, context=Context(), layout=layout, input_level=0
    )
    assert len(ciphers) == 1
    assert captured[0].shape == (8,)
    np.testing.assert_array_equal(captured[0], layout.pack(hidden)[0])


def test_c_form_kv_cache_appends_tokens_and_spills_pages() -> None:
    layout = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=32,
    )
    rng = np.random.default_rng(26091601)
    cache = LayerKvCache(layout=layout)
    tokens = [
        (
            rng.normal(size=(2, 4)).astype(np.float32),
            rng.normal(size=(2, 4)).astype(np.float32),
        )
        for _ in range(5)
    ]
    locations = [cache.append(key, value) for key, value in tokens]
    assert locations[:4] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert locations[4] == (1, 0)
    assert cache.occupied == 5
    assert len(cache.pages) == 2
    assert cache.pages[0].occupied == 4
    assert cache.pages[1].occupied == 1

    first = np.stack([key.reshape(-1) for key, _ in tokens[:4]])
    np.testing.assert_allclose(cache.pages[0].logical_keys(), first, atol=1e-6)
    np.testing.assert_allclose(
        cache.pages[1].logical_values(),
        tokens[4][1].reshape(1, -1),
        atol=1e-6,
    )
    page_keys = np.zeros((4, 8), dtype=np.float32)
    page_keys[:4] = first
    np.testing.assert_allclose(
        cache.pages[0].packed_keys(), layout.pack_key_c(page_keys), atol=1e-6
    )


def test_pack_key_c_roundtrip_matches_query_c_slot_formula() -> None:
    layout = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=32,
    )
    rng = np.random.default_rng(26091602)
    key = rng.normal(size=(4, 2, 4)).astype(np.float32)
    packed = layout.pack_key_c(key)
    np.testing.assert_allclose(layout.unpack_key_c(packed), key.reshape(4, 8))
    assert packed[0, layout.c_slot(1, 2, 0)] == key[2, 0, 1]
    token = layout.pack_token_c(key[2], token_index=2, kv=True, name="key")
    recovered = layout.unpack_key_c(token).reshape(4, 2, 4)
    np.testing.assert_allclose(recovered[2], key[2])
    np.testing.assert_array_equal(recovered[:2], np.zeros((2, 2, 4)))


def test_odd_cipher_pairing_complexity_and_qkv_seq_mismatch() -> None:
    decode = decode_feature_major_layout(hidden_dim=16, slots=128)
    cost = token_pair_complexity(decode)
    assert cost.input_cipher_count == 1
    assert cost.output_cipher_count == 1

    hidden = np.zeros((1, 16), dtype=np.float32)
    attention = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=32,
    )
    q_weight = np.zeros((16, 16), dtype=np.float32)
    kv_weight = np.zeros((8, 16), dtype=np.float32)
    prepared = _prepare_qkv_weights(
        hidden,
        q_weight,
        kv_weight,
        kv_weight,
        attention_layout=attention,
    )
    assert prepared.seq_len == 1


def test_mlp_packed_numpy_accepts_one_token() -> None:
    rng = np.random.default_rng(26091603)
    canonical = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=8, slots=32)
    hidden = rng.normal(scale=0.1, size=(1, 8))
    gate = rng.normal(scale=0.1, size=(12, 8))
    up = rng.normal(scale=0.1, size=(12, 8))
    down = rng.normal(scale=0.1, size=(8, 12))
    silu = lambda value: value / (1.0 + np.exp(-value))

    packed = mlp_packed_numpy(
        hidden, gate, up, down, layout=canonical, silu=silu
    )
    actual = canonical.unpack(packed)
    expected = (silu(hidden @ gate.T) * (hidden @ up.T)) @ down.T
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)


def test_poly_skip_attention_is_identity_residual_then_ffn() -> None:
    rng = np.random.default_rng(26091604)
    layout = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=32,
    )
    hidden_dim = 16
    intermediate = 24
    kv_dim = 8
    weights = LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.05, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        k_proj=rng.normal(scale=0.05, size=(kv_dim, hidden_dim)).astype(
            np.float32
        ),
        v_proj=rng.normal(scale=0.05, size=(kv_dim, hidden_dim)).astype(
            np.float32
        ),
        o_proj=rng.normal(scale=0.05, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.05, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        up_proj=rng.normal(scale=0.05, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        down_proj=rng.normal(scale=0.05, size=(hidden_dim, intermediate)).astype(
            np.float32
        ),
    )
    silu_coeffs = eval_chebyshev_coefficients(_silu, -2.0, 2.0, 12)
    approximations = SimpleNamespace(
        input_norm={"coefficients": [2.0], "fit_interval": [0.0, 1.0]},
        post_attention_norm={"coefficients": [2.0], "fit_interval": [0.0, 1.0]},
        silu={"coefficients": silu_coeffs, "fit_interval": [-2.0, 2.0]},
    )
    hidden = rng.normal(scale=0.1, size=(1, hidden_dim)).astype(np.float32)

    result = poly_transformer_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=layout,
        skip_attention=True,
    )

    gate = hidden @ weights.gate_proj.T
    up = hidden @ weights.up_proj.T
    activated = eval_chebyshev_series_direct(gate, silu_coeffs, -2.0, 2.0) * up
    expected = hidden + activated @ weights.down_proj.T
    np.testing.assert_allclose(result.hidden, expected, rtol=1e-5, atol=1e-5)


def test_skip_attention_is_wired_through_evaluate_and_model() -> None:
    assert signature(evaluate_layer).parameters["skip_attention"].default is False
    assert signature(TransformerLayer.forward).parameters["skip_attention"].default is False
    assert signature(Llama3Model.forward).parameters["skip_attention"].default is False
    assert signature(evaluate_layer).parameters["emit_kv_cache"].default is False
    assert signature(TransformerLayer.forward).parameters["emit_kv_cache"].default is False
    assert signature(Llama3Model.forward).parameters["emit_kv_cache"].default is False
    assert signature(evaluate_decode).parameters["skip_attention"].default is True
    assert "input_ciphers" in signature(evaluate_decode).parameters
    assert "kv_cache_k_ciphers" in signature(evaluate_decode).parameters
    assert "emit_kv_cache" not in signature(evaluate_decode).parameters
    assert LayerSchedule().kv_limbs is None
    assert signature(TransformerLayer.forward_decode).parameters[
        "skip_attention"
    ].default is False
    assert signature(Llama3Model.decode).parameters["skip_attention"].default is False
    assert "kv_caches" in signature(Llama3Model.decode).parameters
    assert (
        signature(Llama3Model.decode).parameters["refresh_post_attention"].default
        is False
    )


def test_decode_layer_rotations_union_bootstrap_keys() -> None:
    feature = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=4096, slots=32768)
    attention = AttentionPairLayout()
    schedule = LayerSchedule()
    base = decode_layer_rotations(
        feature, attention, schedule=schedule, skip_attention=True
    )
    with_bs = decode_layer_rotations(
        feature,
        attention,
        schedule=schedule,
        skip_attention=True,
        include_bootstrap=True,
        log_n=16,
    )
    assert set(base).issubset(set(with_bs))
    assert len(with_bs) > len(base)


@pytest.mark.gpu
def test_decode_fm_residual_and_odd_pairing_roundtrip() -> None:
    rng = np.random.default_rng(26091605)
    layout = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=128, slots=8192)
    left = rng.normal(scale=0.02, size=(1, 128)).astype(np.float32)
    right = np.zeros_like(left)
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 4
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=int(layout.hidden_dim),
        block_out=int(layout.tokens_per_cipher),
        rotations=rmsnorm_rotations(layout),
        include_default_rotations=True,
    )
    left_ciphers = tuple(
        context.encrypt(payload, level=0, slots=layout.slots)
        for payload in layout.pack(left)
    )
    right_ciphers = tuple(
        context.encrypt(payload, level=0, slots=layout.slots)
        for payload in layout.pack(right)
    )
    paired = residual = None
    try:
        assert len(left_ciphers) == 1
        paired = pair_real_ciphers(left_ciphers, crypto_context=context)
        assert len(paired) == 1
        paired_slots = np.asarray(context.decrypt(paired[0]))
        expected_slots = layout.pack(left)[0]
        np.testing.assert_allclose(
            np.asarray(paired_slots.real, dtype=np.float64),
            expected_slots,
            atol=5e-4,
            rtol=5e-4,
        )
        np.testing.assert_allclose(
            np.asarray(paired_slots.imag, dtype=np.float64),
            np.zeros_like(expected_slots),
            atol=5e-4,
            rtol=5e-4,
        )
        residual = residual_add(
            left_ciphers,
            right_ciphers,
            layout=layout,
            crypto_context=context,
            verify=True,
            expected_left=left,
            expected_right=right,
        )
        assert residual.max_abs_diff is not None and residual.max_abs_diff < 1e-6
    finally:
        if residual is not None:
            residual.release()
        for cipher in left_ciphers + right_ciphers + tuple(paired or ()):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_c_page_cipher_add_appends_one_token() -> None:
    rng = np.random.default_rng(26091606)
    layout = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=8192,
    )
    key0 = rng.normal(size=(2, 4)).astype(np.float32)
    key1 = rng.normal(size=(2, 4)).astype(np.float32)
    page = layout.pack_token_c(key0, token_index=0, kv=True, name="key")
    token = layout.pack_token_c(key1, token_index=1, kv=True, name="key")
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 4
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=4,
        block_out=4,
        rotations=(),
        include_default_rotations=False,
    )
    page_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in page
    )
    token_ciphers = tuple(
        context.encrypt(row, level=0, slots=layout.slots) for row in token
    )
    summed = None
    try:
        summed = add_c_pages(
            page_ciphers, token_ciphers, crypto_context=context
        )
        decrypted = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in summed
            ]
        )
        expected = layout.pack_token_c(key0, token_index=0, kv=True, name="key")
        expected = expected + layout.pack_token_c(
            key1, token_index=1, kv=True, name="key"
        )
        np.testing.assert_allclose(decrypted, expected, atol=5e-4, rtol=5e-4)
    finally:
        for cipher in page_ciphers + token_ciphers + tuple(summed or ()):
            release_if_supported(cipher)


def _small_decode_attention() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=8192,
    )


def _small_decode_schedule() -> LayerSchedule:
    return LayerSchedule(
        linear_baby_steps=8,
        linear_anchor_step=8,
        mlp_max_plaintext_rows=16,
        qkv_max_plaintext_rows=16,
        output_max_plaintext_rows=16,
        fuse_rmsnorm_output_scale=False,
        fuse_silu_domain_map=False,
    )


def _mha_decode_attention() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=4,
        slots=8192,
    )


def _pack_decode_qkv_carrier(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> np.ndarray:
    head_dim = int(layout.head_dim)
    half = head_dim // 2
    query_fold = int(layout.query_pair_count) * head_dim
    key_base = query_fold
    value_base = key_base + int(layout.key_value_pair_count) * head_dim
    width = value_base + int(layout.key_value_pair_count) * head_dim
    carrier = np.zeros((width,), dtype=np.complex128)
    query = np.asarray(query, dtype=np.float64).reshape(int(layout.query_heads), head_dim)
    key = np.asarray(key, dtype=np.float64).reshape(
        int(layout.key_value_heads), head_dim
    )
    value = np.asarray(value, dtype=np.float64).reshape(
        int(layout.key_value_heads), head_dim
    )
    for pair_index, heads in enumerate(layout.query_head_pairs()):
        for feature_low in range(half):
            for lane, head in enumerate(heads):
                channel = pair_index * head_dim + 2 * feature_low + lane
                carrier[channel] = 0.5 * (
                    query[int(head), feature_low]
                    + 1j * query[int(head), feature_low + half]
                )
    for pair_index, heads in enumerate(layout.key_value_head_pairs()):
        for feature_low in range(half):
            for lane, head in enumerate(heads):
                channel = key_base + pair_index * head_dim + 2 * feature_low + lane
                carrier[channel] = (
                    key[int(head), feature_low]
                    - 1j * key[int(head), feature_low + half]
                )
    pair_width = 2 * head_dim
    kv_pairs = layout.key_value_head_pairs()
    for group in range(int(layout.key_value_pair_count) // 2):
        even_heads, odd_heads = kv_pairs[2 * group], kv_pairs[2 * group + 1]
        for feature in range(head_dim):
            for lane in (0, 1):
                channel = value_base + group * pair_width + 2 * feature + lane
                even = value[int(even_heads[lane]), feature]
                odd = value[int(odd_heads[lane]), feature]
                carrier[channel] = 0.5 * (even + 1j * odd)
    return carrier


def test_inspect_decode_layer_input_rejects_prefill_shape() -> None:
    prefill = FeatureMajorPrefillLayout(seq_len=8, hidden_dim=16, slots=128)
    with pytest.raises(ValueError, match="seq_len=1"):
        inspect_decode_layer_input(
            (object(),), feature_layout=prefill, crypto_context=None
        )
    decode = decode_feature_major_layout(hidden_dim=16, slots=128)
    with pytest.raises(ValueError, match="1 DecodeFM"):
        inspect_decode_layer_input(
            (), feature_layout=decode, crypto_context=None
        )


@pytest.mark.gpu
def test_decode_layer_input_limbs_and_skip_attention_ffn() -> None:
    rng = np.random.default_rng(26091712)
    feature = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=16, slots=8192)
    attention = _small_decode_attention()
    hidden = rng.normal(scale=0.1, size=(1, 16)).astype(np.float32)
    eps = 1e-5
    variance = np.mean(np.square(hidden, dtype=np.float64), axis=-1)
    fit = (float(np.min(variance) * 0.5), float(np.max(variance) * 2.0 + eps))
    rmsn = {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: 1.0 / np.sqrt(value + eps),
            fit[0],
            fit[1],
            8,
        ),
        "fit_interval": fit,
        "eps": eps,
    }
    silu = {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: value / (1.0 + np.exp(-value)),
            -1.0,
            1.0,
            8,
        ),
        "fit_interval": (-1.0, 1.0),
    }
    weights = LayerWeights(
        input_layernorm=np.zeros((16,), dtype=np.float32),
        q_proj=rng.normal(scale=0.05, size=(16, 16)).astype(np.float32),
        k_proj=rng.normal(scale=0.05, size=(8, 16)).astype(np.float32),
        v_proj=rng.normal(scale=0.05, size=(8, 16)).astype(np.float32),
        o_proj=rng.normal(scale=0.05, size=(16, 16)).astype(np.float32),
        post_attention_layernorm=np.zeros((16,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.05, size=(24, 16)).astype(np.float32),
        up_proj=rng.normal(scale=0.05, size=(24, 16)).astype(np.float32),
        down_proj=rng.normal(scale=0.05, size=(16, 24)).astype(np.float32),
    )
    approximations = LayerApproximations(
        input_norm=rmsn,
        post_attention_norm=rmsn,
        silu=silu,
        softmax=None,
    )
    expected = poly_transformer_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        attention_layout=attention,
        skip_attention=True,
    ).hidden
    schedule = _small_decode_schedule()
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 32
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=int(feature.hidden_dim),
        block_out=int(feature.tokens_per_cipher),
        rotations=decode_layer_rotations(
            feature, attention, schedule=schedule, skip_attention=True
        ),
        include_default_rotations=True,
    )
    packed = feature.pack(hidden)
    assert packed.shape == (1, feature.slots)
    ciphers = tuple(
        context.encrypt(row, level=0, slots=feature.slots) for row in packed
    )
    result = None
    try:
        entry = inspect_decode_layer_input(
            ciphers, feature_layout=feature, crypto_context=context
        )
        assert entry["cipher_count"] == 1
        assert entry["live_lane"] == 0
        assert int(entry["cur_limbs"]) == int(context.context.L)
        assert int(entry["level"]) == 0

        result = evaluate_decode(
            hidden,
            layer_idx=0,
            weights=weights,
            approximations=approximations,
            feature_layout=feature,
            attention_layout=attention,
            config=cfg,
            crypto_context=context,
            input_ciphers=ciphers,
            schedule=schedule,
            skip_attention=True,
            verify=True,
            expected_output=expected,
        )
        assert result.output is not None
        assert result.max_abs_diff is not None and result.max_abs_diff < 5e-3
        nested = result.nested_stage_seconds
        assert nested["decode_input"]["cur_limbs"] == entry["cur_limbs"]
        assert "input_rmsnorm" in result.stage_seconds
        assert "mlp" in result.stage_seconds
        np.testing.assert_allclose(result.output, expected, atol=5e-3, rtol=5e-3)
    finally:
        if result is not None:
            result.release()
        for cipher in ciphers:
            release_if_supported(cipher)


def test_decode_fm_qkv_pairing_is_one_complex_carrier() -> None:
    layout = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=32, slots=8192)
    cost = token_pair_complexity(layout)
    assert cost.input_cipher_count == 1
    assert cost.output_cipher_count == 1
    assert layout.cipher_count == 1


def test_decode_prepare_qkv_numpy_packs_broadcast_and_token_c() -> None:
    rng = np.random.default_rng(26091714)
    attention = _small_decode_attention()
    hidden = rng.normal(scale=0.1, size=(1, 16)).astype(np.float32)
    q_weight = rng.normal(scale=0.05, size=(16, 16)).astype(np.float32)
    k_weight = rng.normal(scale=0.05, size=(8, 16)).astype(np.float32)
    v_weight = rng.normal(scale=0.05, size=(8, 16)).astype(np.float32)
    prepared = decode_prepare_qkv_numpy(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        layout=attention,
        position=2,
        token_index=2,
    )
    assert prepared.query.shape == (1, 16)
    assert prepared.key.shape == (1, 8)
    assert prepared.compact_query_c.shape == (1, attention.slots)
    unpacked_q = attention.unpack_compact_query_c(prepared.compact_query_c)
    np.testing.assert_allclose(
        unpacked_q,
        prepared.query.reshape(attention.query_heads, attention.head_dim),
        atol=1e-6,
    )
    assert prepared.key_token_c.shape == (
        attention.key_value_pair_count,
        attention.slots,
    )
    unpacked_k = attention.unpack_key_c(prepared.key_token_c)
    np.testing.assert_allclose(unpacked_k[2], prepared.key[0], atol=1e-6)
    np.testing.assert_allclose(unpacked_k[0], 0.0, atol=1e-7)


@pytest.mark.gpu
def test_decode_layer_rmsn1_then_qkv_matches_numpy() -> None:
    rng = np.random.default_rng(26091713)
    hidden_dim = 32
    feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=8192
    )
    attention = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=4,
        slots=8192,
    )
    hidden = rng.normal(scale=0.1, size=(1, hidden_dim)).astype(np.float32)
    eps = 1e-5
    variance = np.mean(np.square(hidden, dtype=np.float64), axis=-1)
    fit = (float(np.min(variance) * 0.5), float(np.max(variance) * 2.0 + eps))
    rmsn = {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: 1.0 / np.sqrt(value + eps),
            fit[0],
            fit[1],
            8,
        ),
        "fit_interval": fit,
        "eps": eps,
    }
    q_weight = rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32)
    k_weight = rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32)
    v_weight = rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32)
    linear = LinearOperatorConfig(
        baby_steps=8,
        baby_anchor_step=8,
        max_plaintext_rows=32,
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 24
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(feature.tokens_per_cipher),
        rotations=tuple(
            sorted(
                set(rmsnorm_rotations(feature))
                | set(qkv_rotations(feature, operator_config=linear))
            )
        ),
        include_default_rotations=False,
    )
    packed = feature.pack(hidden)
    ciphers = tuple(
        context.encrypt(row, level=0, slots=feature.slots) for row in packed
    )
    input_norm = qkv = None
    try:
        entry = inspect_decode_layer_input(
            ciphers, feature_layout=feature, crypto_context=context
        )
        assert entry["cipher_count"] == 1
        assert entry["live_lane"] == 0
        assert int(entry["cur_limbs"]) == int(context.context.L)
        assert int(entry["level"]) == 0

        input_norm = rmsnorm_fhe(
            ciphers,
            np.zeros((hidden_dim,), dtype=np.float32),
            coeffs=rmsn,
            layout=feature,
            crypto_context=context,
            verify=True,
            expected_input=hidden,
            function_prefix="model.layers.0.input_layernorm.decode",
        )
        assert input_norm.output is not None
        assert input_norm.max_abs_diff is not None and input_norm.max_abs_diff < 5e-3
        assert len(input_norm.ciphers) == 1

        qkv = qkv_fhe(
            input_norm.output,
            q_weight,
            k_weight,
            v_weight,
            attention_layout=attention,
            layout=feature,
            config=cfg,
            operator_config=linear,
            crypto_context=context,
            input_ciphers=input_norm.ciphers,
            verify=True,
        )
        assert qkv.pair_complexity.input_cipher_count == 1
        assert qkv.pair_complexity.output_cipher_count == 1
        assert len(qkv.carrier_ciphers) == 1
        assert qkv.output_layout.seq_len == 1
        assert qkv.output_layout.cipher_count == 1
        assert qkv.max_abs_diff is not None and qkv.max_abs_diff < 5e-3
        assert int(qkv.output_levels[0]) > int(input_norm.output_levels[0])
    finally:
        if qkv is not None:
            qkv.release()
        if input_norm is not None:
            input_norm.release()
        for cipher in ciphers:
            release_if_supported(cipher)


def test_unpack_decode_qkv_carrier_then_c_packs() -> None:
    rng = np.random.default_rng(26091715)
    attention = _mha_decode_attention()
    query = rng.normal(scale=0.1, size=(4, 4)).astype(np.float32)
    key = rng.normal(scale=0.1, size=(4, 4)).astype(np.float32)
    value = rng.normal(scale=0.1, size=(4, 4)).astype(np.float32)
    carrier = _pack_decode_qkv_carrier(query, key, value, layout=attention)
    got_q, got_k, got_v = unpack_decode_qkv_carrier(carrier, layout=attention)
    np.testing.assert_allclose(got_q, query, atol=1e-6)
    np.testing.assert_allclose(got_k, key, atol=1e-6)
    np.testing.assert_allclose(got_v, value, atol=1e-6)
    hidden = rng.normal(scale=0.1, size=(1, 32)).astype(np.float32)
    q_weight = rng.normal(scale=0.05, size=(16, 32)).astype(np.float32)
    k_weight = rng.normal(scale=0.05, size=(16, 32)).astype(np.float32)
    v_weight = rng.normal(scale=0.05, size=(16, 32)).astype(np.float32)
    prepared = decode_prepare_qkv_numpy(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        layout=attention,
        position=2,
        token_index=2,
    )
    logical_q = (hidden @ q_weight.T).reshape(4, 4)
    logical_k = (hidden @ k_weight.T).reshape(4, 4)
    logical_v = (hidden @ v_weight.T).reshape(4, 4)
    packed = _pack_decode_qkv_carrier(
        logical_q, logical_k, logical_v, layout=attention
    )
    q_c, k_c, v_c, _, _, _ = decode_qkv_carrier_to_c_numpy(
        packed, layout=attention, position=2, token_index=2
    )
    np.testing.assert_allclose(q_c, prepared.compact_query_c, atol=1e-5)
    np.testing.assert_allclose(k_c, prepared.key_token_c, atol=1e-5)
    np.testing.assert_allclose(v_c, prepared.value_token_c, atol=1e-5)


@pytest.mark.gpu
def test_decode_qkv_adapter_fhe_matches_numpy() -> None:
    rng = np.random.default_rng(26091716)
    hidden_dim = 32
    feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=8192
    )
    attention = _mha_decode_attention()
    hidden = rng.normal(scale=0.1, size=(1, hidden_dim)).astype(np.float32)
    q_weight = rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32)
    k_weight = rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32)
    v_weight = rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32)
    prepared = decode_prepare_qkv_numpy(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        layout=attention,
        position=2,
        token_index=2,
    )
    linear = LinearOperatorConfig(
        baby_steps=8,
        baby_anchor_step=8,
        max_plaintext_rows=32,
    )

    carrier_layout = LinearCarrierLayout(
        seq_len=1, dimension=hidden_dim, slots=8192
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 8
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(feature.tokens_per_cipher),
        rotations=tuple(
            sorted(
                set(qkv_rotations(feature, operator_config=linear))
                | set(
                    decode_adapter_rotations(
                        carrier_layout, attention, max_baby_offsets=32
                    )
                )
            )
        ),
        include_default_rotations=False,
    )
    ciphers = tuple(
        context.encrypt(row, level=0, slots=feature.slots)
        for row in feature.pack(hidden)
    )
    qkv = adapter = None
    try:
        qkv = qkv_fhe(
            hidden,
            q_weight,
            k_weight,
            v_weight,
            attention_layout=attention,
            layout=feature,
            config=cfg,
            operator_config=linear,
            crypto_context=context,
            input_ciphers=ciphers,
            verify=True,
        )
        assert qkv.max_abs_diff is not None and qkv.max_abs_diff < 5e-3
        adapter = decode_prepare_qkv_fhe(
            qkv.carrier_ciphers,
            carrier_layout=qkv.output_layout,
            attention_layout=attention,
            crypto_context=context,
            position=2,
            token_index=2,
        )
        actual_q = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in adapter.query_ciphers
            ]
        )
        actual_k = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in adapter.key_ciphers
            ]
        )
        actual_v = np.stack(
            [
                np.asarray(context.decrypt(cipher), dtype=np.float64).real
                for cipher in adapter.value_ciphers
            ]
        )
        np.testing.assert_allclose(
            actual_q, prepared.compact_query_c, atol=5e-3, rtol=5e-3
        )
        np.testing.assert_allclose(
            actual_k, prepared.key_token_c, atol=5e-3, rtol=5e-3
        )
        np.testing.assert_allclose(
            actual_v, prepared.value_token_c, atol=5e-3, rtol=5e-3
        )
    finally:
        if adapter is not None:
            adapter.release()
        if qkv is not None:
            qkv.release()
        for cipher in ciphers:
            release_if_supported(cipher)


def _rmsn_entry(hidden: np.ndarray, *, eps: float = 1e-5, degree: int = 8) -> dict:
    variance = np.mean(np.square(hidden, dtype=np.float64), axis=-1)
    fit = (float(np.min(variance) * 0.5), float(np.max(variance) * 2.0 + eps))
    return {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: 1.0 / np.sqrt(value + eps),
            fit[0],
            fit[1],
            int(degree),
        ),
        "fit_interval": fit,
        "eps": eps,
    }


def _decode_layer_numpy(
    hidden: np.ndarray,
    *,
    weights: LayerWeights,
    approximations: LayerApproximations,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    occupied: int = 0,
    position: int = 0,
    key_page: np.ndarray | None = None,
    value_page: np.ndarray | None = None,
    return_cache: bool = False,
):
    attn_in = rmsnorm_poly_numpy(
        hidden,
        approximations.input_norm["coefficients"],
        approximations.input_norm["fit_interval"],
    )
    prepared = decode_prepare_qkv_numpy(
        attn_in,
        weights.q_proj,
        weights.k_proj,
        weights.v_proj,
        layout=attention_layout,
        position=int(position),
        token_index=int(occupied),
    )
    if int(occupied) == 0:
        key_page = prepared.key_token_c
        value_page = prepared.value_token_c
    else:
        if key_page is None or value_page is None:
            raise ValueError("occupied>0 requires key_page and value_page.")
        key_page = np.asarray(key_page) + prepared.key_token_c
        value_page = np.asarray(value_page) + prepared.value_token_c
    _, _, attn_out = decode_attention_alg2_numpy(
        prepared.query,
        key_page,
        value_page,
        layout=attention_layout,
        occupied=int(occupied) + 1,
        config=approximations.softmax,
        layer_idx=0,
    )
    packed_fm = output_c_to_decode_fm_numpy(
        attn_out,
        attention_layout=attention_layout,
        feature_layout=feature_layout,
    )
    wo = decode_w_o_numpy(
        feature_layout.unpack(packed_fm),
        weights.o_proj,
        feature_layout=feature_layout,
    )
    residual = hidden + wo
    mlp_in = rmsnorm_poly_numpy(
        residual,
        approximations.post_attention_norm["coefficients"],
        approximations.post_attention_norm["fit_interval"],
    )
    silu = approximations.silu
    lo, hi = (float(value) for value in silu["fit_interval"])
    gate = mlp_in @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = mlp_in @ np.asarray(weights.up_proj, dtype=np.float32).T
    activated = eval_chebyshev_series_direct(
        gate, silu["coefficients"], lo, hi
    ) * up
    hidden_out = residual + activated @ np.asarray(
        weights.down_proj, dtype=np.float32
    ).T
    if return_cache:
        return hidden_out, key_page, value_page
    return hidden_out


def _decode_attention_residual_numpy(
    hidden: np.ndarray,
    *,
    weights: LayerWeights,
    rmsn1: dict,
    softmax,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    occupied: int = 0,
    position: int = 0,
    key_page: np.ndarray | None = None,
    value_page: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    attn_in = rmsnorm_poly_numpy(
        hidden, rmsn1["coefficients"], rmsn1["fit_interval"]
    )
    prepared = decode_prepare_qkv_numpy(
        attn_in,
        weights.q_proj,
        weights.k_proj,
        weights.v_proj,
        layout=attention_layout,
        position=int(position),
        token_index=int(occupied),
    )
    if int(occupied) == 0:
        key_page = prepared.key_token_c
        value_page = prepared.value_token_c
    else:
        key_page = np.asarray(key_page) + prepared.key_token_c
        value_page = np.asarray(value_page) + prepared.value_token_c
    _, _, attn_out = decode_attention_alg2_numpy(
        prepared.query,
        key_page,
        value_page,
        layout=attention_layout,
        occupied=int(occupied) + 1,
        config=softmax,
        layer_idx=0,
    )
    wo = decode_w_o_numpy(
        feature_layout.unpack(
            output_c_to_decode_fm_numpy(
                attn_out,
                attention_layout=attention_layout,
                feature_layout=feature_layout,
            )
        ),
        weights.o_proj,
        feature_layout=feature_layout,
    )
    return hidden + wo, key_page, value_page


@pytest.mark.gpu
def test_decode_layer_with_attention_matches_numpy() -> None:
    rng = np.random.default_rng(26091717)
    hidden_dim = 32
    feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=8192
    )
    attention = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=8,
        key_value_heads=4,
        slots=8192,
    )
    hidden = rng.normal(scale=0.1, size=(1, hidden_dim)).astype(np.float32)
    weights = LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.05, size=(32, hidden_dim)).astype(np.float32),
        k_proj=rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.05, size=(hidden_dim, hidden_dim)).astype(np.float32),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.05, size=(48, hidden_dim)).astype(np.float32),
        up_proj=rng.normal(scale=0.05, size=(48, hidden_dim)).astype(np.float32),
        down_proj=rng.normal(scale=0.05, size=(hidden_dim, 48)).astype(np.float32),
    )
    rmsn1 = _rmsn_entry(hidden)
    attn_in = rmsnorm_poly_numpy(
        hidden, rmsn1["coefficients"], rmsn1["fit_interval"]
    )
    prepared = decode_prepare_qkv_numpy(
        attn_in,
        weights.q_proj,
        weights.k_proj,
        weights.v_proj,
        layout=attention,
        position=0,
        token_index=0,
    )
    softmax = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.02, 256.0)
    )
    _, _, attn_out = decode_attention_alg2_numpy(
        prepared.query,
        prepared.key_token_c,
        prepared.value_token_c,
        layout=attention,
        occupied=1,
        config=softmax,
        layer_idx=0,
    )
    wo = decode_w_o_numpy(
        feature.unpack(
            output_c_to_decode_fm_numpy(
                attn_out, attention_layout=attention, feature_layout=feature
            )
        ),
        weights.o_proj,
        feature_layout=feature,
    )
    residual = hidden + wo
    rmsn2 = _rmsn_entry(residual)
    silu = {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: value / (1.0 + np.exp(-value)),
            -1.0,
            1.0,
            8,
        ),
        "fit_interval": (-1.0, 1.0),
    }
    approximations = LayerApproximations(
        input_norm=rmsn1,
        post_attention_norm=rmsn2,
        silu=silu,
        softmax=softmax,
    )
    expected = _decode_layer_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
        occupied=0,
        position=0,
    )
    schedule = _small_decode_schedule()
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 64
    cfg.simulator.device = "cuda"
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(feature.tokens_per_cipher),
        rotations=decode_layer_rotations(
            feature, attention, schedule=schedule, skip_attention=False
        ),
        include_default_rotations=True,
    )
    ciphers = tuple(
        context.encrypt(row, level=0, slots=feature.slots)
        for row in feature.pack(hidden)
    )
    result = None
    try:
        result = evaluate_decode(
            hidden,
            layer_idx=0,
            weights=weights,
            approximations=approximations,
            feature_layout=feature,
            attention_layout=attention,
            config=cfg,
            crypto_context=context,
            input_ciphers=ciphers,
            schedule=schedule,
            skip_attention=False,
            occupied=0,
            position=0,
            verify=True,
            expected_output=expected,
        )
        assert result.output is not None
        assert result.max_abs_diff is not None and result.max_abs_diff < 5e-2
        assert len(result.key_c_ciphers) == int(attention.key_value_pair_count)
        np.testing.assert_allclose(result.output, expected, atol=5e-2, rtol=5e-2)
    finally:
        if result is not None:
            result.release()
        for cipher in ciphers:
            release_if_supported(cipher)


def _toy_decode_weights(rng: np.random.Generator, hidden_dim: int = 32) -> LayerWeights:
    return LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.05, size=(32, hidden_dim)).astype(np.float32),
        k_proj=rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.05, size=(16, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.05, size=(hidden_dim, hidden_dim)).astype(np.float32),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.05, size=(48, hidden_dim)).astype(np.float32),
        up_proj=rng.normal(scale=0.05, size=(48, hidden_dim)).astype(np.float32),
        down_proj=rng.normal(scale=0.05, size=(hidden_dim, 48)).astype(np.float32),
    )


def _toy_decode_silu() -> dict:
    return {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: value / (1.0 + np.exp(-value)),
            -1.0,
            1.0,
            8,
        ),
        "fit_interval": (-1.0, 1.0),
    }


def _toy_decode_layouts(
    hidden_dim: int = 32, slots: int = 8192
) -> tuple[FeatureMajorPrefillLayout, AttentionPairLayout]:
    return (
        FeatureMajorPrefillLayout(
            seq_len=1, hidden_dim=hidden_dim, slots=slots
        ),
        AttentionPairLayout(
            seq_len=4,
            head_dim=4,
            query_heads=8,
            key_value_heads=4,
            slots=slots,
        ),
    )


def _fit_decode_approximations(
    hiddens: np.ndarray,
    *,
    weights: LayerWeights,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> LayerApproximations:
    hiddens = np.asarray(hiddens, dtype=np.float32)
    rmsn1 = _rmsn_entry(hiddens)
    softmax = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.02, 256.0)
    )
    residuals = []
    key_page = value_page = None
    for occupied, hidden in enumerate(hiddens):
        residual, key_page, value_page = _decode_attention_residual_numpy(
            hidden.reshape(1, -1),
            weights=weights,
            rmsn1=rmsn1,
            softmax=softmax,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            occupied=occupied,
            position=occupied,
            key_page=key_page,
            value_page=value_page,
        )
        residuals.append(residual)
    return LayerApproximations(
        input_norm=rmsn1,
        post_attention_norm=_rmsn_entry(np.vstack(residuals)),
        silu=_toy_decode_silu(),
        softmax=softmax,
    )


def _sequential_decode_numpy(
    hiddens: np.ndarray,
    *,
    weights: LayerWeights,
    approximations: LayerApproximations,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    key_page = value_page = None
    for occupied, hidden in enumerate(np.asarray(hiddens, dtype=np.float32)):
        hidden = hidden.reshape(1, -1)
        out, key_page, value_page = _decode_layer_numpy(
            hidden,
            weights=weights,
            approximations=approximations,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            occupied=occupied,
            position=occupied,
            key_page=key_page,
            value_page=value_page,
            return_cache=True,
        )
        outputs.append(out)
    return outputs


def test_decode_numpy_next_token_uses_appended_cache() -> None:
    rng = np.random.default_rng(26091719)
    hidden_dim = 32
    feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=8192
    )
    attention = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=8,
        key_value_heads=4,
        slots=8192,
    )
    h0 = rng.normal(scale=0.1, size=(1, hidden_dim)).astype(np.float32)
    h1 = rng.normal(scale=0.1, size=(1, hidden_dim)).astype(np.float32)
    weights = _toy_decode_weights(rng, hidden_dim)
    rmsn1 = _rmsn_entry(np.vstack([h0, h1]))
    softmax = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.02, 256.0)
    )
    res0, key0, value0 = _decode_attention_residual_numpy(
        h0,
        weights=weights,
        rmsn1=rmsn1,
        softmax=softmax,
        feature_layout=feature,
        attention_layout=attention,
        occupied=0,
        position=0,
    )
    res1, _, _ = _decode_attention_residual_numpy(
        h1,
        weights=weights,
        rmsn1=rmsn1,
        softmax=softmax,
        feature_layout=feature,
        attention_layout=attention,
        occupied=1,
        position=1,
        key_page=key0,
        value_page=value0,
    )
    approximations = LayerApproximations(
        input_norm=rmsn1,
        post_attention_norm=_rmsn_entry(np.vstack([res0, res1])),
        silu=_toy_decode_silu(),
        softmax=softmax,
    )
    out0, cache_k, cache_v = _decode_layer_numpy(
        h0,
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
        occupied=0,
        position=0,
        return_cache=True,
    )
    with_cache = _decode_layer_numpy(
        h1,
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
        occupied=1,
        position=1,
        key_page=cache_k,
        value_page=cache_v,
    )
    without_cache = _decode_layer_numpy(
        h1,
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
        occupied=0,
        position=1,
    )
    assert out0.shape == (1, hidden_dim)
    assert not np.allclose(with_cache, without_cache, atol=1e-5)


def test_decode_numpy_next_n_tokens_uses_growing_cache() -> None:
    rng = np.random.default_rng(26091720)
    hidden_dim = 32
    n_tokens = 4
    feature, attention = _toy_decode_layouts(hidden_dim)
    assert n_tokens == int(attention.seq_len)
    hiddens = rng.normal(scale=0.1, size=(n_tokens, hidden_dim)).astype(np.float32)
    weights = _toy_decode_weights(rng, hidden_dim)
    approximations = _fit_decode_approximations(
        hiddens,
        weights=weights,
        feature_layout=feature,
        attention_layout=attention,
    )
    sequential = _sequential_decode_numpy(
        hiddens,
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
    )
    assert len(sequential) == n_tokens
    independent = [
        _decode_layer_numpy(
            hiddens[index : index + 1],
            weights=weights,
            approximations=approximations,
            feature_layout=feature,
            attention_layout=attention,
            occupied=0,
            position=index,
        )
        for index in range(n_tokens)
    ]
    np.testing.assert_allclose(sequential[0], independent[0], atol=1e-5)
    for index in range(1, n_tokens):
        assert not np.allclose(
            sequential[index], independent[index], atol=1e-5
        )


@pytest.mark.gpu
@pytest.mark.parametrize("consume_kv_cache", [False, True])
def test_decode_model_next_token_kv_cache_matches_numpy(
    consume_kv_cache: bool,
) -> None:
    rng = np.random.default_rng(26091718)
    hidden_dim = 32
    feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=8192
    )
    attention = AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=8,
        key_value_heads=4,
        slots=8192,
    )
    h0 = rng.normal(scale=0.1, size=(1, hidden_dim)).astype(np.float32)
    h1 = rng.normal(scale=0.1, size=(1, hidden_dim)).astype(np.float32)
    weights = _toy_decode_weights(rng, hidden_dim)
    rmsn1 = _rmsn_entry(np.vstack([h0, h1]))
    softmax = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.02, 256.0)
    )
    res0, key0, value0 = _decode_attention_residual_numpy(
        h0,
        weights=weights,
        rmsn1=rmsn1,
        softmax=softmax,
        feature_layout=feature,
        attention_layout=attention,
        occupied=0,
        position=0,
    )
    res1, _, _ = _decode_attention_residual_numpy(
        h1,
        weights=weights,
        rmsn1=rmsn1,
        softmax=softmax,
        feature_layout=feature,
        attention_layout=attention,
        occupied=1,
        position=1,
        key_page=key0,
        value_page=value0,
    )
    approximations = LayerApproximations(
        input_norm=rmsn1,
        post_attention_norm=_rmsn_entry(np.vstack([res0, res1])),
        silu=_toy_decode_silu(),
        softmax=softmax,
    )
    expected0, cache_k, cache_v = _decode_layer_numpy(
        h0,
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
        occupied=0,
        position=0,
        return_cache=True,
    )
    expected1 = _decode_layer_numpy(
        h1,
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
        occupied=1,
        position=1,
        key_page=cache_k,
        value_page=cache_v,
    )
    schedule = _small_decode_schedule()
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 64
    cfg.simulator.device = "cuda"
    layer = TransformerLayer(
        index=0,
        weights=weights,
        approximations=approximations,
        config=cfg,
        feature_layout=feature,
        attention_layout=attention,
    )
    model = Llama3Model(
        config=cfg,
        num_layers=1,
        feature_layout=feature,
        attention_layout=attention,
    )
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(feature.tokens_per_cipher),
        rotations=model.required_decode_rotations(
            schedule, skip_attention=False
        ),
        include_default_rotations=True,
    )
    ciphers0 = tuple(
        context.encrypt(row, level=0, slots=feature.slots)
        for row in feature.pack(h0)
    )
    ciphers1 = tuple(
        context.encrypt(row, level=0, slots=feature.slots)
        for row in feature.pack(h1)
    )
    first = second = None
    try:
        first = model.decode(
            h0,
            crypto_context=context,
            input_ciphers=ciphers0,
            layers=(layer,),
            schedule=schedule,
            skip_attention=False,
            occupied=0,
            position=0,
            verify=True,
            expected_output=expected0,
        )
        assert first.occupied == 1
        assert first.position == 1
        assert len(first.kv_caches) == 1
        assert len(first.kv_caches[0][0]) == int(attention.key_value_pair_count)
        assert first.per_layer[0]["max_abs_diff"] is not None
        assert first.per_layer[0]["max_abs_diff"] < 5e-2
        np.testing.assert_allclose(
            decrypt_hidden(
                first.ciphers, crypto_context=context, layout=feature
            ),
            expected0,
            atol=5e-2,
            rtol=5e-2,
        )
        second = model.decode(
            h1,
            crypto_context=context,
            input_ciphers=ciphers1,
            layers=(layer,),
            schedule=schedule,
            skip_attention=False,
            kv_caches=first.kv_caches,
            occupied=first.occupied,
            position=first.position,
            verify=True,
            expected_output=expected1,
            consume_kv_cache=consume_kv_cache,
        )
        assert second.occupied == 2
        assert second.position == 2
        assert second.per_layer[0]["max_abs_diff"] is not None
        assert second.per_layer[0]["max_abs_diff"] < 5e-2
        np.testing.assert_allclose(
            decrypt_hidden(
                second.ciphers, crypto_context=context, layout=feature
            ),
            expected1,
            atol=5e-2,
            rtol=5e-2,
        )
    finally:
        if second is not None:
            second.release()
        if first is not None:
            first.release()
        for cipher in (*ciphers0, *ciphers1):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_decode_model_prefix_bundle_matches_poly_reference() -> None:
    """Model/graph wiring keeps Prefix PT separate from the content cache."""

    rng = np.random.default_rng(26091733)
    hidden_dim = 32
    feature, attention = _toy_decode_layouts(hidden_dim)
    h0 = rng.normal(scale=0.05, size=(1, hidden_dim)).astype(np.float32)
    h1 = rng.normal(scale=0.05, size=(1, hidden_dim)).astype(np.float32)
    weights = _toy_decode_weights(rng, hidden_dim)
    approximations = _fit_decode_approximations(
        np.vstack([h0, h1]),
        weights=weights,
        feature_layout=feature,
        attention_layout=attention,
    )
    prefix_k = 2
    prefix_key = rng.normal(scale=0.02, size=(prefix_k, 16)).astype(np.float32)
    prefix_value = rng.normal(scale=0.02, size=(prefix_k, 16)).astype(np.float32)
    prefix_bundle = PrefixKVBundle.from_logical(
        prefix_key, prefix_value, layout=attention
    )
    expected = poly_decode_layer_numpy(
        h0,
        cache_key=np.empty((0, 16), dtype=np.float32),
        cache_value=np.empty((0, 16), dtype=np.float32),
        weights=weights,
        approximations=approximations,
        attention_layout=attention,
        position=prefix_k,
        prefix_bundle=prefix_bundle,
        use_alg2_softmax=True,
    ).hidden

    schedule = _small_decode_schedule()
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 64
    cfg.simulator.device = "cuda"
    layer = TransformerLayer(
        index=0,
        weights=weights,
        approximations=approximations,
        config=cfg,
        feature_layout=feature,
        attention_layout=attention,
    )
    model = Llama3Model(
        config=cfg,
        num_layers=1,
        feature_layout=feature,
        attention_layout=attention,
    )
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(feature.tokens_per_cipher),
        rotations=model.required_decode_rotations(
            schedule, skip_attention=False
        ),
        include_default_rotations=True,
    )
    ciphers = tuple(
        context.encrypt(row, level=0, slots=feature.slots)
        for row in feature.pack(h0)
    )
    result = None
    try:
        result = model.decode(
            h0,
            crypto_context=context,
            input_ciphers=ciphers,
            layers=(layer,),
            schedule=schedule,
            skip_attention=False,
            occupied=0,
            position=prefix_k,
            prefix_bundles=(prefix_bundle,),
            verify=True,
            expected_output=expected,
        )
        assert result.per_layer[0]["max_abs_diff"] is not None
        assert result.per_layer[0]["max_abs_diff"] < 8e-2
        np.testing.assert_allclose(
            decrypt_hidden(result.ciphers, crypto_context=context, layout=feature),
            expected,
            atol=8e-2,
            rtol=8e-2,
        )
        assert result.occupied == 1
        assert result.position == prefix_k + 1
    finally:
        if result is not None:
            result.release()
        for cipher in ciphers:
            release_if_supported(cipher)


@pytest.mark.gpu
def test_decode_model_next_n_tokens_kv_cache_matches_numpy() -> None:
    rng = np.random.default_rng(26091721)
    hidden_dim = 32
    n_tokens = 4
    feature, attention = _toy_decode_layouts(hidden_dim)
    hiddens = rng.normal(scale=0.1, size=(n_tokens, hidden_dim)).astype(np.float32)
    weights = _toy_decode_weights(rng, hidden_dim)
    approximations = _fit_decode_approximations(
        hiddens,
        weights=weights,
        feature_layout=feature,
        attention_layout=attention,
    )
    expected = _sequential_decode_numpy(
        hiddens,
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
    )
    schedule = _small_decode_schedule()
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = 64
    cfg.simulator.device = "cuda"
    layer = TransformerLayer(
        index=0,
        weights=weights,
        approximations=approximations,
        config=cfg,
        feature_layout=feature,
        attention_layout=attention,
    )
    model = Llama3Model(
        config=cfg,
        num_layers=1,
        feature_layout=feature,
        attention_layout=attention,
    )
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(feature.tokens_per_cipher),
        rotations=model.required_decode_rotations(
            schedule, skip_attention=False
        ),
        include_default_rotations=True,
    )
    encrypted = [
        tuple(
            context.encrypt(row, level=0, slots=feature.slots)
            for row in feature.pack(hiddens[index : index + 1])
        )
        for index in range(n_tokens)
    ]
    previous = None
    try:
        occupied = 0
        position = 0
        caches = None
        for index, hidden in enumerate(hiddens):
            hidden = hidden.reshape(1, -1)
            result = model.decode(
                hidden,
                crypto_context=context,
                input_ciphers=encrypted[index],
                layers=(layer,),
                schedule=schedule,
                skip_attention=False,
                kv_caches=caches,
                occupied=occupied,
                position=position,
                verify=True,
                expected_output=expected[index],
            )
            assert result.occupied == index + 1
            assert result.position == index + 1
            assert result.per_layer[0]["max_abs_diff"] is not None
            assert result.per_layer[0]["max_abs_diff"] < 5e-2
            np.testing.assert_allclose(
                decrypt_hidden(
                    result.ciphers, crypto_context=context, layout=feature
                ),
                expected[index],
                atol=5e-2,
                rtol=5e-2,
            )
            caches = result.kv_caches
            occupied = result.occupied
            position = result.position
            if previous is not None:
                previous.release()
            previous = result
        with pytest.raises(ValueError, match="full"):
            model.decode(
                hiddens[-1:].copy(),
                crypto_context=context,
                input_ciphers=encrypted[-1],
                layers=(layer,),
                schedule=schedule,
                kv_caches=caches,
                occupied=occupied,
                position=position,
            )
    finally:
        if previous is not None:
            previous.release()
        for ciphers in encrypted:
            for cipher in ciphers:
                release_if_supported(cipher)


def _append_c_token(pages, token_c, *, seq_len: int, occupied: int):
    local = int(occupied) % int(seq_len)
    token_c = np.asarray(token_c)
    if local == 0:
        return list(pages) + [token_c]
    pages = [np.asarray(page) for page in pages]
    pages[-1] = pages[-1] + token_c
    return pages


def _occupied_pages(occupied_after: int, *, seq_len: int, n_pages: int) -> list[int]:
    remaining = int(occupied_after)
    pages = []
    for _ in range(int(n_pages)):
        take = min(int(seq_len), remaining)
        pages.append(take)
        remaining -= take
    return pages


def _prefill_c_pages_numpy(
    hiddens: np.ndarray,
    *,
    weights: LayerWeights,
    rmsn1: dict,
    attention_layout: AttentionPairLayout,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    attn_in = rmsnorm_poly_numpy(
        hiddens, rmsn1["coefficients"], rmsn1["fit_interval"]
    )
    key_pages: list[np.ndarray] = []
    value_pages: list[np.ndarray] = []
    for token, row in enumerate(attn_in):
        prepared = decode_prepare_qkv_numpy(
            row.reshape(1, -1),
            weights.q_proj,
            weights.k_proj,
            weights.v_proj,
            layout=attention_layout,
            position=int(token),
            token_index=int(token) % int(attention_layout.seq_len),
        )
        key_pages = _append_c_token(
            key_pages,
            prepared.key_token_c,
            seq_len=int(attention_layout.seq_len),
            occupied=int(token),
        )
        value_pages = _append_c_token(
            value_pages,
            prepared.value_token_c,
            seq_len=int(attention_layout.seq_len),
            occupied=int(token),
        )
    return key_pages, value_pages


def _decode_attn_residual_with_pages_numpy(
    hidden: np.ndarray,
    *,
    weights: LayerWeights,
    approximations: LayerApproximations,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    occupied: int,
    position: int,
    key_pages: list[np.ndarray],
    value_pages: list[np.ndarray],
):
    attn_in = rmsnorm_poly_numpy(
        hidden,
        approximations.input_norm["coefficients"],
        approximations.input_norm["fit_interval"],
    )
    local = int(occupied) % int(attention_layout.seq_len)
    prepared = decode_prepare_qkv_numpy(
        attn_in,
        weights.q_proj,
        weights.k_proj,
        weights.v_proj,
        layout=attention_layout,
        position=int(position),
        token_index=local,
    )
    key_pages = _append_c_token(
        key_pages,
        prepared.key_token_c,
        seq_len=int(attention_layout.seq_len),
        occupied=int(occupied),
    )
    value_pages = _append_c_token(
        value_pages,
        prepared.value_token_c,
        seq_len=int(attention_layout.seq_len),
        occupied=int(occupied),
    )
    occupied_after = int(occupied) + 1
    _, _, attn_out = decode_attention_alg2_pages_numpy(
        prepared.query,
        key_pages,
        value_pages,
        layout=attention_layout,
        occupied_pages=_occupied_pages(
            occupied_after,
            seq_len=int(attention_layout.seq_len),
            n_pages=len(key_pages),
        ),
        config=approximations.softmax,
    )
    packed_fm = output_c_to_decode_fm_numpy(
        attn_out,
        attention_layout=attention_layout,
        feature_layout=feature_layout,
    )
    wo = decode_w_o_numpy(
        feature_layout.unpack(packed_fm),
        weights.o_proj,
        feature_layout=feature_layout,
    )
    return hidden + wo, key_pages, value_pages


def _decode_layer_with_pages_numpy(
    hidden: np.ndarray,
    *,
    weights: LayerWeights,
    approximations: LayerApproximations,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    occupied: int,
    position: int,
    key_pages: list[np.ndarray],
    value_pages: list[np.ndarray],
):
    residual, key_pages, value_pages = _decode_attn_residual_with_pages_numpy(
        hidden,
        weights=weights,
        approximations=approximations,
        feature_layout=feature_layout,
        attention_layout=attention_layout,
        occupied=occupied,
        position=position,
        key_pages=key_pages,
        value_pages=value_pages,
    )
    mlp_in = rmsnorm_poly_numpy(
        residual,
        approximations.post_attention_norm["coefficients"],
        approximations.post_attention_norm["fit_interval"],
    )
    silu = approximations.silu
    lo, hi = (float(value) for value in silu["fit_interval"])
    gate = mlp_in @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = mlp_in @ np.asarray(weights.up_proj, dtype=np.float32).T
    activated = eval_chebyshev_series_direct(
        gate, silu["coefficients"], lo, hi
    ) * up
    hidden_out = residual + activated @ np.asarray(
        weights.down_proj, dtype=np.float32
    ).T
    return hidden_out, residual, key_pages, value_pages


def test_prefill128_numpy_decode_uses_second_kv_page() -> None:
    rng = np.random.default_rng(26091723)
    hidden_dim = 1024
    prefill_len = 128
    n_decode = 2
    feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=32768
    )
    attention = AttentionPairLayout(
        seq_len=128,
        head_dim=128,
        query_heads=8,
        key_value_heads=4,
        slots=32768,
    )
    prefill = rng.normal(scale=0.03, size=(prefill_len, hidden_dim)).astype(
        np.float32
    )
    decode_h = rng.normal(scale=0.03, size=(n_decode, hidden_dim)).astype(
        np.float32
    )
    weights = LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        k_proj=rng.normal(scale=0.005, size=(512, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.005, size=(512, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.005, size=(1536, hidden_dim)).astype(
            np.float32
        ),
        up_proj=rng.normal(scale=0.005, size=(1536, hidden_dim)).astype(np.float32),
        down_proj=rng.normal(scale=0.005, size=(hidden_dim, 1536)).astype(
            np.float32
        ),
    )
    rmsn1 = _rmsn_entry(np.vstack([prefill, decode_h]))
    softmax = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.02, 256.0)
    )
    key_pages, value_pages = _prefill_c_pages_numpy(
        prefill, weights=weights, rmsn1=rmsn1, attention_layout=attention
    )
    assert len(key_pages) == 1
    residuals = []
    pages_k, pages_v = key_pages, value_pages
    for index, hidden in enumerate(decode_h):
        hidden = hidden.reshape(1, -1)
        residual, pages_k, pages_v = _decode_attn_residual_with_pages_numpy(
            hidden,
            weights=weights,
            approximations=LayerApproximations(
                input_norm=rmsn1,
                post_attention_norm=rmsn1,
                silu=_toy_decode_silu(),
                softmax=softmax,
            ),
            feature_layout=feature,
            attention_layout=attention,
            occupied=prefill_len + index,
            position=prefill_len + index,
            key_pages=pages_k,
            value_pages=pages_v,
        )
        residuals.append(residual)
    assert len(pages_k) == 2
    rmsn2 = _rmsn_entry(np.vstack(residuals))
    approximations = LayerApproximations(
        input_norm=rmsn1,
        post_attention_norm=rmsn2,
        silu=_toy_decode_silu(),
        softmax=softmax,
    )
    pages_k, pages_v = _prefill_c_pages_numpy(
        prefill, weights=weights, rmsn1=rmsn1, attention_layout=attention
    )
    with_cache, _, pages_k, pages_v = _decode_layer_with_pages_numpy(
        decode_h[:1],
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
        occupied=prefill_len,
        position=prefill_len,
        key_pages=pages_k,
        value_pages=pages_v,
    )
    without_cache, _, _, _ = _decode_layer_with_pages_numpy(
        decode_h[:1],
        weights=weights,
        approximations=approximations,
        feature_layout=feature,
        attention_layout=attention,
        occupied=0,
        position=prefill_len,
        key_pages=[],
        value_pages=[],
    )
    assert not np.allclose(with_cache, without_cache, atol=1e-5)


@pytest.mark.gpu
def test_prefill128_decode_n_tokens_kv_cache_matches_numpy() -> None:
    rng = np.random.default_rng(26091724)
    hidden_dim = 1024
    prefill_len = 128
    n_decode = 4
    feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=32768
    )
    attention = AttentionPairLayout(
        seq_len=128,
        head_dim=128,
        query_heads=8,
        key_value_heads=4,
        slots=32768,
    )
    prefill = rng.normal(scale=0.03, size=(prefill_len, hidden_dim)).astype(
        np.float32
    )
    decode_h = rng.normal(scale=0.03, size=(n_decode, hidden_dim)).astype(
        np.float32
    )
    weights = LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        k_proj=rng.normal(scale=0.005, size=(512, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.005, size=(512, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.005, size=(1536, hidden_dim)).astype(
            np.float32
        ),
        up_proj=rng.normal(scale=0.005, size=(1536, hidden_dim)).astype(np.float32),
        down_proj=rng.normal(scale=0.005, size=(hidden_dim, 1536)).astype(
            np.float32
        ),
    )
    rmsn1 = _rmsn_entry(np.vstack([prefill, decode_h]))
    softmax = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.02, 256.0)
    )
    key_pages, value_pages = _prefill_c_pages_numpy(
        prefill, weights=weights, rmsn1=rmsn1, attention_layout=attention
    )
    residuals = []
    pages_k, pages_v = [np.asarray(page) for page in key_pages], [
        np.asarray(page) for page in value_pages
    ]
    expected = []
    scratch_k, scratch_v = pages_k, pages_v
    for index, hidden in enumerate(decode_h):
        hidden = hidden.reshape(1, -1)
        residual, scratch_k, scratch_v = _decode_attn_residual_with_pages_numpy(
            hidden,
            weights=weights,
            approximations=LayerApproximations(
                input_norm=rmsn1,
                post_attention_norm=rmsn1,
                silu=_toy_decode_silu(),
                softmax=softmax,
            ),
            feature_layout=feature,
            attention_layout=attention,
            occupied=prefill_len + index,
            position=prefill_len + index,
            key_pages=scratch_k,
            value_pages=scratch_v,
        )
        residuals.append(residual)
    approximations = LayerApproximations(
        input_norm=rmsn1,
        post_attention_norm=_rmsn_entry(np.vstack(residuals)),
        silu=_toy_decode_silu(),
        softmax=softmax,
    )
    pages_k, pages_v = _prefill_c_pages_numpy(
        prefill, weights=weights, rmsn1=rmsn1, attention_layout=attention
    )
    for index, hidden in enumerate(decode_h):
        hidden = hidden.reshape(1, -1)
        out, _, pages_k, pages_v = _decode_layer_with_pages_numpy(
            hidden,
            weights=weights,
            approximations=approximations,
            feature_layout=feature,
            attention_layout=attention,
            occupied=prefill_len + index,
            position=prefill_len + index,
            key_pages=pages_k,
            value_pages=pages_v,
        )
        expected.append(out)
    schedule = LayerSchedule(
        linear_baby_steps=16,
        linear_anchor_step=16,
        fuse_rmsnorm_output_scale=False,
        fuse_silu_domain_map=False,
        evict_rotation_cuda_cache_before_output_projection=True,
        evict_rotation_cuda_cache_before_mlp=True,
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 16
    cfg.simulator.maxLevelsRemaining = 37
    cfg.simulator.device = "cuda"
    layer = TransformerLayer(
        index=0,
        weights=weights,
        approximations=approximations,
        config=cfg,
        feature_layout=feature,
        attention_layout=attention,
    )
    model = Llama3Model(
        config=cfg,
        num_layers=1,
        feature_layout=feature,
        attention_layout=attention,
    )
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(feature.tokens_per_cipher),
        rotations=model.required_decode_rotations(
            schedule, skip_attention=False, include_bootstrap=True
        ),
        include_default_rotations=True,
    )
    cache_k = tuple(
        context.encrypt(row, level=0, slots=attention.slots)
        for row in key_pages[0]
    )
    cache_v = tuple(
        context.encrypt(row, level=0, slots=attention.slots)
        for row in value_pages[0]
    )
    encrypted = [
        tuple(
            context.encrypt(row, level=0, slots=feature.slots)
            for row in feature.pack(decode_h[index : index + 1])
        )
        for index in range(n_decode)
    ]
    previous = None
    bootstrap = None
    try:
        bootstrap = prepare_bootstrap_operator(context, schedule)
        caches = [(cache_k, cache_v)]
        occupied = prefill_len
        position = prefill_len
        for index, hidden in enumerate(decode_h):
            hidden = hidden.reshape(1, -1)
            result = model.decode(
                hidden,
                crypto_context=context,
                input_ciphers=encrypted[index],
                layers=(layer,),
                schedule=schedule,
                bootstrap_operator=bootstrap,
                refresh_post_attention=True,
                skip_attention=False,
                kv_caches=caches,
                occupied=occupied,
                position=position,
                verify=True,
                expected_output=expected[index],
                max_kv_pages=2,
            )
            assert result.occupied == occupied + 1
            assert len(result.kv_caches[0][0]) == 2 * int(
                attention.key_value_pair_count
            )
            assert result.per_layer[0]["max_abs_diff"] is not None
            assert result.per_layer[0]["max_abs_diff"] < 8e-2
            assert result.per_layer[0]["residual_bootstrap_calls"] >= 1
            assert result.per_layer[0]["softmax_bootstrap_calls"] >= 1
            assert (
                "post_attention_bootstrap_rmsnorm"
                in result.per_layer[0]["stage_seconds"]
            )
            caches = result.kv_caches
            occupied = result.occupied
            position = result.position
            if previous is not None:
                previous.release()
            previous = result
    finally:
        if previous is not None:
            previous.release()
        if bootstrap is not None:
            bootstrap.release()
        for cipher in (*cache_k, *cache_v):
            release_if_supported(cipher)
        for ciphers in encrypted:
            for cipher in ciphers:
                release_if_supported(cipher)


@pytest.mark.gpu
def test_prefill128_decode_hidden4096_with_bootstrap() -> None:
    rng = np.random.default_rng(26091725)
    hidden_dim = 4096
    intermediate = 14336
    prefill_len = 128
    n_decode = 2
    feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=32768
    )
    attention = AttentionPairLayout()
    prefill = rng.normal(scale=0.03, size=(prefill_len, hidden_dim)).astype(
        np.float32
    )
    decode_h = rng.normal(scale=0.03, size=(n_decode, hidden_dim)).astype(
        np.float32
    )
    weights = LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        k_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.005, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        up_proj=rng.normal(scale=0.005, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        down_proj=rng.normal(scale=0.005, size=(hidden_dim, intermediate)).astype(
            np.float32
        ),
    )
    rmsn1 = _rmsn_entry(np.vstack([prefill, decode_h]))
    softmax = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.02, 256.0)
    )
    key_pages, value_pages = _prefill_c_pages_numpy(
        prefill, weights=weights, rmsn1=rmsn1, attention_layout=attention
    )
    residuals = []
    expected = []
    scratch_k, scratch_v = [np.asarray(page) for page in key_pages], [
        np.asarray(page) for page in value_pages
    ]
    for index, hidden in enumerate(decode_h):
        hidden = hidden.reshape(1, -1)
        residual, scratch_k, scratch_v = _decode_attn_residual_with_pages_numpy(
            hidden,
            weights=weights,
            approximations=LayerApproximations(
                input_norm=rmsn1,
                post_attention_norm=rmsn1,
                silu=_toy_decode_silu(),
                softmax=softmax,
            ),
            feature_layout=feature,
            attention_layout=attention,
            occupied=prefill_len + index,
            position=prefill_len + index,
            key_pages=scratch_k,
            value_pages=scratch_v,
        )
        residuals.append(residual)
    approximations = LayerApproximations(
        input_norm=rmsn1,
        post_attention_norm=_rmsn_entry(np.vstack(residuals)),
        silu=_toy_decode_silu(),
        softmax=softmax,
    )
    pages_k, pages_v = _prefill_c_pages_numpy(
        prefill, weights=weights, rmsn1=rmsn1, attention_layout=attention
    )
    for index, hidden in enumerate(decode_h):
        hidden = hidden.reshape(1, -1)
        out, _, pages_k, pages_v = _decode_layer_with_pages_numpy(
            hidden,
            weights=weights,
            approximations=approximations,
            feature_layout=feature,
            attention_layout=attention,
            occupied=prefill_len + index,
            position=prefill_len + index,
            key_pages=pages_k,
            value_pages=pages_v,
        )
        expected.append(out)
    schedule = LayerSchedule(
        linear_baby_steps=64,
        linear_anchor_step=64,
        fuse_rmsnorm_output_scale=False,
        fuse_silu_domain_map=False,
        evict_rotation_cuda_cache_before_output_projection=True,
        evict_rotation_cuda_cache_before_mlp=True,
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 16
    cfg.simulator.maxLevelsRemaining = 37
    cfg.simulator.device = "cuda"
    layer = TransformerLayer(
        index=0,
        weights=weights,
        approximations=approximations,
        config=cfg,
        feature_layout=feature,
        attention_layout=attention,
    )
    model = Llama3Model(
        config=cfg,
        num_layers=1,
        feature_layout=feature,
        attention_layout=attention,
    )
    rotations = model.required_decode_rotations(
        schedule, skip_attention=False, include_bootstrap=True
    )
    print(
        f"hidden=4096 rotations={len(rotations)} n_decode={n_decode} "
        f"q_pairs={attention.query_pair_count} "
        f"kv_pairs={attention.key_value_pair_count}",
        flush=True,
    )
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(feature.tokens_per_cipher),
        rotations=rotations,
        include_default_rotations=True,
    )
    print(f"keygen {context.keygen_wall_time_s:.1f}s", flush=True)
    cache_k = tuple(
        context.encrypt(row, level=0, slots=attention.slots)
        for row in key_pages[0]
    )
    cache_v = tuple(
        context.encrypt(row, level=0, slots=attention.slots)
        for row in value_pages[0]
    )
    encrypted = [
        tuple(
            context.encrypt(row, level=0, slots=feature.slots)
            for row in feature.pack(decode_h[index : index + 1])
        )
        for index in range(n_decode)
    ]
    previous = None
    bootstrap = None
    try:
        bootstrap = prepare_bootstrap_operator(context, schedule)
        print(f"bootstrap_setup {bootstrap.setup_seconds:.1f}s", flush=True)
        caches = [(cache_k, cache_v)]
        occupied = prefill_len
        position = prefill_len
        for index, hidden in enumerate(decode_h):
            hidden = hidden.reshape(1, -1)
            result = model.decode(
                hidden,
                crypto_context=context,
                input_ciphers=encrypted[index],
                layers=(layer,),
                schedule=schedule,
                bootstrap_operator=bootstrap,
                refresh_post_attention=True,
                skip_attention=False,
                kv_caches=caches,
                occupied=occupied,
                position=position,
                verify=True,
                expected_output=expected[index],
                max_kv_pages=2,
            )
            stages = result.per_layer[0]["stage_seconds"]
            print(
                f"token {index} occupied={occupied} "
                f"wall={result.wall_seconds:.1f}s "
                f"kernel={result.per_layer[0]['kernel_wall_seconds']:.1f}s "
                f"softmax_bs={result.per_layer[0]['softmax_bootstrap_calls']} "
                f"residual_bs={result.per_layer[0]['residual_bootstrap_calls']} "
                f"max_abs={result.per_layer[0]['max_abs_diff']:.3e} "
                f"stages={{{', '.join(f'{k}={v:.2f}' for k, v in stages.items())}}}",
                flush=True,
            )
            assert result.occupied == occupied + 1
            assert len(result.kv_caches[0][0]) == 2 * int(
                attention.key_value_pair_count
            )
            assert result.per_layer[0]["max_abs_diff"] is not None
            assert result.per_layer[0]["max_abs_diff"] < 8e-2
            assert result.per_layer[0]["residual_bootstrap_calls"] >= 1
            caches = result.kv_caches
            occupied = result.occupied
            position = result.position
            if previous is not None:
                previous.release()
            previous = result
    finally:
        if previous is not None:
            previous.release()
        if bootstrap is not None:
            bootstrap.release()
        for cipher in (*cache_k, *cache_v):
            release_if_supported(cipher)
        for ciphers in encrypted:
            for cipher in ciphers:
                release_if_supported(cipher)


@pytest.mark.gpu
def test_prefill_layer0_emits_c_cache_for_decode() -> None:
    """Prefill Layer0 emit_cache_c → Decode, no numpy-encrypted fake cache."""

    rng = np.random.default_rng(26091726)
    hidden_dim = 4096
    intermediate = 14336
    prefill_len = 128
    n_decode = 1
    prefill_feature = FeatureMajorPrefillLayout()
    decode_feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=32768
    )
    attention = AttentionPairLayout()
    prefill = rng.normal(scale=0.03, size=(prefill_len, hidden_dim)).astype(
        np.float32
    )
    decode_h = rng.normal(scale=0.03, size=(n_decode, hidden_dim)).astype(
        np.float32
    )
    weights = LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        k_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.005, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        up_proj=rng.normal(scale=0.005, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        down_proj=rng.normal(scale=0.005, size=(hidden_dim, intermediate)).astype(
            np.float32
        ),
    )
    rmsn1 = _rmsn_entry(np.vstack([prefill, decode_h]))
    softmax = DecodeAlg2Poly.synthetic(
        k=1, degree=8, invsqrt_interval=(0.02, 256.0)
    )
    key_pages, value_pages = _prefill_c_pages_numpy(
        prefill, weights=weights, rmsn1=rmsn1, attention_layout=attention
    )
    residuals = []
    scratch_k, scratch_v = [np.asarray(page) for page in key_pages], [
        np.asarray(page) for page in value_pages
    ]
    for index, hidden in enumerate(decode_h):
        hidden = hidden.reshape(1, -1)
        residual, scratch_k, scratch_v = _decode_attn_residual_with_pages_numpy(
            hidden,
            weights=weights,
            approximations=LayerApproximations(
                input_norm=rmsn1,
                post_attention_norm=rmsn1,
                silu=_toy_decode_silu(),
                softmax=softmax,
            ),
            feature_layout=decode_feature,
            attention_layout=attention,
            occupied=prefill_len + index,
            position=prefill_len + index,
            key_pages=scratch_k,
            value_pages=scratch_v,
        )
        residuals.append(residual)
    approximations = LayerApproximations(
        input_norm=rmsn1,
        post_attention_norm=_rmsn_entry(np.vstack(residuals)),
        silu=_toy_decode_silu(),
        softmax=softmax,
    )
    pages_k, pages_v = _prefill_c_pages_numpy(
        prefill, weights=weights, rmsn1=rmsn1, attention_layout=attention
    )
    expected = []
    for index, hidden in enumerate(decode_h):
        hidden = hidden.reshape(1, -1)
        out, _, pages_k, pages_v = _decode_layer_with_pages_numpy(
            hidden,
            weights=weights,
            approximations=approximations,
            feature_layout=decode_feature,
            attention_layout=attention,
            occupied=prefill_len + index,
            position=prefill_len + index,
            key_pages=pages_k,
            value_pages=pages_v,
        )
        expected.append(out)
    kv_limbs = 12
    prefill_schedule = LayerSchedule(
        evict_rotation_cuda_cache_before_output_projection=True,
        evict_rotation_cuda_cache_before_mlp=True,
        kv_limbs=kv_limbs,
    )
    decode_schedule = LayerSchedule(
        linear_baby_steps=64,
        linear_anchor_step=64,
        fuse_rmsnorm_output_scale=False,
        fuse_silu_domain_map=False,
        evict_rotation_cuda_cache_before_output_projection=True,
        evict_rotation_cuda_cache_before_mlp=True,
        kv_limbs=kv_limbs,
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 16
    cfg.simulator.maxLevelsRemaining = 37
    cfg.simulator.device = "cuda"
    prefill_layer = TransformerLayer(
        index=0,
        weights=weights,
        approximations=approximations,
        config=cfg,
        feature_layout=prefill_feature,
        attention_layout=attention,
    )
    decode_layer = TransformerLayer(
        index=0,
        weights=weights,
        approximations=approximations,
        config=cfg,
        feature_layout=decode_feature,
        attention_layout=attention,
    )
    decode_model = Llama3Model(
        config=cfg,
        num_layers=1,
        feature_layout=decode_feature,
        attention_layout=attention,
    )
    rotations = tuple(
        sorted(
            set(
                required_rotations(
                    prefill_feature,
                    attention,
                    schedule=prefill_schedule,
                    emit_kv_cache=True,
                )
            )
            | set(
                decode_layer_rotations(
                    decode_feature,
                    attention,
                    schedule=decode_schedule,
                    skip_attention=False,
                    include_bootstrap=True,
                )
            )
        )
    )
    print(
        f"layer0 emit-cache rotations={len(rotations)} n_decode={n_decode}",
        flush=True,
    )
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(prefill_feature.tokens_per_cipher),
        rotations=rotations,
        include_default_rotations=True,
    )
    print(f"keygen {context.keygen_wall_time_s:.1f}s", flush=True)
    prefill_ciphers = tuple(
        context.encrypt(row, level=0, slots=prefill_feature.slots)
        for row in prefill_feature.pack(prefill)
    )
    decode_ciphers = tuple(
        context.encrypt(row, level=0, slots=decode_feature.slots)
        for row in decode_feature.pack(decode_h[:1])
    )
    prefill_result = decode_result = None
    bootstrap = None
    cache_k = cache_v = ()
    try:
        bootstrap = prepare_bootstrap_operator(context, prefill_schedule)
        print(f"bootstrap_setup {bootstrap.setup_seconds:.1f}s", flush=True)
        prefill_result = prefill_layer.forward(
            prefill,
            crypto_context=context,
            input_ciphers=prefill_ciphers,
            bootstrap_operator=bootstrap,
            schedule=prefill_schedule,
            emit_kv_cache=True,
        )
        cache_k = tuple(prefill_result.key_c_ciphers)
        cache_v = tuple(prefill_result.value_c_ciphers)
        prefill_result.key_c_ciphers = ()
        prefill_result.value_c_ciphers = ()
        print(
            f"prefill wall={prefill_result.wall_seconds:.1f}s "
            f"k_c={len(cache_k)} v_c={len(cache_v)} "
            f"k_level={context.level_for_cipher(cache_k[0])} "
            f"k_limbs={cache_k[0].state.cur_limbs} "
            f"v_level={context.level_for_cipher(cache_v[0])} "
            f"v_limbs={cache_v[0].state.cur_limbs}",
            flush=True,
        )
        assert len(cache_k) == int(attention.key_value_pair_count)
        assert len(cache_v) == int(attention.key_value_pair_count)
        assert cache_k[0].state.cur_limbs == kv_limbs
        assert cache_v[0].state.cur_limbs == kv_limbs
        assert all(int(cipher.state.cur_limbs) == kv_limbs for cipher in cache_k)
        assert all(int(cipher.state.cur_limbs) == kv_limbs for cipher in cache_v)
        prefill_result.release()
        prefill_result = None
        for cipher in prefill_ciphers:
            release_if_supported(cipher)
        prefill_ciphers = ()
        trim_device_allocator_cache(context.device)
        decode_result = decode_model.decode(
            decode_h[:1],
            crypto_context=context,
            input_ciphers=decode_ciphers,
            layers=(decode_layer,),
            schedule=decode_schedule,
            bootstrap_operator=bootstrap,
            refresh_post_attention=True,
            skip_attention=False,
            kv_caches=[(cache_k, cache_v)],
            occupied=prefill_len,
            position=prefill_len,
            verify=True,
            expected_output=expected[0],
            max_kv_pages=2,
        )
        stages = decode_result.per_layer[0]["stage_seconds"]
        print(
            f"decode occupied={prefill_len} "
            f"wall={decode_result.wall_seconds:.1f}s "
            f"softmax_bs={decode_result.per_layer[0]['softmax_bootstrap_calls']} "
            f"residual_bs={decode_result.per_layer[0]['residual_bootstrap_calls']} "
            f"max_abs={decode_result.per_layer[0]['max_abs_diff']:.3e} "
            f"stages={{{', '.join(f'{k}={v:.2f}' for k, v in stages.items())}}}",
            flush=True,
        )
        assert decode_result.occupied == prefill_len + 1
        decode_k, decode_v = decode_result.kv_caches[0]
        assert all(int(cipher.state.cur_limbs) == kv_limbs for cipher in decode_k)
        assert all(int(cipher.state.cur_limbs) == kv_limbs for cipher in decode_v)
        assert decode_result.per_layer[0]["max_abs_diff"] is not None
        assert decode_result.per_layer[0]["max_abs_diff"] < 8e-2
        assert decode_result.per_layer[0]["softmax_bootstrap_calls"] >= 1
    finally:
        if prefill_result is not None:
            prefill_result.release()
        if decode_result is not None:
            decode_result.release()
        if bootstrap is not None:
            bootstrap.release()
        for cipher in (*cache_k, *cache_v, *prefill_ciphers, *decode_ciphers):
            release_if_supported(cipher)


@pytest.mark.gpu
def test_prefill_two_layer_emits_c_cache_for_decode() -> None:
    """Prefill emit through Llama3Model.forward, then 2-layer Decode.

    Nonlinear polys start from the Prefill artifacts.  Random-weight smokes
    fall outside those RMSN intervals, so the plaintext stack is sampled and
    the same Prefill degrees are refit.  Softmax Alg2 coefficients are reused
    as-is; Decode already bootstraps exp and each λ round.
    """

    rng = np.random.default_rng(26091727)
    num_layers = 2
    hidden_dim = 4096
    intermediate = 14336
    prefill_len = 128
    kv_limbs = 12
    prefill_feature = FeatureMajorPrefillLayout()
    decode_feature = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=hidden_dim, slots=32768
    )
    attention = AttentionPairLayout()
    prefill = rng.normal(scale=0.03, size=(prefill_len, hidden_dim)).astype(
        np.float32
    )
    decode_h = rng.normal(scale=0.03, size=(1, hidden_dim)).astype(np.float32)
    weights = LayerWeights(
        input_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        q_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        k_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        v_proj=rng.normal(scale=0.005, size=(1024, hidden_dim)).astype(np.float32),
        o_proj=rng.normal(scale=0.005, size=(hidden_dim, hidden_dim)).astype(
            np.float32
        ),
        post_attention_layernorm=np.zeros((hidden_dim,), dtype=np.float32),
        gate_proj=rng.normal(scale=0.005, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        up_proj=rng.normal(scale=0.005, size=(intermediate, hidden_dim)).astype(
            np.float32
        ),
        down_proj=rng.normal(scale=0.005, size=(hidden_dim, intermediate)).astype(
            np.float32
        ),
    )
    prefill_config = ApproximationConfig(
        rmsnorm_path=DEFAULT_RMS,
        silu_path=DEFAULT_SILU,
        softmax_path=DEFAULT_SOFTMAX,
    )
    layer_approx, fit_reports = fit_stack_from_plaintext(
        prefill,
        (weights, weights),
        attention_layout=attention,
        decode_hidden=decode_h,
        prefill_config=prefill_config,
    )
    for report in fit_reports:
        print(
            f"poly L{report.layer} reused={report.reused} degrees={report.degrees}",
            flush=True,
        )
    prefill_schedule = LayerSchedule(
        evict_rotation_cuda_cache_before_output_projection=True,
        evict_rotation_cuda_cache_before_mlp=True,
        kv_limbs=kv_limbs,
    )
    decode_schedule = LayerSchedule(
        linear_baby_steps=64,
        linear_anchor_step=64,
        evict_rotation_cuda_cache_before_output_projection=True,
        evict_rotation_cuda_cache_before_mlp=True,
        kv_limbs=kv_limbs,
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 16
    cfg.simulator.maxLevelsRemaining = 37
    cfg.simulator.device = "cuda"
    prefill_layers = tuple(
        TransformerLayer(
            index=index,
            weights=weights,
            approximations=layer_approx[index],
            config=cfg,
            feature_layout=prefill_feature,
            attention_layout=attention,
        )
        for index in range(num_layers)
    )
    decode_layers = tuple(
        TransformerLayer(
            index=index,
            weights=weights,
            approximations=layer_approx[index],
            config=cfg,
            feature_layout=decode_feature,
            attention_layout=attention,
        )
        for index in range(num_layers)
    )
    prefill_model = Llama3Model(
        config=cfg,
        num_layers=num_layers,
        feature_layout=prefill_feature,
        attention_layout=attention,
    )
    decode_model = Llama3Model(
        config=cfg,
        num_layers=num_layers,
        feature_layout=decode_feature,
        attention_layout=attention,
    )
    rotations = tuple(
        sorted(
            set(
                prefill_model.required_rotations(
                    prefill_schedule, emit_kv_cache=True
                )
            )
            | set(
                decode_model.required_decode_rotations(
                    decode_schedule,
                    skip_attention=False,
                    include_bootstrap=True,
                )
            )
        )
    )
    print(
        f"two-layer emit-cache rotations={len(rotations)} layers={num_layers}",
        flush=True,
    )
    context = create_sim_context(
        cfg.simulator,
        input_dim=hidden_dim,
        block_out=int(prefill_feature.tokens_per_cipher),
        rotations=rotations,
        include_default_rotations=True,
    )
    print(f"keygen {context.keygen_wall_time_s:.1f}s", flush=True)
    prefill_ciphers = tuple(
        context.encrypt(row, level=0, slots=prefill_feature.slots)
        for row in prefill_feature.pack(prefill)
    )
    decode_ciphers = tuple(
        context.encrypt(row, level=0, slots=decode_feature.slots)
        for row in decode_feature.pack(decode_h)
    )
    prefill_result = decode_result = None
    bootstrap = None
    try:
        bootstrap = prepare_bootstrap_operator(context, prefill_schedule)
        print(f"bootstrap_setup {bootstrap.setup_seconds:.1f}s", flush=True)
        prefill_result = prefill_model.forward(
            prefill,
            crypto_context=context,
            input_ciphers=prefill_ciphers,
            layers=prefill_layers,
            bootstrap_operator=bootstrap,
            schedule=prefill_schedule,
            emit_kv_cache=True,
            progress_callback=lambda row: print(
                f"prefill layer={row['layer']} wall={row['wall_seconds']:.1f}s "
                f"residual_bs={row['residual_bootstrap_calls']}",
                flush=True,
            ),
        )
        assert len(prefill_result.kv_caches) == num_layers
        for layer_index, (cache_k, cache_v) in enumerate(prefill_result.kv_caches):
            print(
                f"prefill cache layer={layer_index} "
                f"k_c={len(cache_k)} v_c={len(cache_v)} "
                f"k_limbs={cache_k[0].state.cur_limbs} "
                f"v_limbs={cache_v[0].state.cur_limbs}",
                flush=True,
            )
            assert len(cache_k) == int(attention.key_value_pair_count)
            assert len(cache_v) == int(attention.key_value_pair_count)
            assert all(
                int(cipher.state.cur_limbs) == kv_limbs for cipher in cache_k
            )
            assert all(
                int(cipher.state.cur_limbs) == kv_limbs for cipher in cache_v
            )
        for cipher in prefill_ciphers:
            release_if_supported(cipher)
        prefill_ciphers = ()
        trim_device_allocator_cache(context.device)
        decode_result = decode_model.decode(
            decode_h,
            crypto_context=context,
            input_ciphers=decode_ciphers,
            layers=decode_layers,
            schedule=decode_schedule,
            bootstrap_operator=bootstrap,
            refresh_post_attention=True,
            skip_attention=False,
            kv_caches=prefill_result.kv_caches,
            occupied=prefill_len,
            position=prefill_len,
            max_kv_pages=2,
            progress_callback=lambda row: print(
                f"decode layer={row['layer']} wall={row['wall_seconds']:.1f}s "
                f"softmax_bs={row['softmax_bootstrap_calls']} "
                f"residual_bs={row['residual_bootstrap_calls']} "
                f"out_levels={row['output_levels']}",
                flush=True,
            ),
        )
        print(
            f"decode occupied={decode_result.occupied} "
            f"wall={decode_result.wall_seconds:.1f}s layers={len(decode_result.per_layer)}",
            flush=True,
        )
        assert decode_result.occupied == prefill_len + 1
        assert len(decode_result.per_layer) == num_layers
        assert len(decode_result.kv_caches) == num_layers
        assert prefill_result.per_layer[1]["residual_bootstrap_calls"] > (
            prefill_result.per_layer[0]["residual_bootstrap_calls"]
        )
        for cache_k, cache_v in decode_result.kv_caches:
            assert all(
                int(cipher.state.cur_limbs) == kv_limbs for cipher in cache_k
            )
            assert all(
                int(cipher.state.cur_limbs) == kv_limbs for cipher in cache_v
            )
        hidden = decrypt_hidden(
            decode_result.ciphers,
            crypto_context=context,
            layout=decode_feature,
        )
        print(
            f"decode hidden shape={hidden.shape} "
            f"finite={bool(np.isfinite(hidden).all())}",
            flush=True,
        )
        assert hidden.shape == decode_h.shape
        assert np.isfinite(hidden).all()
    finally:
        if prefill_result is not None:
            prefill_result.release()
        if decode_result is not None:
            decode_result.release()
        if bootstrap is not None:
            bootstrap.release()
        for cipher in (*prefill_ciphers, *decode_ciphers):
            release_if_supported(cipher)
