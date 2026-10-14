from __future__ import annotations

import json
import time

import numpy as np
import pytest

from llama3fhe.backend import (
    create_sim_context,
    cuda_memory_snapshot,
    release_if_supported,
    synchronize_device,
)
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.codec import decrypt_complex_rows
from llama3fhe.operators.attention.pairfold.pairfold128 import (
    pv_pairfold128_numpy,
    qk_pairfold128_numpy,
)
from llama3fhe.operators.attention.pairfold.pairfold_decode import (
    pairfold_decode_rotations,
    pv_pairfold_pages_fhe,
    qk_pairfold_pages_fhe,
)
from llama3fhe.operators.attention.pairfold.pairfold_route import (
    pairfold_kv_from_carrier_fhe,
    pairfold_kv_from_carrier_numpy,
    pairfold_query_from_carrier_fhe,
    pairfold_query_from_carrier_numpy,
    pairfold_route_rotations,
)
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.pairfold.pairfold_qkv import (
    pairfold_qkv_fhe,
    pairfold_qkv_rotations,
)
from llama3fhe.operators.linear.prefill.qkv_kernel import ComplexTokenDiagonalLayout


def _random_weight(
    rng: np.random.Generator, rows: int, columns: int, *, scale: float
) -> np.ndarray:
    """Allocate directly in fp32 so this full-width gate stays memory bounded."""

    result = rng.standard_normal((int(rows), int(columns)), dtype=np.float32)
    result *= np.float32(scale)
    return result


def _state(cipher, *, crypto_context) -> dict[str, int]:  # noqa: ANN001
    return {
        "level": int(crypto_context.level_for_cipher(cipher)),
        "cur_limbs": int(cipher.state.cur_limbs),
        "scale_degree": int(cipher.state.scale_degree),
    }


@pytest.mark.gpu
def test_pairfold_projection_kv_and_streaming_q_gpu_decrypt_gate() -> None:
    """Exercise the real 4096-wide P3 path, including every physical slot.

    The test deliberately extracts only one Q ``(group, offset)`` after the
    four persistent K/V outputs.  This pins the Decode ownership contract:
    the projection carrier is borrowed by both routes and eight Q ciphertexts
    are never materialized together.
    """

    layout = PairFold128Layout()
    dimension = int(layout.carrier_dimension)
    token_column = 127
    position = 37
    query_group = 1
    query_offset = 3
    rng = np.random.default_rng(26092231)

    hidden = rng.standard_normal((1, dimension), dtype=np.float32)
    hidden *= np.float32(0.015)
    q_weight = _random_weight(
        rng,
        layout.query_heads * layout.head_dim,
        dimension,
        scale=0.002,
    )
    k_weight = _random_weight(
        rng,
        layout.key_value_heads * layout.head_dim,
        dimension,
        scale=0.002,
    )
    v_weight = _random_weight(
        rng,
        layout.key_value_heads * layout.head_dim,
        dimension,
        scale=0.002,
    )
    logical_query = (hidden.astype(np.float64) @ q_weight.T).reshape(
        layout.query_heads, layout.head_dim
    )
    logical_key = (hidden.astype(np.float64) @ k_weight.T).reshape(
        layout.key_value_heads, layout.head_dim
    )
    logical_value = (hidden.astype(np.float64) @ v_weight.T).reshape(
        layout.key_value_heads, layout.head_dim
    )
    expected_carrier = layout.pack_qkv_carrier(
        logical_query, logical_key, logical_value
    )[0]

    operator_config = LinearOperatorConfig(
        baby_steps=64,
        baby_anchor_step=64,
        max_plaintext_rows=512,
        constant_cache_mode="none",
        raw_weight_policy="device",
    )
    config = Llama3CKKSConfig()
    config.simulator.logN = 16
    config.simulator.maxLevelsRemaining = 4
    config.simulator.dnum = 1
    config.simulator.device = "cuda"
    config.qkv_linear.dtype = "float64"
    route_rotations = pairfold_route_rotations(
        layout,
        token_columns=token_column,
        include_kv=True,
        include_query=True,
    )
    rotations = pairfold_qkv_rotations(
        layout,
        operator_config=operator_config,
        extra_rotations=route_rotations,
    )

    context_start = time.perf_counter()
    context = create_sim_context(
        config.simulator,
        input_dim=dimension,
        block_out=layout.native_token_lanes,
        rotations=rotations,
        include_default_rotations=False,
    )
    context_seconds = time.perf_counter() - context_start
    input_layout = ComplexTokenDiagonalLayout(
        seq_len=1,
        dimension=dimension,
        slots=layout.slots,
    )
    input_cipher = context.encrypt(
        input_layout.pack(hidden)[0], level=0, slots=layout.slots
    )
    input_before = decrypt_complex_rows(
        (input_cipher,), crypto_context=context
    )[0]

    projection = None
    token_write = None
    query_cipher = None
    try:
        projection = pairfold_qkv_fhe(
            hidden,
            q_weight,
            k_weight,
            v_weight,
            layout=layout,
            config=config,
            operator_config=operator_config,
            extra_rotations=route_rotations,
            crypto_context=context,
            input_ciphers=(input_cipher,),
            verify=False,
        )
        synchronize_device(context.device)
        actual_carrier = decrypt_complex_rows(
            (projection.carrier_cipher,), crypto_context=context
        )[0]

        # A full-slot assertion catches both a 3072-active-prefix truncation
        # and leakage into the seven inactive token lanes of every channel.
        active_slots = np.asarray(
            [
                channel * layout.native_token_lanes
                for channel in layout.active_carrier_channels()
            ],
            dtype=np.int64,
        )
        inactive = np.ones((layout.slots,), dtype=bool)
        inactive[active_slots] = False
        np.testing.assert_allclose(
            actual_carrier[active_slots],
            expected_carrier[active_slots],
            atol=3e-6,
            rtol=3e-5,
        )
        np.testing.assert_allclose(
            actual_carrier[inactive],
            0.0,
            atol=3e-6,
            rtol=0.0,
        )
        actual_q, actual_k, actual_v = layout.unpack_qkv_carrier(
            actual_carrier
        )
        np.testing.assert_allclose(
            actual_q, logical_query, atol=6e-6, rtol=3e-5
        )
        np.testing.assert_allclose(
            actual_k, logical_key, atol=6e-6, rtol=3e-5
        )
        np.testing.assert_allclose(
            actual_v, logical_value, atol=6e-6, rtol=3e-5
        )

        kv_profile: dict[str, float] = {}
        kv_start = time.perf_counter()
        token_write = pairfold_kv_from_carrier_fhe(
            projection.carrier_cipher,
            carrier_layout=projection.output_layout,
            layout=layout,
            crypto_context=context,
            position=position,
            token_column=token_column,
            batched_sparse_bsgs=True,
            shared_baby_rotations=True,
            fallback_on_error=False,
            profile=kv_profile,
        )
        synchronize_device(context.device)
        kv_seconds = time.perf_counter() - kv_start
        actual_key_rows = decrypt_complex_rows(
            tuple(token_write.key_groups), crypto_context=context
        )
        actual_value_rows = decrypt_complex_rows(
            tuple(token_write.value_groups), crypto_context=context
        )
        expected_key_rows, expected_value_rows = pairfold_kv_from_carrier_numpy(
            expected_carrier,
            layout=layout,
            position=position,
            token_column=token_column,
        )
        np.testing.assert_allclose(
            actual_key_rows,
            expected_key_rows,
            atol=8e-6,
            rtol=5e-5,
        )
        np.testing.assert_allclose(
            actual_value_rows,
            expected_value_rows,
            atol=8e-6,
            rtol=5e-5,
        )
        assert kv_profile["batched_sparse_bsgs"] == 1.0
        assert kv_profile["shared_baby_rotations"] == 1.0
        assert kv_profile["shared_basis_size"] == 2.0
        assert kv_profile["shared_fallbacks"] == 0.0

        query_profile: dict[str, float] = {}
        query_start = time.perf_counter()
        query_cipher = pairfold_query_from_carrier_fhe(
            projection.carrier_cipher,
            carrier_layout=projection.output_layout,
            layout=layout,
            crypto_context=context,
            group=query_group,
            query_offset=query_offset,
            position=position,
            batched_sparse_bsgs=True,
            fallback_on_error=False,
            profile=query_profile,
        )
        synchronize_device(context.device)
        query_seconds = time.perf_counter() - query_start
        actual_query_row = decrypt_complex_rows(
            (query_cipher,), crypto_context=context
        )[0]
        expected_query_row = pairfold_query_from_carrier_numpy(
            expected_carrier,
            layout=layout,
            group=query_group,
            query_offset=query_offset,
            position=position,
        )
        np.testing.assert_allclose(
            actual_query_row,
            expected_query_row,
            atol=8e-6,
            rtol=5e-5,
        )

        # Both route APIs borrow the complete projection carrier, and the
        # projection kernel in turn borrows its encrypted DecodeFM input.
        carrier_after_routes = decrypt_complex_rows(
            (projection.carrier_cipher,), crypto_context=context
        )[0]
        input_after = decrypt_complex_rows(
            (input_cipher,), crypto_context=context
        )[0]
        np.testing.assert_allclose(
            carrier_after_routes, actual_carrier, atol=3e-7, rtol=3e-7
        )
        np.testing.assert_allclose(
            input_after, input_before, atol=3e-7, rtol=3e-7
        )

        states = {
            "input": _state(input_cipher, crypto_context=context),
            "carrier": _state(
                projection.carrier_cipher, crypto_context=context
            ),
            "kv": [
                _state(cipher, crypto_context=context)
                for cipher in token_write.handles
            ],
            "query": _state(query_cipher, crypto_context=context),
        }
        assert states["input"]["level"] == 0
        assert states["carrier"]["level"] == 1
        assert {state["level"] for state in states["kv"]} == {2}
        assert states["query"]["level"] == 2
        assert {
            state["cur_limbs"] + state["level"]
            for state in [states["input"], states["carrier"], *states["kv"], states["query"]]
        } == {int(context.context.L)}
        assert {
            state["scale_degree"]
            for state in [states["input"], states["carrier"], *states["kv"], states["query"]]
        } == {1}

        print(
            "PAIRFOLD_GPU_GATE "
            + json.dumps(
                {
                    "context_seconds": context_seconds,
                    "projection_seconds": projection.wall_seconds,
                    "projection_stages": projection.stage_seconds,
                    "kv_route_seconds": kv_seconds,
                    "kv_profile": kv_profile,
                    "one_query_route_seconds": query_seconds,
                    "query_profile": query_profile,
                    "rotation_key_count": len(rotations),
                    "states": states,
                    "memory": cuda_memory_snapshot(context.device),
                },
                sort_keys=True,
            )
        )
    finally:
        if query_cipher is not None:
            release_if_supported(query_cipher)
        if token_write is not None:
            token_write.release()
            assert token_write.is_released
        if projection is not None:
            projection.release()
            projection.release()
        release_if_supported(input_cipher)


@pytest.mark.gpu
def test_pairfold_prefix_plus_content_qk_pv_gpu_decrypt_gate() -> None:
    """Run production streaming QK and Prefix+content PV on real CUDA FHE."""

    layout = PairFold128Layout()
    carrier_layout = LinearCarrierLayout(
        seq_len=1,
        dimension=layout.carrier_dimension,
        slots=layout.slots,
    )
    position = 19
    prefix_k = 2
    occupied = 3
    rng = np.random.default_rng(26092232)
    query = rng.normal(
        scale=0.01, size=(layout.query_heads, layout.head_dim)
    )
    zero_kv = np.zeros(
        (layout.key_value_heads, layout.head_dim), dtype=np.float64
    )
    carrier = layout.pack_qkv_carrier(query, zero_kv, zero_kv)[0]
    prefix_key = rng.normal(
        scale=0.01,
        size=(prefix_k, layout.key_value_heads, layout.head_dim),
    )
    content_key = rng.normal(
        scale=0.01,
        size=(occupied, layout.key_value_heads, layout.head_dim),
    )
    prefix_value = rng.normal(scale=0.02, size=prefix_key.shape)
    content_value = rng.normal(scale=0.02, size=content_key.shape)
    packed_prefix_key = layout.pack_key(prefix_key)
    packed_content_key = layout.pack_key(content_key)
    packed_prefix_value = layout.pack_value(prefix_value)
    packed_content_value = layout.pack_value(content_value)
    packed_queries = np.stack(
        [
            pairfold_query_from_carrier_numpy(
                carrier,
                layout=layout,
                group=group,
                query_offset=query_offset,
                position=position,
            )
            for group in range(layout.key_value_groups)
            for query_offset in range(layout.gqa_ratio)
        ]
    )
    expected_prefix_score = qk_pairfold128_numpy(
        packed_queries,
        packed_prefix_key,
        layout=layout,
        occupied=prefix_k,
    )
    expected_content_score = qk_pairfold128_numpy(
        packed_queries,
        packed_content_key,
        layout=layout,
        occupied=occupied,
    )
    expected_output = (
        pv_pairfold128_numpy(
            expected_prefix_score,
            packed_prefix_value,
            layout=layout,
            occupied=prefix_k,
        )
        + pv_pairfold128_numpy(
            expected_content_score,
            packed_content_value,
            layout=layout,
            occupied=occupied,
        )
    )[0]

    config = Llama3CKKSConfig()
    config.simulator.logN = 16
    config.simulator.maxLevelsRemaining = 7
    config.simulator.dnum = 1
    config.simulator.device = "cuda"
    route_rotations = pairfold_route_rotations(
        layout,
        token_columns=0,
        include_kv=False,
        include_query=True,
    )
    rotations = tuple(
        sorted(set(route_rotations) | set(pairfold_decode_rotations(layout)))
    )
    context_start = time.perf_counter()
    context = create_sim_context(
        config.simulator,
        input_dim=1,
        block_out=1,
        rotations=rotations,
        include_default_rotations=False,
    )
    context_seconds = time.perf_counter() - context_start
    carrier_cipher = context.encrypt(carrier, level=0, slots=layout.slots)
    key_ciphers = tuple(
        context.encrypt(row, level=1, slots=layout.slots)
        for row in packed_content_key
    )
    value_ciphers = tuple(
        context.encrypt(row, level=1, slots=layout.slots)
        for row in packed_content_value
    )
    carrier_before = decrypt_complex_rows(
        (carrier_cipher,), crypto_context=context
    )[0]
    key_before = decrypt_complex_rows(key_ciphers, crypto_context=context)
    value_before = decrypt_complex_rows(value_ciphers, crypto_context=context)

    score_pages: tuple[tuple[object], ...] = ()
    output: tuple[object, ...] = ()
    try:
        extractor_profiles: list[dict[str, float]] = []

        def extract_one(cipher, *, group: int, query_offset: int):  # noqa: ANN001
            child_profile: dict[str, float] = {}
            extractor_profiles.append(child_profile)
            return pairfold_query_from_carrier_fhe(
                cipher,
                carrier_layout=carrier_layout,
                layout=layout,
                crypto_context=context,
                group=group,
                query_offset=query_offset,
                position=position,
                batched_sparse_bsgs=True,
                fallback_on_error=False,
                profile=child_profile,
            )

        qk_profile: dict[str, float] = {}
        qk_started = time.perf_counter()
        score_pages = qk_pairfold_pages_fhe(
            carrier_cipher,
            (key_ciphers,),
            query_extractor=extract_one,
            layout=layout,
            crypto_context=context,
            occupied_pages=(occupied,),
            key_prefix=prefix_key,
            prefix_k=prefix_k,
            profile=qk_profile,
        )
        synchronize_device(context.device)
        qk_seconds = time.perf_counter() - qk_started
        actual_scores = tuple(
            decrypt_complex_rows(page, crypto_context=context)[0]
            for page in score_pages
        )
        np.testing.assert_allclose(
            actual_scores[0],
            expected_prefix_score[0],
            atol=2e-5,
            rtol=2e-4,
        )
        np.testing.assert_allclose(
            actual_scores[1],
            expected_content_score[0],
            atol=2e-5,
            rtol=2e-4,
        )
        assert qk_profile["query_peak_live_ciphers"] == 1.0
        assert qk_profile["query_extract_count"] == 8.0
        assert len(extractor_profiles) == 8

        pv_profile: dict[str, float] = {}
        pv_started = time.perf_counter()
        output = pv_pairfold_pages_fhe(
            (score_pages[1],),
            (value_ciphers,),
            layout=layout,
            crypto_context=context,
            occupied_pages=(occupied,),
            prefix_probability_ciphers=score_pages[0],
            value_prefix=prefix_value,
            prefix_k=prefix_k,
            profile=pv_profile,
        )
        synchronize_device(context.device)
        pv_seconds = time.perf_counter() - pv_started
        actual_output = decrypt_complex_rows(
            output, crypto_context=context
        )[0]
        np.testing.assert_allclose(
            actual_output,
            expected_output,
            atol=5e-5,
            rtol=5e-4,
        )
        inactive = np.ones((layout.slots,), dtype=bool)
        inactive[np.flatnonzero(expected_output)] = False
        np.testing.assert_allclose(
            actual_output[inactive], 0.0, atol=5e-5, rtol=0.0
        )

        # QK/PV borrow their carrier, cache pages and score pages.
        np.testing.assert_allclose(
            decrypt_complex_rows(
                (carrier_cipher,), crypto_context=context
            )[0],
            carrier_before,
            atol=3e-7,
            rtol=3e-7,
        )
        np.testing.assert_allclose(
            decrypt_complex_rows(key_ciphers, crypto_context=context),
            key_before,
            atol=3e-7,
            rtol=3e-7,
        )
        np.testing.assert_allclose(
            decrypt_complex_rows(value_ciphers, crypto_context=context),
            value_before,
            atol=3e-7,
            rtol=3e-7,
        )
        scores_after_pv = tuple(
            decrypt_complex_rows(page, crypto_context=context)[0]
            for page in score_pages
        )
        for after, before in zip(scores_after_pv, actual_scores, strict=True):
            np.testing.assert_allclose(after, before, atol=3e-7, rtol=3e-7)

        states = {
            "carrier": _state(carrier_cipher, crypto_context=context),
            "keys": [
                _state(cipher, crypto_context=context) for cipher in key_ciphers
            ],
            "values": [
                _state(cipher, crypto_context=context)
                for cipher in value_ciphers
            ],
            "scores": [
                _state(page[0], crypto_context=context) for page in score_pages
            ],
            "output": _state(output[0], crypto_context=context),
        }
        assert states["carrier"]["level"] == 0
        assert {state["level"] for state in states["keys"]} == {1}
        assert {state["level"] for state in states["values"]} == {1}
        assert {state["level"] for state in states["scores"]} == {3}
        assert states["output"]["level"] == 6
        assert {
            state["scale_degree"]
            for state in [
                states["carrier"],
                *states["keys"],
                *states["values"],
                *states["scores"],
                states["output"],
            ]
        } == {1}
        print(
            "PAIRFOLD_GPU_QK_PV_GATE "
            + json.dumps(
                {
                    "context_seconds": context_seconds,
                    "qk_seconds": qk_seconds,
                    "qk_profile": qk_profile,
                    "pv_seconds": pv_seconds,
                    "pv_profile": pv_profile,
                    "rotation_key_count": len(rotations),
                    "states": states,
                    "memory": cuda_memory_snapshot(context.device),
                },
                sort_keys=True,
            )
        )
    finally:
        for page in score_pages:
            for cipher in page:
                release_if_supported(cipher)
        for cipher in output:
            release_if_supported(cipher)
        for cipher in key_ciphers + value_ciphers + (carrier_cipher,):
            release_if_supported(cipher)
