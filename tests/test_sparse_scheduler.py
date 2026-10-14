from __future__ import annotations

import numpy as np
import pytest

from llama3fhe import Llama3CKKSConfig
from llama3fhe.backend import (
    create_sim_context,
    release_if_supported,
    synchronize_device,
)
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.operators.attention.codec import decrypt_complex_rows
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.operators.attention.prefill.input import (
    SparseDiagonalPlan,
    _apply_sparse_routes,
    _apply_sparse_routes_batched,
    _key_shear_routes,
    _layout_parameters,
    _value_delta_routes,
    choose_sparse_diagonal_plan,
)
from llama3fhe.operators.attention.sparse_bsgs_scheduler import (
    SparseMaskRoute,
    build_sparse_route_schedule,
    materialize_sparse_batch_masks,
)


def _production_params() -> dict[str, int]:
    return _layout_parameters(
        LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768),
        AttentionPairLayout(),
    )


def _schedule(routes: dict[int, np.ndarray], *, padded_rows=None):
    params = _production_params()
    plan = choose_sparse_diagonal_plan(
        tuple(routes), slots=params["slots"], max_baby_offsets=32
    )
    schedule = build_sparse_route_schedule(
        diagonal_offsets=plan.diagonal_offsets,
        baby_offsets=plan.baby_offsets,
        modulus=plan.modulus,
        slots=params["slots"],
        batch_cap=32,
        padded_plaintext_rows=padded_rows,
    )
    return plan, schedule


def test_production_v_delta_schedule_preserves_routes_rotations_and_cap() -> None:
    params = _production_params()
    routes = _value_delta_routes(params)
    plan, schedule = _schedule(routes)

    assert len(routes) == schedule.route_rows == 2062
    assert len(schedule.baby_offsets) == 32
    assert len(schedule.groups) == 138
    assert len(schedule.batches) == 69
    assert max(group.row_count for group in schedule.groups) == 15
    assert max(batch.encoded_rows for batch in schedule.batches) <= 32
    assert set(schedule.diagonal_offsets) == set(plan.diagonal_offsets)
    assert schedule.baby_rotations + schedule.giant_rotations == (
        plan.rotations_per_cipher
    )


@pytest.mark.parametrize(
    ("upper", "route_rows", "padding_rows", "encode_batches"),
    ((False, 127, 1, 4), (True, 128, 0, 4)),
)
def test_production_key_shear_schedule_preserves_padded_ptct_count(
    upper: bool,
    route_rows: int,
    padding_rows: int,
    encode_batches: int,
) -> None:
    routes = _key_shear_routes(_production_params(), upper=upper)
    plan, schedule = _schedule(routes, padded_rows=128)

    assert schedule.route_rows == route_rows
    assert schedule.padding_rows == padding_rows
    assert schedule.encoded_rows == 128
    assert len(schedule.batches) == encode_batches
    assert schedule.baby_rotations + schedule.giant_rotations == (
        plan.rotations_per_cipher
    )


def test_batched_sparse_masks_equal_reference_roll_and_keep_padding_zero() -> None:
    params = _production_params()
    routes = _key_shear_routes(params, upper=False)
    _, schedule = _schedule(routes, padded_rows=128)

    seen: set[int] = set()
    for batch in schedule.batches:
        masks, metadata = materialize_sparse_batch_masks(
            batch, routes=routes, slots=params["slots"]
        )
        assert masks.shape == (batch.encoded_rows, params["slots"])
        for group, start, stop in metadata:
            assert stop - start == group.row_count
            for local, offset in enumerate(group.diagonal_offsets):
                reference = np.zeros((params["slots"],), dtype=np.float64)
                reference[routes[offset]] = 1.0
                reference = np.roll(reference, int(group.giant_offset))
                np.testing.assert_array_equal(masks[start + local], reference)
                seen.add(int(offset))
        if batch.padding_rows:
            assert not np.any(masks[batch.route_rows :])
    assert seen == set(routes)


def test_batch_cap_rejects_splitting_a_production_giant_group() -> None:
    params = _production_params()
    routes = _value_delta_routes(params)
    plan = choose_sparse_diagonal_plan(
        tuple(routes), slots=params["slots"], max_baby_offsets=32
    )
    with pytest.raises(ValueError, match="largest complete giant group"):
        build_sparse_route_schedule(
            diagonal_offsets=plan.diagonal_offsets,
            baby_offsets=plan.baby_offsets,
            modulus=plan.modulus,
            slots=params["slots"],
            batch_cap=14,
        )


def test_sparse_scheduler_options_are_validated_by_the_config() -> None:
    """The batch cap is checked where it is declared, not at each use site."""

    assert AttentionOperatorConfig(sparse_input_batch_cap=16) is not None
    for invalid in (0, 33, -1):
        with pytest.raises(ValueError, match="sparse_input_batch_cap"):
            AttentionOperatorConfig(sparse_input_batch_cap=invalid)


@pytest.mark.gpu
def test_batched_sparse_bsgs_fhe_matches_reference_with_complex_routes() -> None:
    # EasyFHE's GPU NTT kernel used by this repository supports the production
    # ring size; keep the route set tiny while exercising logN=16 slots.
    slots = 32768
    route_indices = {
        -4: (17, 31),
        -3: (8, 29),
        0: (0, 7, 19),
        1: (1, 9, 21),
        4: (4, 11),
        5: (5, 13),
    }
    routes = {
        offset: SparseMaskRoute(
            destinations=np.asarray(indices, dtype=np.int64),
            coefficients=np.asarray(
                tuple(1.0 if index % 2 == 0 else -1.0j for index in indices),
                dtype=np.complex128,
            ),
        )
        for offset, indices in route_indices.items()
    }
    plan = SparseDiagonalPlan(
        modulus=4,
        diagonal_offsets=(-4, -3, 0, 1, 4, 5),
        baby_offsets=(0, 1),
        giant_offsets=(-4, 0, 4),
    )
    config = Llama3CKKSConfig()
    config.simulator.logN = 16
    config.simulator.maxLevelsRemaining = 4
    config.simulator.device = "cuda"
    context = create_sim_context(
        config.simulator,
        input_dim=32,
        block_out=8,
        rotations=(-4, 1, 4),
        include_default_rotations=False,
    )
    values = np.linspace(-0.05, 0.05, slots, dtype=np.float64)
    cipher = context.encrypt(values, level=0, slots=slots)
    reference: tuple[object, ...] = ()
    batched: tuple[object, ...] = ()
    try:
        reference = _apply_sparse_routes(
            (cipher,),
            routes=routes,
            plan=plan,
            slots=slots,
            crypto_context=context,
            function_prefix="test.sparse.reference",
            padded_plaintext_rows=8,
        )
        batched = _apply_sparse_routes_batched(
            (cipher,),
            routes=routes,
            plan=plan,
            slots=slots,
            crypto_context=context,
            function_prefix="test.sparse.batched",
            padded_plaintext_rows=8,
            batch_cap=4,
        )
        synchronize_device(context.device)
        assert context.level_for_cipher(reference[0]) == 1
        assert context.level_for_cipher(batched[0]) == 1
        actual_reference = decrypt_complex_rows(
            reference, crypto_context=context
        )[0]
        actual_batched = decrypt_complex_rows(
            batched, crypto_context=context
        )[0]
        np.testing.assert_allclose(
            actual_batched,
            actual_reference,
            atol=2e-7,
            rtol=2e-7,
        )
        expected = np.zeros((slots,), dtype=np.complex128)
        for offset, route in routes.items():
            expected[route.destinations] += (
                values[(route.destinations + int(offset)) % slots]
                * route.coefficients
            )
        np.testing.assert_allclose(
            actual_batched,
            expected,
            atol=2e-7,
            rtol=2e-7,
        )
    finally:
        for value in reference + batched + (cipher,):
            release_if_supported(value)


def test_sparse_batch_mask_indices_reproduce_dense_masks() -> None:
    from llama3fhe.operators.attention.sparse_bsgs_scheduler import (
        sparse_batch_mask_indices,
    )

    slots = 64
    routes = {
        -4: np.asarray([1, 5, 9]),
        0: np.asarray([0, 7]),
        3: SparseMaskRoute(
            destinations=np.asarray([2, 60]),
            coefficients=np.asarray([1.0, -1.0j]),
        ),
        4: np.asarray([63]),
    }
    schedule = build_sparse_route_schedule(
        diagonal_offsets=(-4, 0, 3, 4),
        baby_offsets=(0, 3),
        modulus=4,
        slots=slots,
        batch_cap=4,
        padded_plaintext_rows=6,
    )
    for batch in schedule.batches:
        dense, dense_meta = materialize_sparse_batch_masks(
            batch, routes=routes, slots=slots
        )
        rows, cols, coeffs, meta = sparse_batch_mask_indices(
            batch, routes=routes, slots=slots
        )
        rebuilt = np.zeros(dense.shape, dtype=np.complex128)
        rebuilt[rows, cols] = coeffs
        np.testing.assert_array_equal(rebuilt, dense.astype(np.complex128))
        assert meta == dense_meta
