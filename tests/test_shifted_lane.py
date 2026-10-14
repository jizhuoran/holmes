from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.layouts.shifted_lane import (
    ShiftedLaneLayout,
    bsgs_matvec,
    diagonal_plaintext,
    shifted_lane_linear,
)
from llama3fhe.operators.linear import (
    LinearOperator,
    LinearOperatorConfig,
    LinearWeight,
    linear_rotations,
)


def test_layout_shapes_and_lane_shifts() -> None:
    layout = ShiftedLaneLayout(hidden_dim=64, slots=512)
    assert layout.lanes == 8
    assert layout.shift == 8
    assert layout.fold_offsets() == (1, 2, 4)
    assert layout.lane0_to_shifted_offsets() == tuple(63 * l for l in range(1, 8))
    x = np.arange(64, dtype=np.float64)
    carrier = layout.pack(x)
    assert layout.is_shifted_lane(carrier)
    for lane in range(8):
        np.testing.assert_array_equal(layout.unpack(carrier, lane=lane), x)
    assert carrier[layout.slot(3, 2)] == x[(3 + 2 * 8) % 64]


def test_lane0_expansion_matches_pack_and_fold_is_lane0_sum() -> None:
    layout = ShiftedLaneLayout(hidden_dim=64, slots=512)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(64)
    lane0 = layout.pack_lane0(x)
    np.testing.assert_allclose(layout.lane0_to_shifted(lane0), layout.pack(x))
    # a dirty carrier needs the mask
    dirty = lane0 + rng.standard_normal(512) * (1 - layout.lane_mask(0))
    np.testing.assert_allclose(layout.lane0_to_shifted(dirty, masked=True), layout.pack(x))
    v = rng.standard_normal(512)
    folded = layout.fold_lanes(v)
    np.testing.assert_allclose(folded[::8], v.reshape(64, 8).sum(axis=1))


@pytest.mark.parametrize("baby_steps", (1, 2, 4, 8))
def test_sl_bsgs_oracle_matches_matvec(baby_steps: int) -> None:
    layout = ShiftedLaneLayout(hidden_dim=64, slots=512)
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((64, 64))
    x = rng.standard_normal(64)
    out = shifted_lane_linear(weight, x, layout=layout, baby_steps=baby_steps)
    np.testing.assert_allclose(out[::8], weight @ x, atol=1e-12)


def test_sl_chunks_sum_to_full_product_like_the_all_reduce() -> None:
    layout = ShiftedLaneLayout(hidden_dim=64, slots=512)
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((64, 64))
    x = rng.standard_normal(64)
    carrier = layout.pack(x)
    giants = list(range(layout.shift // 2))  # baby 2 -> 4 giants
    parts = [
        bsgs_matvec(weight, carrier, layout=layout, baby_steps=2, lane_shift=layout.shift, giant_indices=[g])
        for g in giants
    ]
    np.testing.assert_allclose(layout.fold_lanes(sum(parts))[::8], weight @ x, atol=1e-12)


def test_lane_shift_zero_reproduces_the_lane_replicated_packing() -> None:
    layout = ShiftedLaneLayout(hidden_dim=64, slots=512)
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((64, 64))
    x = rng.standard_normal(64)
    out = bsgs_matvec(weight, layout.pack_lane0(x), layout=layout, baby_steps=8, lane_shift=0)
    np.testing.assert_allclose(out[::8], weight @ x, atol=1e-12)
    assert np.all(out.reshape(64, 8)[:, 1:] == 0)
    # the LR diagonal equals the existing _cpu_packer formula
    row = diagonal_plaintext(weight, layout=layout, baby=3, giant=2, baby_steps=8, lane_shift=0)
    feature = np.arange(512) // 8
    np.testing.assert_array_equal(row, weight[(feature - 16) % 64, (feature + 3) % 64])


def test_shifted_lane_plan_has_fewer_diagonals_and_extra_keys() -> None:
    config = LinearOperatorConfig(baby_steps=64, baby_anchor_step=64, max_plaintext_rows=64, lane_shift=512)
    plan = config.plan(dimension=4096, token_lanes=8)
    assert plan.diagonals == 512
    assert plan.giant_steps == 8
    assert plan.plaintext_chunks == 8
    assert plan.fold_offsets == (1, 2, 4)
    assert plan.lane_expand_offsets == tuple(4095 * l for l in range(1, 8))
    keys = set(linear_rotations(dimension=4096, token_lanes=8, operator_config=config))
    assert {1, 2, 4, 4095, 4095 * 7} <= keys
    with pytest.raises(ValueError):
        LinearOperatorConfig(lane_shift=256).plan(dimension=4096, token_lanes=8)


@pytest.mark.gpu
@pytest.mark.parametrize("use_gpu_packer", (True, False))
def test_shifted_lane_linear_matches_oracle_on_gpu(use_gpu_packer: bool) -> None:
    from llama3fhe.backend import create_sim_context
    from llama3fhe.config import SimulatorConfig
    from llama3fhe.operators.linear.shifted_lane import (
        expand_lane0_to_shifted_lanes,
        fold_shifted_lanes,
        mask_lane0,
    )

    dimension, lanes = 1024, 8  # slots 8192 (logN 14)
    config = LinearOperatorConfig(
        baby_steps=16,
        baby_anchor_step=16,
        max_plaintext_rows=32,  # 2 giants per chunk -> 4 chunks
        lane_shift=dimension // lanes,
        raw_weight_policy="device" if use_gpu_packer else "host",
    )
    rotations = linear_rotations(dimension=dimension, token_lanes=lanes, operator_config=config)
    context = create_sim_context(
        SimulatorConfig(logN=14, maxLevelsRemaining=4, dnum=1, dcrtBits=59, firstMod=60, device="cuda"),
        input_dim=dimension,
        block_out=lanes,
        rotations=rotations,
        include_default_rotations=False,
    )
    layout = ShiftedLaneLayout(hidden_dim=dimension, slots=8192)
    rng = np.random.default_rng(5)
    weight = rng.normal(scale=0.05, size=(dimension, dimension))
    x = rng.normal(scale=1.0, size=dimension)
    expected = weight @ x

    cipher = context.encrypt(layout.pack_lane0(x), level=0)
    plan = config.plan(dimension=dimension, token_lanes=lanes)
    with LinearOperator(crypto_context=context, token_lanes=lanes, config=config) as operator, LinearWeight(
        weight, dtype=np.float64, crypto_context=context
    ) as prepared:
        sl_input = expand_lane0_to_shifted_lanes(cipher, plan=plan, crypto_context=context)
        assert layout.is_shifted_lane(np.asarray(context.decrypt(sl_input)).real, atol=1e-6)
        # split the chunks over two "ranks" and sum before the fold
        partial_a, _ = operator.apply((sl_input,), prepared, chunk_indices=(0, 1), rescale=False)
        partial_b, _ = operator.apply((sl_input,), prepared, chunk_indices=(2, 3), rescale=False)
        summed = context.fhe.homo_add(partial_a[0], partial_b[0], context.context)
        summed = context.fhe.rescale(summed, context.context)
        folded = fold_shifted_lanes(summed, plan=plan, crypto_context=context)
        clean = mask_lane0(folded, plan=plan, crypto_context=context)

    decrypted = np.asarray(context.decrypt(folded)).real
    np.testing.assert_allclose(decrypted[::lanes], expected, atol=1e-5, rtol=1e-5)
    masked = np.asarray(context.decrypt(clean)).real
    np.testing.assert_allclose(masked[::lanes], expected, atol=1e-5, rtol=1e-5)
    assert np.abs(masked.reshape(dimension, lanes)[:, 1:]).max() < 1e-5
