from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.backend import create_sim_context
from llama3fhe.config import SimulatorConfig
from llama3fhe.operators.linear import (
    LLAMA3_LINEAR_SCHEDULES,
    LinearOperator,
    LinearOperatorConfig,
    LinearWeight,
    linear_rotations,
)
from llama3fhe.operators.linear import operator as linear_operator_impl
from llama3fhe.operators.linear.prefill.qkv_kernel import (
    complex_token_diagonal_qkv_rotations,
)


def test_production_llama3_linear_schedules_are_named_and_bounded() -> None:
    schedules = LLAMA3_LINEAR_SCHEDULES
    for operator_config in (
        schedules.qkv,
        schedules.w_o,
        schedules.gate_up,
        schedules.down,
    ):
        assert operator_config.baby_steps == 64
        assert operator_config.baby_anchor_step == 64
        assert operator_config.max_plaintext_rows == 512
        assert operator_config.reuse_baby_rotations is True
        assert operator_config.hoist_strategy == "normal"
        assert operator_config.constant_cache_mode == "none"
        assert operator_config.raw_weight_policy == "device"


def test_diagonal_linear_production_plan_is_operator_independent() -> None:
    plan = LinearOperatorConfig(
        baby_steps=32,
        max_plaintext_rows=512,
        reuse_baby_rotations=True,
        hoist_strategy="ext_double_hoist",
    ).plan(dimension=4096, token_lanes=8)

    assert plan.giant_steps == 128
    assert plan.chunk_giants == 16
    assert plan.chunk_sizes == (16,) * 8
    assert plan.plaintext_chunks == 8
    assert plan.giant_offset == 256
    assert plan.baby_offsets == tuple(8 * index for index in range(32))
    assert LinearOperatorConfig().reuse_baby_rotations is True


def test_large_baby_schedule_caps_fast_rotation_keys_at_32() -> None:
    config = LinearOperatorConfig(
        baby_steps=32,
        baby_anchor_step=32,
        max_plaintext_rows=512,
    )
    plan = config.plan(dimension=4096, token_lanes=8)
    rotations = set(
        linear_rotations(
            dimension=4096,
            token_lanes=8,
            operator_config=config,
        )
    )

    assert tuple(block.width for block in plan.baby_blocks) == (32,)
    assert plan.giant_offset == 256
    assert plan.baby_rotation_count == 31
    assert plan.baby_key_offsets == tuple(8 * index for index in range(1, 32))
    assert set(plan.baby_key_offsets) <= rotations
    assert 256 not in plan.baby_key_offsets

    large = LinearOperatorConfig(
        baby_steps=128,
        baby_anchor_step=32,
        max_plaintext_rows=512,
    ).plan(dimension=4096, token_lanes=8)
    assert tuple(block.width for block in large.baby_blocks) == (32,) * 4
    assert large.baby_rotation_count == 127
    assert large.baby_key_offsets == tuple(8 * index for index in range(1, 33))
    assert 264 not in large.baby_key_offsets


def test_baby_step_64_can_use_one_64_way_fast_rotation() -> None:
    config = LinearOperatorConfig(
        baby_steps=64,
        baby_anchor_step=64,
        max_plaintext_rows=512,
    )
    plan = config.plan(dimension=4096, token_lanes=8)
    rotations = set(
        linear_rotations(
            dimension=4096,
            token_lanes=8,
            operator_config=config,
        )
    )

    assert plan.anchor_enabled is False
    assert tuple(block.width for block in plan.baby_blocks) == (64,)
    assert plan.baby_rotation_count == 63
    assert plan.baby_key_offsets == tuple(8 * index for index in range(1, 64))
    assert set(plan.baby_key_offsets) <= rotations
    # The production default is normal hoisting: 63 baby keys plus the seven
    # global chunk-alignment offsets and one conjugation key.
    assert len(rotations) == 71


@pytest.mark.parametrize(
    ("strategy", "baby_steps", "anchor_step", "expected_keys"),
    [
        ("normal", 32, 32, 39),
        ("ext_normal", 32, 32, 39),
        ("ext_double_hoist", 32, 32, 53),
        ("normal", 64, 64, 71),
        ("ext_normal", 64, 64, 71),
        ("ext_double_hoist", 64, 64, 77),
        ("normal", 128, 64, 72),
        ("ext_normal", 128, 64, 72),
        ("ext_double_hoist", 128, 64, 74),
    ],
)
def test_linear_rotation_keys_follow_the_actual_giant_schedule(
    strategy: str,
    baby_steps: int,
    anchor_step: int,
    expected_keys: int,
) -> None:
    config = LinearOperatorConfig(
        baby_steps=int(baby_steps),
        baby_anchor_step=int(anchor_step),
        max_plaintext_rows=512,
        hoist_strategy=str(strategy),
    )
    rotations = linear_rotations(
        dimension=4096,
        token_lanes=8,
        operator_config=config,
    )
    assert len(rotations) == int(expected_keys)
    assert len(rotations) == len(set(rotations))


def test_named_qkv_rotation_helper_delegates_to_the_generic_linear_plan() -> None:
    generic = linear_rotations(
        dimension=4096,
        token_lanes=8,
        slots=32768,
        include_conjugation=True,
        operator_config=LinearOperatorConfig(baby_steps=32),
    )
    named_helper = complex_token_diagonal_qkv_rotations(
        dimension=4096,
        real_token_lanes=8,
        slots=32768,
        operator_config=LinearOperatorConfig(baby_steps=32),
    )

    assert generic == named_helper
    assert 4 * 32768 - 1 in generic


def _small_cpu_context():
    return create_sim_context(
        SimulatorConfig(
            logN=8,
            maxLevelsRemaining=3,
            dnum=1,
            dcrtBits=30,
            firstMod=35,
            device="cpu",
        ),
        input_dim=16,
        block_out=8,
        rotations=(),
        include_default_rotations=False,
    )


@pytest.mark.gpu
def test_linear_weight_uses_constant_bundle_cache_stage() -> None:
    context = _small_cpu_context()
    weight = np.eye(16, dtype=np.float64)

    expected = {
        "none": (0, 0, 0, 2, 2),
        "middle": (1, 0, 1, 1, 2),
        "plain": (0, 1, 0, 1, 1),
        "both": (1, 1, 0, 1, 1),
    }
    for mode, counts in expected.items():
        config = LinearOperatorConfig(
            baby_steps=4,
            max_plaintext_rows=16,
            constant_cache_mode=mode,
            raw_weight_policy="host",
        )
        with LinearOperator(
            crypto_context=context, token_lanes=8, config=config
        ) as operator, LinearWeight(
            weight, dtype=np.float64, crypto_context=context
        ) as prepared:
            plan = operator.plan(prepared.dimension)
            chunk = plan.chunks[0]
            prepared.plaintext(
                plan, chunk, level=0, operator_config=config
            )
            prepared.plaintext(
                plan, chunk, level=0, operator_config=config
            )
            info = prepared.cache_info()
            actual = (
                info["middle_entries"],
                info["plain_entries"],
                info["middle_hits"],
                info["middle_misses"],
                info["plain_misses"],
            )
            assert actual == counts


def test_linear_operator_can_prepare_babies_before_weight_construction(
    monkeypatch,
) -> None:
    class FakeContext:
        context = object()
        device = "cpu"

        @staticmethod
        def level_for_cipher(cipher) -> int:
            del cipher
            return 3

    source = (object(), object())
    prepared = []

    def fake_prepare(cipher, offsets, context, **kwargs):
        value = (object(),)
        prepared.append((cipher, offsets, context, kwargs, value))
        return value

    monkeypatch.setattr(
        linear_operator_impl.fhe,
        "prepare_hoisted_baby_rotations",
        fake_prepare,
    )
    operator = LinearOperator(
        crypto_context=FakeContext(),
        token_lanes=2,
        config=LinearOperatorConfig(
            baby_steps=4,
            baby_anchor_step=4,
            max_plaintext_rows=8,
        ),
    )
    first = operator.prepare_baby_rotations(source, dimension=8)
    second = operator.prepare_baby_rotations(source, dimension=8)

    assert first["baby_cache_misses"] == 2.0
    assert first["baby_cache_hits"] == 0.0
    # The first pass built both bases, so it owns their device time; the
    # second is a pure cache hit and does no backend work.
    assert first["baby_precompute"] > 0.0
    assert second["baby_cache_misses"] == 0.0
    assert second["baby_cache_hits"] == 2.0
    assert second["baby_precompute"] == 0.0
    assert len(prepared) == 2
    assert operator.cache_info() == {
        "baby_entries": 2,
        "baby_hits": 2,
        "baby_misses": 2,
    }
    operator.close()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "hoist_strategy", ("normal", "ext_normal", "ext_double_hoist")
)
def test_cached_linear_plaintext_is_reusable_by_fhe_operator(
    hoist_strategy: str,
) -> None:
    config = LinearOperatorConfig(
        baby_steps=8,
        baby_anchor_step=-1,
        max_plaintext_rows=64,
        constant_cache_mode="plain",
        raw_weight_policy="device",
        hoist_strategy=str(hoist_strategy),
    )
    simulator = SimulatorConfig(
        logN=14,
        maxLevelsRemaining=3,
        dnum=1,
        dcrtBits=59,
        firstMod=60,
        device="cuda",
    )
    context = create_sim_context(
        simulator,
        input_dim=64,
        block_out=128,
        rotations=linear_rotations(
            dimension=64, token_lanes=128, operator_config=config
        ),
        include_default_rotations=False,
    )
    rng = np.random.default_rng(26081431)
    values = rng.normal(scale=0.02, size=(128, 64))
    weight = rng.normal(scale=0.02, size=(64, 64))
    cipher = context.encrypt(values.T.reshape(-1), level=0, slots=8192)

    with LinearOperator(
        crypto_context=context, token_lanes=128, config=config
    ) as operator, LinearWeight(
        weight, dtype=np.float64, crypto_context=context
    ) as prepared:
        first, first_profile = operator.apply((cipher,), prepared)
        second, second_profile = operator.apply((cipher,), prepared)
        info = prepared.cache_info()
        operator_info = operator.cache_info()

    expected = values @ weight.T
    for output in (first[0], second[0]):
        actual = np.asarray(context.decrypt(output)).real.reshape(64, 128).T
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
    assert info["plain_entries"] == 1
    assert info["plain_hits"] == 1
    assert operator_info == {
        "baby_entries": 1,
        "baby_hits": 1,
        "baby_misses": 1,
    }
    assert first_profile["baby_precompute"] > 0.0
    assert first_profile["baby_precompute_enqueue"] > 0.0
    assert second_profile["baby_precompute"] == 0.0
