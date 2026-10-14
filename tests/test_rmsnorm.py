from __future__ import annotations

import numpy as np
import pytest

from llama3fhe import Llama3CKKSConfig
from llama3fhe.approx.chebyshev import eval_chebyshev_coefficients
from llama3fhe.approx.rmsnorm import rmsnorm_numpy
from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.norm.refresh import (
    adaptive_bootstrap_output_level_drop,
)
from llama3fhe.operators.norm.rmsnorm import (
    residual_add,
    rmsnorm_complexity,
    rmsnorm_fhe,
    rmsnorm_rotations,
)


def test_feature_major_production_contract_and_roundtrip() -> None:
    layout = FeatureMajorPrefillLayout()
    assert layout.tokens_per_cipher == 8
    assert layout.cipher_count == 16
    assert layout.slot(0, 0) == 0
    assert layout.slot(1, 0) == 8
    assert layout.slot(4095, 7) == 32767

    small = FeatureMajorPrefillLayout(
        seq_len=10, hidden_dim=16, slots=128
    )
    rows = np.arange(160, dtype=np.float32).reshape(10, 16)
    packed = small.pack(rows)
    assert packed.shape == (2, 128)
    assert packed[0, small.slot(3, 2)] == rows[2, 3]
    assert packed[1, small.slot(3, 1)] == rows[9, 3]
    np.testing.assert_array_equal(small.unpack(packed), rows)


def test_rmsnorm_numpy_reference() -> None:
    rng = np.random.default_rng(26081401)
    hidden = rng.normal(scale=0.2, size=(9, 16)).astype(np.float32)
    gamma = rng.normal(loc=1.0, scale=0.02, size=(16,)).astype(np.float32)

    actual = rmsnorm_numpy(hidden, gamma, eps=1e-5)
    mean_square = np.mean(np.square(hidden, dtype=np.float32), axis=-1)
    expected = (
        hidden
        * np.reciprocal(np.sqrt(mean_square + np.float32(1e-5)))[:, None]
        * gamma[None, :]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)


def test_feature_major_production_rmsnorm_complexity() -> None:
    layout = FeatureMajorPrefillLayout()
    complexity = rmsnorm_complexity(
        layout, polynomial_degree=32, apply_gamma=True
    )
    assert complexity.input_cipher_count == 16
    assert complexity.mean_square_ct_ct_multiplications == 16
    assert complexity.mean_square_rotations == 192
    assert complexity.slim_pack_pt_ct_multiplications == 16
    assert complexity.slim_pack_rotations == 0
    assert complexity.slim_extract_pt_ct_multiplications == 16
    assert complexity.slim_extract_rotations == 0
    assert complexity.scale_broadcast_rotations == 192
    assert complexity.output_ct_ct_multiplications == 16
    assert complexity.gamma_pt_ct_multiplications == 16
    assert complexity.explicit_multiplicative_depth_excluding_polynomial == 5
    fused = rmsnorm_complexity(
        layout, polynomial_degree=32, apply_gamma=False
    )
    assert fused.gamma_pt_ct_multiplications == 0
    assert fused.explicit_multiplicative_depth_excluding_polynomial == 4


def test_residual_bootstrap_level_drop_is_clamped_to_next_consumer() -> None:
    # A layer-1 attention RMSNorm consumes 8 polynomial + 5 structural
    # levels.  From a 28-limb bootstrap, dropping five still leaves the ten
    # limbs needed to reach the pre-exp checkpoint.
    assert adaptive_bootstrap_output_level_drop(
        current_limbs=28,
        requested_level_drop=5,
        rmsnorm_depth=13,
        minimum_output_limbs=10,
    ) == 5

    # Its FFN RMSNorm consumes 7 + 5 levels and the degree-96 SiLU path needs
    # thirteen input limbs, so only three of the requested five are safe.
    assert adaptive_bootstrap_output_level_drop(
        current_limbs=28,
        requested_level_drop=5,
        rmsnorm_depth=12,
        minimum_output_limbs=13,
    ) == 3

    # Deep per-layer RMSNorm polynomials may consume the whole budget; in
    # that case retain every bootstrap limb and let the existing downstream
    # checkpoint handle the unavoidable refresh.
    assert adaptive_bootstrap_output_level_drop(
        current_limbs=28,
        requested_level_drop=5,
        rmsnorm_depth=18,
        minimum_output_limbs=12,
    ) == 0


def _small_feature_major_context(
    layout: FeatureMajorPrefillLayout,
    *,
    depth: int,
):
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14
    cfg.simulator.maxLevelsRemaining = int(depth)
    cfg.simulator.device = "cuda"
    return create_sim_context(
        cfg.simulator,
        input_dim=int(layout.hidden_dim),
        block_out=int(layout.tokens_per_cipher),
        rotations=rmsnorm_rotations(layout),
        include_default_rotations=False,
    )


@pytest.mark.gpu
def test_feature_major_fhe_residual_preserves_layout() -> None:
    rng = np.random.default_rng(26081402)
    layout = FeatureMajorPrefillLayout(
        seq_len=128, hidden_dim=128, slots=8192
    )
    left = rng.normal(scale=0.02, size=(128, 128)).astype(np.float32)
    right = rng.normal(scale=0.02, size=(128, 128)).astype(np.float32)
    context = _small_feature_major_context(layout, depth=4)
    left_ciphers = tuple(
        context.encrypt(payload, level=0, slots=layout.slots)
        for payload in layout.pack(left)
    )
    right_ciphers = tuple(
        context.encrypt(payload, level=1, slots=layout.slots)
        for payload in layout.pack(right)
    )
    result = None
    try:
        result = residual_add(
            left_ciphers,
            right_ciphers,
            layout=layout,
            crypto_context=context,
            verify=True,
            expected_left=left,
            expected_right=right,
        )
        assert result.output_levels == (1,)
        # Both pairs entered at different levels, so both were aligned.
        assert result.modulus_aligned_pairs == 2
        assert result.max_abs_diff is not None and result.max_abs_diff < 1e-7
    finally:
        if result is not None:
            result.release()
        for cipher in left_ciphers + right_ciphers:
            release_if_supported(cipher)


def test_periodic_slim_row_mask_is_periodic_and_distinct() -> None:
    from llama3fhe.operators.norm.rmsnorm import (
        _mapped_active_offset,
        _slim_row_mask,
        slim_period_slots,
    )

    layout = FeatureMajorPrefillLayout()
    period = slim_period_slots(layout, 16)
    assert period == 16 * 8 == 128
    legacy = _slim_row_mask(layout, 5, value=1.0)
    periodic = _slim_row_mask(layout, 5, value=1.0, period_rows=16)
    # the legacy row is the first period of the periodic mask
    np.testing.assert_array_equal(periodic[:period], legacy[:period])
    np.testing.assert_array_equal(
        periodic.reshape(-1, period), np.tile(periodic[:period], (layout.slots // period, 1))
    )
    # distinct ciphers of one rank land on disjoint rows; the offsets add up
    rows = (4, 5)
    offset = _mapped_active_offset(layout, 2.0, cipher_indices=rows, period_rows=16)
    assert offset.max() == 2.0
    assert int((offset != 0).sum()) == len(rows) * 8 * (layout.slots // period)
    # 16 % 8 == 0 collides with cipher 0 under period 8
    with pytest.raises(ValueError):
        from llama3fhe.operators.norm.rmsnorm import _check_slim_period

        _check_slim_period(layout, 8, (0, 8))
    with pytest.raises(ValueError):
        _slim_row_mask(layout, 0, value=1.0, period_rows=12)


def test_slim_refresh_pre_scale_defaults_for_dense_periodic_rows() -> None:
    """A dense slim row (one rank owning most of a period) gets the 1/16 pre-scale by default."""

    from llama3fhe.operators.norm.rmsnorm import (
        SLIM_DENSE_REFRESH_FACTOR,
        _slim_refresh_pre_scale_factor,
        _slim_row_density,
    )

    hidden = 4096
    assert _slim_row_density(2, 16, hidden_dim=hidden) == 0.125
    # 4 ranks -> 1/4 stays on the configured factor, 2 ranks (1/2) and 1 rank (1) are dense
    assert _slim_row_density(4, 16, hidden_dim=hidden) == 0.25
    assert _slim_row_density(16, 16, hidden_dim=hidden) == 1.0
    # legacy packing: one row per cipher out of hidden_dim
    assert _slim_row_density(16, None, hidden_dim=hidden) == 16 / hidden
    common = dict(current_limbs=6, required_output_limbs=18)
    assert _slim_refresh_pre_scale_factor(configured_factor=1.0, slim_density=0.125, **common) == 1.0
    assert _slim_refresh_pre_scale_factor(configured_factor=1.0, slim_density=0.25, **common) == 1.0
    assert (
        _slim_refresh_pre_scale_factor(configured_factor=1.0, slim_density=0.5, **common)
        == SLIM_DENSE_REFRESH_FACTOR
    )
    assert (
        _slim_refresh_pre_scale_factor(configured_factor=1.0, slim_density=1.0, **common)
        == SLIM_DENSE_REFRESH_FACTOR
    )
    # an explicit factor wins; no refresh / too few limbs -> no pre-scale at all
    assert _slim_refresh_pre_scale_factor(configured_factor=0.03125, slim_density=1.0, **common) == 0.03125
    assert _slim_refresh_pre_scale_factor(
        configured_factor=1.0, slim_density=1.0, current_limbs=20, required_output_limbs=18
    ) == 1.0
    assert _slim_refresh_pre_scale_factor(
        configured_factor=1.0, slim_density=1.0, current_limbs=2, required_output_limbs=18
    ) == 1.0


@pytest.mark.gpu
@pytest.mark.parametrize("slim_period_rows", [None, 4, 2])
def test_feature_major_fhe_rmsnorm_periodic_slim_pack_matches_legacy(
    slim_period_rows: int | None,
) -> None:
    """The periodic slim pack is a pure re-layout: same output, same levels."""

    rng = np.random.default_rng(26081404)
    layout = FeatureMajorPrefillLayout(
        seq_len=128, hidden_dim=128, slots=8192
    )
    # O(1) mean squares keep the periodic pack mask (alpha / hidden_dim, now
    # dense in the slot vector) inside the encoder's 61-bit coefficient range
    # at this small context's 2^59 scaling factor.
    hidden = rng.normal(scale=0.5, size=(128, 128)).astype(np.float32)
    gamma = rng.normal(loc=1.0, scale=0.01, size=(128,)).astype(np.float32)
    eps = 1e-5
    means = np.mean(np.square(hidden, dtype=np.float64), axis=-1)
    fit_interval = (float(np.min(means) * 0.8), float(np.max(means) * 1.2))
    coeffs = {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: 1.0 / np.sqrt(value + eps),
            fit_interval[0],
            fit_interval[1],
            16,
        ),
        "fit_interval": fit_interval,
        "eps": eps,
    }
    context = _small_feature_major_context(layout, depth=16)
    input_ciphers = tuple(
        context.encrypt(payload, level=0, slots=layout.slots)
        for payload in layout.pack(hidden)
    )
    result = None
    try:
        result = rmsnorm_fhe(
            input_ciphers,
            gamma,
            coeffs=coeffs,
            layout=layout,
            crypto_context=context,
            verify=True,
            expected_input=hidden,
            slim_period_rows=slim_period_rows,
        )
        assert result.output_levels == (10,)
        assert result.max_abs_diff is not None and result.max_abs_diff < 3e-3
    finally:
        if result is not None:
            result.release()
        for cipher in input_ciphers:
            release_if_supported(cipher)


@pytest.mark.gpu
@pytest.mark.parametrize("defer_output_scale", [False, True])
def test_feature_major_fhe_rmsnorm_uses_one_slim_polynomial(
    defer_output_scale: bool,
) -> None:
    rng = np.random.default_rng(26081403)
    layout = FeatureMajorPrefillLayout(
        seq_len=128, hidden_dim=128, slots=8192
    )
    hidden = rng.normal(scale=0.02, size=(128, 128)).astype(np.float32)
    gamma = rng.normal(loc=1.0, scale=0.01, size=(128,)).astype(np.float32)
    eps = 1e-5
    means = np.mean(np.square(hidden, dtype=np.float64), axis=-1)
    fit_interval = (float(np.min(means) * 0.8), float(np.max(means) * 1.2))
    coeffs = {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: 1.0 / np.sqrt(value + eps),
            fit_interval[0],
            fit_interval[1],
            16,
        ),
        "fit_interval": fit_interval,
        "eps": eps,
    }
    context = _small_feature_major_context(layout, depth=16)
    input_value_scale = 2.0 if defer_output_scale else 1.0
    physical_hidden = hidden * input_value_scale
    input_ciphers = tuple(
        context.encrypt(payload, level=0, slots=layout.slots)
        for payload in layout.pack(physical_hidden)
    )
    result = None
    try:
        result = rmsnorm_fhe(
            input_ciphers,
            gamma,
            coeffs=coeffs,
            layout=layout,
            crypto_context=context,
            verify=True,
            expected_input=hidden,
            input_value_scale=input_value_scale,
            defer_output_scale=bool(defer_output_scale),
        )
        assert result.complexity.rsqrt_polynomial_evaluations == 1
        assert result.output_levels == (
            (9,) if defer_output_scale else (10,)
        )
        assert result.max_abs_diff is not None and result.max_abs_diff < 3e-4
        if defer_output_scale:
            assert result.deferred_output_scale is not None
            assert result.output is not None
            logical = result.output * result.deferred_output_scale[None, :]
            expected = rmsnorm_numpy(hidden, gamma, eps=eps)
            np.testing.assert_allclose(logical, expected, atol=5e-4, rtol=5e-4)
        else:
            assert result.deferred_output_scale is None
    finally:
        if result is not None:
            result.release()
        for cipher in input_ciphers:
            release_if_supported(cipher)


