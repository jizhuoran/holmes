"""Shifted-lane (SL) decode linears wired into the decode operators.

CPU: the decode policy (:func:`llama3fhe.graph._decode_linear_config`) and the
key families.  GPU: the SL PairFold QKV kernel (folded / partial + fold) and
the SL decode MLP shard on a small layout, both against numpy references.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.graph import _decode_linear_config, _linear_config
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.linear import LinearOperatorConfig
from llama3fhe.parallel.decode.decode_keys import decode_rotation_families
from llama3fhe.parallel.decode.decode_stations import decode_qkv_chunk_count
from llama3fhe.parallel.presets import preset_by_name


def _schedules():
    base = preset_by_name("a100-high").layer_schedule()
    return base, replace(base, decode_shifted_lanes=True)


def test_decode_linear_config_is_lane_replicated_unless_enabled() -> None:
    base, sl = _schedules()
    assert _decode_linear_config(base, base.qkv_max_plaintext_rows, base.qkv_hoist_strategy) == _linear_config(
        base, base.qkv_max_plaintext_rows, base.qkv_hoist_strategy
    )
    config = _decode_linear_config(sl, sl.qkv_max_plaintext_rows, sl.qkv_hoist_strategy)
    assert config.lane_shift == 512
    assert config.baby_steps == sl.decode_sl_baby_steps == 32
    assert config.max_plaintext_rows == 64
    plan = config.plan(dimension=4096, token_lanes=8)
    assert plan.shifted_lanes and plan.diagonals == 512 and plan.plaintext_chunks == 8
    for baby in (8, 16, 32, 64):
        plan = _decode_linear_config(
            replace(sl, decode_sl_baby_steps=baby), 512, "normal"
        ).plan(dimension=4096, token_lanes=8)
        assert plan.plaintext_chunks == 8, baby
    assert decode_qkv_chunk_count(sl) == 8
    assert decode_qkv_chunk_count(base) == 4096 // int(base.qkv_max_plaintext_rows)


def test_decode_key_families_carry_fold_and_expand_offsets_only_under_sl() -> None:
    base, sl = _schedules()
    feature_layout = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=4096, slots=32768)
    lr = decode_rotation_families(base, feature_layout=feature_layout, pairfold_layout=PairFold128Layout())
    sl_fam = decode_rotation_families(sl, feature_layout=feature_layout, pairfold_layout=PairFold128Layout())
    expand = {4095 * lane for lane in range(1, 8)}
    for family in ("qkv", "mlp"):
        assert {1, 2, 4}.issubset(sl_fam[family]) and expand.issubset(sl_fam[family])
        assert not ({1, 2, 4} & set(lr[family]))
    # W_O and the non-linear families are untouched by the SL switch
    for family in ("w_o", "rmsnorm", "route", "attention"):
        assert lr[family] == sl_fam[family]


def _pairfold_qkv_setup(operator_config: LinearOperatorConfig):
    from llama3fhe.backend import create_sim_context
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.linear.pairfold.pairfold_qkv import pairfold_qkv_rotations
    from llama3fhe.operators.linear.prefill.qkv_kernel import ComplexTokenDiagonalLayout

    layout = PairFold128Layout()
    dimension = int(layout.carrier_dimension)
    rng = np.random.default_rng(20261003)
    hidden = rng.normal(scale=0.05, size=(1, dimension)).astype(np.float32)
    q_weight = rng.normal(scale=0.02, size=(layout.query_heads * layout.head_dim, dimension)).astype(np.float32)
    k_weight = rng.normal(scale=0.02, size=(layout.key_value_heads * layout.head_dim, dimension)).astype(np.float32)
    v_weight = rng.normal(scale=0.02, size=(layout.key_value_heads * layout.head_dim, dimension)).astype(np.float32)
    logical_query = (hidden.astype(np.float64) @ q_weight.T).reshape(layout.query_heads, layout.head_dim)
    logical_key = (hidden.astype(np.float64) @ k_weight.T).reshape(layout.key_value_heads, layout.head_dim)
    logical_value = (hidden.astype(np.float64) @ v_weight.T).reshape(layout.key_value_heads, layout.head_dim)
    expected_carrier = layout.pack_qkv_carrier(logical_query, logical_key, logical_value)[0]

    config = Llama3CKKSConfig()
    config.simulator.logN = 16
    config.simulator.maxLevelsRemaining = 4
    config.simulator.dnum = 1
    config.simulator.device = "cuda"
    config.qkv_linear.dtype = "float64"
    context = create_sim_context(
        config.simulator,
        input_dim=dimension,
        block_out=layout.native_token_lanes,
        rotations=pairfold_qkv_rotations(layout, operator_config=operator_config),
        include_default_rotations=False,
    )
    input_layout = ComplexTokenDiagonalLayout(seq_len=1, dimension=dimension, slots=layout.slots)
    cipher = context.encrypt(input_layout.pack(hidden)[0], level=0, slots=layout.slots)
    active = np.asarray([c * layout.native_token_lanes for c in layout.active_carrier_channels()], dtype=np.int64)
    return SimpleNamespace(
        layout=layout, config=config, context=context, hidden=hidden, weights=(q_weight, k_weight, v_weight),
        cipher=cipher, expected_carrier=expected_carrier, active_slots=active,
    )


@pytest.mark.gpu
def test_shifted_lane_pairfold_qkv_matches_carrier_and_partial_fold() -> None:
    from llama3fhe.backend import release_if_supported
    from llama3fhe.operators.attention.codec import decrypt_complex_rows
    from llama3fhe.operators.linear.pairfold.pairfold_qkv import pairfold_qkv_fhe, pairfold_qkv_fold_carrier

    operator_config = LinearOperatorConfig(
        baby_steps=16, baby_anchor_step=16, max_plaintext_rows=64, lane_shift=512, raw_weight_policy="device"
    )
    s = _pairfold_qkv_setup(operator_config)
    q_weight, k_weight, v_weight = s.weights
    common = dict(layout=s.layout, config=s.config, operator_config=operator_config, crypto_context=s.context)
    full = None
    partials = []
    folded = None
    try:
        full = pairfold_qkv_fhe(s.hidden, q_weight, k_weight, v_weight, input_ciphers=(s.cipher,), verify=True, **common)
        assert full.max_abs_diff is not None and full.max_abs_diff < 3e-6
        carrier = decrypt_complex_rows((full.carrier_cipher,), crypto_context=s.context)[0]
        np.testing.assert_allclose(carrier[s.active_slots], s.expected_carrier[s.active_slots], atol=3e-6, rtol=3e-5)
        assert "input_lane_expand" in full.stage_seconds
        # the input is not consumed and stays lane-0 clean
        back = np.asarray(s.context.decrypt(s.cipher)).reshape(-1, 8)
        assert np.abs(back[:, 1:]).max() < 1e-6

        # two "ranks": chunks (0..3) and (4..7), unfolded partials, sum, fold once
        for chunks in ((0, 1, 2, 3), (4, 5, 6, 7)):
            partials.append(
                pairfold_qkv_fhe(
                    s.hidden, q_weight, k_weight, v_weight, input_ciphers=(s.cipher,), verify=False,
                    chunk_indices=chunks, fold=False, **common,
                )
            )
        summed = s.context.fhe.homo_add(partials[0].carrier_cipher, partials[1].carrier_cipher, s.context.context)
        try:
            folded = pairfold_qkv_fold_carrier(summed, layout=s.layout, operator_config=operator_config, crypto_context=s.context)
        finally:
            release_if_supported(summed)
        carrier2 = decrypt_complex_rows((folded,), crypto_context=s.context)[0]
        np.testing.assert_allclose(carrier2[s.active_slots], s.expected_carrier[s.active_slots], atol=3e-6, rtol=3e-5)
    finally:
        if full is not None:
            full.release()
        for p in partials:
            p.release()
        release_if_supported(folded)
        release_if_supported(s.cipher)


@pytest.mark.gpu
@pytest.mark.parametrize("fuse_silu_domain_map", [False, True])
def test_shifted_lane_decode_mlp_shard_small_matches_reference(fuse_silu_domain_map: bool) -> None:
    from llama3fhe.approx.chebyshev import chebyshev_ps_mul_depth, eval_chebyshev_coefficients
    from llama3fhe.backend import create_sim_context, release_if_supported
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.linear.prefill.mlp import mlp_rotations
    from llama3fhe.operators.linear.shard.decode_mlp_shard import decode_mlp_shard_fhe
    from llama3fhe.parallel.decode.decode_ownership import DecodeRankOwnership

    rng = np.random.default_rng(26081433)
    layout = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=512, slots=8192)  # 16 token lanes
    hidden = rng.normal(scale=0.01, size=(1, 512)).astype(np.float32)
    gate = rng.normal(scale=0.01, size=(768, 512)).astype(np.float32)
    up = rng.normal(scale=0.01, size=(768, 512)).astype(np.float32)
    down = rng.normal(scale=0.01, size=(512, 768)).astype(np.float32)
    config = Llama3CKKSConfig()
    config.simulator.logN = 14
    config.simulator.maxLevelsRemaining = 18
    config.simulator.device = "cuda"
    config.qkv_linear.input_level = 0
    fit_interval = (-0.1, 0.1)
    silu_entry = {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: value / (1.0 + np.exp(-value)), fit_interval[0], fit_interval[1], 16
        ),
        "fit_interval": fit_interval,
    }
    # lane_shift = 512 / 16 = 32 diagonals, 8-row chunks -> 4 chunks per page
    sl_linear = LinearOperatorConfig(
        baby_steps=8, baby_anchor_step=8, max_plaintext_rows=8, hoist_strategy="normal", lane_shift=32,
        raw_weight_policy="device",
    )
    context = create_sim_context(
        config.simulator,
        input_dim=layout.hidden_dim,
        block_out=layout.tokens_per_cipher,
        rotations=mlp_rotations(layout, operator_config=sl_linear),
        include_default_rotations=False,
    )
    inputs = tuple(context.encrypt(row, level=0, slots=layout.slots) for row in layout.pack(hidden))
    transport = SimpleNamespace(world_size=1, rank=0)
    ownership = DecodeRankOwnership(0, 1, mlp_pages=2)
    result = decode_mlp_shard_fhe(
        hidden, gate, up, down,
        silu_entry=silu_entry, layout=layout, ownership=ownership, transport=transport, config=config,
        operator_config=sl_linear, crypto_context=context, input_ciphers=inputs, verify=True,
        fuse_silu_domain_map=bool(fuse_silu_domain_map), tag="mlp", split_gate_up=True,
    )
    try:
        assert len(result.ciphers) == 1
        assert result.max_abs_diff is not None and result.max_abs_diff < 2e-5, result.max_abs_diff
        assert {"input_lane_expand", "gate_up_fold_mask", "down_fold_mask", "silu_gated"} <= set(result.stage_seconds)
        # lanes 1.. of the output are clean (lane-0 mask at the exit)
        out = np.asarray(context.decrypt(result.ciphers[0])).real.reshape(-1, layout.tokens_per_cipher)
        assert np.abs(out[:, 1:]).max() < 1e-5
        # two extra levels vs the lane-replicated kernel (two lane-0 masks)
        consumed = int(result.output_levels[0]) - int(context.level_for_cipher(inputs[0]))
        assert consumed == chebyshev_ps_mul_depth(silu_entry["coefficients"]) + 4 - int(fuse_silu_domain_map) + 2
    finally:
        result.release()
        for cipher in inputs:
            release_if_supported(cipher)
