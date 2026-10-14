"""Decode / Prefill+Decode rotation-key plans of the T2 benches (CPU only)."""

from __future__ import annotations

import pytest

from bench.t2_decode import (
    decode_key_plan,
    decode_layouts,
    decode_schedule,
    resolve_attention_limb_cap,
)
from bench.t2_prefill_decode import (
    combined_preset,
    decode_preset_and_schedule,
    rank_key_plan,
    rank_phase_key_plans,
)
from llama3fhe.operators.attention.pairfold.pairfold_decode import (
    pv_pairfold_rotations,
    pv_probability_route_rotations,
)
from llama3fhe.parallel.decode.decode_keys import build_decode_key_plan
from llama3fhe.parallel.prefill.keys import normalize_rotation, rotation_key_bytes
from llama3fhe.parallel.presets import preset_by_name

GIB = 2**30
pytestmark = pytest.mark.parametrize("preset_name", ["rtx6000pro-high", "a100-high"])


def _resident_gib(plan, *, alpha: int = 9, special_limbs: int = 8) -> float:
    limits = plan.limb_limits()
    return sum(
        rotation_key_bytes(ring_dim=plan.ring_dim, limbs=limits.get(r, plan.full_limbs), alpha=alpha,
                           special_limbs=special_limbs)
        for r in plan.resident_rotations
    ) / GIB


def test_resolve_attention_limb_cap_defaults_to_the_preset_and_zero_means_uncapped(preset_name) -> None:
    preset = preset_by_name(preset_name)
    assert resolve_attention_limb_cap(preset, None) == 13
    assert resolve_attention_limb_cap(preset, 0) is None
    assert resolve_attention_limb_cap(preset, 11) == 11


def test_decode_plan_caps_attn_core_by_default(preset_name) -> None:
    preset = preset_by_name(preset_name)
    capped = decode_key_plan(preset, kv_limbs=12, attention_limb_cap=None, attn_resident=False, layers=range(32))
    uncapped = decode_key_plan(preset, kv_limbs=12, attention_limb_cap=0, attn_resident=False, layers=range(32))
    assert capped.tier("attn_core").limb_cap == 13
    assert uncapped.tier("attn_core").limb_cap is None
    assert 13 < capped.full_limbs
    assert len(capped.limb_limits()) > len(uncapped.limb_limits())


def test_decode_probability_route_keys_can_keep_more_limbs_than_other_attention_keys(preset_name) -> None:
    preset = preset_by_name(preset_name)
    feature, attention, pairfold = decode_layouts()
    schedule = decode_schedule(preset, kv_limbs=12)
    kwargs = dict(
        schedule=schedule,
        feature_layout=feature,
        attention_layout=attention,
        pairfold_layout=pairfold,
        attention_limb_cap=13,
    )
    baseline = build_decode_key_plan(preset, **kwargs)
    raised = build_decode_key_plan(preset, probability_route_limb_cap=19, **kwargs)
    route = {
        normalize_rotation(rotation, ring_dim=raised.ring_dim)
        for rotation in pv_probability_route_rotations(pairfold)
    }
    assert route <= {
        normalize_rotation(rotation, ring_dim=raised.ring_dim)
        for rotation in pv_pairfold_rotations(pairfold)
    }
    assert 17408 in route
    assert baseline.limb_limits()[17408] == 13
    assert raised.limb_limits()[17408] == 19
    assert set(raised.union) == set(baseline.union)
    assert raised.tier("attn_pv").rotation_set == route
    for rotation, limit in baseline.limb_limits().items():
        if rotation not in route:
            assert raised.limb_limits()[rotation] == limit


def test_combined_plan_cap_removes_the_full_chain_promotion_of_shared_attention_rotations(preset_name) -> None:
    preset = combined_preset(preset_by_name(preset_name))
    decode_preset, schedule = decode_preset_and_schedule(preset, kv_limbs=12, shifted_lanes=True, sl_baby=16)
    layers = tuple(range(32))
    capped, sizes = rank_key_plan(preset, 0, decode_preset=decode_preset, decode_schedule=schedule, layers=layers)
    uncapped, _ = rank_key_plan(preset, 0, decode_preset=decode_preset, decode_schedule=schedule, layers=layers,
                                attention_limb_cap=0)
    assert set(capped.union) == set(uncapped.union)
    assert sizes["prefill_union"] > 0 and sizes["decode_union"] > 0
    saved = _resident_gib(uncapped) - _resident_gib(capped)
    assert saved > 6.0, saved
    assert _resident_gib(capped) < 48.0


def test_phase_prefill_plan_does_not_warm_decode_only_bootstrap_rotations(preset_name) -> None:
    preset = combined_preset(preset_by_name(preset_name))
    decode_preset, schedule = decode_preset_and_schedule(preset, kv_limbs=12, shifted_lanes=True, sl_baby=16)
    merged, sizes, prefill_phase, decode = rank_phase_key_plans(
        preset, 0, decode_preset=decode_preset, decode_schedule=schedule, layers=tuple(range(32))
    )

    # The merged context still contains the complete union used by both
    # phases.  Only the warm-up target is stripped, so Decode can load its
    # bootstrap keys after the Prefill peak has passed.
    assert len(prefill_phase.tier("bootstrap")) < len(merged.tier("bootstrap"))
    assert len(prefill_phase.union) < sizes["prefill_union"]
    assert set(merged.union) == set(prefill_phase.union) | set(decode.union)
