"""Decode sparse bootstrap programs: schedule knobs, log_slots derivation, rotation plans."""

from __future__ import annotations

from llama3fhe.graph import decode_layer_rotations, decode_sparse_bootstrap_log_slots
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.model import with_decode_sparse_bootstrap
from llama3fhe.operators.nonlinear.bootstrap import BootstrapOperator
from llama3fhe.schedule import LayerSchedule


def _decode_layouts():
    prefill = FeatureMajorPrefillLayout()
    decode = FeatureMajorPrefillLayout(
        seq_len=1, hidden_dim=int(prefill.hidden_dim), slots=int(prefill.slots)
    )
    return decode, AttentionPairLayout()


def test_production_decode_layouts_want_slim_3_and_merged_page_12() -> None:
    decode, attention = _decode_layouts()
    assert decode_sparse_bootstrap_log_slots(decode, attention) == (3, 12)
    # period-1 slim row only / merged-page only
    assert decode_sparse_bootstrap_log_slots(decode, None) == (3,)
    assert decode_sparse_bootstrap_log_slots(
        decode, attention, slim_period_rows=None
    ) == (12,)
    assert decode_sparse_bootstrap_log_slots(
        decode, attention, slim_period_rows=None, merged_page_softmax=False
    ) == ()


def test_with_decode_sparse_bootstrap_fills_both_schedule_fields() -> None:
    decode, attention = _decode_layouts()
    base = LayerSchedule()
    assert base.bootstrap_sparse_log_slots == ()
    assert base.decode_rmsnorm_slim_period_rows is None
    schedule = with_decode_sparse_bootstrap(base, decode, attention)
    assert schedule.bootstrap_sparse_log_slots == (3, 12)
    assert schedule.decode_rmsnorm_slim_period_rows == 1
    # idempotent and merges user extras
    again = with_decode_sparse_bootstrap(schedule, decode, attention)
    assert again.bootstrap_sparse_log_slots == (3, 12)
    extra = with_decode_sparse_bootstrap(
        LayerSchedule(bootstrap_sparse_log_slots=(9,)), decode, attention
    )
    assert extra.bootstrap_sparse_log_slots == (3, 9, 12)


def test_decode_rotations_grow_by_exactly_the_sparse_program_rotations() -> None:
    decode, attention = _decode_layouts()
    dense = LayerSchedule()
    sparse = with_decode_sparse_bootstrap(dense, decode, attention)
    kwargs = dict(schedule=None, skip_attention=False, include_bootstrap=True)
    base = set(decode_layer_rotations(decode, attention, **{**kwargs, "schedule": dense}))
    grown = set(decode_layer_rotations(decode, attention, **{**kwargs, "schedule": sparse}))
    assert base <= grown
    expected_extra = set(
        BootstrapOperator.required_rotations(
            log_n=16,
            log_slots=int(dense.bootstrap_log_slots),
            level_budget=tuple(dense.bootstrap_level_budget),
            sparse_log_slots=(3, 12),
        )
    ) - set(
        BootstrapOperator.required_rotations(
            log_n=16,
            log_slots=int(dense.bootstrap_log_slots),
            level_budget=tuple(dense.bootstrap_level_budget),
        )
    )
    assert grown - base == {r for r in expected_extra if r} - base
    assert grown - base
    # without include_bootstrap the sparse programs change nothing
    assert set(decode_layer_rotations(decode, attention, schedule=dense, skip_attention=False)) == set(
        decode_layer_rotations(decode, attention, schedule=sparse, skip_attention=False)
    )
