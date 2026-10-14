"""CPU contracts for stage-exact Delta-attention rotation keys."""

from __future__ import annotations

import pytest

from llama3fhe import graph
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.attention.audit import (
    delta_attention_rotations,
    delta_pv_rotations,
    delta_qk_rotations,
)
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.schedule import LayerSchedule


def _production_qk_rotations(
    layout: AttentionPairLayout, config: AttentionOperatorConfig
) -> tuple[int, ...]:
    return delta_qk_rotations(
        layout,
        complex_feature_folding=config.qk_complex_feature_folding,
        key_features_pre_folded=config.qk_key_features_pre_folded,
        query_features_pre_folded=config.qk_query_features_pre_folded,
        log_n=16,
    )


def _production_pv_rotations(
    layout: AttentionPairLayout,
    config: AttentionOperatorConfig,
    *,
    inputs_pre_folded: bool,
) -> tuple[int, ...]:
    # pv_delta_fhe treats its runtime inputs_pre_folded fact as applying to
    # both operands, so the audit maps it to both stage-level parameters.
    return delta_pv_rotations(
        layout,
        complex_delta_folding=config.pv_complex_delta_folding,
        probability_pre_folded=inputs_pre_folded,
        value_pre_folded=inputs_pre_folded,
        log_n=16,
    )


def test_production_delta_rotation_key_counts_are_stage_exact() -> None:
    layout = AttentionPairLayout()
    config = AttentionOperatorConfig()

    qk = _production_qk_rotations(layout, config)
    pv = _production_pv_rotations(layout, config, inputs_pre_folded=False)
    pv_pre_folded = _production_pv_rotations(
        layout, config, inputs_pre_folded=True
    )

    assert len(qk) == 190
    assert len(pv) == 193
    assert len(pv_pre_folded) == 190
    assert len(qk) == len(set(qk))
    assert len(pv) == len(set(pv))
    assert len(pv_pre_folded) == len(set(pv_pre_folded))


@pytest.mark.parametrize(
    (
        "qk_folded",
        "key_pre_folded",
        "query_pre_folded",
        "pv_folded",
        "probability_pre_folded",
        "value_pre_folded",
    ),
    [
        (False, False, False, False, False, False),
        (True, False, False, False, False, False),
        (True, True, True, True, False, False),
        (True, True, True, True, True, True),
        (True, False, True, True, True, False),
    ],
)
def test_legacy_delta_rotation_union_matches_stage_apis(
    qk_folded: bool,
    key_pre_folded: bool,
    query_pre_folded: bool,
    pv_folded: bool,
    probability_pre_folded: bool,
    value_pre_folded: bool,
) -> None:
    layout = AttentionPairLayout()
    qk = delta_qk_rotations(
        layout,
        complex_feature_folding=qk_folded,
        key_features_pre_folded=key_pre_folded,
        query_features_pre_folded=query_pre_folded,
    )
    pv = delta_pv_rotations(
        layout,
        complex_delta_folding=pv_folded,
        probability_pre_folded=probability_pre_folded,
        value_pre_folded=value_pre_folded,
    )
    legacy = delta_attention_rotations(
        layout,
        include_qk_feature_fold=qk_folded,
        qk_key_features_pre_folded=key_pre_folded,
        qk_query_features_pre_folded=query_pre_folded,
        include_pv_delta_fold=pv_folded,
        pv_probability_pre_folded=probability_pre_folded,
        pv_value_pre_folded=value_pre_folded,
    )

    assert set(legacy) == set(qk) | set(pv)


def test_stage_rotation_apis_validate_prefold_contracts() -> None:
    layout = AttentionPairLayout()

    with pytest.raises(ValueError, match="pre-folded K"):
        delta_qk_rotations(layout, key_features_pre_folded=True)
    with pytest.raises(ValueError, match="pre-folded Q"):
        delta_qk_rotations(layout, query_features_pre_folded=True)
    with pytest.raises(ValueError, match="pre-folded P/V"):
        delta_pv_rotations(layout, probability_pre_folded=True)
    with pytest.raises(ValueError, match="pre-folded P/V"):
        delta_pv_rotations(layout, value_pre_folded=True)


def test_qk_eviction_keep_set_covers_pv_prefold_helpers() -> None:
    attention = AttentionPairLayout()
    config = AttentionOperatorConfig(pv_prefold_after_softmax=True)

    keep = graph._attention_exclusive_cache_keep_rotations(
        FeatureMajorPrefillLayout(),
        attention,
        schedule=LayerSchedule(pv_prefold_after_softmax=True),
        operator_config=config,
        log_n=16,
    )
    qk_keep = set(keep["qk_to_softmax"])
    pv_keep = set(keep["pv_to_pairfm"])
    complete_pv = set(
        delta_pv_rotations(
            attention,
            complex_delta_folding=True,
            probability_pre_folded=False,
            value_pre_folded=False,
            log_n=16,
        )
    )

    assert complete_pv <= qk_keep
    assert {-128, 128, 16384} <= qk_keep
    assert pv_keep <= qk_keep
