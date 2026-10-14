from __future__ import annotations

from llama3fhe.graph import (
    LayerSchedule,
    operation_audit,
    required_rotations,
)
from llama3fhe.layouts import AttentionPairLayout, FeatureMajorPrefillLayout


def test_feature_major_layer0_static_contract() -> None:
    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()

    audit = operation_audit(feature, attention)
    stages = audit["stages"]

    assert audit["shape"] == {
        "seq_len": 128,
        "hidden_dim": 4096,
        "intermediate_width": 14336,
        "canonical_cipher_count": 16,
    }
    assert stages["feature_major_qkv_to_attention"][
        "requires_input_layout_conversion"
    ] is False
    assert stages["feature_major_qkv_to_attention"][
        "pt_ct_multiplications"
    ] == 72_876
    assert stages["pairfm_output_conversion"]["conversion_pt_ct"] == 6_400
    assert stages["pairfm_output_conversion"]["conversion_depth"] == 3
    assert stages["w_o"]["pt_ct_multiplications"] == 32_768
    assert stages["mlp"]["pt_ct_multiplications"] == 393_216
    # Nine, not ten: the production schedule fuses the SiLU domain map into
    # the gate weight, which saves one level. This test builds the audit from
    # LayerSchedule() defaults, so it sees whatever production sees.
    assert stages["mlp"]["multiplicative_depth"] == 9
    # Zero, because the production schedule borrows the pre-bootstrap
    # residual instead of materializing a refreshed copy.
    assert stages["residuals"]["bootstrap_identity_rescales"] == 0
    assert audit["settings"]["softmax_lambda_bootstrap_iterations"] == 1
    assert (
        audit["settings"]["match_softmax_single_pass_bootstrap_level"]
        is True
    )
    # The audit and the runtime read the same LayerSchedule, so the audit's
    # stated settings are the execution defaults by construction.
    defaults = LayerSchedule()
    assert defaults.softmax_lambda_bootstrap_iterations == 1
    assert defaults.match_softmax_single_pass_bootstrap_level is True
    # Disabling the borrow is what costs one identity rescale per cipher on
    # each of the two residual paths.
    materialized = operation_audit(
        feature,
        attention,
        schedule=LayerSchedule(preserve_prebootstrap_residual=False),
    )
    assert materialized["stages"]["residuals"][
        "bootstrap_identity_rescales"
    ] == 32


def test_feature_major_layer0_rotation_union_is_compact_and_has_conjugation() -> None:
    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    rotations = required_rotations(feature, attention)

    # The bounded/reused-baby backend requests the exact local/anchor key
    # union instead of the old direct-baby superset.
    assert len(rotations) == 713
    assert len(rotations) == len(set(rotations))
    assert 4 * feature.slots - 1 in rotations

    direct_64 = required_rotations(
        feature,
        attention,
        schedule=LayerSchedule(linear_baby_steps=64, linear_anchor_step=64),
    )
    assert len(direct_64) == 713
    assert 8 * 63 in direct_64
