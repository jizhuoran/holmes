from __future__ import annotations

import inspect

import numpy as np
import pytest

from llama3fhe.layouts.attention import (
    AttentionPairLayout,
)
from llama3fhe.layouts.feature_major import (
    FeatureMajorPrefillLayout,
)
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.operators.attention.prefill import input as attention_input_module
from llama3fhe.operators.attention import rope_cos_sin
from llama3fhe.operators.attention.audit import (
    delta_attention_rotations,
    delta_softmax_alg2_complexity,
    delta_softmax_rotations,
)
from llama3fhe.operators.attention.config import (
    AttentionOperatorConfig,
)
from llama3fhe.operators.attention.prefill.input import (
    complex_token_attention_rotations,
)
from llama3fhe.operators.attention.prefill.operator import (
    attention_fhe_delta_alg2,
    attention_to_w_o_rotations,
)
from llama3fhe.operators.attention.prefill.output import (
    pairfm_output_rotations,
)
from llama3fhe.operators.attention.prefill.softmax import (
    DeltaSoftmaxAlg2Result,
    delta_causal_mask,
    softmax_alg2_fhe_delta,
)


def test_delta_softmax_result_release_detaches_and_is_idempotent() -> None:
    released: list[str] = []

    class _Cipher:
        def __init__(self, name: str) -> None:
            self.name = str(name)

        def release(self) -> None:
            released.append(self.name)
            assert result.probability_ciphers == ()

    outputs = (_Cipher("first"), _Cipher("second"))
    result = DeltaSoftmaxAlg2Result(
        probability_ciphers=outputs,
        stage_seconds={},
        output_levels=(),
    )

    result.release()
    result.release()

    assert result.probability_ciphers == ()
    assert released == ["first", "second"]


def test_delta_causal_mask_excludes_padding_and_safes_invalid_queries() -> None:
    layout = AttentionPairLayout()
    token_valid = np.zeros(layout.seq_len, dtype=bool)
    token_valid[-2:] = True
    mask = delta_causal_mask(
        layout, token_valid_mask=token_valid
    )[: 2 * layout.seq_len * layout.seq_len]
    valid = mask.reshape(layout.seq_len, layout.seq_len, 2)[:, :, 0]

    # Every padding query retains one harmless self sentinel, while valid
    # queries see only causal, non-padding keys.
    np.testing.assert_array_equal(valid[:, :126].sum(axis=0), np.ones(126))
    assert valid[:, 126].sum() == 1
    assert valid[:, 127].sum() == 2
    assert valid[0, 126] == 1
    assert valid[0, 127] == 1
    assert valid[127, 127] == 1


def test_attention_package_preserves_oracle_import_contract() -> None:
    cos, sin = rope_cos_sin(seq_len=4, head_dim=4)
    assert cos.shape == sin.shape == (4, 4)
    np.testing.assert_allclose(cos[:, :2], cos[:, 2:])


def test_attention_config_rejects_prefold_without_complex_pv() -> None:
    with pytest.raises(ValueError, match="requires pv_complex"):
        AttentionOperatorConfig(
            pv_complex_delta_folding=False,
            pv_prefold_after_softmax=True,
        )


def test_attention_config_bounds_sparse_input_batching() -> None:
    assert AttentionOperatorConfig(
        batched_sparse_input_bsgs=True,
        sparse_input_batch_cap=32,
    ).batched_sparse_input_bsgs
    with pytest.raises(ValueError, match="sparse_input_batch_cap"):
        AttentionOperatorConfig(sparse_input_batch_cap=33)


def test_attention_config_bounds_pairfm_post_refresh_limbs() -> None:
    assert AttentionOperatorConfig().pairfm_post_refresh_input_limbs == 6
    assert (
        AttentionOperatorConfig(
            pairfm_post_refresh_input_limbs=6
        ).pairfm_post_refresh_input_limbs
        == 6
    )
    with pytest.raises(ValueError, match="post_refresh_input_limbs"):
        AttentionOperatorConfig(pairfm_post_refresh_input_limbs=1)


def test_attention_config_defaults_to_matched_one_pass_lambda_refresh() -> None:
    config = AttentionOperatorConfig()
    assert config.softmax_lambda_bootstrap_iterations == 1
    assert config.match_softmax_single_pass_bootstrap_level is True

    reference = AttentionOperatorConfig(
        softmax_lambda_bootstrap_iterations=2,
        match_softmax_single_pass_bootstrap_level=False,
    )
    assert reference.softmax_lambda_bootstrap_iterations == 2
    assert reference.match_softmax_single_pass_bootstrap_level is False

    # The whole chain reads one AttentionOperatorConfig, so every schedule
    # default exists in exactly one place: neither the orchestrator nor the
    # kernels below it may declare a competing one.
    for entry in (attention_fhe_delta_alg2, softmax_alg2_fhe_delta):
        parameters = inspect.signature(entry).parameters
        assert "operator_config" in parameters
        assert not [
            name
            for name in parameters
            if name
            in {
                "lambda_bootstrap_iterations",
                "main_bootstrap_iterations",
                "match_single_pass_bootstrap_level",
                "reuse_round_square",
                "rotation_mode",
                "rotation_chunk_size",
            }
        ]


def test_attention_input_reads_its_sparse_scheduler_policy_from_the_config() -> None:
    """The sparse-BSGS policy is carried by the config, not by loose keywords.

    The batched scheduler, its per-call cap and the fallback switch changed
    performance in the 2026-08 sweep, so which values reach the stage matters.
    They are read off AttentionOperatorConfig, which is what the caller passes.
    """

    config = AttentionOperatorConfig(
        batched_sparse_input_bsgs=True,
        sparse_input_batch_cap=16,
        sparse_input_fallback=False,
    )
    parameters = inspect.signature(
        attention_input_module.prepare_attention_inputs_fhe
    ).parameters
    # One config object, not six exploded knobs.
    assert "operator_config" in parameters
    for exploded in (
        "batched_sparse_bsgs",
        "sparse_batch_cap",
        "sparse_scheduler_fallback",
        "sparse_max_baby_offsets",
        "shear_baby_steps",
    ):
        assert exploded not in parameters, exploded

    # The values the stage reads come from the config object itself.
    assert config.batched_sparse_input_bsgs is True
    assert config.sparse_input_batch_cap == 16
    assert config.sparse_input_fallback is False


def test_prefill_shared_sparse_routes_preserve_transform_and_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V_C and Delta-V sharing keeps the two output axes deterministic."""

    class _Cipher:
        def __init__(self, name: str) -> None:
            self.name = name
            self.release_count = 0

        def release(self) -> None:
            self.release_count += 1

    inputs = (_Cipher("input0"), _Cipher("input1"))
    plans = (
        ({}, attention_input_module.SparseDiagonalPlan(1, (0,), (0,), (0,)), "value_c"),
        ({}, attention_input_module.SparseDiagonalPlan(1, (0,), (0,), (0,)), "value_delta"),
    )
    calls: list[str] = []

    def shared(cipher, *, transforms, profiles, **_kwargs):  # noqa: ANN001
        calls.append(cipher.name)
        for index, profile in enumerate(profiles):
            profile["wall_seconds"] = float(index + 1)
        return tuple(
            _Cipher(f"{name}.{cipher.name}") for _routes, _plan, name in transforms
        )

    monkeypatch.setattr(
        attention_input_module, "_apply_sparse_routes_batched_shared", shared
    )
    profiles = [{}, {}]
    outputs = attention_input_module._apply_sparse_routes_shared_selected(
        inputs,
        transforms=plans,
        slots=8,
        crypto_context=object(),
        use_batched=True,
        batch_cap=32,
        fallback_on_error=False,
        profiles=profiles,
    )

    assert calls == ["input0", "input1"]
    assert [[cipher.name for cipher in group] for group in outputs] == [
        ["value_c.input0", "value_c.input1"],
        ["value_delta.input0", "value_delta.input1"],
    ]
    assert profiles[0]["wall_seconds"] == 2.0
    assert profiles[1]["wall_seconds"] == 4.0
    assert all(cipher.release_count == 0 for cipher in inputs)
    for group in outputs:
        for cipher in group:
            cipher.release()


def test_prefill_shared_sparse_routes_fall_back_to_independent_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cipher:
        def __init__(self, name: str) -> None:
            self.name = name

        def release(self) -> None:
            pass

    inputs = (_Cipher("input0"),)
    plans = (
        ({}, attention_input_module.SparseDiagonalPlan(1, (0,), (0,), (0,)), "value_c"),
        ({}, attention_input_module.SparseDiagonalPlan(1, (0,), (0,), (0,)), "value_delta"),
    )
    selected_calls: list[str] = []

    def fail_shared(*_args, **_kwargs):  # noqa: ANN001
        raise RuntimeError("batch API unavailable")

    def selected(_ciphers, *, function_prefix, profile, **_kwargs):  # noqa: ANN001
        selected_calls.append(function_prefix)
        profile["wall_seconds"] = 0.25
        profile["batched"] = 0.0
        return (_Cipher(function_prefix),)

    monkeypatch.setattr(
        attention_input_module, "_apply_sparse_routes_batched_shared", fail_shared
    )
    monkeypatch.setattr(
        attention_input_module, "_apply_sparse_routes_selected", selected
    )
    profiles = [{}, {}]
    with pytest.warns(RuntimeWarning, match="independent transforms"):
        outputs = attention_input_module._apply_sparse_routes_shared_selected(
            inputs,
            transforms=plans,
            slots=8,
            crypto_context=object(),
            use_batched=True,
            batch_cap=32,
            fallback_on_error=True,
            profiles=profiles,
        )

    assert selected_calls == ["value_c", "value_delta"]
    assert [[cipher.name for cipher in group] for group in outputs] == [
        ["value_c"],
        ["value_delta"],
    ]
    assert all(profile["shared_fallbacks"] == 1.0 for profile in profiles)


def test_attention_to_w_o_rotation_union_matches_component_union() -> None:
    carrier = LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768)
    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    config = AttentionOperatorConfig()
    expected = set(
        complex_token_attention_rotations(
            carrier,
            attention,
            sparse_max_baby_offsets=config.sparse_max_baby_offsets,
            shear_baby_steps=config.shear_baby_steps,
        )
    )
    expected.update(
        delta_attention_rotations(
            attention,
            include_qk_feature_fold=True,
            qk_key_features_pre_folded=True,
            qk_query_features_pre_folded=True,
            include_pv_delta_fold=True,
            pv_probability_pre_folded=False,
            pv_value_pre_folded=False,
        )
    )
    expected.update(delta_softmax_rotations(attention))
    expected.update(pairfm_output_rotations(feature, attention))
    actual = attention_to_w_o_rotations(
        carrier,
        feature,
        attention,
        operator_config=config,
    )
    assert set(actual) == expected
    assert len(actual) == len(set(actual))


def test_alg2_static_audit_has_one_causal_mask_before_rounds() -> None:
    layout = AttentionPairLayout()
    cost = delta_softmax_alg2_complexity(layout)

    # Four rounds retain the three CT-CT operations used by the current
    # implementation (row-sum square, y*lambda, output square), but no longer
    # spend 16 PT-CT multiplications and one level per round remasking zeros.
    assert cost.explicit_ct_ct_multiplications == 4 * 3 * 16
    assert cost.pt_ct_multiplications == 2 * 16 + 4 * (16 + 1 + 16)
    assert cost.lambda_bootstraps == 4

    reused = delta_softmax_alg2_complexity(
        layout, reuse_round_square=True
    )
    # Per round: 16 y-squares + one slim lambda-square + 16 final products.
    assert reused.explicit_ct_ct_multiplications == 4 * (2 * 16 + 1)
    assert reused.explicit_relinearizations == 4 * (2 * 16 + 1)
    assert reused.explicit_ct_ct_rescales == 4 * (2 * 16 + 1)

    direct_fill = delta_softmax_alg2_complexity(
        layout,
        reuse_round_square=True,
        direct_slim_inactive_fill=True,
    )
    fused_map = delta_softmax_alg2_complexity(
        layout,
        reuse_round_square=True,
        fuse_slim_chebyshev_map=True,
    )
    # The slim pack has already selected disjoint active rows.  Direct public
    # filling, and the stronger pack+Chebyshev-map fusion, both remove the
    # otherwise redundant one-plaintext active-row sanitizer per round.
    expected_pt_ct = 2 * 16 + 4 * (16 + 16)
    assert direct_fill.pt_ct_multiplications == expected_pt_ct
    assert fused_map.pt_ct_multiplications == expected_pt_ct
