from __future__ import annotations

import pytest

from llama3fhe.application.args import (
    DEFAULT_RMS,
    DEFAULT_SILU,
    DEFAULT_SOFTMAX,
    JOINT_TOL_V1_RMS,
    JOINT_TOL_V1_SILU,
    JOINT_TOL_V1_SOFTMAX,
    build_parser,
    resolve_poly_paths,
)
from llama3fhe.application.settings import resolve_settings
from llama3fhe.config import Llama3CKKSConfig


def _settings(*argv: str, layer0: bool = False):
    args = build_parser(layer0_preset=layer0).parse_args(list(argv))
    config = Llama3CKKSConfig.from_json(args.config)
    return resolve_settings(args, config)


def test_input_level_has_one_source_per_shape() -> None:
    """The four entry shapes are distinct: conflating any two changes the budget."""

    depth = Llama3CKKSConfig.from_json(build_parser().parse_args([]).config)
    depth = int(depth.simulator.maxLevelsRemaining)

    fresh = _settings(layer0=True)
    assert (fresh.input_level, fresh.input_level_source) == (0, "single_layer_fresh")

    steady = _settings("--steady-state-layer", layer0=True)
    assert steady.input_level == depth - 1
    assert steady.input_level_source == "steady_state_boundary"

    full = _settings("--num-layers", "32")
    assert full.input_level == depth + 1 - full.bootstrap_post_levels
    assert full.input_level_source == "bootstrap_output_budget"

    explicit = _settings("--input-level", "19", layer0=True)
    assert (explicit.input_level, explicit.input_level_source) == (19, "explicit")

    # A steady layer enters near the top of the chain, a full-model run enters at
    # the bootstrap's output budget. They are far apart on purpose.
    assert steady.input_level != full.input_level


def test_softmax_position_resolves_auto_from_the_layer_shape() -> None:
    assert _settings("--num-layers", "32").softmax_bootstrap_position == "pre"
    assert (
        _settings("--steady-state-layer", layer0=True)
        .softmax_bootstrap_position
        == "pre"
    )
    # A fresh layer0 has no input refresh to share budget with.
    assert _settings(layer0=True).softmax_bootstrap_position == "off"
    # An explicit request always wins over the inferred one.
    assert (
        _settings("--num-layers", "32", "--softmax-bootstrap-position", "post")
        .softmax_bootstrap_position
        == "post"
    )


def test_pairfm_limbs_distinguishes_absent_from_disabled() -> None:
    assert _settings("--num-layers", "32").pairfm_post_refresh_input_limbs == 6
    assert (
        _settings("--num-layers", "32", "--pairfm-post-refresh-input-limbs", "0")
        .pairfm_post_refresh_input_limbs
        is None
    )
    assert (
        _settings("--num-layers", "32", "--pairfm-post-refresh-input-limbs", "5")
        .pairfm_post_refresh_input_limbs
        == 5
    )


def test_impossible_combinations_fail_before_any_setup() -> None:
    with pytest.raises(ValueError, match="num-layers"):
        _settings("--num-layers", "33")
    with pytest.raises(ValueError, match="bootstrap-post-levels"):
        _settings("--num-layers", "32", "--bootstrap-post-levels", "999")
    with pytest.raises(ValueError, match="input-level"):
        _settings("--input-level", "999", layer0=True)


def test_report_shape_is_shared_by_both_report_paths() -> None:
    report = _settings("--num-layers", "32").as_report()
    assert set(report) == {
        "context_depth",
        "input_level",
        "input_level_source",
        "bootstrap_post_levels",
        "softmax_bootstrap_position",
        "key_mode",
        "auto_load_keys",
    }
    assert report["softmax_bootstrap_position"] == "pre"


def test_key_mode_resolves_to_easyfhe_key_residency() -> None:
    eager = _settings(layer0=True)
    assert eager.key_mode == "eager"
    assert eager.auto_load_keys is True

    host = _settings("--key-mode", "host", layer0=True)
    assert host.key_mode == "host"
    assert host.auto_load_keys is False


def test_warmup_scope_auto_picks_the_cheapest_representative_shape() -> None:
    # A multi-layer run warms one steady layer, not another 32-layer pass.
    assert _settings("--num-layers", "32").warmup_scope == "steady-layer"
    assert _settings(layer0=True).warmup_scope == "model"
    assert (
        _settings("--num-layers", "32", "--warmup-scope", "model").warmup_scope
        == "model"
    )


def test_weights_source_and_scope_must_agree() -> None:
    with pytest.raises(ValueError, match="one-layer system preset"):
        _settings("--num-layers", "32", "--weights-source", "synthetic")


def test_poly_pack_joint_tol_v1_fills_reduced_artifacts() -> None:
    args = resolve_poly_paths(
        build_parser().parse_args(["--poly-pack", "joint_tol_v1"])
    )
    assert args.rms_coeffs == JOINT_TOL_V1_RMS
    assert args.silu_coeffs == JOINT_TOL_V1_SILU
    assert args.softmax_coeffs == JOINT_TOL_V1_SOFTMAX
    production = resolve_poly_paths(build_parser().parse_args([]))
    assert production.rms_coeffs == DEFAULT_RMS
    assert production.silu_coeffs == DEFAULT_SILU
    assert production.softmax_coeffs == DEFAULT_SOFTMAX
    override = resolve_poly_paths(
        build_parser().parse_args(
            ["--poly-pack", "joint_tol_v1", "--silu-coeffs", str(DEFAULT_SILU)]
        )
    )
    assert override.silu_coeffs == DEFAULT_SILU
    assert override.rms_coeffs == JOINT_TOL_V1_RMS
