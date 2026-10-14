from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import llama3fhe.graph as graph
from llama3fhe import Llama3CKKSConfig
from llama3fhe.application import runner as model_app
from llama3fhe.application.args import build_parser
from llama3fhe.application.runner import run
from llama3fhe.model import (
    LayerApproximations,
    LayerWeights,
    TransformerLayer,
)


def _dummy_weights() -> LayerWeights:
    value = np.zeros((1,), dtype=np.float32)
    return LayerWeights(
        input_layernorm=value,
        q_proj=value,
        k_proj=value,
        v_proj=value,
        o_proj=value,
        post_attention_layernorm=value,
        gate_proj=value,
        up_proj=value,
        down_proj=value,
    )


def test_layer0_application_audit_is_a_model_free_fast_path() -> None:
    args = build_parser(layer0_preset=True).parse_args(["--audit-only"])
    payload = run(args)
    assert payload["backend"] == "feature_major"
    assert payload["mode"] == "audit"
    assert payload["model_scope"] == "layer0"
    assert payload["audit"]["shape"]["canonical_cipher_count"] == 16


def test_constructed_layer0_model_delegates_to_the_graph(monkeypatch) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_layer(hidden_states, **kwargs):
        captured["hidden_shape"] = tuple(hidden_states.shape)
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(graph, "evaluate", fake_layer)
    weights = _dummy_weights()
    approximations = LayerApproximations(
        input_norm={"name": "attn"},
        post_attention_norm={"name": "ffn"},
        silu={"name": "silu"},
        softmax=object(),
    )
    config = Llama3CKKSConfig()
    model = TransformerLayer(0, weights, approximations, config)

    context = object()
    inputs = (object(),)
    result = model.forward(
        np.zeros((128, 4096), dtype=np.float32),
        crypto_context=context,
        input_ciphers=inputs,
    )

    assert result is sentinel
    assert captured["hidden_shape"] == (128, 4096)
    assert captured["layer_idx"] == 0
    assert captured["weights"] is weights
    assert captured["approximations"] is approximations
    assert captured["config"] is config


def test_diagnostic_reference_uses_indexed_alg2_prefill(monkeypatch) -> None:
    hidden = np.zeros((4, 8), dtype=np.float32)
    token_valid = np.ones((4,), dtype=bool)
    weights = object()
    approximations = object()
    attention_layout = object()
    prefix_bundle = object()
    expected = object()
    captured: dict[str, object] = {}

    def fake_prefill(values, **kwargs):
        captured["hidden"] = values
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        model_app, "poly_prefill_layer_numpy", fake_prefill
    )
    layer = SimpleNamespace(
        index=7,
        weights=weights,
        approximations=approximations,
        rope_theta=12345.0,
    )

    result = model_app._poly_layer_diagnostic_reference(
        hidden,
        layer=layer,
        attention_layout=attention_layout,
        prefix_bundle=prefix_bundle,
        token_valid_mask=token_valid,
    )

    assert result is expected
    assert captured["hidden"] is hidden
    assert captured["weights"] is weights
    assert captured["approximations"] is approximations
    assert captured["attention_layout"] is attention_layout
    assert captured["layer_idx"] == 7
    assert captured["rope_theta"] == 12345.0
    assert captured["use_alg2_softmax"] is True
    assert captured["prefix_bundle"] is prefix_bundle
    assert captured["token_valid_mask"] is token_valid


def test_layer0_cli_overrides_are_optional() -> None:
    args = build_parser(layer0_preset=True).parse_args([])
    assert args.num_layers == 1
    assert args.weights_source == "synthetic"
    assert args.default_input_ids_file is None
    assert args.depth is None
    assert args.device is None
    assert args.key_mode == "eager"
    assert args.warmup == 0
    assert args.warmup_scope == "auto"
    assert args.profile_operator_breakdown is False
    assert args.repetitions == 1
    assert args.pairfm_bootstrap_mode == "auto"
    assert args.softmax_complex_pre_exp_bootstrap is True
    assert args.softmax_reuse_round_square is True
    assert args.softmax_fuse_slim_chebyshev_map is True
    assert args.adaptive_rmsnorm_slim_refresh is True
    assert args.batched_sparse_attention_input is True
    assert args.fuse_rmsnorm_output_scale is True
    assert args.fuse_silu_domain_map is True
    assert args.residual_bootstrap_level_drop == 0
    assert args.preserve_prebootstrap_residual is True
    assert args.profile_device_memory is False
    assert args.plaintext_cache_gb == 0.0
    assert args.plaintext_cache_max_source_mb == 1.0
    assert args.linear_hoist_strategy == "normal"
    assert args.softmax_lambda_bootstrap_iterations == 1
    assert args.softmax_single_pass_match_two_pass_level is True
    assert args.residual_bootstrap_iterations == 1

    disabled = build_parser(layer0_preset=True).parse_args(
        [
            "--no-preserve-prebootstrap-residual",
            "--no-adaptive-rmsnorm-slim-refresh",
            "--no-batched-sparse-attention-input",
            "--no-fuse-rmsnorm-output-scale",
            "--no-fuse-silu-domain-map",
            "--profile-device-memory",
            "--softmax-lambda-bootstrap-iterations",
            "2",
            "--no-softmax-single-pass-match-two-pass-level",
            "--residual-bootstrap-iterations",
            "2",
        ]
    )
    assert disabled.preserve_prebootstrap_residual is False
    assert disabled.adaptive_rmsnorm_slim_refresh is False
    assert disabled.batched_sparse_attention_input is False
    assert disabled.fuse_rmsnorm_output_scale is False
    assert disabled.fuse_silu_domain_map is False
    assert disabled.profile_device_memory is True
    assert disabled.softmax_lambda_bootstrap_iterations == 2
    assert disabled.softmax_single_pass_match_two_pass_level is False
    assert disabled.residual_bootstrap_iterations == 2


def test_prebootstrap_residual_selection_borrows_the_original_stream() -> None:
    original = (object(), object())
    refreshed = (object(), object())

    selected, source = graph._select_residual_identity(
        original,
        refreshed,
        preserve_prebootstrap=True,
        refresh_was_run=True,
    )
    assert selected is original
    assert source == "prebootstrap"

    selected, source = graph._select_residual_identity(
        original,
        refreshed,
        preserve_prebootstrap=False,
        refresh_was_run=True,
    )
    assert selected is refreshed
    assert source == "bootstrap_refreshed"


def test_device_memory_snapshot_is_inert_on_cpu() -> None:
    from llama3fhe.backend import cuda_memory_snapshot

    assert cuda_memory_snapshot("cpu") == {"available": False}


def test_steady_state_default_uses_the_two_limb_boundary() -> None:
    args = build_parser(layer0_preset=True).parse_args(
        ["--audit-only", "--steady-state-layer"]
    )
    payload = run(args)
    assert payload["schedule"]["context_depth"] == 37
    assert payload["schedule"]["input_level"] == 36
    assert payload["schedule"]["input_level_source"] == "steady_state_boundary"
    assert payload["schedule"]["bootstrap_post_levels"] == 19
    assert payload["schedule"]["softmax_bootstrap_position"] == "pre"


def test_explicit_layer0_input_level_is_not_reinterpreted() -> None:
    args = build_parser(layer0_preset=True).parse_args(
        [
            "--audit-only",
            "--steady-state-layer",
            "--input-level",
            "0",
        ]
    )
    payload = run(args)
    assert payload["schedule"]["input_level"] == 0
    assert payload["schedule"]["input_level_source"] == "explicit"


def test_audit_records_prebootstrap_residual_policy() -> None:
    args = build_parser(layer0_preset=True).parse_args(
        ["--audit-only", "--preserve-prebootstrap-residual"]
    )
    payload = run(args)
    assert payload["schedule"]["preserve_prebootstrap_residual"] is True


def test_audit_records_softmax_lambda_bootstrap_policy() -> None:
    args = build_parser(layer0_preset=True).parse_args(
        [
            "--audit-only",
            "--softmax-lambda-bootstrap-iterations",
            "2",
            "--no-softmax-single-pass-match-two-pass-level",
        ]
    )
    payload = run(args)
    assert payload["schedule"]["softmax_lambda_bootstrap_iterations"] == 2
    assert (
        payload["schedule"][
            "softmax_single_pass_match_two_pass_level"
        ]
        is False
    )
    assert payload["audit"]["settings"][
        "softmax_lambda_bootstrap_iterations"
    ] == 2
    assert (
        payload["audit"]["settings"][
            "match_softmax_single_pass_bootstrap_level"
        ]
        is False
    )


def test_layer0_application_warmup_and_measured_samples(monkeypatch) -> None:
    calls = {"forward": 0, "release": 0, "bootstrap_release": 0}
    context_auto_load_keys: list[bool] = []
    preserve_options: list[bool] = []
    batched_sparse_options: list[bool] = []
    lambda_iterations: list[int] = []
    match_lambda_levels: list[bool] = []
    bootstrap_operators: list[object] = []

    class FakeContext:
        device = "cpu"
        keygen_wall_time_s = 0.5

        @staticmethod
        def configure_plaintext_cache(**_kwargs):
            pass

        @staticmethod
        def clear_plaintext_cache():
            pass

    class FakeBootstrapOperator:
        setup_seconds = 0.25

        @staticmethod
        def release():
            calls["bootstrap_release"] += 1

    class FakeResult:
        ciphers = tuple(np.zeros((1,)) for _ in range(16))
        wall_seconds = 10.0
        bootstrap_calls = 0
        native_bootstrap_calls = 0
        per_layer = [
            {
                "stage_seconds": {"qkv": 4.0},
                "nested_stage_seconds": {"qkv_pack_encode": 2.0},
                "output_levels": (7,),
            }
        ]

        @staticmethod
        def release():
            calls["release"] += 1

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def required_rotations(schedule=None, **_kwargs):
            return ()

        @staticmethod
        def forward(_hidden, **kwargs):
            calls["forward"] += 1
            schedule = kwargs["schedule"]
            preserve_options.append(
                bool(schedule.preserve_prebootstrap_residual)
            )
            batched_sparse_options.append(
                bool(schedule.batched_sparse_attention_input)
            )
            lambda_iterations.append(
                int(schedule.softmax_lambda_bootstrap_iterations)
            )
            match_lambda_levels.append(
                bool(schedule.match_softmax_single_pass_bootstrap_level)
            )
            bootstrap_operators.append(kwargs["bootstrap_operator"])
            return FakeResult()

    monkeypatch.setattr(
        model_app,
        "_synthetic_inputs",
        lambda *_args, **_kwargs: (
            np.zeros((128, 4096), dtype=np.float32),
            _dummy_weights(),
        ),
    )

    class FakeApproximationConfig:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def for_layer(_layer_idx):
            return LayerApproximations(
                input_norm={"coefficients": [0.0, 1.0]},
                post_attention_norm={"coefficients": [0.0, 1.0]},
                silu={"coefficients": [0.0, 1.0]},
                softmax=object(),
            )

        @staticmethod
        def validate_layers(_count):
            return None

        @staticmethod
        def degree_summary(_layer_idx):
            return {"input_rmsnorm": 1, "post_attention_rmsnorm": 1, "silu": 1}

        paths = {"rmsnorm": "rms", "silu": "silu", "softmax": "softmax"}

    monkeypatch.setattr(
        model_app, "ApproximationConfig", FakeApproximationConfig
    )
    monkeypatch.setattr(model_app, "Llama3Model", FakeModel)
    def fake_create_context(*_args, **kwargs):
        context_auto_load_keys.append(bool(kwargs["auto_load_keys"]))
        return FakeContext()

    monkeypatch.setattr(model_app, "_create_context", fake_create_context)
    monkeypatch.setattr(
        model_app,
        "_encrypt_hidden",
        lambda *_args, **_kwargs: tuple(np.zeros((1,)) for _ in range(16)),
    )
    monkeypatch.setattr(
        model_app,
        "prepare_bootstrap_operator",
        lambda *_args, **_kwargs: FakeBootstrapOperator(),
    )
    monkeypatch.setattr(
        model_app,
        "decrypt_hidden",
        lambda *_args, **_kwargs: np.zeros((128, 4096), dtype=np.float32),
    )
    args = build_parser(layer0_preset=True).parse_args(
        [
            "--warmup",
            "1",
            "--repetitions",
            "2",
            "--profile-operator-breakdown",
            "--key-mode",
            "host",
            "--preserve-prebootstrap-residual",
            "--batched-sparse-attention-input",
        ]
    )

    payload = run(args)

    assert calls == {
        "forward": 3,
        "release": 3,
        "bootstrap_release": 1,
    }
    assert context_auto_load_keys == [False]
    assert preserve_options == [True, True, True]
    assert batched_sparse_options == [True, True, True]
    assert lambda_iterations == [1, 1, 1]
    assert match_lambda_levels == [True, True, True]
    assert len({id(value) for value in bootstrap_operators}) == 1
    assert payload["performance"]["warmup"] == 1
    assert payload["performance"]["repetitions"] == 2
    assert payload["performance"]["median_wall_seconds"] == 10.0
    assert payload["performance"]["warmup_scope"] == "model"
    assert payload["performance"]["warmup_layers_per_repetition"] == 1
    assert payload["performance"]["median_wall_seconds"] == 10.0
    assert (
        payload["performance"][
            "median_compute_excluding_prepare_seconds"
        ]
        == 8.0
    )
    assert payload["performance"]["application_bootstrap_prepare_seconds"] >= 0
    assert len(payload["performance"]["samples"]) == 2
    assert payload["performance"]["samples"][0]["per_layer"] == FakeResult.per_layer


def test_slim_extra_softmax_checkpoints_default_and_env(monkeypatch) -> None:
    from types import SimpleNamespace

    from llama3fhe.operators.attention.prefill.softmax import (
        SLIM_MID_ALG2_LAYERS,
        SLIM_POST_ALG2_LAYERS,
        SLIM_POST_ALG2_MIN_ROUNDS,
        SLIM_POST_EXP_LAYERS,
        extra_softmax_checkpoint_enabled,
    )

    slim = SimpleNamespace(config=SimpleNamespace(output_levels=19))
    legacy = SimpleNamespace(config=SimpleNamespace(output_levels=23))
    for name in (
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP",
        "LLAMA_FHE_SOFTMAX_POST_EXP_BOOTSTRAP",
        "LLAMA_FHE_SOFTMAX_MID_ALG2_BOOTSTRAP",
    ):
        monkeypatch.delenv(name, raising=False)

    assert extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 0, SLIM_POST_ALG2_LAYERS, slim
    )
    assert extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 14, SLIM_POST_ALG2_LAYERS, slim
    )
    assert not extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 15, SLIM_POST_ALG2_LAYERS, slim
    )
    # k-aware default: a reduced-k table (L2 k 3->2) skips the post-Alg2 BS,
    # the production k keeps it.
    assert extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 2, SLIM_POST_ALG2_LAYERS, slim,
        rounds=3, min_rounds=SLIM_POST_ALG2_MIN_ROUNDS,
    )
    assert not extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 2, SLIM_POST_ALG2_LAYERS, slim,
        rounds=2, min_rounds=SLIM_POST_ALG2_MIN_ROUNDS,
    )
    assert extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_EXP_BOOTSTRAP", 31, SLIM_POST_EXP_LAYERS, slim
    )
    assert extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_MID_ALG2_BOOTSTRAP", 31, SLIM_MID_ALG2_LAYERS, slim
    )
    assert not extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 0, SLIM_POST_ALG2_LAYERS, legacy
    )

    monkeypatch.setenv("LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", "off")
    assert not extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 0, SLIM_POST_ALG2_LAYERS, slim
    )
    monkeypatch.setenv("LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", "15")
    assert extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 15, SLIM_POST_ALG2_LAYERS, slim
    )
    monkeypatch.setenv("LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", "1")
    assert extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 1, SLIM_POST_ALG2_LAYERS, slim
    )
    assert not extra_softmax_checkpoint_enabled(
        "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", 0, SLIM_POST_ALG2_LAYERS, slim
    )
