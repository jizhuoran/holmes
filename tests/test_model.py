from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe import Llama3CKKSConfig
from llama3fhe.application.approx_config import ApproximationConfig
from llama3fhe.application.args import (
    DEFAULT_INPUT_IDS,
    _fixture_expected_token_id,
    build_parser,
)
from llama3fhe.application.runner import _raise_on_failed_expectation, run
from llama3fhe.model import LayerSchedule, Llama3Model


class _Cipher:
    def __init__(self, name: str):
        self.name = name
        self.releases = 0

    def release(self) -> None:
        self.releases += 1


class _Layer:
    def __init__(self, index: int, calls: list[dict[str, object]]):
        self.index = index
        self.calls = calls

    def forward(self, hidden, **kwargs):
        output = tuple(
            _Cipher(f"layer-{self.index}-{cipher_index}")
            for cipher_index in range(16)
        )
        bootstrap = kwargs["bootstrap_operator"]
        bootstrap.calls += 10
        bootstrap.native_calls = int(
            getattr(bootstrap, "native_calls", 0)
        ) + 7
        self.calls.append(
            {
                "index": self.index,
                "refresh_input": kwargs["refresh_input"],
                "hidden_nonzero": bool(np.any(hidden)),
                "qk_output_level": kwargs["schedule"].qk_output_level,
                "preserve_prebootstrap_residual": bool(
                    kwargs["schedule"].preserve_prebootstrap_residual
                ),
            }
        )

        class Result:
            ciphers = output
            wall_seconds = 1.0
            stage_seconds = {"layer": 1.0}
            nested_stage_seconds = {}
            output_levels = (7,)
            softmax_bootstrap_calls = 2
            residual_bootstrap_calls = 8
            softmax_native_bootstrap_calls = 3
            residual_native_bootstrap_calls = 4

            def release(self):
                for cipher in self.ciphers:
                    cipher.release()
                self.ciphers = ()

        return Result()


def test_model_chains_layers_and_refreshes_only_later_inputs() -> None:
    config = Llama3CKKSConfig()
    model = Llama3Model(config=config, num_layers=3)
    calls: list[dict[str, object]] = []
    bootstrap = SimpleNamespace(calls=30, native_calls=12)
    initial = tuple(_Cipher(f"input-{index}") for index in range(16))
    result = model.forward(
        np.ones((128, 4096), dtype=np.float32),
        crypto_context=object(),
        input_ciphers=initial,
        layers=(_Layer(index, calls) for index in range(3)),
        bootstrap_operator=bootstrap,
        schedule=LayerSchedule(preserve_prebootstrap_residual=True),
    )

    assert [row["index"] for row in calls] == [0, 1, 2]
    assert [row["refresh_input"] for row in calls] == [False, True, True]
    assert [row["hidden_nonzero"] for row in calls] == [True, False, False]
    assert all(row["qk_output_level"] is None for row in calls)
    assert all(row["preserve_prebootstrap_residual"] for row in calls)
    assert len(result.per_layer) == 3
    assert result.bootstrap_setup_seconds == 0.0
    assert result.bootstrap_calls == 30
    assert result.native_bootstrap_calls == 21
    assert sum(
        row["softmax_bootstrap_calls"] + row["residual_bootstrap_calls"]
        for row in result.per_layer
    ) == result.bootstrap_calls
    assert sum(
        row["softmax_native_bootstrap_calls"]
        + row["residual_native_bootstrap_calls"]
        for row in result.per_layer
    ) == result.native_bootstrap_calls
    assert all(cipher.releases == 0 for cipher in initial)
    output = result.ciphers[0]
    result.release()
    assert output.releases == 1

    # Reusing the same prepared operator must report only this forward's
    # counter delta, never the preceding run or its warmup history.
    second = model.forward(
        np.zeros((128, 4096), dtype=np.float32),
        crypto_context=object(),
        input_ciphers=initial,
        layers=(_Layer(index, calls) for index in range(3)),
        bootstrap_operator=bootstrap,
    )
    assert second.bootstrap_calls == 30
    assert second.native_bootstrap_calls == 21
    assert bootstrap.calls == 90
    assert bootstrap.native_calls == 54
    second.release()


def test_model_accepts_consecutive_layers_not_starting_at_zero() -> None:
    config = Llama3CKKSConfig()
    model = Llama3Model(config=config, num_layers=2)
    calls: list[dict[str, object]] = []
    bootstrap = SimpleNamespace(calls=0, native_calls=0)
    initial = tuple(_Cipher(f"input-{index}") for index in range(16))
    result = model.forward(
        np.ones((128, 4096), dtype=np.float32),
        crypto_context=object(),
        input_ciphers=initial,
        layers=(_Layer(index, calls) for index in (15, 16)),
        bootstrap_operator=bootstrap,
        schedule=LayerSchedule(preserve_prebootstrap_residual=True),
    )
    assert [row["index"] for row in calls] == [15, 16]
    assert [row["refresh_input"] for row in calls] == [False, True]
    result.release()

    with pytest.raises(ValueError, match="consecutive"):
        model.forward(
            np.ones((128, 4096), dtype=np.float32),
            crypto_context=object(),
            input_ciphers=initial,
            layers=(_Layer(index, calls) for index in (15, 17)),
            bootstrap_operator=bootstrap,
        )


def test_approximation_config_selects_each_layer() -> None:
    root = Path(__file__).resolve().parents[1] / "assets/polynomials"
    approximations = ApproximationConfig(
        rmsnorm_path=root / "rmsnorm_poly_coeffs.json",
        silu_path=root / "silu_poly_coeffs.json",
        softmax_path=root / "softmax_poly_coeffs.json",
    )
    layer0 = approximations.for_layer(0)
    layer31 = approximations.for_layer(31)

    assert len(layer0.input_norm["coefficients"]) - 1 == 16
    assert len(layer0.post_attention_norm["coefficients"]) - 1 == 32
    assert len(layer0.silu["coefficients"]) - 1 == 48
    assert len(layer31.input_norm["coefficients"]) - 1 == 128
    assert len(layer31.post_attention_norm["coefficients"]) - 1 == 128
    assert len(layer31.silu["coefficients"]) - 1 == 384
    assert layer0.softmax is layer31.softmax
    with pytest.raises(KeyError):
        approximations.for_layer(32)


def test_full_model_audit_does_not_require_a_checkpoint() -> None:
    args = build_parser().parse_args(["--audit-only", "--num-layers", "32"])
    assert args.warmup == 1
    assert args.warmup_scope == "auto"
    assert args.input_ids_file is None
    assert args.default_input_ids_file == DEFAULT_INPUT_IDS
    assert _fixture_expected_token_id(DEFAULT_INPUT_IDS) == 12366
    payload = run(args)
    assert payload["model_scope"] == "full_model"
    assert payload["num_layers"] == 32
    assert payload["inter_layer"] == "bootstrap fused into next input RMSNorm"
    assert payload["schedule"]["context_depth"] == 37
    assert payload["schedule"]["input_level"] == 19
    assert (
        payload["schedule"]["input_level_source"]
        == "bootstrap_output_budget"
    )
    assert payload["schedule"]["softmax_bootstrap_position"] == "pre"
    assert payload["schedule"]["adaptive_rmsnorm_slim_refresh"] is True
    assert payload["schedule"]["softmax_lambda_bootstrap_iterations"] == 1
    assert (
        payload["schedule"][
            "softmax_single_pass_match_two_pass_level"
        ]
        is True
    )
    assert payload["audit"]["settings"][
        "softmax_lambda_bootstrap_iterations"
    ] == 1
    assert payload["schedule"]["residual_bootstrap_iterations"] == 1
    assert payload["schedule"]["pairfm_bootstrap_mode"] == "auto"
    assert payload["schedule"]["pairfm_post_refresh_input_limbs"] == 6
    assert payload["schedule"]["warmup"] == 1
    assert payload["schedule"]["warmup_scope"] == "steady-layer"


def test_next_token_expectation_fails_only_after_a_report_exists() -> None:
    _raise_on_failed_expectation(
        {
            "expectation": {
                "expected_token_id": 12366,
                "actual_token_id": 12366,
                "passed": True,
            }
        }
    )
    with pytest.raises(SystemExit, match="expected 12366, got 311"):
        _raise_on_failed_expectation(
            {
                "expectation": {
                    "expected_token_id": 12366,
                    "actual_token_id": 311,
                    "passed": False,
                }
            }
        )

    args = build_parser().parse_args(
        ["--audit-only", "--expect-token-id", "12366"]
    )
    with pytest.raises(ValueError, match="real-weight model execution"):
        run(args)
