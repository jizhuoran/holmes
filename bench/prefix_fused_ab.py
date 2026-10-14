#!/usr/bin/env python3
"""One-rank H100 A/B for no-Prefix, compact, pages, and fused Softmax.

Example:
    env_fhe/bin/python bench/prefix_fused_ab.py --layer 10 --limbs 24 --repeat 3

The bounded score fixture exercises the selected layer's fitted polynomial
domain.  Its public score scale differs from QK's real 1/sqrt(128) factor, so
this is an operator comparison, not an end-to-end model accuracy result. The
no-Prefix arm defaults to the production no-Prefix polynomial artifact; pass
the Prefix-k2 artifact through --no-prefix-poly to isolate path cost with one
shared coefficient set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama3fhe.approx.prefix import dual_softmax_alg2_numpy, dual_softmax_numpy
from llama3fhe.approx.softmax import SoftmaxPolyConfig, load_softmax_poly_coeffs
from llama3fhe.backend import (
    create_sim_context,
    release_if_supported,
    synchronize_device,
)
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.operators.attention.prefill.softmax import softmax_alg2_fhe_delta
from llama3fhe.operators.attention.prefix.prefix_softmax import (
    prefix_softmax_rotations,
    softmax_alg2_prefix_partitions_fhe,
)
from llama3fhe.operators.nonlinear.bootstrap import BootstrapConfig, BootstrapOperator

DEFAULT_POLY = ROOT / "assets/polynomials/softmax_poly_coeffs_prefix_k2.json"


def _fixture(config, *, layer: int, layout: AttentionPairLayout):
    scale = max(10.0, config.U_hat_for(layer) / 3.5)
    high = (config.U_hat_for(layer) - 1.0) / scale
    next_high = (config.U_hat_for(layer) - 3.0) / scale
    low = (config.U_hat_for(layer) - 25.0) / scale
    prefix = np.empty((int(layout.query_heads), 128, 2), dtype=np.float32)
    prefix[:, :, 0] = high
    prefix[:, :, 1] = next_high
    content = np.full(
        (int(layout.query_heads), 128, 128), low, dtype=np.float32
    )
    for query in range(128):
        content[:, query, query] = high
    factor = scale * math.sqrt(128.0)
    expected_p, expected_c, fit_report = dual_softmax_alg2_numpy(
        prefix * factor,
        content * factor,
        layout=layout,
        config=config,
        layer_idx=layer,
        prefix_k=2,
    )
    exact_p, exact_c = dual_softmax_numpy(
        prefix * scale, content * scale, prefix_k=2
    )
    reference = {
        "exp_oob_frac": fit_report["exp_oob_frac"],
        "invsqrt_oob_fracs": [
            row["oob_frac"] for row in fit_report["invsqrt_rounds"]
        ],
        "max_abs_vs_exact": max(
            float(np.max(np.abs(expected_p - exact_p))),
            float(np.max(np.abs(expected_c - exact_c))),
        ),
    }
    return prefix, content, scale, expected_p, expected_c, reference


def _worst_error(actual: np.ndarray, expected: np.ndarray) -> dict:
    errors = np.abs(actual - expected)
    index = tuple(int(i) for i in np.unravel_index(np.argmax(errors), errors.shape))
    return {
        "index": list(index),
        "actual": float(actual[index]),
        "expected": float(expected[index]),
        "abs_error": float(errors[index]),
    }


def _run_once(
    pipeline: str,
    prefix_ciphers,
    content_ciphers,
    *,
    layout,
    context,
    config,
    layer,
    scale,
    bootstrap,
    operator_config,
    refresh_before_exp,
    expected_p,
    expected_c,
    expected_no_prefix_p,
    expected_no_prefix_c,
    no_prefix_config,
):
    if pipeline != "no_prefix":
        resolved_pipeline = {
            "compact": "dual",
            "partitioned": "dual",
            "pages": "pages",
            "fused": "fused",
        }[pipeline]
        os.environ["LLAMA_FHE_PREFIX_PIPELINE"] = resolved_pipeline
        os.environ["LLAMA_FHE_PREFIX_PACKING"] = (
            "partitioned" if pipeline == "partitioned" else "compact"
        )
    stages: dict[str, float] = {}
    synchronize_device(context.device)
    started = time.perf_counter()
    calls_before = int(bootstrap.calls)
    native_before = int(bootstrap.native_calls)
    out_p = out_c = ()
    try:
        if pipeline == "no_prefix":
            result = softmax_alg2_fhe_delta(
                content_ciphers,
                layout=layout,
                layer_idx=layer,
                config=no_prefix_config,
                crypto_context=context,
                operator_config=operator_config,
                score_scale=scale,
                bootstrap_operator=bootstrap,
                refresh_before_exp=refresh_before_exp,
            )
            out_c = result.probability_ciphers
            result.probability_ciphers = ()
            stages.update(result.stage_seconds)
            expected_p = expected_no_prefix_p
            expected_c = expected_no_prefix_c
        else:
            out_p, out_c = softmax_alg2_prefix_partitions_fhe(
                prefix_ciphers,
                content_ciphers,
                layout=layout,
                crypto_context=context,
                config=config,
                prefix_k=2,
                layer_idx=layer,
                content_form="delta",
                score_scale=scale,
                bootstrap_operator=bootstrap,
                refresh_before_exp=refresh_before_exp,
                operator_config=operator_config,
                stage_seconds=stages,
            )
        synchronize_device(context.device)
        wall = time.perf_counter() - started
        packed_p = (
            np.stack([np.asarray(context.decrypt(ct)).real for ct in out_p])
            if out_p
            else None
        )
        packed_c = np.stack([np.asarray(context.decrypt(ct)).real for ct in out_c])
        actual_p = (
            layout.unpack_prefix_scores_prefill(packed_p, prefix_k=2)
            if packed_p is not None
            else expected_p
        )
        actual_c = layout.unpack_score_delta(packed_c)
        actual_row_sums = actual_p.sum(axis=-1) + actual_c.sum(axis=-1)
        expected_row_sums = expected_p.sum(axis=-1) + expected_c.sum(axis=-1)
        return {
            "pipeline": pipeline,
            "wall_s": wall,
            "bootstrap_calls": int(bootstrap.calls) - calls_before,
            "bootstrap_native_passes": int(bootstrap.native_calls) - native_before,
            "prefix_max_abs_error": (
                float(np.max(np.abs(actual_p - expected_p))) if out_p else None
            ),
            "content_max_abs_error": float(np.max(np.abs(actual_c - expected_c))),
            "tail_prefix_max_abs_error": float(
                np.max(np.abs(actual_p[:, -2:] - expected_p[:, -2:]))
            ) if out_p else None,
            "worst_prefix": _worst_error(actual_p, expected_p) if out_p else None,
            "worst_content": _worst_error(actual_c, expected_c),
            "row_sum_max_abs_error": float(
                np.max(np.abs(actual_row_sums - expected_row_sums))
            ),
            "prefix_min_limbs": (
                min(int(ct.state.cur_limbs) for ct in out_p) if out_p else None
            ),
            "content_min_limbs": min(int(ct.state.cur_limbs) for ct in out_c),
            "stage_seconds": stages,
        }
    finally:
        for cipher in out_p + out_c:
            release_if_supported(cipher)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--limbs", type=int, default=24)
    parser.add_argument("--prefix-limbs", type=int)
    parser.add_argument("--content-limbs", type=int)
    parser.add_argument(
        "--encrypt-limbs", type=int,
        help="Encrypt at this limb count, then modulus-drop to input limbs.",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--include-partitioned", action="store_true")
    parser.add_argument("--include-pages", action="store_true")
    parser.add_argument("--refresh-before-exp", action="store_true")
    parser.add_argument("--main-iterations", type=int, default=1, choices=(1, 2))
    parser.add_argument("--lambda-iterations", type=int, default=1, choices=(1, 2))
    parser.add_argument("--poly", type=Path, default=DEFAULT_POLY)
    parser.add_argument(
        "--no-prefix-poly",
        type=Path,
        default=ROOT / "assets/polynomials/softmax_poly_coeffs.json",
        help="No-Prefix coefficient artifact; use --poly to match coefficients for path-only A/B.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    prefix_limbs = args.limbs if args.prefix_limbs is None else args.prefix_limbs
    content_limbs = args.limbs if args.content_limbs is None else args.content_limbs
    if min(prefix_limbs, content_limbs) < 2 or max(prefix_limbs, content_limbs) > 38:
        parser.error("input limbs must lie in [2, 38]")
    if args.encrypt_limbs is not None and not (
        max(prefix_limbs, content_limbs) <= args.encrypt_limbs <= 38
    ):
        parser.error("--encrypt-limbs must be >= both input limbs and <= 38")

    layout = AttentionPairLayout(
        seq_len=128, head_dim=128, query_heads=4, key_value_heads=2, slots=32768
    )
    config = SoftmaxPolyConfig.from_artifact(load_softmax_poly_coeffs(args.poly))
    no_prefix_config = SoftmaxPolyConfig.from_artifact(
        load_softmax_poly_coeffs(args.no_prefix_poly)
    )
    if (
        config.k_for(args.layer) != no_prefix_config.k_for(args.layer)
        or config.U_hat_for(args.layer) != no_prefix_config.U_hat_for(args.layer)
        or config.stats_min_for(args.layer) != no_prefix_config.stats_min_for(args.layer)
        or config.layer_exp[args.layer].get("degree")
        != no_prefix_config.layer_exp[args.layer].get("degree")
    ):
        parser.error(
            "No-Prefix coefficients must use the same layer k, U_hat, stats_min, "
            "and exp degree as the Prefix coefficients."
        )
    prefix, content, scale, expected_p, expected_c, reference = _fixture(
        config, layer=args.layer, layout=layout
    )
    expected_no_prefix_p, expected_no_prefix_c, no_prefix_fit_report = (
        dual_softmax_alg2_numpy(
            np.empty(prefix.shape[:2] + (0,), dtype=np.float32),
            content * (scale * math.sqrt(128.0)),
            layout=layout,
            config=no_prefix_config,
            layer_idx=args.layer,
            prefix_k=0,
        )
    )
    operator_config = AttentionOperatorConfig(
        refresh_softmax_exp_inputs=args.refresh_before_exp,
        pair_softmax_pre_exp_bootstrap=args.refresh_before_exp,
        reuse_softmax_round_square=True,
        fuse_softmax_slim_chebyshev_map=True,
        softmax_main_bootstrap_iterations=args.main_iterations,
        softmax_lambda_bootstrap_iterations=args.lambda_iterations,
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.maxLevelsRemaining = 37
    rotations = set(prefix_softmax_rotations(layout, prefix_k=2, fused=True))
    rotations.update(BootstrapOperator.required_rotations(log_n=16))
    context = create_sim_context(
        cfg.simulator,
        input_dim=128,
        block_out=128,
        rotations=tuple(sorted(rotations)),
        include_default_rotations=True,
    )
    bootstrap = BootstrapOperator.prepare(
        context,
        BootstrapConfig(
            log_slots=15,
            level_budget=(4, 4),
            output_levels=19,
            iterations=1,
            precision_bits=14,
        ),
    )
    def encrypt_input(rows, input_limbs):
        initial_limbs = args.encrypt_limbs or input_limbs
        outputs = []
        try:
            for row in rows:
                cipher = context.encrypt(
                    row, level=context.context.L - initial_limbs, slots=32768
                )
                if initial_limbs != input_limbs:
                    state = cipher.state.replace(
                        cur_limbs=input_limbs,
                        scale_degree=1,
                        scaling_factor=None,
                    )
                    try:
                        dropped = context.fhe.align_to(
                            cipher, state, context.context
                        )
                    finally:
                        release_if_supported(cipher)
                    cipher = dropped
                outputs.append(cipher)
            return tuple(outputs)
        except Exception:
            for cipher in outputs:
                release_if_supported(cipher)
            raise

    prefix_ciphers = encrypt_input(
        layout.pack_prefix_scores_prefill(prefix), prefix_limbs
    )
    content_ciphers = encrypt_input(
        layout.pack_score_delta(content), content_limbs
    )
    try:
        pipelines = tuple(
            pipeline
            for pipeline, enabled in (
                ("no_prefix", True),
                ("partitioned", args.include_partitioned),
                ("compact", True),
                ("pages", args.include_pages),
                ("fused", True),
            )
            if enabled
        )
        for pipeline in pipelines:
            _run_once(
                pipeline, prefix_ciphers, content_ciphers,
                layout=layout, context=context, config=config,
                layer=args.layer, scale=scale, bootstrap=bootstrap,
                operator_config=operator_config,
                refresh_before_exp=args.refresh_before_exp,
                expected_p=expected_p, expected_c=expected_c,
                expected_no_prefix_p=expected_no_prefix_p,
                expected_no_prefix_c=expected_no_prefix_c,
                no_prefix_config=no_prefix_config,
            )
        runs = []
        for index in range(args.repeat):
            order = pipelines if index % 2 == 0 else tuple(reversed(pipelines))
            for pipeline in order:
                runs.append(
                    _run_once(
                        pipeline, prefix_ciphers, content_ciphers,
                        layout=layout, context=context, config=config,
                        layer=args.layer, scale=scale, bootstrap=bootstrap,
                        operator_config=operator_config,
                        refresh_before_exp=args.refresh_before_exp,
                        expected_p=expected_p, expected_c=expected_c,
                        expected_no_prefix_p=expected_no_prefix_p,
                        expected_no_prefix_c=expected_no_prefix_c,
                        no_prefix_config=no_prefix_config,
                    )
                )
        medians = {
            pipeline: statistics.median(
                row["wall_s"] for row in runs if row["pipeline"] == pipeline
            )
            for pipeline in pipelines
        }
        report = {
            "layer": args.layer,
            "k": config.k_for(args.layer),
            "prefix_limbs": prefix_limbs,
            "content_limbs": content_limbs,
            "prefix_poly": str(args.poly),
            "no_prefix_poly": str(args.no_prefix_poly),
            "encrypt_limbs": args.encrypt_limbs,
            "refresh_before_exp": args.refresh_before_exp,
            "main_iterations": args.main_iterations,
            "lambda_iterations": args.lambda_iterations,
            "numpy_reference": reference,
            "no_prefix_numpy_reference": {
                "exp_oob_frac": no_prefix_fit_report["exp_oob_frac"],
                "invsqrt_oob_fracs": [
                    row["oob_frac"]
                    for row in no_prefix_fit_report["invsqrt_rounds"]
                ],
            },
            "median_wall_s": medians,
            "compact_wall_divided_by_no_prefix_wall": (
                medians["compact"] / medians["no_prefix"]
            ),
            "fused_wall_divided_by_no_prefix_wall": (
                medians["fused"] / medians["no_prefix"]
            ),
            "pages_wall_divided_by_no_prefix_wall": (
                medians["pages"] / medians["no_prefix"]
                if "pages" in medians
                else None
            ),
            "compact_wall_divided_by_pages_wall": (
                medians["compact"] / medians["pages"]
                if "pages" in medians
                else None
            ),
            "compact_wall_divided_by_fused_wall": (
                medians["compact"] / medians["fused"]
            ),
            "runs": runs,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        for cipher in prefix_ciphers + content_ciphers:
            release_if_supported(cipher)
        bootstrap.release()


if __name__ == "__main__":
    main()
