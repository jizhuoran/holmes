#!/usr/bin/env python3
"""Run and compare the matched one-pass Softmax-lambda schedule.

The script is deliberately dry-run by default.  Pass ``--execute`` only on a
machine reserved for the two sequential 32-layer FHE evaluations per case.
The reference explicitly restores the former two-pass Lambda policy; the
candidate exercises the application defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "weights" / "Meta-Llama-3-8B-QuaRot"
DEFAULT_CONFIG = PROJECT_ROOT / "assets" / "feature_major.json"
DEFAULT_FIXTURE = PROJECT_ROOT / "assets" / "prefill128_input_ids.json"

MULTI_PROMPT_SUITE = (
    ("arithmetic", "The result of 2 + 2 is"),
    ("sky", "The color of a clear daytime sky is"),
    ("austen", "The author of Pride and Prejudice is"),
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or "prompt"


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_outputs(
    reference_hidden: Path,
    candidate_hidden: Path,
    *,
    reference_report: Path | None = None,
    candidate_report: Path | None = None,
) -> dict[str, object]:
    """Return stable numerical and prediction deltas for one A/B case."""

    reference = np.asarray(np.load(reference_hidden), dtype=np.float64)
    candidate = np.asarray(np.load(candidate_hidden), dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"output shape mismatch: {reference.shape} != {candidate.shape}"
        )
    delta = candidate - reference
    absolute = np.abs(delta)
    last = absolute[-1]
    result: dict[str, object] = {
        "shape": list(reference.shape),
        "all": {
            "max_abs": float(np.max(absolute)),
            "mean_abs": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        },
        "last_token": {
            "max_abs": float(np.max(last)),
            "mean_abs": float(np.mean(last)),
            "rmse": float(np.sqrt(np.mean(np.square(delta[-1])))),
        },
    }
    if reference_report is not None and candidate_report is not None:
        ref = _load_json(reference_report)
        cand = _load_json(candidate_report)
        ref_top = [int(row["token_id"]) for row in ref.get("top_k", ())]
        cand_top = [int(row["token_id"]) for row in cand.get("top_k", ())]
        result["prediction"] = {
            "reference_next_token": ref.get("next_token"),
            "candidate_next_token": cand.get("next_token"),
            "reference_top_k_ids": ref_top,
            "candidate_top_k_ids": cand_top,
            "top_k_identical": ref_top == cand_top,
        }
        result["performance"] = {
            "reference_online_seconds": ref.get("performance", {}).get(
                "median_online_seconds"
            ),
            "candidate_online_seconds": cand.get("performance", {}).get(
                "median_online_seconds"
            ),
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results" / "lambda_ab",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--include-multi-prompt-suite",
        action="store_true",
        help="Also run three deterministic non-fixture prompts.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="LABEL=TEXT",
        help="Add a prompt case; may be repeated.",
    )
    return parser


def _cases(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    cases: list[tuple[str, str | None]] = [("paris", None)]
    if bool(args.include_multi_prompt_suite):
        cases.extend(MULTI_PROMPT_SUITE)
    for item in args.prompt:
        label, separator, prompt = str(item).partition("=")
        if not separator or not label.strip() or not prompt.strip():
            raise ValueError("--prompt must have the form LABEL=TEXT")
        cases.append((_slug(label), prompt))
    labels = [label for label, _ in cases]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate case labels: {labels}")
    return cases


def _command(
    args: argparse.Namespace,
    *,
    label: str,
    prompt: str | None,
    policy: str,
) -> tuple[list[str], Path, Path]:
    stem = args.output_dir / f"{label}_{policy}"
    report = stem.with_suffix(".json")
    hidden = stem.with_suffix(".npy")
    command = [
        str(args.python),
        str(PROJECT_ROOT / "run_model.py"),
        "--num-layers",
        "32",
        "--weights-source",
        "real",
        "--model-dir",
        str(args.model_dir),
        "--config",
        str(args.config),
        "--input-level",
        "19",
        "--softmax-bootstrap-position",
        "pre",
        "--warmup",
        "1",
        "--warmup-scope",
        "steady-layer",
        "--repetitions",
        "1",
        "--output-json",
        str(report),
        "--output-hidden-npy",
        str(hidden),
    ]
    if prompt is None:
        command.extend(("--input-ids-file", str(DEFAULT_FIXTURE)))
    else:
        command.extend(("--prompt", prompt))
    if policy == "two_pass_reference":
        command.extend(
            (
                "--softmax-lambda-bootstrap-iterations",
                "2",
                "--no-softmax-single-pass-match-two-pass-level",
            )
        )
    elif policy != "one_pass_default":
        raise ValueError(f"unknown policy: {policy}")
    return command, report, hidden


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = _cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    summary: dict[str, object] = {
        "candidate": "matched one-pass Lambda bootstrap application defaults",
        "reference": "explicit two-pass Lambda bootstrap",
        "cases": {},
    }
    for label, prompt in cases:
        artifacts: dict[str, tuple[Path, Path]] = {}
        for policy in ("two_pass_reference", "one_pass_default"):
            command, report, hidden = _command(
                args, label=label, prompt=prompt, policy=policy
            )
            print(shlex.join(command))
            artifacts[policy] = (report, hidden)
            if bool(args.execute):
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=True,
                )
        if bool(args.execute):
            ref_report, ref_hidden = artifacts["two_pass_reference"]
            cand_report, cand_hidden = artifacts["one_pass_default"]
            summary["cases"][label] = compare_outputs(
                ref_hidden,
                cand_hidden,
                reference_report=ref_report,
                candidate_report=cand_report,
            )
    if bool(args.execute):
        summary_path = args.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"wrote {summary_path}")
    else:
        print("dry run only; pass --execute to launch the sequential GPU runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
