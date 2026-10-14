#!/usr/bin/env python3
"""Fit a two-page Alg2 Softmax table from 129- and 256-token Poly traces.

The encrypted two-page path multiplies valid second-block exp outputs by
1/sqrt(2) in its public post-exp mask. The plaintext Poly path applies the
same map; this tool samples its inverse-square-root inputs without changing
the 128-token table.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from llama3fhe.application.approx_config import ApproximationConfig
from llama3fhe.application.args import DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
from llama3fhe.application.checkpoint import (
    CheckpointMetadata,
    WeightLoader,
    load_layer_weights,
)
from llama3fhe.approx import layer as layer_mod
from llama3fhe.approx.chebyshev import (
    chebyshev_ps_mul_depth,
    compute_degrees_ps,
    eval_chebyshev_coefficients,
    eval_chebyshev_series_direct,
)
from llama3fhe.approx.softmax import SoftmaxPolyConfig
from llama3fhe.layouts import AttentionPairLayout

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "weights/Meta-Llama-3-8b-quarot-safetensors-v2"
DEFAULT_OUT = ROOT / "assets/polynomials/softmax_poly_coeffs_prefill256.json"
DEFAULT_128_IDS = ROOT / "assets/prefill128_input_ids.json"
DEFAULT_129_IDS = ROOT / "assets/prefill129_input_ids.json"
PAGE_SIZE = 128
TOL = 2.0**-12


def _input_ids(path: Path, *, expected: int) -> np.ndarray:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    ids = np.asarray(fixture["input_ids"], dtype=np.int64)
    if ids.shape != (expected,):
        raise ValueError(f"{path} has {ids.shape} IDs; expected ({expected},).")
    return ids


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _grid(lo: float, hi: float) -> np.ndarray:
    linear = np.linspace(lo, hi, 3000, dtype=np.float64)
    if lo > 0:
        return np.concatenate((linear, np.geomspace(lo, hi, 3000)))
    return linear


def _fit(func, lo: float, hi: float, *, initial_degree: int, tol: float):
    xs = _grid(lo, hi)
    expected = np.asarray(func(xs), dtype=np.float64)

    def evaluate(degree: int):
        coeffs = eval_chebyshev_coefficients(
            lambda x: float(func(x)), lo, hi, degree
        )
        actual = eval_chebyshev_series_direct(xs, coeffs, lo, hi)
        error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
        return coeffs, error

    high = max(1, int(initial_degree))
    low = high - 1
    coeffs, error = evaluate(high)
    while error > tol:
        low = high
        high *= 2
        if high > 4096:
            raise RuntimeError(
                f"degree >4096 required on [{lo}, {hi}] for tolerance {tol}"
            )
        coeffs, error = evaluate(high)
    while high - low > 8:
        middle = 2 * ((low + high + 2) // 4)
        candidate, candidate_error = evaluate(middle)
        if candidate_error <= tol:
            high, coeffs, error = middle, candidate, candidate_error
        else:
            low = middle
    return high, coeffs, error


def _bounds(entry: dict) -> tuple[float, float]:
    return tuple(float(v) for v in (entry.get("chebyshev_fhe") or entry)["fit_interval"])


def _required_interval(
    old: tuple[float, float], observed: tuple[float, float], *, margin: float
) -> tuple[float, float] | None:
    old_lo, old_hi = old
    seen_lo, seen_hi = observed
    if seen_lo >= old_lo and seen_hi <= old_hi:
        return None
    if seen_lo >= old_lo:
        lo = old_lo
    elif old_lo > 0:
        lo = min(old_lo, seen_lo * (1.0 - margin))
    else:
        lo = min(old_lo, seen_lo - margin * (old_hi - old_lo))
    hi = old_hi if seen_hi <= old_hi else seen_hi * (1.0 + margin)
    return float(lo), float(hi)


def _fit_stage(table: dict, layer: int, stage: str, observed, *, margin: float):
    key = str(layer)
    old = table["invsqrt_alg2"][stage]["by_layer"][key]
    interval = _required_interval(_bounds(old), observed, margin=margin)
    if interval is None:
        return None
    lo, hi = interval
    degree, coefficients, error = _fit(
        lambda x: np.power(x, -0.5),
        lo,
        hi,
        initial_degree=int(old["degree"]),
        tol=TOL,
    )
    entry = copy.deepcopy(old)
    common = {
        "degree": degree,
        "fit_interval": [lo, hi],
        "coefficients": coefficients.tolist(),
        "basis": "chebyshev",
        "alpha": 0.5,
    }
    entry.update(common)
    entry.update(
        objective="absolute",
        fit_method="prefill256_calibrated_chebyshev",
        uniform_max_abs=error,
    )
    entry["chebyshev_fhe"] = common.copy()
    table["invsqrt_alg2"][stage]["by_layer"][key] = entry
    if stage == "j1_rough":
        table["layers"][key]["j1_rough_fit_interval"] = [lo, hi]
    return {
        "layer": layer,
        "stage": stage,
        "old_interval": list(_bounds(old)),
        "interval": [lo, hi],
        "old_degree": int(old["degree"]),
        "degree": degree,
        "old_depth": chebyshev_ps_mul_depth(old["coefficients"]),
        "depth": chebyshev_ps_mul_depth(coefficients),
        "uniform_max_abs": error,
    }


def _fit_exp(table: dict, layer: int, observed, *, margin: float):
    key = str(layer)
    old = table["layers"][key]
    interval = _required_interval(
        tuple(float(v) for v in old["fit_interval"]), observed, margin=margin
    )
    if interval is None:
        return None
    lo, hi = interval
    degree, coefficients, error = _fit(
        np.exp, lo, hi, initial_degree=int(old["degree"]), tol=TOL
    )
    ps_k, ps_m = compute_degrees_ps(degree)
    old_degree = int(old["degree"])
    old_depth = chebyshev_ps_mul_depth(old["coefficients"])
    old_interval = list(old["fit_interval"])
    old.update(
        degree=degree,
        coefficients=coefficients.tolist(),
        fit_interval=[lo, hi],
        exp_fit_interval=[lo, hi],
        max_error=error,
        max_error_target=TOL,
        ps_k=ps_k,
        ps_m=ps_m,
        fit_method="prefill256_calibrated_chebyshev",
    )
    return {
        "layer": layer,
        "stage": "exp",
        "old_interval": old_interval,
        "interval": [lo, hi],
        "old_degree": old_degree,
        "degree": degree,
        "old_depth": old_depth,
        "depth": chebyshev_ps_mul_depth(coefficients),
        "uniform_max_abs": error,
    }


def _merge_bounds(rows: list[dict], field: str) -> tuple[float, float]:
    ranges = [row[field] for row in rows]
    return min(pair[0] for pair in ranges), max(pair[1] for pair in ranges)


def _run_stack(
    *, loader, metadata, hidden_by_length: dict[int, np.ndarray], approx, exact_invsqrt: bool
):
    hidden = {length: values.copy() for length, values in hidden_by_length.items()}
    records: dict[int, list[dict]] = {}
    original_eval = layer_mod.eval_chebyshev_series_direct
    row_sum_inputs: list[tuple[float, float]] = []

    def mapped_eval(values, coeffs, lo, hi):
        values = np.asarray(values)
        if values.ndim == 3 and values.shape[-1] == 1:
            row_sum_inputs.append((float(values.min()), float(values.max())))
            if exact_invsqrt:
                return np.power(values, -0.5).astype(np.float32)
        return original_eval(values, coeffs, lo, hi)

    layer_mod.eval_chebyshev_series_direct = mapped_eval
    try:
        for layer in range(int(metadata.num_hidden_layers)):
            weights = load_layer_weights(loader, layer)
            records[layer] = []
            for length in sorted(hidden):
                row_sum_inputs.clear()
                token_valid_mask = None
                if length < hidden[length].shape[0]:
                    token_valid_mask = np.arange(hidden[length].shape[0]) < length
                result = layer_mod.poly_prefill_layer_numpy(
                    hidden[length],
                    weights=weights,
                    approximations=approx.for_layer(layer),
                    attention_layout=AttentionPairLayout(),
                    layer_idx=layer,
                    rope_theta=float(metadata.rope_theta),
                    token_valid_mask=token_valid_mask,
                )
                hidden[length] = result.hidden
                if not np.isfinite(result.hidden).all():
                    raise ArithmeticError(f"nonfinite hidden at layer {layer}, length {length}")
                report = result.softmax_report
                expected_rounds = int(approx.for_layer(layer).softmax.k_for(layer))
                if len(row_sum_inputs) != expected_rounds:
                    raise AssertionError(
                        f"layer {layer}, length {length}: {len(row_sum_inputs)} "
                        f"row sums vs {expected_rounds} rounds"
                    )
                records[layer].append(
                    {
                        "length": length,
                        "exp": [report["exp_min"], report["exp_max"]],
                        "j1_rough": list(row_sum_inputs[0]),
                        "j2_precise": [
                            min(pair[0] for pair in row_sum_inputs[1:]),
                            max(pair[1] for pair in row_sum_inputs[1:]),
                        ] if len(row_sum_inputs) > 1 else None,
                    }
                )
            print(f"calibrated L{layer:02d} ({"exact" if exact_invsqrt else "poly"})", flush=True)
    finally:
        layer_mod.eval_chebyshev_series_direct = original_eval
    return records, hidden


def _refit(table: dict, records: dict, *, margin: float) -> list[dict]:
    changed = []
    for layer, samples in records.items():
        exp = _fit_exp(table, layer, _merge_bounds(samples, "exp"), margin=margin)
        if exp is not None:
            changed.append(exp)
        j1 = _fit_stage(
            table, layer, "j1_rough", _merge_bounds(samples, "j1_rough"), margin=margin
        )
        if j1 is not None:
            changed.append(j1)
        if samples[0]["j2_precise"] is not None:
            j2 = _fit_stage(
                table, layer, "j2_precise", _merge_bounds(samples, "j2_precise"), margin=margin
            )
            if j2 is not None:
                changed.append(j2)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOFTMAX)
    parser.add_argument("--input-ids-128", type=Path, default=DEFAULT_128_IDS)
    parser.add_argument("--input-ids-129", type=Path, default=DEFAULT_129_IDS)
    parser.add_argument("--input-ids-256", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--margin", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.0 <= args.margin < 1.0:
        raise ValueError("margin must be in [0, 1).")

    metadata = CheckpointMetadata.from_json(args.model_dir / "config.json")
    loader = WeightLoader(args.model_dir, tensor_dtype=np.float32)
    ids128 = _input_ids(args.input_ids_128, expected=128)
    ids129 = _input_ids(args.input_ids_129, expected=129)
    ids256 = (
        _input_ids(args.input_ids_256, expected=256)
        if args.input_ids_256 is not None
        else np.tile(ids128, 2)
    )
    embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
    ids129_padded = np.pad(ids129, (0, 256 - len(ids129)), constant_values=ids129[-1])
    hidden = {
        129: np.asarray(embedding[ids129_padded], dtype=np.float32),
        256: np.asarray(embedding[ids256], dtype=np.float32),
    }
    del embedding

    table = json.loads(args.source.read_text(encoding="utf-8"))
    table["max_prefill_pages"] = 2
    table["variant_meta"] = dict(table.get("variant_meta") or {})
    table["variant_meta"]["prefill256"] = {
        "source": _portable_path(args.source),
        "calibration_lengths": [129, 256],
        "calibration_fixture_129": _portable_path(args.input_ids_129),
        "calibration_fixture_256": (
            _portable_path(args.input_ids_256)
            if args.input_ids_256 is not None
            else f"{_portable_path(args.input_ids_128)} repeated twice"
        ),
        "padded_width": 256,
        "page_size": PAGE_SIZE,
        "second_block_exp_scale": 1.0 / math.sqrt(2.0),
        "margin": args.margin,
        "absolute_tolerance": TOL,
        "note": "Calibration is sample-based; 3+ pages need a separate artifact.",
    }
    approx = ApproximationConfig(
        rmsnorm_path=DEFAULT_RMS,
        silu_path=DEFAULT_SILU,
        softmax_path=args.source,
    )
    exact_records, exact_hidden = _run_stack(
        loader=loader, metadata=metadata, hidden_by_length=hidden,
        approx=approx, exact_invsqrt=True,
    )
    changes = _refit(table, exact_records, margin=args.margin)
    for attempt in range(2):
        approx._softmax = SoftmaxPolyConfig.from_artifact(table)
        approx._layers.clear()
        poly_records, poly_hidden = _run_stack(
            loader=loader, metadata=metadata, hidden_by_length=hidden,
            approx=approx, exact_invsqrt=False,
        )
        additional = _refit(table, poly_records, margin=args.margin)
        changes.extend(additional)
        if not additional:
            break
    else:
        raise RuntimeError("Poly validation still exceeded the fitted domains.")

    differences = {
        str(length): {
            "max_abs": float(np.max(np.abs(poly_hidden[length] - exact_hidden[length]))),
            "mean_abs": float(np.mean(np.abs(poly_hidden[length] - exact_hidden[length]))),
        }
        for length in hidden
    }
    table["variant_meta"]["prefill256"].update(
        refit=changes,
        poly_vs_exact_invsqrt_hidden=differences,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}; {len(changes)} refits; differences={differences}", flush=True)


if __name__ == "__main__":
    main()
