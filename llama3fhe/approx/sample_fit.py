"""Reuse Prefill nonlinear artifacts, or refit them from a plaintext stack.

RMSNorm and SiLU are the same functions as Prefill; Decode Softmax is the
same Alg2 coefficients with a different packing.  This module samples the
exact plaintext layer, keeps a Prefill artifact when every sample sits in
its interval, and otherwise refits Chebyshev coefficients on a padded
interval at the Prefill degree.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from llama3fhe.approx.chebyshev import chebyshev_degree, eval_chebyshev_coefficients
from llama3fhe.approx.prefix_stack import (
    build_prefix_kv_bundles,
    run_exact_content_layer,
    softmax_valid_mask,
)
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("no finite samples to fit.")
    return array


def _covers(values, interval) -> bool:
    lo, hi = (float(interval[0]), float(interval[1]))
    array = _finite(values)
    return bool(np.min(array) >= lo and np.max(array) <= hi)


def _pad_positive(values, *, eps: float, pad: float) -> tuple[float, float]:
    array = _finite(values)
    lo = max(float(eps), float(np.min(array)) * (1.0 - float(pad)))
    hi = max(lo + float(eps), float(np.max(array)) * (1.0 + float(pad)))
    return lo, hi


def _pad_signed(values, *, pad: float, floor: float) -> tuple[float, float]:
    array = _finite(values)
    span = max(float(np.max(np.abs(array))) * (1.0 + float(pad)), float(floor))
    return -span, span


def _entry_degree(entry: dict[str, Any], *, default: int) -> int:
    if "degree" in entry:
        return int(entry["degree"])
    coeffs = entry.get("coefficients")
    if coeffs is None:
        return int(default)
    return max(int(default), int(chebyshev_degree(coeffs)))


def _fit_rmsn(values, *, degree: int, eps: float, pad: float) -> dict[str, Any]:
    lo, hi = _pad_positive(values, eps=eps, pad=pad)
    coeffs = eval_chebyshev_coefficients(
        lambda value: 1.0 / math.sqrt(float(value) + float(eps)),
        lo,
        hi,
        int(degree),
    )
    return {
        "coefficients": np.asarray(coeffs, dtype=np.float64),
        "fit_interval": [lo, hi],
        "eps": float(eps),
        "degree": int(degree),
        "basis": "chebyshev",
        "objective": "1/sqrt(s+eps)",
        "method": "sample_fit",
    }


def _fit_silu(values, *, degree: int, pad: float) -> dict[str, Any]:
    lo, hi = _pad_signed(values, pad=pad, floor=1.0)
    coeffs = eval_chebyshev_coefficients(
        lambda value: float(value) / (1.0 + math.exp(-float(value))),
        lo,
        hi,
        int(degree),
    )
    return {
        "coefficients": np.asarray(coeffs, dtype=np.float64),
        "fit_interval": [lo, hi],
        "degree": int(degree),
        "basis": "chebyshev",
        "method": "sample_fit",
    }


def _invsqrt_interval(entry: dict[str, Any]) -> tuple[float, float]:
    cheb = entry.get("chebyshev_fhe") or entry
    lo, hi = (float(value) for value in cheb["fit_interval"])
    return lo, hi


def _softmax_live_and_row_ss(
    scores: np.ndarray,
    *,
    softmax,
    layer_idx: int,
    prefix_k: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError(f"GQA scores must be [H,Q,K], got {scores.shape}.")
    valid = softmax_valid_mask(scores, prefix_k=int(prefix_k))
    vals = scores[:, valid]
    k = int(softmax.k_for(layer_idx))
    u_hat = float(softmax.U_hat_for(layer_idx))
    scale = 1.0 / float(2**k)
    shift = -u_hat * scale
    row_ss = []
    for head in scores:
        for query_index, row in enumerate(head):
            live = np.asarray(row[valid[query_index]], dtype=np.float64)
            y = np.exp(live * scale + shift)
            row_ss.append(float(np.sum(y * y)))
    return vals, np.asarray(row_ss, dtype=np.float64)


def _softmax_covers(
    scores: np.ndarray,
    softmax,
    *,
    layer_idx: int,
    prefix_k: int = 0,
) -> bool:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError(f"GQA scores must be [H,Q,K], got {scores.shape}.")
    vals, row_ss = _softmax_live_and_row_ss(
        scores, softmax=softmax, layer_idx=layer_idx, prefix_k=int(prefix_k)
    )
    u_hat = float(softmax.U_hat_for(layer_idx))
    stats_min = float(softmax.stats_min_for(layer_idx))
    if float(np.max(vals)) > u_hat or float(np.min(vals)) < stats_min:
        return False
    k = int(softmax.k_for(layer_idx))
    mapped = vals / float(2**k) - u_hat / float(2**k)
    exp_lo, exp_hi = (
        float(value) for value in softmax.layer_exp[layer_idx]["fit_interval"]
    )
    if not _covers(mapped, (exp_lo, exp_hi)):
        return False
    j1 = softmax.poly_for_layer(softmax.invsqrt_alg2["j1_rough"], layer_idx)
    return _covers(row_ss, _invsqrt_interval(j1))


def _fit_softmax(
    scores: np.ndarray,
    *,
    softmax,
    layer_idx: int,
    pad: float,
    prefix_k: int = 0,
):
    scores = np.asarray(scores, dtype=np.float64)
    valid = softmax_valid_mask(scores, prefix_k=int(prefix_k))
    vals = _finite(scores[:, valid])
    k = int(softmax.k_for(layer_idx))
    degree = _entry_degree(softmax.layer_exp[layer_idx], default=8)
    u_hat = float(np.max(vals)) * (1.0 + float(pad))
    stats_min = min(float(np.min(vals)) * (1.0 + float(pad)), 0.0)
    scale = 1.0 / float(2**k)
    mapped = vals * scale - u_hat * scale
    exp_lo = float(np.min(mapped)) - float(pad) * abs(float(np.min(mapped)))
    exp_hi = float(np.max(mapped)) + float(pad) * abs(float(np.max(mapped)))
    if exp_hi <= exp_lo:
        exp_hi = exp_lo + 1e-3
    row_ss = []
    shift = -u_hat * scale
    for head in scores:
        for query_index, row in enumerate(head):
            live = np.asarray(row[valid[query_index]], dtype=np.float64)
            y = np.exp(live * scale + shift)
            row_ss.append(float(np.sum(y * y)))
    inv_lo, inv_hi = _pad_positive(row_ss, eps=1e-4, pad=pad)
    return DecodeAlg2Poly.synthetic(
        k=k,
        degree=degree,
        U_hat=u_hat,
        stats_min=stats_min,
        exp_interval=(exp_lo, exp_hi),
        invsqrt_interval=(inv_lo, inv_hi),
    )


def run_exact_layer(
    hidden: np.ndarray,
    *,
    weights,
    attention_layout: AttentionPairLayout,
    rope_theta: float = 500000.0,
    eps: float = 1e-5,
    prefix_bundle=None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One plaintext layer: exact RMSNorm, causal GQA (optional prefix), exact SiLU."""

    return run_exact_content_layer(
        hidden,
        weights=weights,
        layout=attention_layout,
        prefix_bundle=prefix_bundle,
        rope_theta=float(rope_theta),
        eps=float(eps),
    )


@dataclass(frozen=True)
class FittedLayerReport:
    layer: int
    reused: dict[str, bool]
    degrees: dict[str, int]


def fit_layer_from_plaintext(
    hidden: np.ndarray,
    *,
    weights,
    attention_layout: AttentionPairLayout,
    layer_idx: int = 0,
    extra_rmsn1: np.ndarray | None = None,
    prefill=None,
    rope_theta: float = 500000.0,
    eps: float = 1e-5,
    pad: float = 1.0,
    prefix_bundle=None,
):
    """Sample one exact layer and reuse or refit its Prefill polynomials.

    ``prefill`` is a production :class:`LayerApproximations` (or ``None`` to
    load layer ``layer_idx`` from the release artifacts).  ``prefix_bundle``
    switches sampling to the FHE kv_cache protocol (content 128, RoPE at k).
    """

    from llama3fhe.graph import LayerApproximations

    layer_idx = int(layer_idx)
    if prefill is None:
        from llama3fhe.application.approx_config import ApproximationConfig
        from llama3fhe.application.args import (
            DEFAULT_RMS,
            DEFAULT_SILU,
            DEFAULT_SOFTMAX,
        )

        prefill = ApproximationConfig(
            rmsnorm_path=DEFAULT_RMS,
            silu_path=DEFAULT_SILU,
            softmax_path=DEFAULT_SOFTMAX,
        ).for_layer(layer_idx)

    output, samples = run_exact_layer(
        hidden,
        weights=weights,
        attention_layout=attention_layout,
        rope_theta=float(rope_theta),
        eps=float(eps),
        prefix_bundle=prefix_bundle,
    )
    rmsn1_var = samples["rmsn1_var"]
    if extra_rmsn1 is not None:
        extra = np.mean(
            np.square(np.asarray(extra_rmsn1, dtype=np.float64)),
            axis=-1,
        )
        rmsn1_var = np.concatenate([rmsn1_var.reshape(-1), extra.reshape(-1)])

    reused = {}
    if _covers(rmsn1_var, prefill.input_norm["fit_interval"]):
        input_norm = prefill.input_norm
        reused["input_norm"] = True
    else:
        input_norm = _fit_rmsn(
            rmsn1_var,
            degree=_entry_degree(prefill.input_norm, default=16),
            eps=float(prefill.input_norm.get("eps", eps)),
            pad=pad,
        )
        reused["input_norm"] = False

    if _covers(samples["rmsn2_var"], prefill.post_attention_norm["fit_interval"]):
        post_norm = prefill.post_attention_norm
        reused["post_attention_norm"] = True
    else:
        post_norm = _fit_rmsn(
            samples["rmsn2_var"],
            degree=_entry_degree(prefill.post_attention_norm, default=16),
            eps=float(prefill.post_attention_norm.get("eps", eps)),
            pad=pad,
        )
        reused["post_attention_norm"] = False

    if _covers(samples["silu_gate"], prefill.silu["fit_interval"]):
        silu = prefill.silu
        reused["silu"] = True
    else:
        silu = _fit_silu(
            samples["silu_gate"],
            degree=_entry_degree(prefill.silu, default=48),
            pad=pad,
        )
        reused["silu"] = False

    prefix_k = 0 if prefix_bundle is None else int(prefix_bundle.prefix_k)
    if _softmax_covers(
        samples["scores"], prefill.softmax, layer_idx=layer_idx, prefix_k=prefix_k
    ):
        softmax = prefill.softmax
        reused["softmax"] = True
        softmax_degree = _entry_degree(
            prefill.softmax.layer_exp[layer_idx], default=8
        )
    else:
        softmax = _fit_softmax(
            samples["scores"],
            softmax=prefill.softmax,
            layer_idx=layer_idx,
            pad=pad,
            prefix_k=prefix_k,
        )
        reused["softmax"] = False
        softmax_degree = int(chebyshev_degree(softmax.exp_entry["coefficients"]))

    approximations = LayerApproximations(
        input_norm=input_norm,
        post_attention_norm=post_norm,
        silu=silu,
        softmax=softmax,
    )
    report = FittedLayerReport(
        layer=layer_idx,
        reused=reused,
        degrees={
            "input_norm": _entry_degree(input_norm, default=16),
            "post_attention_norm": _entry_degree(post_norm, default=16),
            "silu": _entry_degree(silu, default=48),
            "softmax_exp": softmax_degree,
        },
    )
    return approximations, output, report


def fit_stack_from_plaintext(
    prefill_hidden: np.ndarray,
    layer_weights: Sequence[Any],
    *,
    attention_layout: AttentionPairLayout,
    decode_hidden: np.ndarray | None = None,
    prefill_config=None,
    rope_theta: float = 500000.0,
    eps: float = 1e-5,
    pad: float = 1.0,
    prefix_hidden: np.ndarray | None = None,
) -> tuple[list, list[FittedLayerReport]]:
    """Fit one approximation object per layer along the exact Prefill stack.

    ``prefix_hidden`` is the k-token PP residual at layer 0 (embeddings of
    ``\\n``×k).  When set, each layer is sampled with the matching C-form
    prefix cache page so RMSN/SiLU/Softmax intervals are content-128 only.
    """

    hidden = np.asarray(prefill_hidden, dtype=np.float32)
    extra = None if decode_hidden is None else np.asarray(decode_hidden, dtype=np.float32)
    prefix_bundles = None
    if prefix_hidden is not None:
        prefix_bundles, _ = build_prefix_kv_bundles(
            prefix_hidden,
            layer_weights,
            layout=attention_layout,
            rope_theta=float(rope_theta),
            eps=float(eps),
        )
    approximations = []
    reports = []
    for index, weights in enumerate(layer_weights):
        prefill = None
        if prefill_config is not None:
            prefill = prefill_config.for_layer(int(index))
        bundle = None if prefix_bundles is None else prefix_bundles[index]
        fitted, hidden, report = fit_layer_from_plaintext(
            hidden,
            weights=weights,
            attention_layout=attention_layout,
            layer_idx=int(index),
            extra_rmsn1=extra if index == 0 else None,
            prefill=prefill,
            rope_theta=float(rope_theta),
            eps=float(eps),
            pad=float(pad),
            prefix_bundle=bundle,
        )
        approximations.append(fitted)
        reports.append(report)
    return approximations, reports


def _jsonify(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def export_fitted_poly_artifacts(
    approximations: Sequence[Any],
    reports: Sequence[FittedLayerReport],
    *,
    rms_path: str | Path,
    silu_path: str | Path,
    softmax_path: str | Path,
    output_dir: str | Path,
    prefix_k: int,
) -> dict[str, str]:
    """Write patched RMSN / SiLU / Softmax JSONs.  Unchanged layers stay as-is."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rms = json.loads(Path(rms_path).read_text(encoding="utf-8"))
    silu = json.loads(Path(silu_path).read_text(encoding="utf-8"))
    softmax = json.loads(Path(softmax_path).read_text(encoding="utf-8"))
    rms["source"] = f"prefix_k{int(prefix_k)}_sample_fit"
    silu["source"] = f"prefix_k{int(prefix_k)}_sample_fit"
    softmax["source"] = f"prefix_k{int(prefix_k)}_sample_fit"
    softmax["prefix_k"] = int(prefix_k)
    softmax["use_content_only"] = True
    for approx, report in zip(approximations, reports, strict=True):
        key = str(int(report.layer))
        if not report.reused.get("input_norm", True):
            rms.setdefault("branches", {}).setdefault("attn", {})[key] = _jsonify(
                approx.input_norm
            )
        if not report.reused.get("post_attention_norm", True):
            rms.setdefault("branches", {}).setdefault("ffn", {})[key] = _jsonify(
                approx.post_attention_norm
            )
        if not report.reused.get("silu", True):
            silu.setdefault("per_layer", {}).setdefault(key, {}).setdefault(
                "tiers", {}
            )["standard"] = _jsonify(approx.silu)
        if not report.reused.get("softmax", True):
            fitted = approx.softmax
            row = dict(softmax.get("layers", {}).get(key) or {})
            row.update(
                {
                    "layer": int(report.layer),
                    "U_hat": float(fitted.U_hat_for(report.layer)),
                    "stats_min": float(fitted.stats_min_for(report.layer)),
                    "k": int(fitted.k_for(report.layer)),
                    "fit_interval": list(fitted.exp_entry["fit_interval"]),
                    "exp_fit_interval": list(fitted.exp_entry["fit_interval"]),
                    "coefficients": _jsonify(fitted.exp_entry["coefficients"]),
                    "degree": int(chebyshev_degree(fitted.exp_entry["coefficients"])),
                    "method": "sample_fit_prefix",
                }
            )
            softmax.setdefault("layers", {})[key] = row
            inv = fitted.invsqrt_alg2["j1_rough"]
            j1 = softmax.setdefault("invsqrt_alg2", {}).setdefault("j1_rough", {})
            by_layer = j1.setdefault("by_layer", {})
            coeffs = _jsonify(inv["coefficients"])
            interval = list(inv["fit_interval"])
            degree = int(chebyshev_degree(inv["coefficients"]))
            cheb = inv.get("chebyshev_fhe") or {
                "degree": degree,
                "fit_interval": interval,
                "coefficients": coeffs,
                "basis": "chebyshev",
                "alpha": 0.5,
            }
            by_layer[key] = {
                "fit_interval": interval,
                "degree": degree,
                "coefficients": coeffs,
                "alpha": float(inv.get("alpha", 0.5)),
                "basis": str(inv.get("basis", "chebyshev")),
                "objective": "absolute",
                "fit_method": "sample_fit_prefix",
                "chebyshev_fhe": _jsonify(cheb),
            }
    paths = {
        "rmsnorm": str(output_dir / f"rmsnorm_poly_coeffs_prefix_k{int(prefix_k)}.json"),
        "silu": str(output_dir / f"silu_poly_coeffs_prefix_k{int(prefix_k)}.json"),
        "softmax": str(output_dir / f"softmax_poly_coeffs_prefix_k{int(prefix_k)}.json"),
    }
    Path(paths["rmsnorm"]).write_text(json.dumps(rms) + "\n")
    Path(paths["silu"]).write_text(json.dumps(silu) + "\n")
    Path(paths["softmax"]).write_text(json.dumps(softmax) + "\n")
    return paths


__all__ = [
    "FittedLayerReport",
    "export_fitted_poly_artifacts",
    "fit_layer_from_plaintext",
    "fit_stack_from_plaintext",
    "run_exact_layer",
]
