from __future__ import annotations

"""Plaintext poly transformer layer used as the FHE diagnostic reference.

Polynomial RMSNorm, Chebyshev SiLU, and either exact causal GQA or the same
Alg2 Softmax polynomials the encrypted kernel evaluates.  Diagnostic diffs
against this graph isolate FHE bugs from approximation error.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..layouts.attention import AttentionPairLayout
from ..positions import content_rope_start_pos
from .attention import apply_rope_heads, attention_gqa_numpy, rope_cos_sin
from .chebyshev import eval_chebyshev_series_direct
from .rmsnorm import is_rmsn_weight, rmsnorm_poly_numpy


@dataclass(frozen=True)
class PolyLayerResult:
    hidden: np.ndarray
    silu_gate_max_abs: float | None
    silu_oob_frac: float | None
    silu_fit_lo: float
    silu_fit_hi: float
    rmsn1_s_max: float | None = None
    rmsn1_oob_frac: float | None = None
    rmsn1_fit: tuple[float, float] | None = None
    rmsn2_s_max: float | None = None
    rmsn2_oob_frac: float | None = None
    rmsn2_fit: tuple[float, float] | None = None


def _poly_entry_interval(entry: dict) -> tuple[np.ndarray, float, float]:
    coeffs = np.asarray(entry["coefficients"], dtype=np.float64)
    lo, hi = (float(value) for value in entry["fit_interval"])
    return coeffs, lo, hi


def _rmsnorm_poly(
    values: np.ndarray,
    entry: dict,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the FHE RMSNorm polynomial and its logical output gamma.

    The encrypted graph may defer gamma and absorb it into its following
    plaintext linear weights.  Applying it here is mathematically identical
    and keeps this NumPy reference independent of that scheduling choice.
    """

    coeffs, lo, hi = _poly_entry_interval(entry)
    normalized = rmsnorm_poly_numpy(
        values, coeffs, (lo, hi), skip_clamp=True
    )
    if weight is None:
        return normalized
    gamma = np.asarray(weight, dtype=np.float32).reshape(-1)
    if is_rmsn_weight(gamma):
        return normalized
    expected = (int(normalized.shape[-1]),)
    if gamma.shape != expected:
        raise ValueError(
            f"RMSNorm gamma must have shape {expected}, got {gamma.shape}."
        )
    return np.asarray(normalized * gamma, dtype=np.float32)


def _silu_poly(values: np.ndarray, entry: dict) -> np.ndarray:
    coeffs, lo, hi = _poly_entry_interval(entry)
    return eval_chebyshev_series_direct(values, coeffs, lo, hi)


def _variance_domain(values: np.ndarray, entry: dict) -> tuple[float | None, float | None, tuple[float, float]]:
    """Token-wise mean(x²) vs the rsqrt Chebyshev interval (FHE does not clip)."""

    s = np.mean(np.square(np.asarray(values, dtype=np.float32)), axis=-1)
    lo, hi = (float(v) for v in entry["fit_interval"])
    finite = np.isfinite(s)
    if not finite.any():
        return None, None, (lo, hi)
    s_f = s[finite]
    oob = float(np.mean((s < lo) | (s > hi)))
    return float(np.max(s_f)), oob, (lo, hi)


def poly_transformer_layer_numpy(
    hidden: np.ndarray,
    *,
    weights,
    approximations,
    attention_layout: AttentionPairLayout,
    rope_theta: float = 500000.0,
    skip_attention: bool = False,
    prefix_hidden: np.ndarray | None = None,
    prefix_bundle=None,
) -> PolyLayerResult:
    """One transformer layer with poly RMSNorm / SiLU and exact Softmax.

    ``skip_attention=True`` keeps RMSN1 as a discarded diagnostic and uses an
    identity attention residual, matching ``graph.evaluate(..., skip_attention=True)``.

    Prefix tokens stay off the content residual: ``prefix_hidden`` (k tokens)
    is PP-only (RMSN/QKV/RoPE at positions ``0..k-1``).  Content RoPE starts
    at ``k``.  ``prefix_bundle`` supplies already-RoPE'd C-form K/V instead.
    """

    residual = np.asarray(hidden, dtype=np.float32)
    seq_len, hidden_dim = residual.shape
    expected_q = (hidden_dim, hidden_dim)
    if weights.q_proj.shape != expected_q:
        raise ValueError(
            f"q_proj must have shape {expected_q}, got {weights.q_proj.shape}."
        )
    kv_width = int(attention_layout.key_value_heads) * int(attention_layout.head_dim)
    if weights.k_proj.shape != (kv_width, hidden_dim):
        raise ValueError(
            "k_proj must have shape "
            f"{(kv_width, hidden_dim)}, got {weights.k_proj.shape}."
        )

    rmsn1_s_max, rmsn1_oob_frac, rmsn1_fit = _variance_domain(
        residual, approximations.input_norm
    )
    attn_in = _rmsnorm_poly(
        residual, approximations.input_norm, weights.input_layernorm
    )
    if not skip_attention:
        from .prefix import (
            PrefixKVBundle,
            attention_prefix_gqa_numpy,
        )

        query = attn_in @ np.asarray(weights.q_proj, dtype=np.float32).T
        key = attn_in @ np.asarray(weights.k_proj, dtype=np.float32).T
        value = attn_in @ np.asarray(weights.v_proj, dtype=np.float32).T
        prefix_k = 0
        key_p = value_p = None
        if prefix_bundle is not None and prefix_hidden is not None:
            raise ValueError("pass prefix_hidden or prefix_bundle, not both.")
        if prefix_bundle is not None:
            bundle = prefix_bundle
            if not isinstance(bundle, PrefixKVBundle):
                bundle = PrefixKVBundle.from_logical(
                    bundle.key, bundle.value, layout=attention_layout
                )
            prefix_k = int(bundle.prefix_k)
            key_p = bundle.key
            value_p = bundle.value
        elif prefix_hidden is not None:
            prefix_residual = np.asarray(prefix_hidden, dtype=np.float32)
            prefix_k = int(prefix_residual.shape[0])
            prefix_in = _rmsnorm_poly(
                prefix_residual,
                approximations.input_norm,
                weights.input_layernorm,
            )
            key_p = prefix_in @ np.asarray(weights.k_proj, dtype=np.float32).T
            value_p = prefix_in @ np.asarray(weights.v_proj, dtype=np.float32).T
            cos_p, sin_p = rope_cos_sin(
                seq_len=prefix_k,
                head_dim=int(attention_layout.head_dim),
                theta=float(rope_theta),
                start_pos=0,
            )
            key_p = apply_rope_heads(
                key_p,
                cos_p,
                sin_p,
                num_heads=int(attention_layout.key_value_heads),
                head_dim=int(attention_layout.head_dim),
            )
        cos, sin = rope_cos_sin(
            seq_len=int(seq_len),
            head_dim=int(attention_layout.head_dim),
            theta=float(rope_theta),
            start_pos=int(prefix_k),
        )
        query = apply_rope_heads(
            query,
            cos,
            sin,
            num_heads=int(attention_layout.query_heads),
            head_dim=int(attention_layout.head_dim),
        )
        key = apply_rope_heads(
            key,
            cos,
            sin,
            num_heads=int(attention_layout.key_value_heads),
            head_dim=int(attention_layout.head_dim),
        )
        if prefix_k:
            attn_out = attention_prefix_gqa_numpy(
                query, key_p, key, value_p, value, layout=attention_layout
            ).output
        else:
            _, _, attn_out = attention_gqa_numpy(
                query, key, value, layout=attention_layout
            )
        residual = residual + attn_out @ np.asarray(weights.o_proj, dtype=np.float32).T

    rmsn2_s_max, rmsn2_oob_frac, rmsn2_fit = _variance_domain(
        residual, approximations.post_attention_norm
    )
    mlp_in = _rmsnorm_poly(
        residual,
        approximations.post_attention_norm,
        weights.post_attention_layernorm,
    )
    gate = mlp_in @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = mlp_in @ np.asarray(weights.up_proj, dtype=np.float32).T
    silu_entry = approximations.silu
    _, lo, hi = _poly_entry_interval(silu_entry)
    finite_gate = np.isfinite(gate)
    if finite_gate.any():
        gate_abs = np.abs(gate[finite_gate])
        silu_gate_max_abs = float(np.max(gate_abs))
        silu_oob_frac = float(np.mean((gate < lo) | (gate > hi)))
    else:
        silu_gate_max_abs = None
        silu_oob_frac = None
    activated = _silu_poly(gate, silu_entry) * up
    residual = residual + activated @ np.asarray(weights.down_proj, dtype=np.float32).T
    return PolyLayerResult(
        hidden=np.asarray(residual, dtype=np.float32),
        silu_gate_max_abs=silu_gate_max_abs,
        silu_oob_frac=silu_oob_frac,
        silu_fit_lo=lo,
        silu_fit_hi=hi,
        rmsn1_s_max=rmsn1_s_max,
        rmsn1_oob_frac=rmsn1_oob_frac,
        rmsn1_fit=rmsn1_fit,
        rmsn2_s_max=rmsn2_s_max,
        rmsn2_oob_frac=rmsn2_oob_frac,
        rmsn2_fit=rmsn2_fit,
    )


def _oob_frac(values: np.ndarray, lo: float, hi: float) -> float | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.mean((array < float(lo)) | (array > float(hi))))


def _as_heads(
    values: np.ndarray,
    *,
    heads: int,
    head_dim: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[1:] == (int(heads), int(head_dim)):
        return values
    width = int(heads) * int(head_dim)
    if values.ndim == 2 and int(values.shape[1]) == width:
        return values.reshape(int(values.shape[0]), int(heads), int(head_dim))
    raise ValueError(
        f"{name} must be [seq, {width}] or [seq, {heads}, {head_dim}], got {values.shape}."
    )


def _gqa_unscaled_scores(
    query: np.ndarray,
    key: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unscaled ``[H,Q,K]`` scores plus GQA-repeated K/V heads."""

    query = _as_heads(
        query,
        heads=int(layout.query_heads),
        head_dim=int(layout.head_dim),
        name="query",
    )
    key = _as_heads(
        key,
        heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
        name="key",
    )
    ratio = layout.gqa_ratio
    key_for_q = key[:, np.arange(int(layout.query_heads)) // ratio, :]
    scores = np.einsum("qhd,khd->hqk", query, key_for_q, optimize=True)
    return np.asarray(scores, dtype=np.float64), query, key_for_q


def _content_causal_valid_mask(
    seq_len: int,
    *,
    token_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Logical counterpart of the encrypted Delta causal/padding mask.

    An invalid query retains its public self sentinel so Alg2 never receives
    an all-zero row, while invalid keys cannot affect a valid query.
    """

    seq_len = int(seq_len)
    queries = np.arange(seq_len, dtype=np.int64)[:, None]
    keys = np.arange(seq_len, dtype=np.int64)[None, :]
    valid = keys <= queries
    if token_valid_mask is None:
        return valid
    token_valid = np.asarray(token_valid_mask, dtype=bool).reshape(-1)
    if token_valid.shape != (seq_len,):
        raise ValueError(
            "token_valid_mask must have shape "
            f"{(seq_len,)}, got {token_valid.shape}."
        )
    valid &= token_valid[None, :]
    valid &= token_valid[:, None]
    invalid_queries = np.flatnonzero(~token_valid)
    valid[invalid_queries, invalid_queries] = True
    return valid


def softmax_alg2_poly_scores_numpy(
    scores: np.ndarray,
    *,
    config,
    layer_idx: int,
    head_dim: int,
    valid: np.ndarray | None = None,
    variant: str = "alg2",
    prefill_page_size: int | None = None,
    query_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Chebyshev Alg2 on unscaled GQA scores ``[heads, queries, keys]``.

    Same public map as Prefill/Decode FHE: valid slots
    ``(s/√d - U_hat)/2^k``, invalid ``(stats_min - U_hat)/2^k``, then
    Chebyshev exp and ``k`` invsqrt rounds.  Invalid keys are zeroed after
    each polynomial so they do not enter ``‖y‖₂``.

    ``variant="alg2"`` is the per-round form ``y ← (y·λ_t)²``.
    ``variant="schain"`` is the algebraically equivalent S-chain
    (Algorithm 4′): the main track only squares, ``P_t = P_{t-1}²``, the
    row sums ``S_t = Σ_i (P_t)_i`` are λ-free, the auxiliary track
    accumulates ``Λ_{t+1} = (Λ_t·λ_t)²`` with ``λ_t = rsqrt_t(Λ_t²·S_t)``,
    and the output is the single product ``P_k·Λ_{k+1}``.  Both variants
    consume the same exp / j1 / j2 tables and see identical rsqrt inputs.
    """

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError(f"Alg2 scores must be [H,Q,K], got {scores.shape}.")
    _, n_query, n_key = scores.shape
    if valid is None:
        valid = np.ones((n_query, n_key), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (n_query, n_key):
            raise ValueError(
                f"Alg2 valid mask {valid.shape} != {(n_query, n_key)}."
            )
    if variant not in ("alg2", "schain"):
        raise ValueError(f"Unknown Alg2 variant {variant!r}.")
    layer_idx = int(layer_idx)
    k = int(config.k_for(layer_idx))
    scale = 1.0 / math.sqrt(float(head_dim))
    shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    invalid = float(config.stats_min_for(layer_idx)) / float(2**k)
    mapped = np.where(
        valid[None, :, :],
        scores * (scale / float(2**k)) + shift,
        shift + invalid,
    )
    exp_entry = config.layer_exp[layer_idx]
    exp_lo, exp_hi = (float(value) for value in exp_entry["fit_interval"])
    live = mapped[:, valid]
    report: dict[str, object] = {
        "k": k,
        "variant": variant,
        "exp_interval": (exp_lo, exp_hi),
        "exp_min": float(np.min(live)) if live.size else None,
        "exp_max": float(np.max(live)) if live.size else None,
        "exp_oob_frac": _oob_frac(live, exp_lo, exp_hi),
        "invsqrt_rounds": [],
    }
    y = eval_chebyshev_series_direct(
        mapped,
        np.asarray(exp_entry["coefficients"], dtype=np.float64),
        exp_lo,
        exp_hi,
    ).astype(np.float64)
    y = np.where(valid[None, :, :], y, 0.0)
    if prefill_page_size is not None:
        page_size = int(prefill_page_size)
        if page_size <= 0 or n_query != n_key:
            raise ValueError(
                "paged Prefill requires a positive page size and square scores."
            )
        if query_valid_mask is None:
            query_valid = np.ones(n_query, dtype=bool)
        else:
            query_valid = np.asarray(query_valid_mask, dtype=bool).reshape(-1)
            if query_valid.shape != (n_query,):
                raise ValueError(
                    f"query_valid_mask must have shape ({n_query},), got {query_valid.shape}."
                )
        page_counts = np.arange(n_query) // page_size + 1
        amplitude = np.where(query_valid, 1.0 / np.sqrt(page_counts), 1.0)
        y *= amplitude[None, :, None]
        report["prefill_max_page_count"] = int(page_counts[-1])
    rough = config.invsqrt_alg2["j1_rough"]
    precise = config.invsqrt_alg2["j2_precise"]
    all_inv_in_range = True
    # S-chain auxiliary track: Lambda_1 = 1 per (head, query) row.
    big_lambda = np.ones(y.shape[:-1] + (1,), dtype=np.float64)
    for round_index in range(1, k + 1):
        poly = config.poly_for_layer(
            rough if round_index == 1 else precise, layer_idx
        )
        cheb = poly.get("chebyshev_fhe") or poly
        lo, hi = (float(value) for value in cheb["fit_interval"])
        coeffs = np.asarray(cheb["coefficients"], dtype=np.float64)
        if variant == "alg2":
            row_sum = np.sum(y * y, axis=-1, keepdims=True)
        else:
            # Main track squares only; the lambda-free row sum S_t is
            # rescaled by Lambda_t^2 on the auxiliary track so that the
            # rsqrt input equals Alg2's ||y^{(t-1)}||^2 exactly.
            y = np.square(y)
            y = np.where(valid[None, :, :], y, 0.0)
            row_sum = np.square(big_lambda) * np.sum(
                y, axis=-1, keepdims=True
            )
        oob = _oob_frac(row_sum, lo, hi)
        in_range = oob is not None and oob == 0.0
        all_inv_in_range = all_inv_in_range and in_range
        report["invsqrt_rounds"].append(
            {
                "round": round_index,
                "interval": (lo, hi),
                "min": float(np.min(row_sum)),
                "max": float(np.max(row_sum)),
                "oob_frac": oob,
            }
        )
        lam = eval_chebyshev_series_direct(row_sum, coeffs, lo, hi).astype(
            np.float64
        )
        if variant == "alg2":
            y = np.square(y * lam)
            y = np.where(valid[None, :, :], y, 0.0)
        else:
            big_lambda = np.square(big_lambda * lam)
    if variant == "schain":
        y = y * big_lambda
        y = np.where(valid[None, :, :], y, 0.0)
    report["exp_in_range"] = bool(
        report["exp_oob_frac"] is not None and report["exp_oob_frac"] == 0.0
    )
    report["invsqrt_in_range"] = bool(all_inv_in_range)
    return np.asarray(y, dtype=np.float32), report


def softmax_schain_poly_scores_numpy(
    scores: np.ndarray,
    *,
    config,
    layer_idx: int,
    head_dim: int,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """S-chain (Algorithm 4′) form of :func:`softmax_alg2_poly_scores_numpy`."""

    return softmax_alg2_poly_scores_numpy(
        scores,
        config=config,
        layer_idx=layer_idx,
        head_dim=head_dim,
        valid=valid,
        variant="schain",
    )


def _project_qkv(hidden: np.ndarray, weights):
    attn_in = np.asarray(hidden, dtype=np.float32)
    query = attn_in @ np.asarray(weights.q_proj, dtype=np.float32).T
    key = attn_in @ np.asarray(weights.k_proj, dtype=np.float32).T
    value = attn_in @ np.asarray(weights.v_proj, dtype=np.float32).T
    return query, key, value


def _rope_qkv(
    query: np.ndarray,
    key: np.ndarray,
    *,
    layout: AttentionPairLayout,
    rope_theta: float,
    start_pos: int,
):
    cos, sin = rope_cos_sin(
        seq_len=int(query.shape[0]),
        head_dim=int(layout.head_dim),
        theta=float(rope_theta),
        start_pos=int(start_pos),
    )
    query = apply_rope_heads(
        query,
        cos,
        sin,
        num_heads=int(layout.query_heads),
        head_dim=int(layout.head_dim),
    )
    key = apply_rope_heads(
        key,
        cos,
        sin,
        num_heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
    )
    return query, key


def _attention_out_from_probs(
    probabilities: np.ndarray,
    value: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> np.ndarray:
    value = _as_heads(
        value,
        heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
        name="value",
    )
    ratio = layout.gqa_ratio
    value_for_q = value[:, np.arange(int(layout.query_heads)) // ratio, :]
    output = np.einsum("hqk,khd->qhd", probabilities, value_for_q, optimize=True)
    seq_len = int(probabilities.shape[1])
    return output.reshape(seq_len, -1).astype(np.float32)


def _mlp_poly(residual: np.ndarray, weights, approximations) -> tuple[np.ndarray, dict[str, object]]:
    rmsn2_s_max, rmsn2_oob_frac, rmsn2_fit = _variance_domain(
        residual, approximations.post_attention_norm
    )
    mlp_in = _rmsnorm_poly(
        residual,
        approximations.post_attention_norm,
        weights.post_attention_layernorm,
    )
    gate = mlp_in @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = mlp_in @ np.asarray(weights.up_proj, dtype=np.float32).T
    silu_entry = approximations.silu
    _, lo, hi = _poly_entry_interval(silu_entry)
    finite_gate = np.isfinite(gate)
    if finite_gate.any():
        silu_oob_frac = float(np.mean((gate < lo) | (gate > hi)))
        silu_gate_max_abs = float(np.max(np.abs(gate[finite_gate])))
    else:
        silu_oob_frac = None
        silu_gate_max_abs = None
    activated = _silu_poly(gate, silu_entry) * up
    hidden = residual + activated @ np.asarray(weights.down_proj, dtype=np.float32).T
    return np.asarray(hidden, dtype=np.float32), {
        "silu_oob_frac": silu_oob_frac,
        "silu_gate_max_abs": silu_gate_max_abs,
        "silu_fit": (lo, hi),
        "rmsn2_s_max": rmsn2_s_max,
        "rmsn2_oob_frac": rmsn2_oob_frac,
        "rmsn2_fit": rmsn2_fit,
    }


@dataclass(frozen=True)
class PolyKVLayerResult:
    hidden: np.ndarray
    key: np.ndarray
    value: np.ndarray
    rmsn1_oob_frac: float | None
    rmsn2_oob_frac: float | None
    silu_gate_max_abs: float | None
    silu_oob_frac: float | None
    silu_fit_lo: float
    silu_fit_hi: float
    softmax_exp_in_range: bool | None
    softmax_invsqrt_in_range: bool | None
    softmax_report: dict[str, object] | None


def poly_prefill_layer_numpy(
    hidden: np.ndarray,
    *,
    weights,
    approximations,
    attention_layout: AttentionPairLayout,
    layer_idx: int = 0,
    rope_theta: float = 500000.0,
    use_alg2_softmax: bool = True,
    prefix_hidden: np.ndarray | None = None,
    prefix_bundle=None,
    token_valid_mask: np.ndarray | None = None,
) -> PolyKVLayerResult:
    """Prefill poly layer.  Returns RoPE'd K/V for a Decode cache.

    Optional prefix stays off the content residual.  Content RoPE uses
    ``start_pos=k``.  Softmax reuses the layer's existing Alg2 coefficients.
    """

    residual = np.asarray(hidden, dtype=np.float32)
    rmsn1_s_max, rmsn1_oob_frac, _rmsn1_fit = _variance_domain(
        residual, approximations.input_norm
    )
    del rmsn1_s_max, _rmsn1_fit
    attn_in = _rmsnorm_poly(
        residual, approximations.input_norm, weights.input_layernorm
    )
    query, key, value = _project_qkv(attn_in, weights)
    from .prefix import (
        PrefixKVBundle,
        dual_softmax_alg2_numpy,
        dual_softmax_numpy,
        pv_content_numpy,
        pv_prefix_numpy,
        qk_content_numpy,
        qk_prefix_numpy,
    )

    prefix_k = 0
    key_p = value_p = None
    if prefix_bundle is not None and prefix_hidden is not None:
        raise ValueError("pass prefix_hidden or prefix_bundle, not both.")
    if prefix_bundle is not None:
        bundle = prefix_bundle
        if not isinstance(bundle, PrefixKVBundle):
            bundle = PrefixKVBundle.from_logical(
                bundle.key, bundle.value, layout=attention_layout
            )
        prefix_k = int(bundle.prefix_k)
        key_p = bundle.key
        value_p = bundle.value
    elif prefix_hidden is not None:
        prefix_residual = np.asarray(prefix_hidden, dtype=np.float32)
        prefix_k = int(prefix_residual.shape[0])
        prefix_in = _rmsnorm_poly(
            prefix_residual,
            approximations.input_norm,
            weights.input_layernorm,
        )
        _q_p, key_p, value_p = _project_qkv(prefix_in, weights)
        del _q_p
        cos_p, sin_p = rope_cos_sin(
            seq_len=prefix_k,
            head_dim=int(attention_layout.head_dim),
            theta=float(rope_theta),
            start_pos=0,
        )
        key_p = apply_rope_heads(
            key_p,
            cos_p,
            sin_p,
            num_heads=int(attention_layout.key_value_heads),
            head_dim=int(attention_layout.head_dim),
        )

    query, key = _rope_qkv(
        query,
        key,
        layout=attention_layout,
        rope_theta=rope_theta,
        start_pos=content_rope_start_pos(
            prefix_k=prefix_k,
            token_valid_mask=token_valid_mask,
            seq_len=int(residual.shape[0]),
        ),
    )
    seq_len = int(residual.shape[0])
    softmax_report = None
    if prefix_k:
        scores_p = qk_prefix_numpy(
            query, key_p, layout=attention_layout, scale=False
        )
        scores_c = qk_content_numpy(
            query, key, layout=attention_layout, scale=False
        )
        if use_alg2_softmax:
            probabilities_p, probabilities_c, softmax_report = dual_softmax_alg2_numpy(
                scores_p,
                scores_c,
                layout=attention_layout,
                config=approximations.softmax,
                layer_idx=int(layer_idx),
                prefix_k=prefix_k,
                token_valid_mask=token_valid_mask,
            )
        else:
            scale = np.float32(math.sqrt(float(attention_layout.head_dim)))
            probabilities_p, probabilities_c = dual_softmax_numpy(
                scores_p / scale,
                scores_c / scale,
                prefix_k=prefix_k,
            )
        attn_out = pv_prefix_numpy(
            probabilities_p, value_p, layout=attention_layout
        ) + pv_content_numpy(probabilities_c, value, layout=attention_layout)
    else:
        scores, _, _ = _gqa_unscaled_scores(query, key, layout=attention_layout)
        if use_alg2_softmax:
            valid = _content_causal_valid_mask(
                seq_len, token_valid_mask=token_valid_mask
            )
            page_kwargs = (
                {
                    "prefill_page_size": int(attention_layout.seq_len),
                    "query_valid_mask": token_valid_mask,
                }
                if seq_len > int(attention_layout.seq_len)
                else {}
            )
            probabilities, softmax_report = softmax_alg2_poly_scores_numpy(
                scores,
                config=approximations.softmax,
                layer_idx=int(layer_idx),
                head_dim=int(attention_layout.head_dim),
                valid=valid,
                **page_kwargs,
            )
        else:
            scale = np.float32(math.sqrt(float(attention_layout.head_dim)))
            scaled = scores / scale
            valid = _content_causal_valid_mask(
                seq_len, token_valid_mask=token_valid_mask
            )
            masked = np.where(valid[None, :, :], scaled, -np.inf)
            row_max = np.max(masked, axis=-1, keepdims=True)
            exponentials = np.where(valid[None, :, :], np.exp(masked - row_max), 0.0)
            probabilities = exponentials / np.sum(
                exponentials, axis=-1, keepdims=True
            )
        attn_out = _attention_out_from_probs(
            probabilities, value, layout=attention_layout
        )
    residual = residual + attn_out @ np.asarray(weights.o_proj, dtype=np.float32).T
    hidden_out, mlp = _mlp_poly(residual, weights, approximations)
    return PolyKVLayerResult(
        hidden=hidden_out,
        key=np.asarray(key, dtype=np.float32),
        value=np.asarray(value, dtype=np.float32),
        rmsn1_oob_frac=rmsn1_oob_frac,
        rmsn2_oob_frac=mlp["rmsn2_oob_frac"],
        silu_gate_max_abs=mlp["silu_gate_max_abs"],
        silu_oob_frac=mlp["silu_oob_frac"],
        silu_fit_lo=float(mlp["silu_fit"][0]),
        silu_fit_hi=float(mlp["silu_fit"][1]),
        softmax_exp_in_range=(
            None if softmax_report is None else bool(softmax_report["exp_in_range"])
        ),
        softmax_invsqrt_in_range=(
            None
            if softmax_report is None
            else bool(softmax_report["invsqrt_in_range"])
        ),
        softmax_report=softmax_report,
    )


def poly_decode_layer_numpy(
    hidden: np.ndarray,
    *,
    cache_key: np.ndarray,
    cache_value: np.ndarray,
    weights,
    approximations,
    attention_layout: AttentionPairLayout,
    layer_idx: int = 0,
    position: int,
    rope_theta: float = 500000.0,
    use_alg2_softmax: bool = True,
    prefix_bundle=None,
) -> PolyKVLayerResult:
    """One Decode token with optional plaintext Prefix KV.

    ``cache_key``/``cache_value`` contain only the content history.  When a
    ``prefix_bundle`` is supplied, its already-RoPE'd K and plain V are
    prepended logically for QK/PV, but are never appended to the returned
    content cache.  This mirrors the FHE decode path where Prefix is a PT
    page and content remains CT.
    """

    residual = np.asarray(hidden, dtype=np.float32)
    if residual.ndim != 2 or int(residual.shape[0]) != 1:
        raise ValueError(f"Decode hidden must be [1, H], got {residual.shape}.")
    _, rmsn1_oob_frac, _ = _variance_domain(residual, approximations.input_norm)
    attn_in = _rmsnorm_poly(
        residual, approximations.input_norm, weights.input_layernorm
    )
    query, key, value = _project_qkv(attn_in, weights)
    query, key = _rope_qkv(
        query,
        key,
        layout=attention_layout,
        rope_theta=rope_theta,
        start_pos=int(position),
    )
    cache_key = np.asarray(cache_key, dtype=np.float32)
    cache_value = np.asarray(cache_value, dtype=np.float32)
    keys = np.concatenate([cache_key, key], axis=0)
    values = np.concatenate([cache_value, value], axis=0)
    softmax_report = None
    if prefix_bundle is not None:
        from .prefix import PrefixKVBundle

        bundle = prefix_bundle
        if not isinstance(bundle, PrefixKVBundle):
            bundle = PrefixKVBundle.from_logical(
                bundle.key, bundle.value, layout=attention_layout
            )
        prefix_k = int(bundle.prefix_k)
        if prefix_k <= 0:
            raise ValueError("prefix_bundle must contain at least one token.")
        key_prefix = np.asarray(bundle.key, dtype=np.float32)
        value_prefix = np.asarray(bundle.value, dtype=np.float32)
        q_heads = _as_heads(
            query,
            heads=int(attention_layout.query_heads),
            head_dim=int(attention_layout.head_dim),
            name="query",
        )
        p_heads = _as_heads(
            key_prefix,
            heads=int(attention_layout.key_value_heads),
            head_dim=int(attention_layout.head_dim),
            name="prefix key",
        )
        query_to_kv = np.arange(int(attention_layout.query_heads)) // int(
            attention_layout.gqa_ratio
        )
        p_for_q = p_heads[:, query_to_kv, :]
        scores_p = np.einsum("qhd,khd->hqk", q_heads, p_for_q, optimize=True)
        scores_c, _, _ = _gqa_unscaled_scores(query, keys, layout=attention_layout)
        scores = np.concatenate([scores_p, scores_c], axis=-1)
        # A single decode query is at the end of the content history, so all
        # cached content columns (and the current token) are valid.  The
        # prefill dual-softmax helper intentionally requires a square causal
        # content matrix and therefore is not suitable for this [1, K] case.
        valid = np.ones((1, int(scores.shape[-1])), dtype=bool)
        if use_alg2_softmax:
            probabilities, softmax_report = softmax_alg2_poly_scores_numpy(
                scores,
                config=approximations.softmax,
                layer_idx=int(layer_idx),
                head_dim=int(attention_layout.head_dim),
                valid=valid,
            )
        else:
            scaled = scores / np.float32(math.sqrt(float(attention_layout.head_dim)))
            shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        probabilities_p = probabilities[..., :prefix_k]
        probabilities_c = probabilities[..., prefix_k:]
        probabilities = np.concatenate([probabilities_p, probabilities_c], axis=-1)
        values = np.concatenate([value_prefix, values], axis=0)
    else:
        scores, _, _ = _gqa_unscaled_scores(query, keys, layout=attention_layout)
        if use_alg2_softmax:
            probabilities, softmax_report = softmax_alg2_poly_scores_numpy(
                scores,
                config=approximations.softmax,
                layer_idx=int(layer_idx),
                head_dim=int(attention_layout.head_dim),
            )
        else:
            scale = np.float32(math.sqrt(float(attention_layout.head_dim)))
            scaled = scores / scale
            row_max = np.max(scaled, axis=-1, keepdims=True)
            exponentials = np.exp(scaled - row_max)
            probabilities = exponentials / np.sum(
                exponentials, axis=-1, keepdims=True
            )
    attn_out = _attention_out_from_probs(
        probabilities, values, layout=attention_layout
    )
    residual = residual + attn_out @ np.asarray(weights.o_proj, dtype=np.float32).T
    hidden_out, mlp = _mlp_poly(residual, weights, approximations)
    return PolyKVLayerResult(
        hidden=hidden_out,
        key=np.asarray(key, dtype=np.float32),
        value=np.asarray(value, dtype=np.float32),
        rmsn1_oob_frac=rmsn1_oob_frac,
        rmsn2_oob_frac=mlp["rmsn2_oob_frac"],
        silu_gate_max_abs=mlp["silu_gate_max_abs"],
        silu_oob_frac=mlp["silu_oob_frac"],
        silu_fit_lo=float(mlp["silu_fit"][0]),
        silu_fit_hi=float(mlp["silu_fit"][1]),
        softmax_exp_in_range=(
            None if softmax_report is None else bool(softmax_report["exp_in_range"])
        ),
        softmax_invsqrt_in_range=(
            None
            if softmax_report is None
            else bool(softmax_report["invsqrt_in_range"])
        ),
        softmax_report=softmax_report,
    )


def poly_prefill_stack_numpy(
    hidden: np.ndarray,
    layer_weights,
    *,
    approx_config,
    attention_layout: AttentionPairLayout,
    start_layer: int = 0,
    rope_theta: float = 500000.0,
    use_alg2_softmax: bool = True,
    prefix_bundles=None,
    token_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run every Prefill layer.  Returns ``(hidden, keys, values)``.

    ``keys`` / ``values`` are ``[layers, seq, kv_width]`` RoPE'd C-form sources.
    """

    hidden = np.asarray(hidden, dtype=np.float32)
    start_layer = int(start_layer)
    if start_layer < 0:
        raise ValueError("start_layer must be non-negative.")
    if prefix_bundles is not None and len(prefix_bundles) != len(layer_weights):
        raise ValueError(
            "prefix_bundles must match layer_weights length: "
            f"{len(prefix_bundles)} vs {len(layer_weights)}."
        )
    keys = []
    values = []
    for local_layer_idx, weights in enumerate(layer_weights):
        layer_idx = start_layer + int(local_layer_idx)
        prefix_bundle = (
            None
            if prefix_bundles is None
            else prefix_bundles[int(local_layer_idx)]
        )
        result = poly_prefill_layer_numpy(
            hidden,
            weights=weights,
            approximations=approx_config.for_layer(int(layer_idx)),
            attention_layout=attention_layout,
            layer_idx=int(layer_idx),
            rope_theta=float(rope_theta),
            use_alg2_softmax=bool(use_alg2_softmax),
            prefix_bundle=prefix_bundle,
            token_valid_mask=token_valid_mask,
        )
        hidden = result.hidden
        keys.append(result.key)
        values.append(result.value)
    return hidden, np.stack(keys, axis=0), np.stack(values, axis=0)


__all__ = [
    "PolyKVLayerResult",
    "PolyLayerResult",
    "poly_decode_layer_numpy",
    "poly_prefill_layer_numpy",
    "poly_prefill_stack_numpy",
    "poly_transformer_layer_numpy",
    "softmax_alg2_poly_scores_numpy",
]
