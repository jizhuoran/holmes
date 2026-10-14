from __future__ import annotations

"""Decode Alg2 Softmax on compact Decode scores.  Not Delta ``softmax_alg2_fhe_delta``.

Public map, Chebyshev exp / ``1/√s``, and ``y←(y λ)^2`` reuse Prefill
coefficients.  Row reduction is the C token-axis fold (stride 2); ``λ`` is
isolated on column 0 and broadcast back to the occupied prefix.
"""

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from llama3fhe.approx.chebyshev import (
    chebyshev_ps_mul_depth,
    eval_chebyshev_series_direct,
)
from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout

from ...nonlinear.bootstrap import bootstrap_tag
from ...nonlinear.elementwise import eval_exp_cipher, eval_invsqrt_cipher
from ...primitives import mul_cipher_rescale, square_rescale
from .decode import (
    _add_packed_rows,
    _add_scalar_ciphers,
    _homo_rotate,
    _mul_packed_rows,
    _rotate_add_tree,
    _token_broadcast_cipher,
    _token_reduce_cipher,
    decode_broadcast_tokens_fhe,
    decode_reduce_tokens_fhe,
    softmax_alg2_row_numpy,
)


def _scores_rows(
    scores: np.ndarray, *, layout: AttentionPairLayout
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    packed_shape = (int(layout.query_pair_count), int(layout.slots))
    compact_shape = (1, int(layout.slots))
    if scores.ndim == 2 and scores.shape == compact_shape:
        scores = layout.unpack_compact_decode_scores(scores)
    elif scores.ndim == 2 and scores.shape == packed_shape:
        scores = layout.unpack_decode_scores(scores)
    if scores.ndim == 3:
        if scores.shape[1] != 1:
            raise ValueError("decode Alg2 expects one query row or [H,S].")
        scores = scores[:, 0, :]
    expected = (int(layout.query_heads), int(layout.seq_len))
    if scores.shape != expected:
        raise ValueError(f"decode Alg2 scores {scores.shape} vs {expected}.")
    return scores


def _prefix_valid(layout: AttentionPairLayout, occupied: int | None) -> np.ndarray:
    seq = int(layout.seq_len)
    occupied = seq if occupied is None else int(occupied)
    if occupied < 0 or occupied > seq:
        raise ValueError(f"occupied={occupied} outside [0, {seq}].")
    valid = np.zeros((seq,), dtype=bool)
    valid[:occupied] = True
    return valid


def _head_rows_tuple(
    head_rows, *, layout: AttentionPairLayout
) -> tuple[int, ...] | None:
    """Validated, sorted, de-duplicated query-head rows (``None`` = all heads)."""

    if head_rows is None:
        return None
    heads = int(layout.query_heads)
    rows = tuple(sorted({int(row) for row in head_rows}))
    if not rows:
        raise ValueError("head_rows must name at least one query head.")
    for row in rows:
        if not 0 <= row < heads:
            raise IndexError(f"head_rows entry {row} outside [0, {heads}).")
    return rows


def _head_rows_valid(
    layout: AttentionPairLayout, head_rows
) -> np.ndarray | None:
    """``[query_heads]`` bool: which head rows carry real scores."""

    rows = _head_rows_tuple(head_rows, layout=layout)
    if rows is None:
        return None
    owned = np.zeros((int(layout.query_heads),), dtype=bool)
    owned[list(rows)] = True
    return owned


def _head_rows_mask(
    layout: AttentionPairLayout, head_rows
) -> np.ndarray | None:
    """Packed compact-Decode mask: 1 on ``head_rows``; ``None`` = no masking.

    Multi-card decode gives a rank only some ``(group, query_offset)``
    streams, so the other head rows of its compact score pages are exactly
    zero.  A zero row is *not* a valid softmax row: its uniform ``exp`` makes
    ``sum(y^2)`` leave the per-layer ``1/sqrt`` fit interval (layer 2 / 23
    blow up to 1e100+), and the resulting garbage slots poison every slot of
    the ciphertext through bootstrap and the slot-mixing linear transforms.
    With ``head_rows`` those rows are treated like invalid prefix columns
    (public map -> ``stats_min``, masked to zero after ``exp``) and their
    ``1/sqrt`` input is replaced by the safe fill constant.
    """

    rows = _head_rows_tuple(head_rows, layout=layout)
    if rows is None:
        return None
    mask = np.zeros(
        (int(layout.query_heads), int(layout.seq_len)), dtype=np.float32
    )
    mask[list(rows), :] = 1.0
    return layout.pack_compact_decode_scores(mask, dtype=np.float64)


def _with_head_rows(mask: np.ndarray, head_mask: np.ndarray | None) -> np.ndarray:
    """Restrict a packed compact-Decode mask to the owned head rows."""

    if head_mask is None:
        return mask
    return np.asarray(mask, dtype=np.float64) * np.asarray(head_mask, dtype=np.float64)


def apply_decode_alg2_public_map_numpy(
    scores: np.ndarray,
    *,
    layout: AttentionPairLayout,
    config,
    layer_idx: int,
    occupied: int | None = None,
    score_scale: float | None = None,
    head_rows=None,
) -> np.ndarray:
    """Valid slots: ``(s/√d - U_hat)/2^k``; invalid: ``(stats_min - U_hat)/2^k``.

    ``head_rows`` (multi-card decode) marks every slot of the other head rows
    invalid as well.
    """

    scores = _scores_rows(scores, layout=layout)
    valid = _prefix_valid(layout, occupied)
    owned = _head_rows_valid(layout, head_rows)
    k = int(config.k_for(layer_idx))
    score_scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    invalid = float(config.stats_min_for(layer_idx)) / float(2**k)
    mapped = np.full(scores.shape, shift + invalid, dtype=np.float64)
    mapped[:, valid] = scores[:, valid] * (score_scale / float(2**k)) + shift
    if owned is not None:
        mapped[~owned, :] = shift + invalid
    return mapped.astype(np.float32)


def apply_decode_alg2_public_map_fhe(
    score_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int,
    occupied: int | None = None,
    score_scale: float | None = None,
    head_rows=None,
) -> tuple[object, ...]:
    """Prefill Alg2 public map on compact Decode score packing.  Does not consume inputs.

    ``head_rows`` (multi-card decode): the other head rows are mapped like
    invalid columns (see :func:`_head_rows_mask`).
    """

    score_ciphers = tuple(score_ciphers)
    if len(score_ciphers) != 1:
        raise ValueError(
            f"Decode Alg2 expects 1 compact score cipher, got {len(score_ciphers)}."
        )
    layout._require_compact_q()
    k = int(config.k_for(layer_idx))
    score_scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    valid = _with_head_rows(
        layout.compact_decode_prefix_mask(occupied),
        _head_rows_mask(layout, head_rows),
    )
    multiplier = np.asarray(valid, dtype=np.float64) * (
        score_scale / float(2**k)
    )
    shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    invalid = (1.0 - np.asarray(valid, dtype=np.float64)) * (
        float(config.stats_min_for(layer_idx)) / float(2**k)
    )
    scaled = shifted = None
    try:
        scaled = _mul_packed_rows(
            score_ciphers,
            multiplier,
            name="decode.alg2.valid_scale",
            crypto_context=crypto_context,
        )
        shifted = _add_scalar_ciphers(
            scaled, shift, crypto_context=crypto_context
        )
        return _add_packed_rows(
            shifted,
            invalid,
            name="decode.alg2.invalid",
            crypto_context=crypto_context,
        )
    finally:
        for cipher in tuple(scaled or ()) + tuple(shifted or ()):
            release_if_supported(cipher)


def _invsqrt_cheb(poly_entry: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    cheb = poly_entry.get("chebyshev_fhe") or poly_entry
    lo, hi = (float(value) for value in cheb["fit_interval"])
    coeffs = np.asarray(cheb["coefficients"], dtype=np.float64)
    return coeffs, lo, hi


def softmax_alg2_poly_numpy(
    scores: np.ndarray,
    *,
    layout: AttentionPairLayout,
    config,
    layer_idx: int,
    occupied: int | None = None,
    score_scale: float | None = None,
    rounds: int | None = None,
    head_rows=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Chebyshev Alg2 on a Decode score row. Returns ``(P, interval report)``.

    ``head_rows`` mirrors the FHE kernel's multi-card handling: the other head
    rows are invalid everywhere (probabilities exactly zero) and their
    ``1/sqrt`` input is the safe fill constant instead of the zero row sum.
    """

    scores = _scores_rows(scores, layout=layout)
    valid = _prefix_valid(layout, occupied)
    owned = _head_rows_valid(layout, head_rows)
    mapped = apply_decode_alg2_public_map_numpy(
        scores,
        layout=layout,
        config=config,
        layer_idx=layer_idx,
        occupied=occupied,
        score_scale=score_scale,
        head_rows=head_rows,
    ).astype(np.float64)
    exp_entry = config.layer_exp[int(layer_idx)]
    exp_lo, exp_hi = (float(value) for value in exp_entry["fit_interval"])
    mapped_valid = mapped[:, valid]
    if owned is not None:
        mapped_valid = mapped_valid[owned, :]
    report: dict[str, Any] = {
        "k": int(config.k_for(layer_idx)),
        "occupied": int(np.count_nonzero(valid)),
        "exp_interval": (exp_lo, exp_hi),
        "exp_min": float(np.min(mapped_valid)) if mapped_valid.size else None,
        "exp_max": float(np.max(mapped_valid)) if mapped_valid.size else None,
        "exp_in_range": bool(
            mapped_valid.size
            and float(np.min(mapped_valid)) >= exp_lo
            and float(np.max(mapped_valid)) <= exp_hi
        ),
        "invsqrt_rounds": [],
    }
    y = eval_chebyshev_series_direct(
        mapped,
        np.asarray(exp_entry["coefficients"], dtype=np.float64),
        exp_lo,
        exp_hi,
    ).astype(np.float64)
    y = np.where(valid[None, :], y, 0.0)
    if owned is not None:
        y = np.where(owned[:, None], y, 0.0)
    k = int(config.k_for(layer_idx) if rounds is None else rounds)
    rough = config.invsqrt_alg2["j1_rough"]
    precise = config.invsqrt_alg2["j2_precise"]
    for round_index in range(1, k + 1):
        poly = config.poly_for_layer(
            rough if round_index == 1 else precise, int(layer_idx)
        )
        coeffs, lo, hi = _invsqrt_cheb(poly)
        row_sum = np.sum(y * y, axis=-1, keepdims=True)
        if owned is not None:
            row_sum = np.where(
                owned[:, None], row_sum, _safe_invsqrt_fill(poly)
            )
        row_vals = row_sum.reshape(-1)
        if owned is not None:
            row_vals = row_vals[owned]
        round_report = {
            "round": round_index,
            "interval": (lo, hi),
            "min": float(np.min(row_vals)),
            "max": float(np.max(row_vals)),
            "in_range": bool(
                float(np.min(row_vals)) >= lo and float(np.max(row_vals)) <= hi
            ),
        }
        report["invsqrt_rounds"].append(round_report)
        lam = eval_chebyshev_series_direct(row_sum, coeffs, lo, hi).astype(
            np.float64
        )
        y = np.square(y * lam)
        y = np.where(valid[None, :], y, 0.0)
        if owned is not None:
            y = np.where(owned[:, None], y, 0.0)
    probabilities = np.asarray(y, dtype=np.float32)
    exact = softmax_alg2_row_numpy(
        scores,
        layout=layout,
        occupied=int(np.count_nonzero(valid)),
        k=k,
        scale=True,
    )
    difference = np.abs(probabilities - exact)
    if owned is not None:
        difference = difference[owned, :]
    report["vs_exact_max_abs"] = float(np.max(difference))
    return probabilities, report


def _align_cipher_pair(left, right, *, crypto_context):
    left_limbs = int(left.state.cur_limbs)
    right_limbs = int(right.state.cur_limbs)
    left_degree = int(left.state.scale_degree)
    right_degree = int(right.state.scale_degree)
    left_scale = left.state.scaling_factor
    right_scale = right.state.scaling_factor
    same_scale = (
        left_scale is None
        and right_scale is None
    ) or (
        left_scale is not None
        and right_scale is not None
        and math.isclose(
            float(left_scale), float(right_scale), rel_tol=1e-12, abs_tol=0.0
        )
    )
    if (
        left_limbs == right_limbs
        and left_degree == right_degree
        and same_scale
    ):
        return left, right, False, False
    if left_limbs == right_limbs:
        # With equal modulus depth choose the larger scale degree as the
        # common target: aligning a degree-1 ciphertext *up* to degree 2 is
        # depth-safe, while the reverse direction would ask align_to to
        # create a state with more consumed depth than its source has.
        if left_degree > right_degree:
            return (
                left,
                crypto_context.fhe.align_to(
                    right, left.state, crypto_context.context
                ),
                False,
                True,
            )
        return (
            crypto_context.fhe.align_to(
                left, right.state, crypto_context.context
            ),
            right,
            True,
            False,
        )
    if left_limbs > right_limbs:
        return (
            crypto_context.fhe.align_to(left, right.state, crypto_context.context),
            right,
            True,
            False,
        )
    return (
        left,
        crypto_context.fhe.align_to(right, left.state, crypto_context.context),
        False,
        True,
    )


def _safe_invsqrt_fill(poly_entry: dict[str, Any]) -> float:
    _, lo, hi = _invsqrt_cheb(poly_entry)
    return float(min(max(0.25, lo), hi))


def _release_tuple(ciphers) -> None:
    for cipher in ciphers or ():
        release_if_supported(cipher)


class _BroadcastLayerExp:
    """One synthetic exp poly, indexed like the 32-layer production pack."""

    def __init__(self, entry: dict[str, Any]):
        self._entry = dict(entry)

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        del layer_idx
        return dict(self._entry)


@dataclass(frozen=True)
class DecodeAlg2Poly:
    """Minimal Alg2 artifact for Decode FHE smokes (not the 32-layer pack)."""

    k: int
    U_hat: float
    stats_min: float
    exp_entry: dict[str, Any]
    invsqrt_entry: dict[str, Any]

    def k_for(self, layer_idx: int) -> int:
        del layer_idx
        return int(self.k)

    def U_hat_for(self, layer_idx: int) -> float:
        del layer_idx
        return float(self.U_hat)

    def stats_min_for(self, layer_idx: int) -> float:
        del layer_idx
        return float(self.stats_min)

    @property
    def layer_exp(self) -> _BroadcastLayerExp:
        return _BroadcastLayerExp(self.exp_entry)

    @property
    def invsqrt_alg2(self) -> dict[str, dict[str, Any]]:
        entry = dict(self.invsqrt_entry)
        return {"j1_rough": entry, "j2_precise": entry}

    def poly_for_layer(self, poly_entry: dict[str, Any], layer_idx: int):
        del layer_idx
        return poly_entry

    @classmethod
    def synthetic(
        cls,
        *,
        k: int = 1,
        degree: int = 8,
        U_hat: float = 1.0,
        stats_min: float = -4.0,
        exp_interval: tuple[float, float] = (-2.0, 1.0),
        invsqrt_interval: tuple[float, float] = (0.05, 8.0),
    ) -> "DecodeAlg2Poly":
        """Low-degree exp / invsqrt for Decode FHE smokes (not production).

        Default degree 8 uses a well-tested Paterson–Stockmeyer split.
        Degree 2 fails FHE eval with a shallower-to-deeper align error.
        """

        from llama3fhe.approx.chebyshev import eval_chebyshev_coefficients

        exp_lo, exp_hi = (float(value) for value in exp_interval)
        inv_lo, inv_hi = (float(value) for value in invsqrt_interval)
        exp_coeffs = eval_chebyshev_coefficients(math.exp, exp_lo, exp_hi, int(degree))
        inv_coeffs = eval_chebyshev_coefficients(
            lambda value: 1.0 / math.sqrt(max(float(value), inv_lo)),
            inv_lo,
            inv_hi,
            int(degree),
        )
        return cls(
            k=int(k),
            U_hat=float(U_hat),
            stats_min=float(stats_min),
            exp_entry={
                "fit_interval": [exp_lo, exp_hi],
                "coefficients": np.asarray(exp_coeffs, dtype=np.float64),
            },
            invsqrt_entry={
                "fit_interval": [inv_lo, inv_hi],
                "coefficients": np.asarray(inv_coeffs, dtype=np.float64),
                "chebyshev_fhe": {
                    "fit_interval": [inv_lo, inv_hi],
                    "coefficients": np.asarray(inv_coeffs, dtype=np.float64),
                },
            },
        )


def _decode_exp_need_limbs(exp_entry: dict[str, Any]) -> int:
    """Limbs the public-mapped scores must exceed before Chebyshev exp."""

    return int(chebyshev_ps_mul_depth(exp_entry["coefficients"])) + 3


def _invsqrt_need_limbs(poly_entry: dict[str, Any]) -> int:
    """Limbs the Alg2 λ input must exceed before the Prefill invsqrt poly."""

    coeffs, _, _ = _invsqrt_cheb(poly_entry)
    return int(chebyshev_ps_mul_depth(coeffs)) + 3


#: Limbs one Alg2 round consumes on a ``y`` that sits at or below the
#: broadcast-λ level: the ``y·λ`` rescale, the ``square`` rescale and the
#: prefix-mask rescale (:func:`_alg2_round_fhe`).  A higher ``y`` is first
#: aligned down to λ, which only makes the round cheaper for this bound.
ALG2_ROUND_Y_LIMBS = 3
#: Limbs a ciphertext must still carry to be bootstrapped (the mod-raise
#: rescales it once first: ``rescale: cur_limbs must be > 1``).
BOOTSTRAP_MIN_INPUT_LIMBS = 2
#: ``y`` is refreshed *before* a round when it has at most this many limbs:
#: the round itself plus the minimum the post-Softmax probability refresh
#: (or a later round's refresh) needs.  With the production Meta-BTS exit
#: the ``k <= 4`` layers never reach this bound and keep their schedule
#: unchanged; ``k = 5`` (layer 31) refreshes ``y`` once before its last round
#: instead of ending the chain at 1 limb, where the probability refresh
#: failed (``rescale: cur_limbs must be > 1, got 1``).
ALG2_ROUND_NEED_LIMBS = ALG2_ROUND_Y_LIMBS + BOOTSTRAP_MIN_INPUT_LIMBS


def _refresh_row_ciphers(
    ciphers,
    *,
    bootstrap_operator,
    iterations: int | None,
    precision_bits: int | None,
    tag: str | None = None,
    periodic_slots: int | None = None,
):
    """Refresh one score-row stream.  Consumes ``ciphers`` when a refresh runs.

    ``tag`` labels the bootstraps for the bench timers (:func:`bootstrap_tag`).
    ``periodic_slots`` promises the message is periodic with that slot period
    and routes the refresh through the operator's matching sparse-slot
    program (:meth:`BootstrapOperator.refresh`).
    """

    ciphers = tuple(ciphers)
    if bootstrap_operator is None:
        return ciphers
    if tag is not None:
        with bootstrap_tag(tag):
            return _refresh_row_ciphers(
                ciphers,
                bootstrap_operator=bootstrap_operator,
                iterations=iterations,
                precision_bits=precision_bits,
                periodic_slots=periodic_slots,
            )
    selected_iterations = (
        int(bootstrap_operator.config.iterations)
        if iterations is None
        else int(iterations)
    )
    selected_precision = (
        int(bootstrap_operator.config.precision_bits)
        if precision_bits is None
        else int(precision_bits)
    )
    sparse_kwargs = {} if periodic_slots is None else {"periodic_slots": int(periodic_slots)}
    refreshed: list[object] = []
    try:
        for cipher in ciphers:
            refreshed.append(
                bootstrap_operator.refresh(
                    cipher,
                    iterations=selected_iterations,
                    precision_bits=selected_precision,
                    **sparse_kwargs,
                )
            )
        result = tuple(refreshed)
        refreshed = []
        _release_tuple(ciphers)
        return result
    finally:
        _release_tuple(refreshed)


def _refresh_row_ciphers_if_short(
    ciphers,
    *,
    bootstrap_operator,
    need_limbs: int,
    iterations: int | None,
    precision_bits: int | None,
    tag: str | None = None,
    periodic_slots: int | None = None,
):
    """Refresh when any ciphertext has ``cur_limbs <= need_limbs``."""

    ciphers = tuple(ciphers)
    if bootstrap_operator is None or not ciphers:
        return ciphers
    if min(int(cipher.state.cur_limbs) for cipher in ciphers) > int(need_limbs):
        return ciphers
    return _refresh_row_ciphers(
        ciphers,
        bootstrap_operator=bootstrap_operator,
        iterations=iterations,
        precision_bits=precision_bits,
        tag=tag,
        periodic_slots=periodic_slots,
    )


def softmax_alg2_row_fhe(
    score_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int = 0,
    occupied: int | None = None,
    score_scale: float | None = None,
    rounds: int | None = None,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    head_rows=None,
) -> tuple[object, ...]:
    """Chebyshev Alg2 on Decode scores.

    Passing ``bootstrap_operator`` refreshes ``y`` after exp and refreshes
    each round's λ the way Prefill does, so Decode can consume the same
    Alg2 coefficients (including production ``k`` and invsqrt degree).
    ``y`` is refreshed again before a round that would run the chain dry
    (:data:`ALG2_ROUND_NEED_LIMBS`; only the ``k = 5`` layer needs it).

    ``head_rows`` (multi-card decode): only these query-head rows carry real
    scores; the others are forced to zero probability (:func:`_head_rows_mask`).
    """

    score_ciphers = tuple(score_ciphers)
    head_mask = _head_rows_mask(layout, head_rows)
    mapped = y_ciphers = None
    try:
        mapped = apply_decode_alg2_public_map_fhe(
            score_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            config=config,
            layer_idx=layer_idx,
            occupied=occupied,
            score_scale=score_scale,
            head_rows=head_rows,
        )
        exp_entry = config.layer_exp[int(layer_idx)]
        mapped = _refresh_row_ciphers_if_short(
            mapped,
            bootstrap_operator=bootstrap_operator,
            need_limbs=_decode_exp_need_limbs(exp_entry),
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.pre_exp",
        )
        exp_outputs = []
        for cipher in mapped:
            exp_outputs.append(
                eval_exp_cipher(cipher, exp_entry, crypto_context=crypto_context)
            )
        y_ciphers = tuple(exp_outputs)
        exp_outputs = []
        mapped = None
        masked = _mul_packed_rows(
            y_ciphers,
            _with_head_rows(layout.compact_decode_prefix_mask(occupied), head_mask),
            name="decode.alg2.post_exp",
            crypto_context=crypto_context,
        )
        _release_tuple(y_ciphers)
        y_ciphers = _refresh_row_ciphers(
            masked,
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.post_exp",
        )
        k = int(config.k_for(layer_idx) if rounds is None else rounds)
        rough = config.invsqrt_alg2["j1_rough"]
        precise = config.invsqrt_alg2["j2_precise"]
        for round_index in range(1, k + 1):
            poly = config.poly_for_layer(
                rough if round_index == 1 else precise, int(layer_idx)
            )
            y_ciphers = _refresh_row_ciphers_if_short(
                y_ciphers,
                bootstrap_operator=bootstrap_operator,
                need_limbs=ALG2_ROUND_NEED_LIMBS,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag=f"softmax.r{round_index}.y",
            )
            nxt = _alg2_round_fhe(
                y_ciphers,
                poly_entry=poly,
                layout=layout,
                crypto_context=crypto_context,
                occupied=occupied,
                round_index=round_index,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_precision_bits=bootstrap_precision_bits,
                head_mask=head_mask,
            )
            _release_tuple(y_ciphers)
            y_ciphers = nxt
        result = y_ciphers
        y_ciphers = None
        return result
    finally:
        _release_tuple(mapped)
        _release_tuple(y_ciphers)


def _alg2_round_fhe(
    y_ciphers: tuple[object, ...],
    *,
    poly_entry: dict[str, Any],
    layout: AttentionPairLayout,
    crypto_context,
    occupied: int | None,
    round_index: int,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    head_mask: np.ndarray | None = None,
) -> tuple[object, ...]:
    squared = reduced = isolated = filled = lam = lam_bcast = scaled = None
    try:
        squared = tuple(
            square_rescale(cipher, crypto_context=crypto_context)
            for cipher in y_ciphers
        )
        reduced = decode_reduce_tokens_fhe(
            squared, layout=layout, crypto_context=crypto_context
        )
        isolated = _mul_packed_rows(
            reduced,
            layout.compact_decode_token0_mask(),
            name=f"decode.alg2.r{round_index}.sum0",
            crypto_context=crypto_context,
        )
        fill = _invsqrt_fill_mask(layout, poly_entry, head_mask=head_mask)
        filled = _add_packed_rows(
            isolated,
            fill,
            name=f"decode.alg2.r{round_index}.invsqrt_fill",
            crypto_context=crypto_context,
        )
        _release_tuple(isolated)
        isolated = None
        filled = _refresh_row_ciphers_if_short(
            filled,
            bootstrap_operator=bootstrap_operator,
            need_limbs=_invsqrt_need_limbs(poly_entry),
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag=f"softmax.r{round_index}.fill",
        )
        lam_raw = []
        for cipher in filled:
            lam_raw.append(
                eval_invsqrt_cipher(
                    cipher, poly_entry, crypto_context=crypto_context
                )
            )
        lam = tuple(lam_raw)
        lam_raw = []
        lam = _refresh_row_ciphers(
            lam,
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag=f"softmax.r{round_index}.lam",
        )
        lam_bcast = decode_broadcast_tokens_fhe(
            lam, layout=layout, crypto_context=crypto_context, occupied=occupied
        )
        products = []
        for y_cipher, lam_cipher in zip(y_ciphers, lam_bcast, strict=True):
            aligned_y, aligned_lam, owned_y, owned_lam = _align_cipher_pair(
                y_cipher, lam_cipher, crypto_context=crypto_context
            )
            try:
                products.append(
                    mul_cipher_rescale(
                        aligned_y, aligned_lam, crypto_context=crypto_context
                    )
                )
            finally:
                if owned_y:
                    release_if_supported(aligned_y)
                if owned_lam:
                    release_if_supported(aligned_lam)
        scaled = tuple(products)
        products = []
        squared_y = tuple(
            square_rescale(cipher, crypto_context=crypto_context)
            for cipher in scaled
        )
        remasked = _mul_packed_rows(
            squared_y,
            _with_head_rows(layout.compact_decode_prefix_mask(occupied), head_mask),
            name=f"decode.alg2.r{round_index}.prefix",
            crypto_context=crypto_context,
        )
        _release_tuple(squared_y)
        return remasked
    finally:
        _release_tuple(squared)
        _release_tuple(reduced)
        _release_tuple(isolated)
        _release_tuple(filled)
        _release_tuple(lam)
        _release_tuple(lam_bcast)
        _release_tuple(scaled)


def _invsqrt_fill_mask(
    layout: AttentionPairLayout,
    poly_entry: dict[str, Any],
    *,
    head_mask: np.ndarray | None,
) -> np.ndarray:
    """Safe ``1/sqrt`` input everywhere except the owned rows' column-0 sums.

    Column 0 of an owned row holds ``sum(y^2)``; every other slot (and, with
    ``head_mask``, column 0 of the rows this rank does not own, whose sum is
    zero) receives the in-interval fill constant so the Chebyshev ``1/sqrt``
    never extrapolates.
    """

    token0 = _with_head_rows(layout.compact_decode_token0_mask(), head_mask)
    return (1.0 - np.asarray(token0, dtype=np.float64)) * (
        _safe_invsqrt_fill(poly_entry)
    )


def _release_pages(pages) -> None:
    for page in pages or ():
        _release_tuple(page)


def _sum_aligned_ciphers(ciphers, *, crypto_context):
    ciphers = tuple(ciphers)
    if not ciphers:
        raise ValueError("cannot sum an empty cipher list.")
    if len(ciphers) == 1:
        return ciphers[0].deep_copy()
    acc = ciphers[0]
    owned = False
    try:
        for cipher in ciphers[1:]:
            left, right, owned_left, owned_right = _align_cipher_pair(
                acc, cipher, crypto_context=crypto_context
            )
            try:
                summed = crypto_context.fhe.homo_add(
                    left, right, crypto_context.context
                )
            finally:
                if owned_left:
                    release_if_supported(left)
                if owned_right:
                    release_if_supported(right)
            if owned:
                release_if_supported(acc)
            acc = summed
            owned = True
        result = acc
        acc = None
        return result
    finally:
        if owned:
            release_if_supported(acc)


def softmax_alg2_row_pages_fhe(
    score_pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    occupied_pages: tuple[int, ...] | list[int],
    layer_idx: int = 0,
    rounds: int | None = None,
    score_scale: float | None = None,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    head_rows=None,
) -> tuple[tuple[object, ...], ...]:
    """Alg2 across C-form score pages that share one 1-query softmax.

    ``head_rows`` (multi-card decode): only these query-head rows carry real
    scores on every page; the others are forced to zero probability
    (:func:`_head_rows_mask`).  ``None`` keeps the single-card kernel.
    """

    score_pages = tuple(tuple(page) for page in score_pages)
    occupied_pages = tuple(int(value) for value in occupied_pages)
    if len(score_pages) != len(occupied_pages):
        raise ValueError("score pages and occupied_pages must have the same length.")
    if not score_pages:
        raise ValueError("softmax pages require at least one score page.")
    if len(score_pages) >= 2 and softmax_merge_pages_enabled() and (
        len(score_pages) <= int(layout.decode_page_merge_groups)
    ):
        return _softmax_alg2_merged_pages_fhe(
            score_pages,
            layout=layout,
            crypto_context=crypto_context,
            config=config,
            occupied_pages=occupied_pages,
            layer_idx=layer_idx,
            rounds=rounds,
            score_scale=score_scale,
            bootstrap_operator=bootstrap_operator,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_precision_bits=bootstrap_precision_bits,
            head_rows=head_rows,
        )
    head_mask = _head_rows_mask(layout, head_rows)
    mapped_pages: list[tuple[object, ...]] = []
    y_pages: list[tuple[object, ...]] = []
    try:
        for page, occupied in zip(score_pages, occupied_pages, strict=True):
            mapped = apply_decode_alg2_public_map_fhe(
                page,
                layout=layout,
                crypto_context=crypto_context,
                config=config,
                layer_idx=layer_idx,
                occupied=occupied,
                score_scale=score_scale,
                head_rows=head_rows,
            )
            exp_entry = config.layer_exp[int(layer_idx)]
            with bootstrap_tag("softmax.pre_exp"):
                mapped = _refresh_row_ciphers_if_short(
                    mapped,
                    bootstrap_operator=bootstrap_operator,
                    need_limbs=_decode_exp_need_limbs(exp_entry),
                    iterations=bootstrap_iterations,
                    precision_bits=bootstrap_precision_bits,
                )
            mapped_pages.append(mapped)
            exp_outputs = tuple(
                eval_exp_cipher(cipher, exp_entry, crypto_context=crypto_context)
                for cipher in mapped
            )
            masked = _mul_packed_rows(
                exp_outputs,
                _with_head_rows(
                    layout.compact_decode_prefix_mask(occupied), head_mask
                ),
                name="decode.alg2.post_exp",
                crypto_context=crypto_context,
            )
            _release_tuple(exp_outputs)
            y_pages.append(masked)
        _release_pages(mapped_pages)
        mapped_pages = []
        if bootstrap_operator is not None:
            with bootstrap_tag("softmax.post_exp"):
                y_pages = [
                    _refresh_row_ciphers(
                        page,
                        bootstrap_operator=bootstrap_operator,
                        iterations=bootstrap_iterations,
                        precision_bits=bootstrap_precision_bits,
                    )
                    for page in y_pages
                ]
        k = int(config.k_for(layer_idx) if rounds is None else rounds)
        rough = config.invsqrt_alg2["j1_rough"]
        precise = config.invsqrt_alg2["j2_precise"]
        for round_index in range(1, k + 1):
            poly = config.poly_for_layer(
                rough if round_index == 1 else precise, int(layer_idx)
            )
            with bootstrap_tag(f"softmax.r{round_index}.y"):
                y_pages = [
                    _refresh_row_ciphers_if_short(
                        page,
                        bootstrap_operator=bootstrap_operator,
                        need_limbs=ALG2_ROUND_NEED_LIMBS,
                        iterations=bootstrap_iterations,
                        precision_bits=bootstrap_precision_bits,
                    )
                    for page in y_pages
                ]
            nxt = _alg2_round_pages_fhe(
                y_pages,
                poly_entry=poly,
                layout=layout,
                crypto_context=crypto_context,
                occupied_pages=occupied_pages,
                round_index=round_index,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_precision_bits=bootstrap_precision_bits,
                head_mask=head_mask,
            )
            _release_pages(y_pages)
            y_pages = list(nxt)
        result = tuple(y_pages)
        y_pages = []
        return result
    finally:
        _release_pages(mapped_pages)
        _release_pages(y_pages)


def _alg2_round_pages_fhe(
    y_pages: list[tuple[object, ...]] | tuple[tuple[object, ...], ...],
    *,
    poly_entry: dict[str, Any],
    layout: AttentionPairLayout,
    crypto_context,
    occupied_pages: tuple[int, ...],
    round_index: int,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    head_mask: np.ndarray | None = None,
) -> tuple[tuple[object, ...], ...]:
    y_pages = [tuple(page) for page in y_pages]
    squared_pages: list[tuple[object, ...]] = []
    reduced_pages: list[tuple[object, ...]] = []
    isolated_pages: list[tuple[object, ...]] = []
    combined = filled = lam = None
    next_pages: list[tuple[object, ...]] = []
    try:
        for y_ciphers in y_pages:
            squared = tuple(
                square_rescale(cipher, crypto_context=crypto_context)
                for cipher in y_ciphers
            )
            squared_pages.append(squared)
            reduced = decode_reduce_tokens_fhe(
                squared, layout=layout, crypto_context=crypto_context
            )
            reduced_pages.append(reduced)
            isolated_pages.append(
                _mul_packed_rows(
                    reduced,
                    layout.compact_decode_token0_mask(),
                    name=f"decode.alg2.r{round_index}.sum0",
                    crypto_context=crypto_context,
                )
            )
        pair_count = len(y_pages[0])
        if len(isolated_pages) == 1:
            # ``isolated_pages`` owns these sums.  The general borrowed
            # helper deep-copies a one-item sequence, which is unnecessary
            # for the common single-page Decode steady state.
            combined = tuple(isolated_pages[0])
            isolated_pages[0] = ()
        else:
            combined = tuple(
                _sum_aligned_ciphers(
                    tuple(page[pair] for page in isolated_pages),
                    crypto_context=crypto_context,
                )
                for pair in range(pair_count)
            )
        fill = _invsqrt_fill_mask(layout, poly_entry, head_mask=head_mask)
        filled = _add_packed_rows(
            combined,
            fill,
            name=f"decode.alg2.r{round_index}.invsqrt_fill",
            crypto_context=crypto_context,
        )
        # ``_add_packed_rows`` borrows its input rows.  Release the combined
        # sums explicitly; in the single-page case these are the transferred
        # isolated rows rather than a disposable deep copy.
        _release_tuple(combined)
        combined = None
        with bootstrap_tag(f"softmax.r{round_index}.fill"):
            filled = _refresh_row_ciphers_if_short(
                filled,
                bootstrap_operator=bootstrap_operator,
                need_limbs=_invsqrt_need_limbs(poly_entry),
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
            )
        lam = tuple(
            eval_invsqrt_cipher(cipher, poly_entry, crypto_context=crypto_context)
            for cipher in filled
        )
        with bootstrap_tag(f"softmax.r{round_index}.lam"):
            lam = _refresh_row_ciphers(
                lam,
                bootstrap_operator=bootstrap_operator,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
            )
        for y_ciphers, occupied in zip(y_pages, occupied_pages, strict=True):
            lam_bcast = decode_broadcast_tokens_fhe(
                lam, layout=layout, crypto_context=crypto_context, occupied=occupied
            )
            products = []
            try:
                for y_cipher, lam_cipher in zip(y_ciphers, lam_bcast, strict=True):
                    aligned_y, aligned_lam, owned_y, owned_lam = _align_cipher_pair(
                        y_cipher, lam_cipher, crypto_context=crypto_context
                    )
                    try:
                        products.append(
                            mul_cipher_rescale(
                                aligned_y, aligned_lam, crypto_context=crypto_context
                            )
                        )
                    finally:
                        if owned_y:
                            release_if_supported(aligned_y)
                        if owned_lam:
                            release_if_supported(aligned_lam)
                scaled = tuple(products)
                products = []
                squared_y = tuple(
                    square_rescale(cipher, crypto_context=crypto_context)
                    for cipher in scaled
                )
                _release_tuple(scaled)
                remasked = _mul_packed_rows(
                    squared_y,
                    _with_head_rows(
                        layout.compact_decode_prefix_mask(occupied), head_mask
                    ),
                    name=f"decode.alg2.r{round_index}.prefix",
                    crypto_context=crypto_context,
                )
                _release_tuple(squared_y)
                next_pages.append(remasked)
            finally:
                _release_tuple(products)
                _release_tuple(lam_bcast)
        result = tuple(next_pages)
        next_pages = []
        return result
    finally:
        _release_pages(squared_pages)
        _release_pages(reduced_pages)
        _release_pages(isolated_pages)
        _release_tuple(combined)
        _release_tuple(filled)
        _release_tuple(lam)
        _release_pages(next_pages)


# --------------------------------------------------------------------------
# Merged-page Softmax: every CT cache page of one query shares one ciphertext
# --------------------------------------------------------------------------

#: ``LLAMA_FHE_SOFTMAX_MERGE_PAGES=0`` falls back to the per-page kernel.
SOFTMAX_MERGE_PAGES_ENV = "LLAMA_FHE_SOFTMAX_MERGE_PAGES"


SOFTMAX_SPARSE_BS_ENV = "LLAMA_FHE_SOFTMAX_SPARSE_BS"


#: ``LLAMA_FHE_SOFTMAX_SCHAIN=1`` runs the ``k = 2`` merged-page Softmax as the
#: S-chain (one aux bootstrap instead of two) -- see :func:`_schain_merged_tail`.
SOFTMAX_SCHAIN_ENV = "LLAMA_FHE_SOFTMAX_SCHAIN"


def softmax_schain_enabled() -> bool:
    """Whether ``k = 2`` merged-page Softmax layers use the S-chain tail (off by default)."""

    return os.environ.get(SOFTMAX_SCHAIN_ENV, "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


#: Limbs the S-chain probability must still carry so the PV station does not
#: have to refresh it (``_refresh_row_ciphers_if_short(need_limbs=6)``).
SCHAIN_PROBABILITY_MIN_LIMBS = 7


def schain_limb_schedule(config, layer_idx: int, *, exit_limbs: int) -> dict[str, int]:
    """Static limb plan of :func:`_schain_merged_tail` for one layer.

    ``exit_limbs`` is the bootstrap exit level (18 on the production L3
    program).  Returns the limbs of ``lam1`` (after the j1 polynomial),
    ``pack`` (the co-packed bootstrap input) and ``probability`` (what the
    kernel hands to PV).  :func:`schain_applicable` requires ``pack >=
    BOOTSTRAP_MIN_INPUT_LIMBS`` and ``probability >=
    SCHAIN_PROBABILITY_MIN_LIMBS``; otherwise the S-chain would cost an extra
    refresh and the Alg2 rounds are kept.
    """

    layer_idx = int(layer_idx)
    rough = config.poly_for_layer(config.invsqrt_alg2["j1_rough"], layer_idx)
    precise = config.poly_for_layer(config.invsqrt_alg2["j2_precise"], layer_idx)
    d1 = int(chebyshev_ps_mul_depth(_invsqrt_cheb(rough)[0])) + 1  # unfused map
    d2 = int(chebyshev_ps_mul_depth(_invsqrt_cheb(precise)[0]))  # map fused into the mask
    exit_limbs = int(exit_limbs)
    # y0 (exit) -> y1 = y0^2 -> S1 = isolate(reduce(y1)) -> rsqrt_j1
    lam1 = exit_limbs - 2 - d1
    # lam1 -> lam1^2 -> lam1^4 -> T = lam1^4 * S2 (co-packed with c*lam1 on token 1)
    pack = lam1 - 3
    # exit -> alpha*token0 mask -> rsqrt_j2 -> mu = lam2 * lam1'^2 -> Lambda = mu^2 -> y2' * Lambda
    probability = exit_limbs - 1 - d2 - 1 - 1 - 1
    return {"lam1": int(lam1), "pack": int(pack), "probability": int(probability)}


def schain_applicable(config, layer_idx: int, k: int, *, exit_limbs: int | None) -> bool:
    """``k == 2``, the env switch is on and the static limb plan needs no extra refresh."""

    if int(k) != 2 or not softmax_schain_enabled():
        return False
    if exit_limbs is None:
        return True  # no bootstrap operator (sim smoke): depth is the caller's business
    plan = schain_limb_schedule(config, layer_idx, exit_limbs=exit_limbs)
    return (
        plan["pack"] >= BOOTSTRAP_MIN_INPUT_LIMBS
        and plan["probability"] >= SCHAIN_PROBABILITY_MIN_LIMBS
    )


def _schain_copack_scale(rough: dict[str, Any], precise: dict[str, Any]) -> float:
    """``c`` so that ``c * lam1`` sits around the j2 fit interval (geometric centres).

    The co-packed bootstrap input holds ``T`` (inside the j2 interval) on
    token 0 and ``c * lam1`` on token 1; the refresh noise is relative to the
    ciphertext's largest slot, so both must have the same order of magnitude.
    """

    _, lo1, hi1 = _invsqrt_cheb(rough)
    _, lo2, hi2 = _invsqrt_cheb(precise)
    lam1_centre = (max(lo1, 1e-12) * max(hi1, 1e-12)) ** -0.25
    t_centre = math.sqrt(max(lo2, 1e-12) * max(hi2, 1e-12))
    return float(t_centre / lam1_centre)


def _add_aligned(left, right, *, crypto_context):
    """``left + right`` at the lower of the two levels (neither input consumed)."""

    aligned_left, aligned_right, owned_left, owned_right = _align_cipher_pair(
        left, right, crypto_context=crypto_context
    )
    try:
        return _homo_add(aligned_left, aligned_right, crypto_context=crypto_context)
    finally:
        if owned_left:
            release_if_supported(aligned_left)
        if owned_right:
            release_if_supported(aligned_right)


def _mul_aligned(left, right, *, crypto_context):
    """``left * right`` (relin + rescale) at the lower level (neither input consumed)."""

    aligned_left, aligned_right, owned_left, owned_right = _align_cipher_pair(
        left, right, crypto_context=crypto_context
    )
    try:
        return mul_cipher_rescale(aligned_left, aligned_right, crypto_context=crypto_context)
    finally:
        if owned_left:
            release_if_supported(aligned_left)
        if owned_right:
            release_if_supported(aligned_right)


def _schain_merged_tail(
    y,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int,
    masks: "_MergedPageMasks",
    page_count: int,
    stride: int,
    groups: int,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    sparse_period: int | None,
    own,
    drop,
) -> list[tuple[object, ...]]:
    """``k = 2`` Alg2 tail as the S-chain: same probabilities, one aux bootstrap.

    Alg2 with ``k = 2`` computes ``P = (y0 lam1)^4 lam2^2`` with ``lam1 =
    rsqrt(sum y0^2)`` and ``lam2 = rsqrt(sum (y0 lam1)^4) = rsqrt(lam1^4 sum
    y0^4)``.  The S-chain keeps the main track un-normalised (``y2' = y0^4``,
    two squarings, no ``lam`` multiply in between) and folds every ``lam``
    into one row scalar ``Lambda = lam1^4 lam2^2 = (lam1^2 lam2)^2`` applied
    once at the end.  Both row sums (``S1 = sum y1'``, ``S2 = sum y2'``) come
    straight off the main track, so the aux track is one dependency chain
    ``S1 -> lam1 -> T = lam1^4 S2 -> lam2 -> Lambda``; the single refresh sits
    on ``T`` and carries a scaled copy of ``lam1`` on token column 1 of the
    same (stride-periodic) ciphertext, so no second ``lam`` bootstrap is
    needed.  The Chebyshev map of the j2 polynomial is fused into the token-0
    mask after the refresh (``alpha`` in the mask, ``beta`` and the mapped
    safe fill in the add).  Limb plan: :func:`schain_limb_schedule`.
    """

    layer_idx = int(layer_idx)
    rough = config.poly_for_layer(config.invsqrt_alg2["j1_rough"], layer_idx)
    precise = config.poly_for_layer(config.invsqrt_alg2["j2_precise"], layer_idx)
    fill_mask = 1.0 - masks.merged_token0
    copack = _schain_copack_scale(rough, precise)

    # main track: y1' = y0^2, y2' = y0^4 (causal zeros stay zero under squaring)
    y1 = own(square_rescale(y, crypto_context=crypto_context))
    y2 = own(square_rescale(y1, crypto_context=crypto_context))
    drop(y)

    # S1 = sum y0^2 = row sum of y1', rough rsqrt
    reduced = own(_token_reduce_cipher(y1, layout=layout, crypto_context=crypto_context))
    drop(y1)
    isolated = own(
        _mul_packed_rows(
            (reduced,), masks.merged_token0, name="decode.schain.s1.sum0", crypto_context=crypto_context
        )[0]
    )
    drop(reduced)
    total1 = own(_rotate_add_tree(isolated, stride=stride, count=groups, crypto_context=crypto_context))
    drop(isolated)
    filled1 = own(
        _add_packed_rows(
            (total1,),
            fill_mask * _safe_invsqrt_fill(rough),
            name="decode.schain.s1.invsqrt_fill",
            crypto_context=crypto_context,
        )[0]
    )
    drop(total1)
    refreshed = _refresh_row_ciphers_if_short(
        (filled1,),
        bootstrap_operator=bootstrap_operator,
        need_limbs=_invsqrt_need_limbs(rough),
        iterations=bootstrap_iterations,
        precision_bits=bootstrap_precision_bits,
        tag="softmax.schain.s1.fill",
        periodic_slots=sparse_period,
    )
    if refreshed[0] is not filled1:
        drop(filled1)
        filled1 = own(refreshed[0])
    lam1 = own(eval_invsqrt_cipher(filled1, rough, crypto_context=crypto_context))
    drop(filled1)

    # S2 = sum y0^4 = row sum of y2' (token-0 only, no fill: it is multiplied, not fed to a poly)
    reduced = own(_token_reduce_cipher(y2, layout=layout, crypto_context=crypto_context))
    isolated = own(
        _mul_packed_rows(
            (reduced,), masks.merged_token0, name="decode.schain.s2.sum0", crypto_context=crypto_context
        )[0]
    )
    drop(reduced)
    total2 = own(_rotate_add_tree(isolated, stride=stride, count=groups, crypto_context=crypto_context))
    drop(isolated)

    # T = lam1^4 * S2 on token 0; c * lam1 on token 1; one sparse refresh for both
    lam1_sq = own(square_rescale(lam1, crypto_context=crypto_context))
    lam1_q = own(square_rescale(lam1_sq, crypto_context=crypto_context))
    drop(lam1_sq)
    t_cipher = own(_mul_aligned(lam1_q, total2, crypto_context=crypto_context))
    drop(lam1_q)
    drop(total2)
    lam1_copy = own(
        _mul_packed_rows(
            (lam1,), masks.merged_token0 * copack, name="decode.schain.lam1.copack", crypto_context=crypto_context
        )[0]
    )
    drop(lam1)
    lam1_shift = own(_homo_rotate(lam1_copy, -2, crypto_context=crypto_context))  # token 0 -> token 1
    drop(lam1_copy)
    pack = own(_add_aligned(t_cipher, lam1_shift, crypto_context=crypto_context))
    drop(t_cipher)
    drop(lam1_shift)
    refreshed = _refresh_row_ciphers(
        (pack,),
        bootstrap_operator=bootstrap_operator,
        iterations=bootstrap_iterations,
        precision_bits=bootstrap_precision_bits,
        tag="softmax.schain.pack",
        periodic_slots=sparse_period,
    )
    if refreshed[0] is not pack:
        drop(pack)
        pack = own(refreshed[0])

    # lam2 = rsqrt_j2(T) with the Chebyshev map fused into the token-0 mask
    _, lo2, hi2 = _invsqrt_cheb(precise)
    alpha = 2.0 / (hi2 - lo2)
    beta = -1.0 - 2.0 * lo2 / (hi2 - lo2)
    mapped_safe = alpha * _safe_invsqrt_fill(precise) + beta
    t_scaled = own(
        _mul_packed_rows(
            (pack,), masks.merged_token0 * alpha, name="decode.schain.t.alpha", crypto_context=crypto_context
        )[0]
    )
    t_mapped = own(
        _add_packed_rows(
            (t_scaled,),
            masks.merged_token0 * beta + fill_mask * mapped_safe,
            name="decode.schain.t.mapped_fill",
            crypto_context=crypto_context,
        )[0]
    )
    drop(t_scaled)
    lam2 = own(
        eval_invsqrt_cipher(
            t_mapped, precise, crypto_context=crypto_context, input_is_chebyshev_mapped=True
        )
    )
    drop(t_mapped)
    # lam1 back from token 1 (token-0 only after the mask, which also undoes c)
    lam1_back = own(_homo_rotate(pack, 2, crypto_context=crypto_context))
    drop(pack)
    lam1_b = own(
        _mul_packed_rows(
            (lam1_back,),
            masks.merged_token0 * (1.0 / copack),
            name="decode.schain.lam1.uncopack",
            crypto_context=crypto_context,
        )[0]
    )
    drop(lam1_back)
    lam1_b_sq = own(square_rescale(lam1_b, crypto_context=crypto_context))
    drop(lam1_b)
    # Lambda = (lam1^2 lam2)^2, token-0 only (lam1_b_sq zeroes lam2's fill slots)
    mu = own(_mul_aligned(lam2, lam1_b_sq, crypto_context=crypto_context))
    drop(lam2)
    drop(lam1_b_sq)
    big = own(square_rescale(mu, crypto_context=crypto_context))
    drop(mu)
    big_b = own(_token_broadcast_cipher(big, layout=layout, crypto_context=crypto_context))
    drop(big)
    # un-merge y2' page by page (prefix mask on the main track), then P_g = y2'_g * Lambda.
    # Lambda is stride-periodic after the group tree, so no rotation is needed on it.
    outputs: list[tuple[object, ...]] = []
    for group in range(int(page_count)):
        back = own(_homo_rotate(y2, stride * group, crypto_context=crypto_context))
        page_y = own(
            _mul_packed_rows(
                (back,),
                masks.page_valid[group],
                name=f"decode.schain.prefix.g{group}",
                crypto_context=crypto_context,
            )[0]
        )
        drop(back)
        page_out = _mul_aligned(page_y, big_b, crypto_context=crypto_context)
        drop(page_y)
        outputs.append((page_out,))
    drop(y2)
    drop(big_b)
    return outputs


def softmax_sparse_bootstrap_enabled() -> bool:
    """Whether the merged-page Softmax may route its periodic rows through a sparse bootstrap program.

    On by default; the operator still has to carry a program of period
    ``decode_page_merge_stride`` slots (``preset.bootstrap_extra_sparse_log_slots``),
    otherwise the dense program is used.  ``LLAMA_FHE_SOFTMAX_SPARSE_BS=0``
    forces dense (A/B baseline).
    """

    return os.environ.get(SOFTMAX_SPARSE_BS_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def merged_page_sparse_period(layout, bootstrap_operator) -> int | None:
    """Slot period of the merged-page ``total`` / ``filled`` / ``lambda`` rows if it can go sparse.

    The packed payload (``groups * stride`` slots) tiles the slot vector, so
    after the cyclic group rotate-add every ``stride``-slot block of the
    merged ciphertext carries the same ``sum(y^2)`` row, ``fill_mask = 1 -
    merged_token0`` is itself a ``stride``-periodic pattern, and ``1/sqrt`` is
    slot-wise: those rows are exactly periodic with period ``stride`` and may
    be bootstrapped as a ``log2(stride)``-slot sparse view (production: 4096
    slots, log_slots 12).  ``None`` when the operator has no such program,
    the payload does not tile the slot vector, or the env switch is off.
    """

    if bootstrap_operator is None or not softmax_sparse_bootstrap_enabled():
        return None
    stride = int(layout.decode_page_merge_stride)
    groups = int(layout.decode_page_merge_groups)
    if int(layout.slots) % (stride * groups) or stride & (stride - 1):
        return None
    probe = getattr(bootstrap_operator, "supports_periodic_slots", None)
    if not callable(probe) or not probe(stride):
        return None
    return stride


def softmax_merge_pages_enabled() -> bool:
    return os.environ.get(SOFTMAX_MERGE_PAGES_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def softmax_page_merge_rotations(layout) -> tuple[int, ...]:
    """Rotation keys of :func:`_softmax_alg2_merged_pages_fhe` for any layout
    exposing ``seq_len`` / ``head_dim`` / ``query_heads`` (compact
    :class:`AttentionPairLayout` or ``PairFold128Layout``)."""

    pairs = int(layout.query_heads) // 2
    stride = 2 * int(layout.seq_len) * pairs
    groups = int(layout.head_dim) // pairs
    rotations: list[int] = []
    for group in range(1, groups):
        rotations.append(-stride * group)
        rotations.append(stride * group)
    bit = 1
    while bit < groups:
        rotations.append(stride * bit)
        bit <<= 1
    return tuple(dict.fromkeys(rotations))


@dataclass(frozen=True)
class _MergedPageMasks:
    """Packed ``[1, slots]`` masks of a merged-page Softmax.

    ``page_valid[g]`` is page ``g``'s own (prefix ∧ owned head rows) mask on
    rows ``0 .. pairs-1`` (used before the merge and after the un-merge);
    ``merged_valid`` is their union after shifting page ``g`` by ``g *
    stride`` slots; ``merged_token0`` selects column 0 of the owned rows in
    *every* group (also the empty ones), so after the cyclic group
    rotate-add every group's column 0 carries the full ``sum(y^2)`` and the
    ``1/sqrt`` fill never has to cover a total-plus-fill slot.
    """

    page_valid: tuple[np.ndarray, ...]
    merged_valid: np.ndarray
    merged_token0: np.ndarray


def _merged_page_masks(
    layout: AttentionPairLayout,
    occupied_pages: tuple[int, ...],
    head_mask: np.ndarray | None,
) -> _MergedPageMasks:
    stride = int(layout.decode_page_merge_stride)
    groups = int(layout.decode_page_merge_groups)
    slots = int(layout.slots)
    if stride * groups != int(layout.payload_slots):
        raise ValueError(
            "merged pages need stride*groups == payload_slots "
            f"({stride}*{groups} != {layout.payload_slots})."
        )
    page_valid = tuple(
        np.asarray(
            _with_head_rows(layout.compact_decode_prefix_mask(occupied), head_mask),
            dtype=np.float64,
        ).reshape(1, slots)
        for occupied in occupied_pages
    )
    merged_valid = np.zeros((1, slots), dtype=np.float64)
    for group, valid in enumerate(page_valid):
        merged_valid += np.roll(valid, stride * group, axis=1)
    token0 = np.asarray(
        _with_head_rows(layout.compact_decode_token0_mask(), head_mask), dtype=np.float64
    ).reshape(1, slots)
    merged_token0 = np.zeros((1, slots), dtype=np.float64)
    for group in range(groups):
        merged_token0 += np.roll(token0, stride * group, axis=1)
    return _MergedPageMasks(
        page_valid=page_valid,
        merged_valid=np.minimum(merged_valid, 1.0),
        merged_token0=np.minimum(merged_token0, 1.0),
    )


def _homo_add(left, right, *, crypto_context):
    return crypto_context.fhe.homo_add(left, right, crypto_context.context)


def _softmax_alg2_merged_pages_fhe(
    score_pages: tuple[tuple[object, ...], ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    occupied_pages: tuple[int, ...],
    layer_idx: int,
    rounds: int | None,
    score_scale: float | None,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    head_rows,
) -> tuple[tuple[object, ...], ...]:
    """Alg2 over ``G`` compact score pages packed into one ciphertext.

    Slot-for-slot the same arithmetic as :func:`softmax_alg2_row_pages_fhe`
    (public map, exp, prefix mask, ``k`` rounds of ``y <- (y*λ)^2`` with
    ``λ = 1/sqrt(sum y^2)``), but page ``g`` lives on rows ``g*pairs ..`` of
    one ciphertext: one exp, one post-exp bootstrap and one set of round
    multiplications serve every page, and the cross-page ``sum(y^2)`` is a
    3-rotation group tree instead of per-page adds.  Returns one
    probability ciphertext per page, back on rows ``0 .. pairs-1``.
    """

    layout._require_compact_q()
    pages = tuple(tuple(page) for page in score_pages)
    for page in pages:
        if len(page) != 1:
            raise ValueError("merged-page Softmax expects 1 compact cipher per page.")
    stride = int(layout.decode_page_merge_stride)
    groups = int(layout.decode_page_merge_groups)
    head_mask = _head_rows_mask(layout, head_rows)
    masks = _merged_page_masks(layout, occupied_pages, head_mask)
    k = int(config.k_for(layer_idx) if rounds is None else rounds)
    exp_entry = config.layer_exp[int(layer_idx)]
    scale = (
        1.0 / math.sqrt(float(layout.head_dim)) if score_scale is None else float(score_scale)
    )
    shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    invalid_value = float(config.stats_min_for(layer_idx)) / float(2**k)
    rough = config.invsqrt_alg2["j1_rough"]
    precise = config.invsqrt_alg2["j2_precise"]
    fill_mask = 1.0 - masks.merged_token0
    # ``total`` / ``filled`` / ``lam`` are ``stride``-periodic (see
    # merged_page_sparse_period); ``y`` is not (every page differs).
    sparse_period = merged_page_sparse_period(layout, bootstrap_operator)
    live: list[object] = []  # everything still to release

    def own(cipher):
        live.append(cipher)
        return cipher

    def drop(cipher):
        if cipher is not None and cipher in live:
            live.remove(cipher)
            release_if_supported(cipher)

    try:
        # 1. public map, page by page, merged by rotation into one ciphertext
        merged = None
        for group, page in enumerate(pages):
            scaled = own(
                _mul_packed_rows(
                    page,
                    masks.page_valid[group] * (scale / float(2**k)),
                    name=f"decode.alg2m.valid_scale.g{group}",
                    crypto_context=crypto_context,
                )[0]
            )
            rotated = own(_homo_rotate(scaled, -stride * group, crypto_context=crypto_context))
            drop(scaled)
            if merged is None:
                merged = rotated
            else:
                summed = own(_homo_add(merged, rotated, crypto_context=crypto_context))
                drop(merged)
                drop(rotated)
                merged = summed
        shifted = own(_add_scalar_ciphers((merged,), shift, crypto_context=crypto_context)[0])
        drop(merged)
        mapped = own(
            _add_packed_rows(
                (shifted,),
                (1.0 - masks.merged_valid) * invalid_value,
                name="decode.alg2m.invalid",
                crypto_context=crypto_context,
            )[0]
        )
        drop(shifted)
        # 2. exp, prefix mask, post-exp refresh (one bootstrap for all pages)
        refreshed = _refresh_row_ciphers_if_short(
            (mapped,),
            bootstrap_operator=bootstrap_operator,
            need_limbs=_decode_exp_need_limbs(exp_entry),
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.pre_exp",
        )
        if refreshed[0] is not mapped:
            live.remove(mapped)  # consumed by the refresh
            mapped = own(refreshed[0])
        exp_out = own(eval_exp_cipher(mapped, exp_entry, crypto_context=crypto_context))
        drop(mapped)
        y = own(
            _mul_packed_rows(
                (exp_out,), masks.merged_valid, name="decode.alg2m.post_exp", crypto_context=crypto_context
            )[0]
        )
        drop(exp_out)
        refreshed = _refresh_row_ciphers(
            (y,),
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.post_exp",
        )
        if refreshed[0] is not y:
            live.remove(y)
            y = own(refreshed[0])
        # 3a. k = 2 S-chain tail (one aux bootstrap) when enabled and limb-feasible
        # (``y`` just left the unconditional post-exp refresh, so its limbs
        # are the operator's exit level.)
        if schain_applicable(
            config,
            layer_idx,
            k,
            exit_limbs=(None if bootstrap_operator is None else int(y.state.cur_limbs)),
        ):
            outputs = _schain_merged_tail(
                y,
                layout=layout,
                crypto_context=crypto_context,
                config=config,
                layer_idx=layer_idx,
                masks=masks,
                page_count=len(pages),
                stride=stride,
                groups=groups,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_precision_bits=bootstrap_precision_bits,
                sparse_period=sparse_period,
                own=own,
                drop=drop,
            )
            result = tuple(outputs)
            outputs = []
            return result
        # 3. Alg2 rounds on the merged ciphertext
        outputs: list[tuple[object, ...]] = []
        for round_index in range(1, k + 1):
            poly = config.poly_for_layer(rough if round_index == 1 else precise, int(layer_idx))
            refreshed = _refresh_row_ciphers_if_short(
                (y,),
                bootstrap_operator=bootstrap_operator,
                need_limbs=ALG2_ROUND_NEED_LIMBS,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag=f"softmax.r{round_index}.y",
            )
            if refreshed[0] is not y:
                live.remove(y)
                y = own(refreshed[0])
            squared = own(square_rescale(y, crypto_context=crypto_context))
            reduced = own(_token_reduce_cipher(squared, layout=layout, crypto_context=crypto_context))
            drop(squared)
            isolated = own(
                _mul_packed_rows(
                    (reduced,),
                    masks.merged_token0,
                    name=f"decode.alg2m.r{round_index}.sum0",
                    crypto_context=crypto_context,
                )[0]
            )
            drop(reduced)
            total = own(
                _rotate_add_tree(isolated, stride=stride, count=groups, crypto_context=crypto_context)
            )
            drop(isolated)
            filled = own(
                _add_packed_rows(
                    (total,),
                    fill_mask * _safe_invsqrt_fill(poly),
                    name=f"decode.alg2m.r{round_index}.invsqrt_fill",
                    crypto_context=crypto_context,
                )[0]
            )
            drop(total)
            refreshed = _refresh_row_ciphers_if_short(
                (filled,),
                bootstrap_operator=bootstrap_operator,
                need_limbs=_invsqrt_need_limbs(poly),
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag=f"softmax.r{round_index}.fill",
                periodic_slots=sparse_period,
            )
            if refreshed[0] is not filled:
                live.remove(filled)
                filled = own(refreshed[0])
            lam = own(eval_invsqrt_cipher(filled, poly, crypto_context=crypto_context))
            drop(filled)
            refreshed = _refresh_row_ciphers(
                (lam,),
                bootstrap_operator=bootstrap_operator,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag=f"softmax.r{round_index}.lam",
                periodic_slots=sparse_period,
            )
            if refreshed[0] is not lam:
                live.remove(lam)
                lam = own(refreshed[0])
            # broadcast λ along the token axis of every merged row (same three
            # steps as decode_broadcast_tokens_fhe: isolate, tree, remask)
            lam0 = own(
                _mul_packed_rows(
                    (lam,), masks.merged_token0, name="decode.alg2m.lambda.token0", crypto_context=crypto_context
                )[0]
            )
            drop(lam)
            lam_b = own(_token_broadcast_cipher(lam0, layout=layout, crypto_context=crypto_context))
            drop(lam0)
            lam_bcast = own(
                _mul_packed_rows(
                    (lam_b,), masks.merged_valid, name="decode.alg2m.lambda.prefix", crypto_context=crypto_context
                )[0]
            )
            drop(lam_b)
            aligned_y, aligned_lam, owned_y, owned_lam = _align_cipher_pair(
                y, lam_bcast, crypto_context=crypto_context
            )
            try:
                product = own(mul_cipher_rescale(aligned_y, aligned_lam, crypto_context=crypto_context))
            finally:
                if owned_y:
                    release_if_supported(aligned_y)
                if owned_lam:
                    release_if_supported(aligned_lam)
            drop(lam_bcast)
            drop(y)
            squared_y = own(square_rescale(product, crypto_context=crypto_context))
            drop(product)
            if round_index < k:
                y = own(
                    _mul_packed_rows(
                        (squared_y,),
                        masks.merged_valid,
                        name=f"decode.alg2m.r{round_index}.prefix",
                        crypto_context=crypto_context,
                    )[0]
                )
                drop(squared_y)
                continue
            # last round: un-merge straight into the per-page prefix mask
            for group in range(len(pages)):
                back = own(_homo_rotate(squared_y, stride * group, crypto_context=crypto_context))
                page_out = _mul_packed_rows(
                    (back,),
                    masks.page_valid[group],
                    name=f"decode.alg2m.r{round_index}.prefix.g{group}",
                    crypto_context=crypto_context,
                )
                drop(back)
                outputs.append(tuple(page_out))
            drop(squared_y)
        if k < 1:
            raise ValueError("Alg2 needs at least one round.")
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in live:
            release_if_supported(cipher)
        live.clear()


__all__ = [
    "ALG2_ROUND_NEED_LIMBS",
    "SOFTMAX_SCHAIN_ENV",
    "SCHAIN_PROBABILITY_MIN_LIMBS",
    "schain_applicable",
    "schain_limb_schedule",
    "softmax_schain_enabled",
    "SOFTMAX_MERGE_PAGES_ENV",
    "SOFTMAX_SPARSE_BS_ENV",
    "merged_page_sparse_period",
    "softmax_merge_pages_enabled",
    "softmax_sparse_bootstrap_enabled",
    "softmax_page_merge_rotations",
    "ALG2_ROUND_Y_LIMBS",
    "BOOTSTRAP_MIN_INPUT_LIMBS",
    "DecodeAlg2Poly",
    "apply_decode_alg2_public_map_fhe",
    "apply_decode_alg2_public_map_numpy",
    "softmax_alg2_poly_numpy",
    "softmax_alg2_row_fhe",
    "softmax_alg2_row_pages_fhe",
]
