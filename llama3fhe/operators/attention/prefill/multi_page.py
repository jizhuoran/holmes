"""Fixed-width Delta pages for variable-length encrypted Prefill.

Each page has the ordinary 128-by-128 Delta geometry.  Earlier key pages
use a full attention mask; the current page uses its causal mask.  All pages
for one query block share the same row denominator in every Alg2 round.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout

from ...primitives import accumulate_owned
from .softmax import (
    delta_causal_mask,
    delta_reduce_rows_fhe,
    softmax_alg2_fhe_delta,
)


@dataclass
class DeltaSoftmaxAlg2PagesResult:
    probability_pages: tuple[tuple[object, ...], ...]
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]
    slim_period_rows: int | None
    slim_sparse_bootstraps: int

    def release(self) -> None:
        pages = self.probability_pages
        self.probability_pages = ()
        for page in pages:
            for cipher in page:
                release_if_supported(cipher)


def _valid_tokens(
    mask: np.ndarray | None, *, seq_len: int, name: str
) -> np.ndarray:
    if mask is None:
        return np.ones(seq_len, dtype=bool)
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    if valid.shape != (seq_len,):
        raise ValueError(f"{name} must have shape ({seq_len},), got {valid.shape}.")
    return valid


def delta_full_page_mask(
    layout: AttentionPairLayout,
    *,
    query_valid_mask: np.ndarray | None = None,
    key_valid_mask: np.ndarray | None = None,
    dtype=np.float64,
) -> np.ndarray:
    """Mask one earlier, fully visible key page in Delta slot order.

    Invalid query columns are empty here.  The current causal page supplies
    their self slot so the shared inverse-square-root input stays nonzero.
    """

    seq_len = int(layout.seq_len)
    queries = np.arange(seq_len, dtype=np.int64)
    deltas = np.arange(seq_len, dtype=np.int64)
    keys = (queries[None, :] + deltas[:, None]) % seq_len
    query_valid = _valid_tokens(
        query_valid_mask, seq_len=seq_len, name="query_valid_mask"
    )
    key_valid = _valid_tokens(key_valid_mask, seq_len=seq_len, name="key_valid_mask")
    active = query_valid[None, :] & key_valid[keys]
    payload = np.repeat(active[:, :, None], 2, axis=2).reshape(-1).astype(dtype)
    return np.tile(payload, int(layout.payload_repetitions))


def prefill_delta_page_masks(
    layout: AttentionPairLayout,
    *,
    historical_key_valid_masks: Sequence[np.ndarray | None] = (),
    current_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Return historical full-page masks followed by the current causal mask."""

    return (
        *(
            delta_full_page_mask(
                layout,
                query_valid_mask=current_valid_mask,
                key_valid_mask=key_valid,
            )
            for key_valid in historical_key_valid_masks
        ),
        delta_causal_mask(layout, token_valid_mask=current_valid_mask),
    )


def delta_page_exp_amplitude_mask(
    layout: AttentionPairLayout,
    *,
    page_count: int,
    query_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize live query rows before the first shared Alg2 row sum.

    All pages use the same factor, which cancels in the first normalization.
    Padding sentinels stay at unit amplitude because they have no history.
    """

    page_count = int(page_count)
    if page_count < 1:
        raise ValueError("page_count must be positive.")
    seq_len = int(layout.seq_len)
    query_valid = _valid_tokens(
        query_valid_mask, seq_len=seq_len, name="query_valid_mask"
    )
    factors = np.where(query_valid, 1.0 / math.sqrt(page_count), 1.0)
    payload = np.broadcast_to(
        factors[None, :, None], (seq_len, seq_len, 2)
    ).reshape(-1)
    return np.tile(payload, int(layout.payload_repetitions))


def _checked_numpy_pages(
    score_pages: Sequence[np.ndarray],
    page_masks: Sequence[np.ndarray],
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, np.ndarray]:
    if not score_pages or len(score_pages) != len(page_masks):
        raise ValueError("score_pages and page_masks need the same nonzero length.")
    pairs = int(layout.query_pair_count)
    slots = int(layout.slots)
    scores = np.stack([np.asarray(page) for page in score_pages])
    if scores.shape != (len(score_pages), pairs, slots):
        raise ValueError(
            f"score pages must have shape ({pairs}, {slots}) each, "
            f"got {scores.shape}."
        )
    if np.iscomplexobj(scores) or not np.isfinite(scores).all():
        raise ValueError("score pages must contain finite real values.")
    expanded_masks = []
    for mask in page_masks:
        value = np.asarray(mask)
        if value.shape == (slots,):
            value = np.broadcast_to(value, (pairs, slots))
        if value.shape != (pairs, slots):
            raise ValueError(
                f"page masks must have shape ({slots},) or ({pairs}, {slots}), "
                f"got {value.shape}."
            )
        if not np.all((value == 0) | (value == 1)):
            raise ValueError("page masks must be binary.")
        expanded_masks.append(value.astype(bool, copy=False))
    masks = np.stack(expanded_masks)
    shape = (
        len(score_pages),
        pairs,
        int(layout.payload_repetitions),
        int(layout.seq_len),
        int(layout.seq_len),
        2,
    )
    scores = scores.astype(np.float64, copy=False).reshape(shape)
    masks = masks.reshape(shape)
    if not masks.any(axis=(0, 3)).all():
        raise ValueError("each query/head lane needs an active score on some page.")
    return scores, masks


def softmax_alg2_pages_numpy(
    score_pages: Sequence[np.ndarray],
    page_masks: Sequence[np.ndarray],
    *,
    layout: AttentionPairLayout,
    rounds: int,
    score_scale: float | None = None,
    u_hat: float = 0.0,
    query_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Slot oracle for one shared Alg2 denominator across Delta pages.

    This uses exact exp and inverse square root.  It checks packing, masks and
    row coupling; polynomial and CKKS approximation error are outside its scope.
    """

    rounds = int(rounds)
    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}.")
    scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    if not math.isfinite(scale) or not math.isfinite(float(u_hat)):
        raise ValueError("score_scale and u_hat must be finite.")
    scores, masks = _checked_numpy_pages(score_pages, page_masks, layout=layout)
    mapped = np.where(
        masks, (scores * scale - float(u_hat)) / float(2**rounds), -np.inf
    )
    y = np.exp(mapped)
    query_valid = _valid_tokens(
        query_valid_mask, seq_len=int(layout.seq_len), name="query_valid_mask"
    )
    amplitudes = np.where(query_valid, 1.0 / math.sqrt(len(score_pages)), 1.0)
    y *= amplitudes[None, None, None, None, :, None]
    for _ in range(rounds):
        squared = y * y
        # Sum pages in the same Delta slots first.  Row reduction is linear,
        # so this is the same Z as reducing each page separately and adding.
        all_page_squares = np.sum(squared, axis=0)
        row_sum = np.sum(all_page_squares, axis=2)
        if not np.isfinite(row_sum).all() or np.any(row_sum <= 0):
            raise ValueError("Alg2 row sum is non-finite or zero.")
        y = squared / row_sum[None, :, :, None, :, :]
    return tuple(
        page.reshape(int(layout.query_pair_count), int(layout.slots)) for page in y
    )


def softmax_alg2_fhe_delta_pages(
    score_pages: Sequence[tuple[object, ...]],
    page_masks: Sequence[np.ndarray],
    *,
    layout: AttentionPairLayout,
    layer_idx: int,
    config,
    crypto_context,
    operator_config=None,
    score_scale: float | None = None,
    bootstrap_operator=None,
    refresh_before_exp: bool | None = None,
    refresh_after_exp: bool | None = None,
    rounds: int | None = None,
    current_valid_mask: np.ndarray | None = None,
) -> DeltaSoftmaxAlg2PagesResult:
    """Run one Alg2 normalization across fixed-width Delta score pages.

    Score pages and masks remain caller-owned.  Historical pages should carry
    full key masks and the final current page its causal mask; use
    :func:`prefill_delta_page_masks` to construct these masks.  The fitted
    inverse-square-root interval must cover the combined page row sum.
    """

    pages = tuple(tuple(page) for page in score_pages)
    masks = tuple(np.asarray(mask, dtype=np.float64) for mask in page_masks)
    if not pages or len(pages) != len(masks):
        raise ValueError("score_pages and page_masks need the same nonzero length.")
    max_pages = int(getattr(config, "max_prefill_pages", len(pages)))
    if len(pages) > max_pages:
        raise ValueError(
            f"Softmax coefficients support at most {max_pages} Prefill page(s), "
            f"got {len(pages)}."
        )
    pair_count = int(layout.query_pair_count)
    if any(len(page) != pair_count for page in pages):
        raise ValueError(f"each score page must contain {pair_count} ciphertexts.")
    if any(mask.shape != (int(layout.slots),) for mask in masks):
        raise ValueError(f"each page mask must have shape ({layout.slots},).")
    scores = tuple(cipher for page in pages for cipher in page)
    states = tuple(getattr(cipher, "state", None) for cipher in scores)
    if all(state is not None for state in states):
        limbs = {int(state.cur_limbs) for state in states}
        degrees = {int(state.scale_degree) for state in states}
        scales = tuple(float(state.scaling_factor) for state in states)
        if len(limbs) != 1 or len(degrees) != 1 or not all(
            math.isclose(scale, scales[0], rel_tol=1e-6)
            for scale in scales[1:]
        ):
            raise ValueError("Delta score pages must have matching limb count and scale.")
    if any(int(getattr(cipher, "slots", layout.slots)) != int(layout.slots) for cipher in scores):
        raise ValueError("Delta score page ciphertexts must use the layout slot count.")
    result = softmax_alg2_fhe_delta(
        pages[0],
        layout=layout,
        layer_idx=layer_idx,
        config=config,
        crypto_context=crypto_context,
        operator_config=operator_config,
        score_scale=score_scale,
        bootstrap_operator=bootstrap_operator,
        refresh_before_exp=refresh_before_exp,
        refresh_after_exp=refresh_after_exp,
        rounds=rounds,
        token_valid_mask=current_valid_mask,
        _score_pages=pages,
        _page_masks=masks,
    )
    try:
        probability_pages = tuple(
            result.probability_ciphers[index * pair_count : (index + 1) * pair_count]
            for index in range(len(pages))
        )
        if any(len(page) != pair_count for page in probability_pages):
            raise RuntimeError("multi-page Delta Alg2 returned the wrong page count.")
        output = DeltaSoftmaxAlg2PagesResult(
            probability_pages=probability_pages,
            stage_seconds=result.stage_seconds,
            output_levels=result.output_levels,
            slim_period_rows=result.slim_period_rows,
            slim_sparse_bootstraps=result.slim_sparse_bootstraps,
        )
        result.probability_ciphers = ()
        return output
    finally:
        result.release()


def delta_reduce_page_squares_fhe(
    square_pages: Sequence[tuple[object, ...]],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Add same-slot page squares, then run one Delta row reduction.

    Inputs remain caller-owned.  The caller must supply equal-level squares;
    this function owns and releases only its temporary page sum.
    """

    if not square_pages:
        raise ValueError("at least one square page is required.")
    pairs = int(layout.query_pair_count)
    if any(len(page) != pairs for page in square_pages):
        raise ValueError(f"each square page must contain {pairs} ciphertexts.")
    states = [
        getattr(cipher, "state", None)
        for page in square_pages
        for cipher in page
    ]
    if all(state is not None for state in states):
        limbs = {int(state.cur_limbs) for state in states}
        degrees = {int(state.scale_degree) for state in states}
        scales = [float(state.scaling_factor) for state in states]
        if len(limbs) != 1 or len(degrees) != 1 or not all(
            math.isclose(scale, scales[0], rel_tol=1e-6)
            for scale in scales[1:]
        ):
            raise ValueError(
                "Delta page squares must have matching limb count and scale "
                "before the shared row sum."
            )
    summed: list[object | None] = [None] * pairs
    try:
        for page in square_pages:
            for index, cipher in enumerate(page):
                summed[index] = accumulate_owned(
                    summed[index], cipher.deep_copy(), crypto_context=crypto_context
                )
        return delta_reduce_rows_fhe(
            tuple(summed), layout=layout, crypto_context=crypto_context
        )
    finally:
        for cipher in summed:
            release_if_supported(cipher)
