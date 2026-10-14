from __future__ import annotations

"""Ciphertext-side helpers of the shifted-lane (SL) decode packing.

The numpy contract lives in :mod:`llama3fhe.layouts.shifted_lane`; this
module provides the two rotations stages a :class:`LinearOperator` with
``plan.lane_shift > 0`` needs around its MAC:

* :func:`expand_lane0_to_shifted_lanes` -- lane-0-only carrier -> SL form
  (``token_lanes - 1`` hoisted rotations + adds; optional lane-0 mask that
  costs one level when lanes 1.. are not already zero);
* :func:`fold_shifted_lanes` -- MAC output -> lane 0 holds ``W x``
  (``log2(token_lanes)`` rotate-and-add, no level).  Lanes 1.. hold
  cross-channel garbage afterwards: mask them (:func:`mask_lane0`) before any
  non-linear stage or before the vector is added to the residual stream.

In a chunked (multi-rank) linear every rank MACs its own giant chunks and the
partial accumulators are all-reduced *before* the fold, so the fold runs once
on the summed output.
"""

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.operators.primitives import add_owned, mul_plain_rescale

from .config import LinearPlan


def lane0_mask_vector(plan: LinearPlan, *, slots: int) -> np.ndarray:
    """Plaintext vector selecting lane 0 of every channel."""

    slots = int(slots)
    if slots != int(plan.dimension) * int(plan.token_lanes):
        raise ValueError(
            f"lane mask expects slots={plan.dimension * plan.token_lanes}, got {slots}."
        )
    mask = np.zeros((slots,), dtype=np.float64)
    mask[:: int(plan.token_lanes)] = 1.0
    return mask


def mask_lane0(cipher, *, plan: LinearPlan, crypto_context):
    """``cipher * lane0_mask``, rescaled (one level). Input is not consumed."""

    level = int(crypto_context.level_for_cipher(cipher))
    plain = crypto_context.plaintext(
        lane0_mask_vector(plan, slots=int(crypto_context.max_slots)),
        name=f"sl.lane0_mask.{plan.dimension}x{plan.token_lanes}",
        level=level,
    )
    return mul_plain_rescale(cipher, plain, crypto_context=crypto_context)


def expand_lane0_to_shifted_lanes(
    cipher,
    *,
    plan: LinearPlan,
    crypto_context,
    mask: bool = False,
):
    """Build the SL carrier from a carrier whose lane 0 holds ``x``.

    ``mask=True`` first zeroes lanes 1.. (one level); leave it ``False`` when
    the producer already guarantees zero lanes (e.g. the RMSNorm gamma
    plaintext carried the lane-0 mask).  Input is not consumed.
    """

    if not plan.shifted_lanes:
        raise ValueError("expand_lane0_to_shifted_lanes needs a shifted-lane plan.")
    fhe = crypto_context.fhe
    context = crypto_context.context
    base = mask_lane0(cipher, plan=plan, crypto_context=crypto_context) if mask else None
    source = cipher if base is None else base
    try:
        batch = fhe.fast_rotate(source, tuple(plan.lane_expand_offsets), context)
        rotated = list(fhe.unpack_cipher_batch(batch))
        del batch
        out = source.deep_copy()
        for term in rotated:
            out = add_owned(out, term, crypto_context=crypto_context)
        return out
    finally:
        if base is not None:
            release_if_supported(base)


def fold_shifted_lanes(cipher, *, plan: LinearPlan, crypto_context):
    """Rotate-and-add by 1, 2, 4, ... so lane 0 holds the lane sum.

    Lanes 1.. of the result are garbage (their windows cross into the next
    channel).  Input is not consumed.
    """

    if not plan.shifted_lanes:
        raise ValueError("fold_shifted_lanes needs a shifted-lane plan.")
    fhe = crypto_context.fhe
    context = crypto_context.context
    out = cipher.deep_copy()
    for offset in plan.fold_offsets:
        rotated = fhe.homo_rotate(out, int(offset), context)
        out = add_owned(out, rotated, crypto_context=crypto_context)
    return out


__all__ = [
    "expand_lane0_to_shifted_lanes",
    "fold_shifted_lanes",
    "lane0_mask_vector",
    "mask_lane0",
]
