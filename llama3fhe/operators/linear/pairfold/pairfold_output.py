from __future__ import annotations

"""Folded PairFold128 output projection.

The Decode-native attention path produces one sparse carrier whose active
query-residue channels hold ``z = O_low + i*O_high``.  This module applies

``A = 0.5 * (W_low - i*W_high)``

with one complex diagonal projection, then recovers the real DecodeFM result
as ``A*z + conj(A*z)``.  The input ciphertext is borrowed; the returned
ciphertext is owned by the caller.
"""

import time
from dataclasses import dataclass

import numpy as np

from llama3fhe.backend import (
    release_if_supported,
)
from llama3fhe.backend import (
    profile_sync_device as sync_device,
)
from llama3fhe.layouts.pairfold128 import PairFold128Layout

from ..config import LLAMA3_LINEAR_SCHEDULES, LinearOperatorConfig, linear_rotations
from ..operator import LinearOperator
from ..structured_weight import StructuredLinearWeight, structured_linear_weight

# Folded W_O has one plaintext multiplication/rescale and no preceding
# O_C->DecodeFM rescale.  Two input limbs are therefore the tight graph-level
# precondition; a caller seeing fewer limbs must refresh before entering this
# operator.  Keeping the decision outside this borrowed-input wrapper avoids
# silently changing bootstrap policy when the graph is switched A/B.
PAIRFOLD_W_O_MIN_INPUT_LIMBS = 2


@dataclass(frozen=True)
class PairFoldOutputWeight:
    """Structured ``A`` descriptor and its sparse physical input support."""

    descriptor: StructuredLinearWeight
    physical_width: int
    active_input_channels: tuple[int, ...]

    @property
    def active_input_count(self) -> int:
        return len(self.active_input_channels)


def _pairfold_output_columns(
    layout: PairFold128Layout,
) -> tuple[np.ndarray, np.ndarray]:
    """Map sparse folded-O channels to low/high logical model columns."""

    dimension = int(layout.carrier_dimension)
    low_columns = np.full((dimension,), -1, dtype=np.int64)
    high_columns = np.full((dimension,), -1, dtype=np.int64)
    half = int(layout.fold_dim)
    for group in range(int(layout.key_value_groups)):
        for pair in range(int(layout.pair_blocks)):
            for lane in range(int(layout.head_lanes)):
                for query_offset in range(int(layout.gqa_ratio)):
                    head = layout.query_head(
                        group, pair, lane, query_offset
                    )
                    residue = layout.query_residue(group, query_offset)
                    head_base = int(head) * int(layout.head_dim)
                    for feature in range(half):
                        channel = layout.carrier_channel(
                            pair, feature, residue, lane
                        )
                        low_columns[channel] = head_base + feature
                        high_columns[channel] = head_base + half + feature
    return low_columns, high_columns


def pairfold_w_o_weight(
    weight: np.ndarray,
    *,
    layout: PairFold128Layout | None = None,
) -> PairFoldOutputWeight:
    """Build ``A=.5*(W_low-iW_high)`` without a dense physical permutation."""

    layout = PairFold128Layout() if layout is None else layout
    dimension = int(layout.carrier_dimension)
    weight = np.asarray(weight)
    if weight.shape != (dimension, dimension):
        raise ValueError(
            f"PairFold W_O must have shape {(dimension, dimension)}, "
            f"got {weight.shape}."
        )
    if np.iscomplexobj(weight) or not np.issubdtype(weight.dtype, np.floating):
        raise TypeError(f"PairFold W_O must be a real floating matrix, got {weight.dtype}.")

    low_columns, high_columns = _pairfold_output_columns(layout)
    active = tuple(int(value) for value in np.flatnonzero(low_columns >= 0))
    rows = np.arange(dimension, dtype=np.int64)
    sources = np.zeros((dimension,), dtype=np.int64)
    descriptor = structured_linear_weight(
        (weight,),
        dimension=dimension,
        real_source=sources,
        real_row=rows,
        real_scale=0.5,
        imag_source=sources,
        imag_row=rows,
        imag_scale=-0.5,
        # Keep the legacy public map meaningful for introspection while the
        # independent maps drive CPU and GPU packing.
        column=low_columns,
        real_column=low_columns,
        imag_column=high_columns,
    )
    return PairFoldOutputWeight(
        descriptor=descriptor,
        physical_width=dimension,
        active_input_channels=active,
    )


def pairfold_w_o_rotations(
    layout: PairFold128Layout | None = None,
    *,
    operator_config: LinearOperatorConfig | None = None,
    extra_rotations: tuple[int, ...] | list[int] = (),
) -> tuple[int, ...]:
    """Exact BSGS and output-conjugation key union for folded W_O."""

    layout = PairFold128Layout() if layout is None else layout
    rotations = set(
        linear_rotations(
            dimension=int(layout.carrier_dimension),
            token_lanes=int(layout.native_token_lanes),
            slots=int(layout.slots),
            include_conjugation=True,
            operator_config=(
                LLAMA3_LINEAR_SCHEDULES.w_o
                if operator_config is None
                else operator_config
            ),
        )
    )
    rotations.update(int(value) for value in extra_rotations if int(value))
    return tuple(sorted(rotations))


def pairfold_w_o_needs_refresh(
    input_ciphers: tuple[object, ...] | list[object],
    *,
    crypto_context,
    minimum_input_limbs: int = PAIRFOLD_W_O_MIN_INPUT_LIMBS,
) -> bool:
    """Return whether the graph must refresh before folded W_O.

    This predicate never mutates or consumes its inputs.  It is deliberately
    public so the eventual graph integration can preserve the existing
    conditional output-refresh policy without coupling bootstrapping into the
    linear operator.
    """

    inputs = tuple(input_ciphers)
    if not inputs:
        raise ValueError("PairFold W_O limb check requires at least one input.")
    minimum_input_limbs = int(minimum_input_limbs)
    if minimum_input_limbs <= 1:
        raise ValueError("PairFold W_O minimum_input_limbs must exceed one.")

    limbs: list[int] = []
    for cipher in inputs:
        state = getattr(cipher, "state", None)
        if state is not None and hasattr(state, "cur_limbs"):
            limbs.append(int(state.cur_limbs))
            continue
        context = getattr(crypto_context, "context", None)
        if context is None or not hasattr(context, "L"):
            raise ValueError(
                "PairFold W_O cannot determine input limbs; ciphertext.state."
                "cur_limbs or context.L is required."
            )
        limbs.append(
            int(context.L) - int(crypto_context.level_for_cipher(cipher))
        )
    return min(limbs) < minimum_input_limbs


def pairfold_w_o_fhe(
    input_ciphers: tuple[object, ...] | list[object],
    weight: np.ndarray,
    *,
    layout: PairFold128Layout | None = None,
    crypto_context,
    operator_config: LinearOperatorConfig | None = None,
    dtype: np.dtype = np.complex128,
    minimum_input_limbs: int = PAIRFOLD_W_O_MIN_INPUT_LIMBS,
    profile: dict[str, float] | None = None,
    chunk_indices: tuple[int, ...] | list[int] | None = None,
    finalize: bool = True,
) -> tuple[object]:
    """Project one borrowed folded-O carrier to one owned real DecodeFM CT.

    ``chunk_indices`` restricts the projection to those BSGS plaintext chunks
    (multi-card decode: each rank computes its chunks, the partials add up to
    the full projection -- see :meth:`LinearOperator.apply`).  ``finalize=False``
    returns the (rescaled) complex projection *without* the conjugation step,
    so the caller can all-reduce the partials first and then apply
    :func:`pairfold_w_o_finalize_fhe` once to the sum.
    """

    layout = PairFold128Layout() if layout is None else layout
    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.w_o
        if operator_config is None
        else operator_config
    )
    inputs = tuple(input_ciphers)
    if len(inputs) != 1:
        raise ValueError(
            f"PairFold W_O expects exactly one folded carrier, got {len(inputs)}."
        )
    if hasattr(crypto_context, "max_slots") and int(
        crypto_context.max_slots
    ) != int(layout.slots):
        raise ValueError(
            "PairFold W_O layout/context slot mismatch: "
            f"{layout.slots} != {crypto_context.max_slots}."
        )
    conjugation_rotation = 4 * int(layout.slots) - 1
    actual_conjugation = int(crypto_context.context.M) - 1
    if actual_conjugation != conjugation_rotation:
        raise ValueError(
            "PairFold W_O context/layout conjugation mismatch: "
            f"{actual_conjugation} != {conjugation_rotation}."
        )
    levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in inputs
    }
    if len(levels) != 1:
        raise ValueError("PairFold W_O inputs must enter at one common level.")
    if pairfold_w_o_needs_refresh(
        inputs,
        crypto_context=crypto_context,
        minimum_input_limbs=minimum_input_limbs,
    ):
        raise ValueError(
            "PairFold W_O input has too few limbs; the graph must apply its "
            "conditional output refresh before this borrowed-input operator."
        )
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.complexfloating):
        raise TypeError(f"PairFold W_O projection dtype must be complex, got {dtype}.")

    wall_started = transform_started = time.perf_counter()
    packed_weight = pairfold_w_o_weight(weight, layout=layout)
    if profile is not None:
        profile["weight_transform_cpu"] = float(
            time.perf_counter() - transform_started
        )

    operator = LinearOperator(
        crypto_context=crypto_context,
        token_lanes=int(layout.native_token_lanes),
        config=operator_config,
    )
    projected: tuple[object, ...] = ()
    conjugated = output = None
    close_attempted = False
    try:
        projection_started = time.perf_counter()
        # only forwarded when set so the default (single-card) call is unchanged
        chunk_kwargs = (
            {} if chunk_indices is None else {"chunk_indices": tuple(int(i) for i in chunk_indices)}
        )
        projected, projection_profile = operator.project(
            inputs, packed_weight.descriptor, dtype=dtype, **chunk_kwargs
        )
        if len(projected) != 1:
            raise RuntimeError(
                "PairFold W_O projection must return exactly one ciphertext, "
                f"got {len(projected)}."
            )
        sync_device(crypto_context.device)
        if profile is not None:
            profile["projection"] = float(
                time.perf_counter() - projection_started
            )
            profile.update(
                {
                    f"projection_{name}": float(value)
                    for name, value in projection_profile.items()
                }
            )

        if not finalize:
            # hand the (partial) complex projection to the caller unchanged
            if profile is not None:
                profile["output_conjugation"] = 0.0
                profile["wall"] = float(time.perf_counter() - wall_started)
            close_attempted = True
            operator.close()
            partial = projected[0]
            projected = ()
            return (partial,)

        recovery_started = time.perf_counter()
        conjugated = crypto_context.fhe.homo_rotate(
            projected[0],
            conjugation_rotation,
            crypto_context.context,
        )
        output = crypto_context.fhe.homo_add(
            projected[0], conjugated, crypto_context.context
        )
        sync_device(crypto_context.device)
        if profile is not None:
            profile["output_conjugation"] = float(
                time.perf_counter() - recovery_started
            )
            profile["wall"] = float(time.perf_counter() - wall_started)

        # Close before transferring the result.  If cache cleanup itself
        # fails, ``output`` is still locally owned and the finally block
        # releases it instead of leaking a ciphertext the caller never saw.
        close_attempted = True
        operator.close()
        result = output
        output = None
        return (result,)
    finally:
        try:
            if not close_attempted:
                operator.close()
        finally:
            release_if_supported(output)
            release_if_supported(conjugated)
            for cipher in projected:
                release_if_supported(cipher)


def pairfold_w_o_finalize_fhe(
    projected,
    *,
    layout: PairFold128Layout | None = None,
    crypto_context,
    profile: dict[str, float] | None = None,
) -> tuple[object]:
    """Conjugation step of :func:`pairfold_w_o_fhe` on a borrowed full projection.

    ``projected`` is the (all-reduced) complex W_O projection; returns one
    owned real DecodeFM ciphertext ``projected + conj(projected)``.  Pure
    ciphertext arithmetic (one rotation + one add), so the ranks of a
    multi-card decode can apply it redundantly to the identical sum.
    """

    layout = PairFold128Layout() if layout is None else layout
    conjugation_rotation = 4 * int(layout.slots) - 1
    actual_conjugation = int(crypto_context.context.M) - 1
    if actual_conjugation != conjugation_rotation:
        raise ValueError(
            "PairFold W_O context/layout conjugation mismatch: "
            f"{actual_conjugation} != {conjugation_rotation}."
        )
    started = time.perf_counter()
    conjugated = output = None
    try:
        conjugated = crypto_context.fhe.homo_rotate(
            projected, conjugation_rotation, crypto_context.context
        )
        output = crypto_context.fhe.homo_add(
            projected, conjugated, crypto_context.context
        )
        sync_device(crypto_context.device)
        if profile is not None:
            profile["output_conjugation"] = float(time.perf_counter() - started)
        result = output
        output = None
        return (result,)
    finally:
        release_if_supported(output)
        release_if_supported(conjugated)


def pairfold_w_o_chunk_count(
    operator_config: LinearOperatorConfig,
    *,
    layout: PairFold128Layout | None = None,
) -> int:
    """BSGS plaintext chunks of the W_O projection under ``operator_config``."""

    layout = PairFold128Layout() if layout is None else layout
    plan = operator_config.plan(
        dimension=int(layout.carrier_dimension),
        token_lanes=int(layout.native_token_lanes),
    )
    return int(plan.plaintext_chunks)


__all__ = [
    "PAIRFOLD_W_O_MIN_INPUT_LIMBS",
    "PairFoldOutputWeight",
    "pairfold_w_o_chunk_count",
    "pairfold_w_o_fhe",
    "pairfold_w_o_finalize_fhe",
    "pairfold_w_o_needs_refresh",
    "pairfold_w_o_rotations",
    "pairfold_w_o_weight",
]
