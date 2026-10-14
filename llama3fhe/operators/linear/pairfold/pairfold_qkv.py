from __future__ import annotations

"""Structured Decode QKV weight map for the PairFold128 carrier."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.pairfold128 import PairFold128Layout

from ..config import LLAMA3_LINEAR_SCHEDULES, LinearOperatorConfig
from ..structured_weight import StructuredLinearWeight, structured_linear_weight

if TYPE_CHECKING:
    from llama3fhe.layouts.linear import LinearCarrierLayout

    from ..prefill.qkv_kernel import ComplexTokenDiagonalQKVComplexity


@dataclass(frozen=True)
class PairFoldQKVWeight:
    """PairFold descriptor plus non-contiguous active-row metadata."""

    descriptor: StructuredLinearWeight
    physical_width: int
    active_channel_count: int
    active_channels: tuple[int, ...]
    two_sided_recovery: bool


@dataclass
class PairFoldQKVResult:
    """One owned Decode QKV carrier plus its non-contiguous row metadata."""

    carrier_cipher: object
    output_layout: "LinearCarrierLayout"
    physical_width: int
    active_channel_count: int
    active_channels: tuple[int, ...]
    complexity: "ComplexTokenDiagonalQKVComplexity"
    wall_seconds: float
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]
    max_abs_diff: float | None = None
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def carrier_ciphers(self) -> tuple[object, ...]:
        """Compatibility tuple; PairFold Decode always returns exactly one CT."""

        return (self.carrier_cipher,)

    @property
    def max_active_channel(self) -> int:
        return max(self.active_channels)

    def release(self) -> None:
        """Release the result-owned carrier exactly once."""

        if self._released:
            return
        release_if_supported(self.carrier_cipher)
        self._released = True


def pairfold_qkv_weight(
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    layout: PairFold128Layout,
    input_column_scale: np.ndarray | None = None,
    two_sided_recovery: bool = True,
) -> PairFoldQKVWeight:
    """Build the sparse-row physical weight for one Decode QKV tile.

    The first implementation reuses the existing generic QKV recovery, which
    evaluates both ``x`` and ``conj(x)`` and adds the two results.  Its
    descriptor therefore stores half the desired physical coefficient.  A
    future one-sided real-input kernel sets ``two_sided_recovery=False``.
    """

    dimension = int(layout.carrier_dimension)
    q_weight = np.asarray(q_weight)
    k_weight = np.asarray(k_weight)
    v_weight = np.asarray(v_weight)
    q_shape = (int(layout.query_heads) * int(layout.head_dim), dimension)
    kv_shape = (int(layout.key_value_heads) * int(layout.head_dim), dimension)
    if q_weight.shape != q_shape:
        raise ValueError(f"Q weight must have shape {q_shape}, got {q_weight.shape}.")
    if k_weight.shape != kv_shape or v_weight.shape != kv_shape:
        raise ValueError(
            f"K/V weights must have shape {kv_shape}, got "
            f"{k_weight.shape}/{v_weight.shape}."
        )
    source_dtypes = {q_weight.dtype, k_weight.dtype, v_weight.dtype}
    if len(source_dtypes) != 1:
        raise TypeError(
            "Q/K/V weights must use one dtype, got "
            f"{sorted(str(value) for value in source_dtypes)}."
        )
    if any(np.iscomplexobj(weight) for weight in (q_weight, k_weight, v_weight)):
        raise TypeError("Q/K/V model weights must be real.")

    column_scale = np.ones((dimension,), dtype=np.float64)
    if input_column_scale is not None:
        column_scale = np.asarray(input_column_scale, dtype=np.float64).reshape(-1)
        if column_scale.shape != (dimension,):
            raise ValueError(
                "input_column_scale must have shape "
                f"{(dimension,)}, got {column_scale.shape}."
            )
        if not np.all(np.isfinite(column_scale)):
            raise ValueError("input_column_scale must be finite.")

    real_source = np.full((dimension,), -1, dtype=np.int64)
    real_row = np.full((dimension,), -1, dtype=np.int64)
    real_scale = np.zeros((dimension,), dtype=np.float64)
    imag_source = np.full((dimension,), -1, dtype=np.int64)
    imag_row = np.full((dimension,), -1, dtype=np.int64)
    imag_scale = np.zeros((dimension,), dtype=np.float64)
    recovery_scale = 0.5 if bool(two_sided_recovery) else 1.0
    half = int(layout.fold_dim)

    def assign(
        channel: int,
        *,
        source: int,
        low_row: int,
        high_row: int,
        logical_scale: float,
        high_sign: float,
    ) -> None:
        scale = recovery_scale * float(logical_scale)
        real_source[channel] = int(source)
        real_row[channel] = int(low_row)
        real_scale[channel] = scale
        imag_source[channel] = int(source)
        imag_row[channel] = int(high_row)
        imag_scale[channel] = scale * float(high_sign)

    for group in range(int(layout.key_value_groups)):
        for pair in range(layout.pair_blocks):
            for lane in range(layout.head_lanes):
                kv_head = layout.key_value_head(group, pair, lane)
                for feature in range(half):
                    kv_low = kv_head * int(layout.head_dim) + feature
                    kv_high = kv_low + half
                    assign(
                        layout.carrier_channel(
                            pair, feature, layout.key_residue(group), lane
                        ),
                        source=1,
                        low_row=kv_low,
                        high_row=kv_high,
                        logical_scale=1.0,
                        high_sign=-1.0,
                    )
                    assign(
                        layout.carrier_channel(
                            pair, feature, layout.value_residue(group), lane
                        ),
                        source=2,
                        low_row=kv_low,
                        high_row=kv_high,
                        logical_scale=1.0,
                        high_sign=1.0,
                    )
                    for query_offset in range(layout.gqa_ratio):
                        query_head = layout.query_head(
                            group, pair, lane, query_offset
                        )
                        query_low = query_head * int(layout.head_dim) + feature
                        assign(
                            layout.carrier_channel(
                                pair,
                                feature,
                                layout.query_residue(group, query_offset),
                                lane,
                            ),
                            source=0,
                            low_row=query_low,
                            high_row=query_low + half,
                            logical_scale=0.5,
                            high_sign=1.0,
                        )

    active_channels = layout.active_carrier_channels()
    descriptor = structured_linear_weight(
        (q_weight, k_weight, v_weight),
        dimension=dimension,
        real_source=real_source,
        real_row=real_row,
        real_scale=real_scale,
        imag_source=imag_source,
        imag_row=imag_row,
        imag_scale=imag_scale,
        column=np.arange(dimension, dtype=np.int64),
        column_scale=column_scale,
    )
    return PairFoldQKVWeight(
        descriptor=descriptor,
        physical_width=dimension,
        active_channel_count=len(active_channels),
        active_channels=active_channels,
        two_sided_recovery=bool(two_sided_recovery),
    )


def _two_sided_real_input_reference(
    hidden_states: np.ndarray, descriptor: StructuredLinearWeight
) -> np.ndarray:
    """Apply ``2 * descriptor`` without materializing a 4096-square matrix."""

    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    dimension = int(descriptor.dimension)
    if hidden_states.ndim != 2 or hidden_states.shape[1] != dimension:
        raise ValueError(
            "PairFold QKV reference input must have shape [seq, "
            f"{dimension}], got {hidden_states.shape}."
        )
    output = np.zeros(
        (hidden_states.shape[0], dimension), dtype=np.complex128
    )
    for source_index, source in enumerate(descriptor.sources):
        real_rows = np.flatnonzero(descriptor.real_source == source_index)
        real_active = descriptor.real_column >= 0
        real_physical = np.flatnonzero(real_active)
        if real_rows.size and real_physical.size:
            scaled_input = (
                hidden_states[:, real_physical]
                * descriptor.column_scale[real_physical][None, :]
            )
            values = source[
                np.ix_(
                    descriptor.real_row[real_rows],
                    descriptor.real_column[real_active],
                )
            ]
            output[:, real_rows] += (
                scaled_input @ values.T
            ) * descriptor.real_scale[real_rows][None, :]
        imaginary_rows = np.flatnonzero(
            descriptor.imag_source == source_index
        )
        imag_active = descriptor.imag_column >= 0
        imag_physical = np.flatnonzero(imag_active)
        if imaginary_rows.size and imag_physical.size:
            scaled_input = (
                hidden_states[:, imag_physical]
                * descriptor.column_scale[imag_physical][None, :]
            )
            values = source[
                np.ix_(
                    descriptor.imag_row[imaginary_rows],
                    descriptor.imag_column[imag_active],
                )
            ]
            output[:, imaginary_rows] += 1j * (
                scaled_input @ values.T
            ) * descriptor.imag_scale[imaginary_rows][None, :]
    return 2.0 * output


def pairfold_qkv_rotations(
    layout: PairFold128Layout | None = None,
    *,
    operator_config: LinearOperatorConfig | None = None,
    extra_rotations: tuple[int, ...] | list[int] = (),
) -> tuple[int, ...]:
    """Rotation-key union for the two-sided PairFold QKV projection."""

    # Lazy import avoids a package-initialization cycle: qkv_kernel imports
    # the shared linear symbols exported beside this module.
    from ..prefill.qkv_kernel import complex_token_diagonal_qkv_rotations

    layout = PairFold128Layout() if layout is None else layout
    rotations = set(
        complex_token_diagonal_qkv_rotations(
            dimension=int(layout.carrier_dimension),
            slots=int(layout.slots),
            real_token_lanes=int(layout.native_token_lanes),
            operator_config=(
                LLAMA3_LINEAR_SCHEDULES.qkv
                if operator_config is None
                else operator_config
            ),
        )
    )
    rotations.update(int(value) for value in extra_rotations if int(value))
    return tuple(sorted(rotations))


def pairfold_qkv_fhe(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    layout: PairFold128Layout | None = None,
    config: Llama3CKKSConfig | None = None,
    operator_config: LinearOperatorConfig | None = None,
    extra_rotations: tuple[int, ...] | list[int] = (),
    crypto_context=None,
    input_ciphers: tuple[object, ...] | list[object] | None = None,
    verify: bool = True,
    input_column_scale: np.ndarray | None = None,
    chunk_indices: tuple[int, ...] | list[int] | None = None,
    fold: bool = True,
) -> PairFoldQKVResult:
    """Project one real DecodeFM token into one sparse PairFold carrier.

    The input ciphertext is borrowed and remains caller-owned.  The returned
    ciphertext is owned by :class:`PairFoldQKVResult`.  ``physical_width`` is
    always 4096; the 3072 active channels are deliberately exposed as an
    explicit sparse tuple and never interpreted as a contiguous prefix.

    ``chunk_indices`` (multi-card decode) selects the BSGS plaintext chunks
    this rank evaluates; the carrier is then a partial that the ranks add up
    (:mod:`llama3fhe.parallel.decode.decode_collectives`).  Default: the full carrier.

    ``operator_config.lane_shift > 0`` selects the shifted-lane (SL) one-sided
    kernel (:mod:`..shifted_lane`): the lane-0 input is expanded over the
    eight token lanes, one complex MAC with 512 diagonals replaces the
    two-sided ``(x, conj x)`` pair, and the carrier is the lane fold of the
    MAC output.  ``fold=False`` returns the *unfolded* (partial) MAC output so
    a multi-rank caller can all-reduce first and fold once
    (:func:`pairfold_qkv_fold_carrier`).  Lanes 1.. of a folded carrier hold
    cross-channel garbage; the PairFold routes read lane 0 only.
    """

    from ..prefill.qkv_kernel import _qkv_fhe_complex_token_diagonal_prepared

    layout = PairFold128Layout() if layout is None else layout
    config = Llama3CKKSConfig() if config is None else config
    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.qkv
        if operator_config is None
        else operator_config
    )
    if int(operator_config.lane_shift):
        return _pairfold_qkv_shifted_lanes_fhe(
            hidden_states,
            q_weight,
            k_weight,
            v_weight,
            layout=layout,
            config=config,
            operator_config=operator_config,
            crypto_context=crypto_context,
            input_ciphers=input_ciphers,
            verify=bool(verify),
            input_column_scale=input_column_scale,
            chunk_indices=chunk_indices,
            fold=bool(fold),
        )
    if not fold:
        raise ValueError("fold=False is only meaningful for the shifted-lane QKV kernel.")
    hidden_states = np.asarray(hidden_states)
    if np.iscomplexobj(hidden_states):
        raise TypeError("PairFold QKV hidden_states must be real.")
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    expected_hidden_shape = (1, int(layout.carrier_dimension))
    if hidden_states.shape != expected_hidden_shape:
        raise ValueError(
            "PairFold QKV is Decode-only and requires hidden_states shape "
            f"{expected_hidden_shape}, got {hidden_states.shape}."
        )
    slots = 1 << (int(config.simulator.logN) - 1)
    if slots != int(layout.slots):
        raise ValueError(
            "PairFold QKV layout/config slot mismatch: "
            f"{layout.slots} != {slots}."
        )
    if crypto_context is None or input_ciphers is None:
        raise ValueError(
            "pairfold_qkv_fhe requires an application-owned context and "
            "one encrypted DecodeFM input."
        )
    if hasattr(crypto_context, "max_slots") and int(
        crypto_context.max_slots
    ) != int(layout.slots):
        raise ValueError(
            "PairFold QKV layout/context slot mismatch: "
            f"{layout.slots} != {crypto_context.max_slots}."
        )
    input_ciphers = tuple(input_ciphers)
    if len(input_ciphers) != 1:
        raise ValueError(
            f"PairFold QKV expects exactly one DecodeFM input, got {len(input_ciphers)}."
        )

    wall_start = time.perf_counter()
    weight_transform_start = wall_start
    packed_weight = pairfold_qkv_weight(
        q_weight,
        k_weight,
        v_weight,
        layout=layout,
        input_column_scale=input_column_scale,
        two_sided_recovery=True,
    )
    carrier_reference = (
        _two_sided_real_input_reference(
            hidden_states, packed_weight.descriptor
        )
        if verify
        else None
    )
    weight_transform_seconds = time.perf_counter() - weight_transform_start

    projection = _qkv_fhe_complex_token_diagonal_prepared(
        hidden_states,
        packed_weight.descriptor,
        carrier_width=int(packed_weight.physical_width),
        carrier_weight=None,
        carrier_reference=carrier_reference,
        config=config,
        operator_config=operator_config,
        extra_rotations=extra_rotations,
        crypto_context=crypto_context,
        input_ciphers=input_ciphers,
        verify=bool(verify),
        wall_start=wall_start,
        weight_transform_seconds=weight_transform_seconds,
        chunk_indices=chunk_indices,
    )
    try:
        if len(projection.carrier_ciphers) != 1:
            raise RuntimeError(
                "PairFold Decode projection must return exactly one carrier "
                f"ciphertext, got {len(projection.carrier_ciphers)}."
            )
        if int(projection.carrier_width) != int(packed_weight.physical_width):
            raise RuntimeError(
                "PairFold projection truncated the physical carrier: "
                f"{projection.carrier_width} != {packed_weight.physical_width}."
            )
        output_layout = projection.output_layout
        actual_layout = (
            int(output_layout.seq_len),
            int(output_layout.dimension),
            int(output_layout.slots),
        )
        expected_layout = (1, int(layout.carrier_dimension), int(layout.slots))
        if actual_layout != expected_layout:
            raise RuntimeError(
                "PairFold projection returned the wrong carrier layout: "
                f"{actual_layout} != {expected_layout}."
            )
        result = PairFoldQKVResult(
            carrier_cipher=projection.carrier_ciphers[0],
            output_layout=output_layout,
            physical_width=int(packed_weight.physical_width),
            active_channel_count=int(packed_weight.active_channel_count),
            active_channels=packed_weight.active_channels,
            complexity=projection.complexity,
            wall_seconds=float(projection.wall_seconds),
            stage_seconds=dict(projection.stage_seconds),
            output_levels=projection.output_levels,
            max_abs_diff=projection.max_abs_diff,
        )
        # The generic projection result owned this handle on entry.  Detach
        # it explicitly so only the PairFold wrapper can release it after the
        # successful ownership transfer.
        projection.carrier_ciphers = ()
        return result
    except BaseException:
        projection.release()
        raise


def pairfold_qkv_sl_plan(layout: PairFold128Layout, operator_config: LinearOperatorConfig):
    """The :class:`~..config.LinearPlan` of the SL QKV kernel under ``operator_config``."""

    return operator_config.plan(
        dimension=int(layout.carrier_dimension), token_lanes=int(layout.native_token_lanes)
    )


def pairfold_qkv_fold_carrier(carrier, *, layout: PairFold128Layout, operator_config: LinearOperatorConfig, crypto_context):
    """Lane-fold an (all-reduced) SL QKV MAC output into the carrier (owned result, input borrowed)."""

    from ..shifted_lane import fold_shifted_lanes

    plan = pairfold_qkv_sl_plan(layout, operator_config)
    return fold_shifted_lanes(carrier, plan=plan, crypto_context=crypto_context)


def _pairfold_qkv_shifted_lanes_fhe(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    layout: PairFold128Layout,
    config: Llama3CKKSConfig,
    operator_config: LinearOperatorConfig,
    crypto_context,
    input_ciphers,
    verify: bool,
    input_column_scale: np.ndarray | None,
    chunk_indices,
    fold: bool,
) -> PairFoldQKVResult:
    """One-sided shifted-lane Decode QKV (see :func:`pairfold_qkv_fhe`)."""

    from llama3fhe.backend import profile_sync_device as sync_device

    from ..operator import LinearOperator
    from ..prefill.qkv_kernel import (
        ComplexTokenDiagonalLayout,
        _complex_dtype,
        _decrypt_complex_ciphers,
        complex_token_diagonal_qkv_complexity,
        unpack_complex_carrier_shards,
    )
    from ..shifted_lane import expand_lane0_to_shifted_lanes, fold_shifted_lanes

    hidden_states = np.asarray(hidden_states)
    if np.iscomplexobj(hidden_states):
        raise TypeError("PairFold QKV hidden_states must be real.")
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    dimension = int(layout.carrier_dimension)
    if hidden_states.shape != (1, dimension):
        raise ValueError(f"PairFold QKV requires hidden_states shape {(1, dimension)}, got {hidden_states.shape}.")
    slots = 1 << (int(config.simulator.logN) - 1)
    if slots != int(layout.slots):
        raise ValueError(f"PairFold QKV layout/config slot mismatch: {layout.slots} != {slots}.")
    if crypto_context is None or input_ciphers is None:
        raise ValueError("pairfold_qkv_fhe requires an application-owned context and one encrypted DecodeFM input.")
    if hasattr(crypto_context, "max_slots") and int(crypto_context.max_slots) != int(layout.slots):
        raise ValueError(f"PairFold QKV layout/context slot mismatch: {layout.slots} != {crypto_context.max_slots}.")
    input_ciphers = tuple(input_ciphers)
    if len(input_ciphers) != 1:
        raise ValueError(f"PairFold QKV expects exactly one DecodeFM input, got {len(input_ciphers)}.")
    if verify and (chunk_indices is not None or not fold):
        raise ValueError("QKV verification needs the full, folded projection.")

    wall_start = time.perf_counter()
    packed_weight = pairfold_qkv_weight(
        q_weight, k_weight, v_weight, layout=layout, input_column_scale=input_column_scale, two_sided_recovery=False
    )
    stage_seconds: dict[str, float] = {
        "context_setup": 0.0,
        "weight_transform_cpu": float(time.perf_counter() - wall_start),
    }
    plan = pairfold_qkv_sl_plan(layout, operator_config)
    output_layout = ComplexTokenDiagonalLayout(seq_len=1, dimension=dimension, slots=slots).recovered_layout
    complexity = complex_token_diagonal_qkv_complexity(
        seq_len=1,
        dimension=dimension,
        slots=slots,
        carrier_width=int(packed_weight.physical_width),
        baby_steps=int(plan.baby_steps),
        baby_anchor_step=int(plan.baby_anchor_step),
        max_plaintext_rows=int(plan.max_plaintext_rows),
        reuse_baby_rotations=True,
    )
    operator = LinearOperator(
        crypto_context=crypto_context, token_lanes=int(layout.native_token_lanes), config=operator_config
    )
    sl_input = None
    projected: tuple[object, ...] = ()
    carrier = None
    success = False
    try:
        start = time.perf_counter()
        sl_input = expand_lane0_to_shifted_lanes(input_ciphers[0], plan=plan, crypto_context=crypto_context)
        sync_device(crypto_context.device)
        stage_seconds["input_lane_expand"] = time.perf_counter() - start

        start = time.perf_counter()
        chunk_kwargs = {} if chunk_indices is None else {"chunk_indices": tuple(int(i) for i in chunk_indices)}
        projected, profile = operator.project(
            (sl_input,), packed_weight.descriptor, dtype=_complex_dtype(config), **chunk_kwargs
        )
        sync_device(crypto_context.device)
        stage_seconds["paired_projection"] = time.perf_counter() - start
        for name, seconds in profile.items():
            stage_seconds[f"paired_projection_{name}"] = float(seconds)
        if len(projected) != 1:
            raise RuntimeError(f"SL QKV projection returned {len(projected)} ciphertexts, expected 1.")
        release_if_supported(sl_input)
        sl_input = None

        start = time.perf_counter()
        if fold:
            carrier = fold_shifted_lanes(projected[0], plan=plan, crypto_context=crypto_context)
            release_if_supported(projected[0])
        else:
            carrier = projected[0]
        projected = ()
        sync_device(crypto_context.device)
        stage_seconds["token_group_recovery"] = time.perf_counter() - start

        max_abs_diff = None
        if verify:
            start = time.perf_counter()
            decrypted = _decrypt_complex_ciphers((carrier,), crypto_context=crypto_context).reshape(1, -1)
            carriers = unpack_complex_carrier_shards(
                decrypted, layout=output_layout, output_width=int(packed_weight.physical_width)
            )
            # one-sided weights hold the full coefficient; the shared reference
            # applies ``2 * descriptor`` (two-sided convention) -> halve it.
            expected = 0.5 * _two_sided_real_input_reference(hidden_states, packed_weight.descriptor)
            # the fold leaves cross-channel garbage in lanes 1..; compare lane 0 only
            max_abs_diff = float(np.max(np.abs(carriers - expected)))
            stage_seconds["verify_decrypt"] = time.perf_counter() - start

        success = True
        return PairFoldQKVResult(
            carrier_cipher=carrier,
            output_layout=output_layout,
            physical_width=int(packed_weight.physical_width),
            active_channel_count=int(packed_weight.active_channel_count),
            active_channels=packed_weight.active_channels,
            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={name: float(value) for name, value in stage_seconds.items()},
            output_levels=(int(crypto_context.level_for_cipher(carrier)),),
            max_abs_diff=max_abs_diff,
        )
    finally:
        operator.close()
        release_if_supported(sl_input)
        for cipher in projected:
            release_if_supported(cipher)
        if not success:
            release_if_supported(carrier)


__all__ = [
    "PairFoldQKVResult",
    "PairFoldQKVWeight",
    "pairfold_qkv_fhe",
    "pairfold_qkv_fold_carrier",
    "pairfold_qkv_rotations",
    "pairfold_qkv_sl_plan",
    "pairfold_qkv_weight",
]
