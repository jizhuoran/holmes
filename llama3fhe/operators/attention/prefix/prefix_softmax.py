"""Dual-source Prefill Alg2: C-packed prefix scores + C-packed content scores.

A.3 encrypts NumPy ``S_p`` / ``S_c`` (not QK).  Both partitions use the C
slot formula ``2*(feature*S+token)+lane`` with ``feature=key``, ``token=query``.
Row sums use the same ``2S`` rotation tree as Delta.  Prefix C rows are folded
to feature 0 and broadcast back before they are combined with the content
partition; both then share ``λ = 1/√Z``.  Production Prefill still owns Δ for
the CC kernel; this path never builds a 128×(k+128) ciphertext.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from llama3fhe.approx.chebyshev import chebyshev_ps_mul_depth
from llama3fhe.backend import release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
from llama3fhe.layouts.attention import AttentionPairLayout

from ...linear.complex import pair_real_ciphers, split_complex_ciphers_twice
from ...nonlinear.elementwise import eval_exp_cipher, eval_invsqrt_cipher
from ...primitives import (
    add_owned,
    bundle_plain,
    mul_cipher_rescale,
    mul_plain_rescale,
    square_rescale,
)
from ..config import AttentionOperatorConfig
from ..decode.decode import _add_packed_rows, _add_scalar_ciphers, _mul_packed_rows
from ..decode.decode_softmax import _align_cipher_pair
from ..prefill.softmax import (
    _add_public_fill,
    _fill_delta_slim_poly_input,
    _fill_mapped_delta_slim_poly_input,
    _sanitize_delta_slim_poly_input,
    _single_vector_bundle,
    _softmax_diagnostic_range,
    build_delta_softmax_mask_bundle,
    delta_broadcast_slim_rows_fhe,
    delta_causal_mask,
    delta_pack_row_sums_slim_fhe,
    delta_reduce_rows_fhe,
    delta_slim_active_mask,
    delta_softmax_row_rotations,
    extra_softmax_checkpoint_enabled,
    SLIM_MID_ALG2_LAYERS,
    SLIM_POST_ALG2_LAYERS,
    SLIM_POST_ALG2_MIN_ROUNDS,
    SLIM_POST_EXP_LAYERS,
    _half_rescale,
    _paired_checkpoint_refresh,
)


def prefix_softmax_rotations(
    layout: AttentionPairLayout, *, prefix_k: int = 0, fused: bool | None = None
) -> tuple[int, ...]:
    # Reduction uses the positive tree; Prefix's C-form row sum also needs the
    # inverse tree to broadcast feature 0 back across all rows.
    steps = delta_softmax_row_rotations(layout)
    rotations = set(int(step) for step in steps)
    rotations.update(-int(step) for step in steps)
    # Partitioned Prefix may use the paired real/imaginary bootstrap path.
    # Its split operation needs the CKKS conjugation rotation even when this
    # helper is used standalone (the full model also obtains the same key from
    # the attention-input rotation union).
    rotations.add(4 * int(layout.slots) - 1)
    stride = 2 * int(layout.seq_len)
    # The partitioned slim pack moves pair p's feature-0 sum onto row p.
    rotations.update(
        -pair_index * stride
        for pair_index in range(1, int(layout.query_pair_count))
    )
    prefix_k = int(prefix_k)
    if prefix_k:
        if not 1 <= prefix_k <= int(layout.head_dim):
            raise ValueError(f"prefix_k={prefix_k} outside [1, {layout.head_dim}].")
        if int(layout.query_pair_count) * prefix_k <= int(layout.head_dim):
            # Compact pack/expand and compact lambda extraction move pair p's
            # k-row block between row 0 and row p*k.  Slim packing maps the
            # source row p*k to destination row p.
            for pair_index in range(1, int(layout.query_pair_count)):
                block_shift = pair_index * prefix_k * stride
                rotations.add(block_shift)
                rotations.add(-block_shift)
                rotations.add((pair_index * prefix_k - pair_index) * stride)
        rotations.update(
            prefix_index * stride for prefix_index in range(1, prefix_k)
        )
        if fused is None:
            fused = _prefix_softmax_pipeline() == "fused"
        if fused:
            from .fused_delta import fused_prefix_rotations

            if prefix_k != 2:
                raise ValueError("fused Prefix rotations require prefix_k=2.")
            rotations.update(fused_prefix_rotations(layout))
    return tuple(sorted(rotation for rotation in rotations if rotation))


def _release_tuple(ciphers) -> None:
    for cipher in tuple(ciphers or ()):
        release_if_supported(cipher)


def _diagnostic_state(label: str, ciphers) -> None:
    """Print CKKS state only for an explicitly requested diagnostic run."""

    if os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS", "") != "1":
        return
    states = tuple(ciphers or ())
    if not states:
        print(f"[softmax-state] stage={label} empty=True", flush=True)
        return
    unique = sorted(
        {
            (
                int(cipher.state.cur_limbs),
                int(cipher.state.scale_degree),
                None
                if cipher.state.scaling_factor is None
                else float(cipher.state.scaling_factor),
            )
            for cipher in states
        }
    )
    print(
        f"[softmax-state] stage={label} count={len(states)} states={unique}",
        flush=True,
    )


def _add_cipher_pairs(left, right, *, crypto_context) -> tuple[object, ...]:
    outputs: list[object] = []
    owned: list[object] = []
    try:
        for a, b in zip(left, right, strict=True):
            aligned_a, aligned_b, owned_a, owned_b = _align_cipher_pair(
                a, b, crypto_context=crypto_context
            )
            if owned_a:
                owned.append(aligned_a)
            if owned_b:
                owned.append(aligned_b)
            outputs.append(
                crypto_context.fhe.homo_add(
                    aligned_a, aligned_b, crypto_context.context
                )
            )
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        _release_tuple(outputs)
        _release_tuple(owned)


def _align_partition_streams(
    left: tuple[object, ...],
    right: tuple[object, ...],
    *,
    crypto_context,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Put Prefix and Content streams on one CKKS state before polynomials.

    The plaintext-K Prefix QK path and the ciphertext Delta QK path do not
    consume exactly the same number of rescale levels.  Feeding their mapped
    scores into separate Chebyshev evaluators leaves the two probability
    partitions at different noise/scale states; the later ``Z_p + Z_c`` add
    then relies on implicit alignment after several nonlinear rounds.  Align
    both streams immediately after the public map, when both values still
    have the same semantic scale.  Compact Prefix has one cipher for all
    query pairs, so its stream need not match Content's cipher count.
    """

    if not left or not right:
        raise ValueError("Prefix/Content state alignment requires both streams.")
    aligned_left: list[object] = []
    aligned_right: list[object] = []
    owned_left: list[object] = []
    owned_right: list[object] = []
    try:
        target = min(
            left + right,
            key=lambda cipher: (
                int(cipher.state.cur_limbs),
                -int(cipher.state.scale_degree),
            ),
        ).state
        target_scale = target.scaling_factor
        for sources, outputs, owned in (
            (left, aligned_left, owned_left),
            (right, aligned_right, owned_right),
        ):
            for cipher in sources:
                source_scale = cipher.state.scaling_factor
                same_scale = (
                    source_scale is None and target_scale is None
                ) or (
                    source_scale is not None
                    and target_scale is not None
                    and math.isclose(
                        float(source_scale), float(target_scale), rel_tol=1e-12
                    )
                )
                if (
                    int(cipher.state.cur_limbs) == int(target.cur_limbs)
                    and int(cipher.state.scale_degree) == int(target.scale_degree)
                    and same_scale
                ):
                    outputs.append(cipher)
                else:
                    aligned = crypto_context.fhe.align_to(
                        cipher, target, crypto_context.context
                    )
                    outputs.append(aligned)
                    owned.append(aligned)
        return tuple(aligned_left), tuple(aligned_right)
    except Exception:
        _release_tuple(owned_left)
        _release_tuple(owned_right)
        raise


def _refresh_partition(
    ciphers: tuple[object, ...],
    *,
    bootstrap_operator,
    pair: bool,
    iterations: int | None,
    precision_bits: int,
    match_two_pass_level: bool,
    use_deep_program: bool,
    crypto_context,
    ensure_two_pass_headroom: bool = False,
) -> tuple[object, ...]:
    """Refresh one score partition, optionally as paired complex streams."""

    refreshed: list[object] = []
    paired: tuple[object, ...] = ()
    try:
        checkpoint_inputs = ciphers
        if pair:
            paired = pair_real_ciphers(ciphers, crypto_context=crypto_context)
            checkpoint_inputs = paired
        refresh_many = getattr(bootstrap_operator, "refresh_many", None)
        if refresh_many is not None:
            # whole partition in one call: batched C2S / EvalMod / S2C (both
            # Meta-BTS passes) when the operator has a batch backend
            # (HOLMES_BATCH_BS=1), the plain per-cipher loop otherwise
            sources: list[object] = []
            try:
                for cipher in checkpoint_inputs:
                    sources.append(
                        bootstrap_operator.prepare_two_pass_input(
                            cipher,
                            iterations=iterations,
                            use_deep_program=use_deep_program,
                        )
                        if ensure_two_pass_headroom
                        else cipher
                    )
                refreshed.extend(
                    refresh_many(
                        sources,
                        iterations=iterations,
                        precision_bits=int(precision_bits),
                        match_two_pass_level=bool(match_two_pass_level),
                        use_deep_program=bool(use_deep_program),
                    )
                )
            finally:
                for source, cipher in zip(sources, checkpoint_inputs):
                    if source is not cipher:
                        release_if_supported(source)
            checkpoint_inputs = ()
        for cipher in checkpoint_inputs:
            source = (
                bootstrap_operator.prepare_two_pass_input(
                    cipher,
                    iterations=iterations,
                    use_deep_program=use_deep_program,
                )
                if ensure_two_pass_headroom
                else cipher
            )
            try:
                refreshed.append(
                    bootstrap_operator.refresh(
                        source,
                        iterations=iterations,
                        precision_bits=int(precision_bits),
                        match_two_pass_level=bool(match_two_pass_level),
                        use_deep_program=bool(use_deep_program),
                    )
                )
            finally:
                if source is not cipher:
                    release_if_supported(source)
    except Exception:
        _release_tuple(refreshed)
        raise
    finally:
        _release_tuple(paired)
    _release_tuple(ciphers)
    if pair:
        try:
            return split_complex_ciphers_twice(
                tuple(refreshed), crypto_context=crypto_context
            )
        finally:
            _release_tuple(refreshed)
    return tuple(refreshed)


def _refresh_lambda_with_compact_prefix_probability(
    lambda_cipher,
    prefix_probability,
    *,
    prefix_mask: np.ndarray,
    prefix_gain: np.ndarray,
    bootstrap_operator,
    iterations: int | None,
    match_two_pass_level: bool,
    crypto_context,
):
    """Refresh final-round lambda and compact Prefix probability in one BS.

    The native Delta schedule already refreshes ``lambda`` once per Alg2
    round.  On the final round the compact Prefix probability stream would
    otherwise finish too low for PC PV.  Multiplex the two real ciphertexts
    as real/imaginary components, run the existing lambda bootstrap once,
    split them, and undo the split's factor of two.  Refresh ``y`` rather than
    ``y**2``: after several Alg2 rounds the latter is small enough that
    bootstrap's absolute error is amplified badly by the row scale.  The
    caller squares the refreshed probability before the final update.  This
    retains the original bootstrap call count.

    Compact Prefix only occupies a small row block.  Sending that sparse
    spike field straight through bootstrap rings badly in its inactive rows,
    so fill those public-zero rows with an interior constant before pairing
    and restore the exact mask after splitting.  Neither input is consumed.
    """

    conditioned_prefix: tuple[object, ...] = ()
    filled_prefix: tuple[object, ...] = ()
    aligned_lambda = aligned_prefix = None
    own_lambda = own_prefix = False
    paired: tuple[object, ...] = ()
    refreshed = split = None
    half_lambda = None
    remasked_prefix: tuple[object, ...] = ()
    try:
        conditioned_prefix = _mul_packed_rows(
            (prefix_probability,),
            prefix_gain,
            name="prefix.alg2.final_probability.condition",
            crypto_context=crypto_context,
        )
        prefix_fill = (
            1.0 - np.asarray(prefix_mask, dtype=np.float64)
        ) * 0.25
        filled_prefix = _add_packed_rows(
            conditioned_prefix,
            prefix_fill,
            name="prefix.alg2.final_square.fill",
            crypto_context=crypto_context,
        )
        (
            aligned_lambda,
            aligned_prefix,
            own_lambda,
            own_prefix,
        ) = _align_cipher_pair(
            lambda_cipher,
            filled_prefix[0],
            crypto_context=crypto_context,
        )
        paired = pair_real_ciphers(
            (aligned_lambda, aligned_prefix),
            crypto_context=crypto_context,
            pad_odd=False,
        )
        refreshed = bootstrap_operator.refresh(
            paired[0],
            iterations=iterations,
            match_two_pass_level=bool(match_two_pass_level),
        )
        split = split_complex_ciphers_twice(
            (refreshed,), crypto_context=crypto_context
        )
        half_lambda = _half_rescale(split[0], crypto_context=crypto_context)
        inverse_gain = np.divide(
            0.5 * np.asarray(prefix_mask, dtype=np.float64),
            np.asarray(prefix_gain, dtype=np.float64),
            out=np.zeros_like(prefix_mask, dtype=np.float64),
            where=np.asarray(prefix_gain) != 0.0,
        )
        remasked_prefix = _mul_packed_rows(
            (split[1],),
            inverse_gain,
            name="prefix.alg2.final_square.post_bs.mask",
            crypto_context=crypto_context,
        )
        result = (half_lambda, remasked_prefix[0])
        half_lambda = None
        remasked_prefix = ()
        return result
    finally:
        _release_tuple(conditioned_prefix)
        _release_tuple(filled_prefix)
        _release_tuple(paired)
        release_if_supported(refreshed)
        _release_tuple(split)
        release_if_supported(half_lambda)
        _release_tuple(remasked_prefix)
        if own_lambda:
            release_if_supported(aligned_lambda)
        if own_prefix:
            release_if_supported(aligned_prefix)


def _min_limbs(ciphers: tuple[object, ...]) -> int:
    return min(int(cipher.state.cur_limbs) for cipher in ciphers)


def _prefix_round_refresh_policy() -> str:
    """Return the opt-in Prefix main-track checkpoint policy.

    ``native`` follows the original Delta Softmax invariant: Prefix and
    content are masked once immediately after exp, and the later square / row
    scalar updates preserve those exact zero slots without another PT-CT
    multiplication.  This keeps the main stream on the original checkpoint
    schedule.  ``always`` preserves the landed end-to-end baseline.  ``adaptive`` only
    checkpoints when the two live probability partitions cannot cover the
    remaining Alg2 updates and PairFM's six-limb input floor.  ``off`` is a
    diagnostic mode used to establish the actual depth boundary; it is not a
    production recommendation for high-k layers.
    """

    policy = os.environ.get(
        "LLAMA_FHE_PREFIX_ROUND_REFRESH", "native"
    ).strip().lower()
    if policy not in {"native", "always", "adaptive", "off"}:
        raise ValueError(
            "LLAMA_FHE_PREFIX_ROUND_REFRESH must be one of "
            "'native', 'always', 'adaptive', or 'off'."
        )
    return policy


def _prefix_partition_packing() -> str:
    packing = os.environ.get(
        "LLAMA_FHE_PREFIX_PACKING", "auto"
    ).strip().lower()
    if packing not in {"auto", "partitioned", "compact"}:
        raise ValueError(
            "LLAMA_FHE_PREFIX_PACKING must be 'auto', 'partitioned', or 'compact'."
        )
    return packing


def _resolve_prefix_partition_packing(
    layout: AttentionPairLayout, prefix_k: int
) -> str:
    packing = _prefix_partition_packing()
    if packing == "auto":
        return (
            "compact"
            if int(layout.query_pair_count) * int(prefix_k) <= int(layout.head_dim)
            else "partitioned"
        )
    if packing == "compact":
        layout.prefix_compact_row_count(int(prefix_k))
    return packing


def _prefix_softmax_pipeline() -> str:
    pipeline = os.environ.get(
        "LLAMA_FHE_PREFIX_PIPELINE", "dual"
    ).strip().lower()
    if pipeline not in {"dual", "pages", "fused"}:
        raise ValueError(
            "LLAMA_FHE_PREFIX_PIPELINE must be 'dual', 'pages', or 'fused'."
        )
    return pipeline


def _prefix_remaining_limb_floor(remaining_rounds: int) -> int:
    """Conservative limbs needed after one completed dual Alg2 round.

    Each remaining round consumes one level for ``y**2``, one for the
    multiplication by ``lambda**2``, and one for the public partition mask.
    Six limbs are retained at the probability boundary so the existing
    PairFM policy does not need a new emergency refresh.
    """

    return 3 * max(0, int(remaining_rounds)) + 6


def _prefix_reduce_rows_fhe(
    ciphers: tuple[object, ...],
    *,
    prefix_k: int,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Sum the k C-form prefix features and broadcast like Delta reduce.

    ``delta_reduce_rows_fhe`` sums all 128 feature rows.  Only ``prefix_k`` of
    those hold Prefix scores; the rest are remasked zeros whose square-noise
    dominates later-round Σy² (~0.008).  Fold the k occupied rows onto
    feature 0, remask, then reuse the Delta butterfly to broadcast.
    """

    prefix_k = int(prefix_k)
    if prefix_k < 1:
        raise ValueError(f"prefix_k must be >= 1, got {prefix_k}.")
    stride = 2 * int(layout.seq_len)
    folded: list[object] = []
    try:
        for cipher in ciphers:
            current = cipher.deep_copy()
            # Prefix rows are contiguous and all rows >= prefix_k are already
            # zero after the public mask.  A binary rotate/add tree therefore
            # folds k rows in ceil(log2(k)) rotations; the old linear loop
            # paid one rotation per Prefix token.
            step = 1
            while step < prefix_k:
                rotated = crypto_context.fhe.homo_rotate(
                    current, int(step * stride), crypto_context.context
                )
                try:
                    next_current = crypto_context.fhe.homo_add(
                        current, rotated, crypto_context.context
                    )
                finally:
                    release_if_supported(current)
                    release_if_supported(rotated)
                current = next_current
                step <<= 1
            folded.append(current)
        masked = _mul_packed_rows(
            tuple(folded),
            layout.c_feature_row_mask(0),
            name="prefix.reduce.f0",
            crypto_context=crypto_context,
        )
        _release_tuple(tuple(folded))
        return masked
    except Exception:
        _release_tuple(tuple(folded))
        raise


def _compact_prefix_mask(
    layout: AttentionPairLayout,
    *,
    prefix_k: int,
    pair_index: int | None = None,
    dtype=np.float64,
) -> np.ndarray:
    """Mask the live rows of one compact Prefix partition ciphertext."""

    prefix_k = int(prefix_k)
    row_count = layout.prefix_compact_row_count(prefix_k)
    base = np.zeros(
        (int(layout.head_dim), int(layout.seq_len), 2), dtype=dtype
    )
    if pair_index is None:
        base[:row_count, :, :] = 1.0
    else:
        pair_index = int(pair_index)
        if not 0 <= pair_index < int(layout.query_pair_count):
            raise ValueError(
                f"pair_index={pair_index} outside the query-pair range."
            )
        start = pair_index * prefix_k
        base[start : start + prefix_k, :, :] = 1.0
    row = np.tile(base.reshape(-1), int(layout.payload_repetitions))
    return row.reshape(1, -1)


def _compact_prefix_bootstrap_gain(
    layout: AttentionPairLayout,
    *,
    prefix_k: int,
    dtype=np.float64,
) -> np.ndarray:
    """Public per-query conditioning for the shared final Prefix bootstrap.

    After each Alg2 normalization, a typical probability at query ``q`` is
    proportional to ``1 / (q + prefix_k)``.  Multiplying by the square-root
    of that public row width keeps late-query values above bootstrap's
    absolute error floor without pushing the large first rows out of range.
    The inverse is fused into the post-bootstrap mask.
    """

    prefix_k = int(prefix_k)
    row_count = layout.prefix_compact_row_count(prefix_k)
    base = np.zeros(
        (int(layout.head_dim), int(layout.seq_len), 2), dtype=dtype
    )
    gain = np.sqrt(
        2.0
        * (np.arange(int(layout.seq_len), dtype=dtype) + prefix_k)
        / float(prefix_k)
    )
    base[:row_count, :, :] = gain[None, :, None]
    row = np.tile(base.reshape(-1), int(layout.payload_repetitions))
    return row.reshape(1, -1)


def _compact_prefix_partitions_fhe(
    ciphers: tuple[object, ...],
    *,
    prefix_k: int,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Multiplex 16 sparse Prefix pair streams into one C-grid stream."""

    ciphers = tuple(ciphers)
    if len(ciphers) != int(layout.query_pair_count):
        raise ValueError(
            "compact Prefix packing expects "
            f"{layout.query_pair_count} pair ciphers, got {len(ciphers)}."
        )
    layout.prefix_compact_row_count(prefix_k)
    stride = 2 * int(layout.seq_len)
    accumulator = None
    try:
        for pair_index, cipher in enumerate(ciphers):
            base_row = int(pair_index) * int(prefix_k)
            term = (
                cipher.deep_copy()
                if base_row == 0
                else crypto_context.fhe.homo_rotate(
                    cipher,
                    -base_row * stride,
                    crypto_context.context,
                )
            )
            accumulator = (
                term
                if accumulator is None
                else add_owned(accumulator, term, crypto_context=crypto_context)
            )
        if accumulator is None:
            raise RuntimeError("compact Prefix packing produced no output.")
        result = (accumulator,)
        accumulator = None
        return result
    finally:
        release_if_supported(accumulator)


def _expand_compact_prefix_partitions_fhe(
    ciphers: tuple[object, ...],
    *,
    prefix_k: int,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Expose each compact pair block at rows ``0..k-1`` for Prefix PV.

    Other compact blocks remain in each rotated output, but Prefix PV masks
    rows ``0..k-1`` before using them, so no extra PT-CT mask or level is
    needed here.
    """

    ciphers = tuple(ciphers)
    if len(ciphers) != 1:
        raise ValueError(
            f"compact Prefix expansion expects one cipher, got {len(ciphers)}."
        )
    layout.prefix_compact_row_count(prefix_k)
    stride = 2 * int(layout.seq_len)
    source = ciphers[0]
    outputs: list[object] = []
    try:
        for pair_index in range(int(layout.query_pair_count)):
            base_row = pair_index * int(prefix_k)
            outputs.append(
                source.deep_copy()
                if base_row == 0
                else crypto_context.fhe.homo_rotate(
                    source, base_row * stride, crypto_context.context
                )
            )
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        _release_tuple(outputs)


def _compact_prefix_reduce_rows_fhe(
    ciphers: tuple[object, ...],
    *,
    prefix_k: int,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Reduce each compact pair's ``k`` rows onto its block's first row."""

    ciphers = tuple(ciphers)
    if len(ciphers) != 1:
        raise ValueError(
            f"compact Prefix reduction expects one cipher, got {len(ciphers)}."
        )
    stride = 2 * int(layout.seq_len)
    base_rows = np.zeros(
        (int(layout.head_dim), int(layout.seq_len), 2), dtype=np.float64
    )
    for pair_index in range(int(layout.query_pair_count)):
        base_rows[pair_index * int(prefix_k), :, :] = 1.0
    base_mask = np.tile(
        base_rows.reshape(-1), int(layout.payload_repetitions)
    ).reshape(1, -1)
    accumulator = None
    try:
        accumulator = ciphers[0].deep_copy()
        # Read every row from the original stream.  Rotating an accumulated
        # partial sum would cross-contaminate adjacent k-row blocks when k is
        # not a power of two.  Mask once after the additions instead of paying
        # k plaintext multiplications.
        for prefix_j in range(1, int(prefix_k)):
            shifted = crypto_context.fhe.homo_rotate(
                ciphers[0], prefix_j * stride, crypto_context.context
            )
            accumulator = add_owned(
                accumulator, shifted, crypto_context=crypto_context
            )
        if accumulator is None:
            raise RuntimeError("compact Prefix reduction produced no output.")
        result = _mul_packed_rows(
            (accumulator,),
            base_mask,
            name="prefix.compact.reduce.base_rows",
            crypto_context=crypto_context,
        )
        release_if_supported(accumulator)
        accumulator = None
        return result
    finally:
        release_if_supported(accumulator)


def _compact_prefix_pack_row_sums_slim_fhe(
    reduced_ciphers: tuple[object, ...],
    *,
    prefix_k: int,
    layout: AttentionPairLayout,
    crypto_context,
    masks,
    mask_name_prefix: str,
):
    """Place compact per-pair Prefix sums in the standard slim carrier."""

    reduced_ciphers = tuple(reduced_ciphers)
    if len(reduced_ciphers) != 1:
        raise ValueError(
            "compact Prefix slim packing expects one reduced ciphertext."
        )
    stride = 2 * int(layout.seq_len)
    accumulator = None
    try:
        for pair_index in range(int(layout.query_pair_count)):
            source_row = pair_index * int(prefix_k)
            destination_row = pair_index
            rotation = (source_row - destination_row) * stride
            source = (
                reduced_ciphers[0].deep_copy()
                if rotation == 0
                else crypto_context.fhe.homo_rotate(
                    reduced_ciphers[0], rotation, crypto_context.context
                )
            )
            try:
                plain = bundle_plain(
                    masks,
                    f"{mask_name_prefix}.pair.{pair_index}",
                    source,
                    crypto_context=crypto_context,
                )
                try:
                    selected = mul_plain_rescale(
                        source, plain, crypto_context=crypto_context
                    )
                finally:
                    release_if_supported(plain)
            finally:
                release_if_supported(source)
            accumulator = (
                selected
                if accumulator is None
                else add_owned(
                    accumulator, selected, crypto_context=crypto_context
                )
            )
        if accumulator is None:
            raise RuntimeError("compact Prefix slim packing produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(accumulator)


def _compact_prefix_lambda_pairs_fhe(
    lambda_pairs: tuple[object, ...],
    *,
    prefix_k: int,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Multiplex dense per-pair lambda broadcasts into compact Prefix blocks."""

    lambda_pairs = tuple(lambda_pairs)
    if len(lambda_pairs) != int(layout.query_pair_count):
        raise ValueError(
            f"expected {layout.query_pair_count} lambda pairs, "
            f"got {len(lambda_pairs)}."
        )
    stride = 2 * int(layout.seq_len)
    source_mask = layout.c_prefix_score_mask(int(prefix_k))[:1]
    accumulator = None
    try:
        for pair_index, cipher in enumerate(lambda_pairs):
            selected = _mul_packed_rows(
                (cipher,),
                source_mask,
                name=f"prefix.compact.lambda.p{pair_index}",
                crypto_context=crypto_context,
            )[0]
            base_row = pair_index * int(prefix_k)
            if base_row:
                shifted = crypto_context.fhe.homo_rotate(
                    selected, -base_row * stride, crypto_context.context
                )
                release_if_supported(selected)
                selected = shifted
            accumulator = (
                selected
                if accumulator is None
                else add_owned(
                    accumulator, selected, crypto_context=crypto_context
                )
            )
        if accumulator is None:
            raise RuntimeError("compact Prefix lambda packing produced no output.")
        result = (accumulator,)
        accumulator = None
        return result
    finally:
        release_if_supported(accumulator)


def _compact_prefix_broadcast_rows_fhe(
    reduced_ciphers: tuple[object, ...],
    *,
    prefix_k: int,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Expand compact Prefix sums and broadcast each pair's scalar."""

    expanded = _expand_compact_prefix_partitions_fhe(
        reduced_ciphers,
        prefix_k=prefix_k,
        layout=layout,
        crypto_context=crypto_context,
    )
    isolated = None
    try:
        isolated = _mul_packed_rows(
            expanded,
            layout.c_feature_row_mask(0),
            name="prefix.compact.broadcast.row0",
            crypto_context=crypto_context,
        )
        return _prefix_broadcast_rows_fhe(
            isolated, layout=layout, crypto_context=crypto_context
        )
    finally:
        _release_tuple(expanded)
        _release_tuple(isolated)


def _prefix_pack_row_sums_slim_fhe(
    reduced_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    masks=None,
    mask_name_prefix: str = "prefix.slim",
):
    """Pack feature-0 Prefix row sums into the slim pair slices.

    Each pair cipher holds Σy_p² on C-form feature 0.  Rotate that row onto
    feature ``pair_index`` so the standard slim mask can select it without a
    128-feature butterfly.
    """

    if len(reduced_ciphers) != int(layout.query_pair_count):
        raise ValueError(
            f"prefix slim pack expects {layout.query_pair_count} ciphers, "
            f"got {len(reduced_ciphers)}."
        )
    bundle = (
        build_delta_softmax_mask_bundle(
            layout,
            crypto_context=crypto_context,
            name_prefix=mask_name_prefix,
        )
        if masks is None
        else masks
    )
    stride = 2 * int(layout.seq_len)
    accumulator = None
    try:
        for pair_index, cipher in enumerate(reduced_ciphers):
            source = cipher
            owned = False
            if pair_index:
                source = crypto_context.fhe.homo_rotate(
                    cipher, -int(pair_index) * stride, crypto_context.context
                )
                owned = True
            try:
                plain = bundle_plain(
                    bundle,
                    f"{mask_name_prefix}.pair.{pair_index}",
                    source,
                    crypto_context=crypto_context,
                )
                try:
                    selected = mul_plain_rescale(
                        source, plain, crypto_context=crypto_context
                    )
                finally:
                    release_if_supported(plain)
            finally:
                if owned:
                    release_if_supported(source)
            accumulator = (
                selected
                if accumulator is None
                else add_owned(accumulator, selected, crypto_context=crypto_context)
            )
        if accumulator is None:
            raise RuntimeError("prefix slim pack produced no output.")
        return accumulator
    except Exception:
        if accumulator is not None:
            release_if_supported(accumulator)
        raise


def _prefix_broadcast_rows_fhe(
    reduced_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Broadcast a Prefix row sum from C feature 0 to every feature row.

    ``_prefix_reduce_rows_fhe`` deliberately leaves the sum isolated on
    feature 0.  The regular Delta reducer is a *fold* (it puts a dense input's
    sum on row 0), so using it here would leave the Prefix contribution absent
    from rows 1..D-1.  Alg2 multiplies each occupied Prefix row by the
    corresponding lambda slot; all rows therefore need the same ``Z_prefix``.
    """

    if len(reduced_ciphers) != int(layout.query_pair_count):
        raise ValueError(
            f"prefix broadcast expects {layout.query_pair_count} ciphers, "
            f"got {len(reduced_ciphers)}."
        )
    steps = delta_softmax_row_rotations(layout)
    outputs: list[object] = []
    try:
        for cipher in reduced_ciphers:
            current = cipher.deep_copy()
            for rotation in steps:
                rotated = crypto_context.fhe.homo_rotate(
                    current, -int(rotation), crypto_context.context
                )
                try:
                    next_current = crypto_context.fhe.homo_add(
                        current, rotated, crypto_context.context
                    )
                finally:
                    release_if_supported(current)
                    release_if_supported(rotated)
                current = next_current
            outputs.append(current)
        return tuple(outputs)
    except Exception:
        _release_tuple(outputs)
        raise


def softmax_alg2_prefix_partitions_fhe(
    prefix_score_ciphers: tuple[object, ...] | list[object],
    content_score_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    prefix_k: int,
    layer_idx: int = 0,
    rounds: int | None = None,
    score_scale: float | None = None,
    content_form: str = "c",
    token_valid_mask: np.ndarray | None = None,
    bootstrap_operator=None,
    refresh_before_exp: bool | None = None,
    refresh_after_exp: bool | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    stage_seconds: dict[str, float] | None = None,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Alg2 with shared ``Z = Z_p + Z_c``.  Consumes neither input tuple.

    ``content_form='c'`` is the A.3 demo packing (feature=key).  Production
    CC scores are Delta; pass ``content_form='delta'`` so the causal mask
    matches ``S_Δ``.  After row-reduce both layouts share the query axis,
    so ``Z_p + Z_c`` is a same-slot add.

    Content Softmax follows the original CC program (pre-exp / extra
    checkpoints).  Prefix scores are ``Q_ct × K_pt``, so they inherit Q's
    remaining limbs rather than a fresh encrypt; Prefix is bootstrapped
    before exp only when those limbs cannot finish the polynomial.  Extra
    post-Alg2 checkpoints stay on content (PairFM / W_O).  L31 post-exp and
    mid-Alg2 also refresh Prefix because both partitions run the same k=5
    remaining depth.  Unit tests omit the operator and keep the exact
    dual-source path that does not need slim-pack keys.
    """

    function_started = time.perf_counter()
    pipeline = _prefix_softmax_pipeline()
    if pipeline == "fused":
        from .fused_delta import softmax_alg2_prefix_fused_fhe

        return softmax_alg2_prefix_fused_fhe(
            prefix_score_ciphers,
            content_score_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            config=config,
            prefix_k=prefix_k,
            layer_idx=layer_idx,
            rounds=rounds,
            score_scale=score_scale,
            content_form=content_form,
            token_valid_mask=token_valid_mask,
            bootstrap_operator=bootstrap_operator,
            refresh_before_exp=refresh_before_exp,
            refresh_after_exp=refresh_after_exp,
            operator_config=operator_config,
            stage_seconds=stage_seconds,
        )
    if pipeline == "pages":
        return softmax_alg2_prefix_pages_fhe(
            prefix_score_ciphers,
            content_score_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            config=config,
            prefix_k=prefix_k,
            layer_idx=layer_idx,
            rounds=rounds,
            score_scale=score_scale,
            content_form=content_form,
            token_valid_mask=token_valid_mask,
            bootstrap_operator=bootstrap_operator,
            refresh_before_exp=refresh_before_exp,
            refresh_after_exp=refresh_after_exp,
            operator_config=operator_config,
            stage_seconds=stage_seconds,
        )

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    if refresh_before_exp is None:
        refresh_before_exp = operator_config.refresh_softmax_exp_inputs
    if refresh_after_exp is None:
        refresh_after_exp = operator_config.refresh_softmax_exp_outputs
    refresh_before_exp = bool(refresh_before_exp)
    refresh_after_exp = bool(refresh_after_exp)
    checkpointed = bootstrap_operator is not None
    pair_pre_exp_bootstrap = bool(
        operator_config.pair_softmax_pre_exp_bootstrap and refresh_before_exp
    )
    reuse_round_square = bool(
        operator_config.reuse_softmax_round_square and checkpointed
    )
    fuse_slim_chebyshev_map = bool(
        operator_config.fuse_softmax_slim_chebyshev_map
    )
    direct_slim_inactive_fill = bool(
        operator_config.direct_softmax_slim_inactive_fill
    )
    main_bootstrap_use_deep = bool(operator_config.softmax_main_deep_bootstrap)
    main_bootstrap_iterations = operator_config.softmax_main_bootstrap_iterations
    lambda_bootstrap_iterations = (
        operator_config.softmax_lambda_bootstrap_iterations
    )
    match_single_pass_bootstrap_level = bool(
        operator_config.match_softmax_single_pass_bootstrap_level
    )
    checkpoint_precision_bits = int(
        operator_config.checkpoint_bootstrap_precision_bits
    )
    round_refresh_policy = _prefix_round_refresh_policy()
    prefix_packing = _resolve_prefix_partition_packing(layout, prefix_k)
    if round_refresh_policy == "native" and prefix_packing != "compact":
        # One compact real stream can share the final lambda bootstrap as its
        # imaginary component.  A partitioned Prefix has 16 real streams and
        # cannot use that zero-extra-call checkpoint; preserve its validated
        # conservative schedule.
        round_refresh_policy = "always"
    timings = {} if stage_seconds is None else stage_seconds

    prefix_score_ciphers = tuple(prefix_score_ciphers)
    content_score_ciphers = tuple(content_score_ciphers)
    n_pair = int(layout.query_pair_count)
    if len(prefix_score_ciphers) != n_pair or len(content_score_ciphers) != n_pair:
        raise ValueError(
            f"expected {n_pair} prefix and content score ciphers, "
            f"got {len(prefix_score_ciphers)} and {len(content_score_ciphers)}."
        )
    prefix_k = int(prefix_k)
    compact_prefix_scores: tuple[object, ...] = ()
    working_prefix_scores = prefix_score_ciphers
    if prefix_packing == "compact":
        compact_prefix_scores = _compact_prefix_partitions_fhe(
            prefix_score_ciphers,
            prefix_k=prefix_k,
            layout=layout,
            crypto_context=crypto_context,
        )
        working_prefix_scores = compact_prefix_scores
    k = int(config.k_for(layer_idx) if rounds is None else rounds)
    if k < 1:
        raise ValueError(f"Alg2 k must be >= 1, got {k}.")
    alg2_rounds = int(k)
    score_scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    prefix_mask = (
        _compact_prefix_mask(layout, prefix_k=prefix_k)
        if prefix_packing == "compact"
        else layout.c_prefix_score_mask(prefix_k)
    )
    prefix_bootstrap_gain = (
        _compact_prefix_bootstrap_gain(layout, prefix_k=prefix_k)
        if prefix_packing == "compact"
        else None
    )
    # Prefix scores are valid for *every* content query.  An earlier
    # experimental path kept only the last query after round one (the
    # ``last_prefix_mask`` layout helper still exists for compatibility), but
    # that silently corrupts all earlier rows and, consequently, the K/V
    # snapshots emitted for the next decode step.  Keep the full prefix mask
    # throughout Alg2 just like the content partition.
    content_form = str(content_form)
    if content_form == "c":
        content_mask = layout.c_content_score_causal_mask()
    elif content_form == "delta":
        content_mask = np.tile(
            delta_causal_mask(layout, token_valid_mask=token_valid_mask),
            (n_pair, 1),
        )
    else:
        raise ValueError(f"content_form must be 'c' or 'delta', got {content_form!r}.")
    if refresh_before_exp and refresh_after_exp:
        raise ValueError(
            "Prefix softmax cannot refresh both before and after exp."
        )
    if pair_pre_exp_bootstrap and not refresh_before_exp:
        raise ValueError(
            "pair_pre_exp_bootstrap requires refresh_before_exp=True."
        )
    if (refresh_before_exp or refresh_after_exp) and bootstrap_operator is None:
        raise ValueError(
            "prefix softmax refresh requires a prepared bootstrap operator."
        )
    base_multiplier = score_scale / float(2**k)
    base_shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    base_invalid = float(config.stats_min_for(layer_idx)) / float(2**k)
    exp_entry = config.layer_exp[int(layer_idx)]
    rough = config.invsqrt_alg2["j1_rough"]
    precise = config.invsqrt_alg2["j2_precise"]
    # Low-limb QK output can exhaust exp before a post-exp checkpoint runs.
    # Both score streams are aligned to the lower level after the public map,
    # so either low input requires a pre-exp refresh of both partitions.
    pre_exp_limb_floor = (
        chebyshev_ps_mul_depth(exp_entry["coefficients"]) + 2 * k + 6
    )
    low_prefix = _min_limbs(working_prefix_scores) <= pre_exp_limb_floor
    low_content = _min_limbs(content_score_ciphers) <= pre_exp_limb_floor
    if (low_prefix or low_content) and bootstrap_operator is None:
        raise ValueError(
            "Prefix softmax needs a pre-exp bootstrap for low-limb scores: "
            f"prefix={_min_limbs(working_prefix_scores)}, "
            f"content={_min_limbs(content_score_ciphers)}, "
            f"required>{pre_exp_limb_floor}."
        )
    if low_prefix or low_content:
        if refresh_after_exp:
            raise ValueError(
                "Low-limb Prefix/Content scores require pre-exp rather than "
                "post-exp refresh."
            )
        refresh_before_exp = True
        pair_pre_exp_bootstrap = True
    # Prefix uses the same public score map as Content.  Paired partitioned
    # refresh compensates for split-complex doubling in that map.
    refresh_prefix_before_exp = bool(low_prefix or low_content)
    # Prefix is stored as one real C-form stream per query pair.  The old
    # path refreshed all 16 streams independently, while the content Delta
    # partition used the paired (real/imaginary) bootstrap path.  Apart from
    # doubling the bootstrap count, that made the sparse Prefix probability
    # stream accumulate substantially more bootstrap error than Content.
    # Partitioned Prefix has an even number of streams, so pair it whenever
    # the pre-exp checkpoint is active.  Compact Prefix has one carrier and
    # must retain the scalar path.
    pair_prefix_pre_exp = bool(
        refresh_prefix_before_exp
        and prefix_packing == "partitioned"
        and len(working_prefix_scores) % 2 == 0
    )
    print(
        "[prefix-softmax] "
        f"k={k} refresh_content_before={refresh_before_exp} "
        f"pair_content={pair_pre_exp_bootstrap} "
        f"refresh_prefix_before={refresh_prefix_before_exp} "
        f"pair_prefix={pair_prefix_pre_exp} "
        f"prefix_packing={prefix_packing} "
        f"alg2_rounds={alg2_rounds}/{k} "
        f"limbs_p={_min_limbs(working_prefix_scores) if working_prefix_scores else None} "
        f"limbs_c={_min_limbs(content_score_ciphers) if content_score_ciphers else None}",
        flush=True,
    )

    def _map_constants(*, pair: bool) -> tuple[float, float, float]:
        multiplier = base_multiplier
        shift = base_shift
        invalid = base_invalid
        if pair:
            fit_lo, fit_hi = (
                float(value) for value in exp_entry["fit_interval"]
            )
            map_alpha = 2.0 / (fit_hi - fit_lo)
            map_beta = -1.0 - 2.0 * fit_lo / (fit_hi - fit_lo)
            multiplier *= 0.5 * map_alpha
            shift = 0.5 * (map_alpha * shift + map_beta)
            invalid *= 0.5 * map_alpha
        return multiplier, shift, invalid

    prefix_multiplier, prefix_shift, _prefix_invalid = _map_constants(
        pair=pair_prefix_pre_exp
    )
    # Empty C-form rows are packing, not causal-invalid keys.  stats_min
    # parks 126/128 slots on the Chebyshev endpoint; zeroing them instead
    # makes a sparse spike field that paired bootstrap rings.  Extra fill 0
    # leaves unused slots at the public shift (interior of [-1, 1] after
    # pairing).  _exp_mask still zeros them after exp.
    prefix_invalid = 0.0
    content_multiplier, content_shift, content_invalid = _map_constants(
        pair=pair_pre_exp_bootstrap
    )
    slim_masks = (
        build_delta_softmax_mask_bundle(layout, crypto_context=crypto_context)
        if checkpointed
        else None
    )

    def _mark(name: str, started: float) -> None:
        timings[name] = float(time.perf_counter() - started)

    def _public_map(
        ciphers,
        valid_mask,
        *,
        name: str,
        multiplier: float,
        shift: float,
        invalid: float,
    ) -> tuple[object, ...]:
        scaled = shifted = None
        try:
            scaled = _mul_packed_rows(
                ciphers,
                np.asarray(valid_mask, dtype=np.float64) * multiplier,
                name=f"{name}.valid_scale",
                crypto_context=crypto_context,
            )
            shifted = _add_scalar_ciphers(
                scaled, shift, crypto_context=crypto_context
            )
            invalid_fill = (1.0 - np.asarray(valid_mask, dtype=np.float64)) * invalid
            return _add_packed_rows(
                shifted,
                invalid_fill,
                name=f"{name}.invalid",
                crypto_context=crypto_context,
            )
        finally:
            _release_tuple(scaled)
            _release_tuple(shifted)

    def _exp_mask(
        mapped,
        valid_mask,
        *,
        name: str,
        input_is_chebyshev_mapped: bool,
    ) -> tuple[object, ...]:
        exp_outputs = tuple(
            eval_exp_cipher(
                cipher,
                exp_entry,
                crypto_context=crypto_context,
                input_is_chebyshev_mapped=bool(input_is_chebyshev_mapped),
            )
            for cipher in mapped
        )
        if input_is_chebyshev_mapped:
            _release_tuple(mapped)
        try:
            return _mul_packed_rows(
                exp_outputs,
                valid_mask,
                name=f"{name}.post_exp",
                crypto_context=crypto_context,
            )
        finally:
            _release_tuple(exp_outputs)

    def _update_partition(y_ciphers, squared, lambda_pairs):
        if reuse_round_square:
            nxt = tuple(
                mul_cipher_rescale(
                    left, right, crypto_context=crypto_context
                )
                for left, right in zip(squared, lambda_pairs, strict=True)
            )
            _release_tuple(y_ciphers)
            return nxt
        scaled = []
        try:
            for y_cipher, lam_cipher in zip(y_ciphers, lambda_pairs, strict=True):
                aligned_y, aligned_lam, owned_y, owned_lam = _align_cipher_pair(
                    y_cipher, lam_cipher, crypto_context=crypto_context
                )
                try:
                    scaled.append(
                        mul_cipher_rescale(
                            aligned_y,
                            aligned_lam,
                            crypto_context=crypto_context,
                        )
                    )
                finally:
                    if owned_y:
                        release_if_supported(aligned_y)
                    if owned_lam:
                        release_if_supported(aligned_lam)
            nxt = tuple(
                square_rescale(cipher, crypto_context=crypto_context)
                for cipher in scaled
            )
        finally:
            _release_tuple(scaled)
        _release_tuple(y_ciphers)
        return nxt

    mapped_p = mapped_c = y_p = y_c = None
    try:
        start = time.perf_counter()
        mapped_p = _public_map(
            working_prefix_scores,
            prefix_mask,
            name="prefix.alg2",
            multiplier=prefix_multiplier,
            shift=prefix_shift,
            invalid=prefix_invalid,
        )
        _release_tuple(compact_prefix_scores)
        compact_prefix_scores = ()
        mapped_c = _public_map(
            content_score_ciphers,
            content_mask,
            name="content.alg2",
            multiplier=content_multiplier,
            shift=content_shift,
            invalid=content_invalid,
        )
        old_mapped_p = tuple(mapped_p)
        old_mapped_c = tuple(mapped_c)
        aligned_mapped_p, aligned_mapped_c = _align_partition_streams(
            old_mapped_p, old_mapped_c, crypto_context=crypto_context
        )
        for old, new in zip(old_mapped_p, aligned_mapped_p, strict=True):
            if old is not new:
                release_if_supported(old)
        for old, new in zip(old_mapped_c, aligned_mapped_c, strict=True):
            if old is not new:
                release_if_supported(old)
        mapped_p = aligned_mapped_p
        mapped_c = aligned_mapped_c
        _diagnostic_state("prefix.mapped_p_aligned", mapped_p)
        _diagnostic_state("prefix.mapped_c_aligned", mapped_c)
        _softmax_diagnostic_range(
            "prefix.mapped_p", mapped_p, crypto_context=crypto_context
        )
        _softmax_diagnostic_range(
            "prefix.mapped_c", mapped_c, crypto_context=crypto_context
        )
        _mark("shift_scale_mask", start)
        if refresh_prefix_before_exp or refresh_before_exp:
            start = time.perf_counter()
            if refresh_prefix_before_exp:
                mapped_p = _refresh_partition(
                    mapped_p,
                    bootstrap_operator=bootstrap_operator,
                    pair=pair_prefix_pre_exp,
                    iterations=main_bootstrap_iterations,
                    precision_bits=checkpoint_precision_bits,
                    match_two_pass_level=match_single_pass_bootstrap_level,
                    use_deep_program=main_bootstrap_use_deep,
                    crypto_context=crypto_context,
                    ensure_two_pass_headroom=True,
                )
            if refresh_before_exp:
                mapped_c = _refresh_partition(
                    mapped_c,
                    bootstrap_operator=bootstrap_operator,
                    pair=pair_pre_exp_bootstrap,
                    iterations=main_bootstrap_iterations,
                    precision_bits=checkpoint_precision_bits,
                    match_two_pass_level=match_single_pass_bootstrap_level,
                    use_deep_program=main_bootstrap_use_deep,
                    crypto_context=crypto_context,
                    ensure_two_pass_headroom=True,
                )
            _softmax_diagnostic_range(
                "prefix.pre_exp_p", mapped_p, crypto_context=crypto_context
            )
            _softmax_diagnostic_range(
                "prefix.pre_exp_c", mapped_c, crypto_context=crypto_context
            )
            _mark("pre_exp_bootstrap", start)
        start = time.perf_counter()
        y_p = _exp_mask(
            mapped_p,
            prefix_mask,
            name="prefix.alg2",
            input_is_chebyshev_mapped=pair_prefix_pre_exp,
        )
        mapped_p = None
        y_c = _exp_mask(
            mapped_c,
            content_mask,
            name="content.alg2",
            input_is_chebyshev_mapped=pair_pre_exp_bootstrap,
        )
        mapped_c = None
        _softmax_diagnostic_range(
            "prefix.exp_p", y_p, crypto_context=crypto_context
        )
        _softmax_diagnostic_range(
            "prefix.exp_c", y_c, crypto_context=crypto_context
        )
        _mark("exp", start)
        if (
            bootstrap_operator is not None
            and not refresh_prefix_before_exp
            and round_refresh_policy != "native"
        ):
            start = time.perf_counter()
            # Unused C-form rows are already zero after _exp_mask.  Unpaired
            # bootstrap of that spike field rings; densify, refresh, remask.
            prefix_fill = (1.0 - np.asarray(prefix_mask, dtype=np.float64)) * 0.25
            filled_p = _add_packed_rows(
                y_p,
                prefix_fill,
                name="prefix.alg2.post_exp.fill",
                crypto_context=crypto_context,
            )
            _release_tuple(y_p)
            if prefix_packing == "partitioned":
                # Keep the Prefix probability stream on the same paired
                # bootstrap path as Content.  ``filled_p`` is deliberately
                # dense here; pairing a sparse spike field directly causes
                # the inactive rows to ring through the bootstrap.
                y_p = _paired_checkpoint_refresh(
                    filled_p,
                    bootstrap_operator=bootstrap_operator,
                    label_prefix="alg2.post_exp.prefix",
                    main_bootstrap_iterations=main_bootstrap_iterations,
                    checkpoint_precision_bits=checkpoint_precision_bits,
                    match_single_pass_bootstrap_level=match_single_pass_bootstrap_level,
                    crypto_context=crypto_context,
                )
            else:
                y_p = _refresh_partition(
                    filled_p,
                    bootstrap_operator=bootstrap_operator,
                    pair=False,
                    iterations=main_bootstrap_iterations,
                    precision_bits=checkpoint_precision_bits,
                    match_two_pass_level=match_single_pass_bootstrap_level,
                    use_deep_program=False,
                    crypto_context=crypto_context,
                )
            remasked_p = _mul_packed_rows(
                y_p,
                prefix_mask,
                name="prefix.alg2.post_exp.mask",
                crypto_context=crypto_context,
            )
            _release_tuple(y_p)
            y_p = remasked_p
            _softmax_diagnostic_range(
                "prefix.exp_p_refreshed", y_p, crypto_context=crypto_context
            )
            _mark("prefix_post_exp_bootstrap", start)
        if extra_softmax_checkpoint_enabled(
            "LLAMA_FHE_SOFTMAX_POST_EXP_BOOTSTRAP",
            int(layer_idx),
            SLIM_POST_EXP_LAYERS,
            bootstrap_operator,
        ):
            start = time.perf_counter()
            y_c = _paired_checkpoint_refresh(
                y_c,
                bootstrap_operator=bootstrap_operator,
                label_prefix="alg2.post_exp.content",
                main_bootstrap_iterations=main_bootstrap_iterations,
                checkpoint_precision_bits=checkpoint_precision_bits,
                match_single_pass_bootstrap_level=match_single_pass_bootstrap_level,
                crypto_context=crypto_context,
            )
            _mark("post_exp_bootstrap", start)
        if (
            bootstrap_operator is not None
            and _min_limbs(y_c) <= 3
        ):
            start = time.perf_counter()
            y_c = _paired_checkpoint_refresh(
                y_c,
                bootstrap_operator=bootstrap_operator,
                label_prefix="alg2.post_exp.content.limbs",
                main_bootstrap_iterations=main_bootstrap_iterations,
                checkpoint_precision_bits=checkpoint_precision_bits,
                match_single_pass_bootstrap_level=match_single_pass_bootstrap_level,
                crypto_context=crypto_context,
            )
            _mark("post_exp_limb_bootstrap", start)
        if refresh_after_exp:
            start = time.perf_counter()
            y_c = _refresh_partition(
                y_c,
                bootstrap_operator=bootstrap_operator,
                pair=False,
                iterations=main_bootstrap_iterations,
                precision_bits=checkpoint_precision_bits,
                match_two_pass_level=match_single_pass_bootstrap_level,
                use_deep_program=False,
                crypto_context=crypto_context,
            )
            _mark("post_exp_bootstrap", start)
        for round_index in range(1, alg2_rounds + 1):
            round_start = time.perf_counter()
            poly = config.poly_for_layer(
                rough if round_index == 1 else precise, int(layer_idx)
            )
            # Both partitions participate in every Alg2 normalization round.
            # The old implementation squared Prefix only in round one, which
            # made k>1 Prefix probabilities effectively one-round softmax.
            squared_p = tuple(
                square_rescale(cipher, crypto_context=crypto_context)
                for cipher in y_p
            )
            squared_c = tuple(
                square_rescale(cipher, crypto_context=crypto_context)
                for cipher in y_c
            )
            reduced_p = reduced_c = combined = lam = None
            reduced_p0 = reduced_c0 = broadcast_p = broadcast_c = None
            lambda_pairs = compact_lambda_pairs = None
            refreshed_prefix_square: tuple[object, ...] = ()
            nxt_p = nxt_c = remask_p = remask_c = None
            try:
                prefix_reducer = (
                    _compact_prefix_reduce_rows_fhe
                    if prefix_packing == "compact"
                    else _prefix_reduce_rows_fhe
                )
                reduced_p = prefix_reducer(
                    squared_p,
                    prefix_k=prefix_k,
                    layout=layout,
                    crypto_context=crypto_context,
                )
                reduced_c = delta_reduce_rows_fhe(
                    squared_c, layout=layout, crypto_context=crypto_context
                )
                if reduced_p:
                    _softmax_diagnostic_range(
                        f"prefix.Z_p.r{round_index}",
                        reduced_p,
                        crypto_context=crypto_context,
                    )
                _softmax_diagnostic_range(
                    f"prefix.Z_c.r{round_index}",
                    reduced_c,
                    crypto_context=crypto_context,
                )
                if checkpointed:
                    mapped_masks = None
                    mapped_alpha = 1.0
                    mapped_beta = 0.0
                    pack_prefix = "softmax.slim"
                    if fuse_slim_chebyshev_map:
                        cheb = poly.get("chebyshev_fhe") or poly
                        lo, hi = (float(value) for value in cheb["fit_interval"])
                        pack_prefix = f"softmax.slim.mapped.{round_index}"
                        mapped_masks = build_delta_softmax_mask_bundle(
                            layout,
                            crypto_context=crypto_context,
                            coefficient=2.0 / (hi - lo),
                            name_prefix=pack_prefix,
                        )
                    pack_masks = (
                        mapped_masks if mapped_masks is not None else slim_masks
                    )
                    slim_c = delta_pack_row_sums_slim_fhe(
                        reduced_c,
                        layout=layout,
                        crypto_context=crypto_context,
                        masks=pack_masks,
                        mask_name_prefix=pack_prefix,
                    )
                    # Add the Prefix contribution to the same slim row-sum
                    # carrier on every round.  Prefix and content share the
                    # query axis after reduction, so this is a plain CT add
                    # (with the usual level alignment) and avoids a second
                    # inverse-sqrt evaluation.
                    if prefix_packing == "compact":
                        slim_p = _compact_prefix_pack_row_sums_slim_fhe(
                            reduced_p,
                            prefix_k=prefix_k,
                            layout=layout,
                            crypto_context=crypto_context,
                            masks=pack_masks,
                            mask_name_prefix=pack_prefix,
                        )
                    else:
                        slim_p = _prefix_pack_row_sums_slim_fhe(
                            reduced_p,
                            layout=layout,
                            crypto_context=crypto_context,
                            masks=pack_masks,
                            # ``pack_masks`` is the same bundle used by the
                            # content row-sum carrier.  Reuse its canonical
                            # vector names; appending ``.prefix`` asks the
                            # ConstantBundle for a vector that was never built.
                            mask_name_prefix=pack_prefix,
                        )
                    _diagnostic_state(
                        f"prefix.slim_c.r{round_index}", (slim_c,)
                    )
                    _diagnostic_state(
                        f"prefix.slim_p.r{round_index}", (slim_p,)
                    )
                    slim_pair = _add_cipher_pairs(
                        (slim_c,), (slim_p,), crypto_context=crypto_context
                    )
                    release_if_supported(slim_c)
                    release_if_supported(slim_p)
                    slim = slim_pair[0]
                    _diagnostic_state(
                        f"prefix.slim_sum.r{round_index}", (slim,)
                    )
                    _softmax_diagnostic_range(
                        f"prefix.Z.r{round_index}",
                        (slim,),
                        crypto_context=crypto_context,
                    )
                    sanitize_slim = (
                        _fill_mapped_delta_slim_poly_input
                        if fuse_slim_chebyshev_map
                        else _fill_delta_slim_poly_input
                        if direct_slim_inactive_fill
                        else _sanitize_delta_slim_poly_input
                    )
                    sanitized_slim = sanitize_slim(
                        slim,
                        poly,
                        layout=layout,
                        crypto_context=crypto_context,
                        name=f"prefix.alg2.j{round_index}.slim",
                    )
                    release_if_supported(slim)
                    _softmax_diagnostic_range(
                        f"prefix.slim_input.r{round_index}",
                        (sanitized_slim,),
                        crypto_context=crypto_context,
                    )
                    polynomial_depth = chebyshev_ps_mul_depth(poly["coefficients"])
                    total_polynomial_depth = polynomial_depth + (
                        0 if fuse_slim_chebyshev_map else 1
                    )
                    if int(sanitized_slim.state.cur_limbs) <= int(
                        total_polynomial_depth
                    ) + 1:
                        previous = sanitized_slim
                        sanitized_slim = bootstrap_operator.refresh(
                            previous,
                            iterations=lambda_bootstrap_iterations,
                            match_two_pass_level=bool(
                                match_single_pass_bootstrap_level
                            ),
                        )
                        if sanitized_slim is not previous:
                            release_if_supported(previous)
                    lambda_cipher = eval_invsqrt_cipher(
                        sanitized_slim,
                        poly,
                        crypto_context=crypto_context,
                        input_is_chebyshev_mapped=fuse_slim_chebyshev_map,
                    )
                    release_if_supported(sanitized_slim)
                    previous = lambda_cipher
                    if (
                        round_refresh_policy == "native"
                        and prefix_packing == "compact"
                        and reuse_round_square
                        and round_index == alg2_rounds
                    ):
                        (
                            lambda_cipher,
                            refreshed_prefix_probability,
                        ) = _refresh_lambda_with_compact_prefix_probability(
                            previous,
                            y_p[0],
                            prefix_mask=prefix_mask,
                            prefix_gain=prefix_bootstrap_gain,
                            bootstrap_operator=bootstrap_operator,
                            iterations=lambda_bootstrap_iterations,
                            match_two_pass_level=bool(
                                match_single_pass_bootstrap_level
                            ),
                            crypto_context=crypto_context,
                        )
                        try:
                            refreshed_prefix_square = (
                                square_rescale(
                                    refreshed_prefix_probability,
                                    crypto_context=crypto_context,
                                ),
                            )
                        finally:
                            release_if_supported(
                                refreshed_prefix_probability
                            )
                    else:
                        lambda_cipher = bootstrap_operator.refresh(
                            previous,
                            iterations=lambda_bootstrap_iterations,
                            match_two_pass_level=bool(
                                match_single_pass_bootstrap_level
                            ),
                        )
                    if lambda_cipher is not previous:
                        release_if_supported(previous)
                    _softmax_diagnostic_range(
                        f"prefix.lambda.r{round_index}",
                        (lambda_cipher,),
                        crypto_context=crypto_context,
                    )
                    lambda_for_broadcast = lambda_cipher
                    if reuse_round_square:
                        lambda_for_broadcast = square_rescale(
                            lambda_cipher, crypto_context=crypto_context
                        )
                        release_if_supported(lambda_cipher)
                    lambda_pairs = delta_broadcast_slim_rows_fhe(
                        lambda_for_broadcast,
                        layout=layout,
                        crypto_context=crypto_context,
                        masks=slim_masks,
                    )
                    release_if_supported(lambda_for_broadcast)
                    if prefix_packing == "compact":
                        compact_lambda_pairs = _compact_prefix_lambda_pairs_fhe(
                            lambda_pairs,
                            prefix_k=prefix_k,
                            layout=layout,
                            crypto_context=crypto_context,
                        )
                    if reuse_round_square:
                        _release_tuple(y_p)
                        y_p = ()
                        _release_tuple(y_c)
                        y_c = ()
                    nxt_c = _update_partition(y_c, squared_c, lambda_pairs)
                    if round_refresh_policy == "native":
                        # Both partitions were masked immediately after exp.
                        # Squaring and multiplying by one row scalar preserve
                        # exact invalid zeros, so repeating the public mask
                        # here only burns a level and forces avoidable
                        # probability-stream bootstraps.  This is the same
                        # invariant used by the native Delta Softmax path.
                        y_c = tuple(nxt_c)
                        nxt_c = None
                    else:
                        remask_c = _mul_packed_rows(
                            tuple(nxt_c),
                            content_mask,
                            name=f"content.alg2.r{round_index}.mask",
                            crypto_context=crypto_context,
                        )
                        _release_tuple(nxt_c)
                        nxt_c = None
                        y_c = remask_c
                        remask_c = None
                    nxt_p = _update_partition(
                        y_p,
                        (
                            refreshed_prefix_square
                            if refreshed_prefix_square
                            else squared_p
                        ),
                        compact_lambda_pairs
                        if prefix_packing == "compact"
                        else lambda_pairs,
                    )
                    if round_refresh_policy == "native":
                        y_p = tuple(nxt_p)
                        nxt_p = None
                    else:
                        remask_p = _mul_packed_rows(
                            tuple(nxt_p),
                            prefix_mask,
                            name=f"prefix.alg2.r{round_index}.mask",
                            crypto_context=crypto_context,
                        )
                        _release_tuple(nxt_p)
                        nxt_p = None
                        y_p = remask_p
                        remask_p = None
                    _softmax_diagnostic_range(
                        f"prefix.round{round_index}_p",
                        y_p,
                        crypto_context=crypto_context,
                    )
                    _softmax_diagnostic_range(
                        f"prefix.round{round_index}_c",
                        y_c,
                        crypto_context=crypto_context,
                    )
                    remaining_rounds = int(alg2_rounds) - int(round_index)
                    min_prefix_limbs = _min_limbs(y_p)
                    min_content_limbs = _min_limbs(y_c)
                    remaining_limb_floor = _prefix_remaining_limb_floor(
                        remaining_rounds
                    )
                    refresh_after_round = bool(
                        remaining_rounds > 0
                        and (
                            round_refresh_policy == "always"
                            or (
                                round_refresh_policy == "adaptive"
                                and min(min_prefix_limbs, min_content_limbs)
                                <= remaining_limb_floor
                            )
                        )
                    )
                    print(
                        "[prefix-softmax-limbs] "
                        f"layer={int(layer_idx)} round={round_index}/{alg2_rounds} "
                        f"policy={round_refresh_policy} "
                        f"prefix={min_prefix_limbs} content={min_content_limbs} "
                        f"remaining={remaining_rounds} floor={remaining_limb_floor} "
                        f"refresh={refresh_after_round}",
                        flush=True,
                    )
                    if refresh_after_round:
                        prefix_fill = (
                            1.0 - np.asarray(prefix_mask, dtype=np.float64)
                        ) * 0.25
                        filled_p = _add_packed_rows(
                            y_p,
                            prefix_fill,
                            name=f"prefix.alg2.r{round_index}.fill",
                            crypto_context=crypto_context,
                        )
                        _release_tuple(y_p)
                        if prefix_packing == "partitioned":
                            y_p = _paired_checkpoint_refresh(
                                filled_p,
                                bootstrap_operator=bootstrap_operator,
                                label_prefix=f"alg2.round_{round_index}.prefix",
                                main_bootstrap_iterations=main_bootstrap_iterations,
                                checkpoint_precision_bits=checkpoint_precision_bits,
                                match_single_pass_bootstrap_level=match_single_pass_bootstrap_level,
                                crypto_context=crypto_context,
                            )
                        else:
                            y_p = _refresh_partition(
                                filled_p,
                                bootstrap_operator=bootstrap_operator,
                                pair=False,
                                iterations=main_bootstrap_iterations,
                                precision_bits=checkpoint_precision_bits,
                                match_two_pass_level=match_single_pass_bootstrap_level,
                                use_deep_program=False,
                                crypto_context=crypto_context,
                            )
                        remasked_p = _mul_packed_rows(
                            y_p,
                            prefix_mask,
                            name=f"prefix.alg2.r{round_index}.post_bs.mask",
                            crypto_context=crypto_context,
                        )
                        _release_tuple(y_p)
                        y_p = remasked_p
                        y_c = _paired_checkpoint_refresh(
                            y_c,
                            bootstrap_operator=bootstrap_operator,
                            label_prefix=f"alg2.round_{round_index}.content",
                            main_bootstrap_iterations=main_bootstrap_iterations,
                            checkpoint_precision_bits=checkpoint_precision_bits,
                            match_single_pass_bootstrap_level=match_single_pass_bootstrap_level,
                            crypto_context=crypto_context,
                        )
                else:
                    # Both partitions may be sparse after causal masking.  The
                    # reducer places a dense input's sum on every row, but a
                    # sparse input's sum only on the occupied rows.  Isolate
                    # feature 0 first, then use the inverse rotation tree to
                    # broadcast one shared ``Z`` carrier to every row.
                    if prefix_packing == "compact":
                        broadcast_p = _compact_prefix_broadcast_rows_fhe(
                            reduced_p,
                            prefix_k=prefix_k,
                            layout=layout,
                            crypto_context=crypto_context,
                        )
                    else:
                        reduced_p0 = _mul_packed_rows(
                            reduced_p,
                            layout.c_feature_row_mask(0),
                            name=f"prefix.reduce.p0.r{round_index}",
                            crypto_context=crypto_context,
                        )
                        broadcast_p = _prefix_broadcast_rows_fhe(
                            reduced_p0, layout=layout, crypto_context=crypto_context
                        )
                    reduced_c0 = _mul_packed_rows(
                        reduced_c,
                        layout.c_feature_row_mask(0),
                        name=f"prefix.reduce.c0.r{round_index}",
                        crypto_context=crypto_context,
                    )
                    broadcast_c = _prefix_broadcast_rows_fhe(
                        reduced_c0, layout=layout, crypto_context=crypto_context
                    )
                    combined = _add_cipher_pairs(
                        broadcast_p, broadcast_c, crypto_context=crypto_context
                    )
                    lam = tuple(
                        eval_invsqrt_cipher(
                            cipher, poly, crypto_context=crypto_context
                        )
                        for cipher in combined
                    )
                    if prefix_packing == "compact":
                        compact_lambda_pairs = _compact_prefix_lambda_pairs_fhe(
                            lam,
                            prefix_k=prefix_k,
                            layout=layout,
                            crypto_context=crypto_context,
                        )
                    nxt_p = _update_partition(
                        y_p,
                        squared_p,
                        compact_lambda_pairs
                        if prefix_packing == "compact"
                        else lam,
                    )
                    nxt_c = _update_partition(y_c, squared_c, lam)
                    if round_refresh_policy == "native":
                        y_p = tuple(nxt_p)
                        y_c = tuple(nxt_c)
                        nxt_p = nxt_c = None
                    else:
                        remask_p = _mul_packed_rows(
                            tuple(nxt_p),
                            prefix_mask,
                            name=f"prefix.alg2.r{round_index}.mask",
                            crypto_context=crypto_context,
                        )
                        remask_c = _mul_packed_rows(
                            tuple(nxt_c),
                            content_mask,
                            name=f"content.alg2.r{round_index}.mask",
                            crypto_context=crypto_context,
                        )
                        _release_tuple(nxt_p)
                        _release_tuple(nxt_c)
                        nxt_p = nxt_c = None
                        y_p = remask_p
                        y_c = remask_c
                        remask_p = remask_c = None
            finally:
                _release_tuple(squared_p)
                _release_tuple(squared_c)
                _release_tuple(reduced_p)
                _release_tuple(reduced_c)
                _release_tuple(reduced_p0)
                _release_tuple(reduced_c0)
                _release_tuple(broadcast_p)
                _release_tuple(broadcast_c)
                _release_tuple(combined)
                _release_tuple(lam)
                _release_tuple(lambda_pairs)
                _release_tuple(compact_lambda_pairs)
                _release_tuple(refreshed_prefix_square)
                _release_tuple(nxt_p)
                _release_tuple(nxt_c)
                _release_tuple(remask_p)
                _release_tuple(remask_c)
            _mark(f"alg2_round_{round_index}", round_start)
            mid_round = (int(k) + 1) // 2
            if (
                checkpointed
                and round_index == mid_round
                and mid_round < int(k)
                and extra_softmax_checkpoint_enabled(
                    "LLAMA_FHE_SOFTMAX_MID_ALG2_BOOTSTRAP",
                    int(layer_idx),
                    SLIM_MID_ALG2_LAYERS,
                    bootstrap_operator,
                )
            ):
                start = time.perf_counter()
                if prefix_packing == "compact":
                    prefix_fill = (
                        1.0 - np.asarray(prefix_mask, dtype=np.float64)
                    ) * 0.25
                    filled_p = _add_packed_rows(
                        y_p,
                        prefix_fill,
                        name=f"prefix.alg2.mid_round_{round_index}.fill",
                        crypto_context=crypto_context,
                    )
                    _release_tuple(y_p)
                    y_p = _refresh_partition(
                        filled_p,
                        bootstrap_operator=bootstrap_operator,
                        pair=False,
                        iterations=main_bootstrap_iterations,
                        precision_bits=checkpoint_precision_bits,
                        match_two_pass_level=match_single_pass_bootstrap_level,
                        use_deep_program=False,
                        crypto_context=crypto_context,
                    )
                    remasked_p = _mul_packed_rows(
                        y_p,
                        prefix_mask,
                        name=f"prefix.alg2.mid_round_{round_index}.post_bs.mask",
                        crypto_context=crypto_context,
                    )
                    _release_tuple(y_p)
                    y_p = remasked_p
                else:
                    y_p = _paired_checkpoint_refresh(
                        y_p,
                        bootstrap_operator=bootstrap_operator,
                        label_prefix=f"alg2.mid_round_{round_index}.prefix",
                        main_bootstrap_iterations=main_bootstrap_iterations,
                        checkpoint_precision_bits=checkpoint_precision_bits,
                        match_single_pass_bootstrap_level=match_single_pass_bootstrap_level,
                        crypto_context=crypto_context,
                    )
                y_c = _paired_checkpoint_refresh(
                    y_c,
                    bootstrap_operator=bootstrap_operator,
                    label_prefix=f"alg2.mid_round_{round_index}.content",
                    main_bootstrap_iterations=main_bootstrap_iterations,
                    checkpoint_precision_bits=checkpoint_precision_bits,
                    match_single_pass_bootstrap_level=match_single_pass_bootstrap_level,
                    crypto_context=crypto_context,
                )
                _mark("mid_alg2_bootstrap", start)
        if checkpointed and extra_softmax_checkpoint_enabled(
            "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP",
            int(layer_idx),
            SLIM_POST_ALG2_LAYERS,
            bootstrap_operator,
            rounds=int(alg2_rounds),
            min_rounds=SLIM_POST_ALG2_MIN_ROUNDS,
        ):
            start = time.perf_counter()
            y_c = _paired_checkpoint_refresh(
                y_c,
                bootstrap_operator=bootstrap_operator,
                label_prefix="alg2.post_alg2.content",
                main_bootstrap_iterations=main_bootstrap_iterations,
                checkpoint_precision_bits=checkpoint_precision_bits,
                match_single_pass_bootstrap_level=match_single_pass_bootstrap_level,
                crypto_context=crypto_context,
            )
            _mark("post_alg2_bootstrap", start)
        if prefix_packing == "compact":
            expanded_p = _expand_compact_prefix_partitions_fhe(
                y_p,
                prefix_k=prefix_k,
                layout=layout,
                crypto_context=crypto_context,
            )
            _release_tuple(y_p)
            y_p = ()
            result = (expanded_p, tuple(y_c))
        else:
            result = (tuple(y_p), tuple(y_c))
        if os.environ.get("LLAMA_FHE_PREFIX_PROFILE", "") == "1":
            sync_device(crypto_context.device)
            print(
                "[prefix-softmax-profile] "
                f"layer={int(layer_idx)} packing={prefix_packing} "
                f"policy={round_refresh_policy} "
                f"total={time.perf_counter() - function_started:.6f} "
                f"output_prefix_limbs={_min_limbs(result[0])} "
                f"output_content_limbs={_min_limbs(result[1])} "
                f"stages={dict(timings)}",
                flush=True,
            )
        capture_path = os.environ.get(
            "LLAMA_FHE_PREFIX_CAPTURE_NPZ", ""
        ).strip()
        if capture_path:
            packed_prefix = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher)).real
                    for cipher in result[0]
                ]
            )
            packed_content = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher)).real
                    for cipher in result[1]
                ]
            )
            logical_prefix = layout.unpack_prefix_scores_prefill(
                packed_prefix, prefix_k=prefix_k
            )
            logical_content = (
                layout.unpack_score_delta(packed_content)
                if content_form == "delta"
                else layout.unpack_prefix_scores_prefill(
                    packed_content, prefix_k=int(layout.seq_len)
                )
            )
            np.savez_compressed(
                capture_path,
                prefix=logical_prefix,
                content=logical_content,
                prefix_limbs=np.asarray([_min_limbs(result[0])]),
                content_limbs=np.asarray([_min_limbs(result[1])]),
            )
            print(
                f"[prefix-softmax-capture] path={capture_path} "
                f"prefix_shape={logical_prefix.shape} "
                f"content_shape={logical_content.shape}",
                flush=True,
            )
        y_p = y_c = None
        return result
    finally:
        _release_tuple(compact_prefix_scores)
        _release_tuple(mapped_p)
        _release_tuple(mapped_c)
        _release_tuple(y_p)
        _release_tuple(y_c)


def softmax_alg2_prefix_pages_fhe(
    prefix_score_ciphers: tuple[object, ...] | list[object],
    content_score_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    prefix_k: int,
    layer_idx: int = 0,
    rounds: int | None = None,
    score_scale: float | None = None,
    content_form: str = "delta",
    token_valid_mask: np.ndarray | None = None,
    bootstrap_operator=None,
    refresh_before_exp: bool | None = None,
    refresh_after_exp: bool | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    stage_seconds: dict[str, float] | None = None,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Experimental page-wise Prefill Prefix/Content Softmax.

    This is the structural counterpart of Decode's ``row_pages`` path.  The
    Prefix C-page (compact when it fits, otherwise partitioned) and the
    Content Delta page are treated as two page descriptors: both pass through
    one shared map/exp/reduce/``Z``/lambda/Alg2 loop, while each page retains
    its native reducer and mask.  The layout is intentionally not changed
    here; the compact Prefix page is still one ciphertext and the Content page
    is still the existing Delta carrier.  Thus this path proves the unified
    normalization semantics before attempting a higher-risk triangular Content
    packing redesign.
    """

    pipeline_started = time.perf_counter()
    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    prefix_packing = _resolve_prefix_partition_packing(layout, prefix_k)
    content_form = str(content_form)
    if content_form not in {"c", "delta"}:
        raise ValueError(
            "the page-wise Prefill pipeline requires content_form='c' or 'delta'."
        )
    if refresh_before_exp is None:
        refresh_before_exp = operator_config.refresh_softmax_exp_inputs
    if refresh_after_exp is None:
        refresh_after_exp = operator_config.refresh_softmax_exp_outputs
    refresh_before_exp = bool(refresh_before_exp)
    refresh_after_exp = bool(refresh_after_exp)
    checkpointed = bootstrap_operator is not None
    pair_pre_exp_bootstrap = bool(
        operator_config.pair_softmax_pre_exp_bootstrap and refresh_before_exp
    )
    reuse_round_square = bool(
        operator_config.reuse_softmax_round_square and checkpointed
    )
    fuse_slim_chebyshev_map = bool(
        operator_config.fuse_softmax_slim_chebyshev_map
    )
    direct_slim_inactive_fill = bool(
        operator_config.direct_softmax_slim_inactive_fill
    )
    round_refresh_policy = _prefix_round_refresh_policy()
    # The experimental pages implementation has not yet adopted the native
    # one-mask lifetime rule below.  Keep its previous conservative schedule
    # when no explicit diagnostic policy was requested.
    if round_refresh_policy == "native":
        round_refresh_policy = "always"
    timings = {} if stage_seconds is None else stage_seconds
    prefix_score_ciphers = tuple(prefix_score_ciphers)
    content_score_ciphers = tuple(content_score_ciphers)
    n_pair = int(layout.query_pair_count)
    if len(prefix_score_ciphers) != n_pair or len(content_score_ciphers) != n_pair:
        raise ValueError(
            f"expected {n_pair} prefix and content score ciphers, got "
            f"{len(prefix_score_ciphers)} and {len(content_score_ciphers)}."
        )
    prefix_k = int(prefix_k)
    k = int(config.k_for(layer_idx) if rounds is None else rounds)
    if k < 1:
        raise ValueError(f"Alg2 k must be >= 1, got {k}.")
    base_invalid = float(config.stats_min_for(layer_idx)) / float(2**k)
    compact_scores: tuple[object, ...] = ()
    if (
        prefix_packing == "compact"
        and int(layout.query_pair_count) * prefix_k > int(layout.head_dim)
    ):
        # Keep pages semantics for geometries that cannot fit all pair blocks
        # in one C-grid.  The native partitioned carrier is the safe fallback;
        # no ciphertext is silently truncated.
        prefix_packing = "partitioned"
    if prefix_packing == "compact":
        compact_scores = _compact_prefix_partitions_fhe(
            prefix_score_ciphers,
            prefix_k=prefix_k,
            layout=layout,
            crypto_context=crypto_context,
        )
        working_prefix_scores = compact_scores
        prefix_mask = _compact_prefix_mask(layout, prefix_k=prefix_k)
    else:
        # A compact Prefix page is a performance optimization, not a
        # correctness requirement.  If the requested prefix occupies more
        # than one C-grid's available feature rows, retain the native
        # partitioned carrier while using the same page-wise normalization
        # control flow below.
        working_prefix_scores = prefix_score_ciphers
        prefix_mask = layout.c_prefix_score_mask(prefix_k)
    if content_form == "c":
        content_mask = layout.c_content_score_causal_mask()
        # Match the native C-form path: causal-invalid slots are parked at
        # stats_min during the public map and removed by the post-exp mask.
        content_invalid = base_invalid
    else:
        content_mask = np.tile(
            delta_causal_mask(layout, token_valid_mask=token_valid_mask),
            (n_pair, 1),
        )
        content_invalid = base_invalid
    score_scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    if refresh_before_exp and refresh_after_exp:
        _release_tuple(compact_scores)
        raise ValueError("Prefix page Softmax cannot refresh both before and after exp.")
    if (refresh_before_exp or refresh_after_exp) and not checkpointed:
        _release_tuple(compact_scores)
        raise ValueError("Prefix page refresh requires a bootstrap operator.")
    base_multiplier = score_scale / float(2**k)
    base_shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    exp_entry = config.layer_exp[int(layer_idx)]
    rough = config.invsqrt_alg2["j1_rough"]
    precise = config.invsqrt_alg2["j2_precise"]
    main_bootstrap_iterations = operator_config.softmax_main_bootstrap_iterations
    lambda_bootstrap_iterations = operator_config.softmax_lambda_bootstrap_iterations
    match_single_pass = bool(
        operator_config.match_softmax_single_pass_bootstrap_level
    )
    precision_bits = int(operator_config.checkpoint_bootstrap_precision_bits)
    main_bootstrap_use_deep = bool(operator_config.softmax_main_deep_bootstrap)
    slim_masks = (
        build_delta_softmax_mask_bundle(layout, crypto_context=crypto_context)
        if checkpointed
        else None
    )

    def mark(name: str, started: float) -> None:
        timings[name] = float(time.perf_counter() - started)

    def public_map(ciphers, valid_mask, *, name, multiplier, shift, invalid):
        scaled = shifted = None
        try:
            scaled = _mul_packed_rows(
                ciphers,
                np.asarray(valid_mask, dtype=np.float64) * multiplier,
                name=f"{name}.valid_scale",
                crypto_context=crypto_context,
            )
            shifted = _add_scalar_ciphers(
                scaled, shift, crypto_context=crypto_context
            )
            return _add_packed_rows(
                shifted,
                (1.0 - np.asarray(valid_mask, dtype=np.float64)) * invalid,
                name=f"{name}.invalid",
                crypto_context=crypto_context,
            )
        finally:
            _release_tuple(scaled)
            _release_tuple(shifted)

    def exp_mask(mapped, valid_mask, *, name):
        exp_outputs = tuple(
            eval_exp_cipher(
                cipher,
                exp_entry,
                crypto_context=crypto_context,
                input_is_chebyshev_mapped=False,
            )
            for cipher in mapped
        )
        try:
            return _mul_packed_rows(
                exp_outputs,
                valid_mask,
                name=f"{name}.post_exp",
                crypto_context=crypto_context,
            )
        finally:
            _release_tuple(exp_outputs)

    def update_partition(y_ciphers, squared, lambda_ciphers):
        if reuse_round_square:
            outputs = tuple(
                mul_cipher_rescale(
                    left, right, crypto_context=crypto_context
                )
                for left, right in zip(squared, lambda_ciphers, strict=True)
            )
            _release_tuple(y_ciphers)
            return outputs
        scaled = []
        try:
            for y_cipher, lam_cipher in zip(
                y_ciphers, lambda_ciphers, strict=True
            ):
                aligned_y, aligned_lam, owned_y, owned_lam = _align_cipher_pair(
                    y_cipher, lam_cipher, crypto_context=crypto_context
                )
                try:
                    scaled.append(
                        mul_cipher_rescale(
                            aligned_y,
                            aligned_lam,
                            crypto_context=crypto_context,
                        )
                    )
                finally:
                    if owned_y:
                        release_if_supported(aligned_y)
                    if owned_lam:
                        release_if_supported(aligned_lam)
            outputs = tuple(
                square_rescale(cipher, crypto_context=crypto_context)
                for cipher in scaled
            )
        finally:
            _release_tuple(scaled)
        _release_tuple(y_ciphers)
        return outputs

    def map_page_pair():
        # Both pages share the same public map and exp entry.  The only
        # page-specific state is the validity mask and the ciphertext tuple.
        prefix = public_map(
            working_prefix_scores,
            prefix_mask,
            name="prefix.pages",
            multiplier=base_multiplier,
            shift=base_shift,
            invalid=0.0,
        )
        content = public_map(
            content_score_ciphers,
            content_mask,
            name="content.pages",
            multiplier=base_multiplier,
            shift=base_shift,
            invalid=content_invalid,
        )
        return prefix, content

    mapped_p = mapped_c = y_p = y_c = None
    try:
        started = time.perf_counter()
        mapped_p, mapped_c = map_page_pair()
        mark("pages.shift_scale_mask", started)
        refresh_prefix_before_exp = bool(
            refresh_before_exp
            and mapped_p
            and _min_limbs(mapped_p) <= 4
        )
        if refresh_before_exp:
            started = time.perf_counter()
            if refresh_prefix_before_exp:
                mapped_p = _refresh_partition(
                    mapped_p,
                    bootstrap_operator=bootstrap_operator,
                    pair=False,
                    iterations=main_bootstrap_iterations,
                    precision_bits=precision_bits,
                    match_two_pass_level=match_single_pass,
                    use_deep_program=main_bootstrap_use_deep,
                    crypto_context=crypto_context,
                )
            mapped_c = _refresh_partition(
                mapped_c,
                bootstrap_operator=bootstrap_operator,
                pair=pair_pre_exp_bootstrap,
                iterations=main_bootstrap_iterations,
                precision_bits=precision_bits,
                match_two_pass_level=match_single_pass,
                use_deep_program=main_bootstrap_use_deep,
                crypto_context=crypto_context,
            )
            mark("pages.pre_exp_bootstrap", started)
        started = time.perf_counter()
        y_p = exp_mask(mapped_p, prefix_mask, name="prefix.pages")
        mapped_p = None
        y_c = exp_mask(mapped_c, content_mask, name="content.pages")
        mapped_c = None
        mark("pages.exp", started)

        if bootstrap_operator is not None and not refresh_prefix_before_exp:
            started = time.perf_counter()
            filled = _add_packed_rows(
                y_p,
                (1.0 - np.asarray(prefix_mask, dtype=np.float64)) * 0.25,
                name="prefix.pages.post_exp.fill",
                crypto_context=crypto_context,
            )
            _release_tuple(y_p)
            y_p = _refresh_partition(
                filled,
                bootstrap_operator=bootstrap_operator,
                pair=False,
                iterations=main_bootstrap_iterations,
                precision_bits=precision_bits,
                match_two_pass_level=match_single_pass,
                use_deep_program=False,
                crypto_context=crypto_context,
            )
            remasked = _mul_packed_rows(
                y_p,
                prefix_mask,
                name="prefix.pages.post_exp.mask",
                crypto_context=crypto_context,
            )
            _release_tuple(y_p)
            y_p = remasked
            mark("pages.prefix_post_exp_bootstrap", started)

        for round_index in range(1, k + 1):
            round_started = time.perf_counter()
            poly = config.poly_for_layer(
                rough if round_index == 1 else precise, int(layer_idx)
            )
            squared_p = tuple(
                square_rescale(cipher, crypto_context=crypto_context)
                for cipher in y_p
            )
            squared_c = tuple(
                square_rescale(cipher, crypto_context=crypto_context)
                for cipher in y_c
            )
            reduced_p = reduced_c = slim_p = slim_c = slim = None
            lambda_cipher = lambda_for_broadcast = None
            lambda_pairs = compact_lambda_pairs = None
            reduced_p0 = None
            p0 = c0 = bp = bc = combined = None
            nxt_p = nxt_c = remask_p = remask_c = None
            try:
                reduced_p = (
                    _compact_prefix_reduce_rows_fhe(
                        squared_p,
                        prefix_k=prefix_k,
                        layout=layout,
                        crypto_context=crypto_context,
                    )
                    if prefix_packing == "compact"
                    else _prefix_reduce_rows_fhe(
                        squared_p,
                        prefix_k=prefix_k,
                        layout=layout,
                        crypto_context=crypto_context,
                    )
                )
                # The Content page must keep its native reducer in both C and
                # Delta forms.  ``_prefix_reduce_rows_fhe`` assumes the
                # compact Prefix-page geometry (one packed prefix page); it
                # is not equivalent to the triangular C-page reducer even
                # when ``prefix_k == seq_len``.  Reusing the native reducer
                # preserves the row multiplicities and causal layout that the
                # existing, validated dual pipeline uses.
                reduced_c = delta_reduce_rows_fhe(
                    squared_c, layout=layout, crypto_context=crypto_context
                )
                if checkpointed:
                    mapped_masks = None
                    pack_prefix = "softmax.slim"
                    if fuse_slim_chebyshev_map:
                        cheb = poly.get("chebyshev_fhe") or poly
                        lo, hi = (float(value) for value in cheb["fit_interval"])
                        pack_prefix = f"softmax.slim.mapped.{round_index}"
                        mapped_masks = build_delta_softmax_mask_bundle(
                            layout,
                            crypto_context=crypto_context,
                            coefficient=2.0 / (hi - lo),
                            name_prefix=pack_prefix,
                        )
                    pack_masks = mapped_masks if mapped_masks is not None else slim_masks
                    slim_c = delta_pack_row_sums_slim_fhe(
                        reduced_c,
                        layout=layout,
                        crypto_context=crypto_context,
                        masks=pack_masks,
                        mask_name_prefix=pack_prefix,
                    )
                    slim_p = (
                        _compact_prefix_pack_row_sums_slim_fhe(
                            reduced_p,
                            prefix_k=prefix_k,
                            layout=layout,
                            crypto_context=crypto_context,
                            masks=pack_masks,
                            mask_name_prefix=pack_prefix,
                        )
                        if prefix_packing == "compact"
                        else _prefix_pack_row_sums_slim_fhe(
                            reduced_p,
                            layout=layout,
                            crypto_context=crypto_context,
                            masks=pack_masks,
                            mask_name_prefix=pack_prefix,
                        )
                    )
                    slim = _add_cipher_pairs(
                        (slim_c,), (slim_p,), crypto_context=crypto_context
                    )[0]
                    sanitize = (
                        _fill_mapped_delta_slim_poly_input
                        if fuse_slim_chebyshev_map
                        else _fill_delta_slim_poly_input
                        if direct_slim_inactive_fill
                        else _sanitize_delta_slim_poly_input
                    )
                    sanitized = sanitize(
                        slim,
                        poly,
                        layout=layout,
                        crypto_context=crypto_context,
                        name=f"prefix.pages.r{round_index}.slim",
                    )
                    release_if_supported(slim)
                    slim = None
                    depth = chebyshev_ps_mul_depth(poly["coefficients"])
                    depth += 0 if fuse_slim_chebyshev_map else 1
                    if int(sanitized.state.cur_limbs) <= int(depth) + 1:
                        previous = sanitized
                        sanitized = bootstrap_operator.refresh(
                            previous,
                            iterations=lambda_bootstrap_iterations,
                            match_two_pass_level=match_single_pass,
                        )
                        if sanitized is not previous:
                            release_if_supported(previous)
                    lambda_cipher = eval_invsqrt_cipher(
                        sanitized,
                        poly,
                        crypto_context=crypto_context,
                        input_is_chebyshev_mapped=fuse_slim_chebyshev_map,
                    )
                    release_if_supported(sanitized)
                    previous = lambda_cipher
                    lambda_cipher = bootstrap_operator.refresh(
                        previous,
                        iterations=lambda_bootstrap_iterations,
                        match_two_pass_level=match_single_pass,
                    )
                    if lambda_cipher is not previous:
                        release_if_supported(previous)
                    lambda_for_broadcast = lambda_cipher
                    lambda_cipher = None
                    if reuse_round_square:
                        squared_lambda = square_rescale(
                            lambda_for_broadcast, crypto_context=crypto_context
                        )
                        release_if_supported(lambda_for_broadcast)
                        lambda_for_broadcast = squared_lambda
                    lambda_pairs = delta_broadcast_slim_rows_fhe(
                        lambda_for_broadcast,
                        layout=layout,
                        crypto_context=crypto_context,
                        masks=slim_masks,
                    )
                    if prefix_packing == "compact":
                        compact_lambda_pairs = _compact_prefix_lambda_pairs_fhe(
                            lambda_pairs,
                            prefix_k=prefix_k,
                            layout=layout,
                            crypto_context=crypto_context,
                        )
                else:
                    if prefix_packing == "compact":
                        p0 = _compact_prefix_broadcast_rows_fhe(
                            reduced_p,
                            prefix_k=prefix_k,
                            layout=layout,
                            crypto_context=crypto_context,
                        )
                    else:
                        reduced_p0 = _mul_packed_rows(
                            reduced_p,
                            layout.c_feature_row_mask(0),
                            name=f"prefix.pages.r{round_index}.row0",
                            crypto_context=crypto_context,
                        )
                        p0 = _prefix_broadcast_rows_fhe(
                            reduced_p0,
                            layout=layout,
                            crypto_context=crypto_context,
                        )
                        _release_tuple(reduced_p0)
                        reduced_p0 = None
                    c0 = _mul_packed_rows(
                        reduced_c,
                        layout.c_feature_row_mask(0),
                        name=f"content.pages.r{round_index}.row0",
                        crypto_context=crypto_context,
                    )
                    # ``_compact_prefix_broadcast_rows_fhe`` already isolates
                    # each compact Prefix block and broadcasts its row sum.
                    # Broadcasting it a second time would multiply the
                    # Prefix contribution by the butterfly fan-out and make
                    # the shared Z denominator too large.  Partitioned Prefix
                    # uses the explicit broadcast above and follows the same
                    # ownership convention.
                    bp = p0
                    p0 = None
                    bc = _prefix_broadcast_rows_fhe(
                        c0, layout=layout, crypto_context=crypto_context
                    )
                    combined = _add_cipher_pairs(
                        bp, bc, crypto_context=crypto_context
                    )
                    lambda_pairs = tuple(
                        eval_invsqrt_cipher(
                            cipher, poly, crypto_context=crypto_context
                        )
                        for cipher in combined
                    )
                    if prefix_packing == "compact":
                        compact_lambda_pairs = _compact_prefix_lambda_pairs_fhe(
                            lambda_pairs,
                            prefix_k=prefix_k,
                            layout=layout,
                            crypto_context=crypto_context,
                        )
                if reuse_round_square:
                    _release_tuple(y_p)
                    y_p = ()
                    _release_tuple(y_c)
                    y_c = ()
                nxt_p = update_partition(
                    y_p,
                    squared_p,
                    compact_lambda_pairs
                    if prefix_packing == "compact"
                    else lambda_pairs,
                )
                remask_p = _mul_packed_rows(
                    nxt_p,
                    prefix_mask,
                    name=f"prefix.pages.r{round_index}.mask",
                    crypto_context=crypto_context,
                )
                _release_tuple(nxt_p)
                nxt_p = None
                y_p = remask_p
                remask_p = None
                nxt_c = update_partition(y_c, squared_c, lambda_pairs)
                remask_c = _mul_packed_rows(
                    nxt_c,
                    content_mask,
                    name=f"content.pages.r{round_index}.mask",
                    crypto_context=crypto_context,
                )
                _release_tuple(nxt_c)
                nxt_c = None
                y_c = remask_c
                remask_c = None
                if checkpointed:
                    remaining = int(k) - int(round_index)
                    should_refresh = bool(
                        remaining > 0
                        and (
                            round_refresh_policy == "always"
                            or (
                                round_refresh_policy == "adaptive"
                                and min(_min_limbs(y_p), _min_limbs(y_c))
                                <= _prefix_remaining_limb_floor(remaining)
                            )
                        )
                    )
                    if should_refresh:
                        filled = _add_packed_rows(
                            y_p,
                            (1.0 - np.asarray(prefix_mask)) * 0.25,
                            name=f"prefix.pages.r{round_index}.fill",
                            crypto_context=crypto_context,
                        )
                        _release_tuple(y_p)
                        y_p = _refresh_partition(
                            filled,
                            bootstrap_operator=bootstrap_operator,
                            pair=False,
                            iterations=main_bootstrap_iterations,
                            precision_bits=precision_bits,
                            match_two_pass_level=match_single_pass,
                            use_deep_program=False,
                            crypto_context=crypto_context,
                        )
                        remasked = _mul_packed_rows(
                            y_p,
                            prefix_mask,
                            name=f"prefix.pages.r{round_index}.post_bs.mask",
                            crypto_context=crypto_context,
                        )
                        _release_tuple(y_p)
                        y_p = remasked
                        y_c = _paired_checkpoint_refresh(
                            y_c,
                            bootstrap_operator=bootstrap_operator,
                            label_prefix=f"pages.round_{round_index}.content",
                            main_bootstrap_iterations=main_bootstrap_iterations,
                            checkpoint_precision_bits=precision_bits,
                            match_single_pass_bootstrap_level=match_single_pass,
                            crypto_context=crypto_context,
                        )
            finally:
                _release_tuple(squared_p)
                _release_tuple(squared_c)
                _release_tuple(reduced_p)
                _release_tuple(reduced_c)
                release_if_supported(slim_p)
                release_if_supported(slim_c)
                release_if_supported(slim)
                release_if_supported(lambda_cipher)
                release_if_supported(lambda_for_broadcast)
                _release_tuple(lambda_pairs)
                _release_tuple(compact_lambda_pairs)
                _release_tuple(reduced_p0)
                _release_tuple(p0)
                _release_tuple(c0)
                _release_tuple(bp)
                _release_tuple(bc)
                _release_tuple(combined)
                _release_tuple(nxt_p)
                _release_tuple(nxt_c)
                _release_tuple(remask_p)
                _release_tuple(remask_c)
            mark(f"pages.alg2_round_{round_index}", round_started)

        if prefix_packing == "compact":
            expanded_p = _expand_compact_prefix_partitions_fhe(
                y_p,
                prefix_k=prefix_k,
                layout=layout,
                crypto_context=crypto_context,
            )
            _release_tuple(y_p)
            y_p = ()
        else:
            expanded_p = tuple(y_p)
            y_p = None
        result = (expanded_p, tuple(y_c))
        y_c = None
        if os.environ.get("LLAMA_FHE_PREFIX_PROFILE", "") == "1":
            sync_device(crypto_context.device)
            print(
                "[prefix-softmax-profile] "
                f"layer={int(layer_idx)} pipeline=pages packing={prefix_packing} "
                f"total={time.perf_counter() - pipeline_started:.6f} "
                f"output_prefix_limbs={_min_limbs(result[0])} "
                f"output_content_limbs={_min_limbs(result[1])} "
                f"stages={dict(timings)}",
                flush=True,
            )
        return result
    finally:
        _release_tuple(compact_scores)
        _release_tuple(mapped_p)
        _release_tuple(mapped_c)
        _release_tuple(y_p)
        _release_tuple(y_c)


__all__ = [
    "prefix_softmax_rotations",
    "softmax_alg2_prefix_pages_fhe",
    "softmax_alg2_prefix_partitions_fhe",
]
