from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import (
    profile_sync_device as sync_device,  # stage-timer sync, no-op under T2_STAGE_SYNC=0
)
from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout

from ....approx.chebyshev import chebyshev_ps_mul_depth
from ...linear import pair_real_ciphers, split_complex_ciphers_twice
from ...nonlinear.bootstrap import BootstrapOperator
from ...primitives import (
    add_owned as _add_owned,
)
from ...primitives import (
    bundle_plain as _bundle_plain,
)
from ...primitives import (
    mul_cipher_rescale as _mul_cipher_rescale,
)
from ...primitives import (
    mul_plain_rescale as _mul_plain_rescale,
)
from ...primitives import (
    square_rescale as _square_rescale,
)
from ..config import AttentionOperatorConfig


def _softmax_diagnostic_range(label: str, ciphers, *, crypto_context) -> None:
    """Print decrypted ranges only for an explicitly requested debug run."""

    if os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS", "") != "1":
        return
    values = np.stack(
        [np.asarray(crypto_context.decrypt(cipher)) for cipher in ciphers]
    )
    finite = np.isfinite(values)
    finite_values = values[finite]
    real_values = finite_values.real
    levels = sorted(
        {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    )
    imag_max = None
    complex_request = os.environ.get(
        "LLAMA_FHE_STAGE_DIAGNOSTICS_COMPLEX", ""
    ).strip()
    complex_labels = {
        value.strip() for value in complex_request.split(",") if value.strip()
    }
    if complex_request == "1" or label in complex_labels:
        rotated = [
            fhe.homo_mul_i(cipher, crypto_context.context, negative=True)
            for cipher in ciphers
        ]
        try:
            imag = np.stack(
                [np.asarray(crypto_context.decrypt(c)) for c in rotated]
            )
        finally:
            for c in rotated:
                release_if_supported(c)
        imag_max = float(np.max(np.abs(imag.real)))
    states = sorted(
        {
            (
                int(cipher.state.cur_limbs),
                int(cipher.state.scale_degree),
                round(float(np.log2(float(cipher.state.scaling_factor))), 4),
                bool(getattr(cipher, "is_ext", False)),
                int(getattr(cipher, "slots", 0)),
            )
            for cipher in ciphers
        }
    )
    print(
        f"[softmax-diagnostic] stage={label} "
        f"finite={bool(np.all(finite))} "
        f"real_min={None if not real_values.size else float(np.min(real_values))} "
        f"real_max_signed={None if not real_values.size else float(np.max(real_values))} "
        f"real_max={None if not finite_values.size else float(np.max(np.abs(finite_values.real)))} "
        f"imag_max={imag_max} "
        f"real_mean={None if not finite_values.size else float(np.mean(np.abs(finite_values.real)))} "
        f"levels={levels} states={states}",
        flush=True,
    )


@dataclass
class DeltaSoftmaxAlg2Result:
    probability_ciphers: tuple[object, ...]
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]
    #: Delta-row period of the slim pack (``None`` = legacy one-row-per-pair).
    slim_period_rows: int | None = None
    #: Alg2 slim refreshes (slim input + lambda) that ran through the sparse program.
    slim_sparse_bootstraps: int = 0

    def release(self) -> None:
        owned = self.probability_ciphers
        self.probability_ciphers = ()
        for cipher in owned:
            release_if_supported(cipher)


def delta_softmax_row_rotations(layout: AttentionPairLayout) -> tuple[int, ...]:
    stride = 2 * int(layout.seq_len)
    return tuple(stride * (1 << bit) for bit in range(int(layout.seq_len).bit_length() - 1))


def delta_causal_mask(
    layout: AttentionPairLayout,
    *,
    token_valid_mask: np.ndarray | None = None,
    dtype=np.float64,
) -> np.ndarray:
    """Public causal/padding mask in ``S_Delta/P_Delta`` slot order.

    Invalid query rows retain only their self slot.  Those rows are not model
    outputs, but a single public-safe entry keeps the inverse-square-root
    polynomial away from the undefined all-zero row while preventing padding
    keys from influencing any valid query.
    """

    seq_len = int(layout.seq_len)
    queries = np.arange(seq_len, dtype=np.int64)
    deltas = np.arange(seq_len, dtype=np.int64)
    keys = (queries[None, :] + deltas[:, None]) % seq_len
    valid = keys <= queries[None, :]
    if token_valid_mask is not None:
        token_valid = np.asarray(token_valid_mask, dtype=bool).reshape(-1)
        if token_valid.shape != (seq_len,):
            raise ValueError(
                "token_valid_mask must have shape "
                f"{(seq_len,)}, got {token_valid.shape}."
            )
        valid &= token_valid[keys]
        valid &= token_valid[None, :]
        invalid_queries = ~token_valid
        valid[0, invalid_queries] = True
    base = np.repeat(valid[:, :, None], 2, axis=2).reshape(-1).astype(dtype)
    return np.tile(base, int(layout.payload_repetitions))


def softmax_slim_period_rows(layout: AttentionPairLayout) -> int:
    """Delta-row period ``P`` of the periodic slim pack: the smallest power of two >= ``query_pair_count``.

    Pair ``i`` then owns every Delta row ``d == i (mod P)`` of the slim
    ciphertext instead of row ``i`` only, so the slim message repeats every
    ``P * 2 * seq_len`` slots (:func:`softmax_slim_period_slots`) and its
    refreshes may run through a sparse-slot bootstrap program of that period
    (the same trick as the periodic slim RMSNorm row).  ``P`` is 2 for a
    W=8 rank (2 Q pairs), 4 for W=4, 16 for the full single-card layout.
    """

    pairs = int(layout.query_pair_count)
    if pairs <= 0:
        raise ValueError("query_pair_count must be positive.")
    period = 1 << (pairs - 1).bit_length()
    if period > int(layout.seq_len):
        raise ValueError(
            f"periodic slim pack needs query_pair_count <= seq_len, got {pairs} > {layout.seq_len}."
        )
    return int(period)


def softmax_slim_period_slots(layout: AttentionPairLayout, period_rows: int | None = None) -> int:
    """Slot period of the periodic slim message (``P * 2 * seq_len``)."""

    period_rows = softmax_slim_period_rows(layout) if period_rows is None else _check_slim_period_rows(layout, period_rows)
    return int(period_rows) * 2 * int(layout.seq_len)


def _check_slim_period_rows(layout: AttentionPairLayout, period_rows: int) -> int:
    period_rows = int(period_rows)
    if period_rows <= 0 or period_rows & (period_rows - 1):
        raise ValueError(f"slim period_rows must be a power of two, got {period_rows}.")
    if period_rows < int(layout.query_pair_count):
        raise ValueError(
            f"slim period_rows={period_rows} cannot hold {layout.query_pair_count} query pairs on distinct residues."
        )
    if period_rows > int(layout.seq_len):
        raise ValueError(f"slim period_rows={period_rows} exceeds seq_len={layout.seq_len}.")
    return period_rows


def _pair_slice_mask(
    layout: AttentionPairLayout, pair_index: int, *, period_rows: int | None = None
) -> np.ndarray:
    """Slim destination of pair ``pair_index``: Delta row ``pair_index`` (legacy) or every row ``== pair_index (mod P)``."""

    pair_index = int(pair_index)
    if pair_index < 0 or pair_index >= int(layout.query_pair_count):
        raise ValueError(f"pair_index={pair_index} outside the query-pair range.")
    if pair_index >= int(layout.seq_len):
        raise ValueError("slim Delta packing requires query_pair_count <= seq_len.")
    base = np.zeros((int(layout.seq_len), int(layout.seq_len), 2), dtype=np.float64)
    if period_rows is None:
        base[pair_index, :, :] = 1.0
    else:
        period_rows = _check_slim_period_rows(layout, period_rows)
        base[pair_index::period_rows, :, :] = 1.0
    return np.tile(base.reshape(-1), int(layout.payload_repetitions))


def build_delta_softmax_mask_bundle(
    layout: AttentionPairLayout,
    *,
    crypto_context,
    coefficient: float = 1.0,
    name_prefix: str = "softmax.slim",
    period_rows: int | None = None,
):
    """Pair slice masks of the slim pack / broadcast; ``period_rows`` selects the periodic layout."""

    vectors: dict[str, np.ndarray] = {}
    for pair_index in range(int(layout.query_pair_count)):
        values = float(coefficient) * _pair_slice_mask(layout, pair_index, period_rows=period_rows)
        vectors[f"{name_prefix}.pair.{pair_index}"] = values.reshape(-1)
    return crypto_context.constant_bundle(
        vectors,
        cache_key=(
            "softmax.pair_masks",
            str(name_prefix),
            float(coefficient),
            None if period_rows is None else int(period_rows),
        ),
    )


def _single_vector_bundle(
    values: np.ndarray,
    *,
    name: str,
    crypto_context,
):
    return crypto_context.constant_bundle(
        {str(name): np.asarray(values, dtype=np.float64).reshape(-1)},
        cache_key=("softmax.vector", str(name)),
    )


def delta_slim_active_mask(
    layout: AttentionPairLayout, *, dtype=np.float64, period_rows: int | None = None
) -> np.ndarray:
    """Slim rows that carry a query pair (legacy: rows ``< pairs``; periodic: residues ``< pairs`` mod ``P``)."""

    base = np.zeros((int(layout.seq_len), int(layout.seq_len), 2), dtype=dtype)
    pairs = int(layout.query_pair_count)
    if period_rows is None:
        base[:pairs, :, :] = 1.0
    else:
        period_rows = _check_slim_period_rows(layout, period_rows)
        for residue in range(pairs):
            base[residue::period_rows, :, :] = 1.0
    return np.tile(base.reshape(-1), int(layout.payload_repetitions))


def _slim_fit_interval(poly_entry: dict) -> tuple[float, float]:
    """The Chebyshev fit interval of a slim rsqrt/exp polynomial entry."""

    cheb = poly_entry.get("chebyshev_fhe") or poly_entry
    lo, hi = cheb["fit_interval"]
    return float(lo), float(hi)


def _slim_safe_input(lo: float, hi: float) -> float:
    """A public value inside the fit interval for the inactive slim rows.

    Those rows carry no query, so any in-domain constant works; 0.25 is
    clamped into ``[lo, hi]`` rather than using an endpoint so the polynomial
    is never evaluated at the edge of its fit.
    """

    return min(max(0.25, lo), hi)


def _add_public_fill(cipher, values: np.ndarray, *, name: str, crypto_context):
    """Add a public vector to ``cipher`` at its own level; ``cipher`` is kept."""

    bundle = _single_vector_bundle(
        values, name=name, crypto_context=crypto_context
    )
    plain = _bundle_plain(bundle, name, cipher, crypto_context=crypto_context)
    try:
        return crypto_context.fhe.homo_add_pt(
            cipher, plain, crypto_context.context
        )
    finally:
        release_if_supported(plain)


def _sanitize_delta_slim_poly_input(
    slim_cipher,
    poly_entry: dict,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    name: str,
    period_rows: int | None = None,
):
    """Remask the active slim rows, then fill the inactive ones.

    The general form, used when neither the direct fill nor the fused
    Chebyshev map is selected: it costs one plaintext multiplication because it
    does not assume the inactive rows are already zero.
    """

    active_mask = delta_slim_active_mask(layout, period_rows=period_rows)
    bundle = _single_vector_bundle(
        active_mask, name=f"{name}.active", crypto_context=crypto_context
    )
    plain = _bundle_plain(
        bundle, f"{name}.active", slim_cipher, crypto_context=crypto_context
    )
    try:
        active = _mul_plain_rescale(
            slim_cipher, plain, crypto_context=crypto_context
        )
    finally:
        release_if_supported(plain)
    lo, hi = _slim_fit_interval(poly_entry)
    try:
        return _add_public_fill(
            active,
            (1.0 - active_mask) * _slim_safe_input(lo, hi),
            name=f"{name}.inactive",
            crypto_context=crypto_context,
        )
    finally:
        release_if_supported(active)


def _fill_delta_slim_poly_input(
    slim_cipher,
    poly_entry: dict,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    name: str,
    period_rows: int | None = None,
):
    """Fill already-zero inactive slim rows without another PT multiplication.

    ``delta_pack_row_sums_slim_fhe`` selects one disjoint destination row per
    query pair and leaves every inactive row zero.  Adding the public safe
    polynomial input there is therefore equivalent to remasking the active
    rows first, but consumes no multiplicative level.
    """

    lo, hi = _slim_fit_interval(poly_entry)
    inactive = 1.0 - delta_slim_active_mask(layout, period_rows=period_rows)
    if not np.any(inactive):
        # periodic pack with P == query_pair_count: every slim row is live,
        # nothing to fill (keep the caller's ownership contract: return a copy)
        return slim_cipher.deep_copy()
    return _add_public_fill(
        slim_cipher,
        inactive * _slim_safe_input(lo, hi),
        name=f"{name}.inactive",
        crypto_context=crypto_context,
    )


def _fill_mapped_delta_slim_poly_input(
    slim_cipher,
    poly_entry: dict,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    name: str,
    period_rows: int | None = None,
):
    """Finish a Chebyshev-domain map already scaled in slim-pack masks.

    The slim-pack mask supplied the alpha factor, so only the beta offset
    remains for the active rows; the inactive rows get the mapped safe value.
    """

    active_mask = delta_slim_active_mask(layout, period_rows=period_rows)
    lo, hi = _slim_fit_interval(poly_entry)
    alpha = 2.0 / (hi - lo)
    beta = -1.0 - 2.0 * lo / (hi - lo)
    mapped_safe = alpha * _slim_safe_input(lo, hi) + beta
    return _add_public_fill(
        slim_cipher,
        active_mask * beta + (1.0 - active_mask) * mapped_safe,
        name=f"{name}.mapped_fill",
        crypto_context=crypto_context,
    )


def _apply_vector_mul_rescale(
    ciphers: tuple[object, ...],
    values: np.ndarray,
    *,
    name: str,
    crypto_context,
) -> tuple[object, ...]:
    bundle = _single_vector_bundle(
        values,
        name=name,
        crypto_context=crypto_context,
    )
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            plain = _bundle_plain(
                bundle,
                name,
                cipher,
                crypto_context=crypto_context,
            )
            try:
                outputs.append(
                    _mul_plain_rescale(cipher, plain, crypto_context=crypto_context)
                )
            finally:
                release_if_supported(plain)
        return tuple(outputs)
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def _add_vector_chunks(
    ciphers: tuple[object, ...],
    values: np.ndarray,
    *,
    name: str,
    crypto_context,
    max_abs_chunk: float = 1.0,
) -> tuple[object, ...]:
    values = np.asarray(values, dtype=np.float64)
    chunk_count = max(1, int(math.ceil(float(np.max(np.abs(values))) / float(max_abs_chunk))))
    chunk = values / float(chunk_count)
    outputs = tuple(cipher.deep_copy() for cipher in ciphers)
    try:
        for index in range(chunk_count):
            bundle = _single_vector_bundle(
                chunk,
                name=f"{name}.{index}",
                crypto_context=crypto_context,
                )
            next_outputs: list[object] = []
            completed = False
            try:
                for cipher in outputs:
                    plain = _bundle_plain(
                        bundle,
                        f"{name}.{index}",
                        cipher,
                        crypto_context=crypto_context,
                    )
                    try:
                        next_outputs.append(
                            crypto_context.fhe.homo_add_pt(
                                cipher, plain, crypto_context.context
                            )
                        )
                    finally:
                        release_if_supported(plain)
                completed = True
            except Exception:
                for cipher in next_outputs:
                    release_if_supported(cipher)
                raise
            finally:
                for cipher in outputs:
                    release_if_supported(cipher)
                outputs = ()
            if completed:
                outputs = tuple(next_outputs)
        return outputs
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def _transform_delta_pages(
    ciphers: tuple[object, ...],
    vectors: tuple[np.ndarray, ...],
    *,
    pair_count: int,
    name: str,
    crypto_context,
    add: bool = False,
) -> tuple[object, ...]:
    """Apply a distinct public vector to each page, preserving page order."""

    outputs: list[object] = []
    try:
        for page_index, vector in enumerate(vectors):
            source = ciphers[page_index * pair_count : (page_index + 1) * pair_count]
            page_name = name if len(vectors) == 1 else f"{name}.page{page_index}"
            transformed = (
                _add_vector_chunks(
                    source, vector, name=page_name, crypto_context=crypto_context
                )
                if add
                else _apply_vector_mul_rescale(
                    source, vector, name=page_name, crypto_context=crypto_context
                )
            )
            outputs.extend(transformed)
        return tuple(outputs)
    except Exception:
        for cipher in outputs:
            release_if_supported(cipher)
        raise


def delta_reduce_rows_fhe(
    value_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
) -> tuple[object, ...]:
    """Sum all 128 Delta entries and broadcast each row sum over Delta."""

    if len(value_ciphers) != int(layout.query_pair_count):
        raise ValueError(
            f"Delta row reduction expects {layout.query_pair_count} ciphers, got {len(value_ciphers)}."
        )
    outputs: list[object] = []
    try:
        for cipher in value_ciphers:
            current = cipher.deep_copy()
            for rotation in delta_softmax_row_rotations(layout):
                rotated = crypto_context.fhe.homo_rotate(
                    current, int(rotation), crypto_context.context
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
        for cipher in outputs:
            release_if_supported(cipher)
        raise


_MAX_ENCODED_BITS = 61  # easyfhe.fhe.constants.MAX_ENCODED_BITS (encoder cap)


def _mapped_mask_coefficient_fits(coefficient: float, cipher) -> bool:
    """Can a mask scaled by ``coefficient`` be encoded at ``cipher``'s scale?"""

    scale = getattr(getattr(cipher, "state", None), "scaling_factor", None)
    if scale is None:
        return True
    scaled = float(coefficient) * float(scale)
    return scaled <= 0 or math.log2(scaled) < _MAX_ENCODED_BITS - 0.05


def delta_pack_row_sums_slim_fhe(
    reduced_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    masks=None,
    mask_name_prefix: str = "softmax.slim",
):
    """Pack every pair's row sums into a distinct Delta slice of one cipher."""

    if len(reduced_ciphers) != int(layout.query_pair_count):
        raise ValueError(
            f"slim pack expects {layout.query_pair_count} ciphers, got {len(reduced_ciphers)}."
        )
    bundle = (
        build_delta_softmax_mask_bundle(
            layout,
            crypto_context=crypto_context,
        )
        if masks is None
        else masks
    )
    accumulator = None
    try:
        for pair_index, cipher in enumerate(reduced_ciphers):
            plain = _bundle_plain(
                bundle,
                f"{mask_name_prefix}.pair.{pair_index}",
                cipher,
                crypto_context=crypto_context,
            )
            try:
                selected = _mul_plain_rescale(cipher, plain, crypto_context=crypto_context)
            finally:
                release_if_supported(plain)
            accumulator = (
                selected
                if accumulator is None
                else _add_owned(accumulator, selected, crypto_context=crypto_context)
            )
        if accumulator is None:
            raise RuntimeError("slim row-sum pack produced no output.")
        return accumulator
    except Exception:
        if accumulator is not None:
            release_if_supported(accumulator)
        raise


def delta_slim_broadcast_rotations(
    layout: AttentionPairLayout, *, period_rows: int | None = None
) -> tuple[int, ...]:
    """Doubling steps that spread one slim pair slice over all Delta rows.

    Legacy pack: the slice sits on one row, ``log2(seq_len)`` steps
    (:func:`delta_softmax_row_rotations`).  Periodic pack: after the
    ``pair_index * stride`` alignment the slice already sits on every ``P``-th
    row, so only the ``P - 1`` rows inside each period are missing:
    ``log2(P)`` steps of ``stride * 2**b`` -- a prefix of the legacy chain,
    no new keys.
    """

    if period_rows is None:
        return delta_softmax_row_rotations(layout)
    period_rows = _check_slim_period_rows(layout, period_rows)
    stride = 2 * int(layout.seq_len)
    count = int(period_rows).bit_length() - 1
    return tuple(stride * (1 << bit) for bit in range(count))


def delta_broadcast_slim_rows_fhe(
    slim_cipher,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    masks=None,
    period_rows: int | None = None,
) -> tuple[object, ...]:
    """Extract each slim pair slice and broadcast it over all Delta blocks.

    ``masks`` must have been built with the same ``period_rows`` (see
    :func:`build_delta_softmax_mask_bundle`); the periodic layout keeps the
    ``pair_index * stride`` alignment rotation and shortens the doubling chain.
    """

    bundle = (
        build_delta_softmax_mask_bundle(layout, crypto_context=crypto_context, period_rows=period_rows)
        if masks is None
        else masks
    )
    stride = 2 * int(layout.seq_len)
    steps = delta_slim_broadcast_rotations(layout, period_rows=period_rows)
    outputs: list[object] = []
    source = slim_cipher
    expanded = None
    try:
        if int(slim_cipher.slots) != int(crypto_context.max_slots):
            raise ValueError(
                "slim broadcast requires the full Delta slot count."
            )
        for pair_index in range(int(layout.query_pair_count)):
            plain = _bundle_plain(
                bundle,
                f"softmax.slim.pair.{pair_index}",
                source,
                crypto_context=crypto_context,
            )
            try:
                selected = _mul_plain_rescale(source, plain, crypto_context=crypto_context)
            finally:
                release_if_supported(plain)
            if pair_index:
                current = crypto_context.fhe.homo_rotate(
                    selected, pair_index * stride, crypto_context.context
                )
                release_if_supported(selected)
            else:
                current = selected
            for positive_rotation in steps:
                rotated = crypto_context.fhe.homo_rotate(
                    current, -int(positive_rotation), crypto_context.context
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
        for cipher in outputs:
            release_if_supported(cipher)
        raise
    finally:
        release_if_supported(expanded)


def _half_rescale(cipher, *, crypto_context):
    """Undo the 2x from ``split_complex_ciphers_twice`` after a paired bootstrap.

    The pre-exp checkpoint folds that 0.5 into the public score plaintext
    instead. Extra checkpoints after exp / Alg2 must apply it here; omitting
    it maps the probability stream to ~1e20.
    """

    encoded = fhe.encode_scalar(
        0.5,
        cur_limbs=int(cipher.state.cur_limbs),
        scale_degree=1,
        scaling_factor=cipher.state.scaling_factor,
        context=crypto_context.context,
    )
    return fhe.homo_mul_scalar_rescale(
        cipher, encoded, crypto_context.context
    )


# Extra main-track Softmax bootstraps for the slim (depth-37 / post-19)
# context. Mid-layer k=2 fits in 19 limbs without them. High-k layers
# (0,1: k=4; 2,14: k=3; 31: k=5) would otherwise exhaust PairFM's 6-limb
# floor. Layer 31 also refreshes after exp, before round-1 λ: post-Alg2 is
# too late because λ is degree 512 on the slim program.
SLIM_POST_ALG2_LAYERS = frozenset({0, 1, 2, 14})
# The post-Alg2 refresh is only needed when the layer really runs >= this
# many rounds: a reduced-k softmax table (k ladder) that brings L2/L14 to
# k=2 lands on the same main-track budget as the mid layers and skips it.
SLIM_POST_ALG2_MIN_ROUNDS = 3
SLIM_POST_EXP_LAYERS = frozenset({31})
SLIM_MID_ALG2_LAYERS = frozenset({31})
_LEGACY_BOOTSTRAP_POST_LEVELS = 23


def _parse_layer_enable_env(raw: str, layer_idx: int) -> bool:
    text = raw.strip().lower()
    if text in {"", "false", "off", "none"}:
        return False
    if text in {"all", "true"}:
        return True
    return int(layer_idx) in {
        int(part) for part in text.split(",") if part.strip() != ""
    }


def extra_softmax_checkpoint_enabled(
    name: str,
    layer_idx: int,
    default_layers: frozenset[int],
    bootstrap_operator: BootstrapOperator | None,
    *,
    rounds: int | None = None,
    min_rounds: int | None = None,
) -> bool:
    """Return whether an extra Softmax checkpoint should run on ``layer_idx``.

    An explicit env var always wins (comma layer list, ``all``/``true``, or
    ``off``). ``1`` is layer 1, not a boolean. Unset env uses the slim
    schedule only when the bootstrap returns fewer than the legacy 23 levels,
    and (when ``rounds`` / ``min_rounds`` are given) only if the layer's Alg2
    round count reaches ``min_rounds``.
    """

    raw = os.environ.get(name)
    if raw is not None:
        return _parse_layer_enable_env(raw, layer_idx)
    if bootstrap_operator is None:
        return False
    if int(bootstrap_operator.config.output_levels) >= _LEGACY_BOOTSTRAP_POST_LEVELS:
        return False
    if rounds is not None and min_rounds is not None and int(rounds) < int(min_rounds):
        return False
    return int(layer_idx) in default_layers


def _paired_checkpoint_refresh(
    y_ciphers: tuple[object, ...],
    *,
    bootstrap_operator: BootstrapOperator,
    label_prefix: str,
    main_bootstrap_iterations: int | None,
    checkpoint_precision_bits: int,
    match_single_pass_bootstrap_level: bool,
    crypto_context,
) -> tuple[object, ...]:
    """Pair, bootstrap, split, and undo the 2x conjugate scale."""

    refreshed_y: list[object] = []
    paired_y: tuple[object, ...] = ()
    try:
        paired_y = pair_real_ciphers(y_ciphers, crypto_context=crypto_context)
        for index, cipher in enumerate(paired_y):
            refreshed_y.append(
                bootstrap_operator.refresh(
                    cipher,
                    iterations=main_bootstrap_iterations,
                    precision_bits=int(checkpoint_precision_bits),
                    match_two_pass_level=bool(
                        match_single_pass_bootstrap_level
                    ),
                )
            )
    except Exception:
        for cipher in refreshed_y:
            release_if_supported(cipher)
        raise
    finally:
        for cipher in paired_y:
            release_if_supported(cipher)
        cipher = None
        paired_y = ()
    for cipher in y_ciphers:
        release_if_supported(cipher)
    cipher = None
    y_ciphers = ()
    split_y: tuple[object, ...] = ()
    halved_y: list[object] = []
    try:
        split_y = split_complex_ciphers_twice(
            tuple(refreshed_y), crypto_context=crypto_context
        )
        for cipher in split_y:
            halved_y.append(
                _half_rescale(cipher, crypto_context=crypto_context)
            )
    except Exception:
        for cipher in halved_y:
            release_if_supported(cipher)
        cipher = None
        halved_y.clear()
        raise
    finally:
        for cipher in refreshed_y:
            release_if_supported(cipher)
        for cipher in split_y:
            release_if_supported(cipher)
        cipher = None
        refreshed_y.clear()
        split_y = ()
    outputs = tuple(halved_y)
    halved_y.clear()
    return outputs


def softmax_alg2_fhe_delta(
    score_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    layer_idx: int,
    config,
    crypto_context,
    operator_config: AttentionOperatorConfig | None = None,
    score_scale: float | None = None,
    bootstrap_operator: BootstrapOperator | None = None,
    refresh_before_exp: bool | None = None,
    refresh_after_exp: bool | None = None,
    token_valid_mask: np.ndarray | None = None,
    rounds: int | None = None,
    fused_prefix_plan=None,
    _score_pages: tuple[tuple[object, ...], ...] | None = None,
    _page_masks: tuple[np.ndarray, ...] | None = None,
) -> DeltaSoftmaxAlg2Result:
    """Run polynomial Alg2 directly on paired ``S_Delta`` ciphertexts.

    ``config`` is the fitted softmax artifact; ``operator_config`` is the
    execution policy. Lambda refresh is owned by this Softmax operator:
    supplying a prepared :class:`BootstrapOperator` makes the required Alg2
    bootstraps explicit without leaking callback scheduling into the model.

    The two refresh flags stay parameters because the caller resolves them per
    layer -- layer zero has no steady-state pre-exp checkpoint -- and pass
    ``None`` to accept the policy's own value.
    """

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    if refresh_before_exp is None:
        refresh_before_exp = operator_config.refresh_softmax_exp_inputs
    if refresh_after_exp is None:
        refresh_after_exp = operator_config.refresh_softmax_exp_outputs
    main_bootstrap_use_deep = bool(operator_config.softmax_main_deep_bootstrap)
    pair_pre_exp_bootstrap = bool(
        operator_config.pair_softmax_pre_exp_bootstrap
    )
    reuse_round_square = bool(operator_config.reuse_softmax_round_square)
    direct_slim_inactive_fill = bool(
        operator_config.direct_softmax_slim_inactive_fill
    )
    fuse_slim_chebyshev_map = bool(
        operator_config.fuse_softmax_slim_chebyshev_map
    )
    # Periodic slim pack (``softmax_slim_periodic``): pair i on every Delta
    # row == i (mod P); the slim refreshes take the sparse program of period
    # P * 2 * seq_len slots when the operator carries one, dense otherwise
    # (the pack itself is a pure re-layout either way).
    slim_period_rows = (
        softmax_slim_period_rows(layout)
        if bool(getattr(operator_config, "softmax_slim_periodic", False))
        else None
    )
    slim_periodic_slots = (
        None
        if slim_period_rows is None
        else int(softmax_slim_period_slots(layout, slim_period_rows))
    )
    slim_sparse_bootstrap = bool(
        slim_periodic_slots is not None
        and bootstrap_operator is not None
        and bool(getattr(bootstrap_operator, "supports_periodic_slots", lambda _p: False)(slim_periodic_slots))
        and os.environ.get("LLAMA_FHE_SOFTMAX_SPARSE_BS", "1") != "0"
    )
    slim_refresh_periodic_slots = slim_periodic_slots if slim_sparse_bootstrap else None
    slim_sparse_bootstraps = 0
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

    from ...nonlinear.elementwise import (
        eval_exp_cipher,
        eval_invsqrt_cipher,
    )

    pair_count = int(layout.query_pair_count)
    if _score_pages is None:
        if _page_masks is not None:
            raise ValueError("_page_masks requires _score_pages.")
        score_pages = (tuple(score_ciphers),)
    else:
        score_pages = tuple(tuple(page) for page in _score_pages)
        if not score_pages or _page_masks is None:
            raise ValueError("multi-page Delta Alg2 requires pages and their masks.")
        if fused_prefix_plan is not None:
            raise ValueError("fused Prefix must be supplied as an already merged Delta page.")
    if any(len(page) != pair_count for page in score_pages):
        raise ValueError(
            f"Delta Alg2 expects {pair_count} score ciphers per page, "
            f"got {[len(page) for page in score_pages]}."
        )
    page_count = len(score_pages)
    score_ciphers = tuple(cipher for page in score_pages for cipher in page)
    layer_idx = int(layer_idx)
    k_layer = int(config.k_for(layer_idx) if rounds is None else rounds)
    if k_layer < 1:
        raise ValueError(f"Alg2 k must be >= 1, got {k_layer}.")
    score_scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    if _page_masks is None:
        valid = (
            delta_causal_mask(layout, token_valid_mask=token_valid_mask)
            if fused_prefix_plan is None
            else fused_prefix_plan.active_mask
        )
        page_valid = (np.asarray(valid, dtype=np.float64),)
    else:
        if len(_page_masks) != page_count:
            raise ValueError("one public mask is required for each Delta page.")
        page_valid = tuple(np.asarray(mask, dtype=np.float64) for mask in _page_masks)
        if any(mask.shape != (int(layout.slots),) for mask in page_valid):
            raise ValueError(f"each Delta page mask must have shape ({layout.slots},).")
        if any(not np.all((mask == 0) | (mask == 1)) for mask in page_valid):
            raise ValueError("Delta page masks must be binary.")
        active = np.stack(page_valid).reshape(
            page_count,
            int(layout.payload_repetitions),
            int(layout.seq_len),
            int(layout.seq_len),
            2,
        )
        if not active.any(axis=(0, 2)).all():
            raise ValueError("every query/head lane needs an active Delta score.")
        valid = page_valid[0]
    pair_pre_exp_bootstrap = bool(pair_pre_exp_bootstrap)
    reuse_round_square = bool(reuse_round_square)
    direct_slim_inactive_fill = bool(direct_slim_inactive_fill)
    fuse_slim_chebyshev_map = bool(fuse_slim_chebyshev_map)
    if pair_pre_exp_bootstrap and not bool(refresh_before_exp):
        raise ValueError(
            "pair_pre_exp_bootstrap requires refresh_before_exp=True."
        )
    masks = build_delta_softmax_mask_bundle(
        layout, crypto_context=crypto_context, period_rows=slim_period_rows
    )
    slim_masks = masks
    stage_seconds: dict[str, float] = {}
    fine_profile = os.environ.get("LLAMA_FHE_FINE_PROFILE", "") == "1"

    def fine_mark(name: str, started: float) -> float:
        """Close one opt-in, synchronization-delimited profile interval."""

        sync_device(crypto_context.device)
        finished = time.perf_counter()
        stage_seconds[str(name)] = float(finished - started)
        return finished

    y_ciphers: tuple[object, ...] = ()
    success = False
    try:
        start = time.perf_counter()
        exp_entry = config.layer_exp[layer_idx]
        score_multiplier = score_scale / float(2**k_layer)
        public_shift = -float(config.U_hat_for(layer_idx)) / float(2**k_layer)
        invalid_value = float(config.stats_min_for(layer_idx)) / float(2**k_layer)
        # In the paired checkpoint path each complex split returns
        # ``2*Re, 2*Im``.  Pre-map and halve the two real streams inside the
        # score plaintext that already consumes this level.  The split then
        # recovers the exact Chebyshev-domain input, so both the extra 0.5
        # rescale and the polynomial domain-map level disappear.
        if pair_pre_exp_bootstrap:
            fit_lo, fit_hi = (float(value) for value in exp_entry["fit_interval"])
            map_alpha = 2.0 / (fit_hi - fit_lo)
            map_beta = -1.0 - 2.0 * fit_lo / (fit_hi - fit_lo)
            score_multiplier *= 0.5 * map_alpha
            public_shift = 0.5 * (map_alpha * public_shift + map_beta)
            invalid_value *= 0.5 * map_alpha
        # A fused Prefix stream has already folded this public coefficient
        # into the three source masks used to merge Content and Prefix.
        y_ciphers = (
            _transform_delta_pages(
                score_ciphers,
                tuple(mask * score_multiplier for mask in page_valid),
                pair_count=pair_count,
                name="softmax.delta.valid_scale",
                crypto_context=crypto_context,
            )
            if fused_prefix_plan is None
            else tuple(cipher.deep_copy() for cipher in score_ciphers)
        )
        shifted: list[object] = []
        for cipher in y_ciphers:
            encoded = crypto_context.fhe.encode_scalar(
                public_shift,
                cur_limbs=int(cipher.state.cur_limbs),
                scale_degree=int(cipher.state.scale_degree),
                scaling_factor=cipher.state.scaling_factor,
                context=crypto_context.context,
            )
            shifted.append(
                crypto_context.fhe.homo_add_scalar(
                    cipher, encoded, crypto_context.context
                )
            )
        for cipher in y_ciphers:
            release_if_supported(cipher)
        cipher = None
        y_ciphers = tuple(shifted)
        shifted.clear()
        invalid_correction = tuple(
            (1.0 - mask) * invalid_value for mask in page_valid
        )
        corrected = _transform_delta_pages(
            y_ciphers,
            invalid_correction,
            pair_count=pair_count,
            name="softmax.delta.invalid",
            crypto_context=crypto_context,
            add=True,
        )
        for cipher in y_ciphers:
            release_if_supported(cipher)
        cipher = None
        y_ciphers = corrected
        corrected = ()
        _softmax_diagnostic_range(
            "shift_scale_mask", y_ciphers, crypto_context=crypto_context
        )
        stage_seconds["shift_scale_mask"] = time.perf_counter() - start

        if bool(refresh_before_exp) and bool(refresh_after_exp):
            raise ValueError(
                "Softmax main stream cannot be refreshed both before and after exp."
            )
        if bool(refresh_before_exp):
            if bootstrap_operator is None:
                raise ValueError(
                    "refresh_before_exp requires a prepared bootstrap operator."
                )
            start = time.perf_counter()
            refreshed_y: list[object] = []
            paired_y: tuple[object, ...] = ()
            try:
                checkpoint_inputs = y_ciphers
                if pair_pre_exp_bootstrap:
                    paired_y = pair_real_ciphers(
                        y_ciphers, crypto_context=crypto_context
                    )
                    checkpoint_inputs = paired_y
                sources: list[object] = []
                try:
                    for cipher in checkpoint_inputs:
                        sources.append(
                            bootstrap_operator.prepare_two_pass_input(
                                cipher,
                                iterations=main_bootstrap_iterations,
                                use_deep_program=main_bootstrap_use_deep,
                            )
                            if fused_prefix_plan is not None
                            else cipher
                        )
                    refresh_many = getattr(bootstrap_operator, "refresh_many", None)
                    if refresh_many is not None:
                        # the whole checkpoint in one call: batched C2S / EvalMod /
                        # S2C when the operator has a batch backend (HOLMES_BATCH_BS=1),
                        # the same per-cipher refresh loop otherwise
                        refreshed_y.extend(
                            refresh_many(
                                sources,
                                iterations=main_bootstrap_iterations,
                                precision_bits=int(checkpoint_precision_bits),
                                match_two_pass_level=bool(
                                    match_single_pass_bootstrap_level
                                ),
                                use_deep_program=bool(main_bootstrap_use_deep),
                            )
                        )
                    else:
                        for source in sources:
                            refreshed_y.append(
                                bootstrap_operator.refresh(
                                    source,
                                    iterations=main_bootstrap_iterations,
                                    precision_bits=int(checkpoint_precision_bits),
                                    match_two_pass_level=bool(
                                        match_single_pass_bootstrap_level
                                    ),
                                    use_deep_program=bool(main_bootstrap_use_deep),
                                )
                            )
                finally:
                    for source, cipher in zip(sources, checkpoint_inputs):
                        if source is not cipher:
                            release_if_supported(source)
            except Exception:
                for cipher in refreshed_y:
                    release_if_supported(cipher)
                raise
            finally:
                for cipher in paired_y:
                    release_if_supported(cipher)
                cipher = None
                paired_y = ()
                checkpoint_inputs = ()
            for cipher in y_ciphers:
                release_if_supported(cipher)
            cipher = None
            if pair_pre_exp_bootstrap:
                y_ciphers = ()
                try:
                    y_ciphers = split_complex_ciphers_twice(
                        tuple(refreshed_y), crypto_context=crypto_context
                    )
                finally:
                    for cipher in refreshed_y:
                        release_if_supported(cipher)
                    cipher = None
            else:
                y_ciphers = tuple(refreshed_y)
            refreshed_y.clear()
            _softmax_diagnostic_range(
                "pre_exp_bootstrap", y_ciphers, crypto_context=crypto_context
            )
            stage_seconds["pre_exp_bootstrap"] = (
                time.perf_counter() - start
            )

        start = time.perf_counter()
        exp_outputs: list[object] = []
        for cipher in y_ciphers:
            exp_outputs.append(
                eval_exp_cipher(
                    cipher,
                    exp_entry,
                    crypto_context=crypto_context,
                    input_is_chebyshev_mapped=pair_pre_exp_bootstrap,
                )
            )
        cipher = None
        # A normal polynomial evaluation consumes its input while mapping the
        # domain.  The paired path has already performed that map publicly,
        # so release the still-owned split inputs explicitly.
        if pair_pre_exp_bootstrap:
            for cipher in y_ciphers:
                release_if_supported(cipher)
            cipher = None
        y_ciphers = tuple(exp_outputs)
        exp_outputs.clear()
        if page_count > 1:
            from .multi_page import delta_page_exp_amplitude_mask

            amplitude = delta_page_exp_amplitude_mask(
                layout,
                page_count=page_count,
                query_valid_mask=token_valid_mask,
            )
            post_exp_masks = tuple(mask * amplitude for mask in page_valid)
        else:
            post_exp_masks = page_valid
        masked_exp = _transform_delta_pages(
            y_ciphers,
            post_exp_masks,
            pair_count=pair_count,
            name="softmax.delta.post_exp_valid",
            crypto_context=crypto_context,
        )
        for cipher in y_ciphers:
            release_if_supported(cipher)
        cipher = None
        y_ciphers = masked_exp
        masked_exp = ()
        _softmax_diagnostic_range(
            "exp", y_ciphers, crypto_context=crypto_context
        )
        stage_seconds["exp"] = time.perf_counter() - start

        if extra_softmax_checkpoint_enabled(
            "LLAMA_FHE_SOFTMAX_POST_EXP_BOOTSTRAP",
            layer_idx,
            SLIM_POST_EXP_LAYERS,
            bootstrap_operator,
        ):
            if bootstrap_operator is None:
                raise ValueError(
                    "post-exp bootstrap requires a prepared bootstrap operator."
                )
            start = time.perf_counter()
            y_ciphers = _paired_checkpoint_refresh(
                y_ciphers,
                bootstrap_operator=bootstrap_operator,
                label_prefix="alg2.post_exp",
                main_bootstrap_iterations=main_bootstrap_iterations,
                checkpoint_precision_bits=int(checkpoint_precision_bits),
                match_single_pass_bootstrap_level=bool(
                    match_single_pass_bootstrap_level
                ),
                crypto_context=crypto_context,
            )
            _softmax_diagnostic_range(
                "post_exp_bootstrap",
                y_ciphers,
                crypto_context=crypto_context,
            )
            stage_seconds["post_exp_bootstrap"] = (
                time.perf_counter() - start
            )

        if bool(refresh_after_exp):
            if bootstrap_operator is None:
                raise ValueError(
                    "refresh_after_exp requires a prepared bootstrap operator."
                )
            start = time.perf_counter()
            refreshed_y: list[object] = []
            try:
                for index, cipher in enumerate(y_ciphers):
                    refreshed_y.append(
                        bootstrap_operator.refresh(
                            cipher,
                            iterations=main_bootstrap_iterations,
                            precision_bits=int(checkpoint_precision_bits),
                            match_two_pass_level=bool(
                                match_single_pass_bootstrap_level
                            ),
                        )
                    )
            except Exception:
                for cipher in refreshed_y:
                    release_if_supported(cipher)
                raise
            for cipher in y_ciphers:
                release_if_supported(cipher)
            cipher = None
            y_ciphers = tuple(refreshed_y)
            refreshed_y.clear()
            stage_seconds["post_exp_bootstrap"] = (
                time.perf_counter() - start
            )

        rough = config.invsqrt_alg2["j1_rough"]
        precise = config.invsqrt_alg2["j2_precise"]
        for round_index in range(1, k_layer + 1):
            if fine_profile:
                sync_device(crypto_context.device)
            round_start = time.perf_counter()
            fine_started = round_start
            squared_for_sum = tuple(
                _square_rescale(cipher, crypto_context=crypto_context)
                for cipher in y_ciphers
            )
            if page_count > 1:
                from .multi_page import delta_reduce_page_squares_fhe

                reduced = delta_reduce_page_squares_fhe(
                    tuple(
                        squared_for_sum[index * pair_count : (index + 1) * pair_count]
                        for index in range(page_count)
                    ),
                    layout=layout,
                    crypto_context=crypto_context,
                )
            else:
                row_sum_input = (
                    squared_for_sum
                    if fused_prefix_plan is None
                    else fused_prefix_plan.row_sum_input(
                        squared_for_sum, crypto_context=crypto_context
                    )
                )
                try:
                    reduced = delta_reduce_rows_fhe(
                        row_sum_input,
                        layout=layout,
                        crypto_context=crypto_context,
                    )
                finally:
                    if fused_prefix_plan is not None:
                        for cipher in row_sum_input:
                            release_if_supported(cipher)
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.square_reduce",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.row_sum",
                reduced,
                crypto_context=crypto_context,
            )
            if reuse_round_square:
                previous_y = y_ciphers
                y_ciphers = ()
                for cipher in previous_y:
                    release_if_supported(cipher)
                cipher = None
                previous_y = ()
            else:
                for cipher in squared_for_sum:
                    release_if_supported(cipher)
                cipher = None
                squared_for_sum = ()
            poly = config.poly_for_layer(
                rough if round_index == 1 else precise,
                layer_idx,
            )
            mapped_mask_prefix = f"softmax.slim.mapped.{round_index}"
            mapped_masks = None
            fuse_round = fuse_slim_chebyshev_map
            if fuse_round:
                cheb = poly.get("chebyshev_fhe") or poly
                lo, hi = (float(value) for value in cheb["fit_interval"])
                if not _mapped_mask_coefficient_fits(2.0 / (hi - lo), reduced[0]):
                    # The fused alpha factor would push the mask plaintext past
                    # the encoder's 2^61 cap at this cipher scale (seen with
                    # the periodic slim pack at coeff-layer 27: alpha 6.4 at a
                    # 2^59 scale).  Fall back to the unfused map this round.
                    fuse_round = False
            if fuse_round:
                mapped_masks = build_delta_softmax_mask_bundle(
                    layout,
                    crypto_context=crypto_context,
                    coefficient=2.0 / (hi - lo),
                    name_prefix=mapped_mask_prefix,
                    period_rows=slim_period_rows,
                )
            slim = delta_pack_row_sums_slim_fhe(
                reduced,
                layout=layout,
                crypto_context=crypto_context,
                masks=(
                    mapped_masks if mapped_masks is not None else slim_masks
                ),
                mask_name_prefix=(
                    mapped_mask_prefix
                    if mapped_masks is not None
                    else "softmax.slim"
                ),
            )
            mapped_masks = None
            for cipher in reduced:
                release_if_supported(cipher)
            cipher = None
            reduced = ()

            sanitize_slim = (
                _fill_mapped_delta_slim_poly_input
                if fuse_round
                else _fill_delta_slim_poly_input
                if direct_slim_inactive_fill
                else _sanitize_delta_slim_poly_input
            )
            sanitized_slim = sanitize_slim(
                slim,
                poly,
                layout=layout,
                crypto_context=crypto_context,
                name=f"softmax.delta.j{round_index}.slim",
                period_rows=slim_period_rows,
            )
            release_if_supported(slim)
            slim = None
            sanitize_slim = None
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.slim_pack",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.slim_input",
                (sanitized_slim,),
                crypto_context=crypto_context,
            )
            # A streamed layer can reach this compact boundary with too few
            # limbs for the deep inverse-square-root polynomial.  Refresh its
            # input only when needed; the normal Layer0 path retains the
            # original post-polynomial refresh schedule.
            #
            # This refresh feeds the rsqrt polynomial directly. Use two-pass
            # Meta-BTS to control imaginary error near the lower fit edge;
            # polynomial evaluation can amplify that error across rounds.
            polynomial_depth = chebyshev_ps_mul_depth(poly["coefficients"])
            total_polynomial_depth = polynomial_depth + (
                0 if fuse_round else 1
            )
            if (
                bootstrap_operator is not None
                and int(sanitized_slim.state.cur_limbs)
                <= int(total_polynomial_depth) + 1
            ):
                previous = sanitized_slim
                sanitized_slim = bootstrap_operator.refresh(
                    previous,
                    iterations=2,
                    match_two_pass_level=False,
                    periodic_slots=slim_refresh_periodic_slots,
                )
                if slim_refresh_periodic_slots is not None:
                    slim_sparse_bootstraps += 1
                if sanitized_slim is not previous:
                    release_if_supported(previous)
                previous = None
                _softmax_diagnostic_range(
                    f"round_{round_index}.slim_input_refreshed",
                    (sanitized_slim,),
                    crypto_context=crypto_context,
                )
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.slim_input_refresh",
                    fine_started,
                )
            lambda_cipher = eval_invsqrt_cipher(
                sanitized_slim,
                poly,
                crypto_context=crypto_context,
                input_is_chebyshev_mapped=fuse_round,
            )
            if fuse_round:
                release_if_supported(sanitized_slim)
            sanitized_slim = None
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.lambda_polynomial",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.lambda_poly",
                (lambda_cipher,),
                crypto_context=crypto_context,
            )
            if bootstrap_operator is not None:
                previous = lambda_cipher
                lambda_cipher = bootstrap_operator.refresh(
                    previous,
                    iterations=lambda_bootstrap_iterations,
                    match_two_pass_level=bool(
                        match_single_pass_bootstrap_level
                    ),
                    periodic_slots=slim_refresh_periodic_slots,
                )
                if slim_refresh_periodic_slots is not None:
                    slim_sparse_bootstraps += 1
                if lambda_cipher is not previous:
                    release_if_supported(previous)
                previous = None
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.lambda_refresh",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.lambda_refreshed",
                (lambda_cipher,),
                crypto_context=crypto_context,
            )
            lambda_for_broadcast = lambda_cipher
            if reuse_round_square:
                lambda_for_broadcast = _square_rescale(
                    lambda_cipher, crypto_context=crypto_context
                )
                release_if_supported(lambda_cipher)
                lambda_cipher = None
            lambda_pairs = delta_broadcast_slim_rows_fhe(
                lambda_for_broadcast,
                layout=layout,
                crypto_context=crypto_context,
                masks=masks,
                period_rows=slim_period_rows,
            )
            if fused_prefix_plan is not None:
                adjusted_pairs = fused_prefix_plan.lambda_for_main(
                    lambda_pairs, crypto_context=crypto_context
                )
                for cipher in lambda_pairs:
                    release_if_supported(cipher)
                lambda_pairs = adjusted_pairs
            release_if_supported(lambda_for_broadcast)
            lambda_for_broadcast = None
            lambda_cipher = None
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.lambda_square_broadcast",
                    fine_started,
                )

            if reuse_round_square:
                squared = tuple(
                    _mul_cipher_rescale(
                        left,
                        lambda_pairs[index % pair_count],
                        crypto_context=crypto_context,
                    )
                    for index, left in enumerate(squared_for_sum)
                )
                for cipher in squared_for_sum + lambda_pairs:
                    release_if_supported(cipher)
                cipher = None
                squared_for_sum = ()
                lambda_pairs = ()
            else:
                scaled = tuple(
                    _mul_cipher_rescale(
                        left,
                        lambda_pairs[index % pair_count],
                        crypto_context=crypto_context,
                    )
                    for index, left in enumerate(y_ciphers)
                )
                for cipher in y_ciphers + lambda_pairs:
                    release_if_supported(cipher)
                cipher = None
                y_ciphers = ()
                lambda_pairs = ()
                squared = tuple(
                    _square_rescale(cipher, crypto_context=crypto_context)
                    for cipher in scaled
                )
                for cipher in scaled:
                    release_if_supported(cipher)
                cipher = None
                scaled = ()
            # ``y_ciphers`` was causally masked immediately after exp.
            # Multiplication by a row scalar and squaring cannot turn an
            # exact invalid zero into model data, so keep the squared stream
            # directly instead of spending one PT-CT level per round on the
            # same public mask.  The post-exp mask remains the sole explicit
            # causal-zero enforcement before the four Alg2 rounds.
            y_ciphers = squared
            squared = ()
            if fine_profile:
                fine_started = fine_mark(
                    f"alg2_round_{round_index}.main_update",
                    fine_started,
                )
            _softmax_diagnostic_range(
                f"round_{round_index}.output",
                y_ciphers,
                crypto_context=crypto_context,
            )
            stage_seconds[f"alg2_round_{round_index}"] = (
                time.perf_counter() - round_start
            )
            mid_round = (int(k_layer) + 1) // 2
            if (
                round_index == mid_round
                and mid_round < int(k_layer)
                and extra_softmax_checkpoint_enabled(
                    "LLAMA_FHE_SOFTMAX_MID_ALG2_BOOTSTRAP",
                    layer_idx,
                    SLIM_MID_ALG2_LAYERS,
                    bootstrap_operator,
                )
            ):
                if bootstrap_operator is None:
                    raise ValueError(
                        "mid-Alg2 bootstrap requires a prepared bootstrap "
                        "operator."
                    )
                start = time.perf_counter()
                y_ciphers = _paired_checkpoint_refresh(
                    y_ciphers,
                    bootstrap_operator=bootstrap_operator,
                    label_prefix=f"alg2.mid_round_{round_index}",
                    main_bootstrap_iterations=main_bootstrap_iterations,
                    checkpoint_precision_bits=int(checkpoint_precision_bits),
                    match_single_pass_bootstrap_level=bool(
                        match_single_pass_bootstrap_level
                    ),
                    crypto_context=crypto_context,
                )
                _softmax_diagnostic_range(
                    f"mid_alg2_bootstrap_after_round_{round_index}",
                    y_ciphers,
                    crypto_context=crypto_context,
                )
                stage_seconds["mid_alg2_bootstrap"] = (
                    time.perf_counter() - start
                )


        if extra_softmax_checkpoint_enabled(
            "LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP",
            layer_idx,
            SLIM_POST_ALG2_LAYERS,
            bootstrap_operator,
            rounds=int(k_layer),
            min_rounds=SLIM_POST_ALG2_MIN_ROUNDS,
        ):
            if bootstrap_operator is None:
                raise ValueError(
                    "post-Alg2 bootstrap requires a prepared bootstrap operator."
                )
            _softmax_diagnostic_range(
                "pre_post_alg2_bootstrap",
                y_ciphers,
                crypto_context=crypto_context,
            )
            start = time.perf_counter()
            y_ciphers = _paired_checkpoint_refresh(
                y_ciphers,
                bootstrap_operator=bootstrap_operator,
                label_prefix="alg2.post_alg2",
                main_bootstrap_iterations=main_bootstrap_iterations,
                checkpoint_precision_bits=int(checkpoint_precision_bits),
                match_single_pass_bootstrap_level=bool(
                    match_single_pass_bootstrap_level
                ),
                crypto_context=crypto_context,
            )
            _softmax_diagnostic_range(
                "post_alg2_bootstrap",
                y_ciphers,
                crypto_context=crypto_context,
            )
            stage_seconds["post_alg2_bootstrap"] = (
                time.perf_counter() - start
            )

        success = True
        return DeltaSoftmaxAlg2Result(
            probability_ciphers=y_ciphers,
            stage_seconds={name: float(value) for name, value in stage_seconds.items()},
            output_levels=tuple(
                sorted({int(crypto_context.level_for_cipher(cipher)) for cipher in y_ciphers})
            ),
            slim_period_rows=slim_period_rows,
            slim_sparse_bootstraps=int(slim_sparse_bootstraps),
        )
    finally:
        if not success:
            for cipher in y_ciphers:
                release_if_supported(cipher)
