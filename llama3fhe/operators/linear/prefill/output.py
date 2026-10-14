from __future__ import annotations

"""Feature-major output projection.

Attention owns the ``O_C -> PairFM -> feature-major`` layout conversion.
This module begins at the linear boundary: eight complex token-pair carriers
whose feature channels are in query-pair order, and ends at the sixteen real
canonical feature-major shards consumed by the residual path.
"""

import time

import numpy as np

from llama3fhe.backend import profile_sync_device, release_all, release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

from ..complex import pair_real_ciphers, split_complex_ciphers_twice
from ..config import (
    LLAMA3_LINEAR_SCHEDULES,
    LinearOperatorConfig,
    linear_rotations,
    structured_weight_packing_enabled,
)
from ..operator import LinearOperator
from ..structured_weight import structured_linear_weight


def pair_interleaved_packed_to_logical(
    head_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    head_dim: int,
) -> np.ndarray:
    """Map physical ``(pair, feature, lane)`` channels to model channels."""

    mapping = np.asarray(
        [
            head * int(head_dim) + feature
            for first, second in head_pairs
            for feature in range(int(head_dim))
            for head in (int(first), int(second))
        ],
        dtype=np.int64,
    )
    if mapping.size and not np.array_equal(
        np.sort(mapping), np.arange(mapping.size, dtype=np.int64)
    ):
        raise ValueError("head pairs must cover every logical head exactly once.")
    return mapping


def _validate_w_o_layouts(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> int:
    if int(feature_layout.slots) != int(attention_layout.slots):
        raise ValueError("feature-major and attention slot counts must match.")
    if int(feature_layout.seq_len) != int(attention_layout.seq_len):
        raise ValueError("feature-major and attention sequence lengths must match.")
    dimension = int(feature_layout.hidden_dim)
    if dimension != int(attention_layout.query_heads) * int(attention_layout.head_dim):
        raise ValueError("feature-major dimension must equal query_heads*head_dim.")
    if int(feature_layout.cipher_count) != int(attention_layout.query_pair_count):
        raise ValueError("W_O requires one feature-major shard per query pair.")
    if int(feature_layout.seq_len) != (
        int(feature_layout.cipher_count) * int(feature_layout.tokens_per_cipher)
    ):
        raise ValueError("W_O requires completely filled feature-major token shards.")
    return int(feature_layout.cipher_count)


def _w_o_carriers_fhe(
    input_carriers: tuple[object, ...] | list[object],
    weight: np.ndarray,
    *,
    expected_carrier_count: int,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    operator_config: LinearOperatorConfig | None = None,
    dtype: np.dtype = np.float64,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Apply the production W_O kernel to a carrier subset.

    Every PairFM carrier is an independent token-pair shard. Keeping this
    implementation shared by the full single-card entry point and the
    resident one-carrier entry point guarantees that their plaintext weight
    transform, BSGS schedule, rescale, and complex split remain identical.
    """

    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.w_o
        if operator_config is None
        else operator_config
    )
    # The stage code below reads the policy as plain locals; binding them here
    # keeps one object at the boundary.
    baby_steps = int(operator_config.baby_steps)
    baby_anchor_step = int(operator_config.baby_anchor_step)
    max_plaintext_rows = int(operator_config.max_plaintext_rows)
    reuse_baby_rotations = bool(operator_config.reuse_baby_rotations)
    hoist_strategy = str(operator_config.hoist_strategy)

    carriers = tuple(input_carriers)
    shard_count = _validate_w_o_layouts(feature_layout, attention_layout)
    expected_carrier_count = int(expected_carrier_count)
    if expected_carrier_count <= 0 or expected_carrier_count > shard_count // 2:
        raise ValueError("expected W_O carrier count is outside the layout")
    if len(carriers) != expected_carrier_count:
        raise ValueError(
            f"feature-major W_O expects {expected_carrier_count} carriers."
        )
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in carriers}
    if len(levels) != 1:
        raise ValueError("feature-major W_O inputs must enter at one common level.")
    if int(crypto_context.max_slots) != int(feature_layout.slots):
        raise ValueError("crypto context and feature-major slot counts must match.")

    dimension = int(feature_layout.hidden_dim)
    weight = np.asarray(weight)
    if weight.shape != (dimension, dimension):
        raise ValueError(f"W_O must have shape {(dimension, dimension)}, got {weight.shape}.")
    weight_transform_start = time.perf_counter()
    mapping = pair_interleaved_packed_to_logical(
        attention_layout.query_head_pairs(),
        head_dim=int(attention_layout.head_dim),
    )
    if structured_weight_packing_enabled():
        rows = np.arange(dimension, dtype=np.int64)
        physical_weight = structured_linear_weight(
            (weight,),
            dimension=dimension,
            real_source=np.zeros((dimension,), dtype=np.int64),
            real_row=rows,
            real_scale=0.5,
            column=mapping,
        )
    else:
        physical_weight = np.asarray(0.5 * weight[:, mapping], dtype=dtype)
    if profile is not None:
        profile["weight_transform_cpu"] = float(
            time.perf_counter() - weight_transform_start
        )

    projected: tuple[object, ...] = ()
    outputs: tuple[object, ...] = ()
    projection_profile: dict[str, float] = {}
    projection_operator = LinearOperator(
        crypto_context=crypto_context,
        token_lanes=int(feature_layout.tokens_per_cipher),
        config=LinearOperatorConfig(
            baby_steps=int(baby_steps),
            baby_anchor_step=int(baby_anchor_step),
            max_plaintext_rows=int(max_plaintext_rows),
            reuse_baby_rotations=bool(reuse_baby_rotations),
            hoist_strategy=str(hoist_strategy),
        ),
    )
    try:
        projected, projection_profile = projection_operator.project(
            carriers,
            physical_weight,
            dtype=np.dtype(dtype),
        )
        split_start = time.perf_counter()
        outputs = split_complex_ciphers_twice(
            projected, crypto_context=crypto_context
        )
        profile_sync_device(crypto_context.device)
        if profile is not None:
            profile.update(
                {str(key): float(value) for key, value in projection_profile.items()}
            )
            profile["output_split"] = float(time.perf_counter() - split_start)
        return outputs
    except Exception:
        release_all(outputs)
        raise
    finally:
        projection_operator.close()
        release_all(projected)


def w_o_fhe(
    input_carriers: tuple[object, ...] | list[object],
    weight: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    operator_config: LinearOperatorConfig | None = None,
    dtype: np.dtype = np.float64,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Apply W_O to all eight carriers and return sixteen FM shards.

    Each input carrier is X[2r] + i*X[2r+1]. The input feature channels are
    pair-major. A 0.5 and pair-major-to-logical permutation are folded into
    the plaintext weight, so the final level-free split is unscaled.
    """

    expected = int(feature_layout.cipher_count) // 2
    return _w_o_carriers_fhe(
        input_carriers,
        weight,
        expected_carrier_count=expected,
        feature_layout=feature_layout,
        attention_layout=attention_layout,
        crypto_context=crypto_context,
        operator_config=operator_config,
        dtype=dtype,
        profile=profile,
    )


def w_o_carrier_shard_fhe(
    input_carrier: object,
    weight: np.ndarray,
    *,
    carrier_index: int,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    operator_config: LinearOperatorConfig | None = None,
    dtype: np.dtype = np.float64,
    profile: dict[str, float] | None = None,
) -> tuple[object, object]:
    """Project one resident PairFM carrier into its two FM shards.

    carrier_index fixes ownership/global output indices but does not alter
    the arithmetic: all token-pair carriers use the same complete W_O matrix.
    No gather, reduction, or ciphertext transfer is required.
    """

    carrier_count = int(feature_layout.cipher_count) // 2
    carrier_index = int(carrier_index)
    if carrier_index < 0 or carrier_index >= carrier_count:
        raise ValueError(
            f"carrier_index must be in [0,{carrier_count}), got {carrier_index}"
        )
    outputs = _w_o_carriers_fhe(
        (input_carrier,),
        weight,
        expected_carrier_count=1,
        feature_layout=feature_layout,
        attention_layout=attention_layout,
        crypto_context=crypto_context,
        operator_config=operator_config,
        dtype=dtype,
        profile=profile,
    )
    if len(outputs) != 2:
        release_all(outputs)
        raise RuntimeError("one W_O carrier must produce exactly two FM shards")
    return outputs[0], outputs[1]


def w_o_carrier_shards_fhe(
    input_carriers: tuple[object, ...] | list[object],
    weight: np.ndarray,
    *,
    carrier_indices: tuple[int, ...] | list[int],
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    operator_config: LinearOperatorConfig | None = None,
    dtype: np.dtype = np.float64,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Project the adjacent resident PairFM carriers of one rank into FM shards.

    Generalises :func:`w_o_carrier_shard_fhe` to a rank that owns several
    carriers (T2 with fewer than eight ranks): the carriers share one
    plaintext weight transform and one BSGS operator.  The result is the
    interleaved ``2*Re, 2*Im`` split, i.e. FM ciphers ``2c, 2c+1`` for each
    carrier ``c`` in ``carrier_indices`` order -- adjacent carriers therefore
    give the adjacent FM run ``2*c0 .. 2*c0 + 2*len - 1``.
    """

    carriers = tuple(input_carriers)
    indices = tuple(int(v) for v in carrier_indices)
    carrier_count = int(feature_layout.cipher_count) // 2
    if not indices:
        raise ValueError("w_o_carrier_shards_fhe needs at least one carrier")
    if len(carriers) != len(indices):
        raise ValueError(
            f"{len(carriers)} carriers for carrier_indices {indices}"
        )
    for index in indices:
        if index < 0 or index >= carrier_count:
            raise ValueError(
                f"carrier_index must be in [0,{carrier_count}), got {index}"
            )
    if indices != tuple(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"carrier_indices must be adjacent, got {indices}")
    outputs = _w_o_carriers_fhe(
        carriers,
        weight,
        expected_carrier_count=len(carriers),
        feature_layout=feature_layout,
        attention_layout=attention_layout,
        crypto_context=crypto_context,
        operator_config=operator_config,
        dtype=dtype,
        profile=profile,
    )
    if len(outputs) != 2 * len(carriers):
        release_all(outputs)
        raise RuntimeError(
            f"{len(carriers)} W_O carriers must produce {2 * len(carriers)} FM shards"
        )
    return outputs


def _validate_decode_w_o_layout(feature_layout: FeatureMajorPrefillLayout) -> None:
    if int(feature_layout.seq_len) != 1:
        raise ValueError("Decode W_O expects DecodeFM seq_len=1.")
    if int(feature_layout.cipher_count) != 1:
        raise ValueError("Decode W_O expects one feature-major ciphertext.")


def decode_w_o_rotations(
    feature_layout: FeatureMajorPrefillLayout,
    *,
    operator_config: LinearOperatorConfig | None = None,
) -> tuple[int, ...]:
    """BSGS + conjugation keys for one-ciphertext Decode W_O."""

    _validate_decode_w_o_layout(feature_layout)
    return linear_rotations(
        dimension=int(feature_layout.hidden_dim),
        token_lanes=int(feature_layout.tokens_per_cipher),
        slots=int(feature_layout.slots),
        include_conjugation=True,
        operator_config=(
            LLAMA3_LINEAR_SCHEDULES.w_o
            if operator_config is None
            else operator_config
        ),
    )


def decode_w_o_numpy(
    hidden: np.ndarray,
    weight: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
) -> np.ndarray:
    """``hidden @ W_Oᵀ`` on a DecodeFM logical row."""

    _validate_decode_w_o_layout(feature_layout)
    hidden = np.asarray(hidden, dtype=np.float32)
    expected = (1, int(feature_layout.hidden_dim))
    if hidden.shape == (int(feature_layout.cipher_count), int(feature_layout.slots)):
        hidden = feature_layout.unpack(hidden)
    if hidden.shape == (int(feature_layout.hidden_dim),):
        hidden = hidden.reshape(1, -1)
    if hidden.shape != expected:
        raise ValueError(f"Decode W_O hidden must have shape {expected}, got {hidden.shape}.")
    weight = np.asarray(weight, dtype=np.float32)
    dimension = int(feature_layout.hidden_dim)
    if weight.shape != (dimension, dimension):
        raise ValueError(f"W_O must have shape {(dimension, dimension)}, got {weight.shape}.")
    return hidden @ weight.T


def decode_w_o_fhe(
    input_ciphers: tuple[object, ...] | list[object],
    weight: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
    crypto_context,
    operator_config: LinearOperatorConfig | None = None,
    dtype: np.dtype = np.float64,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Apply W_O to one DecodeFM ciphertext.  Skips Prefill PairFM.

    Pairs the single real stream with a zero pad (same algebra as Prefill),
    folds the ``0.5`` split scale into the weight, and returns the real
    DecodeFM shard.  EasyFHE linear kernels are unchanged.
    """

    _validate_decode_w_o_layout(feature_layout)
    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.w_o
        if operator_config is None
        else operator_config
    )
    baby_steps = int(operator_config.baby_steps)
    baby_anchor_step = int(operator_config.baby_anchor_step)
    max_plaintext_rows = int(operator_config.max_plaintext_rows)
    reuse_baby_rotations = bool(operator_config.reuse_baby_rotations)
    hoist_strategy = str(operator_config.hoist_strategy)

    carriers_in = tuple(input_ciphers)
    if len(carriers_in) != 1:
        raise ValueError(
            f"Decode W_O expects 1 DecodeFM ciphertext, got {len(carriers_in)}."
        )
    if int(crypto_context.max_slots) != int(feature_layout.slots):
        raise ValueError("crypto context and DecodeFM slot counts must match.")

    dimension = int(feature_layout.hidden_dim)
    weight = np.asarray(weight)
    if weight.shape != (dimension, dimension):
        raise ValueError(
            f"W_O must have shape {(dimension, dimension)}, got {weight.shape}."
        )
    weight_transform_start = time.perf_counter()
    mapping = np.arange(dimension, dtype=np.int64)
    if structured_weight_packing_enabled():
        rows = np.arange(dimension, dtype=np.int64)
        physical_weight = structured_linear_weight(
            (weight,),
            dimension=dimension,
            real_source=np.zeros((dimension,), dtype=np.int64),
            real_row=rows,
            real_scale=0.5,
            column=mapping,
        )
    else:
        physical_weight = np.asarray(0.5 * weight, dtype=dtype)
    if profile is not None:
        profile["weight_transform_cpu"] = float(
            time.perf_counter() - weight_transform_start
        )

    paired: tuple[object, ...] = ()
    projected: tuple[object, ...] = ()
    split_outputs: tuple[object, ...] = ()
    projection_profile: dict[str, float] = {}
    projection_operator = LinearOperator(
        crypto_context=crypto_context,
        token_lanes=int(feature_layout.tokens_per_cipher),
        config=LinearOperatorConfig(
            baby_steps=int(baby_steps),
            baby_anchor_step=int(baby_anchor_step),
            max_plaintext_rows=int(max_plaintext_rows),
            reuse_baby_rotations=bool(reuse_baby_rotations),
            hoist_strategy=str(hoist_strategy),
        ),
    )
    try:
        pair_start = time.perf_counter()
        paired = pair_real_ciphers(
            carriers_in, crypto_context=crypto_context, pad_odd=True
        )
        if profile is not None:
            profile["input_pairing"] = float(time.perf_counter() - pair_start)
        projected, projection_profile = projection_operator.project(
            paired,
            physical_weight,
            dtype=np.dtype(dtype),
        )
        split_start = time.perf_counter()
        split_outputs = split_complex_ciphers_twice(
            projected, crypto_context=crypto_context
        )
        profile_sync_device(crypto_context.device)
        if profile is not None:
            profile.update(
                {str(key): float(value) for key, value in projection_profile.items()}
            )
            profile["output_split"] = float(time.perf_counter() - split_start)
        if len(split_outputs) < 1:
            raise RuntimeError("Decode W_O split produced no real shard.")
        kept = split_outputs[0]
        dropped = split_outputs[1:]
        split_outputs = ()
        for cipher in dropped:
            release_if_supported(cipher)
        return (kept,)
    except Exception:
        release_all(split_outputs)
        raise
    finally:
        projection_operator.close()
        release_all(paired)
        release_all(projected)
