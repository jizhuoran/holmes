from __future__ import annotations

"""Token-shard parallelism for the production feature-major QKV.

The filename preserves the original experiment name, but the implemented
axis is deliberately *not* the 4096-wide source-feature axis.  A persistent
feature-major ciphertext owns eight complete tokens in production.  Adjacent
ciphertexts can therefore stay on one worker, be paired as ``x + i*y``, and
run through the existing QKV projection independently of every other token
pair.  Worker ``r`` consumes global FM indices ``(2*r, 2*r+1)`` and produces
carrier indices with the same numbers; no homomorphic partial-sum reduction is
required.

The structured Q/K/V weight sources, encoded diagonal routes, and rotation-key
union are still replicated on every worker.  What shrinks is the encrypted
projection batch: one paired input plus its conjugate instead of eight paired
inputs plus their conjugates.  Consequently the baby-rotation basis and the
per-input MAC workspace are one eighth of the unsharded production batch.
"""

import time
from dataclasses import dataclass

import numpy as np

from llama3fhe.backend import release_all
from llama3fhe.backend import profile_sync_device as sync_device  # stage-timer sync, no-op under T2_STAGE_SYNC=0
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

from .. import LLAMA3_LINEAR_SCHEDULES, LinearOperatorConfig
from ..config import structured_weight_packing_enabled
from ..prefill.qkv import pair_feature_major_ciphers
from ..prefill.qkv_kernel import (
    ComplexTokenDiagonalQKVComplexity,
    _carrier_shape,
    _prepare_qkv_weights,
    _projection_weights,
    _qkv_fhe_complex_token_diagonal_prepared,
    _structured_projection_weight,
    complex_token_diagonal_qkv_complexity,
)


@dataclass(frozen=True)
class QKVTokenShardSpec:
    """One worker's adjacent FM inputs and identically indexed outputs.

    ``input_indices`` is an even number of adjacent global FM indices
    (``n * shard_index .. n * shard_index + n - 1`` for ``n`` ciphertexts per
    worker): one adjacent pair on an 8-worker host, two pairs (four
    ciphertexts) on a 4-worker host.
    """

    rank: int
    shard_index: int
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    token_start: int
    token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_indices", tuple(int(v) for v in self.input_indices))
        object.__setattr__(self, "output_indices", tuple(int(v) for v in self.output_indices))

    @property
    def token_stop(self) -> int:
        return int(self.token_start) + int(self.token_count)

    @property
    def cipher_count(self) -> int:
        return len(self.input_indices)


@dataclass(frozen=True)
class QKVTokenShardPlan:
    """Static adjacent-FM-group-per-worker production assignment."""

    input_cipher_count: int
    output_cipher_count: int
    tokens_per_cipher: int
    assignments: tuple[QKVTokenShardSpec, ...]

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(int(spec.rank) for spec in self.assignments)

    def for_rank(self, rank: int) -> QKVTokenShardSpec:
        matches = [spec for spec in self.assignments if int(spec.rank) == int(rank)]
        if len(matches) != 1:
            raise KeyError(f"rank {rank} is not assigned exactly one QKV token shard")
        return matches[0]


def qkv_token_shard_plan(
    layout: FeatureMajorPrefillLayout,
    ranks: tuple[int, ...] | list[int],
) -> QKVTokenShardPlan:
    """Assign each adjacent FM group of ``cipher_count / len(ranks)`` ciphertexts to one worker.

    The group size must be even (the kernel pairs adjacent real FM ciphertexts
    into complex ones): the production boundary has 16 FM ciphertexts, so 8
    ranks own one adjacent pair each and 4 ranks own two.  Requiring an even
    ciphertext count avoids disguising a padded final token group as a full
    worker shard.
    """

    ranks = tuple(int(rank) for rank in ranks)
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be a non-empty unique sequence")
    cipher_count = int(layout.cipher_count)
    if cipher_count % 2:
        raise ValueError(
            "QKV token sharding requires an even feature-major cipher count, "
            f"got {cipher_count}"
        )
    pair_count = cipher_count // 2
    if pair_count % len(ranks):
        raise ValueError(
            "QKV token sharding assigns whole adjacent FM pairs per rank: "
            f"need a divisor of {pair_count} ranks for {cipher_count} ciphertexts, got {len(ranks)}"
        )
    per_rank = cipher_count // len(ranks)

    tokens_per_cipher = int(layout.tokens_per_cipher)
    assignments: list[QKVTokenShardSpec] = []
    for shard_index, rank in enumerate(ranks):
        first = per_rank * shard_index
        token_start = first * tokens_per_cipher
        token_stop = min(
            int(layout.seq_len), (first + per_rank) * tokens_per_cipher
        )
        if token_stop - token_start != per_rank * tokens_per_cipher:
            raise ValueError(
                f"QKV token sharding requires {per_rank} full FM token groups "
                f"per worker; shard {shard_index} contains "
                f"{token_stop - token_start} tokens"
            )
        indices = tuple(range(first, first + per_rank))
        assignments.append(
            QKVTokenShardSpec(
                rank=int(rank),
                shard_index=int(shard_index),
                input_indices=indices,
                output_indices=indices,
                token_start=int(token_start),
                token_count=int(token_stop - token_start),
            )
        )
    return QKVTokenShardPlan(
        input_cipher_count=cipher_count,
        output_cipher_count=cipher_count,
        tokens_per_cipher=tokens_per_cipher,
        assignments=tuple(assignments),
    )


def qkv_token_shard_resource_report(
    layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    ranks: tuple[int, ...] | list[int],
    *,
    operator_config: LinearOperatorConfig | None = None,
    weight_dtype: np.dtype = np.dtype(np.float32),
) -> dict[str, object]:
    """State exactly what the token-shard prototype does and saves."""

    plan = qkv_token_shard_plan(layout, ranks)
    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.qkv
        if operator_config is None
        else operator_config
    )
    hidden_dim = int(layout.hidden_dim)
    carrier_width = (
        int(attention_layout.query_heads) * int(attention_layout.head_dim) // 2
        + int(attention_layout.key_value_heads) * int(attention_layout.head_dim)
    )
    kwargs = {
        "dimension": hidden_dim,
        "slots": int(layout.slots),
        "carrier_width": carrier_width,
        "baby_steps": int(operator_config.baby_steps),
        "baby_anchor_step": int(operator_config.baby_anchor_step),
        "max_plaintext_rows": int(operator_config.max_plaintext_rows),
        "reuse_baby_rotations": bool(operator_config.reuse_baby_rotations),
    }
    full = complex_token_diagonal_qkv_complexity(
        seq_len=int(layout.seq_len), **kwargs
    )
    local = complex_token_diagonal_qkv_complexity(
        seq_len=int(plan.assignments[0].token_count), **kwargs
    )
    q_rows = int(attention_layout.query_heads) * int(attention_layout.head_dim)
    kv_rows = (
        int(attention_layout.key_value_heads) * int(attention_layout.head_dim)
    )
    weight_bytes = (q_rows + 2 * kv_rows) * hidden_dim * np.dtype(
        weight_dtype
    ).itemsize

    def fraction(numerator: int, denominator: int) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    return {
        "axis": "token_shard",
        "source_feature_partial_sum": False,
        "homomorphic_output_reduction": False,
        "public_qkv_fhe_accepts_local_pair": False,
        "execution_boundary": "_qkv_fhe_complex_token_diagonal_prepared",
        "worker_count": len(plan.assignments),
        "assignments": {
            str(spec.rank): {
                "input_indices": list(spec.input_indices),
                "output_indices": list(spec.output_indices),
                "token_range": [spec.token_start, spec.token_stop],
            }
            for spec in plan.assignments
        },
        "full": {
            "real_input_cipher_count": int(layout.cipher_count),
            "paired_input_cipher_count": int(full.input_cipher_count),
            "evaluated_cipher_count": int(full.evaluated_cipher_count),
            "output_carrier_count": int(full.output_cipher_count),
            "baby_rotation_operations": int(full.baby_rotations),
            "pt_ct_multiplications": int(full.pt_ct_multiplications),
        },
        "per_worker": {
            "local_seq_len": int(plan.assignments[0].token_count),
            "real_input_cipher_count": int(plan.assignments[0].cipher_count),
            "paired_input_cipher_count": int(local.input_cipher_count),
            "evaluated_cipher_count": int(local.evaluated_cipher_count),
            "output_carrier_count": int(local.output_cipher_count),
            "baby_rotation_operations": int(local.baby_rotations),
            "pt_ct_multiplications": int(local.pt_ct_multiplications),
            "encrypted_batch_fraction": fraction(
                local.evaluated_cipher_count, full.evaluated_cipher_count
            ),
            "baby_rotation_work_fraction": fraction(
                local.baby_rotations, full.baby_rotations
            ),
            "full_weight_source_bytes": int(weight_bytes),
            "weight_source_fraction": 1.0,
            "encoded_weight_plaintexts": int(local.encoded_weight_plaintexts),
            "rotation_key_union_replicated": True,
        },
    }


def qkv_carrier_reference(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    attention_layout: AttentionPairLayout,
    input_column_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Build the exact heterogeneous carrier oracle without a dense W."""

    prepared = _prepare_qkv_weights(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        attention_layout=attention_layout,
        input_column_scale=input_column_scale,
    )
    static = _carrier_shape(prepared, head_dim=int(attention_layout.head_dim))
    hidden = np.asarray(prepared.hidden_states, dtype=np.float64)
    query = hidden @ np.asarray(prepared.q_weight_for_pack, dtype=np.float64).T
    key = hidden @ np.asarray(prepared.k_weight_for_pack, dtype=np.float64).T
    value = hidden @ np.asarray(prepared.v_weight_for_pack, dtype=np.float64).T
    output = np.zeros(
        (int(prepared.seq_len), int(static.carrier_width)), dtype=np.complex128
    )
    head_dim = int(attention_layout.head_dim)
    folded_pair_width = head_dim
    pair_width = 2 * head_dim
    query_pairs = int(prepared.query_width) // pair_width
    kv_pairs = int(prepared.key_value_width) // pair_width
    for pair in range(query_pairs):
        dst = slice(
            pair * folded_pair_width, (pair + 1) * folded_pair_width
        )
        source = pair * pair_width
        output[:, dst] = 0.5 * query[:, source : source + folded_pair_width]
        output[:, dst] += 0.5j * query[
            :, source + folded_pair_width : source + 2 * folded_pair_width
        ]
    key_base = int(static.query_fold_width)
    for pair in range(kv_pairs):
        dst = slice(
            key_base + pair * folded_pair_width,
            key_base + (pair + 1) * folded_pair_width,
        )
        source = pair * pair_width
        output[:, dst] = key[:, source : source + folded_pair_width]
        output[:, dst] -= 1j * key[
            :, source + folded_pair_width : source + 2 * folded_pair_width
        ]
    value_base = key_base + int(static.key_fold_width)
    for carrier in range(kv_pairs // 2):
        dst = slice(
            value_base + carrier * pair_width,
            value_base + (carrier + 1) * pair_width,
        )
        real_start = 2 * carrier * pair_width
        imag_start = (2 * carrier + 1) * pair_width
        output[:, dst] = 0.5 * value[:, real_start : real_start + pair_width]
        output[:, dst] += 0.5j * value[
            :, imag_start : imag_start + pair_width
        ]
    return output


@dataclass
class QKVTokenShardResult:
    carrier_ciphers: tuple[object, ...]
    spec: QKVTokenShardSpec
    output_layout: LinearCarrierLayout
    carrier_width: int
    complexity: ComplexTokenDiagonalQKVComplexity
    output_levels: tuple[int, ...]
    wall_seconds: float
    stage_seconds: dict[str, float]
    max_abs_diff: float | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        release_all(self.carrier_ciphers)
        self.carrier_ciphers = ()
        self._released = True


def qkv_token_shard_fhe(
    hidden_states: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    *,
    spec: QKVTokenShardSpec,
    attention_layout: AttentionPairLayout,
    layout: FeatureMajorPrefillLayout,
    config: Llama3CKKSConfig | None = None,
    operator_config: LinearOperatorConfig | None = None,
    extra_rotations: tuple[int, ...] | list[int] = (),
    crypto_context=None,
    input_ciphers: tuple[object, ...] | list[object] | None = None,
    verify: bool = False,
    function_prefix: str = "model.layers.0.self_attn.qkv_token_shard",
    input_column_scale: np.ndarray | None = None,
) -> QKVTokenShardResult:
    """Project one worker's adjacent FM group (one or more pairs) through the prepared QKV kernel.

    The kernel is batched: the ``n`` real FM inputs are paired into ``n/2``
    complex ciphertexts and projected together, so the plaintext weight is
    encoded once per call however many pairs the worker owns.
    """

    if crypto_context is None or input_ciphers is None:
        raise ValueError("qkv_token_shard_fhe requires a context and encrypted inputs")
    inputs = tuple(input_ciphers)
    count = len(spec.input_indices)
    if count <= 0 or count % 2:
        raise ValueError(f"QKV token shard needs an even number of FM inputs, spec owns {count}")
    if len(inputs) != count:
        raise ValueError(f"QKV token shard expects {count} FM inputs, got {len(inputs)}")
    if tuple(spec.output_indices) != tuple(spec.input_indices):
        raise ValueError("QKV token shard output indices must equal its input indices")
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    expected = (int(layout.seq_len), int(layout.hidden_dim))
    if hidden_states.shape != expected:
        raise ValueError(
            f"hidden_states must have global shape {expected}, got {hidden_states.shape}"
        )
    expected_indices = tuple(range(count * int(spec.shard_index), count * int(spec.shard_index) + count))
    if tuple(spec.input_indices) != expected_indices:
        raise ValueError(
            f"shard {spec.shard_index} must own FM indices {expected_indices}"
        )
    local_hidden = np.ascontiguousarray(
        hidden_states[int(spec.token_start) : int(spec.token_stop)]
    )
    if local_hidden.shape[0] != int(spec.token_count):
        raise ValueError("token shard range lies outside hidden_states")
    config = Llama3CKKSConfig() if config is None else config
    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.qkv
        if operator_config is None
        else operator_config
    )
    slots = 1 << (int(config.simulator.logN) - 1)
    if int(layout.slots) != slots or int(crypto_context.max_slots) != slots:
        raise ValueError(
            "QKV token shard layout/config/context slot mismatch: "
            f"{layout.slots}/{slots}/{crypto_context.max_slots}"
        )
    local_layout = FeatureMajorPrefillLayout(
        seq_len=int(spec.token_count),
        hidden_dim=int(layout.hidden_dim),
        slots=int(layout.slots),
    )
    if int(local_layout.cipher_count) != count:
        raise ValueError(f"QKV token shard must describe exactly {count} FM ciphertexts")

    wall_start = time.perf_counter()
    paired_inputs: tuple[object, ...] = ()
    projection = None
    stage_seconds: dict[str, float] = {}
    success = False
    try:
        started = time.perf_counter()
        paired_inputs = pair_feature_major_ciphers(
            inputs, layout=local_layout, crypto_context=crypto_context
        )
        sync_device(crypto_context.device)
        stage_seconds["input_pair"] = float(time.perf_counter() - started)
        if len(paired_inputs) != count // 2:
            raise RuntimeError(
                f"{count} FM inputs must form {count // 2} complex inputs, got {len(paired_inputs)}"
            )

        started = time.perf_counter()
        carrier_weight = None
        if structured_weight_packing_enabled():
            _, projection_weight, carrier_width = _structured_projection_weight(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                attention_layout=attention_layout,
                input_column_scale=input_column_scale,
            )
        else:
            _, carrier_weight, projection_weight = _projection_weights(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                attention_layout=attention_layout,
                input_column_scale=input_column_scale,
            )
            carrier_width = int(carrier_weight.shape[0])
        weight_seconds = float(time.perf_counter() - started)
        reference = None
        if verify:
            reference = qkv_carrier_reference(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                attention_layout=attention_layout,
                input_column_scale=input_column_scale,
            )[int(spec.token_start) : int(spec.token_stop)]

        projection = _qkv_fhe_complex_token_diagonal_prepared(
            local_hidden,
            projection_weight,
            carrier_width=int(carrier_width),
            carrier_weight=carrier_weight,
            carrier_reference=reference,
            config=config,
            operator_config=operator_config,
            extra_rotations=extra_rotations,
            crypto_context=crypto_context,
            input_ciphers=paired_inputs,
            verify=bool(verify),
            wall_start=wall_start,
            weight_transform_seconds=weight_seconds,
        )
        if len(projection.carrier_ciphers) != len(spec.output_indices):
            raise RuntimeError(
                "local QKV projection returned the wrong carrier count: "
                f"{len(projection.carrier_ciphers)} != {len(spec.output_indices)}"
            )
        for name, seconds in projection.stage_seconds.items():
            stage_seconds[f"projection.{name}"] = float(seconds)
        outputs = tuple(projection.carrier_ciphers)
        projection.carrier_ciphers = ()
        result = QKVTokenShardResult(
            carrier_ciphers=outputs,
            spec=spec,
            output_layout=projection.output_layout,
            carrier_width=int(projection.carrier_width),
            complexity=projection.complexity,
            output_levels=tuple(projection.output_levels),
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds=stage_seconds,
            max_abs_diff=projection.max_abs_diff,
        )
        success = True
        return result
    finally:
        release_all(paired_inputs)
        if projection is not None:
            projection.release()
        if not success:
            # Outputs remain attached to ``projection`` until the successful
            # ownership transfer above, so its release handles this path.
            pass


__all__ = [
    "QKVTokenShardPlan",
    "QKVTokenShardResult",
    "QKVTokenShardSpec",
    "qkv_carrier_reference",
    "qkv_token_shard_fhe",
    "qkv_token_shard_plan",
    "qkv_token_shard_resource_report",
]
