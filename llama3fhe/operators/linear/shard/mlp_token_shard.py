from __future__ import annotations

"""Resident token-pair sharding for the feature-major Layer0 MLP.

The production feature-major boundary contains sixteen ciphertexts. Each
ciphertext owns eight complete tokens, so adjacent ciphertexts can be paired
as x + i*y and evaluated independently by mlp_fhe. Worker r consumes global
ciphertext indices (2*r, 2*r+1) and produces the same two global indices.
Gate/up, SiLU, the four-page down accumulation, and the final split are all
local to that token pair; no cross-rank partial sum is created and no
collective is mathematically required.

This is deliberately separate from mlp_shard. That module shards the
intermediate feature width and must reduce down-projection partials. It also
uses a sequential bring-up scheduler. Token-pair sharding instead keeps the
full real Layer0 gate/up/down sources on every worker while reducing the
encrypted batch and its transient workspace to one adjacent pair.
"""

from dataclasses import dataclass

import numpy as np

from llama3fhe.backend import release_all
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

from .. import LLAMA3_LINEAR_SCHEDULES
from ..config import LinearOperatorConfig
from ..prefill.mlp import FeatureMajorMLPComplexity, mlp_complexity


@dataclass(frozen=True)
class MLPTokenShardSpec:
    """One rank's globally indexed adjacent feature-major group (one or more pairs)."""

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
class MLPTokenShardPlan:
    """Static one-adjacent-pair-per-logical-worker assignment."""

    input_cipher_count: int
    output_cipher_count: int
    tokens_per_cipher: int
    assignments: tuple[MLPTokenShardSpec, ...]

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(int(spec.rank) for spec in self.assignments)

    def for_rank(self, rank: int) -> MLPTokenShardSpec:
        matches = [
            spec for spec in self.assignments if int(spec.rank) == int(rank)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"rank {rank} is not assigned exactly one MLP token shard"
            )
        return matches[0]


def mlp_token_shard_plan(
    layout: FeatureMajorPrefillLayout,
    ranks: tuple[int, ...] | list[int],
) -> MLPTokenShardPlan:
    """Assign every adjacent FM group of ``cipher_count / len(ranks)`` ciphertexts to one worker.

    The group size must be even (whole adjacent pairs): 8 ranks own one pair
    each of the 16 production ciphertexts, 4 ranks own two.
    """

    ranks = tuple(int(rank) for rank in ranks)
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be a non-empty unique sequence")
    cipher_count = int(layout.cipher_count)
    if cipher_count % 2:
        raise ValueError(
            "MLP token sharding requires an even feature-major cipher count, "
            f"got {cipher_count}"
        )
    pair_count = cipher_count // 2
    if pair_count % len(ranks):
        raise ValueError(
            "MLP token sharding assigns whole adjacent FM pairs per rank: "
            f"need a divisor of {pair_count} ranks for {cipher_count} ciphertexts, "
            f"got {len(ranks)}"
        )
    per_rank = cipher_count // len(ranks)

    tokens_per_cipher = int(layout.tokens_per_cipher)
    assignments: list[MLPTokenShardSpec] = []
    for shard_index, rank in enumerate(ranks):
        first = per_rank * shard_index
        token_start = first * tokens_per_cipher
        token_stop = min(
            int(layout.seq_len), (first + per_rank) * tokens_per_cipher
        )
        token_count = token_stop - token_start
        if token_count != per_rank * tokens_per_cipher:
            raise ValueError(
                f"MLP token sharding requires {per_rank} full FM token groups per "
                f"worker; shard {shard_index} contains {token_count} tokens"
            )
        indices = tuple(range(first, first + per_rank))
        assignments.append(
            MLPTokenShardSpec(
                rank=int(rank),
                shard_index=int(shard_index),
                input_indices=indices,
                output_indices=indices,
                token_start=int(token_start),
                token_count=int(token_count),
            )
        )
    return MLPTokenShardPlan(
        input_cipher_count=cipher_count,
        output_cipher_count=cipher_count,
        tokens_per_cipher=tokens_per_cipher,
        assignments=tuple(assignments),
    )


def mlp_token_shard_resource_report(
    layout: FeatureMajorPrefillLayout,
    ranks: tuple[int, ...] | list[int],
    *,
    intermediate_width: int = 14336,
    silu_polynomial_degree: int = 48,
    silu_multiplicative_depth: int = 6,
    operator_config: LinearOperatorConfig | None = None,
    weight_dtype: np.dtype = np.dtype(np.float32),
) -> dict[str, object]:
    """Describe replicated sources and the exact one-eighth FHE work split."""

    plan = mlp_token_shard_plan(layout, ranks)
    operator = (
        LLAMA3_LINEAR_SCHEDULES.gate_up
        if operator_config is None
        else operator_config
    )
    local_layout = FeatureMajorPrefillLayout(
        seq_len=int(plan.assignments[0].token_count),
        hidden_dim=int(layout.hidden_dim),
        slots=int(layout.slots),
    )
    complexity_kwargs = {
        "intermediate_width": int(intermediate_width),
        "silu_polynomial_degree": int(silu_polynomial_degree),
        "silu_multiplicative_depth": int(silu_multiplicative_depth),
        "baby_steps": int(operator.baby_steps),
        "max_plaintext_rows": int(operator.max_plaintext_rows),
        "reuse_baby_rotations_across_outputs": True,
        "fuse_silu_domain_map": True,
    }
    full = mlp_complexity(layout, **complexity_kwargs)
    local = mlp_complexity(local_layout, **complexity_kwargs)
    hidden_dim = int(layout.hidden_dim)
    width = int(intermediate_width)
    source_elements = 3 * hidden_dim * width
    source_bytes = source_elements * np.dtype(weight_dtype).itemsize

    def fraction(numerator: int, denominator: int) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    return {
        "axis": "token_pair",
        "source_feature_partial_sum": False,
        "homomorphic_output_reduction": False,
        "collective_required": False,
        "worker_count": len(plan.assignments),
        "assignments": {
            str(spec.rank): {
                "input_indices": list(spec.input_indices),
                "output_indices": list(spec.output_indices),
                "token_range": [int(spec.token_start), int(spec.token_stop)],
            }
            for spec in plan.assignments
        },
        "full": full.to_dict(),
        "per_worker": {
            **local.to_dict(),
            "local_seq_len": int(local_layout.seq_len),
            "real_input_cipher_count": 2,
            "real_output_cipher_count": 2,
            "full_weight_source_bytes": int(source_bytes),
            "weight_source_fraction": 1.0,
            "rotation_key_union_replicated": True,
            "gate_up_pt_ct_work_fraction": fraction(
                local.gate_up_pt_ct_multiplications,
                full.gate_up_pt_ct_multiplications,
            ),
            "down_pt_ct_work_fraction": fraction(
                local.down_pt_ct_multiplications,
                full.down_pt_ct_multiplications,
            ),
            "silu_work_fraction": fraction(
                local.silu_polynomial_evaluations,
                full.silu_polynomial_evaluations,
            ),
        },
    }


@dataclass
class MLPTokenShardResult:
    """Two local FM outputs with their global ownership metadata."""

    ciphers: tuple[object, ...]
    spec: MLPTokenShardSpec
    local_layout: FeatureMajorPrefillLayout
    complexity: FeatureMajorMLPComplexity
    output_levels: tuple[int, ...]
    wall_seconds: float
    stage_seconds: dict[str, float]
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        release_all(self.ciphers)
        self.ciphers = ()
        self._released = True


def mlp_token_shard_fhe(
    hidden_states: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    silu_entry: dict,
    spec: MLPTokenShardSpec,
    layout: FeatureMajorPrefillLayout,
    config=None,
    operator_config: LinearOperatorConfig | None = None,
    crypto_context=None,
    input_ciphers: tuple[object, ...] | list[object] | None = None,
    input_column_scale: np.ndarray | None = None,
    fuse_silu_domain_map: bool = False,
    verify: bool = False,
) -> MLPTokenShardResult:
    """Run the unchanged MLP kernel on one rank's resident adjacent FM group."""

    if crypto_context is None or input_ciphers is None:
        raise ValueError("mlp_token_shard_fhe requires a context and encrypted inputs")
    inputs = tuple(input_ciphers)
    count = len(spec.input_indices)
    if count <= 0 or count % 2:
        raise ValueError(f"MLP token shard needs an even number of FM inputs, spec owns {count}")
    if len(inputs) != count:
        raise ValueError(f"MLP token shard expects {count} FM inputs, got {len(inputs)}")
    expected_indices = tuple(
        range(count * int(spec.shard_index), count * int(spec.shard_index) + count)
    )
    if tuple(spec.input_indices) != expected_indices:
        raise ValueError(
            f"shard {spec.shard_index} must own FM indices {expected_indices}"
        )
    if tuple(spec.output_indices) != tuple(spec.input_indices):
        raise ValueError("MLP token shard outputs must retain their input indices")

    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    expected_shape = (int(layout.seq_len), int(layout.hidden_dim))
    if hidden_states.shape != expected_shape:
        raise ValueError(
            f"hidden_states must have global shape {expected_shape}, "
            f"got {hidden_states.shape}"
        )
    local_hidden = np.ascontiguousarray(
        hidden_states[int(spec.token_start) : int(spec.token_stop)]
    )
    if local_hidden.shape != (
        int(spec.token_count),
        int(layout.hidden_dim),
    ):
        raise ValueError("MLP token shard range lies outside hidden_states")
    local_layout = FeatureMajorPrefillLayout(
        seq_len=int(spec.token_count),
        hidden_dim=int(layout.hidden_dim),
        slots=int(layout.slots),
    )
    if int(local_layout.cipher_count) != count:
        raise ValueError(f"MLP token shard must describe exactly {count} FM ciphertexts")

    from ..prefill.mlp import mlp_fhe

    result = mlp_fhe(
        local_hidden,
        gate_weight,
        up_weight,
        down_weight,
        silu_entry=silu_entry,
        layout=local_layout,
        config=config,
        operator_config=operator_config,
        crypto_context=crypto_context,
        input_ciphers=inputs,
        verify=bool(verify),
        input_column_scale=input_column_scale,
        fuse_silu_domain_map=bool(fuse_silu_domain_map),
    )
    try:
        outputs = tuple(result.ciphers)
        if len(outputs) != len(spec.output_indices):
            raise RuntimeError(
                "local MLP returned the wrong output count: "
                f"{len(outputs)} != {len(spec.output_indices)}"
            )
        result.ciphers = ()
        return MLPTokenShardResult(
            ciphers=outputs,
            spec=spec,
            local_layout=local_layout,
            complexity=result.complexity,
            output_levels=tuple(int(value) for value in result.output_levels),
            wall_seconds=float(result.wall_seconds),
            stage_seconds={
                str(name): float(seconds)
                for name, seconds in result.stage_seconds.items()
            },
        )
    finally:
        result.release()


__all__ = [
    "MLPTokenShardPlan",
    "MLPTokenShardResult",
    "MLPTokenShardSpec",
    "mlp_token_shard_fhe",
    "mlp_token_shard_plan",
    "mlp_token_shard_resource_report",
]
