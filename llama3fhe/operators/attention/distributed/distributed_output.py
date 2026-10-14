from __future__ import annotations

"""Distributed Prefill ``O_C -> PairFM`` ownership transpose.

The production attention core leaves two adjacent global query-pair
ciphertexts on each of eight head-shard ranks.  The first two PairFM slot
permutations are source-local.  The final ownership transpose is not: every
token-pair carrier contains a feature block from every source rank.

This module implements that boundary as a reduce-scatter.  Source shard
(*slot*) ``s`` -- the two adjacent query pairs ``2s, 2s+1`` -- creates one
pre-rescale complex partial for every target token-pair carrier.  A partial's
real/imaginary components are the two target token shards, while its two
source feature blocks occupy their final global positions.  Partials are then
copied to their target owners (cross-process transport lives in
``llama3fhe.parallel.prefill.transport``), summed and rescaled once by
:func:`pairfm_target_reduce_fhe`.  No rank ever gathers all sixteen ``O_C``
inputs or all eight final carriers.

The algebra is fixed at eight source slots x eight target carriers.  A rank
may own several slots (``world_size == 4``: two slots, four ``O_C`` per rank);
:func:`pairfm_rank_partials_fhe` runs the source half once per owned slot and
pre-adds the same-target partials on the device (plain modular addition at the
shared pre-rescale level), so the wire still carries one packet per
``(source rank, target carrier)`` and the target still rescales once.

The sharded algebra deliberately differs from the legacy single-card kernel's
"sixteen real owner outputs, then pair" epilogue.  Strong-scaling results
must therefore use this module with all logical shards mapped to one device as
their single-card denominator.
"""

import os
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_all, release_if_supported
from llama3fhe.backend import profile_sync_device as sync_device  # stage-timer sync, no-op under T2_STAGE_SYNC=0
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

from ...linear import pair_real_ciphers, split_complex_ciphers_twice
from ...primitives import accumulate_owned, drop_to_limbs, encode_mask
from ...transform import apply_transform_by_level, transform_plaintext_scope
from ..config import AttentionOperatorConfig
from ..prefill.output import (
    _blocked_masks,
    _geometry,
    _pairfm_masks,
    output_c_to_pairfm_numpy,
    pairfm_output_complexity,
)


PARTIAL_LAYOUT = "attention.pairfm.target_pair_partial"
CARRIER_LAYOUT = "attention.pairfm.w_o"


@dataclass(frozen=True)
class PairFMShardPlacement:
    """One logical source shard and its colocated target-carrier owner."""

    logical_rank: int
    device_rank: int
    query_pair_indices: tuple[int, int]
    target_carrier_index: int
    target_token_shards: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_rank": int(self.logical_rank),
            "device_rank": int(self.device_rank),
            "query_pair_indices": [int(v) for v in self.query_pair_indices],
            "target_carrier_index": int(self.target_carrier_index),
            "target_token_shards": [
                int(v) for v in self.target_token_shards
            ],
        }


@dataclass(frozen=True)
class PairFMReduceScatterPlan:
    query_pair_count: int
    source_shards: int
    target_carriers: int
    placements: tuple[PairFMShardPlacement, ...]
    transport_edges: tuple[dict[str, object], ...]
    remote_transfer_count: int
    local_contribution_count: int

    @property
    def devices(self) -> tuple[int, ...]:
        return tuple(int(row.device_rank) for row in self.placements)

    @property
    def unique_devices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.devices)))

    def to_dict(self) -> dict[str, object]:
        return {
            "query_pair_count": int(self.query_pair_count),
            "source_shards": int(self.source_shards),
            "target_carriers": int(self.target_carriers),
            "placements": [row.to_dict() for row in self.placements],
            "transport_edges": [dict(row) for row in self.transport_edges],
            "remote_transfer_count": int(self.remote_transfer_count),
            "local_contribution_count": int(self.local_contribution_count),
            "unique_devices": [int(v) for v in self.unique_devices],
        }


@dataclass(frozen=True)
class DistributedPairFMComplexity:
    query_pair_count: int
    source_shards: int
    target_carriers: int
    input_ciphertexts_per_source: int
    partial_carriers_per_source: int
    partial_carriers_total: int
    remote_partial_transfers: int
    pairfm_pt_ct: int
    blocked_pt_ct: int
    owner_pt_ct: int
    conversion_pt_ct: int
    pairfm_rotations: int
    blocked_rotations: int
    owner_rotations: int
    conversion_rotations: int
    source_split_conjugations: int
    input_pair_imults: int
    source_split_imults: int
    target_pair_imults: int
    conversion_imults: int
    target_reduce_additions: int
    final_rescales: int
    transform_encoded_rows_per_source: int
    owner_encoded_rows_per_source: int
    distributed_encoded_rows_total: int
    multiplicative_depth: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class PairFMSourcePartialResult:
    """Eight pre-rescale target-pair partials made by one source shard."""

    partial_ciphers: tuple[object, ...]
    source_logical_rank: int
    global_query_pairs: tuple[int, int]
    wall_seconds: float
    stage_seconds: dict[str, float]

    def release(self) -> None:
        release_all(self.partial_ciphers)
        self.partial_ciphers = ()


@dataclass
class PairFMRankPartialResult:
    """Eight pre-rescale target-pair partials made by one *rank* (>= 1 slots).

    ``partial_ciphers[target]`` is the device-local sum over this rank's
    slots of the slot partials for target carrier ``target``; when the rank
    owns exactly one slot it is that slot's partial unchanged.
    """

    partial_ciphers: tuple[object, ...]
    slot_indices: tuple[int, ...]
    global_query_pairs: tuple[int, ...]
    wall_seconds: float
    stage_seconds: dict[str, float]
    slot_wall_seconds: tuple[float, ...]

    def release(self) -> None:
        release_all(self.partial_ciphers)
        self.partial_ciphers = ()


def build_pairfm_reduce_scatter_plan(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    devices: Iterable[int],
) -> PairFMReduceScatterPlan:
    """Build the logical source/target ownership graph.

    ``devices`` maps logical rank to physical CUDA rank.  Eight distinct
    entries describe the production numerator; eight copies of one rank
    describe the matched single-card denominator.
    """

    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    if pairs % 2:
        raise ValueError("distributed PairFM requires an even query-pair count")
    source_shards = pairs // 2
    resolved = tuple(int(value) for value in devices)
    if len(resolved) != source_shards:
        raise ValueError(
            f"PairFM requires {source_shards} logical device placements, "
            f"got {len(resolved)}"
        )
    if any(value < 0 for value in resolved):
        raise ValueError("CUDA device ranks must be non-negative")

    placements = tuple(
        PairFMShardPlacement(
            logical_rank=logical,
            device_rank=resolved[logical],
            query_pair_indices=(2 * logical, 2 * logical + 1),
            target_carrier_index=logical,
            target_token_shards=(2 * logical, 2 * logical + 1),
        )
        for logical in range(source_shards)
    )
    edges: list[dict[str, object]] = []
    remote = 0
    local = 0
    for source in placements:
        for target in placements:
            is_remote = int(source.device_rank) != int(target.device_rank)
            remote += int(is_remote)
            local += int(not is_remote)
            edges.append(
                {
                    "source_logical_rank": int(source.logical_rank),
                    "source_device_rank": int(source.device_rank),
                    "target_carrier_index": int(target.target_carrier_index),
                    "target_device_rank": int(target.device_rank),
                    "remote": bool(is_remote),
                }
            )
    return PairFMReduceScatterPlan(
        query_pair_count=pairs,
        source_shards=source_shards,
        target_carriers=source_shards,
        placements=placements,
        transport_edges=tuple(edges),
        remote_transfer_count=remote,
        local_contribution_count=local,
    )


def distributed_pairfm_complexity(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    devices: Iterable[int],
    first_baby_steps: int = 16,
    block_baby_steps: int = 32,
) -> DistributedPairFMComplexity:
    plan = build_pairfm_reduce_scatter_plan(
        feature_layout, attention_layout, devices
    )
    base = pairfm_output_complexity(
        feature_layout,
        attention_layout,
        first_baby_steps=int(first_baby_steps),
        block_baby_steps=int(block_baby_steps),
    )
    sources = int(plan.source_shards)
    targets = int(plan.target_carriers)
    transform_rows = int(base.pairfm_padded_rows + base.blocked_padded_rows)
    owner_rows_per_source = 2
    return DistributedPairFMComplexity(
        query_pair_count=int(plan.query_pair_count),
        source_shards=sources,
        target_carriers=targets,
        input_ciphertexts_per_source=2,
        partial_carriers_per_source=targets,
        partial_carriers_total=sources * targets,
        remote_partial_transfers=int(plan.remote_transfer_count),
        pairfm_pt_ct=int(base.pairfm_pt_ct),
        blocked_pt_ct=int(base.blocked_pt_ct),
        owner_pt_ct=int(base.owner_pt_ct),
        conversion_pt_ct=int(base.conversion_pt_ct),
        pairfm_rotations=int(base.pairfm_rotations),
        blocked_rotations=int(base.blocked_rotations),
        owner_rotations=int(base.owner_rotations),
        conversion_rotations=int(base.conversion_rotations),
        source_split_conjugations=sources,
        input_pair_imults=sources,
        source_split_imults=sources,
        target_pair_imults=sources * targets,
        conversion_imults=sources * (2 + targets),
        target_reduce_additions=targets * (sources - 1),
        final_rescales=targets,
        transform_encoded_rows_per_source=transform_rows,
        owner_encoded_rows_per_source=owner_rows_per_source,
        distributed_encoded_rows_total=sources
        * (transform_rows + owner_rows_per_source),
        multiplicative_depth=3,
    )


def pairfm_source_partial_carriers_numpy(
    output_c: np.ndarray,
    *,
    source_logical_rank: int,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> np.ndarray:
    """Return one source rank's eight complex target-pair partials."""

    pairs, lanes, pair_feature, _ = _geometry(
        feature_layout, attention_layout
    )
    source_count = pairs // 2
    source_rank = int(source_logical_rank)
    if not 0 <= source_rank < source_count:
        raise IndexError(
            f"source_logical_rank={source_rank} is outside [0, {source_count})"
        )
    values = np.asarray(output_c)
    expected = (pairs, int(attention_layout.slots))
    if values.shape != expected:
        raise ValueError(f"O_C must have shape {expected}, got {values.shape}")

    pairfm = output_c_to_pairfm_numpy(
        values,
        feature_layout=feature_layout,
        attention_layout=attention_layout,
    ).reshape(pairs, pair_feature, pairs, lanes)
    blocked = pairfm.transpose(0, 2, 1, 3)
    result = np.zeros(
        (source_count, int(feature_layout.slots)), dtype=np.complex128
    )
    block = pair_feature * lanes
    for local_source in (0, 1):
        global_source = 2 * source_rank + local_source
        destination = slice(global_source * block, (global_source + 1) * block)
        for target_pair in range(source_count):
            even = blocked[global_source, 2 * target_pair].reshape(-1)
            odd = blocked[global_source, 2 * target_pair + 1].reshape(-1)
            result[target_pair, destination] = even + 1j * odd
    return result


def distributed_pairfm_carriers_numpy(
    output_c: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> np.ndarray:
    """CPU oracle for the complete source-to-target reduce-scatter."""

    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    return sum(
        (
            pairfm_source_partial_carriers_numpy(
                output_c,
                source_logical_rank=source,
                feature_layout=feature_layout,
                attention_layout=attention_layout,
            )
            for source in range(pairs // 2)
        ),
        start=np.zeros(
            (pairs // 2, int(feature_layout.slots)), dtype=np.complex128
        ),
    )


def _source_owner_partials_fhe(
    logical_sources: tuple[object, object],
    *,
    source_logical_rank: int,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    dtype: np.dtype,
    name: str,
    profile: dict[str, float],
) -> tuple[object, ...]:
    """Create eight complex, pre-rescale target-pair partials."""

    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    targets = pairs // 2
    source_rank = int(source_logical_rank)
    if len(logical_sources) != 2:
        raise ValueError("a PairFM source rank must provide two split streams")
    levels = {
        int(crypto_context.level_for_cipher(cipher))
        for cipher in logical_sources
    }
    if len(levels) != 1:
        raise ValueError("PairFM source streams must share one level")
    level = levels.pop()
    block = int(feature_layout.slots) // pairs
    even_accumulators: list[object | None] = [None] * targets
    odd_accumulators: list[object | None] = [None] * targets
    outputs: list[object] = []
    try:
        for local_index, cipher in enumerate(logical_sources):
            global_source = 2 * source_rank + local_index
            started = time.perf_counter()
            mask = np.zeros((int(feature_layout.slots),), dtype=dtype)
            # split_complex_ciphers_twice returns twice the real value.
            mask[global_source * block : (global_source + 1) * block] = 0.5
            plain = None
            try:
                plain = encode_mask(
                    mask,
                    name=f"{name}.source{global_source}",
                    level=level,
                    crypto_context=crypto_context,
                )
                sync_device(crypto_context.device)
                profile["pack_encode"] += time.perf_counter() - started
                started = time.perf_counter()
                for target_pair in range(targets):
                    for parity, accumulators in (
                        (0, even_accumulators),
                        (1, odd_accumulators),
                    ):
                        target_shard = 2 * target_pair + parity
                        offset = (target_shard - global_source) * block
                        rotated = (
                            cipher
                            if offset == 0
                            else crypto_context.fhe.homo_rotate(
                                cipher, offset, crypto_context.context
                            )
                        )
                        try:
                            product = crypto_context.fhe.homo_mul_pt(
                                rotated, plain, crypto_context.context
                            )
                            accumulators[target_pair] = accumulate_owned(
                                accumulators[target_pair],
                                product,
                                crypto_context=crypto_context,
                            )
                        finally:
                            if rotated is not cipher:
                                release_if_supported(rotated)
                sync_device(crypto_context.device)
                profile["online"] += time.perf_counter() - started
            finally:
                release_if_supported(plain)

        started = time.perf_counter()
        for target_pair in range(targets):
            even = even_accumulators[target_pair]
            odd = odd_accumulators[target_pair]
            if even is None or odd is None:
                raise RuntimeError(
                    f"empty PairFM partial accumulator {target_pair}"
                )
            imaginary = None
            combined = None
            try:
                imaginary = fhe.homo_mul_i(
                    odd, crypto_context.context, negative=False
                )
                combined = crypto_context.fhe.homo_add(
                    even, imaginary, crypto_context.context
                )
                outputs.append(combined)
                combined = None
            finally:
                release_if_supported(even)
                release_if_supported(odd)
                release_if_supported(imaginary)
                release_if_supported(combined)
                even_accumulators[target_pair] = None
                odd_accumulators[target_pair] = None
        sync_device(crypto_context.device)
        profile["target_pairing"] += time.perf_counter() - started
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in even_accumulators + odd_accumulators + outputs:
            release_if_supported(cipher)


def pairfm_source_partials_fhe(
    output_ciphers: Iterable[object],
    *,
    source_logical_rank: int,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    bootstrap_operator=None,
    function_prefix: str = "model.layers.0.self_attn.distributed_pairfm",
) -> PairFMSourcePartialResult:
    """Run the source-local half of distributed PairFM on two ``O_C`` CTs."""

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    output_ciphers = tuple(output_ciphers)
    if len(output_ciphers) != 2:
        raise ValueError("each distributed PairFM source requires exactly two O_C CTs")
    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    source_rank = int(source_logical_rank)
    if not 0 <= source_rank < pairs // 2:
        raise IndexError(
            f"source_logical_rank={source_rank} is outside [0, {pairs // 2})"
        )
    input_levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in output_ciphers
    }
    if len(input_levels) != 1:
        raise ValueError("the two local O_C ciphertexts must share one level")
    model_config = Llama3CKKSConfig() if model_config is None else model_config
    if (1 << (int(model_config.simulator.logN) - 1)) != int(
        feature_layout.slots
    ):
        raise ValueError("model logN and feature layout slots do not match")
    if int(crypto_context.max_slots) != int(feature_layout.slots):
        raise ValueError("crypto context and feature layout slots do not match")

    dtype = np.dtype(model_config.layout_transform.dtype)
    first_baby_steps = int(operator_config.pairfm_first_baby_steps)
    block_baby_steps = int(operator_config.pairfm_block_baby_steps)
    pairfm_masks, pairfm_stride = _pairfm_masks(
        feature_layout, attention_layout, dtype=dtype
    )
    blocked_masks, blocked_stride = _blocked_masks(
        feature_layout, attention_layout, dtype=dtype
    )

    paired: tuple[object, ...] = ()
    pairfm: tuple[object, ...] = ()
    blocked: tuple[object, ...] = ()
    logical_sources: tuple[object, ...] = ()
    partials: tuple[object, ...] = ()
    success = False
    wall_started = time.perf_counter()
    stages = {
        "input_pairing": 0.0,
        "input_pair_bootstrap": 0.0,
        "input_pair_post_refresh_modulus_drop": 0.0,
        "pairfm_pack_encode": 0.0,
        "pairfm_online_rescale": 0.0,
        "block_pack_encode": 0.0,
        "block_online_rescale": 0.0,
        "source_split": 0.0,
        "partial_pack_encode": 0.0,
        "partial_online": 0.0,
        "partial_target_pairing": 0.0,
    }
    try:
        started = time.perf_counter()
        paired = pair_real_ciphers(
            output_ciphers, crypto_context=crypto_context, pad_odd=False
        )
        sync_device(crypto_context.device)
        stages["input_pairing"] = time.perf_counter() - started

        should_refresh = bool(operator_config.refresh_output_before_pairfm)
        threshold = operator_config.pairfm_refresh_min_input_limbs
        if threshold is not None:
            minimum = int(threshold)
            if minimum <= 1:
                raise ValueError("pairfm_refresh_min_input_limbs must exceed one")
            should_refresh = should_refresh or min(
                int(cipher.state.cur_limbs) for cipher in paired
            ) < minimum
        if os.environ.get("LLAMA_FHE_PAIRFM_LIMB_TRACE", "") == "1":
            print(
                f"[distributed-pairfm-limb-trace] {function_prefix}: "
                f"source={source_rank} limbs="
                f"{[int(c.state.cur_limbs) for c in paired]} "
                f"threshold={threshold} refresh={should_refresh}",
                flush=True,
            )
        if should_refresh:
            if bootstrap_operator is None:
                raise ValueError(
                    "PairFM input refresh requires a per-rank bootstrap operator"
                )
            started = time.perf_counter()
            refreshed = (
                bootstrap_operator.refresh(
                    paired[0],
                    iterations=int(
                        operator_config.pairfm_refresh_bootstrap_iterations
                    ),
                    precision_bits=int(
                        operator_config.checkpoint_bootstrap_precision_bits
                    ),
                ),
            )
            release_all(paired)
            paired = refreshed
            sync_device(crypto_context.device)
            stages["input_pair_bootstrap"] = time.perf_counter() - started
            target_limbs = operator_config.pairfm_post_refresh_input_limbs
            if target_limbs is not None and int(
                paired[0].state.cur_limbs
            ) != int(target_limbs):
                started = time.perf_counter()
                dropped = drop_to_limbs(
                    paired,
                    target_limbs=int(target_limbs),
                    crypto_context=crypto_context,
                )
                release_all(paired)
                paired = dropped
                sync_device(crypto_context.device)
                stages["input_pair_post_refresh_modulus_drop"] = (
                    time.perf_counter() - started
                )

        profile: dict[str, float] = {}
        pairfm = apply_transform_by_level(
            paired,
            masks=pairfm_masks,
            stride=pairfm_stride,
            baby_steps=first_baby_steps,
            crypto_context=crypto_context,
            name=f"{function_prefix}.source{source_rank}.pairfm",
            profile=profile,
        )
        release_all(paired)
        paired = ()
        stages["pairfm_pack_encode"] = float(profile.get("pack_encode", 0.0))
        stages["pairfm_online_rescale"] = float(
            profile.get("online_rescale", 0.0)
        )

        profile = {}
        blocked = apply_transform_by_level(
            pairfm,
            masks=blocked_masks,
            stride=blocked_stride,
            baby_steps=block_baby_steps,
            crypto_context=crypto_context,
            name=f"{function_prefix}.source{source_rank}.blocked",
            profile=profile,
        )
        release_all(pairfm)
        pairfm = ()
        stages["block_pack_encode"] = float(profile.get("pack_encode", 0.0))
        stages["block_online_rescale"] = float(
            profile.get("online_rescale", 0.0)
        )

        started = time.perf_counter()
        logical_sources = split_complex_ciphers_twice(
            blocked, crypto_context=crypto_context
        )
        sync_device(crypto_context.device)
        stages["source_split"] = time.perf_counter() - started
        release_all(blocked)
        blocked = ()

        profile = {"pack_encode": 0.0, "online": 0.0, "target_pairing": 0.0}
        partials = _source_owner_partials_fhe(
            (logical_sources[0], logical_sources[1]),
            source_logical_rank=source_rank,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            dtype=dtype,
            name=f"{function_prefix}.owner_partial",
            profile=profile,
        )
        release_all(logical_sources)
        logical_sources = ()
        stages["partial_pack_encode"] = float(profile["pack_encode"])
        stages["partial_online"] = float(profile["online"])
        stages["partial_target_pairing"] = float(profile["target_pairing"])

        success = True
        return PairFMSourcePartialResult(
            partial_ciphers=partials,
            source_logical_rank=source_rank,
            global_query_pairs=(2 * source_rank, 2 * source_rank + 1),
            wall_seconds=float(time.perf_counter() - wall_started),
            stage_seconds={str(k): float(v) for k, v in stages.items()},
        )
    finally:
        release_all(paired + pairfm + blocked + logical_sources)
        if not success:
            release_all(partials)


def pairfm_rank_partials_fhe(
    output_ciphers_by_slot,
    *,
    slot_indices: Iterable[int],
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    bootstrap_operator=None,
    function_prefix: str = "model.layers.0.self_attn.distributed_pairfm",
) -> PairFMRankPartialResult:
    """Source half of distributed PairFM for every slot owned by one rank.

    ``output_ciphers_by_slot[i]`` are the two ``O_C`` ciphertexts of slot
    ``slot_indices[i]`` (query pairs ``2s, 2s+1``).  Each slot runs
    :func:`pairfm_source_partials_fhe` unchanged; the eight partials of
    successive slots are then folded together per target carrier with
    ``homo_add`` (all partials leave the source half at one pre-rescale
    level, so the sum is exact and level-free) and the slot partials are
    released.  With one slot this is exactly ``pairfm_source_partials_fhe``.

    Inputs are not consumed.  The caller owns the returned partials.
    """

    slots = tuple(int(v) for v in slot_indices)
    groups = tuple(tuple(group) for group in output_ciphers_by_slot)
    if not slots:
        raise ValueError("pairfm_rank_partials_fhe needs at least one slot")
    if len(groups) != len(slots):
        raise ValueError(
            f"{len(groups)} O_C groups for {len(slots)} slots {slots}"
        )
    if len(set(slots)) != len(slots):
        raise ValueError(f"duplicate slot indices {slots}")
    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    target_count = pairs // 2

    stages: dict[str, float] = {}
    slot_walls: list[float] = []
    accumulated: list[object | None] = [None] * target_count
    wall_started = time.perf_counter()
    # every slot applies the same read-only pairfm / blocked masks at the same
    # level: encode them once per station instead of once per slot
    scope = transform_plaintext_scope()
    memo = scope.__enter__()
    try:
        for slot, ciphers in zip(slots, groups, strict=True):
            source = pairfm_source_partials_fhe(
                ciphers,
                source_logical_rank=slot,
                feature_layout=feature_layout,
                attention_layout=attention_layout,
                crypto_context=crypto_context,
                model_config=model_config,
                operator_config=operator_config,
                bootstrap_operator=bootstrap_operator,
                function_prefix=function_prefix,
            )
            partials = list(source.partial_ciphers)
            source.partial_ciphers = ()
            slot_walls.append(float(source.wall_seconds))
            for name, seconds in source.stage_seconds.items():
                stages[str(name)] = stages.get(str(name), 0.0) + float(seconds)
            if len(partials) != target_count:
                release_all(partials)
                raise RuntimeError(
                    f"slot {slot}: {len(partials)} partials for {target_count} targets"
                )
            started = time.perf_counter()
            try:
                for target in range(target_count):
                    value = partials[target]
                    partials[target] = None
                    if accumulated[target] is not None and int(
                        crypto_context.level_for_cipher(accumulated[target])
                    ) != int(crypto_context.level_for_cipher(value)):
                        release_if_supported(value)
                        raise ValueError(
                            f"slot {slot}: partial level differs from the earlier slots; "
                            "cannot pre-add before the target rescale"
                        )
                    accumulated[target] = accumulate_owned(
                        accumulated[target], value, crypto_context=crypto_context
                    )
            finally:
                # partials not yet folded in (error mid-loop) must not leak
                release_all(value for value in partials if value is not None)
            if len(slots) > 1:
                sync_device(crypto_context.device)
            stages["slot_pre_add"] = stages.get("slot_pre_add", 0.0) + float(
                time.perf_counter() - started
            )
        result = tuple(accumulated)
        accumulated = []
        stages["mask_encode_hits"] = float(memo.hits)
        stages["mask_encode_misses"] = float(memo.misses)
        return PairFMRankPartialResult(
            partial_ciphers=result,
            slot_indices=slots,
            global_query_pairs=tuple(
                pair for slot in slots for pair in (2 * slot, 2 * slot + 1)
            ),
            wall_seconds=float(time.perf_counter() - wall_started),
            stage_seconds={str(k): float(v) for k, v in stages.items()},
            slot_wall_seconds=tuple(slot_walls),
        )
    finally:
        scope.__exit__(None, None, None)
        release_all(value for value in accumulated if value is not None)


def pairfm_target_reduce_fhe(
    partials: Iterable[object],
    *,
    crypto_context,
    synchronize: bool = True,
) -> object:
    """Target-side half of the PairFM reduce-scatter (one rank, one carrier).

    ``partials`` are the pre-rescale complex partials produced for this target
    carrier by every source rank (``pairfm_source_partials_fhe(...)
    .partial_ciphers[target]``), already resident on this device.  They are
    accumulated with a single modular addition chain and rescaled once.  The
    input partials are released; the caller owns the returned carrier.

    The cross-process transport that brings the partials here is the
    responsibility of :mod:`llama3fhe.parallel.prefill.transport`; this function is
    the device-local compute step only.
    """

    raw_values = [value for value in partials]
    if not raw_values:
        raise ValueError("pairfm_target_reduce_fhe requires at least one partial")
    accumulator = None
    output = None
    try:
        for index in range(len(raw_values)):
            value = raw_values[index]
            raw_values[index] = None
            accumulator = accumulate_owned(
                accumulator, value, crypto_context=crypto_context
            )
        output = crypto_context.fhe.rescale(accumulator, crypto_context.context)
        release_if_supported(accumulator)
        accumulator = None
        if bool(synchronize):
            sync_device(crypto_context.device)
        return output
    except Exception:
        release_if_supported(accumulator)
        release_if_supported(output)
        raise
    finally:
        release_all(value for value in raw_values if value is not None)



__all__ = [
    "CARRIER_LAYOUT",
    "PARTIAL_LAYOUT",
    "DistributedPairFMComplexity",
    "PairFMRankPartialResult",
    "PairFMReduceScatterPlan",
    "PairFMShardPlacement",
    "PairFMSourcePartialResult",
    "build_pairfm_reduce_scatter_plan",
    "distributed_pairfm_carriers_numpy",
    "distributed_pairfm_complexity",
    "pairfm_rank_partials_fhe",
    "pairfm_target_reduce_fhe",
    "pairfm_source_partial_carriers_numpy",
    "pairfm_source_partials_fhe",
]
