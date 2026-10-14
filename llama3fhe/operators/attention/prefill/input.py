from __future__ import annotations

import os
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import (
    profile_sync_device as sync_device,  # stage-timer sync, no-op under T2_STAGE_SYNC=0
)
from llama3fhe.backend import release_if_supported
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

from ....approx.attention import apply_rope_heads, rope_cos_sin
from ...linear import split_complex_ciphers_twice
from ...primitives import (
    accumulate_owned as _add_owned,
)
from ...primitives import (
    encode_mask as _encode_mask,
)
from ...transform import apply_transform_by_level, encode_transform_rows
from ..codec import decrypt_complex_rows, repeat_payload_cipher
from ..config import AttentionOperatorConfig
from ..sparse_bsgs_scheduler import (
    SparseMaskRoute,
    SparseRouteBatch,
    SparseRouteGroup,
    build_sparse_route_schedule,
    materialize_sparse_batch_masks,
    sparse_batch_mask_indices,
    sparse_route_arrays,
)
from ..sparse_bsgs_scheduler import (
    canonical_rotation as _canonical_rotation,
)
from ..sparse_fused import SparseFusedAccumulator


@dataclass(frozen=True)
class SparseDiagonalPlan:
    """A sparse BSGS decomposition of arbitrary cyclic matrix diagonals."""

    modulus: int
    diagonal_offsets: tuple[int, ...]
    baby_offsets: tuple[int, ...]
    giant_offsets: tuple[int, ...]

    @property
    def plaintext_rows(self) -> int:
        return len(self.diagonal_offsets)

    @property
    def rotations_per_cipher(self) -> int:
        return max(0, len(self.baby_offsets) - (0 in self.baby_offsets)) + max(
            0, len(self.giant_offsets) - (0 in self.giant_offsets)
        )


@dataclass(frozen=True)
class ComplexTokenAttentionComplexity:
    input_cipher_count: int
    query_output_cipher_count: int
    key_output_cipher_count: int
    value_output_cipher_count: int
    token_lanes: int
    lane_phase_plaintext_rows: int
    lane_phase_pt_ct_multiplications: int
    lane_phase_rotations: int
    block_gather_plaintext_rows: int
    block_gather_pt_ct_multiplications: int
    block_gather_rotations: int
    rectangular_plaintext_rows: int
    rectangular_pt_ct_multiplications: int
    rectangular_rotations: int
    rectangular_bsgs_modulus: int
    feature_expansion_rotations: int
    feature_expansion_conjugations: int
    key_shear_plaintext_rows: int
    key_shear_pt_ct_multiplications: int
    key_shear_rotations: int
    value_delta_plaintext_rows: int
    value_delta_pt_ct_multiplications: int
    value_delta_rotations: int
    value_delta_bsgs_modulus: int
    value_split_conjugations: int
    payload_repeat_rotations: int
    encoded_static_plaintext_rows: int
    pt_ct_multiplications: int
    rotations: int
    conjugations: int
    query_multiplicative_depth: int
    key_multiplicative_depth: int
    value_multiplicative_depth: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class DeltaKVPage:
    """Owned K/V ciphertexts in the Prefill Delta layout."""

    key_ciphers: tuple[object, ...]
    value_ciphers: tuple[object, ...]
    token_valid_mask: np.ndarray | None = None

    def release(self) -> None:
        owned = self.key_ciphers + self.value_ciphers
        self.key_ciphers = ()
        self.value_ciphers = ()
        for cipher in owned:
            release_if_supported(cipher)


@dataclass
class ComplexTokenAttentionResult:
    """Selected QK-ready boundary produced from private linear carriers."""

    query_ciphers: tuple[object, ...]
    key_ciphers: tuple[object, ...]
    value_ciphers: tuple[object, ...]
    wall_seconds: float
    stage_seconds: dict[str, float]
    scheduler_stats: dict[str, float]
    max_abs_diff: dict[str, float] | None = None
    key_c_ciphers: tuple[object, ...] = ()
    value_c_ciphers: tuple[object, ...] = ()
    pairfold_value_ciphers: tuple[object, ...] = ()

    def snapshot_delta_page(
        self, *, token_valid_mask: np.ndarray | None = None
    ) -> DeltaKVPage:
        """Copy the current K/V page before QK/PV release their inputs."""

        copied: list[object] = []
        try:
            for cipher in self.key_ciphers + self.value_ciphers:
                copied.append(cipher.deep_copy())
            mask = (
                None
                if token_valid_mask is None
                else np.asarray(token_valid_mask, dtype=bool).copy()
            )
            page = DeltaKVPage(
                key_ciphers=tuple(copied[: len(self.key_ciphers)]),
                value_ciphers=tuple(copied[len(self.key_ciphers) :]),
                token_valid_mask=mask,
            )
            copied = []
            return page
        finally:
            for cipher in copied:
                release_if_supported(cipher)

    def release_qk_inputs(self) -> None:
        """Release Q/K at their last use while retaining V/cache outputs."""

        owned = self.query_ciphers + self.key_ciphers
        self.query_ciphers = ()
        self.key_ciphers = ()
        for cipher in owned:
            release_if_supported(cipher)

    def release_value_input(self) -> None:
        """Release the Delta V stream without touching emitted cache C."""

        owned = self.value_ciphers
        self.value_ciphers = ()
        for cipher in owned:
            release_if_supported(cipher)

    def release(self) -> None:
        owned = (
            self.query_ciphers
            + self.key_ciphers
            + self.value_ciphers
            + self.key_c_ciphers
            + self.value_c_ciphers
            + self.pairfold_value_ciphers
        )
        self.query_ciphers = ()
        self.key_ciphers = ()
        self.value_ciphers = ()
        self.key_c_ciphers = ()
        self.value_c_ciphers = ()
        self.pairfold_value_ciphers = ()
        for cipher in owned:
            release_if_supported(cipher)


def _layout_parameters(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
) -> dict[str, int]:
    if int(carrier_layout.seq_len) != int(attention_layout.seq_len):
        raise ValueError("carrier and attention layouts must have the same sequence length.")
    if int(carrier_layout.slots) != int(attention_layout.slots):
        raise ValueError("carrier and attention layouts must have the same slot count.")
    token_lanes = int(carrier_layout.token_lanes)
    seq_len = int(attention_layout.seq_len)
    head_dim = int(attention_layout.head_dim)
    if seq_len % token_lanes:
        raise ValueError(
            f"the direct adapter requires seq_len divisible by token_lanes, got {seq_len}/{token_lanes}."
        )
    if head_dim % 2:
        raise ValueError("folded Q/K carriers require an even head dimension.")
    query_pairs = int(attention_layout.query_pair_count)
    kv_pairs = int(attention_layout.key_value_pair_count)
    if kv_pairs % 2:
        raise ValueError("V/V carriers require an even number of KV head pairs.")
    source_count = seq_len // token_lanes
    if source_count != int(carrier_layout.cipher_count):
        raise ValueError("the direct adapter requires fully occupied carrier shards.")
    query_fold_width = query_pairs * head_dim
    key_fold_width = kv_pairs * head_dim
    value_carrier_count = kv_pairs // 2
    carrier_width = query_fold_width + key_fold_width + kv_pairs * head_dim
    if carrier_width > int(carrier_layout.dimension):
        raise ValueError(
            f"heterogeneous carrier width {carrier_width} exceeds carrier dimension "
            f"{carrier_layout.dimension}."
        )
    repetitions = int(attention_layout.payload_repetitions)
    if repetitions & (repetitions - 1):
        raise ValueError("payload repetitions must be a power of two for ciphertext repetition.")
    return {
        "slots": int(carrier_layout.slots),
        "token_lanes": token_lanes,
        "source_count": source_count,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "half": head_dim // 2,
        "query_pairs": query_pairs,
        "kv_pairs": kv_pairs,
        "value_carriers": value_carrier_count,
        "query_fold_width": query_fold_width,
        "key_fold_width": key_fold_width,
        "carrier_width": carrier_width,
        "qk_block": token_lanes * head_dim,
        "value_block": 2 * token_lanes * head_dim,
        "repeat_steps": max(0, repetitions.bit_length() - 1),
    }


def _canonical_offsets(values, *, slots: int) -> tuple[int, ...]:
    return tuple(sorted({_canonical_rotation(int(value), int(slots)) for value in values}))


@lru_cache(maxsize=64)
def _choose_sparse_diagonal_plan_cached(
    diagonal_offsets: tuple[int, ...], slots: int, max_baby_offsets: int
) -> SparseDiagonalPlan:
    offsets = _canonical_offsets(diagonal_offsets, slots=int(slots))
    if not offsets:
        raise ValueError("a sparse diagonal transform requires at least one diagonal.")
    max_baby_offsets = int(max_baby_offsets)
    if max_baby_offsets <= 0:
        raise ValueError("max_baby_offsets must be positive.")

    # Searching up to 4096 finds the production optima (1104 for Q/K and 238
    # for V) without making small layout-contract tests expensive.
    search_stop = max(2, min(int(slots) // 2, 4096))
    best: tuple[int, int, int, int, tuple[int, ...], tuple[int, ...]] | None = None
    for modulus in range(1, search_stop + 1):
        babies = tuple(sorted({int(offset) % modulus for offset in offsets}))
        if len(babies) > max_baby_offsets:
            continue
        giants = tuple(
            sorted(
                {
                    _canonical_rotation(int(offset) - (int(offset) % modulus), int(slots))
                    for offset in offsets
                }
            )
        )
        rotations = len(babies) - int(0 in babies) + len(giants) - int(0 in giants)
        candidate = (rotations, len(babies), len(giants), modulus, babies, giants)
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    if best is None:
        raise ValueError(
            f"no sparse decomposition fits max_baby_offsets={max_baby_offsets}."
        )
    _, _, _, modulus, babies, giants = best
    return SparseDiagonalPlan(
        modulus=int(modulus),
        diagonal_offsets=offsets,
        baby_offsets=tuple(int(value) for value in babies),
        giant_offsets=tuple(int(value) for value in giants),
    )


def choose_sparse_diagonal_plan(
    diagonal_offsets: tuple[int, ...] | list[int],
    *,
    slots: int,
    max_baby_offsets: int = 32,
) -> SparseDiagonalPlan:
    return _choose_sparse_diagonal_plan_cached(
        tuple(int(value) for value in diagonal_offsets),
        int(slots),
        int(max_baby_offsets),
    )


def _value_delta_routes(
    params: dict[str, int], *, sources: tuple[int, ...] | None = None
) -> dict[int, np.ndarray]:
    """Compact V gather -> Delta form.

    ``sources`` restricts the diagonals to the blocks of the given lane
    sources.  The compact carrier holds every other source's block as zeros,
    so a restricted route set applied to a compact carrier gathered from the
    same sources is *exactly* the full transform restricted to those sources;
    summing the per-source-group results reproduces the monolithic output.
    """

    slots = params["slots"]
    token_lanes = params["token_lanes"]
    source_count = params["source_count"]
    seq_len = params["seq_len"]
    head_dim = params["head_dim"]
    value_block = params["value_block"]
    if sources is None:
        sources = tuple(range(source_count))
    else:
        sources = tuple(int(value) for value in sources)
        if any(value < 0 or value >= source_count for value in sources):
            raise IndexError("value delta route sources are outside the source axis.")
    destinations: dict[int, list[int]] = {}
    for source in sources:
        for local_token in range(token_lanes):
            token = source * token_lanes + local_token
            for feature in range(head_dim):
                query = (token - feature) % seq_len
                for lane in (0, 1):
                    source_slot = (
                        source * value_block
                        + 2 * token_lanes * feature
                        + 2 * local_token
                        + lane
                    )
                    dst = 2 * (feature * seq_len + query) + lane
                    offset = _canonical_rotation(source_slot - dst, slots)
                    destinations.setdefault(offset, []).append(dst)
    return {
        int(offset): np.asarray(indices, dtype=np.int64)
        for offset, indices in destinations.items()
    }


def _value_c_routes(
    params: dict[str, int], *, sources: tuple[int, ...] | None = None
) -> dict[int, np.ndarray]:
    """Compact V gather → real C, same source slots as Delta but unsherared.

    ``sources`` restricts the routes to the given lane sources exactly like
    :func:`_value_delta_routes` (the multi-card shard emits a per-source-group
    V_C partial that the consumer sums).
    """

    slots = params["slots"]
    token_lanes = params["token_lanes"]
    source_count = params["source_count"]
    seq_len = params["seq_len"]
    head_dim = params["head_dim"]
    value_block = params["value_block"]
    if sources is None:
        sources = tuple(range(source_count))
    else:
        sources = tuple(int(value) for value in sources)
        if any(value < 0 or value >= source_count for value in sources):
            raise IndexError("value C route sources are outside the source axis.")
    destinations: dict[int, list[int]] = {}
    for source in sources:
        for local_token in range(token_lanes):
            token = source * token_lanes + local_token
            for feature in range(head_dim):
                for lane in (0, 1):
                    source_slot = (
                        source * value_block
                        + 2 * token_lanes * feature
                        + 2 * local_token
                        + lane
                    )
                    dst = 2 * (feature * seq_len + token) + lane
                    offset = _canonical_rotation(source_slot - dst, slots)
                    destinations.setdefault(offset, []).append(dst)
    return {
        int(offset): np.asarray(indices, dtype=np.int64)
        for offset, indices in destinations.items()
    }


@lru_cache(maxsize=8)
def _value_delta_pairfold_routes(
    pair: int,
    occupied: int,
    *,
    seq_len: int = 128,
    head_dim: int = 128,
    slots: int = 32768,
) -> dict[int, SparseMaskRoute]:
    """Unshear one real Delta-V pair and fold it into its PairFold page block."""

    if pair not in (0, 1) or not 1 <= occupied <= seq_len:
        raise ValueError("PairFold V routes need pair 0/1 and a nonempty page.")
    if seq_len != 128 or head_dim != 128 or slots != 32768:
        raise ValueError("Delta-V PairFold routes require the PairFold128 layout.")
    destinations: dict[int, list[int]] = {}
    coefficients: dict[int, list[complex]] = {}
    half = head_dim // 2
    for feature in range(head_dim):
        folded_feature = feature % half
        coefficient = 1j if feature >= half else 1.0
        for token in range(seq_len):
            query = (token - feature) % seq_len
            for lane in (0, 1):
                source = 2 * (feature * seq_len + query) + lane
                destination = (
                    pair * (slots // 2)
                    + 2 * (folded_feature * seq_len + token)
                    + lane
                )
                offset = _canonical_rotation(source - destination, slots)
                destinations.setdefault(offset, []).append(destination)
                coefficients.setdefault(offset, []).append(
                    coefficient if token < occupied else 0.0
                )
    return {
        offset: SparseMaskRoute(
            destinations=np.asarray(indices, dtype=np.int64),
            coefficients=np.asarray(coefficients[offset], dtype=np.complex128),
        )
        for offset, indices in destinations.items()
    }


@lru_cache(maxsize=1)
def _value_delta_pairfold_low_routes() -> dict[int, np.ndarray]:
    """Undo the low four bits of the Delta-V feature shear."""

    seq_len = 128
    slots = 32768
    destinations: dict[int, list[int]] = {}
    for feature in range(128):
        shift = feature % 16
        for query in range(seq_len):
            token = (query + shift) % seq_len
            for lane in (0, 1):
                source = 2 * (feature * seq_len + query) + lane
                destination = 2 * (feature * seq_len + token) + lane
                offset = _canonical_rotation(source - destination, slots)
                destinations.setdefault(offset, []).append(destination)
    return {
        offset: np.asarray(indices, dtype=np.int64)
        for offset, indices in destinations.items()
    }


@lru_cache(maxsize=256)
def _value_delta_pairfold_high_routes(
    pair: int, occupied: int
) -> dict[int, SparseMaskRoute]:
    """Finish the shear and fold the two feature halves into a page block."""

    if pair not in (0, 1) or not 1 <= occupied <= 128:
        raise ValueError("PairFold V routes need pair 0/1 and a nonempty page.")
    seq_len = 128
    slots = 32768
    destinations: dict[int, list[int]] = {}
    coefficients: dict[int, list[complex]] = {}
    for feature in range(128):
        shift = feature - feature % 16
        folded_feature = feature % 64
        coefficient = 1j if feature >= 64 else 1.0
        for intermediate_token in range(seq_len):
            token = (intermediate_token + shift) % seq_len
            for lane in (0, 1):
                source = 2 * (feature * seq_len + intermediate_token) + lane
                destination = (
                    pair * (slots // 2)
                    + 2 * (folded_feature * seq_len + token)
                    + lane
                )
                offset = _canonical_rotation(source - destination, slots)
                destinations.setdefault(offset, []).append(destination)
                coefficients.setdefault(offset, []).append(
                    coefficient if token < occupied else 0.0
                )
    return {
        offset: SparseMaskRoute(
            destinations=np.asarray(indices, dtype=np.int64),
            coefficients=np.asarray(coefficients[offset], dtype=np.complex128),
        )
        for offset, indices in destinations.items()
    }


def _value_delta_pairfold_staged_enabled() -> bool:
    return os.environ.get("LLAMA_FHE_PREFILL_DELTA_PAIRFOLD_V_STAGED", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _qk_axis_masks(params: dict[str, int], *, stage: int) -> tuple[np.ndarray, int]:
    """Two 31-diagonal swaps for ``[source 16][feature 4][feature 16]``."""

    source_count = params["source_count"]
    token_lanes = params["token_lanes"]
    half = params["half"]
    if source_count != 16 or half != 64:
        raise NotImplementedError(
            "the selected two-stage Q/K axis swap currently requires source=16 and half=64."
        )
    high_radix = 4
    low_radix = 16
    suffix = 2 * token_lanes
    masks = np.zeros((31, params["slots"]), dtype=np.float64)
    if int(stage) == 1:
        stride = 3 * low_radix * suffix
        for source in range(source_count):
            for high in range(high_radix):
                relative = source - 5 * high
                row = relative + 15
                for low in range(low_radix):
                    for tail in range(suffix):
                        destination = (
                            ((high * source_count + source) * low_radix + low)
                            * suffix
                            + tail
                        )
                        masks[row, destination] = 1.0
    elif int(stage) == 2:
        stride = 15 * suffix
        for high in range(high_radix):
            for source in range(source_count):
                for low in range(low_radix):
                    relative = source - low
                    row = relative + 15
                    for tail in range(suffix):
                        destination = (
                            ((high * low_radix + low) * source_count + source)
                            * suffix
                            + tail
                        )
                        masks[row, destination] = 1.0
    else:
        raise ValueError(f"Q/K axis stage must be 1 or 2, got {stage}.")
    return masks, int(stride)


def _qk_fused_gather_axis1_masks(
    params: dict[str, int],
    *,
    source: int,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    """Masks for target gather fused with the first Q/K axis swap.

    The input of source ``s`` is the output of :func:`_apply_lane_phase`:

    ``[target][feature_high 4][feature_low 16][token 8][lane 2]``.

    One plaintext multiplication routes a fixed ``(target, high)`` rectangle
    directly into the intermediate

    ``[feature_high 4][source 16][feature_low 16][token 8][lane 2]``.

    There are exactly ``(Q-pairs + K-pairs) * 4`` useful rows per source,
    i.e. 80 rows for production Llama.  Padding is intentionally not part of
    this stage: unlike ordinary regular BSGS, these are heterogeneous target
    accumulators and every useful row already has an independent destination.
    """

    source = int(source)
    if params["source_count"] != 16 or params["half"] != 64:
        raise NotImplementedError(
            "the fused Q/K gather-axis stage requires source=16 and half=64."
        )
    if not 0 <= source < params["source_count"]:
        raise IndexError(f"source {source} is outside the source-shard axis.")
    targets = params["query_pairs"] + params["kv_pairs"]
    suffix = 2 * params["token_lanes"]
    target_width = params["head_dim"] * params["token_lanes"]
    # The destination mask depends on ``source`` and ``high``, but not on the
    # target.  Encode four unique rows and broadcast each over the 20 target
    # rotations; this is the key distinction between 64 reusable rows and
    # 1,280 PT-CT uses in production.
    masks = np.zeros((4, params["slots"]), dtype=np.float64)
    metadata: list[tuple[int, int, int]] = []
    for high in range(4):
        for low in range(16):
            destination = (
                ((high * params["source_count"] + source) * 16 + low)
                * suffix
            )
            masks[high, destination : destination + suffix] = 1.0
    for target in range(targets):
        for high in range(4):
            source_start = target * target_width + high * 16 * suffix
            destination_start = (
                (high * params["source_count"] + source) * 16 * suffix
            )
            offset = _canonical_rotation(
                source_start - destination_start, params["slots"]
            )
            metadata.append((target, high, offset))
    return masks, tuple(metadata)


def _qk_fused_gather_axis1_rotation_count(params: dict[str, int]) -> int:
    """Exact logical rotation count of the correctness-first scheduler."""

    total = 0
    for source in range(params["source_count"]):
        _, metadata = _qk_fused_gather_axis1_masks(params, source=source)
        offsets = {int(offset) for _, _, offset in metadata}
        total += len(offsets) - int(0 in offsets)
    return int(total)


def _key_shear_routes(params: dict[str, int], *, upper: bool) -> dict[int, np.ndarray]:
    """Folded K lower shear or conjugate-derived upper shear."""

    slots = params["slots"]
    seq_len = params["seq_len"]
    half = params["half"]
    destinations: dict[int, list[int]] = {}
    for feature in range(half):
        physical_feature = feature + (half if upper else 0)
        for query in range(seq_len):
            source_token = (query + physical_feature) % seq_len
            for lane in (0, 1):
                source_slot = 2 * (feature * seq_len + source_token) + lane
                dst = 2 * (physical_feature * seq_len + query) + lane
                offset = _canonical_rotation(source_slot - dst, slots)
                destinations.setdefault(offset, []).append(dst)
    return {
        int(offset): np.asarray(indices, dtype=np.int64)
        for offset, indices in destinations.items()
    }


def _block_gather_routes(params: dict[str, int], source: int):
    qk_block = params["qk_block"]
    value_block = params["value_block"]
    q_base = 0
    k_base = params["token_lanes"] * params["query_fold_width"]
    v_base = params["token_lanes"] * (
        params["query_fold_width"] + params["key_fold_width"]
    )
    routes: list[tuple[str, int, int, str]] = []
    for pair in range(params["query_pairs"]):
        routes.append(
            ("q", pair, q_base + pair * qk_block - source * qk_block, "qk")
        )
    for pair in range(params["kv_pairs"]):
        routes.append(
            ("k", pair, k_base + pair * qk_block - source * qk_block, "qk")
        )
    for carrier in range(params["value_carriers"]):
        routes.append(
            (
                "v",
                carrier,
                v_base + carrier * value_block - source * value_block,
                "v",
            )
        )
    return tuple(routes)


def complex_token_attention_complexity(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    *,
    sparse_max_baby_offsets: int = 32,
    shear_baby_steps: int = 16,
) -> ComplexTokenAttentionComplexity:
    if int(shear_baby_steps) != 16:
        raise ValueError(
            "the selected hybrid adapter currently requires shear_baby_steps=16."
        )
    params = _layout_parameters(carrier_layout, attention_layout)
    # The selected production factorization is specialized to the Llama
    # source/feature axes.  Small arbitrary layouts retain the CPU oracle but
    # do not claim this production static audit.
    if params["source_count"] != 16 or params["half"] != 64:
        raise NotImplementedError(
            "the staged production adapter audit requires 16 source shards and head_dim=128."
        )
    lane_rows = params["source_count"] * (2 * params["token_lanes"] - 1)
    lane_rotations = params["source_count"] * (
        2 * params["token_lanes"] - 2
    )
    qk_transforms = params["query_pairs"] + params["kv_pairs"]
    fused_axis1_unique_rows = params["source_count"] * 4
    fused_axis1_ptct = params["source_count"] * qk_transforms * 4
    fused_axis1_rotations = _qk_fused_gather_axis1_rotation_count(params)
    axis_rows = 32  # 31 useful rows padded to one 16x2 BSGS group.
    axis_rotations_per_cipher = 1 + 15 + 1
    rectangular_ptct = fused_axis1_ptct + axis_rows * qk_transforms
    rectangular_rotations = (
        fused_axis1_rotations + axis_rotations_per_cipher * qk_transforms
    )

    lower_routes = _key_shear_routes(params, upper=False)
    upper_routes = _key_shear_routes(params, upper=True)
    lower_plan = choose_sparse_diagonal_plan(
        tuple(lower_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    upper_plan = choose_sparse_diagonal_plan(
        tuple(upper_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    shear_ptct = 256 * params["kv_pairs"]
    shear_rotations = (
        lower_plan.rotations_per_cipher + upper_plan.rotations_per_cipher
    ) * params["kv_pairs"]

    value_routes = _value_delta_routes(params)
    value_gather_rows = params["source_count"]
    value_gather_ptct = (
        params["source_count"] * params["value_carriers"]
    )
    value_gather_rotations = sum(
        len(
            {
                _canonical_rotation(route[2], params["slots"])
                for route in _block_gather_routes(params, source)
                if route[0] == "v"
            }
        )
        - int(
            0
            in {
                _canonical_rotation(route[2], params["slots"])
                for route in _block_gather_routes(params, source)
                if route[0] == "v"
            }
        )
        for source in range(params["source_count"])
    )
    value_plan = choose_sparse_diagonal_plan(
        tuple(value_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    value_final_ciphers = params["value_carriers"]
    value_ptct = value_plan.plaintext_rows * value_final_ciphers
    value_rotations = value_plan.rotations_per_cipher * value_final_ciphers
    feature_count = params["query_pairs"]
    repeat_rotations = params["repeat_steps"] * (
        params["query_pairs"] + 2 * params["kv_pairs"]
    )
    encoded_rows = (
        lane_rows
        + fused_axis1_unique_rows
        + axis_rows
        + value_gather_rows
        + value_plan.plaintext_rows
        + 256
    )
    ptct = (
        lane_rows
        + rectangular_ptct
        + value_gather_ptct
        + value_ptct
        + shear_ptct
    )
    rotations = (
        lane_rotations
        + rectangular_rotations
        + feature_count
        + shear_rotations
        + value_gather_rotations
        + value_rotations
        + repeat_rotations
    )
    conjugations = (
        params["query_pairs"] + params["kv_pairs"] + params["value_carriers"]
    )
    return ComplexTokenAttentionComplexity(
        input_cipher_count=params["source_count"],
        query_output_cipher_count=params["query_pairs"],
        key_output_cipher_count=params["kv_pairs"],
        value_output_cipher_count=params["kv_pairs"],
        token_lanes=params["token_lanes"],
        lane_phase_plaintext_rows=lane_rows,
        lane_phase_pt_ct_multiplications=lane_rows,
        lane_phase_rotations=lane_rotations,
        block_gather_plaintext_rows=value_gather_rows,
        block_gather_pt_ct_multiplications=value_gather_ptct,
        block_gather_rotations=value_gather_rotations,
        rectangular_plaintext_rows=fused_axis1_unique_rows + axis_rows,
        rectangular_pt_ct_multiplications=rectangular_ptct,
        rectangular_rotations=rectangular_rotations,
        rectangular_bsgs_modulus=16,
        feature_expansion_rotations=feature_count,
        feature_expansion_conjugations=params["query_pairs"],
        key_shear_plaintext_rows=256,
        key_shear_pt_ct_multiplications=shear_ptct,
        key_shear_rotations=shear_rotations,
        value_delta_plaintext_rows=value_plan.plaintext_rows,
        value_delta_pt_ct_multiplications=value_ptct,
        value_delta_rotations=value_rotations,
        value_delta_bsgs_modulus=value_plan.modulus,
        value_split_conjugations=params["value_carriers"],
        payload_repeat_rotations=repeat_rotations,
        encoded_static_plaintext_rows=encoded_rows,
        pt_ct_multiplications=ptct,
        rotations=rotations,
        conjugations=conjugations,
        query_multiplicative_depth=3,
        key_multiplicative_depth=4,
        value_multiplicative_depth=3,
    )


def complex_token_attention_rotations(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    *,
    sparse_max_baby_offsets: int = 32,
    shear_baby_steps: int = 16,
    log_n: int = 16,
) -> tuple[int, ...]:
    if int(shear_baby_steps) != 16:
        raise ValueError(
            "the selected hybrid adapter currently requires shear_baby_steps=16."
        )
    params = _layout_parameters(carrier_layout, attention_layout)
    if params["source_count"] != 16 or params["half"] != 64:
        raise NotImplementedError(
            "production staged adapter rotations require 16 source shards and head_dim=128."
        )
    value_routes = _value_delta_routes(params)
    value_plan = choose_sparse_diagonal_plan(
        tuple(value_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    lower_plan = choose_sparse_diagonal_plan(
        tuple(_key_shear_routes(params, upper=False)),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    upper_plan = choose_sparse_diagonal_plan(
        tuple(_key_shear_routes(params, upper=True)),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    rotations = set(
        range(-(params["token_lanes"] - 1), params["token_lanes"])
    )
    for source in range(params["source_count"]):
        _, metadata = _qk_fused_gather_axis1_masks(params, source=source)
        rotations.update(int(offset) for _, _, offset in metadata)
        rotations.update(
            _canonical_rotation(route[2], params["slots"])
            for route in _block_gather_routes(params, source)
            if route[0] == "v"
        )
    rotations.update(value_plan.baby_offsets)
    rotations.update(value_plan.giant_offsets)
    rotations.update(lower_plan.baby_offsets)
    rotations.update(lower_plan.giant_offsets)
    rotations.update(upper_plan.baby_offsets)
    rotations.update(upper_plan.giant_offsets)
    stride = _qk_axis_masks(params, stage=2)[1]
    rotations.add(-15 * int(stride))
    rotations.update(index * int(stride) for index in range(1, 16))
    rotations.add(16 * int(stride))
    rotations.add(-int(attention_layout.payload_slots) // 2)
    rotations.add((1 << (int(log_n) + 1)) - 1)
    step = int(attention_layout.payload_slots)
    while step < int(attention_layout.slots):
        rotations.add(-step)
        step *= 2
    return tuple(sorted(int(value) for value in rotations if int(value)))


def emit_cache_c_rotations(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    *,
    sparse_max_baby_offsets: int = 32,
    direct_pairfold_v: bool = False,
) -> tuple[int, ...]:
    """Extra keys for the selected Prefill V cache-emission route."""

    params = _layout_parameters(carrier_layout, attention_layout)
    if direct_pairfold_v and _value_delta_pairfold_staged_enabled():
        routes = (
            _value_delta_pairfold_low_routes(),
            *(_value_delta_pairfold_high_routes(pair, 128) for pair in (0, 1)),
        )
    elif direct_pairfold_v:
        routes = tuple(_value_delta_pairfold_routes(pair, 128) for pair in (0, 1))
    else:
        routes = (_value_c_routes(params),)
    rotations: set[int] = set()
    for route in routes:
        plan = choose_sparse_diagonal_plan(
            tuple(route),
            slots=params["slots"],
            max_baby_offsets=int(sparse_max_baby_offsets),
        )
        rotations.update(int(value) for value in plan.baby_offsets)
        rotations.update(int(value) for value in plan.giant_offsets)
    return tuple(sorted(value for value in rotations if value))


def _lane_phase_masks(
    *,
    source: int,
    params: dict[str, int],
    cos: np.ndarray,
    sin: np.ndarray,
) -> dict[int, np.ndarray]:
    masks = {
        offset: np.zeros((params["slots"],), dtype=np.complex128)
        for offset in range(-(params["token_lanes"] - 1), params["token_lanes"])
    }
    q_end = params["query_fold_width"]
    k_end = q_end + params["key_fold_width"]
    for channel in range(params["carrier_width"]):
        pair_feature = channel // 2
        head_lane = channel % 2
        for local_token in range(params["token_lanes"]):
            token = source * params["token_lanes"] + local_token
            source_slot = channel * params["token_lanes"] + local_token
            dst = 2 * params["token_lanes"] * pair_feature + 2 * local_token + head_lane
            offset = source_slot - dst
            coefficient: complex = 1.0
            if channel < q_end:
                feature = pair_feature % params["half"]
                coefficient = complex(cos[token, feature], sin[token, feature])
            elif channel < k_end:
                local_pair_feature = (channel - q_end) // 2
                feature = local_pair_feature % params["half"]
                coefficient = complex(cos[token, feature], -sin[token, feature])
            masks[offset][dst] = coefficient
    return masks


def _lane_phase_hoisted_enabled(crypto_context) -> bool:
    """Hoisted lane/phase (``LLAMA_FHE_LANE_PHASE_HOISTED``, default on) when the backend has the batched APIs."""

    if os.environ.get("LLAMA_FHE_LANE_PHASE_HOISTED", "1").lower() in ("0", "false", "no", "off"):
        return False
    return all(
        hasattr(crypto_context.fhe, name)
        for name in ("fast_rotate", "grouped_pairwise_mac", "moddown_from_ext")
    )


def _apply_lane_phase_hoisted(
    ciphers: tuple[object, ...],
    *,
    params: dict[str, int],
    cos: np.ndarray,
    sin: np.ndarray,
    crypto_context,
    function_prefix: str,
    source_indices: tuple[int, ...],
) -> tuple[object, ...]:
    """:func:`_apply_lane_phase` as one hoisted diagonal MAC per carrier.

    Same terms (``2 * token_lanes - 1`` diagonals per carrier, one PT x CT
    product each, one rescale), scheduled like the sparse Q/K/V transforms:
    the carrier is mod-upped once and all diagonals come out of one fast
    rotation batch in the extended basis, the masks are encoded as one row
    batch (a session constant, so the plaintext cache keeps it across
    layers), one grouped MAC accumulates the products and a single ModDown
    closes the sum.  The sequential path pays a full key switch (ModUp +
    inner product + ModDown) per diagonal and one encode per mask.
    """

    outputs: list[object] = []
    try:
        for source, cipher in zip(source_indices, ciphers, strict=True):
            level = int(crypto_context.level_for_cipher(cipher))
            masks = _lane_phase_masks(
                source=source, params=params, cos=cos, sin=sin
            )
            offsets = tuple(
                offset for offset, mask in sorted(masks.items()) if np.any(mask)
            )
            if not offsets:
                raise RuntimeError("lane/phase transform produced no terms.")
            rows = np.stack([masks[offset] for offset in offsets])
            rotations = plaintexts = product = total = None
            try:
                rotations = _fast_baby_rotation_batch(
                    cipher, baby_offsets=offsets, crypto_context=crypto_context
                )
                plaintexts = encode_transform_rows(
                    rows,
                    name=f"{function_prefix}.source{source}.lane_phase",
                    level=level,
                    crypto_context=crypto_context,
                    is_ext=bool(rotations.is_ext),
                )
                product = crypto_context.fhe.grouped_pairwise_mac(
                    rotations, plaintexts, 1, crypto_context.context
                )
                total = (
                    crypto_context.fhe.moddown_from_ext(
                        product, crypto_context.context
                    )
                    if product.is_ext
                    else product.deep_copy()
                )
                outputs.append(
                    crypto_context.fhe.rescale(total, crypto_context.context)
                )
            finally:
                release_if_supported(total)
                release_if_supported(product)
                release_if_supported(plaintexts)
                release_if_supported(rotations)
        return tuple(outputs)
    except Exception:
        for output in outputs:
            release_if_supported(output)
        raise


def _apply_lane_phase(
    ciphers: tuple[object, ...],
    *,
    params: dict[str, int],
    cos: np.ndarray,
    sin: np.ndarray,
    crypto_context,
    function_prefix: str,
    source_indices: tuple[int, ...] | None = None,
) -> tuple[object, ...]:
    if source_indices is None:
        source_indices = tuple(range(len(ciphers)))
    else:
        source_indices = tuple(int(value) for value in source_indices)
    if len(source_indices) != len(ciphers):
        raise ValueError(
            "source_indices must identify every lane/phase input cipher"
        )
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("lane/phase source_indices must be unique")
    if any(
        source < 0 or source >= int(params["source_count"])
        for source in source_indices
    ):
        raise IndexError("lane/phase source index is outside the source axis")
    if _lane_phase_hoisted_enabled(crypto_context):
        return _apply_lane_phase_hoisted(
            ciphers,
            params=params,
            cos=cos,
            sin=sin,
            crypto_context=crypto_context,
            function_prefix=function_prefix,
            source_indices=source_indices,
        )
    outputs: list[object] = []
    try:
        for source, cipher in zip(source_indices, ciphers, strict=True):
            level = int(crypto_context.level_for_cipher(cipher))
            accumulator = None
            masks = _lane_phase_masks(
                source=source, params=params, cos=cos, sin=sin
            )
            try:
                for offset, mask in sorted(masks.items()):
                    plain = rotated = product = None
                    try:
                        plain = _encode_mask(
                            mask,
                            name=f"{function_prefix}.source{source}.diag{offset}",
                            level=level,
                            crypto_context=crypto_context,
                            dtype=np.complex128,
                        )
                        rotated = (
                            cipher
                            if offset == 0
                            else crypto_context.fhe.homo_rotate(
                                cipher, offset, crypto_context.context
                            )
                        )
                        product = crypto_context.fhe.homo_mul_pt(
                            rotated, plain, crypto_context.context
                        )
                        accumulator = _add_owned(
                            accumulator, product, crypto_context=crypto_context
                        )
                        product = None
                    finally:
                        release_if_supported(plain)
                        if rotated is not cipher:
                            release_if_supported(rotated)
                        release_if_supported(product)
                if accumulator is None:
                    raise RuntimeError("lane/phase transform produced no terms.")
                output = crypto_context.fhe.rescale(
                    accumulator, crypto_context.context
                )
                release_if_supported(accumulator)
                accumulator = None
                outputs.append(output)
            finally:
                release_if_supported(accumulator)
        return tuple(outputs)
    except Exception:
        for output in outputs:
            release_if_supported(output)
        raise


def _apply_qk_fused_gather_axis1(
    ciphers: tuple[object, ...],
    *,
    params: dict[str, int],
    crypto_context,
    function_prefix: str,
    target_indices: tuple[int, ...] | None = None,
) -> tuple[object, ...]:
    """Gather every Q/K target while performing axis-swap stage one.

    This is deliberately a correctness-first scheduler.  It batches the 80
    source-specific rotations and plaintext products, then accumulates the
    four feature-high contributions for each target.  All target accumulators
    are rescaled once, so gather and the first axis swap consume one level.
    """

    if len(ciphers) != params["source_count"]:
        raise ValueError(
            f"expected {params['source_count']} source ciphers, got {len(ciphers)}."
        )
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("fused Q/K gather-axis inputs must have one common level.")
    level = levels.pop()
    target_count = params["query_pairs"] + params["kv_pairs"]
    if target_indices is None:
        target_indices = tuple(range(target_count))
    else:
        target_indices = tuple(int(value) for value in target_indices)
    if len(set(target_indices)) != len(target_indices):
        raise ValueError("Q/K gather target_indices must be unique")
    if any(target < 0 or target >= target_count for target in target_indices):
        raise IndexError("Q/K gather target index is outside the target axis")
    if not target_indices:
        return ()
    accumulators: list[object | None] = [None] * len(target_indices)
    outputs: list[object] = []
    try:
        # Four rotations are needed per target.  EasyFHE's current broadcast
        # kernel accepts at most 32 rotations in one call, so process eight
        # targets (4*8 rotations) per hoisted batch at production shape.
        target_batch_size = 8
        for source, cipher in enumerate(ciphers):
            masks, metadata = _qk_fused_gather_axis1_masks(
                params, source=source
            )
            plaintexts = None
            try:
                plaintexts = encode_transform_rows(
                    masks,
                    name=f"{function_prefix}.source{source}",
                    level=level,
                    crypto_context=crypto_context,
                )
                for target_start in range(0, len(target_indices), target_batch_size):
                    target_stop = min(
                        len(target_indices), target_start + target_batch_size
                    )
                    batch_targets = target_indices[target_start:target_stop]
                    batch_metadata = tuple(
                        metadata[4 * target + high]
                        for target in batch_targets
                        for high in range(4)
                    )
                    offsets = tuple(int(item[2]) for item in batch_metadata)
                    rotations = None
                    try:
                        rotations = crypto_context.fhe.fast_rotate(
                            cipher,
                            offsets,
                            crypto_context.context,
                            output_ext=True,
                        )
                        rotation_items = crypto_context.fhe.unpack_cipher_batch(
                            rotations
                        )
                        for local_target, accumulator_index in enumerate(
                            range(target_start, target_stop)
                        ):
                            rotated_target = product = None
                            try:
                                # Materialize this target's four rotations as
                                # one contiguous batch before multiplying.
                                # A strided cipher_like view across targets is
                                # not a valid backend batch on full-head runs.
                                rotated_target = crypto_context.fhe.pack_cipher_batch(
                                    rotation_items[4 * local_target + high]
                                    for high in range(4)
                                )
                                product = crypto_context.fhe.grouped_pairwise_mac(
                                    rotated_target,
                                    plaintexts,
                                    1,
                                    crypto_context.context,
                                )
                                contribution = (
                                    crypto_context.fhe.moddown_from_ext(
                                        product, crypto_context.context
                                    )
                                    if product.is_ext
                                    else product.deep_copy()
                                )
                                accumulators[accumulator_index] = _add_owned(
                                    accumulators[accumulator_index],
                                    contribution,
                                    crypto_context=crypto_context,
                                )
                            finally:
                                release_if_supported(product)
                                release_if_supported(rotated_target)
                    finally:
                        release_if_supported(rotations)
            finally:
                release_if_supported(plaintexts)

        for index, accumulator in enumerate(accumulators):
            if accumulator is None:
                raise RuntimeError(f"fused Q/K output {index} is empty.")
            try:
                output = crypto_context.fhe.rescale(
                    accumulator, crypto_context.context
                )
            finally:
                release_if_supported(accumulator)
                accumulators[index] = None
            outputs.append(output)
        return tuple(outputs)
    except Exception:
        for accumulator in accumulators:
            release_if_supported(accumulator)
        for output in outputs:
            release_if_supported(output)
        raise


def _apply_value_block_gather(
    ciphers: tuple[object, ...],
    *,
    params: dict[str, int],
    crypto_context,
    function_prefix: str,
    target_indices: tuple[int, ...] | None = None,
    source_indices: tuple[int, ...] | None = None,
) -> tuple[object, ...]:
    """Gather only the two complex V/V carriers from lane-phase outputs.

    Q/K use the fused gather-axis stage above.  Keeping this tiny V-only
    gather makes the subsequent direct compact-to-Delta transform one level,
    preserving the selected V depth while using 16 encoded masks and 32
    PT-CT products in production.

    ``source_indices`` restricts the accumulation to those lane sources (the
    T2 own-source partial); ``ciphers`` still has one slot per global source
    but the unused slots may be ``None``.
    """

    if len(ciphers) != params["source_count"]:
        raise ValueError(
            f"expected {params['source_count']} source ciphers, got {len(ciphers)}."
        )
    if source_indices is None:
        source_indices = tuple(range(int(params["source_count"])))
    else:
        source_indices = tuple(int(value) for value in source_indices)
        if len(set(source_indices)) != len(source_indices):
            raise ValueError("V gather source_indices must be unique")
        if any(
            source < 0 or source >= int(params["source_count"])
            for source in source_indices
        ):
            raise IndexError("V gather source index is outside the source axis")
        if not source_indices:
            raise ValueError("V gather needs at least one source")
    used = tuple(ciphers[source] for source in source_indices)
    if any(cipher is None for cipher in used):
        raise ValueError("V gather source slots must hold ciphers")
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in used}
    if len(levels) != 1:
        raise ValueError("V gather inputs must have one common level.")
    level = levels.pop()
    if target_indices is None:
        target_indices = tuple(range(int(params["value_carriers"])))
    else:
        target_indices = tuple(int(value) for value in target_indices)
    if len(set(target_indices)) != len(target_indices):
        raise ValueError("V gather target_indices must be unique")
    if any(
        target < 0 or target >= int(params["value_carriers"])
        for target in target_indices
    ):
        raise IndexError("V gather target index is outside the target axis")
    if not target_indices:
        return ()
    target_positions = {
        target: position for position, target in enumerate(target_indices)
    }
    accumulators: list[object | None] = [None] * len(target_indices)
    outputs: list[object] = []
    try:
        for source in source_indices:
            cipher = ciphers[source]
            mask = np.zeros((params["slots"],), dtype=np.float64)
            start = source * params["value_block"]
            mask[start : start + params["value_block"]] = 1.0
            plain = _encode_mask(
                mask,
                name=f"{function_prefix}.source{source}",
                level=level,
                crypto_context=crypto_context,
                dtype=np.float64,
            )
            cache: dict[int, object] = {}
            try:
                for kind, index, raw_offset, _ in _block_gather_routes(
                    params, source
                ):
                    if kind != "v" or index not in target_positions:
                        continue
                    offset = _canonical_rotation(raw_offset, params["slots"])
                    rotated = cache.get(offset)
                    if rotated is None:
                        rotated = (
                            cipher
                            if offset == 0
                            else crypto_context.fhe.homo_rotate(
                                cipher, offset, crypto_context.context
                            )
                        )
                        cache[offset] = rotated
                    product = crypto_context.fhe.homo_mul_pt(
                        rotated, plain, crypto_context.context
                    )
                    position = target_positions[index]
                    accumulators[position] = _add_owned(
                        accumulators[position],
                        product,
                        crypto_context=crypto_context,
                    )
            finally:
                for offset, rotated in cache.items():
                    if offset and rotated is not cipher:
                        release_if_supported(rotated)
                release_if_supported(plain)

        for index, accumulator in enumerate(accumulators):
            if accumulator is None:
                raise RuntimeError(f"V gather output {index} is empty.")
            output = crypto_context.fhe.rescale(
                accumulator, crypto_context.context
            )
            release_if_supported(accumulator)
            accumulators[index] = None
            outputs.append(output)
        return tuple(outputs)
    except Exception:
        for accumulator in accumulators:
            release_if_supported(accumulator)
        for output in outputs:
            release_if_supported(output)
        raise


def _apply_sparse_routes(
    ciphers: tuple[object, ...],
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    plan: SparseDiagonalPlan,
    slots: int,
    crypto_context,
    function_prefix: str,
    padded_plaintext_rows: int | None = None,
    rescale_outputs: bool = True,
) -> tuple[object, ...]:
    if not ciphers:
        return ()
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("one sparse diagonal transform requires one input level.")
    level = levels.pop()
    by_giant: dict[int, list[tuple[int, int]]] = {}
    for offset in plan.diagonal_offsets:
        baby = int(offset) % int(plan.modulus)
        giant = _canonical_rotation(int(offset) - baby, int(slots))
        by_giant.setdefault(giant, []).append((int(offset), baby))

    rotated_inputs: list[dict[int, object]] = []
    totals: list[object | None] = [None] * len(ciphers)
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            cache: dict[int, object] = {}
            # Register the cache before creating rotations so the outer
            # cleanup also owns a partially constructed cache.
            rotated_inputs.append(cache)
            for baby in plan.baby_offsets:
                cache[baby] = (
                    cipher
                    if baby == 0
                    else crypto_context.fhe.homo_rotate(
                        cipher, baby, crypto_context.context
                    )
                )

        for giant, entries in sorted(by_giant.items()):
            plains: list[tuple[int, object]] = []
            try:
                for offset, baby in entries:
                    destinations, coefficients = sparse_route_arrays(
                        routes[offset]
                    )
                    mask = np.zeros(
                        (int(slots),),
                        dtype=np.result_type(np.float64, coefficients.dtype),
                    )
                    mask[destinations] = coefficients
                    if giant:
                        mask = np.roll(mask, int(giant))
                    plains.append(
                        (
                            baby,
                            _encode_mask(
                                mask,
                                name=f"{function_prefix}.diag{offset}.g{giant}",
                                level=level,
                                crypto_context=crypto_context,
                                dtype=np.complex128 if np.iscomplexobj(mask) else np.float64,
                            ),
                        )
                    )
                for cipher_index, cache in enumerate(rotated_inputs):
                    group = None
                    for baby, plain in plains:
                        product = crypto_context.fhe.homo_mul_pt(
                            cache[baby], plain, crypto_context.context
                        )
                        group = _add_owned(
                            group, product, crypto_context=crypto_context
                        )
                    if group is None:
                        raise RuntimeError("sparse BSGS giant group was empty.")
                    if giant:
                        try:
                            contribution = crypto_context.fhe.homo_rotate(
                                group, giant, crypto_context.context
                            )
                        finally:
                            release_if_supported(group)
                    else:
                        contribution = group
                    totals[cipher_index] = _add_owned(
                        totals[cipher_index],
                        contribution,
                        crypto_context=crypto_context,
                    )
            finally:
                for _, plain in plains:
                    release_if_supported(plain)

        padded_rows = (
            plan.plaintext_rows
            if padded_plaintext_rows is None
            else int(padded_plaintext_rows)
        )
        if padded_rows < plan.plaintext_rows:
            raise ValueError(
                f"padded_plaintext_rows={padded_rows} is below the "
                f"{plan.plaintext_rows} nonzero routes."
            )
        for pad_index in range(padded_rows - plan.plaintext_rows):
            zero_plain = _encode_mask(
                np.zeros((int(slots),), dtype=np.float64),
                name=f"{function_prefix}.pad{pad_index}",
                level=level,
                crypto_context=crypto_context,
                dtype=np.float64,
            )
            try:
                for cipher_index, cipher in enumerate(ciphers):
                    zero_product = crypto_context.fhe.homo_mul_pt(
                        cipher, zero_plain, crypto_context.context
                    )
                    totals[cipher_index] = _add_owned(
                        totals[cipher_index],
                        zero_product,
                        crypto_context=crypto_context,
                    )
            finally:
                release_if_supported(zero_plain)

        for index, total in enumerate(totals):
            if total is None:
                raise RuntimeError(f"sparse diagonal output {index} is empty.")
            if bool(rescale_outputs):
                output = crypto_context.fhe.rescale(
                    total, crypto_context.context
                )
                release_if_supported(total)
            else:
                # Transfer ownership of the pre-rescale PT-CT accumulator to
                # the caller.  Native PairFM V uses this to combine all 16
                # source shards before paying one rescale per target carrier.
                output = total
            totals[index] = None
            outputs.append(output)
        return tuple(outputs)
    except Exception:
        for total in totals:
            release_if_supported(total)
        for output in outputs:
            release_if_supported(output)
        raise
    finally:
        for cipher, cache in zip(ciphers, rotated_inputs):
            for baby, rotated in cache.items():
                if baby and rotated is not cipher:
                    release_if_supported(rotated)


_SPARSE_PHASE_PROFILE = os.environ.get("HOLMES_SPARSE_PROFILE", "0") not in ("", "0", "false", "no")
_SPARSE_NONEXT_BABY = os.environ.get("HOLMES_SPARSE_NONEXT_BABY", "0") not in ("", "0", "false", "no")
#: ``HOLMES_SPARSE_GPU_MASKS=1``: scatter each encode batch's mask rows on the device
#: from their (row, slot, coefficient) nonzeros instead of materialising the dense
#: rows on the host and uploading them (value_delta: 69 x 30 x 32768 f64 = 0.54 GB
#: H2D per layer).  Same rows, same encode; only the transport changes.
_SPARSE_GPU_MASKS = os.environ.get("HOLMES_SPARSE_GPU_MASKS", "0") not in ("", "0", "false", "no")


class _PhaseTimer:
    """Per-phase device-synchronised timing for the sparse scheduler (HOLMES_SPARSE_PROFILE=1).

    Diagnostics only: every ``tick`` syncs the device, so the scheduler wall
    grows while it is enabled.  Disabled it is a no-op.
    """

    __slots__ = ("device", "enabled", "last", "totals")

    def __init__(self, device, enabled: bool) -> None:
        self.device = device
        self.enabled = bool(enabled)
        self.totals: dict[str, float] = {}
        self.last = time.perf_counter() if self.enabled else 0.0

    def tick(self, phase: str) -> None:
        if not self.enabled:
            return
        sync_device(self.device)
        now = time.perf_counter()
        self.totals[phase] = self.totals.get(phase, 0.0) + (now - self.last)
        self.last = now

    def export(self, profile: dict[str, float] | None) -> None:
        if not self.enabled or profile is None:
            return
        for phase, seconds in self.totals.items():
            profile[f"phase.{phase}"] = float(seconds)


def _cipher_batch_slice(batch, start: int, stop: int):
    """Return a borrowed contiguous batch view without a host round trip."""

    start = int(start)
    stop = int(stop)
    if not 0 <= start < stop <= int(batch.batch_size):
        raise IndexError(
            f"invalid cipher batch slice [{start}:{stop}] for size {batch.batch_size}."
        )
    return batch.cipher_like(
        [component[start:stop] for component in batch.cv],
        batch_size=stop - start,
    )


def _sparse_batch_masks(
    batch: SparseRouteBatch,
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    slots: int,
) -> tuple[np.ndarray, tuple[tuple[SparseRouteGroup, int, int], ...]]:
    """Materialize at most 32 sparse rows and their encoded-batch slices."""

    return materialize_sparse_batch_masks(
        batch, routes=routes, slots=int(slots)
    )


def _encode_sparse_batch_on_device(
    batch: SparseRouteBatch,
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    slots: int,
    level: int,
    crypto_context,
    is_ext: bool,
):
    """Scatter one batch's mask rows on the device and encode them.

    Equivalent to ``encode_transform_rows(_sparse_batch_masks(...)[0])``: the
    dense rows are built by one ``index_put`` from the route nonzeros, so the
    host never materialises or uploads them.  Returns ``(plaintext, metadata)``.
    """

    import easyfhe as torch

    row_indices, slot_indices, coefficients, metadata = sparse_batch_mask_indices(
        batch, routes=routes, slots=int(slots)
    )
    device = crypto_context.device
    rows = torch.zeros(
        (int(batch.encoded_rows), int(slots)), dtype=torch.complex128, device=device
    )
    if row_indices.size:
        rows[
            torch.from_numpy(row_indices).to(device),
            torch.from_numpy(slot_indices).to(device),
        ] = torch.from_numpy(np.ascontiguousarray(coefficients, dtype=np.complex128)).to(device)
    bundle = crypto_context.fhe.ConstantBundle(
        vectors={"rows": crypto_context.fhe.PackedRaw(rows)}, cache_mode="none"
    )
    cur_limbs = int(crypto_context.context.L) - int(level)
    plaintext = bundle.plaintext(
        "rows",
        state=fhe.CipherState(
            cur_limbs=cur_limbs,
            scale_degree=1,
            scaling_factor=crypto_context.context.scale_at(cur_limbs),
        ),
        slots=int(crypto_context.max_slots),
        context=crypto_context.context,
        is_ext=bool(is_ext),
        cache=False,
    )
    del rows
    return plaintext, metadata


def _fast_baby_rotation_batch(
    cipher,
    *,
    baby_offsets: tuple[int, ...],
    crypto_context,
    output_ext: bool = True,
):
    if not baby_offsets:
        raise ValueError("baby_offsets must not be empty")
    if len(baby_offsets) > 32:
        raise ValueError("the attention sparse scheduler caps baby batches at 32")
    if len(baby_offsets) == 1 and int(baby_offsets[0]) == 0:
        return crypto_context.fhe.pack_cipher_batch((cipher.deep_copy(),))
    return crypto_context.fhe.fast_rotate(
        cipher,
        baby_offsets,
        crypto_context.context,
        output_ext=bool(output_ext),
    )


def _select_baby_rotation_batch(
    rotation_batch,
    rotation_items: tuple[object, ...],
    *,
    baby_offsets: tuple[int, ...],
    baby_index: dict[int, int],
    crypto_context,
):
    """Select one giant group's babies, borrowing a view when contiguous."""

    indices = tuple(baby_index[int(offset)] for offset in baby_offsets)
    start = min(indices)
    if indices == tuple(range(start, start + len(indices))):
        return _cipher_batch_slice(
            rotation_batch, start, start + len(indices)
        )
    return crypto_context.fhe.pack_cipher_batch(
        rotation_items[index] for index in indices
    )


def _apply_sparse_routes_batched(
    ciphers: tuple[object, ...],
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    plan: SparseDiagonalPlan,
    slots: int,
    crypto_context,
    function_prefix: str,
    padded_plaintext_rows: int | None = None,
    batch_cap: int = 32,
    profile: dict[str, float] | None = None,
    rescale_outputs: bool = True,
) -> tuple[object, ...]:
    """GPU-batched sparse BSGS with one reusable baby basis per input.

    This is a physical scheduling optimization only.  Every nonzero route and
    requested padding row still contributes exactly one logical PT-CT
    multiplication; every nonzero baby and giant offset is unchanged.  The
    differences from :func:`_apply_sparse_routes` are:

    * up to 32 plaintext rows are transferred and encoded together;
    * all baby rotations of one input are produced by one hoisted fast batch;
    * one grouped GPU MAC replaces the Python multiply/add loop of a giant
      group, while preserving that group's route count.
    """

    if not ciphers:
        return ()
    required = (
        "fast_rotate",
        "grouped_pairwise_mac",
        "moddown_from_ext",
        "pack_cipher_batch",
        "unpack_cipher_batch",
    )
    missing = [name for name in required if not hasattr(crypto_context.fhe, name)]
    if missing:
        raise NotImplementedError(
            "batched sparse BSGS requires EasyFHE APIs: " + ", ".join(missing)
        )
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("one sparse diagonal transform requires one input level.")
    level = levels.pop()
    schedule = build_sparse_route_schedule(
        diagonal_offsets=plan.diagonal_offsets,
        baby_offsets=plan.baby_offsets,
        modulus=int(plan.modulus),
        slots=int(slots),
        batch_cap=int(batch_cap),
        padded_plaintext_rows=padded_plaintext_rows,
    )
    if schedule.padding_rows and 0 not in schedule.baby_offsets:
        raise ValueError("zero padding requires a zero-offset baby rotation")

    baby_batches: list[object] = []
    baby_items: list[tuple[object, ...]] = []
    totals: list[object | None] = [None] * len(ciphers)
    outputs: list[object] = []
    started = time.perf_counter()
    timer = _PhaseTimer(crypto_context.device, _SPARSE_PHASE_PROFILE)
    try:
        for cipher in ciphers:
            rotations = _fast_baby_rotation_batch(
                cipher,
                baby_offsets=schedule.baby_offsets,
                crypto_context=crypto_context,
                output_ext=not _SPARSE_NONEXT_BABY,
            )
            baby_batches.append(rotations)
            baby_items.append(
                tuple(crypto_context.fhe.unpack_cipher_batch(rotations))
            )
        timer.tick("baby")

        baby_index = {
            int(offset): index
            for index, offset in enumerate(schedule.baby_offsets)
        }
        fused = SparseFusedAccumulator.create(
            baby_batches, baby_index, crypto_context, schedule
        )
        for batch_index, batch in enumerate(schedule.batches):
            plaintexts = None
            if _SPARSE_GPU_MASKS:
                plaintexts, metadata = _encode_sparse_batch_on_device(
                    batch,
                    routes=routes,
                    slots=int(slots),
                    level=level,
                    crypto_context=crypto_context,
                    is_ext=bool(baby_batches[0].is_ext),
                )
                timer.tick("encode")
            else:
                masks, metadata = _sparse_batch_masks(
                    batch, routes=routes, slots=int(slots)
                )
                timer.tick("mask_host")
            try:
                if plaintexts is None:
                    plaintexts = encode_transform_rows(
                        masks,
                        name=f"{function_prefix}.batch{batch_index}",
                        level=level,
                        crypto_context=crypto_context,
                        is_ext=bool(baby_batches[0].is_ext),
                    )
                    timer.tick("encode")
                if fused is not None:
                    # HOLMES_SPARSE_FUSED: one gathered MAC + one batched ext giant step per encode batch
                    fused.apply_batch(batch_index, plaintexts, totals, _add_owned)
                    timer.tick("fused")
                    continue
                for cipher_index, rotations in enumerate(baby_items):
                    for group, start, stop in metadata:
                        selected = plain_group = product = contribution = None
                        try:
                            selected = _select_baby_rotation_batch(
                                baby_batches[cipher_index],
                                rotations,
                                baby_offsets=group.baby_offsets,
                                baby_index=baby_index,
                                crypto_context=crypto_context,
                            )
                            plain_group = _cipher_batch_slice(
                                plaintexts, start, stop
                            )
                            timer.tick("select")
                            product = crypto_context.fhe.grouped_pairwise_mac(
                                selected,
                                plain_group,
                                1,
                                crypto_context.context,
                            )
                            timer.tick("mac")
                            if product.is_ext:
                                contribution = crypto_context.fhe.moddown_from_ext(
                                    product, crypto_context.context
                                )
                            else:
                                contribution, product = product, None
                            timer.tick("moddown")
                            if int(group.giant_offset):
                                rotated = crypto_context.fhe.homo_rotate(
                                    contribution,
                                    int(group.giant_offset),
                                    crypto_context.context,
                                )
                                release_if_supported(contribution)
                                contribution = rotated
                            timer.tick("rotate")
                            totals[cipher_index] = _add_owned(
                                totals[cipher_index],
                                contribution,
                                crypto_context=crypto_context,
                            )
                            contribution = None
                            timer.tick("add")
                        finally:
                            release_if_supported(contribution)
                            release_if_supported(product)
                            release_if_supported(plain_group)
                            release_if_supported(selected)

                    if int(batch.padding_rows):
                        start = batch.route_rows
                        stop = batch.encoded_rows
                        selected = plain_padding = product = contribution = None
                        try:
                            selected = _select_baby_rotation_batch(
                                baby_batches[cipher_index],
                                rotations,
                                baby_offsets=(0,) * int(batch.padding_rows),
                                baby_index=baby_index,
                                crypto_context=crypto_context,
                            )
                            plain_padding = _cipher_batch_slice(
                                plaintexts, start, stop
                            )
                            product = crypto_context.fhe.grouped_pairwise_mac(
                                selected,
                                plain_padding,
                                1,
                                crypto_context.context,
                            )
                            if product.is_ext:
                                contribution = crypto_context.fhe.moddown_from_ext(
                                    product, crypto_context.context
                                )
                            else:
                                contribution, product = product, None
                            totals[cipher_index] = _add_owned(
                                totals[cipher_index],
                                contribution,
                                crypto_context=crypto_context,
                            )
                            contribution = None
                        finally:
                            release_if_supported(contribution)
                            release_if_supported(product)
                            release_if_supported(plain_padding)
                            release_if_supported(selected)
            finally:
                release_if_supported(plaintexts)

        if fused is not None:
            totals = list(fused.finish(totals, _add_owned))
            timer.tick("fused_moddown")
        for index, total in enumerate(totals):
            if total is None:
                raise RuntimeError(f"sparse diagonal output {index} is empty.")
            if bool(rescale_outputs):
                output = crypto_context.fhe.rescale(
                    total, crypto_context.context
                )
                release_if_supported(total)
            else:
                output = total
            totals[index] = None
            outputs.append(output)
        sync_device(crypto_context.device)
        timer.tick("rescale")
        timer.export(profile)
        if profile is not None:
            profile["batched"] = 1.0
            profile["fused"] = float(fused is not None)
            profile["wall_seconds"] = float(time.perf_counter() - started)
            profile["encode_batches"] = float(len(schedule.batches))
            profile["encoded_rows"] = float(schedule.encoded_rows)
            profile["route_rows"] = float(schedule.route_rows)
            profile["padding_rows"] = float(schedule.padding_rows)
            profile["baby_batches"] = float(len(ciphers))
            profile["baby_rotations"] = float(
                schedule.baby_rotations * len(ciphers)
            )
            profile["giant_groups"] = float(len(schedule.groups))
            profile["giant_rotations"] = float(
                schedule.giant_rotations * len(ciphers)
            )
            profile["logical_pt_ct"] = float(
                schedule.encoded_rows * len(ciphers)
            )
        return tuple(outputs)
    except Exception:
        for total in totals:
            release_if_supported(total)
        for output in outputs:
            release_if_supported(output)
        raise
    finally:
        for batch in baby_batches:
            release_if_supported(batch)


def _apply_sparse_routes_batched_shared(
    cipher,
    *,
    transforms: tuple[
        tuple[dict[int, np.ndarray | SparseMaskRoute], SparseDiagonalPlan, str],
        ...,
    ],
    slots: int,
    crypto_context,
    batch_cap: int = 32,
    profiles: list[dict[str, float]] | None = None,
) -> tuple[object, ...]:
    """Apply several sparse transforms with one shared hoisted baby basis.

    Every transform retains its own diagonal masks, giant groups, accumulator,
    and rescale.  Only the input baby rotations are shared.  The union is
    deliberately capped at 32 so one EasyFHE ``fast_rotate`` batch owns the
    complete basis and its temporary extended ciphertexts remain bounded.
    """

    transforms = tuple(transforms)
    if not transforms:
        return ()
    if profiles is not None and len(profiles) != len(transforms):
        raise ValueError("one shared sparse profile is required per transform.")
    levels = {int(crypto_context.level_for_cipher(cipher))}
    level = levels.pop()
    baby_offsets = tuple(
        sorted(
            {
                int(offset)
                for _, plan, _ in transforms
                for offset in plan.baby_offsets
            }
        )
    )
    if len(baby_offsets) > 32:
        raise ValueError(
            "shared sparse baby basis exceeds the EasyFHE batch cap: "
            f"{len(baby_offsets)} > 32."
        )

    rotation_batch = None
    rotation_items: tuple[object, ...] = ()
    outputs: list[object] = []
    pending_total = None
    shared_started = time.perf_counter()
    try:
        rotation_batch = _fast_baby_rotation_batch(
            cipher,
            baby_offsets=baby_offsets,
            crypto_context=crypto_context,
        )
        rotation_items = tuple(
            crypto_context.fhe.unpack_cipher_batch(rotation_batch)
        )
        baby_index = {
            int(offset): index for index, offset in enumerate(baby_offsets)
        }

        for transform_index, (routes, plan, function_prefix) in enumerate(
            transforms
        ):
            started = time.perf_counter()
            schedule = build_sparse_route_schedule(
                diagonal_offsets=plan.diagonal_offsets,
                baby_offsets=plan.baby_offsets,
                modulus=int(plan.modulus),
                slots=int(slots),
                batch_cap=int(batch_cap),
                padded_plaintext_rows=None,
            )
            total = None
            pending_total = total
            try:
                for batch_index, batch in enumerate(schedule.batches):
                    masks, metadata = _sparse_batch_masks(
                        batch, routes=routes, slots=int(slots)
                    )
                    plaintexts = None
                    try:
                        plaintexts = encode_transform_rows(
                            masks,
                            name=f"{function_prefix}.batch{batch_index}",
                            level=level,
                            crypto_context=crypto_context,
                        )
                        for group, start, stop in metadata:
                            selected = plain_group = product = contribution = None
                            try:
                                selected = _select_baby_rotation_batch(
                                    rotation_batch,
                                    rotation_items,
                                    baby_offsets=group.baby_offsets,
                                    baby_index=baby_index,
                                    crypto_context=crypto_context,
                                )
                                plain_group = _cipher_batch_slice(
                                    plaintexts, start, stop
                                )
                                product = crypto_context.fhe.grouped_pairwise_mac(
                                    selected,
                                    plain_group,
                                    1,
                                    crypto_context.context,
                                )
                                contribution = (
                                    crypto_context.fhe.moddown_from_ext(
                                        product, crypto_context.context
                                    )
                                    if product.is_ext
                                    else product.deep_copy()
                                )
                                if int(group.giant_offset):
                                    rotated = crypto_context.fhe.homo_rotate(
                                        contribution,
                                        int(group.giant_offset),
                                        crypto_context.context,
                                    )
                                    release_if_supported(contribution)
                                    contribution = rotated
                                total = _add_owned(
                                    total,
                                    contribution,
                                    crypto_context=crypto_context,
                                )
                                pending_total = total
                                contribution = None
                            finally:
                                release_if_supported(contribution)
                                release_if_supported(product)
                                release_if_supported(plain_group)
                                release_if_supported(selected)
                    finally:
                        release_if_supported(plaintexts)
                if total is None:
                    raise RuntimeError("shared sparse diagonal output is empty.")
                output = crypto_context.fhe.rescale(
                    total, crypto_context.context
                )
                release_if_supported(total)
                total = pending_total = None
                outputs.append(output)
            finally:
                release_if_supported(total)
                pending_total = None

            if profiles is not None:
                profile = profiles[transform_index]
                profile["wall_seconds"] = float(time.perf_counter() - started)
                profile["route_rows"] = float(schedule.route_rows)
                profile["baby_rotations"] = float(schedule.baby_rotations)
                profile["giant_rotations"] = float(schedule.giant_rotations)
                profile["shared_basis_size"] = float(len(baby_offsets))
                profile["shared_group_size"] = float(len(transforms))

        sync_device(crypto_context.device)
        shared_wall = float(time.perf_counter() - shared_started)
        if profiles is not None:
            for profile in profiles:
                profile["shared_group_wall_seconds"] = shared_wall
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        release_if_supported(pending_total)
        for output in outputs:
            release_if_supported(output)
        release_if_supported(rotation_batch)


def _apply_sparse_routes_selected(
    ciphers: tuple[object, ...],
    *,
    routes: dict[int, np.ndarray | SparseMaskRoute],
    plan: SparseDiagonalPlan,
    slots: int,
    crypto_context,
    function_prefix: str,
    padded_plaintext_rows: int | None,
    use_batched: bool,
    batch_cap: int,
    fallback_on_error: bool,
    profile: dict[str, float] | None = None,
    rescale_outputs: bool = True,
) -> tuple[object, ...]:
    if not bool(use_batched):
        if profile is not None:
            profile["batched"] = 0.0
        return _apply_sparse_routes(
            ciphers,
            routes=routes,
            plan=plan,
            slots=int(slots),
            crypto_context=crypto_context,
            function_prefix=function_prefix,
            padded_plaintext_rows=padded_plaintext_rows,
            rescale_outputs=bool(rescale_outputs),
        )
    try:
        return _apply_sparse_routes_batched(
            ciphers,
            routes=routes,
            plan=plan,
            slots=int(slots),
            crypto_context=crypto_context,
            function_prefix=function_prefix,
            padded_plaintext_rows=padded_plaintext_rows,
            batch_cap=int(batch_cap),
            profile=profile,
            rescale_outputs=bool(rescale_outputs),
        )
    except Exception as exc:
        if not bool(fallback_on_error):
            raise
        warnings.warn(
            f"{function_prefix}: batched sparse BSGS failed; using the "
            f"correctness-first scheduler ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        if profile is not None:
            profile.clear()
            profile["batched"] = 0.0
            profile["fallbacks"] = 1.0
        return _apply_sparse_routes(
            ciphers,
            routes=routes,
            plan=plan,
            slots=int(slots),
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.fallback",
            padded_plaintext_rows=padded_plaintext_rows,
            rescale_outputs=bool(rescale_outputs),
        )


def _apply_sparse_routes_shared_selected(
    ciphers: tuple[object, ...],
    *,
    transforms: tuple[
        tuple[dict[int, np.ndarray | SparseMaskRoute], SparseDiagonalPlan, str],
        ...,
    ],
    slots: int,
    crypto_context,
    use_batched: bool,
    batch_cap: int,
    fallback_on_error: bool,
    profiles: list[dict[str, float]] | None = None,
) -> tuple[tuple[object, ...], ...]:
    """Apply several sparse transforms while sharing each input's baby basis.

    ``_apply_sparse_routes_batched_shared`` shares rotations between
    transforms of one input ciphertext.  Prefill cache emission has two such
    transforms (the compact ``V_C`` snapshot and Delta-V) for every
    ``v_compact`` ciphertext.  This wrapper applies that optimization over the
    whole input tuple and retains the existing independent scheduler as a
    correctness fallback.

    The transform outputs are returned in the same order as ``transforms``;
    each nested tuple follows the input ``ciphers`` order.  Inputs are
    borrowed, while every returned ciphertext is owned by the caller.
    """

    transforms = tuple(transforms)
    if profiles is not None and len(profiles) != len(transforms):
        raise ValueError("one sparse profile is required per shared transform.")
    if not transforms:
        return ()

    # Keep the old scheduler exactly intact when shared batching is disabled.
    if not bool(use_batched):
        outputs: list[tuple[object, ...]] = []
        try:
            for index, (routes, plan, function_prefix) in enumerate(transforms):
                outputs.append(
                    _apply_sparse_routes_selected(
                        ciphers,
                        routes=routes,
                        plan=plan,
                        slots=int(slots),
                        crypto_context=crypto_context,
                        function_prefix=function_prefix,
                        padded_plaintext_rows=None,
                        use_batched=False,
                        batch_cap=int(batch_cap),
                        fallback_on_error=bool(fallback_on_error),
                        profile=None if profiles is None else profiles[index],
                    )
                )
            return tuple(outputs)
        except BaseException:
            for group in outputs:
                for cipher in group:
                    release_if_supported(cipher)
            raise

    outputs_by_transform: list[list[object]] = [
        [] for _ in transforms
    ]
    try:
        for cipher in ciphers:
            child_profiles = [{} for _ in transforms]
            shared_outputs = _apply_sparse_routes_batched_shared(
                cipher,
                transforms=transforms,
                slots=int(slots),
                crypto_context=crypto_context,
                batch_cap=int(batch_cap),
                profiles=child_profiles,
            )
            if len(shared_outputs) != len(transforms):
                for output_group in shared_outputs:
                    for output in output_group:
                        release_if_supported(output)
                raise RuntimeError(
                    "shared sparse scheduler returned an unexpected transform count: "
                    f"{len(shared_outputs)} != {len(transforms)}"
                )
            for index, output in enumerate(shared_outputs):
                outputs_by_transform[index].append(output)
                if profiles is not None:
                    for name, value in child_profiles[index].items():
                        profiles[index][name] = float(
                            profiles[index].get(name, 0.0) + float(value)
                        )
        if profiles is not None:
            for profile in profiles:
                profile["batched"] = 1.0
                profile["shared_fallbacks"] = 0.0
                profile["shared_group_count"] = float(len(ciphers))
        return tuple(tuple(group) for group in outputs_by_transform)
    except BaseException as error:
        for group in outputs_by_transform:
            for cipher in group:
                release_if_supported(cipher)
        if not bool(fallback_on_error):
            raise
        names = ", ".join(str(item[2]) for item in transforms)
        warnings.warn(
            "prefill sparse shared BSGS failed; using independent transforms "
            f"for {names} ({type(error).__name__}: {error})",
            RuntimeWarning,
            stacklevel=2,
        )
        if profiles is not None:
            for profile in profiles:
                profile.clear()
                profile["batched"] = 0.0
                profile["fallbacks"] = 1.0
                profile["shared_fallbacks"] = 1.0
                profile["shared_group_count"] = 0.0
        fallback_outputs: list[tuple[object, ...]] = []
        try:
            for index, (routes, plan, function_prefix) in enumerate(transforms):
                fallback_outputs.append(
                    _apply_sparse_routes_selected(
                        ciphers,
                        routes=routes,
                        plan=plan,
                        slots=int(slots),
                        crypto_context=crypto_context,
                        function_prefix=function_prefix,
                        padded_plaintext_rows=None,
                        use_batched=True,
                        batch_cap=int(batch_cap),
                        fallback_on_error=bool(fallback_on_error),
                        profile=None if profiles is None else profiles[index],
                    )
                )
            if profiles is not None:
                for profile in profiles:
                    profile["shared_fallbacks"] = 1.0
                    profile["shared_group_count"] = 0.0
            return tuple(fallback_outputs)
        except BaseException:
            for group in fallback_outputs:
                for cipher in group:
                    release_if_supported(cipher)
            raise


def _merge_scheduler_profile(
    profile: dict[str, float],
    *,
    label: str,
    stage_seconds: dict[str, float],
    scheduler_stats: dict[str, float],
) -> None:
    """Fold one sparse-scheduler profile into the stage and stats reports.

    ``wall_seconds`` is a duration and belongs with the stage timings; every
    other key is a scheduler counter, and the two aggregate counters
    (``fallbacks``, ``batched_stage_count``) accumulate across stages.
    """

    for name, value in profile.items():
        if name == "wall_seconds":
            stage_seconds[f"{label}.scheduler"] = float(value)
            continue
        scheduler_stats[f"{label}.{name}"] = float(value)
        if name == "fallbacks":
            scheduler_stats["fallbacks"] += float(value)
        elif name == "batched":
            scheduler_stats["batched_stage_count"] += float(value)


def _expand_c_feature_half(
    cipher,
    *,
    upper_sign: int,
    attention_layout: AttentionPairLayout,
    crypto_context,
):
    conjugated = transformed = shifted = None
    try:
        conjugated = crypto_context.fhe.homo_rotate(
            cipher, int(crypto_context.context.M) - 1, crypto_context.context
        )
        transformed = fhe.homo_mul_i(
            conjugated,
            crypto_context.context,
            negative=int(upper_sign) < 0,
        )
        shifted = crypto_context.fhe.homo_rotate(
            transformed,
            -int(attention_layout.payload_slots) // 2,
            crypto_context.context,
        )
        return crypto_context.fhe.homo_add(cipher, shifted, crypto_context.context)
    finally:
        release_if_supported(conjugated)
        release_if_supported(transformed)
        release_if_supported(shifted)


def _repeat_outputs(ciphers: tuple[object, ...], *, layout, crypto_context):
    outputs: list[object] = []
    try:
        for cipher in ciphers:
            outputs.append(
                repeat_payload_cipher(
                    cipher, layout=layout, crypto_context=crypto_context
                )
            )
        return tuple(outputs)
    except Exception:
        for output in outputs:
            release_if_supported(output)
        raise


def prepare_attention_inputs_fhe(
    carrier_ciphers: tuple[object, ...] | list[object],
    *,
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    cos: np.ndarray | None = None,
    sin: np.ndarray | None = None,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    verify: bool = False,
    expected_query: np.ndarray | None = None,
    expected_key: np.ndarray | None = None,
    expected_value: np.ndarray | None = None,
    function_prefix: str = "model.layers.0.self_attn.complex_token_attention",
    direct_pairfold_v: bool = False,
    pairfold_occupied: int = 128,
) -> ComplexTokenAttentionResult:
    """Convert heterogeneous linear carriers directly to QK-ready layouts.

    This avoids materializing another persistent layout. Q/K use a
    lane/phase stage, a target gather fused with the first source-feature axis
    swap, and a second 31-diagonal axis swap.  K then fuses upper-half
    reconstruction into its Delta shear.  V uses the lane/phase
    stage, a tiny carrier gather, and one direct compact-to-Delta transform.
    Relative Q/K/V output depths are 3/4/3.

    ``operator_config.emit_cache_c`` is opt-in.  When set, the adapter also
    materializes folded ``K_C`` / compact ``V_C`` snapshots for the Decode
    cache without changing the Prefill Delta path.  Default stays off so
    Prefill cost is unchanged.

    Sparse-route scheduling is read from ``operator_config``: the release
    default batches mask encoding and reuses hoisted baby rotations
    (``batched_sparse_input_bsgs``, capped by ``sparse_input_batch_cap``).
    ``sparse_input_fallback`` keeps the correctness-first scheduler as the
    recovery path when the batched one raises; set it False to let backend
    errors surface instead.
    """

    model_config = Llama3CKKSConfig() if model_config is None else model_config
    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    # The stage below reads these as plain locals; unpack the schedule once.
    sparse_max_baby_offsets = int(operator_config.sparse_max_baby_offsets)
    qk_score_level_drop = int(operator_config.qk_score_level_drop)
    batched_sparse_bsgs = bool(operator_config.batched_sparse_input_bsgs)
    sparse_batch_cap = int(operator_config.sparse_input_batch_cap)
    sparse_scheduler_fallback = bool(operator_config.sparse_input_fallback)
    emit_cache_c = bool(operator_config.emit_cache_c)
    direct_pairfold_v = bool(direct_pairfold_v)
    staged_pairfold_v = direct_pairfold_v and _value_delta_pairfold_staged_enabled()
    if direct_pairfold_v and not emit_cache_c:
        raise ValueError("direct PairFold V requires cache emission.")
    # RoPE phases are derived from the same config when the caller does not
    # supply them, so this has to happen before their shape is checked.
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must either both be provided or both omitted.")
    if cos is None:
        cos, sin = rope_cos_sin(
            seq_len=int(attention_layout.seq_len),
            head_dim=int(attention_layout.head_dim),
            theta=float(operator_config.rope_theta),
            start_pos=int(operator_config.rope_start_pos),
        )

    carrier_ciphers = tuple(carrier_ciphers)
    params = _layout_parameters(carrier_layout, attention_layout)
    if len(carrier_ciphers) != params["source_count"]:
        raise ValueError(
            f"expected {params['source_count']} linear carrier ciphers, got "
            f"{len(carrier_ciphers)}."
        )
    if not carrier_ciphers:
        raise ValueError("carrier_ciphers must not be empty.")
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in carrier_ciphers}
    if len(levels) != 1:
        raise ValueError("all linear carrier ciphers must enter at one level.")
    cos = np.asarray(cos, dtype=np.float64)
    sin = np.asarray(sin, dtype=np.float64)
    expected_phase_shape = (params["seq_len"], params["head_dim"])
    if cos.shape != expected_phase_shape or sin.shape != expected_phase_shape:
        raise ValueError(f"cos/sin must have shape {expected_phase_shape}.")
    value_routes = _value_delta_routes(params)
    value_plan = choose_sparse_diagonal_plan(
        tuple(value_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    lower_routes = _key_shear_routes(params, upper=False)
    upper_routes = _key_shear_routes(params, upper=True)
    lower_plan = choose_sparse_diagonal_plan(
        tuple(lower_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    upper_plan = choose_sparse_diagonal_plan(
        tuple(upper_routes),
        slots=params["slots"],
        max_baby_offsets=int(sparse_max_baby_offsets),
    )
    axis2_masks, axis2_stride = _qk_axis_masks(params, stage=2)
    wall_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    scheduler_stats: dict[str, float] = {
        "enabled": float(batched_sparse_bsgs),
        "batch_cap": float(sparse_batch_cap),
        "fallback_allowed": float(sparse_scheduler_fallback),
        "fallbacks": 0.0,
        # Two key-shear calls, Delta-V, and any cache-emission transforms.
        "stage_count": float(
            3
            + int(emit_cache_c and not direct_pairfold_v)
            + 2 * int(direct_pairfold_v)
            + int(staged_pairfold_v)
        ),
        "batched_stage_count": 0.0,
    }
    lane_outputs: tuple[object, ...] = ()
    qk_axis1: tuple[object, ...] = ()
    q_compact: tuple[object, ...] = ()
    k_compact: tuple[object, ...] = ()
    v_compact: tuple[object, ...] = ()
    qk_low: tuple[object, ...] = ()
    q_full: tuple[object, ...] = ()
    q_outputs: tuple[object, ...] = ()
    k_lower: tuple[object, ...] = ()
    k_upper_sources: tuple[object, ...] = ()
    k_upper: tuple[object, ...] = ()
    k_combined: tuple[object, ...] = ()
    k_outputs: tuple[object, ...] = ()
    v_axis1_complex: tuple[object, ...] = ()
    v_complex: tuple[object, ...] = ()
    v_split: tuple[object, ...] = ()
    v_outputs: tuple[object, ...] = ()
    k_c_outputs: tuple[object, ...] = ()
    v_c_outputs: tuple[object, ...] = ()
    pairfold_v_outputs: tuple[object, ...] = ()
    v_c_complex: tuple[object, ...] = ()
    success = False
    try:
        start = time.perf_counter()
        lane_outputs = _apply_lane_phase(
            carrier_ciphers,
            params=params,
            cos=cos,
            sin=sin,
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.lane_phase",
        )
        sync_device(crypto_context.device)
        stage_seconds["lane_phase"] = time.perf_counter() - start

        start = time.perf_counter()
        qk_axis1 = _apply_qk_fused_gather_axis1(
            lane_outputs,
            params=params,
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.qk_gather_axis1",
        )
        sync_device(crypto_context.device)
        stage_seconds["qk_gather_axis1"] = time.perf_counter() - start

        # The attention entry budget is sized for the VALUE stream (which is
        # never re-bootstrapped and must reach PairFM pairing at six limbs).
        # The Q/K score stream is reset by the pre-exp softmax bootstrap and
        # only needs the tight schedule (K adapter 4 + QK 2 + preprocessing 1
        # + bootstrap floor 2), so any extra entry limbs can be shed here —
        # the earliest Q/K-only point — making the axis2 transform, expand,
        # key shear, and QK all proportionally cheaper.
        # LLAMA_FHE_QK_LEVEL_DROP overrides the scheduled value for sweeps.
        qk_level_drop = int(
            os.environ.get(
                "LLAMA_FHE_QK_LEVEL_DROP", str(int(qk_score_level_drop))
            )
        )
        if qk_level_drop > 0:
            start = time.perf_counter()
            dropped_qk: list[object] = []
            try:
                for cipher in qk_axis1:
                    target_limbs = (
                        int(cipher.state.cur_limbs) - qk_level_drop
                    )
                    if target_limbs <= 1:
                        raise ValueError(
                            "LLAMA_FHE_QK_LEVEL_DROP leaves too few limbs: "
                            f"cur_limbs={cipher.state.cur_limbs}, "
                            f"drop={qk_level_drop}."
                        )
                    target_state = cipher.state.replace(
                        cur_limbs=target_limbs,
                        scale_degree=1,
                        scaling_factor=None,
                    )
                    dropped_qk.append(
                        fhe.align_to(
                            cipher, target_state, crypto_context.context
                        )
                    )
            except Exception:
                for cipher in dropped_qk:
                    release_if_supported(cipher)
                raise
            for cipher in qk_axis1:
                release_if_supported(cipher)
            qk_axis1 = tuple(dropped_qk)
            sync_device(crypto_context.device)
            stage_seconds["qk_score_level_drop"] = (
                time.perf_counter() - start
            )

        start = time.perf_counter()
        v_compact = _apply_value_block_gather(
            lane_outputs,
            params=params,
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.value_gather",
        )
        for cipher in lane_outputs:
            release_if_supported(cipher)
        lane_outputs = ()
        sync_device(crypto_context.device)
        stage_seconds["value_gather"] = time.perf_counter() - start

        start = time.perf_counter()
        axis_profile: dict[str, float] = {}
        qk_low = apply_transform_by_level(
            qk_axis1,
            masks=axis2_masks,
            stride=axis2_stride,
            baby_steps=16,
            crypto_context=crypto_context,
            name=f"{function_prefix}.qk_axis2",
            profile=axis_profile,
        )
        for cipher in qk_axis1:
            release_if_supported(cipher)
        qk_axis1 = ()
        sync_device(crypto_context.device)
        stage_seconds["qk_axis2"] = time.perf_counter() - start
        for name, seconds in axis_profile.items():
            stage_seconds[f"qk_axis2.{name}"] = float(seconds)

        start = time.perf_counter()
        q_count = params["query_pairs"]
        q_expanded: list[object] = []
        try:
            for cipher in qk_low[:q_count]:
                q_expanded.append(
                    _expand_c_feature_half(
                        cipher,
                        upper_sign=1,
                        attention_layout=attention_layout,
                        crypto_context=crypto_context,
                    )
                )
            q_full = tuple(q_expanded)
            q_expanded = []
        finally:
            for cipher in q_expanded:
                release_if_supported(cipher)
        q_outputs = _repeat_outputs(
            q_full, layout=attention_layout, crypto_context=crypto_context
        )
        q_full = ()

        k_sources = qk_low[q_count:]
        if emit_cache_c:
            cache_k_start = time.perf_counter()
            k_c_expanded: list[object] = []
            try:
                for cipher in k_sources:
                    k_c_expanded.append(
                        _expand_c_feature_half(
                            cipher,
                            upper_sign=1,
                            attention_layout=attention_layout,
                            crypto_context=crypto_context,
                        )
                    )
                k_c_full = tuple(k_c_expanded)
                k_c_expanded = []
            finally:
                for cipher in k_c_expanded:
                    release_if_supported(cipher)
            try:
                k_c_outputs = _repeat_outputs(
                    k_c_full, layout=attention_layout, crypto_context=crypto_context
                )
            finally:
                for cipher in k_c_full:
                    release_if_supported(cipher)
            stage_seconds["key_c_expand"] = time.perf_counter() - cache_k_start
        lower_profile: dict[str, float] = {}
        k_lower = _apply_sparse_routes_selected(
            k_sources,
            routes=lower_routes,
            plan=lower_plan,
            slots=params["slots"],
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.key_shear_lower",
            padded_plaintext_rows=128,
            use_batched=batched_sparse_bsgs,
            batch_cap=sparse_batch_cap,
            fallback_on_error=sparse_scheduler_fallback,
            profile=lower_profile,
        )
        transformed_upper: list[object] = []
        try:
            for cipher in k_sources:
                conjugated = crypto_context.fhe.homo_rotate(
                    cipher,
                    int(crypto_context.context.M) - 1,
                    crypto_context.context,
                )
                try:
                    transformed_upper.append(
                        fhe.homo_mul_i(
                            conjugated,
                            crypto_context.context,
                            negative=True,
                        )
                    )
                finally:
                    release_if_supported(conjugated)
            k_upper_sources = tuple(transformed_upper)
            transformed_upper = []
        finally:
            for cipher in transformed_upper:
                release_if_supported(cipher)
        upper_profile: dict[str, float] = {}
        k_upper = _apply_sparse_routes_selected(
            k_upper_sources,
            routes=upper_routes,
            plan=upper_plan,
            slots=params["slots"],
            crypto_context=crypto_context,
            function_prefix=f"{function_prefix}.key_shear_upper",
            padded_plaintext_rows=128,
            use_batched=batched_sparse_bsgs,
            batch_cap=sparse_batch_cap,
            fallback_on_error=sparse_scheduler_fallback,
            profile=upper_profile,
        )
        for cipher in k_upper_sources:
            release_if_supported(cipher)
        k_upper_sources = ()
        combined: list[object] = []
        try:
            for lower, upper in zip(k_lower, k_upper, strict=True):
                combined.append(
                    crypto_context.fhe.homo_add(
                        lower, upper, crypto_context.context
                    )
                )
            k_combined = tuple(combined)
            combined = []
        finally:
            for cipher in combined:
                release_if_supported(cipher)
            for cipher in k_lower + k_upper:
                release_if_supported(cipher)
            k_lower = k_upper = ()
        k_outputs = _repeat_outputs(
            k_combined, layout=attention_layout, crypto_context=crypto_context
        )
        k_combined = ()
        for cipher in qk_low:
            release_if_supported(cipher)
        qk_low = ()
        sync_device(crypto_context.device)
        stage_seconds["expand_and_key_shear"] = time.perf_counter() - start
        for label, sparse_profile in (
            ("key_shear_lower", lower_profile),
            ("key_shear_upper", upper_profile),
        ):
            _merge_scheduler_profile(
                sparse_profile,
                label=label,
                stage_seconds=stage_seconds,
                scheduler_stats=scheduler_stats,
            )

        start = time.perf_counter()
        shared_value_routes = False
        value_c_profile: dict[str, float] = {}
        value_profile: dict[str, float] = {}
        if emit_cache_c and not direct_pairfold_v:
            value_c_routes = _value_c_routes(params)
            value_c_plan = choose_sparse_diagonal_plan(
                tuple(value_c_routes),
                slots=params["slots"],
                max_baby_offsets=int(sparse_max_baby_offsets),
            )
            if batched_sparse_bsgs:
                shared_value_routes = True
                shared_profiles = [value_c_profile, value_profile]
                shared_outputs = _apply_sparse_routes_shared_selected(
                    v_compact,
                    transforms=(
                        (
                            value_c_routes,
                            value_c_plan,
                            f"{function_prefix}.value_c",
                        ),
                        (
                            value_routes,
                            value_plan,
                            f"{function_prefix}.value_delta",
                        ),
                    ),
                    slots=params["slots"],
                    crypto_context=crypto_context,
                    use_batched=True,
                    batch_cap=sparse_batch_cap,
                    fallback_on_error=sparse_scheduler_fallback,
                    profiles=shared_profiles,
                )
                v_c_complex, v_complex = shared_outputs
            else:
                v_c_complex = _apply_sparse_routes_selected(
                    v_compact,
                    routes=value_c_routes,
                    plan=value_c_plan,
                    slots=params["slots"],
                    crypto_context=crypto_context,
                    function_prefix=f"{function_prefix}.value_c",
                    padded_plaintext_rows=None,
                    use_batched=False,
                    batch_cap=sparse_batch_cap,
                    fallback_on_error=sparse_scheduler_fallback,
                    profile=value_c_profile,
                )
            value_c_split_started = time.perf_counter()
            value_c_split = split_complex_ciphers_twice(
                v_c_complex, crypto_context=crypto_context
            )
            for cipher in v_c_complex:
                release_if_supported(cipher)
            v_c_complex = ()
            value_c_real = value_c_split[0::2]
            value_c_imag = value_c_split[1::2]
            ordered_c: list[object] = []
            for real, imaginary in zip(value_c_real, value_c_imag, strict=True):
                ordered_c.extend((real, imaginary))
            v_c_split = tuple(ordered_c)
            v_c_outputs = _repeat_outputs(
                v_c_split, layout=attention_layout, crypto_context=crypto_context
            )
            for cipher in v_c_split:
                release_if_supported(cipher)
            if shared_value_routes:
                # The shared scheduler reports each transform separately.  Keep
                # the stage breakdown additive even though both transforms were
                # issued by one call.
                stage_seconds["value_c_and_split"] = float(
                    value_c_profile.get("wall_seconds", 0.0)
                    + time.perf_counter() - value_c_split_started
                )
            else:
                stage_seconds["value_c_and_split"] = time.perf_counter() - start
            _merge_scheduler_profile(
                value_c_profile,
                label="value_c",
                stage_seconds=stage_seconds,
                scheduler_stats=scheduler_stats,
            )
            start = time.perf_counter()
        if not shared_value_routes:
            value_profile = {} if not emit_cache_c else value_profile
            v_complex = _apply_sparse_routes_selected(
                v_compact,
                routes=value_routes,
                plan=value_plan,
                slots=params["slots"],
                crypto_context=crypto_context,
                function_prefix=f"{function_prefix}.value_delta",
                padded_plaintext_rows=None,
                use_batched=batched_sparse_bsgs,
                batch_cap=sparse_batch_cap,
                fallback_on_error=sparse_scheduler_fallback,
                profile=value_profile,
            )
        for cipher in v_compact:
            release_if_supported(cipher)
        v_compact = ()
        value_split_started = time.perf_counter()
        value_split = split_complex_ciphers_twice(
            v_complex, crypto_context=crypto_context
        )
        value_real = value_split[0::2]
        value_imag = value_split[1::2]
        ordered: list[object] = []
        for real, imaginary in zip(value_real, value_imag, strict=True):
            ordered.extend((real, imaginary))
        v_split = tuple(ordered)
        v_outputs = _repeat_outputs(
            v_split, layout=attention_layout, crypto_context=crypto_context
        )
        v_split = ()
        for cipher in v_complex:
            release_if_supported(cipher)
        v_complex = ()
        sync_device(crypto_context.device)
        value_delta_label = "value_delta"
        if shared_value_routes:
            stage_seconds[f"{value_delta_label}_and_split"] = float(
                value_profile.get("wall_seconds", 0.0)
                + time.perf_counter() - value_split_started
            )
        else:
            stage_seconds[f"{value_delta_label}_and_split"] = time.perf_counter() - start
        _merge_scheduler_profile(
            value_profile,
            label=value_delta_label,
            stage_seconds=stage_seconds,
            scheduler_stats=scheduler_stats,
        )
        if direct_pairfold_v:
            pairfold_started = time.perf_counter()
            staged_inputs: tuple[object, ...] = ()
            transformed: list[tuple[object, ...]] = []
            try:
                pairfold_inputs = v_outputs
                if staged_pairfold_v:
                    low_routes = _value_delta_pairfold_low_routes()
                    low_plan = choose_sparse_diagonal_plan(
                        tuple(low_routes),
                        slots=params["slots"],
                        max_baby_offsets=sparse_max_baby_offsets,
                    )
                    low_profile: dict[str, float] = {}
                    staged_inputs = _apply_sparse_routes_selected(
                        v_outputs,
                        routes=low_routes,
                        plan=low_plan,
                        slots=params["slots"],
                        crypto_context=crypto_context,
                        function_prefix=f"{function_prefix}.value_pairfold.low",
                        padded_plaintext_rows=None,
                        use_batched=batched_sparse_bsgs,
                        batch_cap=sparse_batch_cap,
                        fallback_on_error=sparse_scheduler_fallback,
                        profile=low_profile,
                    )
                    pairfold_inputs = staged_inputs
                    _merge_scheduler_profile(
                        low_profile,
                        label="value_pairfold_low",
                        stage_seconds=stage_seconds,
                        scheduler_stats=scheduler_stats,
                    )
                for pair in (0, 1):
                    pair_routes = (
                        _value_delta_pairfold_high_routes(pair, int(pairfold_occupied))
                        if staged_pairfold_v
                        else _value_delta_pairfold_routes(pair, int(pairfold_occupied))
                    )
                    pair_plan = choose_sparse_diagonal_plan(
                        tuple(pair_routes),
                        slots=params["slots"],
                        max_baby_offsets=sparse_max_baby_offsets,
                    )
                    pair_profile: dict[str, float] = {}
                    transformed.append(
                        _apply_sparse_routes_selected(
                            tuple(pairfold_inputs[group * 2 + pair] for group in (0, 1)),
                            routes=pair_routes,
                            plan=pair_plan,
                            slots=params["slots"],
                            crypto_context=crypto_context,
                            function_prefix=f"{function_prefix}.value_pairfold.pair{pair}",
                            padded_plaintext_rows=None,
                            use_batched=batched_sparse_bsgs,
                            batch_cap=sparse_batch_cap,
                            fallback_on_error=sparse_scheduler_fallback,
                            profile=pair_profile,
                        )
                    )
                    _merge_scheduler_profile(
                        pair_profile,
                        label=f"value_pairfold_pair{pair}",
                        stage_seconds=stage_seconds,
                        scheduler_stats=scheduler_stats,
                    )
                combined_pairfold: list[object] = []
                try:
                    for group in (0, 1):
                        combined_pairfold.append(
                            crypto_context.fhe.homo_add(
                                transformed[0][group],
                                transformed[1][group],
                                crypto_context.context,
                            )
                        )
                    pairfold_v_outputs = tuple(combined_pairfold)
                    combined_pairfold = []
                finally:
                    for cipher in combined_pairfold:
                        release_if_supported(cipher)
            finally:
                for cipher in staged_inputs:
                    release_if_supported(cipher)
                for group in transformed:
                    for cipher in group:
                        release_if_supported(cipher)
            sync_device(crypto_context.device)
            stage_seconds["value_delta_to_pairfold"] = (
                time.perf_counter() - pairfold_started
            )

        max_abs_diff = None
        if verify:
            if expected_query is None or expected_key is None or expected_value is None:
                raise ValueError(
                    "verify=True requires expected_query, expected_key, and expected_value."
                )
            query_rope = apply_rope_heads(
                expected_query,
                cos,
                sin,
                num_heads=int(attention_layout.query_heads),
                head_dim=int(attention_layout.head_dim),
            )
            key_rope = apply_rope_heads(
                expected_key,
                cos,
                sin,
                num_heads=int(attention_layout.key_value_heads),
                head_dim=int(attention_layout.head_dim),
            )
            actual_q = decrypt_complex_rows(q_outputs, crypto_context=crypto_context)
            actual_k = decrypt_complex_rows(k_outputs, crypto_context=crypto_context)
            actual_v = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher))
                    for cipher in v_outputs
                ]
            )
            wanted_q = attention_layout.pack_query_c_feature_folded(query_rope)
            wanted_k = attention_layout.pack_key_delta_feature_folded(key_rope)
            wanted_v = attention_layout.pack_value_delta(expected_value)
            max_abs_diff = {
                "query_folded": float(np.max(np.abs(actual_q - wanted_q))),
                "key_folded": float(np.max(np.abs(actual_k - wanted_k))),
                "value": float(np.max(np.abs(actual_v - wanted_v))),
            }
            if emit_cache_c:
                actual_k_c = decrypt_complex_rows(
                    k_c_outputs, crypto_context=crypto_context
                )
                wanted_k_c = attention_layout.pack_key_c_feature_folded(key_rope)
                max_abs_diff["key_c_folded"] = float(
                    np.max(np.abs(actual_k_c - wanted_k_c))
                )
                if direct_pairfold_v:
                    from llama3fhe.layouts.pairfold128 import PairFold128Layout

                    actual_pairfold_v = decrypt_complex_rows(
                        pairfold_v_outputs, crypto_context=crypto_context
                    )
                    wanted_pairfold_v = PairFold128Layout().pack_value(expected_value)
                    max_abs_diff["pairfold_value"] = float(
                        np.max(np.abs(actual_pairfold_v - wanted_pairfold_v))
                    )
                else:
                    actual_v_c = np.stack(
                        [
                            np.asarray(crypto_context.decrypt(cipher))
                            for cipher in v_c_outputs
                        ]
                    )
                    wanted_v_c = attention_layout.pack_value_c(expected_value)
                    max_abs_diff["value_c"] = float(
                        np.max(np.abs(actual_v_c - wanted_v_c))
                    )

        success = True
        return ComplexTokenAttentionResult(
            query_ciphers=q_outputs,
            key_ciphers=k_outputs,
            value_ciphers=v_outputs,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={name: float(seconds) for name, seconds in stage_seconds.items()},
            scheduler_stats={
                name: float(value) for name, value in scheduler_stats.items()
            },
            max_abs_diff=max_abs_diff,
            key_c_ciphers=k_c_outputs,
            value_c_ciphers=v_c_outputs,
            pairfold_value_ciphers=pairfold_v_outputs,
        )
    finally:
        for group in (
            lane_outputs,
            qk_axis1,
            q_compact,
            k_compact,
            v_compact,
            qk_low,
            q_full,
            k_lower,
            k_upper_sources,
            k_upper,
            k_combined,
            v_axis1_complex,
            v_complex,
            v_split,
            v_c_complex,
        ):
            for cipher in group:
                release_if_supported(cipher)
        if not success:
            for cipher in q_outputs + k_outputs + v_outputs + k_c_outputs + v_c_outputs + pairfold_v_outputs:
                release_if_supported(cipher)
