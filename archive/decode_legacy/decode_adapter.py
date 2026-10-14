from __future__ import annotations

"""Decode QKV carrier → C-form Q / K_token / V_token.

Prefill ``prepare_attention_inputs_fhe`` requires ``carrier.seq_len == S``
and 16 source ciphers.  Decode has one LinearCarrier (lane 0 live).  This
adapter gathers that shard onto Attention C with the existing sparse-route
BSGS, fusing RoPE into the Q/K coefficients.  EasyFHE kernels are unchanged.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np

from llama3fhe.approx.attention import apply_rope_heads, rope_cos_sin
from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.operators.attention.codec import repeat_payload_cipher
from llama3fhe.operators.attention.prefill.input import (
    SparseDiagonalPlan,
    _apply_sparse_routes_batched_shared,
    _apply_sparse_routes_selected,
    choose_sparse_diagonal_plan,
)
from llama3fhe.operators.attention.sparse_bsgs_scheduler import (
    SparseMaskRoute,
    canonical_rotation,
)
from llama3fhe.operators.primitives import accumulate_owned


def _carrier_bases(layout: AttentionPairLayout) -> tuple[int, int, int, int]:
    head_dim = int(layout.head_dim)
    if head_dim % 2:
        raise ValueError("Decode QKV carrier folding requires even head_dim.")
    query_pairs = int(layout.query_pair_count)
    kv_pairs = int(layout.key_value_pair_count)
    if kv_pairs % 2:
        raise ValueError(
            "Decode V packing needs an even KV pair count, "
            f"got key_value_heads={layout.key_value_heads}."
        )
    query_fold = query_pairs * head_dim
    key_base = query_fold
    value_base = key_base + kv_pairs * head_dim
    return query_fold, key_base, value_base, head_dim


def unpack_decode_qkv_carrier(
    carrier: np.ndarray,
    *,
    layout: AttentionPairLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert heterogeneous QKV packing for one Decode token."""

    carrier = np.asarray(carrier, dtype=np.complex128).reshape(-1)
    query_fold, key_base, value_base, head_dim = _carrier_bases(layout)
    if carrier.size < value_base + int(layout.key_value_pair_count) * head_dim:
        raise ValueError(
            f"carrier width {carrier.size} is below the heterogeneous "
            f"QKV width {value_base + int(layout.key_value_pair_count) * head_dim}."
        )
    half = head_dim // 2
    query = np.zeros(
        (int(layout.query_heads), head_dim), dtype=np.float64
    )
    key = np.zeros(
        (int(layout.key_value_heads), head_dim), dtype=np.float64
    )
    value = np.zeros(
        (int(layout.key_value_heads), head_dim), dtype=np.float64
    )
    for pair_index, heads in enumerate(layout.query_head_pairs()):
        for feature_low in range(half):
            for lane, head in enumerate(heads):
                channel = pair_index * head_dim + 2 * feature_low + lane
                folded = 2.0 * carrier[channel]
                query[int(head), feature_low] = folded.real
                query[int(head), feature_low + half] = folded.imag
    for pair_index, heads in enumerate(layout.key_value_head_pairs()):
        for feature_low in range(half):
            for lane, head in enumerate(heads):
                channel = key_base + pair_index * head_dim + 2 * feature_low + lane
                stored = carrier[channel]
                key[int(head), feature_low] = stored.real
                key[int(head), feature_low + half] = -stored.imag
    pair_width = 2 * head_dim
    kv_pairs = layout.key_value_head_pairs()
    for group in range(int(layout.key_value_pair_count) // 2):
        for pair_offset, heads in enumerate(
            (kv_pairs[2 * group], kv_pairs[2 * group + 1])
        ):
            for feature in range(head_dim):
                for lane, head in enumerate(heads):
                    channel = value_base + group * pair_width + 2 * feature + lane
                    folded = 2.0 * carrier[channel]
                    value[int(head), feature] = (
                        folded.real if pair_offset == 0 else folded.imag
                    )
    return (
        query.astype(np.float32, copy=False),
        key.astype(np.float32, copy=False),
        value.astype(np.float32, copy=False),
    )


def decode_qkv_carrier_to_c_numpy(
    carrier: np.ndarray,
    *,
    layout: AttentionPairLayout,
    position: int = 0,
    token_index: int = 0,
    rope_theta: float = 500000.0,
):
    """NumPy oracle: one-token carrier → compact Q_C and K/V token columns."""

    query, key, value = unpack_decode_qkv_carrier(carrier, layout=layout)
    cos, sin = rope_cos_sin(
        seq_len=1,
        head_dim=int(layout.head_dim),
        theta=float(rope_theta),
        start_pos=int(position),
    )
    query = apply_rope_heads(
        query.reshape(1, -1),
        cos,
        sin,
        num_heads=int(layout.query_heads),
        head_dim=int(layout.head_dim),
    )
    key = apply_rope_heads(
        key.reshape(1, -1),
        cos,
        sin,
        num_heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
    )
    return (
        layout.pack_compact_query_c(query),
        layout.pack_token_c(key, token_index=int(token_index), kv=True, name="key"),
        layout.pack_token_c(
            value, token_index=int(token_index), kv=True, name="value"
        ),
        query,
        key,
        value,
    )


def _record_route(
    buckets: dict[int, dict[int, complex]],
    *,
    source: int,
    destination: int,
    coefficient: complex,
    slots: int,
) -> None:
    if coefficient == 0:
        return
    offset = canonical_rotation(int(source) - int(destination), int(slots))
    dests = buckets.setdefault(int(offset), {})
    dests[int(destination)] = dests.get(int(destination), 0.0) + complex(
        coefficient
    )


def _finalize_routes(
    buckets: dict[int, dict[int, complex]],
) -> dict[int, SparseMaskRoute]:
    routes: dict[int, SparseMaskRoute] = {}
    for offset, dest_map in buckets.items():
        destinations = np.fromiter(dest_map.keys(), dtype=np.int64)
        coefficients = np.asarray(
            [dest_map[int(destination)] for destination in destinations],
            dtype=np.complex128,
        )
        routes[int(offset)] = SparseMaskRoute(destinations, coefficients)
    return routes


def _query_pair_routes(
    pair_index: int,
    *,
    layout: AttentionPairLayout,
    token_lanes: int,
    slots: int,
    cos: np.ndarray,
    sin: np.ndarray,
    dest_token: int = 0,
) -> tuple[dict[int, SparseMaskRoute], dict[int, SparseMaskRoute]]:
    head_dim = int(layout.head_dim)
    half = head_dim // 2
    routes_z: dict[int, dict[int, complex]] = {}
    routes_conj: dict[int, dict[int, complex]] = {}
    for feature_low in range(half):
        rope = complex(float(cos[feature_low]), float(sin[feature_low]))
        conj_rope = complex(rope.real, -rope.imag)
        for lane in (0, 1):
            source = (pair_index * head_dim + 2 * feature_low + lane) * int(
                token_lanes
            )
            dest_lo = layout.c_slot(feature_low, int(dest_token), lane)
            dest_hi = layout.c_slot(feature_low + half, int(dest_token), lane)
            _record_route(
                routes_z,
                source=source,
                destination=dest_lo,
                coefficient=rope,
                slots=slots,
            )
            _record_route(
                routes_conj,
                source=source,
                destination=dest_lo,
                coefficient=conj_rope,
                slots=slots,
            )
            _record_route(
                routes_z,
                source=source,
                destination=dest_hi,
                coefficient=-1j * rope,
                slots=slots,
            )
            _record_route(
                routes_conj,
                source=source,
                destination=dest_hi,
                coefficient=1j * conj_rope,
                slots=slots,
            )
    return _finalize_routes(routes_z), _finalize_routes(routes_conj)


def _key_pair_routes(
    pair_index: int,
    *,
    layout: AttentionPairLayout,
    token_lanes: int,
    slots: int,
    key_base: int,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple[dict[int, SparseMaskRoute], dict[int, SparseMaskRoute]]:
    head_dim = int(layout.head_dim)
    half = head_dim // 2
    routes_z: dict[int, dict[int, complex]] = {}
    routes_conj: dict[int, dict[int, complex]] = {}
    for feature_low in range(half):
        rope = complex(float(cos[feature_low]), float(sin[feature_low]))
        conj_rope = complex(rope.real, -rope.imag)
        for lane in (0, 1):
            source = (
                key_base + pair_index * head_dim + 2 * feature_low + lane
            ) * int(token_lanes)
            dest_lo = layout.c_slot(feature_low, 0, lane)
            dest_hi = layout.c_slot(feature_low + half, 0, lane)
            _record_route(
                routes_z,
                source=source,
                destination=dest_lo,
                coefficient=0.5 * conj_rope,
                slots=slots,
            )
            _record_route(
                routes_conj,
                source=source,
                destination=dest_lo,
                coefficient=0.5 * rope,
                slots=slots,
            )
            _record_route(
                routes_z,
                source=source,
                destination=dest_hi,
                coefficient=0.5j * conj_rope,
                slots=slots,
            )
            _record_route(
                routes_conj,
                source=source,
                destination=dest_hi,
                coefficient=-0.5j * rope,
                slots=slots,
            )
    return _finalize_routes(routes_z), _finalize_routes(routes_conj)


def _value_pair_routes(
    pair_index: int,
    *,
    layout: AttentionPairLayout,
    token_lanes: int,
    slots: int,
    value_base: int,
) -> tuple[dict[int, SparseMaskRoute], dict[int, SparseMaskRoute]]:
    head_dim = int(layout.head_dim)
    pair_width = 2 * head_dim
    group = int(pair_index) // 2
    odd = int(pair_index) % 2
    routes_z: dict[int, dict[int, complex]] = {}
    routes_conj: dict[int, dict[int, complex]] = {}
    coeff_z = -1j if odd else 1.0
    coeff_conj = 1j if odd else 1.0
    for feature in range(head_dim):
        for lane in (0, 1):
            source = (
                value_base + group * pair_width + 2 * feature + lane
            ) * int(token_lanes)
            destination = layout.c_slot(feature, 0, lane)
            _record_route(
                routes_z,
                source=source,
                destination=destination,
                coefficient=coeff_z,
                slots=slots,
            )
            _record_route(
                routes_conj,
                source=source,
                destination=destination,
                coefficient=coeff_conj,
                slots=slots,
            )
    return _finalize_routes(routes_z), _finalize_routes(routes_conj)


def _plan_offsets(
    routes_z: dict[int, SparseMaskRoute],
    routes_conj: dict[int, SparseMaskRoute],
) -> tuple[int, ...]:
    return tuple(dict.fromkeys((*routes_z, *routes_conj)))


def decode_adapter_rotations(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    *,
    max_baby_offsets: int = 32,
) -> tuple[int, ...]:
    """Conjugation, sparse-gather, token-column, and compact-Q keys."""

    _, key_base, value_base, _ = _carrier_bases(attention_layout)
    slots = int(attention_layout.slots)
    token_lanes = int(carrier_layout.token_lanes)
    dummy_phase = np.ones((int(attention_layout.head_dim) // 2,), dtype=np.float64)
    rotations: set[int] = {4 * slots - 1}
    route_sets: list[
        tuple[dict[int, SparseMaskRoute], dict[int, SparseMaskRoute]]
    ] = []
    for pair_index in range(int(attention_layout.query_pair_count)):
        route_sets.append(
            _query_pair_routes(
                pair_index,
                layout=attention_layout,
                token_lanes=token_lanes,
                slots=slots,
                cos=dummy_phase,
                sin=dummy_phase,
            )
        )
    for pair_index in range(int(attention_layout.key_value_pair_count)):
        route_sets.append(
            _key_pair_routes(
                pair_index,
                layout=attention_layout,
                token_lanes=token_lanes,
                slots=slots,
                key_base=key_base,
                cos=dummy_phase,
                sin=dummy_phase,
            )
        )
        route_sets.append(
            _value_pair_routes(
                pair_index,
                layout=attention_layout,
                token_lanes=token_lanes,
                slots=slots,
                value_base=value_base,
            )
        )
    for routes_z, routes_conj in route_sets:
        offsets = _plan_offsets(routes_z, routes_conj)
        if not offsets:
            continue
        plan = choose_sparse_diagonal_plan(
            offsets, slots=slots, max_baby_offsets=int(max_baby_offsets)
        )
        rotations.update(int(value) for value in plan.baby_offsets)
        rotations.update(int(value) for value in plan.giant_offsets)
    bit = 1
    seq = int(attention_layout.seq_len)
    while bit < seq:
        rotations.add(-2 * bit)
        rotations.add(2 * bit)
        bit <<= 1
    for token_index in range(1, seq):
        rotations.add(-2 * int(token_index))
    step = int(attention_layout.payload_slots)
    while step < int(attention_layout.slots):
        rotations.add(-int(step))
        step *= 2
    return tuple(sorted(value for value in rotations if value))


def _scatter_pair(
    cipher,
    conjugated,
    *,
    routes_z: dict[int, SparseMaskRoute],
    routes_conj: dict[int, SparseMaskRoute],
    slots: int,
    crypto_context,
    function_prefix: str,
    max_baby_offsets: int,
    use_batched: bool = False,
    use_output_conjugation: bool = False,
    profile: dict[str, float] | None = None,
):
    wall_start = time.perf_counter()
    pieces: list[object] = []
    try:
        if routes_z:
            start = time.perf_counter()
            plan = choose_sparse_diagonal_plan(
                tuple(routes_z),
                slots=int(slots),
                max_baby_offsets=int(max_baby_offsets),
            )
            if profile is not None:
                profile["z_plan"] = time.perf_counter() - start
                profile["z_diagonals"] = float(len(routes_z))
                profile["z_destinations"] = float(
                    sum(len(route.destinations) for route in routes_z.values())
                )
                profile["z_baby_offsets"] = float(len(plan.baby_offsets))
                profile["z_giant_offsets"] = float(len(plan.giant_offsets))
                profile["z_rotations"] = float(plan.rotations_per_cipher)
            start = time.perf_counter()
            converted = _apply_sparse_routes_selected(
                (cipher,),
                routes=routes_z,
                plan=plan,
                slots=int(slots),
                crypto_context=crypto_context,
                function_prefix=f"{function_prefix}.z",
                padded_plaintext_rows=None,
                use_batched=bool(use_batched),
                batch_cap=32,
                fallback_on_error=True,
            )
            if profile is not None:
                profile["z_apply"] = time.perf_counter() - start
            pieces.append(converted[0])
        if routes_conj and bool(use_output_conjugation):
            if len(pieces) != 1:
                raise RuntimeError(
                    "output-conjugation sparse gather requires one z output."
                )
            start = time.perf_counter()
            pieces.append(
                crypto_context.fhe.homo_rotate(
                    pieces[0],
                    int(crypto_context.context.M) - 1,
                    crypto_context.context,
                )
            )
            if profile is not None:
                profile["output_conjugation"] = time.perf_counter() - start
                profile["conj_apply_elided"] = 1.0
        elif routes_conj:
            start = time.perf_counter()
            plan = choose_sparse_diagonal_plan(
                tuple(routes_conj),
                slots=int(slots),
                max_baby_offsets=int(max_baby_offsets),
            )
            if profile is not None:
                profile["conj_plan"] = time.perf_counter() - start
                profile["conj_diagonals"] = float(len(routes_conj))
                profile["conj_destinations"] = float(
                    sum(len(route.destinations) for route in routes_conj.values())
                )
                profile["conj_baby_offsets"] = float(len(plan.baby_offsets))
                profile["conj_giant_offsets"] = float(len(plan.giant_offsets))
                profile["conj_rotations"] = float(plan.rotations_per_cipher)
            start = time.perf_counter()
            converted = _apply_sparse_routes_selected(
                (conjugated,),
                routes=routes_conj,
                plan=plan,
                slots=int(slots),
                crypto_context=crypto_context,
                function_prefix=f"{function_prefix}.conj",
                padded_plaintext_rows=None,
                use_batched=bool(use_batched),
                batch_cap=32,
                fallback_on_error=True,
            )
            if profile is not None:
                profile["conj_apply"] = time.perf_counter() - start
            pieces.append(converted[0])
        if not pieces:
            raise RuntimeError(f"{function_prefix} produced no C ciphertext.")
        if len(pieces) == 1:
            result = pieces[0]
            pieces = []
            return result
        start = time.perf_counter()
        summed = crypto_context.fhe.homo_add(
            pieces[0], pieces[1], crypto_context.context
        )
        if profile is not None:
            profile["combine"] = time.perf_counter() - start
        for piece in pieces:
            release_if_supported(piece)
        pieces = []
        if profile is not None:
            profile["wall"] = time.perf_counter() - wall_start
        return summed
    except Exception:
        for piece in pieces:
            release_if_supported(piece)
        raise


def _shift_token_column(
    cipher,
    *,
    token_index: int,
    crypto_context,
):
    token_index = int(token_index)
    if token_index == 0:
        return cipher
    shifted = crypto_context.fhe.homo_rotate(
        cipher, -2 * token_index, crypto_context.context
    )
    release_if_supported(cipher)
    return shifted


@dataclass
class DecodeAdapterFHEResult:
    query_ciphers: tuple[object, ...]
    key_ciphers: tuple[object, ...]
    value_ciphers: tuple[object, ...]
    stage_seconds: dict[str, float] = field(default_factory=dict)
    route_stats: dict[str, float] = field(default_factory=dict)

    def release(self) -> None:
        for cipher in self.query_ciphers + self.key_ciphers + self.value_ciphers:
            release_if_supported(cipher)
        self.query_ciphers = ()
        self.key_ciphers = ()
        self.value_ciphers = ()


@dataclass(frozen=True)
class _DecodeRouteSpec:
    kind: str
    pair_index: int
    routes: dict[int, SparseMaskRoute]
    plan: SparseDiagonalPlan
    function_prefix: str
    build_seconds: float

    @property
    def label(self) -> str:
        return f"{self.kind}_pair_{self.pair_index:02d}"


def _greedy_shared_route_groups(
    specs: tuple[_DecodeRouteSpec, ...], *, max_basis_size: int = 32
) -> tuple[tuple[_DecodeRouteSpec, ...], ...]:
    """Group transforms when sharing their baby basis saves rotations."""

    groups = [
        ([spec], {int(value) for value in spec.plan.baby_offsets})
        for spec in specs
    ]
    while True:
        best = None
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                union = groups[left][1] | groups[right][1]
                if len(union) > int(max_basis_size):
                    continue
                saved = (
                    len(groups[left][1]) + len(groups[right][1]) - len(union)
                )
                candidate = (saved, -len(union), -left, -right)
                if saved > 0 and (best is None or candidate > best[0]):
                    best = (candidate, left, right, union)
        if best is None:
            break
        _, left, right, union = best
        groups[left] = (groups[left][0] + groups[right][0], union)
        groups.pop(right)
    return tuple(tuple(group_specs) for group_specs, _ in groups)


def _realify_sparse_output(cipher, *, crypto_context):
    conjugated = None
    try:
        conjugated = crypto_context.fhe.homo_rotate(
            cipher,
            int(crypto_context.context.M) - 1,
            crypto_context.context,
        )
        return crypto_context.fhe.homo_add(
            cipher, conjugated, crypto_context.context
        )
    finally:
        release_if_supported(conjugated)


def _decode_prepare_qkv_shared_baby_fhe(
    cipher,
    *,
    attention_layout: AttentionPairLayout,
    crypto_context,
    token_lanes: int,
    slots: int,
    key_base: int,
    value_base: int,
    phase_cos: np.ndarray,
    phase_sin: np.ndarray,
    token_index: int,
    max_baby_offsets: int,
) -> DecodeAdapterFHEResult:
    """Decode adapter with bounded cross-pair baby-rotation hoisting."""

    wall_start = time.perf_counter()
    specs: list[_DecodeRouteSpec] = []
    stage_seconds: dict[str, float] = {"conjugation": 0.0}
    route_stats: dict[str, float] = {
        "batched_sparse_bsgs": 1.0,
        "output_conjugation": 1.0,
        "shared_baby_rotations": 1.0,
    }

    def add_spec(kind: str, pair_index: int, builder, **kwargs) -> None:
        started = time.perf_counter()
        routes, _ = builder(
            pair_index,
            layout=attention_layout,
            token_lanes=int(token_lanes),
            slots=int(slots),
            **kwargs,
        )
        plan = choose_sparse_diagonal_plan(
            tuple(routes),
            slots=int(slots),
            max_baby_offsets=int(max_baby_offsets),
        )
        elapsed = time.perf_counter() - started
        label = f"{kind}_pair_{pair_index:02d}"
        stage_seconds[f"{label}.route_build_and_plan"] = elapsed
        route_stats[f"{label}.diagonals"] = float(len(routes))
        route_stats[f"{label}.baby_offsets"] = float(len(plan.baby_offsets))
        route_stats[f"{label}.giant_offsets"] = float(len(plan.giant_offsets))
        route_stats[f"{label}.rotations"] = float(plan.rotations_per_cipher)
        specs.append(
            _DecodeRouteSpec(
                kind=kind,
                pair_index=int(pair_index),
                routes=routes,
                plan=plan,
                function_prefix=f"decode.adapter.{kind}{pair_index}.shared",
                build_seconds=elapsed,
            )
        )

    for pair_index in range(int(attention_layout.query_pair_count)):
        add_spec(
            "query",
            pair_index,
            _query_pair_routes,
            cos=phase_cos,
            sin=phase_sin,
        )
    for pair_index in range(int(attention_layout.key_value_pair_count)):
        add_spec(
            "key",
            pair_index,
            _key_pair_routes,
            key_base=int(key_base),
            cos=phase_cos,
            sin=phase_sin,
        )
        add_spec(
            "value",
            pair_index,
            _value_pair_routes,
            value_base=int(value_base),
        )

    groups = _greedy_shared_route_groups(tuple(specs), max_basis_size=32)
    gathered: dict[tuple[str, int], object] = {}
    kind_seconds = {"query": 0.0, "key": 0.0, "value": 0.0}
    compact_q = None
    query_ciphers: list[object] = []
    key_ciphers: list[object] = []
    value_ciphers: list[object] = []
    success = False
    try:
        for group_index, group in enumerate(groups):
            profiles = [{} for _ in group]
            started = time.perf_counter()
            outputs = _apply_sparse_routes_batched_shared(
                cipher,
                transforms=tuple(
                    (spec.routes, spec.plan, spec.function_prefix)
                    for spec in group
                ),
                slots=int(slots),
                crypto_context=crypto_context,
                batch_cap=32,
                profiles=profiles,
            )
            group_wall = time.perf_counter() - started
            stage_seconds[f"shared_group_{group_index:02d}.apply"] = group_wall
            basis = {
                int(offset)
                for spec in group
                for offset in spec.plan.baby_offsets
            }
            unshared = sum(len(spec.plan.baby_offsets) for spec in group)
            route_stats[f"shared_group_{group_index:02d}.size"] = float(
                len(group)
            )
            route_stats[f"shared_group_{group_index:02d}.basis"] = float(
                len(basis)
            )
            route_stats[f"shared_group_{group_index:02d}.baby_saved"] = float(
                unshared - len(basis)
            )
            attributed = group_wall / len(group)
            try:
                for spec, output, profile in zip(
                    group, outputs, profiles, strict=True
                ):
                    realify_start = time.perf_counter()
                    real_output = _realify_sparse_output(
                        output, crypto_context=crypto_context
                    )
                    realify_seconds = time.perf_counter() - realify_start
                    gathered[(spec.kind, spec.pair_index)] = real_output
                    kind_seconds[spec.kind] += attributed + realify_seconds
                    stage_seconds[f"{spec.label}.shared_apply_attributed"] = (
                        attributed
                    )
                    stage_seconds[f"{spec.label}.output_conjugation"] = (
                        realify_seconds
                    )
                    route_stats[f"{spec.label}.shared_basis"] = float(
                        profile["shared_basis_size"]
                    )
            finally:
                for output in outputs:
                    release_if_supported(output)

        for pair_index in range(int(attention_layout.query_pair_count)):
            label = f"query_pair_{pair_index:02d}"
            post_start = time.perf_counter()
            gathered_q = gathered.pop(("query", pair_index))
            placed = _shift_token_column(
                gathered_q,
                token_index=int(pair_index),
                crypto_context=crypto_context,
            )
            compact_q = accumulate_owned(
                compact_q, placed, crypto_context=crypto_context
            )
            elapsed = time.perf_counter() - post_start
            stage_seconds[f"{label}.shift_accumulate"] = elapsed
            kind_seconds["query"] += elapsed
        repeat_start = time.perf_counter()
        query_ciphers.append(
            repeat_payload_cipher(
                compact_q,
                layout=attention_layout,
                crypto_context=crypto_context,
            )
        )
        compact_q = None
        repeat_seconds = time.perf_counter() - repeat_start
        stage_seconds["query_payload_repeat"] = repeat_seconds
        kind_seconds["query"] += repeat_seconds

        for kind, target in (("key", key_ciphers), ("value", value_ciphers)):
            for pair_index in range(int(attention_layout.key_value_pair_count)):
                label = f"{kind}_pair_{pair_index:02d}"
                post_start = time.perf_counter()
                output = gathered.pop((kind, pair_index))
                target.append(
                    repeat_payload_cipher(
                        _shift_token_column(
                            output,
                            token_index=int(token_index),
                            crypto_context=crypto_context,
                        ),
                        layout=attention_layout,
                        crypto_context=crypto_context,
                    )
                )
                elapsed = time.perf_counter() - post_start
                stage_seconds[f"{label}.shift_repeat"] = elapsed
                kind_seconds[kind] += elapsed

        build_seconds = {"query": 0.0, "key": 0.0, "value": 0.0}
        for spec in specs:
            build_seconds[spec.kind] += spec.build_seconds
        stage_seconds["query_routes"] = (
            kind_seconds["query"] + build_seconds["query"]
        )
        stage_seconds["key_routes"] = (
            kind_seconds["key"] + build_seconds["key"]
        )
        stage_seconds["value_routes"] = (
            kind_seconds["value"] + build_seconds["value"]
        )
        stage_seconds["wall"] = time.perf_counter() - wall_start
        route_stats["shared_group_count"] = float(len(groups))
        route_stats["baby_offsets_unshared"] = float(
            sum(len(spec.plan.baby_offsets) for spec in specs)
        )
        route_stats["baby_offsets_shared"] = float(
            sum(
                len(
                    {
                        int(offset)
                        for spec in group
                        for offset in spec.plan.baby_offsets
                    }
                )
                for group in groups
            )
        )
        success = True
        return DecodeAdapterFHEResult(
            query_ciphers=tuple(query_ciphers),
            key_ciphers=tuple(key_ciphers),
            value_ciphers=tuple(value_ciphers),
            stage_seconds=stage_seconds,
            route_stats=route_stats,
        )
    finally:
        if not success:
            release_if_supported(compact_q)
            for output in gathered.values():
                release_if_supported(output)
            for output in query_ciphers + key_ciphers + value_ciphers:
                release_if_supported(output)


def decode_prepare_qkv_fhe(
    carrier_ciphers: tuple[object, ...] | list[object],
    *,
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    position: int = 0,
    token_index: int = 0,
    rope_theta: float = 500000.0,
    max_baby_offsets: int = 32,
    batched_sparse_bsgs: bool | None = None,
    output_conjugation: bool | None = None,
    shared_baby_rotations: bool | None = None,
) -> DecodeAdapterFHEResult:
    """One Decode carrier → compact Q_C and one-column K_C / V_C.

    Does not consume the carrier.  K/V land on column 0 then rotate to
    ``token_index``.  Q pairs land on token columns ``pair_index`` in one
    ciphertext (no token broadcast).
    """

    carrier_ciphers = tuple(carrier_ciphers)
    if int(carrier_layout.seq_len) != 1 or len(carrier_ciphers) != 1:
        raise ValueError(
            "Decode adapter expects one seq_len=1 LinearCarrier ciphertext, "
            f"got seq_len={carrier_layout.seq_len} count={len(carrier_ciphers)}."
        )
    if int(carrier_layout.slots) != int(attention_layout.slots):
        raise ValueError("Decode carrier and attention slot counts must match.")
    _, key_base, value_base, _ = _carrier_bases(attention_layout)
    slots = int(attention_layout.slots)
    token_lanes = int(carrier_layout.token_lanes)
    cos, sin = rope_cos_sin(
        seq_len=1,
        head_dim=int(attention_layout.head_dim),
        theta=float(rope_theta),
        start_pos=int(position),
    )
    phase_cos = np.asarray(
        cos[0, : int(attention_layout.head_dim) // 2], dtype=np.float64
    )
    phase_sin = np.asarray(
        sin[0, : int(attention_layout.head_dim) // 2], dtype=np.float64
    )
    cipher = carrier_ciphers[0]
    conjugated = None
    compact_q = None
    query_ciphers: list[object] = []
    key_ciphers: list[object] = []
    value_ciphers: list[object] = []
    stage_seconds: dict[str, float] = {}
    route_stats: dict[str, float] = {}
    if batched_sparse_bsgs is None:
        batched_sparse_bsgs = bool(
            int(os.environ.get("LLAMA_FHE_DECODE_ADAPTER_BATCHED", "1"))
        )
    if output_conjugation is None:
        output_conjugation = bool(
            int(os.environ.get("LLAMA_FHE_DECODE_ADAPTER_OUTPUT_CONJ", "1"))
        )
    if shared_baby_rotations is None:
        shared_baby_rotations = bool(
            int(os.environ.get("LLAMA_FHE_DECODE_ADAPTER_SHARED_BABY", "1"))
        )
    if (
        bool(shared_baby_rotations)
        and bool(batched_sparse_bsgs)
        and bool(output_conjugation)
    ):
        return _decode_prepare_qkv_shared_baby_fhe(
            cipher,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            token_lanes=token_lanes,
            slots=slots,
            key_base=key_base,
            value_base=value_base,
            phase_cos=phase_cos,
            phase_sin=phase_sin,
            token_index=int(token_index),
            max_baby_offsets=int(max_baby_offsets),
        )
    route_stats["batched_sparse_bsgs"] = float(bool(batched_sparse_bsgs))
    route_stats["output_conjugation"] = float(bool(output_conjugation))
    route_stats["shared_baby_rotations"] = 0.0
    success = False
    try:
        wall_start = time.perf_counter()
        start = time.perf_counter()
        if not bool(output_conjugation):
            conjugated = crypto_context.fhe.homo_rotate(
                cipher,
                int(crypto_context.context.M) - 1,
                crypto_context.context,
            )
        stage_seconds["conjugation"] = time.perf_counter() - start
        query_start = time.perf_counter()
        for pair_index in range(int(attention_layout.query_pair_count)):
            pair_label = f"query_pair_{pair_index:02d}"
            route_start = time.perf_counter()
            routes_z, routes_conj = _query_pair_routes(
                pair_index,
                layout=attention_layout,
                token_lanes=token_lanes,
                slots=slots,
                cos=phase_cos,
                sin=phase_sin,
            )
            stage_seconds[f"{pair_label}.route_build"] = (
                time.perf_counter() - route_start
            )
            scatter_profile: dict[str, float] = {}
            gathered = _scatter_pair(
                cipher,
                conjugated,
                routes_z=routes_z,
                routes_conj=routes_conj,
                slots=slots,
                crypto_context=crypto_context,
                function_prefix=f"decode.adapter.query{pair_index}",
                max_baby_offsets=int(max_baby_offsets),
                use_batched=bool(batched_sparse_bsgs),
                use_output_conjugation=bool(output_conjugation),
                profile=scatter_profile,
            )
            for name, value in scatter_profile.items():
                target = route_stats if name in {
                    "z_diagonals", "z_destinations", "z_baby_offsets",
                    "z_giant_offsets", "z_rotations", "conj_diagonals",
                    "conj_destinations", "conj_baby_offsets",
                    "conj_giant_offsets", "conj_rotations",
                    "conj_apply_elided",
                } else stage_seconds
                target[f"{pair_label}.{name}"] = float(value)
            shift_start = time.perf_counter()
            placed = _shift_token_column(
                gathered,
                token_index=int(pair_index),
                crypto_context=crypto_context,
            )
            stage_seconds[f"{pair_label}.token_shift"] = (
                time.perf_counter() - shift_start
            )
            accumulate_start = time.perf_counter()
            compact_q = accumulate_owned(
                compact_q, placed, crypto_context=crypto_context
            )
            stage_seconds[f"{pair_label}.accumulate"] = (
                time.perf_counter() - accumulate_start
            )
        stage_seconds["query_routes"] = time.perf_counter() - query_start
        start = time.perf_counter()
        query_ciphers.append(
            repeat_payload_cipher(
                compact_q,
                layout=attention_layout,
                crypto_context=crypto_context,
            )
        )
        stage_seconds["query_payload_repeat"] = time.perf_counter() - start
        compact_q = None
        key_elapsed = 0.0
        value_elapsed = 0.0
        for pair_index in range(int(attention_layout.key_value_pair_count)):
            pair_label = f"key_pair_{pair_index:02d}"
            pair_start = time.perf_counter()
            route_start = pair_start
            routes_z, routes_conj = _key_pair_routes(
                pair_index,
                layout=attention_layout,
                token_lanes=token_lanes,
                slots=slots,
                key_base=key_base,
                cos=phase_cos,
                sin=phase_sin,
            )
            stage_seconds[f"{pair_label}.route_build"] = (
                time.perf_counter() - route_start
            )
            scatter_profile = {}
            gathered = _scatter_pair(
                cipher,
                conjugated,
                routes_z=routes_z,
                routes_conj=routes_conj,
                slots=slots,
                crypto_context=crypto_context,
                function_prefix=f"decode.adapter.key{pair_index}",
                max_baby_offsets=int(max_baby_offsets),
                use_batched=bool(batched_sparse_bsgs),
                use_output_conjugation=bool(output_conjugation),
                profile=scatter_profile,
            )
            for name, value in scatter_profile.items():
                target = route_stats if name in {
                    "z_diagonals", "z_destinations", "z_baby_offsets",
                    "z_giant_offsets", "z_rotations", "conj_diagonals",
                    "conj_destinations", "conj_baby_offsets",
                    "conj_giant_offsets", "conj_rotations",
                    "conj_apply_elided",
                } else stage_seconds
                target[f"{pair_label}.{name}"] = float(value)
            finish_start = time.perf_counter()
            key_ciphers.append(
                repeat_payload_cipher(
                    _shift_token_column(
                        gathered,
                        token_index=int(token_index),
                        crypto_context=crypto_context,
                    ),
                    layout=attention_layout,
                    crypto_context=crypto_context,
                )
            )
            stage_seconds[f"{pair_label}.shift_repeat"] = (
                time.perf_counter() - finish_start
            )
            key_elapsed += time.perf_counter() - pair_start
            pair_label = f"value_pair_{pair_index:02d}"
            pair_start = time.perf_counter()
            route_start = pair_start
            routes_z, routes_conj = _value_pair_routes(
                pair_index,
                layout=attention_layout,
                token_lanes=token_lanes,
                slots=slots,
                value_base=value_base,
            )
            stage_seconds[f"{pair_label}.route_build"] = (
                time.perf_counter() - route_start
            )
            scatter_profile = {}
            gathered = _scatter_pair(
                cipher,
                conjugated,
                routes_z=routes_z,
                routes_conj=routes_conj,
                slots=slots,
                crypto_context=crypto_context,
                function_prefix=f"decode.adapter.value{pair_index}",
                max_baby_offsets=int(max_baby_offsets),
                use_batched=bool(batched_sparse_bsgs),
                use_output_conjugation=bool(output_conjugation),
                profile=scatter_profile,
            )
            for name, value in scatter_profile.items():
                target = route_stats if name in {
                    "z_diagonals", "z_destinations", "z_baby_offsets",
                    "z_giant_offsets", "z_rotations", "conj_diagonals",
                    "conj_destinations", "conj_baby_offsets",
                    "conj_giant_offsets", "conj_rotations",
                    "conj_apply_elided",
                } else stage_seconds
                target[f"{pair_label}.{name}"] = float(value)
            finish_start = time.perf_counter()
            value_ciphers.append(
                repeat_payload_cipher(
                    _shift_token_column(
                        gathered,
                        token_index=int(token_index),
                        crypto_context=crypto_context,
                    ),
                    layout=attention_layout,
                    crypto_context=crypto_context,
                )
            )
            stage_seconds[f"{pair_label}.shift_repeat"] = (
                time.perf_counter() - finish_start
            )
            value_elapsed += time.perf_counter() - pair_start
        stage_seconds["key_routes"] = key_elapsed
        stage_seconds["value_routes"] = value_elapsed
        stage_seconds["wall"] = time.perf_counter() - wall_start
        success = True
        return DecodeAdapterFHEResult(
            query_ciphers=tuple(query_ciphers),
            key_ciphers=tuple(key_ciphers),
            value_ciphers=tuple(value_ciphers),
            stage_seconds=stage_seconds,
            route_stats=route_stats,
        )
    finally:
        release_if_supported(conjugated)
        if not success:
            release_if_supported(compact_q)
            for cipher in query_ciphers + key_ciphers + value_ciphers:
                release_if_supported(cipher)


__all__ = [
    "DecodeAdapterFHEResult",
    "decode_adapter_rotations",
    "decode_prepare_qkv_fhe",
    "decode_qkv_carrier_to_c_numpy",
    "unpack_decode_qkv_carrier",
]
