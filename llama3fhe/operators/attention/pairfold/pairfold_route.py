from __future__ import annotations

"""Decode-native routing from one sparse QKV carrier to PairFold128.

The input is the complete 32768-slot ciphertext emitted by the PairFold QKV
projection.  Active carrier rows are sparse across the physical 4096-channel
width; this module therefore never treats the first 3072 rows as an active
prefix.  K/V are extracted once per token, while Q is extracted one ``(g, r)``
stream at a time so eight query ciphertexts never become simultaneously live.
"""

import time
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from llama3fhe.approx.attention import rope_cos_sin
from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.pairfold_kv_cache import PairFoldTokenWrite

from ..prefill.input import (
    SparseDiagonalPlan,
    _apply_sparse_routes_batched_shared,
    _apply_sparse_routes_selected,
)
from ..sparse_bsgs_scheduler import (
    SparseMaskRoute,
    canonical_rotation,
    sparse_route_arrays,
)

PAIRFOLD_ROUTE_MODULUS = 16


@dataclass(frozen=True)
class PairFoldRouteComplexity:
    """Logical work and physical rotations for one Decode routing step."""

    token_column: int
    kv_transform_count: int
    kv_output_cipher_count: int
    kv_diagonal_count: int
    kv_pt_ct_multiplications: int
    kv_shared_baby_basis_size: int
    kv_baby_rotations: int
    kv_giant_rotations: int
    query_stream_count: int
    query_live_cipher_count: int
    query_diagonals_per_stream: int
    query_pt_ct_multiplications: int
    query_baby_rotations: int
    query_giant_rotations: int
    output_rescales: int

    @property
    def pt_ct_multiplications(self) -> int:
        return int(self.kv_pt_ct_multiplications) + int(
            self.query_pt_ct_multiplications
        )

    @property
    def rotations(self) -> int:
        return (
            int(self.kv_baby_rotations)
            + int(self.kv_giant_rotations)
            + int(self.query_baby_rotations)
            + int(self.query_giant_rotations)
        )

    def to_dict(self) -> dict[str, int]:
        values = {str(key): int(value) for key, value in asdict(self).items()}
        values["pt_ct_multiplications"] = self.pt_ct_multiplications
        values["rotations"] = self.rotations
        return values


@dataclass(frozen=True)
class _RouteTransform:
    label: str
    routes: dict[int, SparseMaskRoute]
    plan: SparseDiagonalPlan


def _validate_layouts(
    carrier_layout: LinearCarrierLayout,
    pairfold_layout: PairFold128Layout,
) -> None:
    if int(carrier_layout.seq_len) != 1:
        raise ValueError(
            "PairFold Decode routing requires a seq_len=1 LinearCarrier, "
            f"got {carrier_layout.seq_len}."
        )
    if int(carrier_layout.dimension) != int(pairfold_layout.carrier_dimension):
        raise ValueError(
            "PairFold Decode routing requires physical carrier width "
            f"{pairfold_layout.carrier_dimension}, got {carrier_layout.dimension}."
        )
    if int(carrier_layout.slots) != int(pairfold_layout.slots):
        raise ValueError("carrier and PairFold layouts must have the same slots.")
    if int(carrier_layout.token_lanes) != int(
        pairfold_layout.native_token_lanes
    ):
        raise ValueError(
            "carrier token lanes do not match the PairFold native carrier."
        )
    if int(carrier_layout.cipher_count) != 1:
        raise ValueError("PairFold Decode routing expects exactly one carrier.")


def _full_carrier_payload(
    carrier: np.ndarray, *, layout: PairFold128Layout
) -> np.ndarray:
    payload = np.asarray(carrier)
    if payload.shape == (1, int(layout.slots)):
        payload = payload[0]
    if payload.shape != (int(layout.slots),):
        raise ValueError(
            "PairFold routing requires the complete physical carrier with shape "
            f"{(layout.slots,)}, got {payload.shape}."
        )
    if not np.iscomplexobj(payload):
        payload = payload.astype(np.complex128)
    return payload


def _rope_phase(
    *,
    layout: PairFold128Layout,
    position: int,
    rope_theta: float,
) -> np.ndarray:
    position = int(position)
    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}.")
    cos, sin = rope_cos_sin(
        seq_len=1,
        head_dim=int(layout.head_dim),
        theta=float(rope_theta),
        start_pos=position,
    )
    return np.asarray(
        cos[0, : layout.fold_dim] + 1j * sin[0, : layout.fold_dim],
        dtype=np.complex128,
    )


def _residue_routes(
    *,
    layout: PairFold128Layout,
    residue: int,
    column: int,
    phase: np.ndarray,
) -> dict[int, SparseMaskRoute]:
    phase = np.asarray(phase, dtype=np.complex128).reshape(-1)
    if phase.shape != (layout.fold_dim,):
        raise ValueError(
            f"route phase must have shape {(layout.fold_dim,)}, got {phase.shape}."
        )
    if not np.all(np.isfinite(phase)):
        raise ValueError("route phase must be finite.")

    routes: dict[int, SparseMaskRoute] = {}
    for lane in range(layout.head_lanes):
        offset = canonical_rotation(
            layout.carrier_to_page_rotation(residue, lane, column),
            int(layout.slots),
        )
        destinations = np.asarray(
            [
                layout.page_slot(pair, feature, column, lane)
                for pair in range(layout.pair_blocks)
                for feature in range(layout.fold_dim)
            ],
            dtype=np.int64,
        )
        coefficients = np.tile(phase, layout.pair_blocks)
        routes[int(offset)] = SparseMaskRoute(destinations, coefficients)
    return routes


def _pairfold_plan(
    routes: dict[int, SparseMaskRoute],
    *,
    slots: int,
) -> SparseDiagonalPlan:
    """Use the carrier's 16-slot residue spacing as a shared BSGS radix."""

    offsets = tuple(
        sorted(
            {
                canonical_rotation(int(offset), int(slots))
                for offset in routes
            }
        )
    )
    if not offsets:
        raise ValueError("a PairFold route transform must not be empty.")
    babies = tuple(
        sorted({int(offset) % PAIRFOLD_ROUTE_MODULUS for offset in offsets})
    )
    giants = tuple(
        sorted(
            {
                canonical_rotation(
                    int(offset) - (int(offset) % PAIRFOLD_ROUTE_MODULUS),
                    int(slots),
                )
                for offset in offsets
            }
        )
    )
    return SparseDiagonalPlan(
        modulus=PAIRFOLD_ROUTE_MODULUS,
        diagonal_offsets=offsets,
        baby_offsets=babies,
        giant_offsets=giants,
    )


def _kv_transforms(
    *,
    layout: PairFold128Layout,
    column: int,
    phase: np.ndarray,
) -> tuple[_RouteTransform, ...]:
    column = int(column)
    layout._require_token(column)
    specs: list[_RouteTransform] = []
    for kind in ("key", "value"):
        for group in range(int(layout.key_value_groups)):
            residue = (
                layout.key_residue(group)
                if kind == "key"
                else layout.value_residue(group)
            )
            routes = _residue_routes(
                layout=layout,
                residue=residue,
                column=column,
                phase=np.conjugate(phase)
                if kind == "key"
                else np.ones_like(phase),
            )
            specs.append(
                _RouteTransform(
                    label=f"{kind}.g{group}",
                    routes=routes,
                    plan=_pairfold_plan(routes, slots=int(layout.slots)),
                )
            )
    return tuple(specs)


def _query_transform(
    *,
    layout: PairFold128Layout,
    group: int,
    query_offset: int,
    phase: np.ndarray,
) -> _RouteTransform:
    residue = layout.query_residue(group, query_offset)
    routes = _residue_routes(
        layout=layout,
        residue=residue,
        column=0,
        phase=phase,
    )
    return _RouteTransform(
        label=f"query.g{int(group)}.r{int(query_offset)}",
        routes=routes,
        plan=_pairfold_plan(routes, slots=int(layout.slots)),
    )


def _apply_routes_numpy(
    carrier: np.ndarray,
    *,
    routes: dict[int, SparseMaskRoute],
) -> np.ndarray:
    """Full-slot oracle matching EasyFHE's source-minus-destination rotate."""

    output = np.zeros_like(carrier, dtype=np.result_type(carrier, np.complex128))
    for offset, route in routes.items():
        destinations, coefficients = sparse_route_arrays(route)
        rotated = np.roll(carrier, -int(offset))
        output[destinations] += rotated[destinations] * coefficients
    return output


def pairfold_kv_from_carrier_numpy(
    carrier: np.ndarray,
    *,
    layout: PairFold128Layout,
    position: int,
    token_column: int,
    rope_theta: float = 500000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-slot K/V oracle; returns two K rows and two V rows."""

    payload = _full_carrier_payload(carrier, layout=layout)
    phase = _rope_phase(
        layout=layout, position=int(position), rope_theta=float(rope_theta)
    )
    outputs = tuple(
        _apply_routes_numpy(payload, routes=spec.routes)
        for spec in _kv_transforms(
            layout=layout, column=int(token_column), phase=phase
        )
    )
    return np.stack(outputs[:2]), np.stack(outputs[2:])


def pairfold_query_from_carrier_numpy(
    carrier: np.ndarray,
    *,
    layout: PairFold128Layout,
    group: int,
    query_offset: int,
    position: int,
    rope_theta: float = 500000.0,
) -> np.ndarray:
    """Extract one Q ``(KV group, GQA offset)`` at PairFold column zero."""

    payload = _full_carrier_payload(carrier, layout=layout)
    phase = _rope_phase(
        layout=layout, position=int(position), rope_theta=float(rope_theta)
    )
    spec = _query_transform(
        layout=layout,
        group=int(group),
        query_offset=int(query_offset),
        phase=phase,
    )
    return _apply_routes_numpy(payload, routes=spec.routes)


def _validate_basis_size(
    specs: Iterable[_RouteTransform], *, max_baby_offsets: int
) -> tuple[int, ...]:
    limit = int(max_baby_offsets)
    if not 0 < limit <= 32:
        raise ValueError("max_baby_offsets must be in [1, 32].")
    basis = tuple(
        sorted(
            {
                int(offset)
                for spec in specs
                for offset in spec.plan.baby_offsets
            }
        )
    )
    if len(basis) > limit:
        raise ValueError(
            f"PairFold shared baby basis has {len(basis)} offsets, "
            f"exceeding max_baby_offsets={limit}."
        )
    return basis


def pairfold_route_complexity(
    layout: PairFold128Layout,
    *,
    token_column: int,
    max_baby_offsets: int = 32,
) -> PairFoldRouteComplexity:
    phase = np.ones((layout.fold_dim,), dtype=np.complex128)
    kv_specs = _kv_transforms(
        layout=layout, column=int(token_column), phase=phase
    )
    kv_basis = _validate_basis_size(
        kv_specs, max_baby_offsets=int(max_baby_offsets)
    )
    query_specs = tuple(
        _query_transform(
            layout=layout,
            group=group,
            query_offset=query_offset,
            phase=phase,
        )
        for group in range(int(layout.key_value_groups))
        for query_offset in range(layout.gqa_ratio)
    )
    for spec in query_specs:
        _validate_basis_size((spec,), max_baby_offsets=int(max_baby_offsets))
    return PairFoldRouteComplexity(
        token_column=int(token_column),
        kv_transform_count=len(kv_specs),
        kv_output_cipher_count=len(kv_specs),
        kv_diagonal_count=sum(len(spec.routes) for spec in kv_specs),
        kv_pt_ct_multiplications=sum(len(spec.routes) for spec in kv_specs),
        kv_shared_baby_basis_size=len(kv_basis),
        kv_baby_rotations=len(kv_basis) - int(0 in kv_basis),
        kv_giant_rotations=sum(
            len(spec.plan.giant_offsets) - int(0 in spec.plan.giant_offsets)
            for spec in kv_specs
        ),
        query_stream_count=len(query_specs),
        query_live_cipher_count=1,
        query_diagonals_per_stream=2,
        query_pt_ct_multiplications=sum(
            len(spec.routes) for spec in query_specs
        ),
        query_baby_rotations=sum(
            len(spec.plan.baby_offsets) - int(0 in spec.plan.baby_offsets)
            for spec in query_specs
        ),
        query_giant_rotations=sum(
            len(spec.plan.giant_offsets) - int(0 in spec.plan.giant_offsets)
            for spec in query_specs
        ),
        output_rescales=len(kv_specs) + len(query_specs),
    )


def pairfold_route_rotations(
    layout: PairFold128Layout,
    *,
    token_columns: int | Iterable[int] | None = None,
    include_kv: bool = True,
    include_query: bool = True,
    max_baby_offsets: int = 32,
) -> tuple[int, ...]:
    """Rotation-key union for selected columns (all 128 by default)."""

    if token_columns is None:
        columns = tuple(range(int(layout.seq_len)))
    elif isinstance(token_columns, (int, np.integer)):
        columns = (int(token_columns),)
    else:
        columns = tuple(int(column) for column in token_columns)
    phase = np.ones((layout.fold_dim,), dtype=np.complex128)
    rotations: set[int] = set()
    if bool(include_kv):
        for column in columns:
            specs = _kv_transforms(layout=layout, column=column, phase=phase)
            _validate_basis_size(
                specs, max_baby_offsets=int(max_baby_offsets)
            )
            for spec in specs:
                rotations.update(spec.plan.baby_offsets)
                rotations.update(spec.plan.giant_offsets)
    if bool(include_query):
        for group in range(int(layout.key_value_groups)):
            for query_offset in range(layout.gqa_ratio):
                spec = _query_transform(
                    layout=layout,
                    group=group,
                    query_offset=query_offset,
                    phase=phase,
                )
                _validate_basis_size(
                    (spec,), max_baby_offsets=int(max_baby_offsets)
                )
                rotations.update(spec.plan.baby_offsets)
                rotations.update(spec.plan.giant_offsets)
    return tuple(
        sorted(
            canonical_rotation(value, int(layout.slots))
            for value in rotations
            if canonical_rotation(value, int(layout.slots))
        )
    )


def _release_outputs(outputs: Iterable[object], *, borrowed: object) -> None:
    seen: set[int] = set()
    for output in tuple(outputs):
        identity = id(output)
        if identity == id(borrowed) or identity in seen:
            continue
        seen.add(identity)
        try:
            release_if_supported(output)
        except BaseException:
            pass


def _validate_outputs(
    outputs: tuple[object, ...], *, expected: int, borrowed: object
) -> None:
    if len(outputs) != int(expected):
        raise RuntimeError(
            f"PairFold routing expected {expected} outputs, got {len(outputs)}."
        )
    if any(output is None for output in outputs):
        raise RuntimeError("PairFold routing produced a None ciphertext.")
    if len({id(output) for output in outputs}) != len(outputs):
        raise RuntimeError("PairFold routing outputs must not alias.")
    if any(output is borrowed for output in outputs):
        raise RuntimeError("PairFold routing output must not alias its carrier.")


def _apply_one_transform(
    carrier_cipher,
    *,
    spec: _RouteTransform,
    layout: PairFold128Layout,
    crypto_context,
    max_baby_offsets: int,
    batched_sparse_bsgs: bool,
    fallback_on_error: bool,
    profile: dict[str, float] | None,
) -> object:
    _validate_basis_size((spec,), max_baby_offsets=int(max_baby_offsets))
    outputs = _apply_sparse_routes_selected(
        (carrier_cipher,),
        routes=spec.routes,
        plan=spec.plan,
        slots=int(layout.slots),
        crypto_context=crypto_context,
        function_prefix=f"decode.pairfold_route.{spec.label}",
        padded_plaintext_rows=None,
        use_batched=bool(batched_sparse_bsgs),
        batch_cap=32,
        fallback_on_error=bool(fallback_on_error),
        profile=profile,
    )
    try:
        _validate_outputs(outputs, expected=1, borrowed=carrier_cipher)
        return outputs[0]
    except BaseException:
        _release_outputs(outputs, borrowed=carrier_cipher)
        raise


def pairfold_kv_from_carrier_fhe(
    carrier_cipher,
    *,
    carrier_layout: LinearCarrierLayout,
    layout: PairFold128Layout,
    crypto_context,
    position: int,
    token_column: int,
    rope_theta: float = 500000.0,
    max_baby_offsets: int = 32,
    batched_sparse_bsgs: bool = True,
    shared_baby_rotations: bool = True,
    fallback_on_error: bool = True,
    profile: dict[str, float] | None = None,
) -> PairFoldTokenWrite:
    """Borrow one carrier and return an owned 2K+2V token write."""

    _validate_layouts(carrier_layout, layout)
    if carrier_cipher is None:
        raise ValueError("carrier_cipher must not be None.")
    wall_start = time.perf_counter()
    build_start = time.perf_counter()
    phase = _rope_phase(
        layout=layout, position=int(position), rope_theta=float(rope_theta)
    )
    specs = _kv_transforms(
        layout=layout, column=int(token_column), phase=phase
    )
    basis = _validate_basis_size(
        specs, max_baby_offsets=int(max_baby_offsets)
    )
    build_seconds = time.perf_counter() - build_start
    if profile is not None:
        profile.clear()
        profile["route_build_and_plan"] = build_seconds
        profile["transform_count"] = float(len(specs))
        profile["logical_pt_ct"] = float(sum(len(spec.routes) for spec in specs))
        profile["shared_basis_size"] = float(len(basis))
        profile["baby_rotations"] = float(len(basis) - int(0 in basis))
        profile["giant_rotations"] = float(
            sum(
                len(spec.plan.giant_offsets)
                - int(0 in spec.plan.giant_offsets)
                for spec in specs
            )
        )
        profile["output_cipher_count"] = float(len(specs))
        profile["route_modulus"] = float(PAIRFOLD_ROUTE_MODULUS)
        profile["token_column"] = float(token_column)
        profile["batched_sparse_bsgs"] = float(bool(batched_sparse_bsgs))
        profile["shared_baby_rotations"] = float(
            bool(shared_baby_rotations) and bool(batched_sparse_bsgs)
        )

    outputs: tuple[object, ...] = ()
    apply_start = time.perf_counter()
    try:
        if bool(batched_sparse_bsgs) and bool(shared_baby_rotations):
            child_profiles = [{} for _ in specs]
            try:
                outputs = _apply_sparse_routes_batched_shared(
                    carrier_cipher,
                    transforms=tuple(
                        (spec.routes, spec.plan, f"decode.pairfold_route.{spec.label}")
                        for spec in specs
                    ),
                    slots=int(layout.slots),
                    crypto_context=crypto_context,
                    batch_cap=32,
                    profiles=child_profiles,
                )
                if profile is not None:
                    profile["shared_fallbacks"] = 0.0
                    profile["shared_group_count"] = 1.0
            except Exception as error:
                if not bool(fallback_on_error):
                    raise
                warnings.warn(
                    "decode.pairfold_route.kv: shared batched sparse BSGS "
                    f"failed; using per-transform fallback ({type(error).__name__}: "
                    f"{error})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if profile is not None:
                    profile["shared_fallbacks"] = 1.0
                    profile["shared_group_count"] = 0.0
                sequential: list[object] = []
                try:
                    for spec in specs:
                        sequential.append(
                            _apply_one_transform(
                                carrier_cipher,
                                spec=spec,
                                layout=layout,
                                crypto_context=crypto_context,
                                max_baby_offsets=int(max_baby_offsets),
                                batched_sparse_bsgs=bool(batched_sparse_bsgs),
                                fallback_on_error=bool(fallback_on_error),
                                profile=None,
                            )
                        )
                    outputs = tuple(sequential)
                    sequential = []
                finally:
                    _release_outputs(sequential, borrowed=carrier_cipher)
        else:
            sequential = []
            try:
                for spec in specs:
                    child_profile: dict[str, float] = {}
                    sequential.append(
                        _apply_one_transform(
                            carrier_cipher,
                            spec=spec,
                            layout=layout,
                            crypto_context=crypto_context,
                            max_baby_offsets=int(max_baby_offsets),
                            batched_sparse_bsgs=bool(batched_sparse_bsgs),
                            fallback_on_error=bool(fallback_on_error),
                            profile=child_profile,
                        )
                    )
                outputs = tuple(sequential)
                sequential = []
            finally:
                _release_outputs(sequential, borrowed=carrier_cipher)

        _validate_outputs(outputs, expected=4, borrowed=carrier_cipher)
        result = PairFoldTokenWrite(
            key_groups=outputs[:2],
            value_groups=outputs[2:],
            column=int(token_column),
        )
        outputs = ()
        if profile is not None:
            profile["apply"] = time.perf_counter() - apply_start
            profile["wall"] = time.perf_counter() - wall_start
        return result
    finally:
        _release_outputs(outputs, borrowed=carrier_cipher)


def pairfold_query_from_carrier_fhe(
    carrier_cipher,
    *,
    carrier_layout: LinearCarrierLayout,
    layout: PairFold128Layout,
    crypto_context,
    group: int,
    query_offset: int,
    position: int,
    rope_theta: float = 500000.0,
    max_baby_offsets: int = 32,
    batched_sparse_bsgs: bool = True,
    fallback_on_error: bool = True,
    profile: dict[str, float] | None = None,
):
    """Borrow one carrier and return one owned streaming PairFold Q cipher."""

    _validate_layouts(carrier_layout, layout)
    if carrier_cipher is None:
        raise ValueError("carrier_cipher must not be None.")
    wall_start = time.perf_counter()
    build_start = time.perf_counter()
    phase = _rope_phase(
        layout=layout, position=int(position), rope_theta=float(rope_theta)
    )
    spec = _query_transform(
        layout=layout,
        group=int(group),
        query_offset=int(query_offset),
        phase=phase,
    )
    build_seconds = time.perf_counter() - build_start
    scheduler_profile: dict[str, float] = {}
    if profile is not None:
        basis = spec.plan.baby_offsets
        profile.clear()
        profile["route_build_and_plan"] = build_seconds
        profile["logical_pt_ct"] = float(len(spec.routes))
        profile["baby_rotations"] = float(len(basis) - int(0 in basis))
        profile["giant_rotations"] = float(
            len(spec.plan.giant_offsets) - int(0 in spec.plan.giant_offsets)
        )
        profile["output_cipher_count"] = 1.0
        profile["route_modulus"] = float(PAIRFOLD_ROUTE_MODULUS)
        profile["group"] = float(group)
        profile["query_offset"] = float(query_offset)
        profile["batched_sparse_bsgs"] = float(bool(batched_sparse_bsgs))
    output = _apply_one_transform(
        carrier_cipher,
        spec=spec,
        layout=layout,
        crypto_context=crypto_context,
        max_baby_offsets=int(max_baby_offsets),
        batched_sparse_bsgs=bool(batched_sparse_bsgs),
        fallback_on_error=bool(fallback_on_error),
        profile=scheduler_profile,
    )
    if profile is not None:
        profile["apply"] = float(scheduler_profile.get("wall_seconds", 0.0))
        profile["wall"] = time.perf_counter() - wall_start
        for name, value in scheduler_profile.items():
            if name != "wall_seconds":
                profile[f"scheduler.{name}"] = float(value)
    return output


def pairfold_queries_from_carrier_fhe(
    carrier_cipher,
    *,
    carrier_layout: LinearCarrierLayout,
    layout: PairFold128Layout,
    crypto_context,
    streams: Iterable[tuple[int, int]],
    position: int,
    rope_theta: float = 500000.0,
    max_baby_offsets: int = 32,
    batched_sparse_bsgs: bool = True,
    shared_baby_rotations: bool = True,
    fallback_on_error: bool = True,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Extract several Decode Q streams while sharing one baby basis.

    The legacy streaming route invokes :func:`pairfold_query_from_carrier_fhe`
    once per ``(group, query_offset)``.  That is memory efficient, but repeats
    the same carrier baby rotations for every query stream.  This batched form
    keeps the output order supplied by ``streams`` and hoists the common baby
    basis once.  Callers may bound the batch size externally to keep the query
    ciphertext live set small.
    """

    _validate_layouts(carrier_layout, layout)
    if carrier_cipher is None:
        raise ValueError("carrier_cipher must not be None.")
    stream_values = tuple((int(group), int(offset)) for group, offset in streams)
    if not stream_values:
        return ()
    wall_start = time.perf_counter()
    phase = _rope_phase(layout=layout, position=int(position), rope_theta=float(rope_theta))
    specs = tuple(
        _query_transform(
            layout=layout,
            group=group,
            query_offset=query_offset,
            phase=phase,
        )
        for group, query_offset in stream_values
    )
    basis = _validate_basis_size(specs, max_baby_offsets=int(max_baby_offsets))
    if profile is not None:
        profile.clear()
        groups = {int(group) for group, _ in stream_values}
        profile["group"] = float(next(iter(groups))) if len(groups) == 1 else -1.0
        profile["query_stream_count"] = float(len(stream_values))
        profile["stream_count"] = float(len(specs))
        profile["shared_basis_size"] = float(len(basis))
        profile["baby_rotations"] = float(len(basis) - int(0 in basis))
        profile["giant_rotations"] = float(
            sum(len(spec.plan.giant_offsets) - int(0 in spec.plan.giant_offsets) for spec in specs)
        )
        profile["batched_sparse_bsgs"] = float(bool(batched_sparse_bsgs))
        profile["shared_baby_rotations"] = float(bool(shared_baby_rotations) and bool(batched_sparse_bsgs))

    outputs: tuple[object, ...] = ()
    try:
        if bool(batched_sparse_bsgs) and bool(shared_baby_rotations):
            child_profiles = [{} for _ in specs]
            try:
                outputs = _apply_sparse_routes_batched_shared(
                    carrier_cipher,
                    transforms=tuple(
                        (spec.routes, spec.plan, f"decode.pairfold_route.query.g{group}.r{offset}")
                        for spec, (group, offset) in zip(specs, stream_values, strict=True)
                    ),
                    slots=int(layout.slots),
                    crypto_context=crypto_context,
                    batch_cap=32,
                    profiles=child_profiles,
                )
                if profile is not None:
                    profile["shared_fallbacks"] = 0.0
            except Exception as error:
                if not bool(fallback_on_error):
                    raise
                warnings.warn(
                    "decode.pairfold_route.queries: shared batched sparse BSGS "
                    f"failed; using per-stream fallback ({type(error).__name__}: {error})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if profile is not None:
                    profile["shared_fallbacks"] = 1.0
                sequential: list[object] = []
                try:
                    for spec in specs:
                        sequential.append(
                            _apply_one_transform(
                                carrier_cipher,
                                spec=spec,
                                layout=layout,
                                crypto_context=crypto_context,
                                max_baby_offsets=int(max_baby_offsets),
                                batched_sparse_bsgs=bool(batched_sparse_bsgs),
                                fallback_on_error=bool(fallback_on_error),
                                profile=None,
                            )
                        )
                    outputs = tuple(sequential)
                    sequential = []
                finally:
                    _release_outputs(sequential, borrowed=carrier_cipher)
        else:
            sequential: list[object] = []
            try:
                for spec in specs:
                    sequential.append(
                        _apply_one_transform(
                            carrier_cipher,
                            spec=spec,
                            layout=layout,
                            crypto_context=crypto_context,
                            max_baby_offsets=int(max_baby_offsets),
                            batched_sparse_bsgs=bool(batched_sparse_bsgs),
                            fallback_on_error=bool(fallback_on_error),
                            profile=None,
                        )
                    )
                outputs = tuple(sequential)
                sequential = []
            finally:
                _release_outputs(sequential, borrowed=carrier_cipher)
        _validate_outputs(outputs, expected=len(specs), borrowed=carrier_cipher)
        result = outputs
        outputs = ()
        if profile is not None:
            profile["wall"] = float(time.perf_counter() - wall_start)
        return result
    finally:
        _release_outputs(outputs, borrowed=carrier_cipher)


def pairfold_query_group_from_carrier_fhe(
    carrier_cipher,
    *,
    carrier_layout: LinearCarrierLayout,
    layout: PairFold128Layout,
    crypto_context,
    group: int,
    query_offsets: Iterable[int] | None = None,
    position: int,
    rope_theta: float = 500000.0,
    max_baby_offsets: int = 32,
    fallback_on_error: bool = True,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Extract several Q streams of one GQA group with one shared baby basis.

    Serial Decode owns all query streams, so the four offsets of a group can
    share one hoisted rotation batch.  The returned tuple is owned by the
    caller; callers should release each query after its score/PV stream is
    consumed.  Multi-rank Decode should continue using the single-stream
    function because it only evaluates a subset of offsets.
    """

    offsets = tuple(
        range(int(layout.gqa_ratio))
        if query_offsets is None
        else (int(value) for value in query_offsets)
    )
    if not offsets:
        return ()
    if any(value < 0 or value >= int(layout.gqa_ratio) for value in offsets):
        raise ValueError(f"query offsets must be in [0, {layout.gqa_ratio}).")
    return pairfold_queries_from_carrier_fhe(
        carrier_cipher,
        carrier_layout=carrier_layout,
        layout=layout,
        crypto_context=crypto_context,
        streams=tuple((int(group), int(offset)) for offset in offsets),
        position=int(position),
        rope_theta=float(rope_theta),
        max_baby_offsets=int(max_baby_offsets),
        batched_sparse_bsgs=True,
        shared_baby_rotations=True,
        fallback_on_error=bool(fallback_on_error),
        profile=profile,
    )


__all__ = [
    "PAIRFOLD_ROUTE_MODULUS",
    "PairFoldRouteComplexity",
    "pairfold_kv_from_carrier_fhe",
    "pairfold_kv_from_carrier_numpy",
    "pairfold_query_from_carrier_fhe",
    "pairfold_queries_from_carrier_fhe",
    "pairfold_query_group_from_carrier_fhe",
    "pairfold_query_from_carrier_numpy",
    "pairfold_route_complexity",
    "pairfold_route_rotations",
]
