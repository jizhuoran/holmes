"""Source-to-head sharding primitives for the Prefill attention adapter.

The production QKV carrier boundary is token-sharded: sixteen ciphertexts
hold eight tokens each, and every ciphertext contains all Q/K/V heads.  A
correct multi-GPU Adapter therefore cannot run one independent two-cipher
fixture per card.  This module splits the existing linear Adapter at its
first source-local boundary:

* :func:`attention_input_lane_shard_fhe` applies lane/phase to selected
  global source indices;
* callers explicitly redistribute those sixteen lane ciphertexts; and
* :func:`attention_input_target_shard_fhe` accumulates all sources for only
  the Q/K/V head targets assigned to the local card.

Across all ranks this executes every production target once.  Communication
is deliberately left to the scheduler so benchmarks can report it outside
the matched operator-compute wall.

T2 phase-1 split: the V carrier
is built as *own-source partials* -- every rank routes only the two lane
sources it produced (``value_source_indices``) and returns the pre-split
complex partial (``value_return_partial``).  The partials of one carrier are
summed on their consumers and finished with
:func:`attention_input_value_finish_fhe`.  Because the compact V carrier
holds every foreign source block as zeros, the restricted transform is
exactly the full transform restricted to those sources, so the eight-way
sum reproduces the monolithic output up to rescale rounding.
:func:`attention_input_shard_rotations` reports the rotation keys each phase
(Q-only, K-only, own-source V) needs so the key lifecycle can lease them
separately.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import profile_sync_device as sync_device  # stage-timer sync, no-op under T2_STAGE_SYNC=0
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

from ...linear import split_complex_ciphers_twice
from ...transform import apply_transform_by_level
from ..prefill import input as core
from ..config import AttentionOperatorConfig


@dataclass
class AttentionInputLaneShardResult:
    source_indices: tuple[int, ...]
    lane_ciphers: tuple[object, ...]
    wall_seconds: float
    stage_seconds: dict[str, float]

    def release(self) -> None:
        owned = self.lane_ciphers
        self.lane_ciphers = ()
        for cipher in owned:
            release_if_supported(cipher)


@dataclass
class AttentionInputTargetShardResult:
    query_items: tuple[tuple[int, object], ...]
    key_items: tuple[tuple[int, object], ...]
    value_items: tuple[tuple[int, object], ...]
    wall_seconds: float
    stage_seconds: dict[str, float]
    scheduler_stats: dict[str, float]
    #: ``(carrier, cipher)`` pre-split complex V partials when the shard ran
    #: with ``value_return_partial=True`` (then ``value_items`` is empty).
    value_partials: tuple[tuple[int, object], ...] = ()
    #: ``emit_cache_c=True``: ``(key_pair, cipher)`` folded ``K_C`` snapshots
    #: of the own key pairs (the graph's emitted-K_C contract, see
    #: :func:`~llama3fhe.operators.attention.prefill.input.attention_input_adapter_fhe`)
    #: and ``(carrier, cipher)`` complex compact ``V_C`` partials over the
    #: shard's value sources (finish them with
    #: :func:`attention_input_value_finish_fhe` after the exchange).
    key_c_items: tuple[tuple[int, object], ...] = ()
    value_c_partials: tuple[tuple[int, object], ...] = ()

    @property
    def ciphers(self) -> tuple[object, ...]:
        return tuple(
            cipher
            for _, cipher in self.query_items
            + self.key_items
            + self.value_items
            + self.value_partials
            + self.key_c_items
            + self.value_c_partials
        )

    def release(self) -> None:
        owned = self.ciphers
        self.query_items = ()
        self.key_items = ()
        self.value_items = ()
        self.value_partials = ()
        self.key_c_items = ()
        self.value_c_partials = ()
        for cipher in owned:
            release_if_supported(cipher)


def _checked_indices(
    values, *, upper: int, name: str, allow_empty: bool = True
) -> tuple[int, ...]:
    indices = tuple(int(value) for value in values)
    if not allow_empty and not indices:
        raise ValueError(f"{name} must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must be unique")
    if any(index < 0 or index >= int(upper) for index in indices):
        raise IndexError(f"{name} contains an out-of-range index")
    return indices


def attention_input_lane_shard_fhe(
    carrier_ciphers,
    *,
    source_indices,
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    cos,
    sin,
    function_prefix: str,
) -> AttentionInputLaneShardResult:
    """Run exactly the source-local lane/phase work for one rank."""

    ciphers = tuple(carrier_ciphers)
    params = core._layout_parameters(carrier_layout, attention_layout)
    sources = _checked_indices(
        source_indices,
        upper=int(params["source_count"]),
        name="source_indices",
        allow_empty=False,
    )
    if len(ciphers) != len(sources):
        raise ValueError(
            "carrier_ciphers and source_indices must have the same length"
        )
    started = time.perf_counter()
    outputs: tuple[object, ...] = ()
    try:
        outputs = core._apply_lane_phase(
            ciphers,
            params=params,
            cos=cos,
            sin=sin,
            crypto_context=crypto_context,
            function_prefix=str(function_prefix),
            source_indices=sources,
        )
        sync_device(crypto_context.device)
        wall = float(time.perf_counter() - started)
        result = AttentionInputLaneShardResult(
            source_indices=sources,
            lane_ciphers=outputs,
            wall_seconds=wall,
            stage_seconds={"lane_phase": wall},
        )
        outputs = ()
        return result
    finally:
        for cipher in outputs:
            release_if_supported(cipher)


def attention_input_target_shard_fhe(
    lane_ciphers,
    *,
    query_pair_indices,
    key_pair_indices,
    value_carrier_indices,
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    function_prefix: str,
    value_source_indices=None,
    value_return_partial: bool = False,
    emit_cache_c: bool = False,
) -> AttentionInputTargetShardResult:
    """Finish selected head targets after the lane-cipher all-gather.

    ``lane_ciphers`` must contain all sixteen global sources in source order.
    Q and K indices name output pair ciphertexts.  A V carrier names the
    complex compact carrier that splits into V pair indices ``2*i`` and
    ``2*i+1``.

    ``value_source_indices`` restricts the V path to those lane sources (the
    other lane slots may then be ``None`` when no Q/K target is requested);
    with ``value_return_partial`` the V path stops before the Re/Im split and
    returns the rescaled complex partial per carrier in ``value_partials``
    (see :func:`attention_input_value_finish_fhe`).

    ``emit_cache_c`` additionally materialises the Decode-cache snapshots the
    monolithic adapter emits under ``operator_config.emit_cache_c``: folded
    ``K_C`` of every own key pair (``key_c_items``; local, from the same
    sheared-axis sources as the Delta K) and the compact ``V_C`` partial of
    every own carrier over ``value_source_indices`` (``value_c_partials``;
    same exchange / finish as the Delta V partials).  The Prefill Delta path
    is unchanged.
    """

    # Keep the production API surface aligned with the monolithic Adapter.
    # This shard currently has no model-config-dependent branch.
    del model_config
    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    ciphers = tuple(lane_ciphers)
    params = core._layout_parameters(carrier_layout, attention_layout)
    if len(ciphers) != int(params["source_count"]):
        raise ValueError(
            "target shard requires all lane sources in global source order"
        )
    q_indices = _checked_indices(
        query_pair_indices,
        upper=int(params["query_pairs"]),
        name="query_pair_indices",
    )
    k_indices = _checked_indices(
        key_pair_indices,
        upper=int(params["kv_pairs"]),
        name="key_pair_indices",
    )
    v_carriers = _checked_indices(
        value_carrier_indices,
        upper=int(params["value_carriers"]),
        name="value_carrier_indices",
    )
    if not q_indices and not k_indices and not v_carriers:
        raise ValueError("target shard must own at least one output target")
    v_sources: tuple[int, ...] | None = None
    if value_source_indices is not None:
        v_sources = _checked_indices(
            value_source_indices,
            upper=int(params["source_count"]),
            name="value_source_indices",
            allow_empty=False,
        )
    if (q_indices or k_indices) and any(cipher is None for cipher in ciphers):
        raise ValueError("Q/K targets need every lane source cipher")
    if v_carriers and v_sources is None and any(cipher is None for cipher in ciphers):
        raise ValueError("a full V carrier needs every lane source cipher")
    value_return_partial = bool(value_return_partial)

    sparse_max_baby_offsets = int(operator_config.sparse_max_baby_offsets)
    batched_sparse_bsgs = bool(operator_config.batched_sparse_input_bsgs)
    sparse_batch_cap = int(operator_config.sparse_input_batch_cap)
    sparse_scheduler_fallback = bool(operator_config.sparse_input_fallback)
    qk_level_drop = int(
        os.environ.get(
            "LLAMA_FHE_QK_LEVEL_DROP",
            str(int(operator_config.qk_score_level_drop)),
        )
    )
    lower_routes = core._key_shear_routes(params, upper=False)
    upper_routes = core._key_shear_routes(params, upper=True)
    lower_plan = core.choose_sparse_diagonal_plan(
        tuple(lower_routes),
        slots=int(params["slots"]),
        max_baby_offsets=sparse_max_baby_offsets,
    )
    upper_plan = core.choose_sparse_diagonal_plan(
        tuple(upper_routes),
        slots=int(params["slots"]),
        max_baby_offsets=sparse_max_baby_offsets,
    )
    value_routes = core._value_delta_routes(params, sources=v_sources)
    value_plan = core.choose_sparse_diagonal_plan(
        tuple(value_routes),
        slots=int(params["slots"]),
        max_baby_offsets=sparse_max_baby_offsets,
    )
    emit_cache_c = bool(emit_cache_c)
    value_c_routes: dict = {}
    value_c_plan = None
    if emit_cache_c and v_carriers:
        value_c_routes = core._value_c_routes(params, sources=v_sources)
        value_c_plan = core.choose_sparse_diagonal_plan(
            tuple(value_c_routes),
            slots=int(params["slots"]),
            max_baby_offsets=sparse_max_baby_offsets,
        )
    axis2_masks, axis2_stride = core._qk_axis_masks(params, stage=2)

    wall_started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    sparse_stages = (2 if k_indices else 0) + (1 if v_carriers else 0)
    scheduler_stats: dict[str, float] = {
        "enabled": float(batched_sparse_bsgs),
        "batch_cap": float(sparse_batch_cap),
        "fallback_allowed": float(sparse_scheduler_fallback),
        "fallbacks": 0.0,
        "stage_count": float(sparse_stages),
        "batched_stage_count": 0.0,
    }
    qk_axis1: tuple[object, ...] = ()
    qk_low: tuple[object, ...] = ()
    q_full: tuple[object, ...] = ()
    q_outputs: tuple[object, ...] = ()
    k_lower: tuple[object, ...] = ()
    k_upper_sources: tuple[object, ...] = ()
    k_upper: tuple[object, ...] = ()
    k_combined: tuple[object, ...] = ()
    k_outputs: tuple[object, ...] = ()
    v_compact: tuple[object, ...] = ()
    v_complex: tuple[object, ...] = ()
    v_split: tuple[object, ...] = ()
    v_outputs: tuple[object, ...] = ()
    v_partials: tuple[object, ...] = ()
    k_c_outputs: tuple[object, ...] = ()
    v_c_partials: tuple[object, ...] = ()
    success = False
    try:
        if q_indices or k_indices:
            started = time.perf_counter()
            targets = q_indices + tuple(
                int(params["query_pairs"]) + index for index in k_indices
            )
            qk_axis1 = core._apply_qk_fused_gather_axis1(
                ciphers,
                params=params,
                crypto_context=crypto_context,
                function_prefix=f"{function_prefix}.qk_gather_axis1",
                target_indices=targets,
            )
            sync_device(crypto_context.device)
            stage_seconds["qk_gather_axis1"] = float(
                time.perf_counter() - started
            )

            if qk_level_drop > 0:
                started = time.perf_counter()
                dropped: list[object] = []
                try:
                    for cipher in qk_axis1:
                        target_limbs = int(cipher.state.cur_limbs) - qk_level_drop
                        if target_limbs <= 1:
                            raise ValueError(
                                "LLAMA_FHE_QK_LEVEL_DROP leaves too few limbs"
                            )
                        target_state = cipher.state.replace(
                            cur_limbs=target_limbs,
                            scale_degree=1,
                            scaling_factor=None,
                        )
                        dropped.append(
                            fhe.align_to(
                                cipher, target_state, crypto_context.context
                            )
                        )
                except Exception:
                    for cipher in dropped:
                        release_if_supported(cipher)
                    raise
                for cipher in qk_axis1:
                    release_if_supported(cipher)
                qk_axis1 = tuple(dropped)
                sync_device(crypto_context.device)
                stage_seconds["qk_score_level_drop"] = float(
                    time.perf_counter() - started
                )

            started = time.perf_counter()
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
            stage_seconds["qk_axis2"] = float(time.perf_counter() - started)
            for name, seconds in axis_profile.items():
                stage_seconds[f"qk_axis2.{name}"] = float(seconds)

            started = time.perf_counter()
            expanded: list[object] = []
            try:
                for cipher in qk_low[: len(q_indices)]:
                    expanded.append(
                        core._expand_c_feature_half(
                            cipher,
                            upper_sign=1,
                            attention_layout=attention_layout,
                            crypto_context=crypto_context,
                        )
                    )
                q_full = tuple(expanded)
                expanded = []
            finally:
                for cipher in expanded:
                    release_if_supported(cipher)
            q_outputs = core._repeat_outputs(
                q_full,
                layout=attention_layout,
                crypto_context=crypto_context,
            )
            q_full = ()

            k_sources = qk_low[len(q_indices) :]
            if k_sources and emit_cache_c:
                cache_k_start = time.perf_counter()
                k_c_expanded: list[object] = []
                try:
                    for cipher in k_sources:
                        k_c_expanded.append(
                            core._expand_c_feature_half(
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
                    k_c_outputs = core._repeat_outputs(
                        k_c_full, layout=attention_layout, crypto_context=crypto_context
                    )
                finally:
                    for cipher in k_c_full:
                        release_if_supported(cipher)
                sync_device(crypto_context.device)
                stage_seconds["key_c_expand"] = float(time.perf_counter() - cache_k_start)
            if k_sources:
                lower_profile: dict[str, float] = {}
                k_lower = core._apply_sparse_routes_selected(
                    k_sources,
                    routes=lower_routes,
                    plan=lower_plan,
                    slots=int(params["slots"]),
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
                k_upper = core._apply_sparse_routes_selected(
                    k_upper_sources,
                    routes=upper_routes,
                    plan=upper_plan,
                    slots=int(params["slots"]),
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
                k_outputs = core._repeat_outputs(
                    k_combined,
                    layout=attention_layout,
                    crypto_context=crypto_context,
                )
                k_combined = ()
                for label, profile in (
                    ("key_shear_lower", lower_profile),
                    ("key_shear_upper", upper_profile),
                ):
                    core._merge_scheduler_profile(
                        profile,
                        label=label,
                        stage_seconds=stage_seconds,
                        scheduler_stats=scheduler_stats,
                    )
            for cipher in qk_low:
                release_if_supported(cipher)
            qk_low = ()
            sync_device(crypto_context.device)
            stage_seconds["expand_and_key_shear"] = float(
                time.perf_counter() - started
            )

        if v_carriers:
            started = time.perf_counter()
            v_compact = core._apply_value_block_gather(
                ciphers,
                params=params,
                crypto_context=crypto_context,
                function_prefix=f"{function_prefix}.value_gather",
                target_indices=v_carriers,
                source_indices=v_sources,
            )
            sync_device(crypto_context.device)
            stage_seconds["value_gather"] = float(
                time.perf_counter() - started
            )

            shared_value_routes = False
            value_c_profile: dict[str, float] = {}
            value_profile: dict[str, float] = {}
            if emit_cache_c:
                started = time.perf_counter()
                if batched_sparse_bsgs:
                    shared_value_routes = True
                    v_c_partials, v_complex = core._apply_sparse_routes_shared_selected(
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
                        slots=int(params["slots"]),
                        crypto_context=crypto_context,
                        use_batched=True,
                        batch_cap=sparse_batch_cap,
                        fallback_on_error=sparse_scheduler_fallback,
                        profiles=[value_c_profile, value_profile],
                    )
                else:
                    v_c_partials = core._apply_sparse_routes_selected(
                        v_compact,
                        routes=value_c_routes,
                        plan=value_c_plan,
                        slots=int(params["slots"]),
                        crypto_context=crypto_context,
                        function_prefix=f"{function_prefix}.value_c",
                        padded_plaintext_rows=None,
                        use_batched=False,
                        batch_cap=sparse_batch_cap,
                        fallback_on_error=sparse_scheduler_fallback,
                        profile=value_c_profile,
                    )
                sync_device(crypto_context.device)
                stage_seconds["value_c"] = float(
                    value_c_profile.get("wall_seconds", 0.0)
                    if shared_value_routes
                    else time.perf_counter() - started
                )
                core._merge_scheduler_profile(
                    value_c_profile,
                    label="value_c",
                    stage_seconds=stage_seconds,
                    scheduler_stats=scheduler_stats,
                )

            started = time.perf_counter()
            if not shared_value_routes:
                v_complex = core._apply_sparse_routes_selected(
                    v_compact,
                    routes=value_routes,
                    plan=value_plan,
                    slots=int(params["slots"]),
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
            if value_return_partial:
                v_partials = v_complex
                v_complex = ()
            else:
                split_started = time.perf_counter()
                v_split = split_complex_ciphers_twice(
                    v_complex, crypto_context=crypto_context
                )
                v_outputs = core._repeat_outputs(
                    v_split,
                    layout=attention_layout,
                    crypto_context=crypto_context,
                )
                v_split = ()
                for cipher in v_complex:
                    release_if_supported(cipher)
                v_complex = ()
            sync_device(crypto_context.device)
            if shared_value_routes:
                split_seconds = 0.0 if value_return_partial else time.perf_counter() - split_started
                stage_seconds["value_delta_and_split"] = float(
                    value_profile.get("wall_seconds", 0.0) + split_seconds
                )
            else:
                stage_seconds["value_delta_and_split"] = float(
                    time.perf_counter() - started
                )
            core._merge_scheduler_profile(
                value_profile,
                label="value_delta",
                stage_seconds=stage_seconds,
                scheduler_stats=scheduler_stats,
            )

        value_indices = (
            ()
            if value_return_partial
            else tuple(
                output_index
                for carrier in v_carriers
                for output_index in (2 * carrier, 2 * carrier + 1)
            )
        )
        success = True
        return AttentionInputTargetShardResult(
            query_items=tuple(zip(q_indices, q_outputs, strict=True)),
            key_items=tuple(zip(k_indices, k_outputs, strict=True)),
            value_items=tuple(zip(value_indices, v_outputs, strict=True)),
            value_partials=(
                tuple(zip(v_carriers, v_partials, strict=True))
                if value_return_partial
                else ()
            ),
            key_c_items=(
                tuple(zip(k_indices, k_c_outputs, strict=True)) if emit_cache_c else ()
            ),
            value_c_partials=(
                tuple(zip(v_carriers, v_c_partials, strict=True)) if emit_cache_c else ()
            ),
            wall_seconds=float(time.perf_counter() - wall_started),
            stage_seconds={
                str(name): float(seconds)
                for name, seconds in stage_seconds.items()
            },
            scheduler_stats={
                str(name): float(value)
                for name, value in scheduler_stats.items()
            },
        )
    finally:
        for group in (
            qk_axis1,
            qk_low,
            q_full,
            k_lower,
            k_upper_sources,
            k_upper,
            k_combined,
            v_compact,
            v_complex,
            v_split,
        ):
            for cipher in group:
                release_if_supported(cipher)
        if not success:
            for cipher in q_outputs + k_outputs + v_outputs + v_partials + k_c_outputs + v_c_partials:
                release_if_supported(cipher)


def attention_input_value_finish_fhe(
    partials,
    *,
    attention_layout: AttentionPairLayout,
    crypto_context,
) -> tuple[tuple[int, object], ...]:
    """Sum the own-source V partials of each carrier and finish the V pairs.

    ``partials`` is an iterable of ``(carrier, cipher)`` -- typically the
    eight ranks' ``value_partials`` for the carriers this rank consumes.  All
    partials of one carrier must share level and scale (they come from the
    same pipeline on identical lane levels).  Returns ``value_items`` in the
    :class:`AttentionInputTargetShardResult` convention: pair indices
    ``2*carrier`` (``2*Re``) and ``2*carrier + 1`` (``2*Im``), repeated to
    the full slot count.  Inputs are not consumed.
    """

    by_carrier: dict[int, list[object]] = {}
    for carrier, cipher in partials:
        by_carrier.setdefault(int(carrier), []).append(cipher)
    if not by_carrier:
        raise ValueError("value finish needs at least one partial")
    sums: list[object] = []
    split: tuple[object, ...] = ()
    outputs: tuple[object, ...] = ()
    carriers = tuple(sorted(by_carrier))
    try:
        for carrier in carriers:
            group = by_carrier[carrier]
            if len(group) == 1:
                # ``a + (a - a)`` is a fresh, owned copy without a deep_copy API.
                zero = crypto_context.fhe.homo_sub(group[0], group[0], crypto_context.context)
                try:
                    total = crypto_context.fhe.homo_add(group[0], zero, crypto_context.context)
                finally:
                    release_if_supported(zero)
            else:
                total = crypto_context.fhe.homo_add(group[0], group[1], crypto_context.context)
                for cipher in group[2:]:
                    merged = crypto_context.fhe.homo_add(total, cipher, crypto_context.context)
                    release_if_supported(total)
                    total = merged
            sums.append(total)
        split = split_complex_ciphers_twice(tuple(sums), crypto_context=crypto_context)
        outputs = core._repeat_outputs(
            split, layout=attention_layout, crypto_context=crypto_context
        )
        split = ()
        indices = tuple(
            index for carrier in carriers for index in (2 * carrier, 2 * carrier + 1)
        )
        result = tuple(zip(indices, outputs, strict=True))
        outputs = ()
        return result
    finally:
        for cipher in sums:
            release_if_supported(cipher)
        for cipher in split:
            release_if_supported(cipher)
        for cipher in outputs:
            release_if_supported(cipher)


def attention_input_shard_rotations(
    carrier_layout: LinearCarrierLayout,
    attention_layout: AttentionPairLayout,
    *,
    query_pair_indices,
    key_pair_indices,
    value_carrier_indices,
    value_source_indices,
    sparse_max_baby_offsets: int = 32,
    shear_baby_steps: int = 16,
    log_n: int = 16,
    emit_cache_c: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Signed rotation offsets per attention-input phase of one T2 rank.

    Same convention as :func:`~.input.complex_token_attention_rotations`
    (signed slot offsets, conjugation as ``M - 1``); the union of the four
    phases over all ranks is a subset of that monolithic set.

    ``"lane"``  lane/phase of the own sources (source independent);
    ``"query"`` Q-only target shard over all sixteen lanes;
    ``"key"``   K-only target shard (duplicated on every consumer rank);
    ``"value"`` own-source V partial (``value_source_indices`` x carriers)
    plus the Re/Im split and repeat used by the finish step.

    ``emit_cache_c`` adds what :func:`attention_input_target_shard_fhe` with
    ``emit_cache_c=True`` needs on top: the ``K_C`` feature-half expand on the
    key phase and the compact ``V_C`` gather plan of the own sources on the
    value phase (the Decode-cache snapshots of the multi-card Prefill).
    """

    if int(shear_baby_steps) != 16:
        raise ValueError(
            "the selected hybrid adapter currently requires shear_baby_steps=16."
        )
    params = core._layout_parameters(carrier_layout, attention_layout)
    slots = int(params["slots"])
    q_indices = _checked_indices(
        query_pair_indices, upper=int(params["query_pairs"]), name="query_pair_indices"
    )
    k_indices = _checked_indices(
        key_pair_indices, upper=int(params["kv_pairs"]), name="key_pair_indices"
    )
    v_carriers = _checked_indices(
        value_carrier_indices, upper=int(params["value_carriers"]), name="value_carrier_indices"
    )
    v_sources = _checked_indices(
        value_source_indices, upper=int(params["source_count"]), name="value_source_indices"
    )
    cap = int(sparse_max_baby_offsets)
    conjugate = (1 << (int(log_n) + 1)) - 1
    repeat: set[int] = set()
    step = int(attention_layout.payload_slots)
    while step < int(attention_layout.slots):
        repeat.add(-step)
        step *= 2
    stride = int(core._qk_axis_masks(params, stage=2)[1])
    axis2 = {-15 * stride, 16 * stride} | {index * stride for index in range(1, 16)}

    def gather(targets: set[int]) -> set[int]:
        found: set[int] = set()
        if not targets:
            return found
        for source in range(int(params["source_count"])):
            _, metadata = core._qk_fused_gather_axis1_masks(params, source=source)
            found.update(int(offset) for target, _, offset in metadata if target in targets)
        return found

    def plan_keys(routes) -> set[int]:
        if not routes:
            return set()
        plan = core.choose_sparse_diagonal_plan(tuple(routes), slots=slots, max_baby_offsets=cap)
        return set(plan.baby_offsets) | set(plan.giant_offsets)

    lane = set(range(-(int(params["token_lanes"]) - 1), int(params["token_lanes"])))

    query: set[int] = set()
    if q_indices:
        query = gather(set(q_indices)) | axis2 | {-int(attention_layout.payload_slots) // 2, conjugate} | repeat

    key: set[int] = set()
    if k_indices:
        key = gather({int(params["query_pairs"]) + index for index in k_indices}) | axis2 | {conjugate} | repeat
        key |= plan_keys(core._key_shear_routes(params, upper=False))
        key |= plan_keys(core._key_shear_routes(params, upper=True))
        if emit_cache_c:
            # _expand_c_feature_half: conjugate, multiply by i, shift by half a payload
            key |= {-int(attention_layout.payload_slots) // 2}

    value: set[int] = set()
    if v_carriers and v_sources:
        for source in v_sources:
            value.update(
                core._canonical_rotation(route[2], slots)
                for route in core._block_gather_routes(params, source)
                if route[0] == "v" and route[1] in v_carriers
            )
        value |= plan_keys(core._value_delta_routes(params, sources=v_sources))
        if emit_cache_c:
            value |= plan_keys(core._value_c_routes(params, sources=v_sources))
        value |= {conjugate} | repeat

    def finish(values: set[int]) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in values if int(value)))

    return {
        "lane": finish(lane),
        "query": finish(query),
        "key": finish(key),
        "value": finish(value),
    }


__all__ = [
    "AttentionInputLaneShardResult",
    "AttentionInputTargetShardResult",
    "attention_input_lane_shard_fhe",
    "attention_input_shard_rotations",
    "attention_input_target_shard_fhe",
    "attention_input_value_finish_fhe",
]
