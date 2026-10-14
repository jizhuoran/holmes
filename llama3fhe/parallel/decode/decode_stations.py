"""One multi-card *decode* layer (T2-Decode station chain).

This is :func:`llama3fhe.graph.evaluate_decode` (``decode_packing="pairfold128"``,
``refresh_input=True``, ``skip_attention=False``) rewritten as a station chain
that a world of ``W`` ranks walks in lock step.  Every rank holds the whole
DecodeFM hidden state (one ciphertext) and the whole PairFold KV cache of the
layer, so the cheap, depth-critical stations (RMSNorm + bootstrap, KV routing,
folded W_O, residual adds) run *redundantly* on every rank and never
communicate.  Only the three heavy stations are split along the axes of
:class:`~llama3fhe.parallel.decode.decode_ownership.DecodeRankOwnership` and glued
back with a ciphertext all-reduce (:mod:`.decode_collectives`):

=================  ===========================================  ===========
station            split axis                                   collective
=================  ===========================================  ===========
``qkv_projection`` BSGS plaintext chunks (``chunk_indices``)    ``qkv`` (1 CT)
``attention``      ``(group, query)`` streams                   ``folded_o`` (1 CT, level-aligned)
``mlp``            intermediate pages / chunks                  ``mlp/down`` (+ ``mlp/gu/<page>`` with W > pages)
=================  ===========================================  ===========

CKKS addition is exact, so after each collective all ranks hold the
bit-identical ciphertext and stay in lock step without further exchange.
``world_size == 1`` (no collectives, full chunks / streams / pages) is the
single-card graph.  Bootstraps never cross a card.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from llama3fhe.operators.nonlinear.bootstrap import bootstrap_tag
from llama3fhe.backend import release_all, release_if_supported, trim_device_allocator_cache
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.pairfold_kv_cache import PairFoldLayerCache

from .decode_collectives import all_gather_ciphers, all_reduce_ciphers
from .decode_ownership import DecodeRankOwnership

TimedCallback = Callable[[str, Callable[[], Any]], Any]


@dataclass
class DecodeLayerParallelResult:
    """The layer output (identical on every rank) plus per-station accounting."""

    #: owned DecodeFM output ciphertexts (``feature_layout.cipher_count``)
    ciphers: tuple[object, ...]
    #: the (same) cache object that was passed in, with this token appended
    kv_cache: PairFoldLayerCache
    wall_seconds: float
    stage_seconds: dict[str, float]
    nested: dict[str, Any]
    #: ``{station: bootstrap calls}`` (``bootstrap_operator.calls`` deltas)
    bootstrap_calls: dict[str, int]
    native_bootstrap_calls: dict[str, int]
    #: ``{round}_sent/_received/_seconds/_rounds`` of every collective round
    comm: dict[str, Any]
    output_levels: dict[str, Any]
    ownership: DecodeRankOwnership
    _released: bool = field(default=False, init=False, repr=False)

    def release(self) -> None:
        if self._released:
            return
        release_all(self.ciphers)
        self.ciphers = ()
        self._released = True


def _default_timed(name: str, fn: Callable[[], Any]) -> Any:
    return fn()


def decode_shifted_lanes(schedule) -> bool:
    """``True`` when the decode linears run in the shifted-lane (SL) packing."""

    return bool(getattr(schedule, "decode_shifted_lanes", False))


def decode_mlp_input_limbs(schedule, silu_entry) -> int:
    """Limbs the Decode MLP consumes from its input (SiLU PS depth + suffix, SL adds 2 mask levels).

    The SiLU polynomial is per layer (Llama-3-8B standard tier: depth 6 on
    layers 0-28, 7 / 8 / 9 on layers 29 / 30 / 31), so the MLP input -- and the
    rotation keys of the ``linear`` tier (:func:`decode_linear_limb_floor`) --
    must follow the deepest layer that runs, not layer 10.
    """

    from llama3fhe.approx.chebyshev import chebyshev_ps_mul_depth

    mlp_suffix_limbs = int(os.environ.get("LLAMA_FHE_MLP_SUFFIX_LIMBS", "6"))
    if mlp_suffix_limbs < 3:
        raise ValueError("LLAMA_FHE_MLP_SUFFIX_LIMBS must be at least 3.")
    if decode_shifted_lanes(schedule):
        # the SL MLP masks lane 0 at the gate/up exit and at the down exit (+2 levels)
        mlp_suffix_limbs += 2
    return int(chebyshev_ps_mul_depth(silu_entry["coefficients"]) + mlp_suffix_limbs - int(bool(schedule.fuse_silu_domain_map)))


def decode_linear_limb_floor(schedule, approximations, layers) -> int:
    """Smallest ``linear`` rotation-key limb cap that serves the MLP input of every layer in ``layers``.

    A rotation key must hold at least as many Q limbs as the ciphertext it
    rotates; the SL lane expansion rotates the MLP input itself, so the cap is
    the largest :func:`decode_mlp_input_limbs` over the layers (16 for the
    SL chain through layer 31, 13 on layers 0-28 alone).
    """

    return max(decode_mlp_input_limbs(schedule, approximations.for_layer(int(idx)).silu) for idx in layers)


def decode_qkv_chunk_count(schedule) -> int:
    """BSGS plaintext chunks of the 4096-wide QKV projection under ``schedule``
    (the axis :meth:`DecodeRankOwnership.chunk_indices` splits).

    Lane-replicated: ``4096 / qkv_max_plaintext_rows``.  Shifted lanes: the
    512 diagonals in 64-row chunks, i.e. always 8 (``W | 8`` for every W).
    """

    from llama3fhe.graph import _decode_linear_config

    operator_config = _decode_linear_config(
        schedule, int(schedule.qkv_max_plaintext_rows), str(schedule.qkv_hoist_strategy)
    )
    plan = operator_config.plan(dimension=4096, token_lanes=8 if decode_shifted_lanes(schedule) else 1)
    return int(plan.plaintext_chunks)


def _split_kv_limb_drop(
    token,
    *,
    target_limbs: int,
    crypto_context,
    bootstrap_operator,
    bootstrap_iterations: int,
    ownership: DecodeRankOwnership,
    transport,
    comm: dict[str, Any] | None,
):
    """Multi-rank :func:`drop_pairfold_token_to_limbs`: every rank bootstraps only
    its :meth:`DecodeRankOwnership.kv_refresh_indices` of the token's ``K0, K1,
    V0, V1`` and the four results are all-gathered (one round, tag ``kv_refresh``).

    Same contract as the single-card helper: consumes ``token``, returns an
    owned :class:`PairFoldTokenWrite` at ``target_limbs`` whose ciphertexts are
    bit-identical on every rank.  Inputs already at ``target_limbs`` on every
    position take the single-card path (no bootstrap, no exchange).
    """

    from llama3fhe.operators.pairfold_kv_cache import (
        PAIRFOLD_GROUP_COUNT,
        PairFoldTokenWrite,
        drop_pairfold_token_to_limbs,
    )
    from llama3fhe.operators.primitives import drop_to_limbs

    inputs = tuple(token.handles)
    current = tuple(int(c.state.cur_limbs) for c in inputs)
    if all(have == int(target_limbs) for have in current):
        return drop_pairfold_token_to_limbs(token, target_limbs=int(target_limbs), crypto_context=crypto_context)
    mine = ownership.kv_refresh_indices(len(inputs))
    owned: dict[int, Any] = {}
    gathered = None
    try:
        for index in mine:
            cipher = inputs[index]
            refreshed = None
            working = cipher
            if current[index] < int(target_limbs):
                with bootstrap_tag("kv_limb_drop"):
                    refreshed = bootstrap_operator.refresh(cipher, iterations=int(bootstrap_iterations))
                working = refreshed
            try:
                owned[index] = drop_to_limbs((working,), target_limbs=int(target_limbs), crypto_context=crypto_context)[0]
            finally:
                if refreshed is not None:
                    release_if_supported(refreshed)
        gathered = all_gather_ciphers(
            transport, "kv_refresh", owned, count=len(inputs), crypto_context=crypto_context, comm=comm
        )
        owned = {}
        replacement = PairFoldTokenWrite(
            key_groups=tuple(gathered[:PAIRFOLD_GROUP_COUNT]),
            value_groups=tuple(gathered[PAIRFOLD_GROUP_COUNT:]),
            column=token.column,
        )
        gathered = None
    except BaseException:
        release_all(list(owned.values()))
        if gathered is not None:
            release_all(gathered)
        raise
    token.release()
    return replacement


def decode_layer_parallel(
    *,
    crypto_context,
    transport,
    ownership: DecodeRankOwnership,
    weights,
    approximations,
    schedule,
    config,
    input_ciphers,
    kv_cache: PairFoldLayerCache,
    position: int,
    layer_idx: int,
    bootstrap_operator,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    pairfold_layout: PairFold128Layout | None = None,
    rope_theta: float = 500000.0,
    prefix_bundle=None,
    split_gate_up: bool = True,
    split_kv_refresh: bool = True,
    split_w_o: bool = True,
    hidden_stub: np.ndarray | None = None,
    timed: TimedCallback | None = None,
    comm: dict[str, Any] | None = None,
    rms_slim_period_rows: int | None = None,
) -> DecodeLayerParallelResult:
    """Walk one decode layer on this rank; see the module docstring.

    ``input_ciphers`` (the DecodeFM hidden state, identical on every rank) are
    borrowed.  ``kv_cache`` is *mutated* (this token is appended; every rank
    appends the same ciphertexts) and stays owned by the caller.  ``timed``
    is an optional ``timed(name, fn) -> fn()`` hook the bench uses to wrap
    each station (memory probes, key leases, tracer stages); the station
    names are the keys of ``stage_seconds``.  Every rank of the world must
    call this with the same ``layer_idx`` sequence: the collectives match on
    their tags.

    ``rms_slim_period_rows`` is the periodic slim pack of both RMSNorms
    (``preset.rmsnorm_slim_period_rows``; the DecodeFM stream is one cipher,
    so ``1`` is the natural period: slim message period 8 slots, zero
    extract rotations, and the rsqrt refresh runs through the log_slots=3
    sparse bootstrap program when the operator has one).  The merged-page
    Softmax likewise routes its ``filled`` / ``lambda`` refreshes through a
    sparse program of period ``attention_layout.decode_page_merge_stride``
    slots whenever the operator carries one
    (:func:`~llama3fhe.operators.attention.decode.decode_softmax.merged_page_sparse_period`).
    """

    from llama3fhe.approx.chebyshev import chebyshev_ps_mul_depth
    from llama3fhe.graph import _decode_linear_config, _linear_config
    from llama3fhe.operators.attention.pairfold.pairfold_operator import pairfold_attention_fhe
    from llama3fhe.operators.attention.pairfold.pairfold_route import (
        pairfold_kv_from_carrier_fhe,
        pairfold_query_group_from_carrier_fhe,
        pairfold_query_from_carrier_fhe,
    )
    from llama3fhe.operators.linear.shard.decode_mlp_shard import decode_mlp_shard_fhe
    from llama3fhe.operators.linear.pairfold.pairfold_output import (
        pairfold_w_o_chunk_count,
        pairfold_w_o_fhe,
        pairfold_w_o_finalize_fhe,
        pairfold_w_o_needs_refresh,
    )
    from llama3fhe.operators.linear.pairfold.pairfold_qkv import pairfold_qkv_fhe, pairfold_qkv_fold_carrier
    from llama3fhe.operators.norm.refresh import bootstrap_rmsnorm_fhe
    from llama3fhe.operators.norm.rmsnorm import residual_add
    from llama3fhe.operators.pairfold_kv_cache import drop_pairfold_token_to_limbs
    from llama3fhe.operators.primitives import drop_levels

    if bootstrap_operator is None:
        raise ValueError("decode_layer_parallel requires a bootstrap operator (RMSN1/RMSN2 Meta-BTS).")
    if approximations.softmax is None:
        raise ValueError("decode_layer_parallel requires approximations.softmax.")
    if kv_cache is None or kv_cache.is_released:
        raise ValueError("decode_layer_parallel requires a live PairFoldLayerCache.")
    world_size = int(transport.world_size)
    rank = int(transport.rank)
    if int(ownership.rank) != rank or int(ownership.world_size) != world_size:
        raise ValueError(
            f"ownership (rank {ownership.rank} of {ownership.world_size}) does not match the transport "
            f"(rank {rank} of {world_size})"
        )
    timed = _default_timed if timed is None else timed
    comm = {} if comm is None else comm
    rms_slim_period_rows = None if rms_slim_period_rows is None else int(rms_slim_period_rows)
    if rms_slim_period_rows is not None and rms_slim_period_rows <= 0:
        rms_slim_period_rows = None
    pairfold_layout = PairFold128Layout() if pairfold_layout is None else pairfold_layout
    position = int(position)
    layer_idx = int(layer_idx)
    input_ciphers = tuple(input_ciphers)
    if len(input_ciphers) != int(feature_layout.cipher_count):
        raise ValueError(f"expected {feature_layout.cipher_count} DecodeFM input ciphers, got {len(input_ciphers)}.")
    if int(feature_layout.seq_len) != 1 or int(feature_layout.hidden_dim) != int(pairfold_layout.carrier_dimension):
        raise ValueError("decode_layer_parallel requires the production DecodeFM layout (seq_len=1, hidden=4096).")
    if int(attention_layout.slots) != int(feature_layout.slots):
        raise ValueError("Decode FM and attention slot counts must match.")
    if int(kv_cache.content_tokens) >= int(pairfold_layout.seq_len) * max(1, len(kv_cache.pages)):
        # Appending opens a new page in that case; allowed, but positions must stay contiguous.
        pass
    if hidden_stub is None:
        hidden_stub = np.zeros((int(feature_layout.seq_len), int(feature_layout.hidden_dim)), dtype=np.float32)
    hidden_stub = np.asarray(hidden_stub, dtype=np.float32)

    fuse_rmsnorm_output_scale = bool(schedule.fuse_rmsnorm_output_scale)
    fuse_silu_domain_map = bool(schedule.fuse_silu_domain_map)
    silu_entry = approximations.silu
    required_mlp_input_limbs = decode_mlp_input_limbs(schedule, silu_entry)
    minimum_attention_rmsnorm_limbs = 14 + int(os.environ.get("LLAMA_FHE_ATTN_EXTRA_LIMBS", "0"))
    softmax_bootstrap_iterations = (
        schedule.residual_bootstrap_iterations
        if schedule.softmax_main_bootstrap_iterations is None
        else int(schedule.softmax_main_bootstrap_iterations)
    )
    shifted_lanes = decode_shifted_lanes(schedule)
    chunk_count = decode_qkv_chunk_count(schedule)
    qkv_chunks = None if world_size == 1 else ownership.chunk_indices(chunk_count)
    qkv_config = _decode_linear_config(schedule, schedule.qkv_max_plaintext_rows, schedule.qkv_hoist_strategy)
    mlp_config = _decode_linear_config(schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy)
    # W_O stays lane-replicated in this SL version (its PairFold carrier input is
    # not the lane-0 DecodeFM form (shifted lanes).
    w_o_config = _linear_config(schedule, schedule.output_max_plaintext_rows, schedule.output_hoist_strategy)
    w_o_chunks = (
        None
        if world_size == 1 or not split_w_o
        else ownership.chunk_indices(pairfold_w_o_chunk_count(w_o_config, layout=pairfold_layout))
    )
    query_selection = None if ownership.is_full_attention else ownership.query_streams
    shared_query_route = os.environ.get("LLAMA_FHE_DECODE_QUERY_SHARED", "1") != "0"

    wall_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    nested: dict[str, Any] = {
        "output_levels": {},
        "ownership": {
            "rank": rank,
            "world_size": world_size,
            "qkv_chunks": None if qkv_chunks is None else tuple(qkv_chunks),
            "qkv_chunk_count": chunk_count,
            "query_streams": tuple(ownership.query_streams),
            "mlp_pages": tuple(ownership.mlp_page_indices),
            "mlp_partners": tuple(ownership.mlp_partners),
            "mlp_split_gate_up": bool(split_gate_up),
            "shifted_lanes": bool(shifted_lanes),
            "shared_query_route": bool(shared_query_route and query_selection is None),
        },
    }
    bootstrap_calls: dict[str, int] = {}
    native_bootstrap_calls: dict[str, int] = {}

    def counters() -> tuple[int, int]:
        return int(bootstrap_operator.calls), int(bootstrap_operator.native_calls)

    def station(name: str, fn: Callable[[], Any]) -> Any:
        calls_before, native_before = counters()
        started = time.perf_counter()
        result = timed(name, fn)
        stage_seconds[name] = float(stage_seconds.get(name, 0.0) + time.perf_counter() - started)
        calls_after, native_after = counters()
        bootstrap_calls[name] = int(bootstrap_calls.get(name, 0) + calls_after - calls_before)
        native_bootstrap_calls[name] = int(
            native_bootstrap_calls.get(name, 0) + native_after - native_before
        )
        return result

    input_norm = qkv = attention = post_norm = mlp = None
    first_residual = final_residual = None
    carrier = None
    folded = None
    refreshed_o = None
    wo_ciphers: tuple[object, ...] = ()
    mlp_inputs: tuple[object, ...] = ()
    owns_mlp_inputs = False
    token = None
    outputs: tuple[object, ...] = ()
    success = False
    try:
        # ------------------------------------------------- 1. RMSN1 (redundant)
        def _input_rmsnorm():
            return bootstrap_rmsnorm_fhe(
                input_ciphers,
                weights.input_layernorm,
                coeffs=approximations.input_norm,
                layout=feature_layout,
                crypto_context=crypto_context,
                bootstrap_operator=bootstrap_operator,
                pair_inputs=False,
                bootstrap_iterations=int(schedule.residual_bootstrap_iterations),
                bootstrap_precision_bits=int(schedule.residual_bootstrap_precision),
                bootstrap_output_level_drop=int(schedule.residual_bootstrap_level_drop),
                minimum_rmsnorm_output_limbs=minimum_attention_rmsnorm_limbs,
                adaptive_slim_refresh=bool(schedule.adaptive_rmsnorm_slim_refresh),
                defer_output_scale=fuse_rmsnorm_output_scale,
                function_prefix=f"model.layers.{layer_idx}.input_layernorm.decode",
                slim_period_rows=rms_slim_period_rows,
            )

        input_norm = station("input_rmsnorm", _input_rmsnorm)
        nested["input_rmsnorm"] = dict(input_norm.stage_seconds)
        nested["output_levels"]["input_rmsnorm"] = input_norm.output_levels

        # ------------------------------- 2. QKV chunks of this rank + all-reduce
        def _qkv():
            return pairfold_qkv_fhe(
                hidden_stub,
                weights.q_proj,
                weights.k_proj,
                weights.v_proj,
                layout=pairfold_layout,
                config=config,
                operator_config=qkv_config,
                crypto_context=crypto_context,
                input_ciphers=input_norm.ciphers,
                verify=False,
                input_column_scale=input_norm.deferred_output_scale,
                chunk_indices=qkv_chunks,
                # SL, multi-rank: all-reduce the unfolded partial, fold once afterwards
                fold=(qkv_chunks is None) or not shifted_lanes,
            )

        qkv = station("qkv_projection", _qkv)
        nested["qkv_projection"] = dict(qkv.stage_seconds)
        input_norm.release()
        input_norm = None
        carrier_layout = qkv.output_layout

        def _qkv_all_reduce():
            partial = qkv.carrier_cipher
            qkv.carrier_cipher = None  # ownership moves into the collective
            (summed,) = all_reduce_ciphers(
                transport, "qkv", [partial], crypto_context=crypto_context, comm=comm
            )
            if shifted_lanes and qkv_chunks is not None:
                try:
                    return pairfold_qkv_fold_carrier(
                        summed, layout=pairfold_layout, operator_config=qkv_config, crypto_context=crypto_context
                    )
                finally:
                    release_if_supported(summed)
            return summed

        carrier = station("qkv_all_reduce", _qkv_all_reduce)
        qkv.release()
        qkv = None
        nested["output_levels"]["qkv_carrier"] = (int(crypto_context.level_for_cipher(carrier)),)

        # ------------------------------------------ 3. KV route + append (redundant)
        token_column = int(kv_cache.content_tokens) % int(pairfold_layout.seq_len)
        kv_route_profile: dict[str, float] = {}

        def _kv_extract():
            return pairfold_kv_from_carrier_fhe(
                carrier,
                carrier_layout=carrier_layout,
                layout=pairfold_layout,
                crypto_context=crypto_context,
                position=position,
                token_column=token_column,
                rope_theta=float(rope_theta),
                max_baby_offsets=int(schedule.sparse_max_baby_offsets),
                profile=kv_route_profile,
            )

        token = station("kv_extract", _kv_extract)
        nested["pairfold_kv_extract"] = dict(kv_route_profile)
        if schedule.kv_limbs is not None:

            def _kv_limb_drop():
                if ownership.world_size == 1 or not split_kv_refresh:
                    return drop_pairfold_token_to_limbs(
                        token,
                        target_limbs=int(schedule.kv_limbs),
                        crypto_context=crypto_context,
                        bootstrap_operator=bootstrap_operator,
                        bootstrap_iterations=int(schedule.residual_bootstrap_iterations),
                    )
                # split: the four K0/K1/V0/V1 bootstraps are independent -> one per rank, then all-gather
                return _split_kv_limb_drop(
                    token,
                    target_limbs=int(schedule.kv_limbs),
                    crypto_context=crypto_context,
                    bootstrap_operator=bootstrap_operator,
                    bootstrap_iterations=int(schedule.residual_bootstrap_iterations),
                    ownership=ownership,
                    transport=transport,
                    comm=comm,
                )

            token = station("kv_limb_drop", _kv_limb_drop)
        else:
            stage_seconds["kv_limb_drop"] = 0.0

        def _kv_append():
            return kv_cache.append(token, crypto_context=crypto_context)

        page_index, appended_column = station("kv_append", _kv_append)
        token = None
        nested["pairfold_kv_append"] = {
            "page_index": int(page_index),
            "token_column": int(appended_column),
            "content_tokens": int(kv_cache.content_tokens),
            "page_count": int(len(kv_cache.pages)),
        }

        # ---------------------------- 4. attention streams of this rank + all-reduce
        key_pages = tuple(page.key_groups for page in kv_cache.pages)
        value_pages = tuple(page.value_groups for page in kv_cache.pages)
        occupied_pages = tuple(int(page.occupied) for page in kv_cache.pages)
        query_route_profiles: list[dict[str, float]] = []

        def extract_streaming_query(carrier_cipher, *, group: int, query_offset: int):
            route_profile: dict[str, float] = {}
            query = pairfold_query_from_carrier_fhe(
                carrier_cipher,
                carrier_layout=carrier_layout,
                layout=pairfold_layout,
                crypto_context=crypto_context,
                group=int(group),
                query_offset=int(query_offset),
                position=position,
                rope_theta=float(rope_theta),
                max_baby_offsets=int(schedule.sparse_max_baby_offsets),
                profile=route_profile,
            )
            query_route_profiles.append(route_profile)
            return query

        def extract_query_group(carrier_cipher, *, group: int, query_offsets):
            group_profile: dict[str, float] = {}
            extracted = pairfold_query_group_from_carrier_fhe(
                carrier_cipher,
                carrier_layout=carrier_layout,
                layout=pairfold_layout,
                crypto_context=crypto_context,
                group=int(group),
                query_offsets=tuple(int(offset) for offset in query_offsets),
                position=position,
                rope_theta=float(rope_theta),
                max_baby_offsets=int(schedule.sparse_max_baby_offsets),
                profile=group_profile,
            )
            query_route_profiles.append(group_profile)
            return extracted

        def _attention():
            return pairfold_attention_fhe(
                carrier,
                key_pages,
                value_pages,
                query_extractor=extract_streaming_query,
                pairfold_layout=pairfold_layout,
                score_layout=attention_layout,
                crypto_context=crypto_context,
                softmax_config=approximations.softmax,
                occupied_pages=occupied_pages,
                layer_idx=layer_idx,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=softmax_bootstrap_iterations,
                bootstrap_precision_bits=int(schedule.residual_bootstrap_precision),
                prefix_bundle=prefix_bundle,
                query_selection=query_selection,
                query_group_extractor=(
                    extract_query_group
                    if shared_query_route and query_selection is None
                    else None
                ),
            )

        attention = station("decode_attention", _attention)
        attention_stages = dict(attention.stage_seconds)
        nested["decode_attention"] = attention_stages
        nested["decode_attention_profile"] = dict(attention.profile)
        nested["pairfold_query_routes"] = tuple(query_route_profiles)
        release_if_supported(carrier)
        carrier = None

        def _folded_o_all_reduce():
            partial = attention.folded_o_cipher
            if partial is None:
                raise RuntimeError("PairFold attention returned no folded-O ciphertext.")
            attention.folded_o_cipher = None  # ownership moves into the collective
            (summed,) = all_reduce_ciphers(
                transport, "folded_o", [partial], crypto_context=crypto_context, comm=comm, aligned=True
            )
            return summed

        folded = station("folded_o_all_reduce", _folded_o_all_reduce)
        attention.release()
        attention = None
        nested["output_levels"]["folded_o"] = (int(crypto_context.level_for_cipher(folded)),)

        # ------------------------------ 5. folded W_O + residual (redundant)
        def _folded_o_refresh():
            nonlocal folded, refreshed_o
            if not pairfold_w_o_needs_refresh((folded,), crypto_context=crypto_context):
                return False
            with bootstrap_tag("folded_o"):
                refreshed_o = bootstrap_operator.refresh(
                    folded,
                    iterations=int(schedule.residual_bootstrap_iterations),
                    precision_bits=int(schedule.residual_bootstrap_precision),
                )
            if refreshed_o is None or refreshed_o is folded:
                raise RuntimeError("PairFold folded-O refresh must return a distinct owned ciphertext.")
            release_if_supported(folded)
            folded = refreshed_o
            refreshed_o = None
            return True

        refresh_applied = station("folded_o_refresh", _folded_o_refresh)
        nested["folded_o_refresh"] = {"applied": bool(refresh_applied)}
        if bool(schedule.evict_rotation_cuda_cache_before_output_projection):
            clear = getattr(crypto_context.context, "clear_cuda_rotation_cache", None)
            if callable(clear):
                clear()
        trim_device_allocator_cache(crypto_context.device)

        folded_w_o_profile: dict[str, float] = {}

        def _folded_w_o():
            # chunked over the ranks (like QKV): this rank's BSGS chunks of W_O;
            # the conjugation moves behind the all-reduce (w_o_finalize)
            return pairfold_w_o_fhe(
                (folded,),
                weights.o_proj,
                layout=pairfold_layout,
                crypto_context=crypto_context,
                operator_config=w_o_config,
                profile=folded_w_o_profile,
                chunk_indices=w_o_chunks,
                finalize=w_o_chunks is None,
            )

        wo_ciphers = tuple(station("folded_w_o", _folded_w_o))
        nested["folded_w_o"] = dict(folded_w_o_profile)
        release_if_supported(folded)
        folded = None

        def _w_o_all_reduce():
            nonlocal wo_ciphers
            if w_o_chunks is None:
                return wo_ciphers
            partials = list(wo_ciphers)
            wo_ciphers = ()  # ownership moves into the collective
            return tuple(all_reduce_ciphers(transport, "w_o", partials, crypto_context=crypto_context, comm=comm))

        wo_ciphers = tuple(station("w_o_all_reduce", _w_o_all_reduce))

        def _w_o_finalize():
            nonlocal wo_ciphers
            if w_o_chunks is None:
                return wo_ciphers
            (summed,) = wo_ciphers
            wo_ciphers = ()
            try:
                return pairfold_w_o_finalize_fhe(
                    summed, layout=pairfold_layout, crypto_context=crypto_context, profile=folded_w_o_profile
                )
            finally:
                release_if_supported(summed)

        wo_ciphers = tuple(station("w_o_finalize", _w_o_finalize))
        nested["folded_w_o"] = dict(folded_w_o_profile)

        def _attention_residual():
            return residual_add(
                input_ciphers, wo_ciphers, layout=feature_layout, crypto_context=crypto_context, verify=False
            )

        first_residual = station("attention_residual_add", _attention_residual)
        release_all(wo_ciphers)
        wo_ciphers = ()
        nested["output_levels"]["attention_residual"] = first_residual.output_levels

        # ------------------------------------------- 6. RMSN2 + MLP entry (redundant)
        def _post_rmsnorm():
            return bootstrap_rmsnorm_fhe(
                first_residual.ciphers,
                weights.post_attention_layernorm,
                coeffs=approximations.post_attention_norm,
                layout=feature_layout,
                crypto_context=crypto_context,
                bootstrap_operator=bootstrap_operator,
                pair_inputs=False,
                bootstrap_iterations=int(
                    schedule.residual_bootstrap_iterations
                    if schedule.post_attention_residual_bootstrap_iterations is None
                    else schedule.post_attention_residual_bootstrap_iterations
                ),
                bootstrap_precision_bits=int(schedule.residual_bootstrap_precision),
                bootstrap_output_level_drop=int(schedule.residual_bootstrap_level_drop),
                minimum_rmsnorm_output_limbs=required_mlp_input_limbs,
                return_refreshed_inputs=False,
                defer_output_scale=fuse_rmsnorm_output_scale,
                adaptive_slim_refresh=bool(schedule.adaptive_rmsnorm_slim_refresh),
                use_slim_program=schedule.rmsnorm2_slim_log_slots is not None,
                function_prefix=f"model.layers.{layer_idx}.post_attention_layernorm.decode",
                slim_period_rows=rms_slim_period_rows,
            )

        post_norm = station("post_attention_bootstrap_rmsnorm", _post_rmsnorm)
        nested["post_attention_bootstrap_rmsnorm"] = dict(post_norm.stage_seconds)
        nested["output_levels"]["post_attention_rmsnorm"] = post_norm.output_levels

        if bool(schedule.evict_rotation_cuda_cache_before_mlp):
            clear = getattr(crypto_context.context, "clear_cuda_rotation_cache", None)
            if callable(clear):
                clear()
        trim_device_allocator_cache(crypto_context.device)

        minimum_mlp_input_limbs = min(int(cipher.state.cur_limbs) for cipher in post_norm.ciphers)
        refreshed_mlp_input = minimum_mlp_input_limbs < required_mlp_input_limbs
        requested_mlp_input_drop = (
            int(schedule.mlp_input_level_drop) + int(fuse_rmsnorm_output_scale) + int(fuse_silu_domain_map)
        )
        maximum_safe_mlp_input_drop = max(0, minimum_mlp_input_limbs - required_mlp_input_limbs)
        # Enter the MLP at exactly ``required_mlp_input_limbs``: the multi-card linear tier
        # holds its rotation keys at ``limb_caps["linear"]`` (14 on rtx6000pro-high), and a
        # layer whose post-RMSNorm rsqrt is shallower (e.g. layer 30) exits 2 limbs higher
        # than layer 10 -> the SL lane expansion asked for a 15-limb key (8 x Pro 6000,
        # tools/autodl/37 D_all).  Extra limbs buy nothing after the RMSNorm anyway.
        scheduled_mlp_input_drop = maximum_safe_mlp_input_drop

        def _mlp_entry():
            if refreshed_mlp_input:
                raised: list[object] = []
                try:
                    for cipher in post_norm.ciphers:
                        with bootstrap_tag("mlp.entry"):
                            raised.append(
                                bootstrap_operator.refresh(
                                    cipher,
                                    iterations=int(schedule.residual_bootstrap_iterations),
                                    precision_bits=int(schedule.residual_bootstrap_precision),
                                )
                            )
                    raised_limbs = min(int(cipher.state.cur_limbs) for cipher in raised)
                    applied = max(0, raised_limbs - required_mlp_input_limbs)
                    return drop_levels(tuple(raised), applied, crypto_context=crypto_context), applied
                finally:
                    release_all(raised)
            return (
                drop_levels(post_norm.ciphers, scheduled_mlp_input_drop, crypto_context=crypto_context),
                scheduled_mlp_input_drop,
            )

        mlp_inputs, applied_mlp_input_drop = station(
            "pre_mlp_bootstrap" if refreshed_mlp_input else "mlp_input_modulus_drop", _mlp_entry
        )
        mlp_inputs = tuple(mlp_inputs)
        owns_mlp_inputs = True
        minimum_scheduled_mlp_limbs = min(int(cipher.state.cur_limbs) for cipher in mlp_inputs)
        if minimum_scheduled_mlp_limbs < required_mlp_input_limbs:
            raise RuntimeError(
                "Decode MLP input scheduling violated its consumer budget: "
                f"{minimum_scheduled_mlp_limbs} < {required_mlp_input_limbs}."
            )
        nested["mlp_input_modulus_drop"] = {
            "requested": int(requested_mlp_input_drop),
            "maximum_safe": int(maximum_safe_mlp_input_drop),
            "scheduled": int(scheduled_mlp_input_drop),
            "applied": int(applied_mlp_input_drop),
            "overridden_by_emergency_bootstrap": bool(refreshed_mlp_input),
            "before_min_limbs": int(minimum_mlp_input_limbs),
            "after_min_limbs": int(minimum_scheduled_mlp_limbs),
            "required_input_limbs": int(required_mlp_input_limbs),
        }
        nested["output_levels"]["mlp_input"] = tuple(
            sorted({int(crypto_context.level_for_cipher(c)) for c in mlp_inputs})
        )

        # ------------------------------ 7. MLP pages / chunks of this rank + all-reduce
        def _mlp():
            return decode_mlp_shard_fhe(
                hidden_stub,
                weights.gate_proj,
                weights.up_proj,
                weights.down_proj,
                silu_entry=silu_entry,
                layout=feature_layout,
                ownership=ownership,
                transport=transport,
                config=config,
                operator_config=mlp_config,
                crypto_context=crypto_context,
                input_ciphers=mlp_inputs,
                input_column_scale=post_norm.deferred_output_scale,
                fuse_silu_domain_map=fuse_silu_domain_map,
                verify=False,
                tag="mlp",
                split_gate_up=bool(split_gate_up),
            )

        mlp = station("mlp", _mlp)
        nested["mlp"] = dict(mlp.stage_seconds)
        nested["output_levels"]["mlp"] = mlp.output_levels
        for key, value in mlp.comm.items():
            if isinstance(value, float):
                comm[key] = float(comm.get(key, 0.0) + value)
            else:
                comm[key] = int(comm.get(key, 0) + int(value))
        release_all(mlp_inputs)
        mlp_inputs = ()
        owns_mlp_inputs = False
        post_norm.release()
        post_norm = None

        def _mlp_residual():
            return residual_add(
                first_residual.ciphers, mlp.ciphers, layout=feature_layout, crypto_context=crypto_context, verify=False
            )

        final_residual = station("mlp_residual", _mlp_residual)
        nested["output_levels"]["layer_output"] = final_residual.output_levels
        mlp.release()
        mlp = None
        first_residual.release()
        first_residual = None

        outputs = tuple(final_residual.ciphers)
        final_residual.ciphers = ()  # ownership moves to the result
        success = True
        return DecodeLayerParallelResult(
            ciphers=outputs,
            kv_cache=kv_cache,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds=dict(stage_seconds),
            nested=nested,
            bootstrap_calls=dict(bootstrap_calls),
            native_bootstrap_calls=dict(native_bootstrap_calls),
            comm=dict(comm),
            output_levels=dict(nested["output_levels"]),
            ownership=ownership,
        )
    finally:
        for owner in (input_norm, qkv, attention, post_norm, mlp, first_residual):
            if owner is not None:
                owner.release()
        if final_residual is not None and not success:
            final_residual.release()
        release_if_supported(carrier)
        release_if_supported(folded)
        release_if_supported(refreshed_o)
        if token is not None:
            token.release()
        release_all(wo_ciphers)
        if owns_mlp_inputs:
            release_all(mlp_inputs)
        if not success:
            release_all(outputs)


__all__ = ["DecodeLayerParallelResult", "decode_layer_parallel", "decode_qkv_chunk_count", "decode_shifted_lanes"]
