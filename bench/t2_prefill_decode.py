#!/usr/bin/env python3
"""T2 end-to-end Prefill -> Decode on an N-card host (one process per rank).

The multi-card Prefill station chain (``bench/t2_model.py``) emits the Decode
KV cache of every layer while it runs (``attention_input_exchange(...,
emit_kv_cache=True)`` + :func:`~llama3fhe.parallel.prefill.kv_emit.prefill_kv_cache_station`),
rank 0 applies the client-side LM head to the decrypted hidden state and feeds
the greedy token back as the DecodeFM input of the multi-card Decode chain
(``decode_layer_parallel``, ``bench/t2_decode.py``) for ``--decode-tokens``
steps.  One worker state serves both phases: the key plan is the per-tier
union of the Prefill plan (``emit_kv_cache=True``) and the Decode plan, and
the bootstrap operator carries every sparse program the two phases need
(Prefill period-16 slim RMSNorm + periodic Softmax slim, Decode period-1 slim
RMSNorm + merged-page Softmax).

Gold greedy continuation of Paris128 (``hidden_0`` of ``fixtures/model``):
``12366, 13, 578, 6864`` (Prefill token + 3 Decode tokens).

Usage (8 x Pro 6000, from the repository root)::

    python bench/t2_prefill_decode.py --preset a100-high --world-size 8 --transport pinned \\
        --fixtures fixtures/model --model-dir $MODEL_DIR --layers 0-31 --decode-tokens 3 \\
        --sl --sl-baby 16 --warm-keys --output artifacts/t2_prefill_decode/run1

Plaintext Prefix (the cache must contain the selected layers)::

    python bench/t2_prefill_decode.py --preset a100-high --world-size 8 --transport pinned \\
        --fixtures fixtures/model --model-dir $MODEL_DIR --layers 0-31 --decode-tokens 3 \\
        --prefix-kv-cache fixtures/model/prefix2_kv.npz --output artifacts/t2_prefix2/run1

Smoke test on one card (fixtures hold layers 0-5 only; the LM head of a
truncated model is meaningless, ``--verify`` checks every layer against the
poly chain instead)::

    python bench/t2_prefill_decode.py --emulate-single-device --preset h100-emu --layers 0-1 \\
        --decode-tokens 1 --fixtures fixtures/model --model-dir $MODEL_DIR \\
        --poly-kv archive/experiments/artifacts/paris128_prefill_kv_poly.npz --verify
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

from bench.t2_decode import STATION_TIERS as DECODE_STATION_TIERS  # noqa: E402
from bench.t2_decode import ROPE_THETA, _BootstrapTimer  # noqa: E402
from bench.t2_layer import (  # noqa: E402
    PROCESS_TRANSPORTS,
    STATIONS,
    _compare,
    _decrypt_rows,
    _median,
    _release_ciphers,
    add_runtime_knob_args,
    apply_runtime_knobs,
    attn_in_overlap_requested,
    rmsnorm_coeffs_table,
    rmsnorm_exit_kwargs,
    softmax_coeffs_table,
    worker_tiers,
)
from bench.t2_model import WeightPrefetcher, parse_layers  # noqa: E402
from llama3fhe.parallel.launch import expand_devices, spawn_workers  # noqa: E402
from llama3fhe.parallel.presets import preset_by_name  # noqa: E402

GOLD_TOKENS = (12366, 13, 578, 6864)
PREFILL_STATIONS = STATIONS[:4] + ("kv_emit",) + STATIONS[4:]


# ------------------------------------------------------------------ key plans
def decode_rows_for_world(rows: int, world_size: int) -> int:
    """Largest chunk row count <= ``rows`` with ``world_size | 4096 / rows`` (the Decode diagonal split)."""

    rows = int(rows)
    while rows > 1 and (4096 // rows) % int(world_size):
        rows //= 2
    return rows


def decode_sparse_log_slots(preset) -> tuple[int, ...]:
    """Sparse bootstrap programs the Decode chain needs: period-1 slim RMSNorm + merged-page Softmax."""

    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.parallel.worker import production_layouts, slim_sparse_log_slots

    feature_layout, _attention = production_layouts()
    ks: set[int] = set()
    slim = slim_sparse_log_slots(replace(preset, rmsnorm_slim_period_rows=1), feature_layout)
    if slim is not None:
        ks.add(int(slim))
    stride = int(AttentionPairLayout().decode_page_merge_stride)
    ks.add(stride.bit_length() - 1)
    return tuple(sorted(ks))


def combined_preset(preset):
    """``preset`` with the Decode sparse programs added to ``bootstrap_extra_sparse_log_slots``."""

    extra = set(int(k) for k in preset.bootstrap_extra_sparse_log_slots) | set(decode_sparse_log_slots(preset))
    return replace(preset, bootstrap_extra_sparse_log_slots=tuple(sorted(extra)))


def decode_preset_and_schedule(preset, *, kv_limbs: int, shifted_lanes: bool, sl_baby: int | None):
    """The Decode-side preset copy (clamped chunk rows) and its LayerSchedule."""

    rows = decode_rows_for_world(int(preset.linear_max_plaintext_rows), int(preset.world_size))
    decode_preset = replace(preset, linear_max_plaintext_rows=rows)
    schedule = replace(decode_preset.layer_schedule(), kv_limbs=int(kv_limbs))
    if shifted_lanes:
        baby = int(schedule.decode_sl_baby_steps if sl_baby is None else sl_baby)
        schedule = replace(schedule, decode_shifted_lanes=True, decode_sl_baby_steps=baby)
    return decode_preset, schedule


def merge_key_plans(*plans):
    """Per-tier union of several :class:`~llama3fhe.parallel.prefill.keys.KeyPlan` objects.

    A tier present in several plans gets the union of their rotations, the
    lowest residency level and the largest limb cap (``None`` -- full chain --
    wins, a full-chain key serves every lower limb count).
    """

    from llama3fhe.parallel.prefill.keys import KeyPlan, KeyTier

    rotations: dict[str, set[int]] = collections.defaultdict(set)
    levels: dict[str, int] = {}
    caps: dict[str, int | None] = {}
    resident_extra: set[str] = set()
    first = plans[0]
    for plan in plans:
        resident_extra.update(plan.resident_extra)
        for tier in plan.tiers:
            rotations[tier.name].update(int(v) for v in tier.rotations)
            levels[tier.name] = min(levels.get(tier.name, tier.level), tier.level)
            if tier.name in caps:
                old = caps[tier.name]
                caps[tier.name] = None if (old is None or tier.limb_cap is None) else max(int(old), int(tier.limb_cap))
            else:
                caps[tier.name] = tier.limb_cap
    tiers = tuple(KeyTier(name, levels[name], tuple(sorted(rotations[name])), caps[name]) for name in rotations)
    return KeyPlan(tiers=tiers, policy=str(first.policy), ring_dim=int(first.ring_dim),
                   full_limbs=int(first.full_limbs), resident_extra=tuple(sorted(resident_extra)))


def _prefill_key_preset(preset):
    """Remove Decode-only sparse bootstrap programs from the Prefill plan."""

    decode_slots = set(int(v) for v in decode_sparse_log_slots(preset))
    extras = tuple(sorted(int(v) for v in preset.bootstrap_extra_sparse_log_slots
                          if int(v) not in decode_slots))
    return replace(preset, bootstrap_extra_sparse_log_slots=extras)


def rank_phase_key_plans(
    preset,
    rank: int,
    *,
    decode_preset,
    decode_schedule,
    layers,
    attention_limb_cap: int | None = None,
    prefix_k: int = 0,
):
    """Return ``(merged, sizes, prefill, decode)`` plans for one rank.

    The merged plan remains the context/lifecycle compatibility plan.  The
    phase plans are used for selective warm-up, so :func:`rank_key_plan` keeps
    its original two-item return shape for existing callers.
    """

    from llama3fhe.parallel.decode.decode_keys import build_decode_key_plan
    from llama3fhe.parallel.prefill.keys import build_key_plan
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts
    from llama3fhe.parallel.worker import production_layouts

    feature_layout, attention_layout = production_layouts()
    world = ownership_from_layouts(feature_layout, attention_layout, world_size=int(preset.world_size))
    # Keep the legacy Prefill plan for the merged context and summary sizes;
    # the stripped copy is used only for phase-specific warm-up.
    prefill = build_key_plan(
        preset, feature_layout, attention_layout, tiers=worker_tiers(STATIONS),
        schedule=preset.layer_schedule(),
        limb_caps=dict(preset.limb_caps), resident_extra=tuple(preset.key_resident_extra),
        policy=str(preset.key_policy), ownership=world[int(rank)], emit_kv_cache=True,
        prefix_k=int(prefix_k),
    )
    prefill_phase = build_key_plan(
        _prefill_key_preset(preset), feature_layout, attention_layout, tiers=worker_tiers(STATIONS),
        schedule=preset.layer_schedule(),
        limb_caps=dict(preset.limb_caps), resident_extra=tuple(preset.key_resident_extra),
        policy=str(preset.key_policy), ownership=world[int(rank)], emit_kv_cache=True,
        prefix_k=int(prefix_k),
    )
    from bench.t2_decode import decode_layouts, decode_linear_floor, resolve_attention_limb_cap

    decode_feature, decode_attention, pairfold_layout = decode_layouts()
    decode = build_decode_key_plan(
        decode_preset, schedule=decode_schedule, feature_layout=decode_feature, attention_layout=decode_attention,
        pairfold_layout=pairfold_layout, attention_limb_cap=resolve_attention_limb_cap(decode_preset, attention_limb_cap),
        resident_extra=tuple(preset.key_resident_extra),
        linear_limb_floor=decode_linear_floor(decode_schedule, layers),
    )
    merged = merge_key_plans(prefill, decode)
    sizes = {"prefill_union": len(prefill.union), "decode_union": len(decode.union)}
    return merged, sizes, prefill_phase, decode


def rank_key_plan(preset, rank: int, *, decode_preset, decode_schedule, layers,
                  attention_limb_cap: int | None = None, prefix_k: int = 0):
    """Prefill + Decode key plan of ``rank`` with the legacy two-item result."""

    merged, sizes, _prefill, _decode = rank_phase_key_plans(
        preset, rank, decode_preset=decode_preset, decode_schedule=decode_schedule, layers=layers,
        attention_limb_cap=attention_limb_cap, prefix_k=prefix_k,
    )
    return merged, sizes


# ------------------------------------------------------------------ worker
def worker_entry(rank: int, preset, endpoints, *, fixtures: str, layers: tuple[int, ...], decode_tokens: int,
                 input_level: int, decode_input_level: int, kv_limbs: int, model_dir: str | None,
                 poly_kv: str | None, prefix_kv_cache: str | None, prefix_k: int,
                 prefix_model_metadata: dict[str, object] | None,
                 verify: bool, verify_every: int, warm_keys: bool, shifted_lanes: bool,
                 sl_baby: int | None, split_gate_up: bool, split_kv_refresh: bool, split_w_o: bool,
                 kv_bootstrap_iterations: int, prefetch_weights: bool = True, stage_weights: bool = True,
                 attention_limb_cap: int | None = None, emulation=None):
    """One rank: Prefill chain with KV emission, then ``decode_tokens`` Decode steps on the emitted caches."""

    import numpy as np

    import easyfhe

    from llama3fhe.application.approx_config import ApproximationConfig
    from llama3fhe.application.args import DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
    from llama3fhe.approx.attention import rope_cos_sin
    from llama3fhe.approx.prefix_cache_io import load_prefix_kv_cache
    from llama3fhe.approx.rmsnorm import get_layer_coeffs
    from llama3fhe.approx.silu import load_silu_poly_coeffs
    from llama3fhe.approx.softmax import SoftmaxPolyConfig
    from llama3fhe.backend import release_all
    from llama3fhe.graph import LayerWeights as GraphLayerWeights
    from llama3fhe.graph import _linear_config
    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.operators.linear.weight import DEVICE_WEIGHTS
    from llama3fhe.parallel.decode.decode_collectives import broadcast_ciphers
    from llama3fhe.parallel.decode.decode_ownership import DecodeRankOwnership
    from llama3fhe.parallel.decode.decode_stations import decode_layer_parallel
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.prefill.kv_emit import (
        prefill_kv_cache_begin,
        prefill_kv_cache_station,
        prefill_kv_cache_stream_begin,
    )
    from llama3fhe.parallel.prefill.stations import (
        QKV_INPUT_LIMBS,
        attention_core_operator_config,
        attention_core_station,
        attention_input_exchange,
        mlp_input_limbs,
        mlp_station,
        pairfm_exchange,
        qkv_station,
        residual_station,
        rmsnorm_station,
        w_o_station,
    )
    from llama3fhe.parallel.prefill.transport import build_transport
    from llama3fhe.parallel.worker import build_worker_state, load_layer_config, production_layouts
    from bench.t2_decode import decode_layouts

    layers = tuple(int(v) for v in layers)
    emulated = emulation is not None
    prefix_bundles = (
        load_prefix_kv_cache(
            prefix_kv_cache,
            layout=production_layouts()[1],
            expected_layers=layers,
            expected_prefix_k=int(prefix_k),
            expected_rope_theta=ROPE_THETA,
            expected_model_dir=model_dir,
            expected_model_metadata=prefix_model_metadata,
        )
        if prefix_kv_cache is not None
        else None
    )
    if (prefix_bundles is None) != (int(prefix_k) == 0):
        raise ValueError("prefix_k must match the optional Prefix KV cache")
    prefix_by_layer = (
        {} if prefix_bundles is None else dict(zip(layers, prefix_bundles, strict=True))
    )
    decode_preset, decode_sched = decode_preset_and_schedule(preset, kv_limbs=kv_limbs, shifted_lanes=shifted_lanes,
                                                             sl_baby=sl_baby)
    plan, plan_sizes, prefill_plan, decode_plan = rank_phase_key_plans(
        preset, rank, decode_preset=decode_preset, decode_schedule=decode_sched, layers=layers,
        attention_limb_cap=attention_limb_cap, prefix_k=int(prefix_k),
    )
    if emulated:
        from llama3fhe.parallel.memory import MemoryProbe

        transport = emulation.transport
        state = replace(emulation.shared_state, rank=int(rank), transport=transport, memory=MemoryProbe(rank=int(rank)))
        state.memory.reset_peak()
    else:
        transport = build_transport(rank, endpoints)
        state = build_worker_state(rank, preset, transport=transport, tiers=plan.names, trace_keys=True, key_plan=plan)
    probe, context, layout, attention_layout = state.memory, state.context, state.feature_layout, state.attention_layout
    ownership, world, native, life, tracer = state.ownership, state.world, state.native, state.keys, state.tracer
    bootstrap = _BootstrapTimer(state.bootstrap_operator, shared=emulated)
    world_size = int(transport.world_size)
    config = load_layer_config(preset)
    schedule = preset.layer_schedule()
    qkv_config = _linear_config(schedule, schedule.qkv_max_plaintext_rows, schedule.qkv_hoist_strategy)
    mlp_config = _linear_config(schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy)
    w_o_config = _linear_config(schedule, schedule.output_max_plaintext_rows, schedule.output_hoist_strategy)
    decode_feature, decode_attention, pairfold_layout = decode_layouts()
    decode_ownership = DecodeRankOwnership(int(rank), world_size)
    stage_prefix = f"r{rank}." if emulated else ""

    rms_all, rms_table_info = rmsnorm_coeffs_table(DEFAULT_RMS, preset)
    silu_all = load_silu_poly_coeffs(DEFAULT_SILU)["per_layer"]
    softmax_table, softmax_table_info = softmax_coeffs_table(DEFAULT_SOFTMAX, preset)
    softmax_config = SoftmaxPolyConfig.from_artifact(softmax_table)
    approximations = ApproximationConfig(rmsnorm_path=DEFAULT_RMS, silu_path=DEFAULT_SILU, softmax_path=DEFAULT_SOFTMAX)
    approximations._rmsnorm = rms_all
    approximations._softmax = softmax_config
    hidden_dim = int(layout.hidden_dim)
    rows = slice(int(ownership.token_start), int(ownership.token_stop))
    fixtures_dir = Path(fixtures)

    with np.load(fixtures_dir / "hidden_trajectory.npz") as data:
        hidden_in = np.asarray(data[f"hidden_{layers[0]}"], dtype=np.float32)
        oracle_hidden = (
            {idx: np.asarray(data[f"hidden_{idx + 1}"], dtype=np.float32) for idx in layers}
            if not prefix_k else {}
        )
    packed = layout.pack(hidden_in)
    poly_kv_arrays = None
    if poly_kv:
        with np.load(poly_kv) as payload:
            poly_keys = np.asarray(payload["keys"], dtype=np.float32)
            poly_values = np.asarray(payload["values"], dtype=np.float32)
        poly_kv_arrays = {idx: (np.array(poly_keys[idx]), np.array(poly_values[idx])) for idx in layers}

    fuse_silu = bool(schedule.fuse_silu_domain_map)
    carrier_layout = LinearCarrierLayout(seq_len=int(layout.seq_len), dimension=hidden_dim, slots=int(layout.slots))
    cos, sin = rope_cos_sin(seq_len=int(attention_layout.seq_len), head_dim=int(attention_layout.head_dim),
                            theta=float(AttentionOperatorConfig().rope_theta), start_pos=int(prefix_k))
    operator_config = AttentionOperatorConfig(
        sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
        shear_baby_steps=int(schedule.shear_baby_steps),
        pv_complex_delta_folding=bool(schedule.pv_complex_delta_folding),
        pv_prefold_after_softmax=bool(schedule.pv_prefold_after_softmax),
        batched_sparse_input_bsgs=bool(schedule.batched_sparse_attention_input),
        qk_complex_feature_folding=True, qk_key_features_pre_folded=True,
        qk_query_features_pre_scaled=True, qk_query_features_pre_folded=True,
        pairfm_refresh_bootstrap_iterations=int(preset.pairfm_refresh_bootstrap_iterations),
        softmax_slim_periodic=bool(getattr(preset, "softmax_slim_periodic", False)),
    )
    stage_device = str(context.device) if (stage_weights and str(context.device).startswith("cuda")) else None
    prefetcher = WeightPrefetcher(fixtures_dir, layers, enabled=bool(prefetch_weights), device=stage_device)
    hidden_shape_only = np.zeros((int(layout.seq_len), hidden_dim), dtype=np.float32)

    def _stage_dict(values) -> dict[str, float]:
        return {str(k): round(float(v), 4) for k, v in dict(values or {}).items()}

    def timed_station(stations: dict, stage_name: str, name: str, tier: str | None, fn, *, barrier_tag: str | None = None):
        """wall / BS count / BS seconds / peak allocation of one station (the ``--lite`` profile of t2_model)."""

        if barrier_tag is not None:
            transport.barrier(barrier_tag)
        probe.reset_peak()
        calls_before = bootstrap.calls
        bootstrap.take()
        started = time.perf_counter()
        with tracer.stage(stage_name):
            if tier is None:
                value = fn()
            else:
                with life.lease(tier, prefetch=False):
                    value = fn()
        easyfhe.cuda.current_stream().synchronize()
        wall = time.perf_counter() - started
        bs_records = bootstrap.take()
        stats = tracer.stages[stage_name].as_row()
        stations[name] = {
            "wall_s": round(wall, 4), "tier": tier,
            "bootstrap_calls": bootstrap.calls - calls_before,
            "bootstrap_seconds": round(sum(r[0] for r in bs_records), 4),
            "peak_alloc_gib": round(float(easyfhe.cuda.max_memory_allocated()) / float(2**30), 3),
            "keys": {k: stats.get(k) for k in ("requests", "misses", "gib_loaded", "load_seconds")},
        }
        return value

    # ------------------------------------------------------------ Prefill
    caches: dict[int, object] = {}

    def run_prefill_layer(idx: int, inputs, weights, *, verify_layer: bool):
        stations: dict[str, dict] = {}
        tag_prefix = f"model/{idx}"
        prefix = f"t2pd.r{rank}.l{idx}"
        coeffs_attn = get_layer_coeffs(rms_all, "attn", idx)
        coeffs_ffn = get_layer_coeffs(rms_all, "ffn", idx)
        rms_exit = rmsnorm_exit_kwargs(idx, coeffs_attn, preset)
        rms_exit_post = rmsnorm_exit_kwargs(idx, coeffs_ffn, preset)
        silu_entry = silu_all[str(idx)]["tiers"]["standard"]
        mlp_entry = mlp_input_limbs(silu_entry, fuse_silu_domain_map=fuse_silu)
        core_config = attention_core_operator_config(schedule, layer_idx=idx, base=operator_config)
        gamma_in, gamma_post = weights["input_layernorm"], weights["post_attention_layernorm"]
        q_w, k_w, v_w, o_w = weights["q_proj"], weights["k_proj"], weights["v_proj"], weights["o_proj"]
        gate_w, up_w, down_w = weights["gate_proj"], weights["up_proj"], weights["down_proj"]
        weights.register()

        def timed(name, tier, fn, *, barrier=False):
            return timed_station(stations, f"{stage_prefix}p_{name}_{idx}", name, tier, fn,
                                 barrier_tag=f"{tag_prefix}/before/{name}" if barrier else None)

        keep: dict[str, object] = {}
        outputs: tuple = ()
        kv_result = None
        cache_owned = False
        kv_emit_pending = None
        kv_emit_stream_pending = None
        kv_emit_overlap_requested = os.environ.get("T2_PREFILL_KV_EMIT_OVERLAP", "0").lower() in {
            "1", "true", "yes", "on"
        }
        kv_emit_stream_requested = os.environ.get("T2_PREFILL_KV_EMIT_STREAM", "0").lower() in {
            "1", "true", "yes", "on"
        }
        # A side stream may share rotation-key state only when the merged plan
        # keeps every tier resident.  Leased plans would evict keys while the
        # worker is still using them, so they stay on the synchronous path.
        kv_emit_stream_allowed = bool(
            kv_emit_stream_requested
            and str(getattr(getattr(life, "plan", None), "policy", "")) == "resident"
            and str(getattr(context, "device", "")).startswith("cuda")
        )
        kv_emit_overlap = bool(kv_emit_overlap_requested or kv_emit_stream_allowed)
        layer_started = time.perf_counter()
        try:
            rms = timed("rmsnorm_in", None, lambda: rmsnorm_station(
                inputs, gamma_in, coeffs=coeffs_attn, ownership=ownership, layout=layout, context=context,
                bootstrap_operator=bootstrap, keep_residual=True, **rms_exit, function_prefix=f"{prefix}.rms_in"))
            keep["rms"] = rms
            release_all(inputs)
            inputs = ()
            qkv = timed("qkv", "linear", lambda: qkv_station(
                rms.ciphers, hidden_shape_only, q_w, k_w, v_w, ownership=ownership, layout=layout,
                attention_layout=attention_layout, context=context, config=config, operator_config=qkv_config,
                input_column_scale=rms.linear_input_column_scale(), input_limbs=QKV_INPUT_LIMBS,
                function_prefix=f"{prefix}.qkv"))
            keep["qkv"] = qkv
            _release_ciphers(rms.ciphers)
            rms.ciphers = ()
            attn_in = timed("attn_input", None, lambda: attention_input_exchange(
                qkv.carrier_ciphers, ownership=ownership, world=world, transport=transport,
                carrier_layout=carrier_layout, attention_layout=attention_layout, context=context, cos=cos, sin=sin,
                model_config=config, operator_config=operator_config, keys=life, prefetch_keys=False,
                overlap=attn_in_overlap_requested(preset), emit_kv_cache=True,
                function_prefix=f"{prefix}.attn_in", tag=f"{tag_prefix}/attn_in"), barrier=True)
            keep["attn_in"] = attn_in
            stations["attn_input"]["comm_gib"] = {k: round(int(v) / float(2**30), 4) for k, v in attn_in.comm_bytes.items()}
            qkv.release()
            if kv_emit_overlap:
                key_c_items, value_c_items = attn_in.take_kv_cache_items()
                kv_emit_pending = prefill_kv_cache_begin(
                    key_c_items, value_c_items,
                    ownership=ownership, world=world, transport=transport,
                    attention_layout=attention_layout, context=context,
                    pairfold_layout=pairfold_layout,
                    tag=f"{tag_prefix}/kv_emit",
                )
                if kv_emit_stream_allowed:
                    kv_emit_stream_pending = prefill_kv_cache_stream_begin(
                        kv_emit_pending,
                        keys=life,
                        ownership=ownership,
                        world=world,
                        transport=transport,
                        attention_layout=attention_layout,
                        context=context,
                        bootstrap_operator=bootstrap,
                        kv_limbs=int(kv_limbs),
                        bootstrap_iterations=int(kv_bootstrap_iterations),
                        pairfold_layout=pairfold_layout,
                        tag=f"{tag_prefix}/kv_emit",
                    )
                    if kv_emit_stream_pending is not None:
                        # The stream worker now owns the raw pending handle.
                        kv_emit_pending = None
            core = timed("attn_core", None, lambda: attention_core_station(
                attn_in.query_items, attn_in.key_items, attn_in.value_items, ownership=ownership,
                attention_layout=attention_layout, context=context, softmax_config=softmax_config,
                operator_config=core_config, bootstrap_operator=bootstrap, layer_idx=idx,
                keys=life, prefetch_keys=False, prefix_bundle=prefix_by_layer.get(idx),
                function_prefix=f"{prefix}.core"))
            keep["core"] = core
            # The synchronous path detaches snapshots only after attention core.
            if not kv_emit_overlap:
                key_c_items, value_c_items = attn_in.take_kv_cache_items()
            attn_in.release()
            if kv_emit_stream_pending is not None:
                # Conversion/refresh runs on the side stream while Attention
                # Core is active.  Join that event before PairFM so its key
                # work never races the shared resident bootstrap program.
                kv_emit_stream_pending.wait_conversion()
            if kv_emit_stream_pending is None and kv_emit_overlap:
                # The raw route was issued before Attention Core.  The
                # transport allows only one collective in flight, so wait via
                # the station before PairFM starts its next all-to-all.
                kv_result = timed("kv_emit", None, lambda: prefill_kv_cache_station(
                    (), (), ownership=ownership, world=world, transport=transport,
                    attention_layout=attention_layout, context=context,
                    bootstrap_operator=bootstrap,
                    kv_limbs=int(kv_limbs), bootstrap_iterations=int(kv_bootstrap_iterations),
                    pairfold_layout=pairfold_layout, tag=f"{tag_prefix}/kv_emit",
                    raw_pending=kv_emit_pending), barrier=False)
                stations["kv_emit"]["overlapped"] = True
            elif kv_emit_stream_pending is None:
                kv_result = timed("kv_emit", None, lambda: prefill_kv_cache_station(
                    key_c_items, value_c_items, ownership=ownership, world=world, transport=transport,
                    attention_layout=attention_layout, context=context, bootstrap_operator=bootstrap,
                    kv_limbs=int(kv_limbs), bootstrap_iterations=int(kv_bootstrap_iterations),
                    pairfold_layout=pairfold_layout, tag=f"{tag_prefix}/kv_emit"), barrier=True)
            pairfm_after_exchange = (
                kv_emit_stream_pending.allow_final_gather
                if kv_emit_stream_pending is not None else None
            )
            pf = timed("pairfm", None, lambda: pairfm_exchange(
                core.output_items, ownership=ownership, world=world, transport=transport, feature_layout=layout,
                attention_layout=attention_layout, context=context, model_config=config, operator_config=core_config,
                bootstrap_operator=bootstrap, keys=life, prefetch_keys=False,
                function_prefix=f"{prefix}.pairfm", tag=f"{tag_prefix}/pairfm",
                after_exchange=pairfm_after_exchange), barrier=True)
            keep["pf"] = pf
            # The side worker was gated before its final all-gather.  Its
            # transport round now runs while PairFM reduces received partials
            # locally and while the rest of this layer performs W_O, residual,
            # RMSNorm, and MLP.  The cache is not consumed until the next
            # Decode phase, so joining here would serialize independent local
            # work behind the gather.
            core.release()
            wo = timed("w_o", "linear", lambda: w_o_station(
                pf.carriers, o_w, ownership=ownership, layout=layout, attention_layout=attention_layout,
                context=context, operator_config=w_o_config))
            keep["wo"] = tuple(wo)
            pf.release()
            res1 = timed("residual_attn", None, lambda: residual_station(
                rms.residual_inputs, wo, ownership=ownership, layout=layout, context=context))
            keep["res1"] = res1
            rms.release()
            _release_ciphers(keep.pop("wo"))
            rms2 = timed("rmsnorm_post", None, lambda: rmsnorm_station(
                res1.ciphers, gamma_post, coeffs=coeffs_ffn, ownership=ownership, layout=layout, context=context,
                bootstrap_operator=bootstrap, keep_residual=True, **rms_exit_post, function_prefix=f"{prefix}.rms_post"))
            keep["rms2"] = rms2
            res1.release()
            mlp = timed("mlp", "linear", lambda: mlp_station(
                rms2.ciphers, hidden_shape_only, gate_w, up_w, down_w, silu_entry=silu_entry, ownership=ownership,
                layout=layout, context=context, config=config, operator_config=mlp_config,
                input_column_scale=rms2.linear_input_column_scale(), fuse_silu_domain_map=fuse_silu,
                input_limbs=mlp_entry, feature_shard_k=int(preset.mlp_feature_shard_k), transport=transport,
                function_prefix=f"{prefix}.mlp"))
            keep["mlp"] = mlp
            _release_ciphers(rms2.ciphers)
            rms2.ciphers = ()
            res2 = timed("residual_mlp", None, lambda: residual_station(
                rms2.residual_inputs, mlp.ciphers, ownership=ownership, layout=layout, context=context))
            outputs = tuple(res2.ciphers)
            res2.ciphers = ()
            keep["res2"] = res2
            rms2.release()
            mlp.release()

            # Join the resident side stream only after all local post-attention
            # work has been queued.  For the synchronous/fallback paths this
            # is already a completed result; keeping the bookkeeping here
            # gives both paths the same cache ownership boundary.
            if kv_emit_stream_pending is not None:
                kv_result = timed("kv_emit", None, lambda: kv_emit_stream_pending.finish(), barrier=False)
                stations["kv_emit"]["overlapped"] = True
                stations["kv_emit"]["stream"] = "resident-side"
                stations["kv_emit"]["gather_after_pairfm"] = True
                stations["kv_emit"]["background_wall_s"] = round(
                    float(kv_emit_stream_pending.box.get("result_wall_s", 0.0)), 4
                )
            stations["kv_emit"].update({
                "stream_requested": bool(kv_emit_stream_requested),
                "stream_fallback": bool(kv_emit_stream_requested and kv_emit_stream_pending is None),
                "stage_seconds": _stage_dict(kv_result.stage_seconds),
                "comm_gib": {k: round(int(v) / float(2**30), 4) for k, v in kv_result.comm_bytes.items()},
                "comm_seconds": round(kv_result.comm_seconds, 4),
                "contributed": {k: list(v) for k, v in kv_result.contributed.items()},
                "refreshed": list(kv_result.refreshed), "limbs": kv_result.limbs,
            })
            caches[idx] = kv_result.take_cache()
            cache_owned = True
        finally:
            # If Attention Core or a later local stage fails after the
            # split-phase raw route was issued, drain that one in-flight
            # collective before the worker can enter another layer.  The
            # received raw handles are never needed on this error path.
            if kv_emit_pending is not None and not kv_emit_pending._waited:
                try:
                    pending_raw, pending_map, _ = kv_emit_pending.wait_raw()
                    _release_ciphers(pending_raw)
                    _release_ciphers(tuple(pending_map.values()))
                except BaseException:
                    pass
            if kv_emit_stream_pending is not None and not kv_emit_stream_pending._finished:
                try:
                    kv_emit_stream_pending.cancel()
                except BaseException:
                    pass
            if kv_result is not None and not cache_owned:
                release = getattr(kv_result, "release", None)
                if callable(release):
                    try:
                        release()
                    except BaseException:
                        pass
            for value in keep.values():
                release = getattr(value, "release", None)
                if callable(release):
                    release()
                elif isinstance(value, (tuple, list)):
                    _release_ciphers(value)
            if inputs:
                release_all(inputs)
            weights.unregister()
        easyfhe.cuda.current_stream().synchronize()
        record = {
            "layer": int(idx),
            "layer_wall_s": round(time.perf_counter() - layer_started, 4),
            "station_wall_sum_s": round(sum(v["wall_s"] for v in stations.values()), 4),
            "bootstrap_calls": sum(v["bootstrap_calls"] for v in stations.values()),
            "bootstrap_seconds": round(sum(v["bootstrap_seconds"] for v in stations.values()), 4),
            "kv_emit_wall_s": stations["kv_emit"]["wall_s"],
            "output_limbs": [int(c.state.cur_limbs) for c in outputs],
            "cache_limbs": sorted({int(c.state.cur_limbs) for page in caches[idx].pages
                                   for c in (*page.key_groups, *page.value_groups)}),
            "cache_tokens": int(caches[idx].content_tokens),
            "stations": stations,
            "verified": False,
        }
        if verify_layer:
            started = time.perf_counter()
            got = _decrypt_rows(context, layout, ownership, outputs)
            record["oracle"] = _compare(got, oracle_hidden[idx][rows])
            record["verify_seconds"] = round(time.perf_counter() - started, 3)
            record["verified"] = True
        return outputs, record

    inputs = tuple(context.encrypt(packed[index], level=int(input_level), slots=int(layout.slots))
                   for index in ownership.cipher_indices)
    easyfhe.cuda.synchronize()
    prefetcher.start(layers[0])
    first_weights = prefetcher.get(layers[0])

    def warm_phase(target_plan) -> dict:
        """Warm only the rotations needed by one phase of the combined run."""

        started = time.perf_counter()
        per_tier = {}
        for tier_name in target_plan.names:
            tier = target_plan.tier(tier_name)
            count, seconds = life.prefetch(
                tier_name, limbs=tier.limb_cap, rotations=tier.rotations,
            )
            per_tier[str(tier_name)] = {"keys": int(count), "seconds": round(float(seconds), 3)}
        easyfhe.cuda.synchronize()
        return {
            "seconds": round(time.perf_counter() - started, 3),
            "tiers": per_tier,
            "resident_keys_gib": native_cache_inventory(native)["total_gib"],
        }

    warm_keys_report: dict = {"enabled": bool(warm_keys), "phases": {}}
    if warm_keys and not emulated:
        prefill_warm = warm_phase(prefill_plan)
        warm_keys_report["phases"]["prefill"] = prefill_warm
        warm_keys_report.update({
            "prefill_seconds": prefill_warm["seconds"],
            "tiers": prefill_warm["tiers"],
            "seconds": prefill_warm["seconds"],
            "resident_keys_gib": prefill_warm["resident_keys_gib"],
            "resident_keys_gib_after_prefill": prefill_warm["resident_keys_gib"],
        })
    transport.barrier("prefill/start")
    prefill_started = time.perf_counter()
    prefill_records = []
    last = layers[-1]
    for position, idx in enumerate(layers):
        weights = first_weights if position == 0 else prefetcher.get(idx)
        first_weights = None
        if position + 1 < len(layers):
            prefetcher.start(layers[position + 1])
        verify_layer = not prefix_k and (
            (idx == last) or (int(verify_every) > 0 and position % int(verify_every) == 0)
        )
        inputs, record = run_prefill_layer(idx, inputs, weights, verify_layer=verify_layer)
        record["weights_wait_s"] = round(prefetcher.wait_seconds.get(idx, 0.0), 3)
        weights.release()
        prefill_records.append(record)
        if rank == 0:
            print(json.dumps({"phase": "prefill", "layer": idx, "wall_s": record["layer_wall_s"],
                              "kv_emit_s": record["kv_emit_wall_s"], "bs": record["bootstrap_calls"],
                              "bs_s": record["bootstrap_seconds"],
                              **({"rel_max_err": round(record["oracle"]["rel_max_err"], 6)} if verify_layer else {})}),
                              flush=True)
    easyfhe.cuda.synchronize()
    transport.barrier("prefill/end")
    # The Decode plan is warmed only after this barrier.  Record the resident
    # key footprint here so reports can distinguish completed Prefill from the
    # merged Prefill+Decode run.
    if warm_keys and not emulated:
        resident_after_prefill = native_cache_inventory(native)["total_gib"]
        warm_keys_report["resident_keys_gib_after_prefill"] = resident_after_prefill
        warm_keys_report["phases"]["prefill"]["resident_keys_gib_after_prefill"] = resident_after_prefill
    prefill_wall = time.perf_counter() - prefill_started

    # -- client side: decrypt my rows, assemble on rank 0, LM head -> first token
    started = time.perf_counter()
    own_rows = _decrypt_rows(context, layout, ownership, inputs).astype(np.float32)
    _release_ciphers(inputs)
    inputs = ()
    gathered_rows = transport.gather_to(0, "prefill/rows", {"start": int(ownership.token_start), "rows": own_rows})
    loader = metadata = embedding = client_head = None
    generated: list[dict] = []
    prefill_hidden = None
    if rank == 0:
        prefill_hidden = np.zeros((int(layout.seq_len), hidden_dim), dtype=np.float32)
        for item in gathered_rows:
            prefill_hidden[int(item["start"]):int(item["start"]) + len(item["rows"])] = item["rows"]
        if model_dir:
            from llama3fhe.application.checkpoint import CheckpointMetadata, WeightLoader
            from llama3fhe.application.client import ClientHead, top_k_logits

            loader = WeightLoader(Path(model_dir), tensor_dtype=np.float32)
            loader.__enter__()
            metadata = CheckpointMetadata.from_json(Path(model_dir) / "config.json")
            embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
            client_head = ClientHead(loader, metadata)
            logits = client_head.logits(prefill_hidden, token_index=-1)
            top = top_k_logits(logits, 5)
            token_id = int(top[0]["token_id"])
        else:
            token_id = int(GOLD_TOKENS[0])  # no checkpoint: feed the gold token so the Decode chain still runs
            top = None
        generated.append({"step": 0, "source": "prefill", "token_id": token_id, "top_k": top})
    client_prefill_seconds = time.perf_counter() - started
    token_ids = transport.all_gather("prefill/token", int(generated[0]["token_id"]) if rank == 0 else -1)
    token_id = int(token_ids[0])
    prefill_oracle = None if prefix_k else _compare(own_rows, oracle_hidden[last][rows])

    # ------------------------------------------------------------ Decode
    hidden_stub = np.zeros((1, int(decode_feature.hidden_dim)), dtype=np.float32)
    input_comm: dict[str, float] = {}

    def encrypt_decode_input(hidden_row):
        own = None
        if int(rank) == 0:
            packed_row = decode_feature.pack(np.asarray(hidden_row, dtype=np.float32).reshape(1, -1), dtype=np.float64)
            own = [context.encrypt(np.asarray(r).reshape(-1), level=int(decode_input_level), slots=int(decode_feature.slots))
                   for r in packed_row]
        return tuple(broadcast_ciphers(transport, "decode/input", own, root=0, crypto_context=context, comm=input_comm))

    def decrypt_decode(ciphers):
        packed_out = np.stack([np.asarray(context.decrypt(c), dtype=np.float64) for c in ciphers])
        return np.asarray(decode_feature.unpack(packed_out), dtype=np.float32)

    decode_rows: list[dict] = []
    poly_hidden_chain = None
    if warm_keys and not emulated and int(decode_tokens) > 0:
        decode_warm = warm_phase(decode_plan)
        warm_keys_report["phases"]["decode"] = decode_warm
        warm_keys_report["decode_seconds"] = decode_warm["seconds"]
        warm_keys_report["resident_keys_gib"] = decode_warm["resident_keys_gib"]
        warm_keys_report["seconds"] = round(
            float(warm_keys_report.get("prefill_seconds", 0.0)) + float(decode_warm["seconds"]), 3
        )
    elif warm_keys:
        warm_keys_report["seconds"] = warm_keys_report.get("prefill_seconds", 0.0)
    if int(decode_tokens) > 0:
        prefetcher.start(layers[0])
    transport.barrier("decode/start")
    decode_started = time.perf_counter()
    for step in range(int(decode_tokens)):
        token_started = time.perf_counter()
        if rank == 0:
            hidden_row = (embedding[token_id] if embedding is not None else np.zeros(hidden_dim, dtype=np.float32))
            poly_hidden_chain = np.asarray(hidden_row, dtype=np.float32).reshape(1, -1)
        else:
            hidden_row = None
        hidden_ciphers = encrypt_decode_input(hidden_row)
        layer_rows = []
        for layer_index, idx in enumerate(layers):
            weights = prefetcher.get(idx)
            if layer_index + 1 < len(layers):
                prefetcher.start(layers[layer_index + 1])
            elif step + 1 < int(decode_tokens):
                prefetcher.start(layers[0])
            weights.register()
            graph_weights = GraphLayerWeights(**weights.host)
            stations: dict[str, dict] = {}

            def timed(name, fn, _idx=idx, _step=step, _stations=stations):
                return timed_station(_stations, f"{stage_prefix}d_{name}_{_step}_{_idx}", name,
                                     DECODE_STATION_TIERS.get(name), fn)

            position = int(prefix_k) + int(caches[idx].content_tokens)
            layer_start = time.perf_counter()
            result = decode_layer_parallel(
                crypto_context=context, transport=transport, ownership=decode_ownership, weights=graph_weights,
                approximations=approximations.for_layer(idx), schedule=decode_sched, config=config,
                input_ciphers=hidden_ciphers, kv_cache=caches[idx], position=position, layer_idx=idx,
                bootstrap_operator=bootstrap, feature_layout=decode_feature, attention_layout=decode_attention,
                pairfold_layout=pairfold_layout, rope_theta=ROPE_THETA,
                prefix_bundle=prefix_by_layer.get(idx),
                split_gate_up=bool(split_gate_up), split_kv_refresh=bool(split_kv_refresh), split_w_o=bool(split_w_o),
                hidden_stub=hidden_stub, timed=timed, rms_slim_period_rows=1,
            )
            layer_wall = time.perf_counter() - layer_start
            release_all(hidden_ciphers)
            hidden_ciphers = result.ciphers
            result.ciphers = ()
            row = {
                "layer": int(idx), "position": position, "wall_s": round(layer_wall, 4),
                "station_wall_sum_s": round(sum(s["wall_s"] for s in stations.values()), 4),
                "bootstrap_calls_total": int(sum(result.bootstrap_calls.values())),
                "bootstrap_seconds": round(sum(s["bootstrap_seconds"] for s in stations.values()), 4),
                "comm_seconds": round(sum(v for k, v in result.comm.items() if k.endswith("_seconds")), 4),
                "comm_bytes_sent": int(sum(v for k, v in result.comm.items() if k.endswith("_sent"))),
                "stations": stations,
                "weights_wait_s": round(prefetcher.wait_seconds.get(idx, 0.0), 3),
            }
            if verify and poly_kv_arrays is not None and rank == 0:
                from llama3fhe.approx.layer import poly_decode_layer_numpy

                started = time.perf_counter()
                fhe_hidden = decrypt_decode(hidden_ciphers)
                poly = poly_decode_layer_numpy(
                    poly_hidden_chain, cache_key=poly_kv_arrays[idx][0], cache_value=poly_kv_arrays[idx][1],
                    weights=graph_weights, approximations=approximations.for_layer(idx),
                    attention_layout=decode_attention, layer_idx=int(idx), position=position,
                    rope_theta=ROPE_THETA, use_alg2_softmax=True,
                    prefix_bundle=prefix_by_layer.get(idx))
                poly_kv_arrays[idx] = (np.concatenate([poly_kv_arrays[idx][0], poly.key], axis=0),
                                       np.concatenate([poly_kv_arrays[idx][1], poly.value], axis=0))
                poly_hidden_chain = np.asarray(poly.hidden, dtype=np.float32)
                diff = np.abs(fhe_hidden.astype(np.float64) - poly_hidden_chain.astype(np.float64))
                row["verify"] = {"seconds": round(time.perf_counter() - started, 3),
                                 "max_abs_diff_vs_poly_chain": float(diff.max()),
                                 "mean_abs_diff_vs_poly_chain": float(diff.mean()),
                                 "fhe_max_abs": float(np.abs(fhe_hidden).max()),
                                 "poly_max_abs": float(np.abs(poly_hidden_chain).max()),
                                 "finite": bool(np.all(np.isfinite(fhe_hidden)))}
            weights.release()
            layer_rows.append(row)
            if rank == 0:
                print(json.dumps({"phase": "decode", "step": step + 1, "layer": idx, "wall_s": row["wall_s"],
                                  "bs": row["bootstrap_calls_total"], "bs_s": row["bootstrap_seconds"],
                                  "comm_s": row["comm_seconds"],
                                  **({"vs_poly": round(row["verify"]["max_abs_diff_vs_poly_chain"], 5)} if "verify" in row else {})}),
                      flush=True)
        easyfhe.cuda.synchronize()
        layers_wall = time.perf_counter() - token_started
        # -- client: decrypt, LM head, next token
        started = time.perf_counter()
        if rank == 0:
            step_hidden = decrypt_decode(hidden_ciphers)
            if client_head is not None:
                from llama3fhe.application.client import top_k_logits

                logits = client_head.logits(step_hidden, token_index=0)
                top = top_k_logits(logits, 5)
                next_id = int(top[0]["token_id"])
            else:
                next_id = int(GOLD_TOKENS[min(step + 1, len(GOLD_TOKENS) - 1)])
                top = None
            generated.append({"step": step + 1, "source": "decode", "token_id": next_id, "top_k": top,
                              "finite": bool(np.all(np.isfinite(step_hidden)))})
        release_all(hidden_ciphers)
        hidden_ciphers = ()
        next_ids = transport.all_gather(f"decode/token/{step}", int(generated[-1]["token_id"]) if rank == 0 else -1)
        token_id = int(next_ids[0])
        client_seconds = time.perf_counter() - started
        decode_rows.append({
            "step": step + 1, "token_id": token_id, "wall_s": round(time.perf_counter() - token_started, 4),
            "layers_wall_s": round(layers_wall, 4), "client_seconds": round(client_seconds, 3),
            "layer_wall_sum_s": round(sum(r["wall_s"] for r in layer_rows), 4),
            "bootstrap_calls": int(sum(r["bootstrap_calls_total"] for r in layer_rows)),
            "bootstrap_seconds": round(sum(r["bootstrap_seconds"] for r in layer_rows), 4),
            "comm_seconds": round(sum(r["comm_seconds"] for r in layer_rows), 4),
            "layers": layer_rows,
        })
        if rank == 0:
            print(json.dumps({"phase": "decode", "step": step + 1, "token_id": token_id,
                              "wall_s": decode_rows[-1]["wall_s"], "layers_wall_s": decode_rows[-1]["layers_wall_s"],
                              "client_s": decode_rows[-1]["client_seconds"]}), flush=True)
    transport.barrier("decode/end")
    decode_wall = time.perf_counter() - decode_started
    for cache in caches.values():
        cache.release()
    caches.clear()
    if loader is not None:
        loader.__exit__(None, None, None)

    summary = {
        "rank": int(rank), "world_size": world_size, "emulated": bool(emulated),
        "layers": list(layers), "decode_tokens": int(decode_tokens),
        "prefix_k": int(prefix_k), "prefix_kv_cache": prefix_kv_cache,
        "setup_seconds": state.setup_seconds, "key_plan": state.key_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K)),
        "key_phase_plans": {
            "prefill": prefill_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K)),
            "decode": decode_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K)),
        },
        "key_plan_sizes": plan_sizes, "resident_keys_gib": native_cache_inventory(native)["total_gib"],
        "sparse_log_slots": list(preset.bootstrap_extra_sparse_log_slots),
        "decode_rows": int(decode_preset.linear_max_plaintext_rows), "prefill_rows": int(preset.linear_max_plaintext_rows),
        "shifted_lanes": bool(shifted_lanes), "sl_baby": int(decode_sched.decode_sl_baby_steps),
        "warm_keys": warm_keys_report,
        "tables": {"rmsnorm": rms_table_info, "softmax": softmax_table_info},
        "prefill": {"wall_s": round(prefill_wall, 3), "layers": prefill_records,
                    "client_seconds": round(client_prefill_seconds, 3), "own_rows_oracle": prefill_oracle,
                    "kv_emit_sum_s": round(sum(r["kv_emit_wall_s"] for r in prefill_records), 3),
                    "kv_emit_stream_layers": sum(
                        r["stations"]["kv_emit"].get("stream") == "resident-side"
                        for r in prefill_records
                    ),
                    "kv_emit_stream_fallback_layers": sum(
                        bool(r["stations"]["kv_emit"].get("stream_fallback", False))
                        for r in prefill_records
                    ),
                    "kv_emit_background_sum_s": round(sum(
                        float(r["stations"]["kv_emit"].get("background_wall_s", 0.0))
                        for r in prefill_records
                    ), 3),
                    "kv_emit_bs": int(sum(r["stations"]["kv_emit"]["bootstrap_calls"] for r in prefill_records))},
        "decode": {"wall_s": round(decode_wall, 3), "tokens": decode_rows, "input_comm": dict(input_comm)},
        "generated": generated if rank == 0 else None,
        "prefill_hidden": prefill_hidden,
        "transport": {**transport.stats.to_dict(), "backend": type(transport).__name__},
        "timeline": probe.timeline,
    }
    gathered = transport.gather_to(0, "summary", summary)
    transport.barrier("final")
    transport.close()
    if not emulated:
        state.release()
    return gathered if rank == 0 else None


# ------------------------------------------------------------------ driver
def _run_emulated(preset, kwargs: dict):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("LLAMA_FHE_PLAIN_CACHE_GB", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("LLAMA_FHE_TRANSFORM_HOIST", str(preset.transform_hoist_strategy))
    os.environ["T2_WORLD_SIZE"] = str(int(preset.world_size))
    if str(preset.key_policy) != "resident":
        raise SystemExit("--emulate-single-device needs key_policy='resident' (a100-high / h100-emu)")

    import easyfhe

    from llama3fhe.parallel.emulation import run_emulated_world
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.worker import build_worker_state

    decode_preset, decode_sched = decode_preset_and_schedule(preset, kv_limbs=kwargs["kv_limbs"],
                                                             shifted_lanes=kwargs["shifted_lanes"], sl_baby=kwargs["sl_baby"])
    started = time.perf_counter()
    plans = [rank_key_plan(preset, rank, decode_preset=decode_preset, decode_schedule=decode_sched,
                           layers=tuple(kwargs["layers"]), attention_limb_cap=kwargs.get("attention_limb_cap"),
                           prefix_k=int(kwargs.get("prefix_k", 0)))[0]
             for rank in range(int(preset.world_size))]
    union = merge_key_plans(*plans)
    shared = build_worker_state(0, preset, tiers=union.names, trace_keys=True, key_plan=union)
    easyfhe.cuda.synchronize()
    setup = {"seconds": round(time.perf_counter() - started, 2),
             "key_plan": shared.key_plan.summary(alpha=int(shared.native.alpha), special_limbs=int(shared.native.K)),
             "resident_keys_gib": native_cache_inventory(shared.native)["total_gib"]}
    print(json.dumps({"emulation_setup": setup}, default=str), flush=True)
    try:
        outcomes = run_emulated_world(preset, worker_entry, shared_state=shared, kwargs=kwargs)
    finally:
        shared.release()
    return outcomes, setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="a100-high")
    add_runtime_knob_args(parser)
    parser.add_argument("--fixtures", default="fixtures/model")
    parser.add_argument("--model-dir", default=None, help="checkpoint dir (LM head + embedding on rank 0); "
                        "without it the gold tokens are fed so the chain still runs")
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--decode-tokens", type=int, default=3, help="Decode steps after the Prefill token (gold: 3)")
    parser.add_argument("--input-level", type=int, default=35)
    parser.add_argument("--decode-input-level", type=int, default=35)
    parser.add_argument("--kv-limbs", type=int, default=12)
    parser.add_argument("--kv-bootstrap-iterations", type=int, default=1, help="Meta-BTS passes of the kv_emit refresh")
    parser.add_argument("--poly-kv", default=None, help="paris128_prefill_kv_poly.npz for --verify of the Decode layers")
    parser.add_argument("--prefix-kv-cache", default=None, help="plaintext per-layer Prefix KV cache (.npz)")
    parser.add_argument("--poly-kv-matches-prefix", action="store_true",
                        help="declare that --poly-kv was generated with the selected Prefix")
    parser.add_argument("--verify", action="store_true", help="rank 0 decrypts every Decode layer and compares with the poly chain")
    parser.add_argument("--verify-every", type=int, default=0, help="Prefill: decrypt + compare every N layers")
    parser.add_argument("--warm-keys", action="store_true")
    parser.add_argument("--sl", dest="shifted_lanes", action="store_true", default=True, help="(default) SL decode packing")
    parser.add_argument("--no-sl", dest="shifted_lanes", action="store_false")
    parser.add_argument("--sl-baby", type=int, default=16)
    parser.add_argument("--no-split-gate-up", action="store_true")
    parser.add_argument("--no-split-kv-refresh", action="store_true")
    parser.add_argument("--no-split-w-o", action="store_true")
    parser.add_argument("--no-prefetch-weights", action="store_true")
    parser.add_argument("--no-stage-weights", action="store_true")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="*", default=None)
    parser.add_argument("--ranks-per-device", type=int, default=1)
    parser.add_argument("--transport", choices=PROCESS_TRANSPORTS, default=None)
    parser.add_argument("--max-plaintext-rows", type=int, default=None, help="Prefill chunk rows (Decode clamps its own)")
    parser.add_argument("--attention-limb-cap", type=int, default=None,
                        help="Decode attn_core key cap (default: the preset's 13; 0 = full chain, +15 GiB resident)")
    parser.add_argument("--emulate-single-device", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _station_table(records: list[list[dict]], names) -> dict[str, dict]:
    """``{station: {wall_med_s, wall_max_s, bs}}`` over the records of all ranks x layers."""

    table = {}
    for name in names:
        walls = [r["stations"][name]["wall_s"] for per_rank in records for r in per_rank if name in r["stations"]]
        if walls:
            bs = [r["stations"][name]["bootstrap_calls"] for per_rank in records for r in per_rank if name in r["stations"]]
            table[name] = {"wall_med_s": round(_median(walls), 4), "wall_max_s": round(max(walls), 4),
                           "wall_sum_med_per_rank_s": round(_median(sum(r["stations"][name]["wall_s"] for r in per_rank)
                                                                    for per_rank in records), 3),
                           "bs_med": round(_median(bs), 2)}
    return table


def _prefix_cache_token_count(path: str | Path) -> int:
    """Read Prefix occupancy before creating the worker key plans."""

    import numpy as np

    with np.load(path, allow_pickle=False) as cache:
        prefix_k = int(cache["prefix_k"].item())
    if not 1 <= prefix_k <= 128:
        raise ValueError(f"Prefix cache has unsupported prefix_k={prefix_k}")
    return prefix_k


def main() -> int:
    import numpy as np

    args = parse_args()
    preset = preset_by_name(args.preset)
    preset = apply_runtime_knobs(args, preset)
    world_size = int(preset.world_size if args.world_size is None else args.world_size)
    if world_size != int(preset.world_size):
        preset = preset.with_world_size(world_size)
    if world_size not in (2, 4, 8):
        raise SystemExit("--world-size must be 2, 4 or 8 (multi-card Prefill + Decode)")
    if args.transport is not None:
        os.environ["T2_TRANSPORT"] = str(args.transport)
    if not args.emulate_single_device:
        expand_devices(preset.world_size, args.devices, ranks_per_device=int(args.ranks_per_device))
        if int(args.ranks_per_device) > 1:
            preset = preset.for_colocation(int(args.ranks_per_device))
    if args.max_plaintext_rows is not None:
        preset = replace(preset, linear_max_plaintext_rows=int(args.max_plaintext_rows))
    preset = combined_preset(preset)
    layers = parse_layers(args.layers)
    fixtures = Path(args.fixtures).resolve()
    for idx in layers:
        if not (fixtures / f"layer{idx}_weights.npz").exists():
            raise SystemExit(f"missing {fixtures / f'layer{idx}_weights.npz'}")
    if args.verify and args.poly_kv is None:
        raise SystemExit("--verify needs --poly-kv (paris128_prefill_kv_poly.npz)")
    prefix_path = None if args.prefix_kv_cache is None else Path(args.prefix_kv_cache).resolve()
    prefix_k = 0 if prefix_path is None else _prefix_cache_token_count(prefix_path)
    if prefix_k and layers[0] != 0:
        raise SystemExit("Prefix Prefill must start at layer 0; intermediate fixture hidden states have no Prefix")
    if prefix_k and args.model_dir is None:
        raise SystemExit("Prefix cache verification requires --model-dir")
    if prefix_k and args.verify and not args.poly_kv_matches_prefix:
        raise SystemExit("Prefix --verify needs --poly-kv-matches-prefix and a Prefix-matched --poly-kv")
    if prefix_k and int(args.verify_every) > 0:
        raise SystemExit("--verify-every uses the no-Prefix fixture trajectory and is unavailable with Prefix")
    if prefix_k:
        from llama3fhe.approx.prefix_cache_io import model_cache_metadata

        prefix_model_metadata = model_cache_metadata(args.model_dir)
    else:
        prefix_model_metadata = None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "artifacts" / f"t2_prefill_decode_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "fixtures": str(fixtures), "layers": layers, "decode_tokens": int(args.decode_tokens),
        "input_level": int(args.input_level), "decode_input_level": int(args.decode_input_level),
        "kv_limbs": int(args.kv_limbs), "model_dir": None if args.model_dir is None else str(Path(args.model_dir).resolve()),
        "poly_kv": None if args.poly_kv is None else str(Path(args.poly_kv).resolve()),
        "prefix_kv_cache": None if prefix_path is None else str(prefix_path), "prefix_k": int(prefix_k),
        "prefix_model_metadata": prefix_model_metadata,
        "verify": bool(args.verify), "verify_every": int(args.verify_every), "warm_keys": bool(args.warm_keys),
        "shifted_lanes": bool(args.shifted_lanes), "sl_baby": int(args.sl_baby),
        "split_gate_up": not args.no_split_gate_up, "split_kv_refresh": not args.no_split_kv_refresh,
        "split_w_o": not args.no_split_w_o, "kv_bootstrap_iterations": int(args.kv_bootstrap_iterations),
        "prefetch_weights": not args.no_prefetch_weights, "stage_weights": not args.no_stage_weights,
        "attention_limb_cap": args.attention_limb_cap,
    }
    decode_rows = decode_rows_for_world(int(preset.linear_max_plaintext_rows), world_size)
    print(json.dumps({"preset": preset.name, "world_size": world_size, "layers": [layers[0], layers[-1]],
                      "prefix_k": int(prefix_k),
                      "sparse_log_slots": list(preset.bootstrap_extra_sparse_log_slots),
                      "prefill_rows": int(preset.linear_max_plaintext_rows), "decode_rows": decode_rows,
                      "sl": bool(args.shifted_lanes), "sl_baby": int(args.sl_baby)}), flush=True)
    started = time.perf_counter()
    setup: dict = {}
    if args.emulate_single_device:
        outcomes, setup = _run_emulated(preset, kwargs)
    else:
        outcomes = spawn_workers(preset, worker_entry, kwargs=kwargs, devices=args.devices,
                                 ranks_per_device=int(args.ranks_per_device), transport=args.transport)
    total = time.perf_counter() - started
    failed = [o for o in outcomes if not getattr(o, "ok", True)]
    if failed:
        for o in failed:
            print(json.dumps({"rank": o.rank, "error": o.error}), flush=True)
        return 2
    per_rank = outcomes[0].result
    assert per_rank is not None and len(per_rank) == world_size
    per_rank.sort(key=lambda s: int(s["rank"]))
    rank0 = per_rank[0]
    prefill_hidden = rank0.pop("prefill_hidden")
    for s in per_rank[1:]:
        s.pop("prefill_hidden", None)
    last = layers[-1]
    if prefix_k:
        final_error = None
    else:
        with np.load(fixtures / "hidden_trajectory.npz") as data:
            expected = np.asarray(data[f"hidden_{last + 1}"], dtype=np.float32)
        final_error = _compare(np.asarray(prefill_hidden), expected)
    np.save(out_dir / f"hidden_{last + 1}_fhe.npy", np.asarray(prefill_hidden))

    generated = rank0["generated"]
    tokens = [int(g["token_id"]) for g in generated]
    gold = None if prefix_k else list(GOLD_TOKENS[:len(tokens)])
    gold_match = (tokens == gold) if (gold is not None and args.model_dir and last == 31) else None
    prefill_layers = [s["prefill"]["layers"] for s in per_rank]
    decode_layer_records = [[row for tok in s["decode"]["tokens"] for row in tok["layers"]] for s in per_rank]
    prefill_walls = [max(s["prefill"]["layers"][i]["layer_wall_s"] for s in per_rank) for i in range(len(layers))]
    decode_token_walls = [max(s["decode"]["tokens"][i]["wall_s"] for s in per_rank) for i in range(int(args.decode_tokens))]

    def warm_phase_max(field: str):
        values = [s["warm_keys"].get(field) for s in per_rank
                  if s["warm_keys"].get(field) is not None]
        return round(max(float(value) for value in values), 3) if values else None

    result = {
        "preset": preset.name, "world_size": world_size, "emulated_single_device": bool(args.emulate_single_device),
        "prefix_k": int(prefix_k), "prefix_kv_cache": None if prefix_path is None else str(prefix_path),
        "emulation_setup": setup or None, "timestamp": stamp, "layers": list(layers),
        "sparse_log_slots": list(preset.bootstrap_extra_sparse_log_slots),
        "prefill_rows": int(preset.linear_max_plaintext_rows), "decode_rows": decode_rows,
        "shifted_lanes": bool(args.shifted_lanes), "sl_baby": int(args.sl_baby),
        "total_wall_s": round(total, 2), "worker_wall_s": [round(o.wall_seconds, 2) for o in outcomes],
        "setup_seconds": rank0["setup_seconds"], "key_plan": rank0["key_plan"], "key_plan_sizes": rank0["key_plan_sizes"],
        "key_phase_plans": rank0["key_phase_plans"],
        "resident_keys_gib_max": round(max(float(s["resident_keys_gib"]) for s in per_rank), 3),
        "warm_keys_seconds_max": warm_phase_max("seconds"),
        "warm_keys_prefill_seconds_max": warm_phase_max("prefill_seconds"),
        "warm_keys_decode_seconds_max": warm_phase_max("decode_seconds"),
        "resident_keys_gib_after_prefill_max": warm_phase_max("resident_keys_gib_after_prefill"),
        "prefill": {
            "wall_max_s": round(max(s["prefill"]["wall_s"] for s in per_rank), 3),
            "layer_wall_sum_s": round(sum(prefill_walls), 3),
            "layer_wall_first_s": prefill_walls[0],
            "layer_wall_steady_med_s": round(_median(prefill_walls[1:]), 3) if len(prefill_walls) > 1 else None,
            "kv_emit_sum_max_s": round(max(s["prefill"]["kv_emit_sum_s"] for s in per_rank), 3),
            "kv_emit_stream_layers_min": min(s["prefill"].get("kv_emit_stream_layers", 0) for s in per_rank),
            "kv_emit_stream_layers_max": max(s["prefill"].get("kv_emit_stream_layers", 0) for s in per_rank),
            "kv_emit_stream_fallback_layers_max": max(
                s["prefill"].get("kv_emit_stream_fallback_layers", 0) for s in per_rank
            ),
            "kv_emit_background_sum_max_s": round(max(
                s["prefill"].get("kv_emit_background_sum_s", 0.0) for s in per_rank
            ), 3),
            "kv_emit_bs_per_rank": rank0["prefill"]["kv_emit_bs"],
            "client_seconds": rank0["prefill"]["client_seconds"],
            "bootstrap_calls_rank0": int(sum(r["bootstrap_calls"] for r in rank0["prefill"]["layers"])),
            "final_hidden_error": final_error,
            "stations": _station_table(prefill_layers, PREFILL_STATIONS),
            "peak_alloc_gib_max": round(max(st["peak_alloc_gib"] for s in per_rank for r in s["prefill"]["layers"]
                                            for st in r["stations"].values()), 2),
        },
        "decode": {
            "wall_max_s": round(max(s["decode"]["wall_s"] for s in per_rank), 3),
            "token_wall_max_s": decode_token_walls,
            "token_wall_med_s": round(_median(decode_token_walls), 3) if decode_token_walls else None,
            "layer_wall_med_s": round(_median(r["wall_s"] for per in decode_layer_records for r in per), 4)
            if decode_layer_records and decode_layer_records[0] else None,
            "client_seconds": [t["client_seconds"] for t in rank0["decode"]["tokens"]],
            "stations": _station_table(decode_layer_records, list(DECODE_STATION_TIERS) + [
                "input_rmsnorm", "qkv_all_reduce", "kv_append", "folded_o_all_reduce", "folded_o_refresh", "w_o_all_reduce",
                "attention_residual_add", "post_attention_bootstrap_rmsnorm", "mlp_input_modulus_drop", "pre_mlp_bootstrap",
                "mlp_residual"]),
            "verify_worst": (max((r["verify"]["max_abs_diff_vs_poly_chain"] for r in decode_layer_records[0] if "verify" in r),
                                 default=None)),
        },
        "generated": generated, "tokens": tokens,
        "gold_tokens": None if prefix_k else list(GOLD_TOKENS), "gold_match": gold_match,
        "args": vars(args), "ranks": per_rank,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("preset", "world_size", "total_wall_s", "resident_keys_gib_max",
                                              "warm_keys_seconds_max", "warm_keys_prefill_seconds_max",
                                              "warm_keys_decode_seconds_max", "resident_keys_gib_after_prefill_max",
                                              "sparse_log_slots")}))
    print(json.dumps({"prefill": {k: v for k, v in result["prefill"].items() if k not in ("stations", "final_hidden_error")},
                      "final_hidden_error": (None if final_error is None else
                                             {k: final_error[k] for k in ("max_abs_err", "rel_max_err", "finite")})}))
    print(json.dumps({"prefill_stations": result["prefill"]["stations"]}))
    print(json.dumps({"decode": {k: v for k, v in result["decode"].items() if k != "stations"}}))
    print(json.dumps({"decode_stations": result["decode"]["stations"]}))
    print(json.dumps({"tokens": tokens, "gold": gold, "gold_match": gold_match, "output": str(out_dir)}))
    return 0 if (gold_match is None or gold_match) else 1


if __name__ == "__main__":
    raise SystemExit(main())
