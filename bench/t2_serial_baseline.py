"""One card runs the whole 16-CT layer front serially: the T2 strong-scaling denominator.

Same L3 context, presets, station code and key lifecycle as the parallel run
(``bench/t2_layer.py``); the one card simply plays all eight ranks' shares in
turn.  No communication; every leased tier is loaded once per
station (the loop over ranks sits inside the lease), which is how a serial
single-card program would run this chain and is what a rank-summed parallel
number must beat.

    rmsnorm_in  x8 (pair-BS per FM pair)            -> 16 normalised CT
    qkv         x8 (own pair -> own two carriers)   -> 16 QKV carriers
    attn_lane   x8 (own two sources)                -> 16 lane CT
    attn_v      x8 own-source partials + finish     -> 4 V pairs (2 complex carriers)
    attn_q      1 call, all 16 Q pairs
    attn_k      1 call, all 4 K pairs
    attn_core   x8 sub-layout calls (2 Q x 1 KV pair each; the T2 denominator)
                + 1 full-layout call (16 Q x 4 KV; a serial program's own shape)
                each: QK lease -> softmax + PV lease, 3 / 10 BS
    pairfm      x8 source halves (2 O_C -> 8 partials, one 2-pass BS each) under one
                pairfm_output lease, then 8 target reduces -> 8 W_O carriers
    w_o         x8 (own carrier -> own two FM shards)
    residual_attn x8, rmsnorm_post x8 (pair-BS in/out), mlp x8 (gate/up -> SiLU -> down),
    residual_mlp x8                                   -> 16 layer-output CT

Every station's bootstrap seconds are recorded separately (``bootstrap_s``,
``bootstrap_calls``) so the per-type strong-scaling table can split "compute"
from "BS" per station.

Usage::

    python bench/t2_serial_baseline.py --preset a100-high --repeat 2 --core-shards shard \
        --stations rmsnorm_in,qkv,attn_lane,attn_v,attn_q,attn_k,attn_core,pairfm,w_o,rmsnorm_post,mlp \
        --output artifacts/t2_serial/run4_full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 26092802
HIDDEN_SCALE = 0.3
COEFF_LAYER = 10
WEIGHT_SCALE = 0.02
INTERMEDIATE = 14336
TIERS = ("bootstrap", "rmsnorm", "linear", "attn_input_lane", "attn_input_q", "attn_input_k", "attn_input_v",
         "attn_qk", "attn_pv", "softmax", "pairfm_output")
FULL_STATIONS = "rmsnorm_in,qkv,attn_lane,attn_v,attn_q,attn_k,attn_core,pairfm,w_o,rmsnorm_post,mlp"


class _BootstrapTimer:
    """Proxy around the ``BootstrapOperator`` timing every ``refresh`` (same as bench/t2_layer.py)."""

    def __init__(self, inner):
        self._inner = inner
        self.records: list[tuple[float, int, int]] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def calls(self) -> int:
        return int(self._inner.calls)

    def refresh(self, cipher, **kwargs):
        import easyfhe

        easyfhe.cuda.synchronize()
        started = time.perf_counter()
        result = self._inner.refresh(cipher, **kwargs)
        easyfhe.cuda.synchronize()
        self.records.append((time.perf_counter() - started, int(cipher.state.cur_limbs),
                             int(kwargs.get("iterations") or self._inner.config.iterations)))
        return result

    def take(self) -> list[tuple[float, int, int]]:
        taken, self.records = self.records, []
        return taken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="a100-high")
    parser.add_argument("--input-level", type=int, default=35)
    parser.add_argument("--hidden-scale", type=float, default=HIDDEN_SCALE)
    parser.add_argument("--coeff-layer", type=int, default=COEFF_LAYER)
    parser.add_argument("--qkv-limbs", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=2,
                        help="times the attention-input chain is run (iteration 0 is cold)")
    parser.add_argument("--stations", default="rmsnorm_in,qkv,attn_lane,attn_v,attn_q,attn_k",
                        help="comma list; add attn_core for the QK/softmax/PV station (8 sub-layout shards + 1 full call); "
                             "pairfm,w_o,rmsnorm_post,mlp extend to the whole layer (residuals are implied); "
                             "'full' = " + FULL_STATIONS)
    parser.add_argument("--core-shards", default="shard,full",
                        help="attn_core variants: shard (8 per-rank sub-layout calls) and/or full (one 16x4 call); "
                             "the layer tail (pairfm ...) consumes the shard variant's O_C")
    parser.add_argument("--mlp-limbs", type=int, default=None,
                        help="MLP entry modulus (default stations.mlp_input_limbs(silu_entry); 0 = no drop)")
    parser.add_argument("--real-weights", default=None,
                        help="fixtures/layerNN_fixture.npz from bench/export_layer_fixture.py (same weights/hidden as the 8-card run)")
    parser.add_argument("--no-pinned-keys", action="store_true")
    parser.add_argument("--qk-sweep", action="store_true",
                        help="after the chain, time the Q shard at 1/2/4/8/16 pairs and the K shard at 1/2/4 pairs "
                             "(warm keys, one lease per call) to separate fixed per-source cost from per-pair cost")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _gib(value: int) -> float:
    return float(value) / float(2**30)


def _union_rotations(preset, feature_layout, attention_layout, world, schedule):
    """Per sub-tier union over the eight ranks of the phase rotations (signed offsets)."""

    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.distributed.distributed_input import attention_input_shard_rotations
    from llama3fhe.parallel.prefill.ownership import key_pairs_for_rank

    carrier = LinearCarrierLayout(seq_len=int(feature_layout.seq_len), dimension=int(feature_layout.hidden_dim),
                                  slots=int(feature_layout.slots))
    ratio = int(attention_layout.gqa_ratio)
    all_carriers = tuple(range(int(attention_layout.key_value_pair_count) // 2))
    union = {"lane": set(), "query": set(), "key": set(), "value": set()}
    for rank in world:
        phases = attention_input_shard_rotations(
            carrier, attention_layout,
            query_pair_indices=tuple(rank.query_pair_indices),
            key_pair_indices=key_pairs_for_rank(rank, gqa_ratio=ratio),
            value_carrier_indices=all_carriers,
            value_source_indices=tuple(rank.cipher_indices),
            sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
            shear_baby_steps=int(schedule.shear_baby_steps), log_n=int(preset.log_n))
        for name in union:
            union[name].update(int(v) for v in phases[name])
    return {
        "attn_input_lane": tuple(sorted(union["lane"])),
        "attn_input_q": tuple(sorted(union["query"])),
        "attn_input_k": tuple(sorted(union["key"])),
        "attn_input_v": tuple(sorted(union["value"])),
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import easyfhe

    from dataclasses import replace

    from llama3fhe.approx.attention import rope_cos_sin
    from llama3fhe.application.args import DEFAULT_RMS
    from llama3fhe.approx.rmsnorm import get_layer_coeffs, load_rmsnorm_poly_coeffs
    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.operators.attention.distributed.distributed_input import (
        attention_input_lane_shard_fhe,
        attention_input_target_shard_fhe,
        attention_input_value_finish_fhe,
    )
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.prefill.ownership import key_pairs_for_rank
    from llama3fhe.parallel.presets import preset_by_name
    from llama3fhe.parallel.prefill.stations import (
        QKV_INPUT_LIMBS,
        _release,
        attention_core_operator_config,
        attention_core_station,
        mlp_input_limbs,
        mlp_station,
        qkv_station,
        residual_station,
        rmsnorm_station,
        w_o_station,
    )
    from llama3fhe.parallel.worker import build_worker_state, load_layer_config, ownership_from_layouts, production_layouts
    from llama3fhe.graph import _linear_config
    from llama3fhe.application.args import DEFAULT_SILU, DEFAULT_SOFTMAX
    from llama3fhe.approx.silu import load_silu_poly_coeffs
    from llama3fhe.approx.softmax import SoftmaxPolyConfig, load_softmax_poly_coeffs
    from llama3fhe.operators.attention.distributed.distributed_output import pairfm_source_partials_fhe, pairfm_target_reduce_fhe

    preset = preset_by_name(args.preset)
    if args.no_pinned_keys:
        preset = replace(preset, pin_leased_keys=False)
    stations = [s.strip() for s in str(args.stations).replace("full", FULL_STATIONS).split(",") if s.strip()]
    tail = [s for s in ("pairfm", "w_o", "rmsnorm_post", "mlp") if s in stations]
    if tail and any(s not in stations for s in ("attn_lane", "attn_v", "attn_q", "attn_k", "attn_core")):
        raise SystemExit("the layer tail (pairfm/w_o/rmsnorm_post/mlp) needs attn_lane,attn_v,attn_q,attn_k,attn_core")
    if tail and tail != ["pairfm", "w_o", "rmsnorm_post", "mlp"][:len(tail)]:
        raise SystemExit("the layer tail must be a prefix of pairfm,w_o,rmsnorm_post,mlp")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "artifacts" / f"t2_serial_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_layout, attention_layout = production_layouts()
    schedule = preset.layer_schedule()
    qkv_config = _linear_config(schedule, schedule.qkv_max_plaintext_rows, schedule.qkv_hoist_strategy)
    w_o_config = _linear_config(schedule, schedule.output_max_plaintext_rows, schedule.output_hoist_strategy)
    mlp_config = _linear_config(schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy)
    world = ownership_from_layouts(feature_layout, attention_layout, world_size=int(preset.world_size))
    extra = _union_rotations(preset, feature_layout, attention_layout, world, schedule)

    started_all = time.perf_counter()
    state = build_worker_state(0, preset, tiers=TIERS, trace_keys=True, echo_memory=False, extra_rotations=extra)
    probe, context, layout = state.memory, state.context, state.feature_layout
    native, life, tracer = state.native, state.keys, state.tracer
    bootstrap = _BootstrapTimer(state.bootstrap_operator)
    config = load_layer_config(preset)
    print(json.dumps({"setup": state.setup_seconds, "resident_gib": native_cache_inventory(native)["total_gib"],
                      "pinned": None if life.pinned is None else life.pinned.summary(),
                      "key_plan": state.key_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K))},
                     default=str), flush=True)

    # -- inputs: all sixteen FM ciphers of one shared random hidden state (same seed as the 8-card bench)
    rng = np.random.default_rng(SEED)
    hidden_dim = int(layout.hidden_dim)
    kv_dim = int(attention_layout.key_value_heads) * int(attention_layout.head_dim)
    rms_all = load_rmsnorm_poly_coeffs(DEFAULT_RMS)
    coeffs = get_layer_coeffs(rms_all, "attn", int(args.coeff_layer))
    coeffs_ffn = get_layer_coeffs(rms_all, "ffn", int(args.coeff_layer))
    silu_entry = load_silu_poly_coeffs(DEFAULT_SILU)["per_layer"][str(int(args.coeff_layer))]["tiers"]["standard"]
    if args.real_weights:
        from bench.t2_layer import load_layer_fixture

        fixture = load_layer_fixture(args.real_weights)
        hidden = np.asarray(fixture["hidden_in"], dtype=np.float32)
        if hidden.shape != (int(layout.seq_len), hidden_dim):
            raise ValueError(f"fixture hidden_in has shape {hidden.shape}, layout wants {(layout.seq_len, hidden_dim)}")
        gamma = np.asarray(fixture["input_layernorm"], dtype=np.float32)
        gamma_post = np.asarray(fixture["post_attention_layernorm"], dtype=np.float32)
        q_w, k_w, v_w, o_w = (np.asarray(fixture[k], dtype=np.float32) for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
        gate_w, up_w, down_w = (np.asarray(fixture[k], dtype=np.float32) for k in ("gate_proj", "up_proj", "down_proj"))
    else:
        hidden = rng.normal(scale=float(args.hidden_scale), size=(int(layout.seq_len), hidden_dim)).astype(np.float32)
        gamma = gamma_post = np.ones(hidden_dim, dtype=np.float32)
        q_w = rng.normal(scale=WEIGHT_SCALE, size=(hidden_dim, hidden_dim)).astype(np.float32)
        k_w = rng.normal(scale=WEIGHT_SCALE, size=(kv_dim, hidden_dim)).astype(np.float32)
        v_w = rng.normal(scale=WEIGHT_SCALE, size=(kv_dim, hidden_dim)).astype(np.float32)
        o_w = rng.normal(scale=WEIGHT_SCALE, size=(hidden_dim, hidden_dim)).astype(np.float32)
        gate_w = rng.normal(scale=WEIGHT_SCALE, size=(INTERMEDIATE, hidden_dim)).astype(np.float32)
        up_w = rng.normal(scale=WEIGHT_SCALE, size=(INTERMEDIATE, hidden_dim)).astype(np.float32)
        down_w = rng.normal(scale=WEIGHT_SCALE, size=(hidden_dim, INTERMEDIATE)).astype(np.float32)
    packed = layout.pack(hidden)
    fuse_silu = bool(schedule.fuse_silu_domain_map)
    mlp_entry = (mlp_input_limbs(silu_entry, fuse_silu_domain_map=fuse_silu)
                 if args.mlp_limbs is None else (int(args.mlp_limbs) or None))
    inputs: list = []
    probe.snap("10 inputs")

    carrier_layout = LinearCarrierLayout(seq_len=int(layout.seq_len), dimension=int(layout.hidden_dim), slots=int(layout.slots))
    operator_config = AttentionOperatorConfig(
        sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
        shear_baby_steps=int(schedule.shear_baby_steps),
        pv_complex_delta_folding=bool(schedule.pv_complex_delta_folding),
        pv_prefold_after_softmax=bool(schedule.pv_prefold_after_softmax),
        batched_sparse_input_bsgs=bool(schedule.batched_sparse_attention_input),
        qk_complex_feature_folding=True,
        qk_key_features_pre_folded=True,
        qk_query_features_pre_scaled=True,
        qk_query_features_pre_folded=True,
    )
    core_config = attention_core_operator_config(schedule, layer_idx=int(args.coeff_layer), base=operator_config)
    softmax_config = SoftmaxPolyConfig.from_artifact(load_softmax_poly_coeffs(DEFAULT_SOFTMAX))
    core_variants = [s.strip() for s in str(args.core_shards).split(",") if s.strip()]
    ratio = int(attention_layout.gqa_ratio)
    all_carriers = tuple(range(int(attention_layout.key_value_pair_count) // 2))
    all_q_pairs = tuple(range(int(world.query_pair_count)))
    all_k_pairs = tuple(sorted({p for rank in world for p in key_pairs_for_rank(rank, gqa_ratio=ratio)}))
    cos, sin = rope_cos_sin(seq_len=int(attention_layout.seq_len), head_dim=int(attention_layout.head_dim),
                            theta=float(AttentionOperatorConfig().rope_theta), start_pos=0)
    qkv_limbs = QKV_INPUT_LIMBS if args.qkv_limbs is None else (int(args.qkv_limbs) or None)

    results: dict[str, dict] = {}

    def run(name: str, tier: str | None, fn):
        """One station over all ranks under ONE lease of ``tier``."""

        probe.reset_peak()
        easyfhe.cuda.synchronize()
        pinned_before = life.pinned.summary() if life.pinned is not None else None
        bootstrap.take()
        started = time.perf_counter()
        with tracer.stage(name):
            if tier is None:
                value = fn()
            else:
                with life.lease(tier, prefetch=False) as lease:
                    value = fn()
        easyfhe.cuda.synchronize()
        wall = time.perf_counter() - started
        row = probe.snap(f"20 {name}", {"wall_s": round(wall, 4)})
        stats = tracer.stages[name].as_row()
        stats = {k: stats.get(k) for k in ("requests", "misses", "gib_loaded", "load_seconds", "max_limbs", "distinct_rotations")}
        stats.update({"wall_s": wall, "peak_alloc_gib": row.get("peak_alloc_gib"), "alloc_after_gib": row.get("alloc_gib"),
                      "cache_after_gib": native_cache_inventory(native)["total_gib"], "tier": tier})
        bs_records = bootstrap.take()
        stats["bootstrap_s"] = float(sum(r[0] for r in bs_records))
        stats["bootstrap_calls_timed"] = len(bs_records)
        stats["bootstrap_records"] = [(round(r[0], 4), r[1], r[2]) for r in bs_records]
        stats["compute_s"] = wall - stats["bootstrap_s"]
        if tier is not None:
            stats["lease"] = {"evict_s": lease.evict_seconds, "evicted_entries": lease.evicted_entries}
        if pinned_before is not None:
            after = life.pinned.summary()
            stats["pinned"] = {k: round(after[k] - pinned_before[k], 4) for k in ("hits", "misses", "gib_copied")}
        stats["alloc_after_gib"] = row.get("alloc_gib")
        results.setdefault(name, []).append(stats)
        print(json.dumps({"stage": name, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()
                                              if k in ("wall_s", "bootstrap_s", "bootstrap_calls_timed", "gib_loaded", "load_seconds",
                                                       "distinct_rotations", "peak_alloc_gib", "pinned")}},
                         default=str), flush=True)
        return value

    keep: dict[str, object] = {}
    failure = None

    def _front():
        """1.-2. rmsnorm_in + qkv over the eight ranks (re-run per iteration when the layer tail consumes them)."""

        nonlocal inputs
        if not inputs:
            inputs = [context.encrypt(packed[i], level=int(args.input_level), slots=int(layout.slots))
                      for i in range(int(world.cipher_count))]
            easyfhe.cuda.synchronize()
        # 1. RMSNorm entry: eight pair-BS calls
        if "rmsnorm_in" in stations:
            def _rms():
                outs, scales = [], []
                for rank in world:
                    pair = tuple(inputs[i] for i in rank.cipher_indices)
                    res = rmsnorm_station(pair, gamma, coeffs=coeffs, ownership=rank, layout=layout, context=context,
                                          bootstrap_operator=bootstrap, keep_residual=bool(tail), exit_refresh=True,
                                          function_prefix=f"serial.rms_in.{rank.rank}")
                    outs.append(res)
                    scales.append(res.linear_input_column_scale())
                return outs, scales
            rms_list, scales = run("rmsnorm_in", None, _rms)
            keep["rms"] = rms_list
            results["rmsnorm_in"][-1]["output_limbs"] = [list(r.output_limbs) for r in rms_list][:2]
            results["rmsnorm_in"][-1]["bootstrap_calls"] = int(sum(r.bootstrap_calls for r in rms_list))
            _release(*inputs)
            inputs = []

        # 2. QKV: eight token shards under one linear lease (linear is resident in the preset)
        if "qkv" in stations:
            rms_list = keep["rms"]
            old_carriers = keep.pop("carriers", None)
            if old_carriers:
                _release(*(c for c in old_carriers if c is not None))
            def _qkv():
                outs = []
                for rank, rms in zip(world, rms_list, strict=True):
                    outs.append(qkv_station(rms.ciphers, hidden, q_w, k_w, v_w, ownership=rank, layout=layout,
                                            attention_layout=attention_layout, context=context, config=config,
                                            operator_config=qkv_config,
                                            input_column_scale=rms.linear_input_column_scale(), input_limbs=qkv_limbs,
                                            function_prefix=f"serial.qkv.{rank.rank}"))
                    _release(*rms.ciphers)
                    rms.ciphers = ()
                return outs
            qkv_list = run("qkv", "linear", _qkv)
            qkv_stages: dict[str, float] = {}
            for res in qkv_list:
                for k, v in dict(getattr(res, "stage_seconds", {}) or {}).items():
                    qkv_stages[k] = qkv_stages.get(k, 0.0) + float(v)
            results["qkv"][-1]["stage_seconds_sum8"] = qkv_stages
            carriers: list = [None] * int(world.cipher_count)
            for rank, res in zip(world, qkv_list, strict=True):
                for source, cipher in zip(rank.cipher_indices, res.carrier_ciphers, strict=True):
                    carriers[int(source)] = cipher
                res.carrier_ciphers = ()
            keep["carriers"] = carriers
            results["qkv"][-1]["carrier_limbs"] = [int(c.state.cur_limbs) for c in carriers][:2]

    try:
        if not tail:
            _front()

        def _qk_sweep(lanes):
            """Warm-key Q/K target shards at several pair counts (fixed vs per-pair cost)."""

            def _shard(kind: str, count: int):
                q_pairs = tuple(all_q_pairs[:count]) if kind == "q" else ()
                k_pairs = tuple(all_k_pairs[:count]) if kind == "k" else ()
                shard = attention_input_target_shard_fhe(
                    tuple(lanes), query_pair_indices=q_pairs, key_pair_indices=k_pairs, value_carrier_indices=(),
                    carrier_layout=carrier_layout, attention_layout=attention_layout, crypto_context=context,
                    model_config=config, operator_config=operator_config, function_prefix=f"serial.sweep.{kind}{count}")
                stages = dict(shard.stage_seconds)
                shard.release()
                return stages

            for kind, counts in (("q", (1, 2, 4, 8, 16)), ("k", (1, 2, 4))):
                tier = f"attn_input_{kind}"
                for count in counts:
                    name = f"sweep_{kind}{count}"
                    stages = run(name, tier, lambda: _shard(kind, count))
                    results[name][-1]["pairs"] = count
                    results[name][-1]["stage_seconds"] = stages

        def _core_record(name: str, res_list, wall_started: float, keep_outputs: bool = False):
            row = results[name][-1]
            row["calls"] = len(res_list)
            row["bootstrap_calls"] = int(sum(r.bootstrap_calls for r in res_list))
            row["output_limbs"] = sorted({int(v) for r in res_list for v in r.output_limbs})
            row["output_pairs"] = int(sum(len(r.output_items) for r in res_list))
            stages: dict[str, float] = {}
            for r in res_list:
                for k, v in r.stage_seconds.items():
                    stages[k] = stages.get(k, 0.0) + float(v)
            row["stage_seconds_sum"] = stages
            row["stage_seconds_first"] = dict(res_list[0].stage_seconds)
            row["level_trace"] = {k: list(v) for k, v in res_list[0].level_trace.items()}
            row["leases"] = [
                {"tier": r.tier, "prefetch_keys": r.prefetched_keys, "prefetch_s": round(r.prefetch_seconds, 4),
                 "evict_s": round(r.evict_seconds, 4), "evicted_entries": r.evicted_entries, "retained_entries": r.retained_entries}
                for res in res_list for r in res.lease_records
            ]
            row["sub_layout"] = {"query_heads": int(res_list[0].sub_layout.query_heads),
                                 "key_value_heads": int(res_list[0].sub_layout.key_value_heads)}
            # limbs the kernels really asked for, per tier of the plan (cap audit)
            asked = tracer.stages[name].max_limbs
            per_tier = {}
            for tier in ("attn_qk", "attn_pv", "softmax", "bootstrap", "rmsnorm", "linear"):
                rots = set(int(v) for v in state.key_plan.tier(tier).rotations)
                hit = {r: l for r, l in asked.items() if r in rots}
                if hit:
                    per_tier[tier] = {"keys_used": len(hit), "max_limbs": max(hit.values()),
                                      "over_13": sorted((r, l) for r, l in hit.items() if l > 13)[:12]}
            row["asked_limbs"] = per_tier
            kept = None
            if keep_outputs:
                kept = [tuple(r.output_items) for r in res_list]
                for r in res_list:
                    r.output_items = ()
            for r in res_list:
                r.release()
            return kept

        def _attn_core(query_items, key_items, value_items, iteration: int):
            """QK -> softmax -> PV on the Q/K/V the chain just produced (inputs are kept).

            Returns the shard variant's O_C items (one tuple of 2 per rank) when the layer tail needs them.
            """

            q_by = dict(query_items)
            k_by = dict(key_items)
            v_by = dict(value_items)
            o_c = None
            if "shard" in core_variants:
                def _shards():
                    outs = []
                    for rank in world:
                        q_pairs = tuple(int(v) for v in rank.query_pair_indices)
                        k_pairs = key_pairs_for_rank(rank, gqa_ratio=ratio)
                        outs.append(attention_core_station(
                            tuple((i, q_by[i]) for i in q_pairs), tuple((i, k_by[i]) for i in k_pairs),
                            tuple((i, v_by[i]) for i in k_pairs), ownership=rank, attention_layout=attention_layout,
                            context=context, softmax_config=softmax_config, operator_config=core_config,
                            bootstrap_operator=bootstrap, layer_idx=int(args.coeff_layer), keys=life,
                            prefetch_keys=False, function_prefix=f"serial.core.{rank.rank}"))
                    return outs
                started = time.perf_counter()
                res_list = run("attn_core_shard", None, _shards)
                o_c = _core_record("attn_core_shard", res_list, started, keep_outputs=bool(tail))
            if "full" in core_variants:
                def _full():
                    return [attention_core_station(
                        tuple((i, q_by[i]) for i in all_q_pairs), tuple((i, k_by[i]) for i in all_k_pairs),
                        tuple((i, v_by[i]) for i in all_k_pairs), ownership=None, attention_layout=attention_layout,
                        context=context, softmax_config=softmax_config, operator_config=core_config,
                        bootstrap_operator=bootstrap, layer_idx=int(args.coeff_layer), keys=life,
                        prefetch_keys=False, function_prefix="serial.core.full")]
                started = time.perf_counter()
                res_list = run("attn_core_full", None, _full)
                _core_record("attn_core_full", res_list, started)
            return o_c

        def _layer_tail(o_c_by_rank, iteration: int):
            """pairfm -> w_o -> residual -> rmsnorm_post -> mlp -> residual on the 8 ranks' O_C (consumed)."""

            rms_list = keep["rms"]

            # 7. PairFM: 8 source halves under ONE pairfm_output lease, then 8 target reduces (no transport)
            def _pairfm():
                partials_by_src: list[list] = []
                source_stages: dict[str, float] = {}
                for rank, items in zip(world, o_c_by_rank, strict=True):
                    ciphers = tuple(cipher for _, cipher in items)
                    source = pairfm_source_partials_fhe(
                        ciphers, source_logical_rank=int(rank.rank), feature_layout=layout,
                        attention_layout=attention_layout, crypto_context=context, model_config=config,
                        operator_config=core_config, bootstrap_operator=bootstrap,
                        function_prefix=f"serial.pairfm.{rank.rank}.source")
                    partials_by_src.append(list(source.partial_ciphers))
                    source.partial_ciphers = ()
                    for k, v in source.stage_seconds.items():
                        source_stages[str(k)] = source_stages.get(str(k), 0.0) + float(v)
                    _release(*ciphers)
                easyfhe.cuda.synchronize()
                source_done = time.perf_counter()
                carriers_out = []
                for target in range(len(world)):
                    summands = [partials_by_src[src][target] for src in range(len(world))]
                    carriers_out.append(pairfm_target_reduce_fhe(summands, crypto_context=context, synchronize=True))
                return carriers_out, source_stages, source_done

            started = time.perf_counter()
            wo_carriers, source_stages, source_done = run("pairfm", "pairfm_output", _pairfm)
            results["pairfm"][-1]["source_stage_seconds_sum8"] = source_stages
            results["pairfm"][-1]["reduce_s"] = time.perf_counter() - source_done
            results["pairfm"][-1]["carrier_limbs"] = [int(c.state.cur_limbs) for c in wo_carriers][:2]

            # 8. W_O: 8 carriers -> 16 FM shards under one linear lease
            def _wo():
                outs = []
                profiles: dict[str, float] = {}
                for rank, carrier in zip(world, wo_carriers, strict=True):
                    prof: dict[str, float] = {}
                    outs.append(tuple(w_o_station(carrier, o_w, ownership=rank, layout=layout,
                                                  attention_layout=attention_layout, context=context,
                                                  operator_config=w_o_config, profile=prof)))
                    for k, v in prof.items():
                        if "." not in k:
                            profiles[k] = profiles.get(k, 0.0) + float(v)
                    _release(carrier)
                return outs, profiles
            wo_list, wo_profiles = run("w_o", "linear", _wo)
            results["w_o"][-1]["stage_seconds_sum8"] = wo_profiles
            results["w_o"][-1]["output_limbs"] = [int(c.state.cur_limbs) for c in wo_list[0]]
            if "w_o" not in stations:
                _release(*(c for pair in wo_list for c in pair))
                return

            # 9. residual: refreshed layer input + W_O shards
            def _res1():
                outs = []
                for rank, rms, wo in zip(world, rms_list, wo_list, strict=True):
                    outs.append(residual_station(rms.residual_inputs, wo, ownership=rank, layout=layout, context=context))
                    _release(*wo)
                    rms.release()
                return outs
            res1_list = run("residual_attn", None, _res1)
            results["residual_attn"][-1]["modulus_aligned_pairs"] = int(sum(r.modulus_aligned_pairs for r in res1_list))
            results["residual_attn"][-1]["output_limbs"] = [int(c.state.cur_limbs) for c in res1_list[0].ciphers]
            if "rmsnorm_post" not in stations:
                for r in res1_list:
                    r.release()
                return

            # 10. post-attention RMSNorm (pair-BS in/out), keeps the refreshed residual branch
            def _rms2():
                outs = []
                for rank, res1 in zip(world, res1_list, strict=True):
                    outs.append(rmsnorm_station(res1.ciphers, gamma_post, coeffs=coeffs_ffn, ownership=rank, layout=layout,
                                                context=context, bootstrap_operator=bootstrap, keep_residual=True,
                                                exit_refresh=True, function_prefix=f"serial.rms_post.{rank.rank}"))
                    res1.release()
                return outs
            rms2_list = run("rmsnorm_post", None, _rms2)
            results["rmsnorm_post"][-1]["output_limbs"] = [list(r.output_limbs) for r in rms2_list][:2]
            results["rmsnorm_post"][-1]["bootstrap_calls"] = int(sum(r.bootstrap_calls for r in rms2_list))
            stages: dict[str, float] = {}
            for r in rms2_list:
                for k, v in r.stage_seconds.items():
                    stages[str(k)] = stages.get(str(k), 0.0) + float(v)
            results["rmsnorm_post"][-1]["stage_seconds_sum8"] = stages
            if "mlp" not in stations:
                for r in rms2_list:
                    r.release()
                return

            # 11. MLP: 8 token shards under one linear lease
            def _mlp():
                outs = []
                stages: dict[str, float] = {}
                for rank, rms2 in zip(world, rms2_list, strict=True):
                    res = mlp_station(rms2.ciphers, hidden, gate_w, up_w, down_w, silu_entry=silu_entry, ownership=rank,
                                      layout=layout, context=context, config=config, operator_config=mlp_config,
                                      input_column_scale=rms2.linear_input_column_scale(),
                                      fuse_silu_domain_map=fuse_silu, input_limbs=mlp_entry)
                    outs.append(res)
                    for k, v in res.stage_seconds.items():
                        stages[str(k)] = stages.get(str(k), 0.0) + float(v)
                    _release(*rms2.ciphers)
                    rms2.ciphers = ()
                return outs, stages
            mlp_list, mlp_stages = run("mlp", "linear", _mlp)
            results["mlp"][-1]["stage_seconds_sum8"] = mlp_stages
            results["mlp"][-1]["output_limbs"] = [int(c.state.cur_limbs) for c in mlp_list[0].ciphers]
            results["mlp"][-1]["entry_limbs"] = mlp_entry

            # 12. residual: refreshed post-norm input + MLP shards -> 16 layer outputs
            def _res2():
                outs = []
                for rank, rms2, mlp in zip(world, rms2_list, mlp_list, strict=True):
                    outs.append(residual_station(rms2.residual_inputs, mlp.ciphers, ownership=rank, layout=layout, context=context))
                    rms2.release()
                    mlp.release()
                return outs
            res2_list = run("residual_mlp", None, _res2)
            results["residual_mlp"][-1]["modulus_aligned_pairs"] = int(sum(r.modulus_aligned_pairs for r in res2_list))
            results["residual_mlp"][-1]["output_limbs"] = [int(c.state.cur_limbs) for c in res2_list[0].ciphers]
            for r in res2_list:
                r.release()

        # 3.-6. attention input chain, repeated
        attn = [s for s in ("attn_lane", "attn_v", "attn_q", "attn_k", "attn_core") if s in stations]
        for iteration in range(int(args.repeat) if attn else 0):
            if tail:
                _front()
            carriers = keep["carriers"]
            lanes: list = [None] * int(world.cipher_count)
            chain_started = time.perf_counter()
            if "attn_lane" in attn:
                def _lane():
                    for rank in world:
                        own = tuple(carriers[i] for i in rank.cipher_indices)
                        res = attention_input_lane_shard_fhe(own, source_indices=tuple(rank.cipher_indices),
                                                             carrier_layout=carrier_layout, attention_layout=attention_layout,
                                                             crypto_context=context, cos=cos, sin=sin,
                                                             function_prefix=f"serial.lane.{rank.rank}")
                        for source, cipher in zip(rank.cipher_indices, res.lane_ciphers, strict=True):
                            lanes[int(source)] = cipher
                        res.lane_ciphers = ()
                        res.release()
                run("attn_lane", "attn_input_lane", _lane)
            value_items = ()
            if "attn_v" in attn:
                def _value():
                    summands = []
                    partials = []
                    for rank in world:
                        sparse = [None] * len(lanes)
                        for i in rank.cipher_indices:
                            sparse[int(i)] = lanes[int(i)]
                        part = attention_input_target_shard_fhe(
                            tuple(sparse), query_pair_indices=(), key_pair_indices=(), value_carrier_indices=all_carriers,
                            carrier_layout=carrier_layout, attention_layout=attention_layout, crypto_context=context,
                            model_config=config, operator_config=operator_config, function_prefix=f"serial.vpart.{rank.rank}",
                            value_source_indices=tuple(rank.cipher_indices), value_return_partial=True)
                        for c, cipher in part.value_partials:
                            summands.append((int(c), cipher))
                            partials.append(cipher)
                        part.value_partials = ()
                        stage_acc = results.setdefault("_value_partial_stages", {})
                        for k, v in part.stage_seconds.items():
                            stage_acc[k] = stage_acc.get(k, 0.0) + float(v)
                        part.release()
                    items = attention_input_value_finish_fhe(summands, attention_layout=attention_layout, crypto_context=context)
                    _release(*partials)
                    return items
                results.pop("_value_partial_stages", None)
                value_items = run("attn_v", "attn_input_v", _value)
                results["attn_v"][-1]["output_limbs"] = [int(c.state.cur_limbs) for _, c in value_items]
                results["attn_v"][-1]["stage_seconds_sum8"] = results.pop("_value_partial_stages", {})
            query_items = ()
            if "attn_q" in attn:
                q_shard = run("attn_q", "attn_input_q", lambda: attention_input_target_shard_fhe(
                    tuple(lanes), query_pair_indices=all_q_pairs, key_pair_indices=(), value_carrier_indices=(),
                    carrier_layout=carrier_layout, attention_layout=attention_layout, crypto_context=context,
                    model_config=config, operator_config=operator_config, function_prefix="serial.q"))
                query_items = tuple(q_shard.query_items)
                q_shard.query_items = ()
                results["attn_q"][-1]["stage_seconds"] = dict(q_shard.stage_seconds)
                q_shard.release()
                results["attn_q"][-1]["output_limbs"] = [int(c.state.cur_limbs) for _, c in query_items][:2]
                results["attn_q"][-1]["pairs"] = len(query_items)
            key_items = ()
            if "attn_k" in attn:
                k_shard = run("attn_k", "attn_input_k", lambda: attention_input_target_shard_fhe(
                    tuple(lanes), query_pair_indices=(), key_pair_indices=all_k_pairs, value_carrier_indices=(),
                    carrier_layout=carrier_layout, attention_layout=attention_layout, crypto_context=context,
                    model_config=config, operator_config=operator_config, function_prefix="serial.k"))
                key_items = tuple(k_shard.key_items)
                k_shard.key_items = ()
                results["attn_k"][-1]["stage_seconds"] = dict(k_shard.stage_seconds)
                results["attn_k"][-1]["scheduler_stats"] = dict(k_shard.scheduler_stats)
                k_shard.release()
                results["attn_k"][-1]["output_limbs"] = [int(c.state.cur_limbs) for _, c in key_items]
                results["attn_k"][-1]["pairs"] = len(key_items)
            easyfhe.cuda.synchronize()
            chain = time.perf_counter() - chain_started
            results.setdefault("attn_input_chain", []).append({"iteration": iteration, "wall_s": chain})
            print(json.dumps({"attn_input_chain": iteration, "wall_s": round(chain, 3)}), flush=True)
            o_c_by_rank = None
            if "attn_core" in attn and query_items and key_items and value_items:
                o_c_by_rank = _attn_core(query_items, key_items, value_items, iteration)
            _release(*(c for _, c in query_items + key_items + value_items))
            if tail:
                if o_c_by_rank is None or len(o_c_by_rank) != len(world):
                    raise RuntimeError("layer tail needs the attn_core shard variant's O_C (add 'shard' to --core-shards)")
                _layer_tail(o_c_by_rank, iteration)
            if args.qk_sweep and iteration == int(args.repeat) - 1 and all(c is not None for c in lanes):
                _qk_sweep(lanes)
            _release(*(c for c in lanes if c is not None))
    except Exception as exc:  # noqa: BLE001
        import traceback

        failure = {"type": type(exc).__name__, "message": str(exc)[:2000], "traceback": traceback.format_exc()[-4000:]}
        print(json.dumps({"failure": failure["type"], "message": failure["message"][:300]}), flush=True)

    leftover = keep.pop("carriers", None)
    if leftover:
        _release(*(c for c in leftover if c is not None))
    for r in keep.pop("rms", ()) or ():
        r.release()
    _release(*inputs)
    easyfhe.cuda.synchronize()
    summary = {
        "preset": preset.name,
        "pin_leased_keys": bool(preset.pin_leased_keys),
        "mode": "serial_one_card_all_ranks",
        "world_size_played": int(preset.world_size),
        "ciphers": int(world.cipher_count),
        "timestamp": stamp,
        "args": vars(args),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "device_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "setup_seconds": state.setup_seconds,
        "total_wall_s": time.perf_counter() - started_all,
        "failure": failure,
        "key_plan": state.key_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K)),
        "extra_rotations": {k: len(v) for k, v in extra.items()},
        "resident": state.resident_report(),
        "stations": results,
        "memory_trace": list(probe.timeline),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"output": str(out_dir), "failure": None if failure is None else failure["type"],
                      "stations": {k: [round(r["wall_s"], 3) for r in v] for k, v in results.items()}}), flush=True)
    state.release()
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
