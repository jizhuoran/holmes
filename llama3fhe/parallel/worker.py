"""Per-rank state for the T2 runtime: context, resident keys, bootstrap.

A worker is one OS process pinned to one GPU.  It builds, once:

1. its own CKKS context (client + native) whose host key material covers the
   rotation-key union of the requested *tiers* (:mod:`.keys`); keygen is
   deterministic, so every rank ends up with the same secret key and can
   exchange ciphertexts without sharing material;
2. the L3 bootstrap program (``dnum5``, level budget from the preset,
   ``ConstantBundle`` cache mode ``middle`` on 4090 / ``plain`` on A100),
   warmed once so the CUDA key/constant caches are materialised before the
   first timed operator;
3. a :class:`~.keys.KeyLifecycle` that keeps the T0 tiers (BS + RMSNorm)
   resident and brackets every other station with prefetch / evict under the
   preset's ``key_policy``;
4. the transport endpoint for the two per-layer collectives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .prefill.keys import LEASE_POLICY, KeyLifecycle, KeyPlan, KeyTracer, T0_TIERS, build_key_plan
from .memory import MemoryProbe, bootstrap_constant_cache_gib, native_cache_inventory
from .prefill.ownership import RankOwnership, WorldOwnership, ownership_from_layouts
from .presets import ParallelPreset


# ------------------------------------------------------------------ config
def load_layer_config(preset: ParallelPreset, path: str | None = None):
    """``Llama3CKKSConfig`` for the preset: assets/feature_major.json + dnum/depth/logN."""

    from pathlib import Path

    from llama3fhe.config import Llama3CKKSConfig

    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = str(root / "assets" / "feature_major.json")
    config = Llama3CKKSConfig.from_json(path)
    config.simulator.dnum = int(preset.dnum)
    config.simulator.maxLevelsRemaining = int(preset.depth)
    config.simulator.logN = int(preset.log_n)
    # Inside a worker the pinned card is always the exact string "cuda"; some
    # EasyFHE key/bootstrap loaders compare against that literal.
    config.simulator.device = "cuda"
    return config


def production_layouts():
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

    return FeatureMajorPrefillLayout(), AttentionPairLayout()


# --------------------------------------------------------------- key plan
@dataclass(frozen=True)
class RotationPlan:
    """Which rotation keys one rank keeps resident, by operator family."""

    bootstrap: tuple[int, ...]
    rmsnorm: tuple[int, ...]
    qkv: tuple[int, ...]
    attention: tuple[int, ...]
    w_o: tuple[int, ...]
    mlp: tuple[int, ...]

    @property
    def union(self) -> tuple[int, ...]:
        keys: set[int] = set()
        for group in (self.bootstrap, self.rmsnorm, self.qkv, self.attention, self.w_o, self.mlp):
            keys.update(int(v) for v in group)
        return tuple(sorted(keys))

    def summary(self) -> dict[str, int]:
        return {
            "bootstrap": len(self.bootstrap),
            "rmsnorm": len(self.rmsnorm),
            "qkv": len(self.qkv),
            "attention": len(self.attention),
            "w_o": len(self.w_o),
            "mlp": len(self.mlp),
            "union": len(self.union),
        }


def resident_rotations(
    preset: ParallelPreset,
    feature_layout,
    attention_layout,
    *,
    families: tuple[str, ...] = ("bootstrap", "rmsnorm"),
    schedule=None,
) -> RotationPlan:
    """Rotation keys per operator family; ``families`` selects which are non-empty."""

    from llama3fhe.operators.nonlinear.bootstrap import BootstrapOperator
    from llama3fhe.operators.norm.rmsnorm import rmsnorm_rotations

    wanted = set(families)
    bootstrap = tuple(
        int(v)
        for v in BootstrapOperator.required_rotations(
            log_n=int(preset.log_n),
            log_slots=int(preset.bootstrap.log_slots),
            level_budget=tuple(preset.bootstrap.level_budget),
            strategy=str(preset.bootstrap.strategy),
            sparse_log_slots=sparse_bootstrap_log_slots(preset, feature_layout, attention_layout),
        )
    ) if "bootstrap" in wanted else ()
    rmsnorm = (
        tuple(sorted(set(int(v) for v in rmsnorm_rotations(feature_layout))))
        if "rmsnorm" in wanted
        else ()
    )
    qkv: tuple[int, ...] = ()
    attention: tuple[int, ...] = ()
    w_o: tuple[int, ...] = ()
    mlp: tuple[int, ...] = ()
    if wanted & {"qkv", "attention", "w_o", "mlp"}:
        from llama3fhe.graph import _linear_config
        from llama3fhe.layouts.linear import LinearCarrierLayout
        from llama3fhe.operators.attention.prefill.operator import attention_to_w_o_rotations
        from llama3fhe.operators.attention.config import AttentionOperatorConfig
        from llama3fhe.operators.linear.config import linear_rotations
        from llama3fhe.operators.linear.prefill.mlp import mlp_rotations
        from llama3fhe.operators.linear.prefill.qkv import qkv_rotations
        from llama3fhe.schedule import LayerSchedule

        schedule = LayerSchedule() if schedule is None else schedule
        carrier = LinearCarrierLayout(
            seq_len=int(feature_layout.seq_len),
            dimension=int(feature_layout.hidden_dim),
            slots=int(feature_layout.slots),
        )
        if "qkv" in wanted:
            qkv = tuple(
                sorted(
                    qkv_rotations(
                        feature_layout,
                        operator_config=_linear_config(
                            schedule, schedule.qkv_max_plaintext_rows, schedule.qkv_hoist_strategy
                        ),
                    )
                )
            )
        if "attention" in wanted:
            attention = tuple(
                sorted(
                    attention_to_w_o_rotations(
                        carrier,
                        feature_layout,
                        attention_layout,
                        operator_config=AttentionOperatorConfig(
                            sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
                            shear_baby_steps=int(schedule.shear_baby_steps),
                            pv_complex_delta_folding=bool(schedule.pv_complex_delta_folding),
                            pv_prefold_after_softmax=bool(schedule.pv_prefold_after_softmax),
                        ),
                        log_n=int(preset.log_n),
                    )
                )
            )
        if "w_o" in wanted:
            w_o = tuple(
                sorted(
                    linear_rotations(
                        dimension=int(feature_layout.hidden_dim),
                        slots=int(feature_layout.slots),
                        token_lanes=int(feature_layout.tokens_per_cipher),
                        include_conjugation=True,
                        operator_config=_linear_config(
                            schedule,
                            schedule.output_max_plaintext_rows,
                            schedule.output_hoist_strategy,
                        ),
                    )
                )
            )
        if "mlp" in wanted:
            mlp = tuple(
                sorted(
                    mlp_rotations(
                        feature_layout,
                        operator_config=_linear_config(
                            schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy
                        ),
                    )
                )
            )
    return RotationPlan(
        bootstrap=bootstrap, rmsnorm=rmsnorm, qkv=qkv, attention=attention, w_o=w_o, mlp=mlp
    )


# --------------------------------------------------------------- context
def host_rss_gib() -> float:
    """Resident set size of this process (GiB) from ``/proc/self/status``; -1 if unavailable."""

    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / float(2**20)
    except OSError:
        pass
    return -1.0


def _trim_host_heap() -> None:
    """Return freed glibc arenas to the OS (keygen scratch, dropped host keys)."""

    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_context(config, rotations, *, auto_load_keys: bool = True, rotation_key_limb_limits=None):
    """One CKKS context (client + native) on the pinned card.

    Same construction as :func:`llama3fhe.application.session._create_context`
    (kept local so the worker does not import the whole application session).
    ``rotation_key_limb_limits`` (``rotation -> Q limbs``) trims the host key
    of every capped rotation at generation time (:meth:`KeyPlan.limb_limits`).
    """

    import easyfhe
    from easyfhe import fhe

    from llama3fhe.backend import EasyFHEContext

    started = time.perf_counter()
    simulator = config.simulator
    client, native = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=int(simulator.maxLevelsRemaining),
            log_n=int(simulator.logN),
            dnum=int(simulator.dnum),
            dcrt_bits=int(simulator.dcrtBits),
            first_mod=int(simulator.firstMod),
            secret_key_dist=str(simulator.secretKeyDist),
            scale_mode="fixed",
            rescale_policy=simulator.rescale_policy(),
            rotations=tuple(int(v) for v in rotations),
            auto_load_keys=bool(auto_load_keys),
            rotation_key_limb_limits={int(k): int(v) for k, v in dict(rotation_key_limb_limits or {}).items()},
        ),
        device=str(simulator.device),
    )
    return EasyFHEContext(
        tensor=easyfhe,
        fhe=fhe,
        client=client,
        context=native,
        max_slots=int(native.max_slots),
        keygen_wall_time_s=float(time.perf_counter() - started),
    )


def slim_sparse_log_slots(preset: ParallelPreset, feature_layout) -> int | None:
    """``log2`` of the sparse bootstrap slot count the preset's periodic slim RMSNorm pack needs.

    ``None`` when :attr:`~.presets.ParallelPreset.rmsnorm_slim_period_rows` is
    unset or the period already spans the whole slot vector.
    """

    period_rows = getattr(preset, "rmsnorm_slim_period_rows", None)
    if period_rows is None:
        return None
    from llama3fhe.operators.norm.rmsnorm import slim_period_slots

    slots = int(slim_period_slots(feature_layout, int(period_rows)))
    if slots >= int(feature_layout.slots):
        return None
    return slots.bit_length() - 1


def softmax_slim_sparse_log_slots(preset: ParallelPreset, attention_layout) -> int | None:
    """``log2`` of the sparse program the periodic Prefill Softmax slim pack of one rank needs.

    ``None`` unless :attr:`~.presets.ParallelPreset.softmax_slim_periodic` is
    set.  A rank owns ``query_pair_count / world_size`` Q pairs, so its Alg2
    slim ciphertext repeats every ``P * 2 * seq_len`` slots with ``P`` the
    next power of two >= that count
    (:func:`~llama3fhe.operators.attention.prefill.softmax.softmax_slim_period_slots`
    on the rank's sub-layout): 2 * 256 -> 9 at W=8, 10 at W=4, 12 single-card.
    """

    if not bool(getattr(preset, "softmax_slim_periodic", False)) or attention_layout is None:
        return None
    from llama3fhe.operators.attention.prefill.softmax import softmax_slim_period_slots
    from llama3fhe.parallel.prefill.stations import attention_sub_layout

    pairs = int(attention_layout.query_pair_count)
    world = max(1, int(preset.world_size))
    if pairs % world:
        raise ValueError(f"query_pair_count {pairs} is not a multiple of world_size {world}")
    kv_pairs = max(1, int(attention_layout.key_value_pair_count) // world)
    sub = attention_sub_layout(attention_layout, query_pair_count=pairs // world, key_value_pair_count=kv_pairs)
    slots = int(softmax_slim_period_slots(sub))
    if slots >= int(attention_layout.slots):
        return None
    return slots.bit_length() - 1


def sparse_bootstrap_log_slots(
    preset: ParallelPreset, feature_layout, attention_layout=None
) -> tuple[int, ...] | None:
    """All sparse bootstrap programs a rank needs: ``(slim RMSNorm row, Softmax slim row, *preset extras)``.

    Combines :func:`slim_sparse_log_slots`, :func:`softmax_slim_sparse_log_slots`
    (when ``attention_layout`` is given) and
    :attr:`~.presets.ParallelPreset.bootstrap_extra_sparse_log_slots` (e.g. the
    Decode merged-page Softmax period), deduplicated and sorted; ``None`` when
    there is nothing to add.
    """

    ks: set[int] = set()
    slim = slim_sparse_log_slots(preset, feature_layout)
    if slim is not None:
        ks.add(int(slim))
    softmax = softmax_slim_sparse_log_slots(preset, attention_layout)
    if softmax is not None:
        ks.add(int(softmax))
    for k in tuple(getattr(preset, "bootstrap_extra_sparse_log_slots", ()) or ()):
        k = int(k)
        if 0 < k < int(preset.bootstrap.log_slots):
            ks.add(k)
    return tuple(sorted(ks)) or None


def prepare_bootstrap(
    context, preset: ParallelPreset, *, sparse_log_slots: int | tuple[int, ...] | None = None
):
    """Prepared L3 bootstrap operator with the preset's constant cache mode.

    ``sparse_log_slots`` adds the sparse-slot program(s) of the periodic slim
    RMSNorm row / Decode Softmax rows (:func:`sparse_bootstrap_log_slots`);
    their constants take the same cache mode as the dense program.
    """

    from llama3fhe.operators.nonlinear.bootstrap import BootstrapConfig, BootstrapOperator

    config = BootstrapConfig(
        log_slots=int(preset.bootstrap.log_slots),
        level_budget=tuple(preset.bootstrap.level_budget),
        output_levels=int(preset.bootstrap_post_levels),
        iterations=int(preset.bootstrap.iterations),
        precision_bits=int(preset.bootstrap.precision_bits),
        mode=str(preset.bootstrap.mode),
        strategy=str(preset.bootstrap.strategy),
        evict_rotation_cache=False,
        sparse_log_slots=_normalise_sparse_log_slots(sparse_log_slots),
    )
    operator = BootstrapOperator.prepare(context, config)
    operator.program.constants.set_cache_mode(str(preset.bootstrap.cache_mode), clear=True)
    for sparse_program in operator.sparse_programs.values():
        sparse_program.constants.set_cache_mode(str(preset.bootstrap.cache_mode), clear=True)
    return operator


def _normalise_sparse_log_slots(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value),)
    ks = tuple(sorted({int(k) for k in value}))
    return ks or None


def warm_bootstrap(context, operator, *, slots: int, level: int, probe: MemoryProbe | None = None):
    """Run one bootstrap on a throw-away ciphertext so caches are materialised.

    Returns ``(cold_seconds, output_limbs)``.
    """

    import easyfhe
    import numpy as np

    from llama3fhe.backend import release_if_supported

    values = np.random.default_rng(0).uniform(3.0, 5.5, size=int(slots)) / 32.0
    cipher = context.encrypt(values, level=int(level), slots=int(slots))
    easyfhe.cuda.synchronize()
    started = time.perf_counter()
    out = operator.refresh(cipher, iterations=int(operator.config.iterations), synchronize=True)
    cold = time.perf_counter() - started
    limbs = int(out.state.cur_limbs)
    release_if_supported(out)
    release_if_supported(cipher)
    if probe is not None:
        probe.snap("bootstrap warm", {"cold_bs_s": cold, "bs_out_limbs": limbs})
    return cold, limbs


# ---------------------------------------------------------------- state
@dataclass
class WorkerState:
    """Everything one rank keeps for the whole run."""

    rank: int
    preset: ParallelPreset
    world: WorldOwnership
    feature_layout: Any
    attention_layout: Any
    key_plan: KeyPlan
    context: Any
    bootstrap_operator: Any
    keys: KeyLifecycle
    transport: Any = None
    tracer: KeyTracer | None = None
    memory: MemoryProbe = field(default_factory=lambda: MemoryProbe(rank=0))
    setup_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def ownership(self) -> RankOwnership:
        return self.world[self.rank]

    @property
    def native(self):
        return self.context.context

    @property
    def slots(self) -> int:
        return int(self.feature_layout.slots)

    def resident_report(self) -> dict[str, Any]:
        keys = native_cache_inventory(self.native)
        native = self.native
        return {
            "rank": int(self.rank),
            "preset": self.preset.name,
            "key_plan": self.key_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K)),
            "resident_keys_gib": keys["total_gib"],
            "pinned_keys": None if self.keys.pinned is None else self.keys.pinned.summary(),
            "host_rss_gib": host_rss_gib(),
            "bootstrap_constants": bootstrap_constant_cache_gib(self.bootstrap_operator.program),
            "memory": self.memory.last() if self.memory.timeline else {},
            "setup_seconds": dict(self.setup_seconds),
        }

    def release(self) -> None:
        from llama3fhe.backend import release_if_supported

        if self.tracer is not None:
            self.tracer.uninstall()
        self.keys.release()
        release_if_supported(self.bootstrap_operator)
        clear = getattr(self.context, "clear_plaintext_cache", None)
        if callable(clear):
            clear()


def build_worker_state(
    rank: int,
    preset: ParallelPreset,
    *,
    transport=None,
    tiers: tuple[str, ...] = T0_TIERS,
    config=None,
    feature_layout=None,
    attention_layout=None,
    warm: bool = True,
    echo_memory: bool = False,
    trace_keys: bool = False,
    limb_caps=None,
    extra_rotations=None,
    key_plan=None,
    emit_kv_cache: bool = False,
) -> WorkerState:
    """Context + key lifecycle + warmed bootstrap for one rank (call inside the worker).

    ``tiers`` selects which key tiers (:data:`~.keys.TIER_ORDER`) the context
    generates host material for; the default is the T0 residency
    (BS + RMSNorm). ``limb_caps`` overrides the preset's
    ``key_limb_caps``.  With ``trace_keys`` a :class:`~.keys.KeyTracer` is
    installed on the native context and exposed as ``state.tracer``.
    ``extra_rotations`` (tier -> signed offsets) is forwarded to
    :func:`~.keys.build_key_plan`.  ``key_plan`` (a prebuilt
    :class:`~.keys.KeyPlan`, e.g. :func:`~.decode_keys.build_decode_key_plan`)
    replaces that call altogether; ``tiers`` / ``limb_caps`` /
    ``extra_rotations`` are then ignored.  ``emit_kv_cache`` is forwarded to
    :func:`~.keys.build_key_plan` (the Prefill emits the Decode KV cache).
    """

    import easyfhe

    if feature_layout is None or attention_layout is None:
        default_feature, default_attention = production_layouts()
        feature_layout = default_feature if feature_layout is None else feature_layout
        attention_layout = default_attention if attention_layout is None else attention_layout
    world = ownership_from_layouts(feature_layout, attention_layout, world_size=int(preset.world_size))
    if int(world.ciphers_per_rank) != int(preset.ciphers_per_rank):
        raise ValueError(
            f"layout gives {world.ciphers_per_rank} ciphers/rank but preset says {preset.ciphers_per_rank}"
        )
    config = load_layer_config(preset) if config is None else config
    probe = MemoryProbe(rank=int(rank), echo=bool(echo_memory))
    probe.reset_peak()
    probe.snap("00 start")

    caps = dict(preset.limb_caps) if limb_caps is None else {str(k): int(v) for k, v in dict(limb_caps).items()}
    if key_plan is not None:
        plan = key_plan
    else:
        plan = build_key_plan(
            preset,
            feature_layout,
            attention_layout,
            tiers=tuple(tiers),
            schedule=preset.layer_schedule(),
            limb_caps=caps,
            resident_extra=tuple(preset.key_resident_extra),
            policy=str(preset.key_policy),
            ownership=world[int(rank)],
            extra_rotations=extra_rotations,
            emit_kv_cache=bool(emit_kv_cache),
        )
    timings: dict[str, float] = {}

    started = time.perf_counter()
    context = build_context(
        config,
        plan.union,
        auto_load_keys=bool(preset.auto_load_keys),
        rotation_key_limb_limits=plan.limb_limits(),
    )
    timings["context_keygen"] = time.perf_counter() - started
    native = context.context
    probe.snap("01 context", {"key_plan": plan.summary(alpha=int(native.alpha), special_limbs=int(native.K))})
    lifecycle = KeyLifecycle(
        native,
        plan,
        prefetch_mode=str(getattr(preset, "key_prefetch_mode", "lazy")),
        prefetch_margin_gib=float(getattr(preset, "key_prefetch_margin_gib", 1.5)),
        evict_sync=str(getattr(preset, "key_evict_sync", "device")),
    )
    if bool(preset.pin_leased_keys) and str(preset.key_policy) == LEASE_POLICY:
        started = time.perf_counter()
        pinned = lifecycle.pin_leased_keys()
        _trim_host_heap()
        timings["pin_leased_keys"] = time.perf_counter() - started
        probe.snap("01 pinned keys", {"pinned_keys": pinned, "host_rss_gib": host_rss_gib()})
    tracer = KeyTracer(native).install() if trace_keys else None
    lifecycle.tracer = tracer
    if lifecycle.prefetch_mode == "async" and str(preset.key_policy) == LEASE_POLICY:
        # Outermost wrapper: learns the per-rotation limbs each tier really asks for.
        lifecycle.install_profile_hook()

    started = time.perf_counter()
    operator = prepare_bootstrap(
        context, preset, sparse_log_slots=sparse_bootstrap_log_slots(preset, feature_layout, attention_layout)
    )
    timings["bootstrap_prepare"] = time.perf_counter() - started
    probe.snap("02 bootstrap prepared")

    if warm:
        cold, limbs = warm_bootstrap(
            context, operator, slots=int(feature_layout.slots), level=27, probe=probe
        )
        timings["bootstrap_cold"] = cold
        if limbs != int(preset.bootstrap_post_levels) + 1:
            # cur_limbs = level + 1; the preset promises ``bootstrap_post_levels``.
            raise RuntimeError(
                f"bootstrap returned {limbs} limbs, preset expects {preset.bootstrap_post_levels + 1}"
            )
    easyfhe.cuda.synchronize()
    probe.snap("03 resident", {"resident_keys_gib": native_cache_inventory(context.context)["total_gib"]})
    return WorkerState(
        rank=int(rank),
        preset=preset,
        world=world,
        feature_layout=feature_layout,
        attention_layout=attention_layout,
        key_plan=plan,
        context=context,
        bootstrap_operator=operator,
        keys=lifecycle,
        transport=transport,
        tracer=tracer,
        memory=probe,
        setup_seconds=timings,
    )


__all__ = [
    "RotationPlan",
    "WorkerState",
    "build_context",
    "build_worker_state",
    "load_layer_config",
    "prepare_bootstrap",
    "production_layouts",
    "resident_rotations",
    "slim_sparse_log_slots",
    "softmax_slim_sparse_log_slots",
    "sparse_bootstrap_log_slots",
    "warm_bootstrap",
]
