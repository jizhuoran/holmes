"""Rotation-key tiers and their device-side lifecycle for one T2 rank.

Why this exists
---------------
Rotation keys live in tiers so memory-constrained presets can keep bootstrap
and RMSNorm keys resident while leasing keys for other stations:

=====  =================  ==========================================  ==============
tier   name               rotations                                   residency
=====  =================  ==========================================  ==============
T0     ``bootstrap``      Meta-BTS program keys (39)                  whole layer
T0     ``rmsnorm``        RMSNorm fold keys (24)                      whole layer
T1     ``linear``         QKV == W_O == MLP diagonal keys (57 - T0)   preset policy
T2     ``attn_input``     lane/target shard (RoPE, shear, V gather)   one station
T2     ``attn_input_lane``  own-source lane/phase (15)               one phase
T2     ``attn_input_q``     Q-only target shard over 16 lanes        one phase
T2     ``attn_input_k``     K-only target shard (duplicated)         one phase
T2     ``attn_input_v``     own-source V partial + finish            one phase
T2     ``attn_core``      QK / PV delta rotations                     one station
T2     ``attn_qk``          QK-only delta rotations (190)            one phase
T2     ``attn_pv``          PV-only delta rotations (193)            one phase
T2     ``softmax``        softmax fold rotations                      one station
T2     ``pairfm_output``  PairFM source partial rotations             one station
=====  =================  ==========================================  ==============

``attn_input`` is the monolithic tier (single-card oracle, A100 residency);
the four ``attn_input_*`` sub-tiers are the T2 phase-1 split of the same
station and depend on the rank's
ownership.  Likewise ``attn_core`` is the QK+PV union while ``attn_qk`` /
``attn_pv`` are its two lease phases. QK and PV can lease their own key sets.
Neither depends on the pair count, so a per-rank sub-layout shard leases the
same keys as the full layout.  A plan may hold both forms.

QKV, W_O and MLP request the same linear rotation set. Keys shared by
different consumers retain the largest limb cap those consumers require.

Two EasyFHE facts make the lifecycle cheap:

* ``Context.get_rotation_key_for_limbs(rot, cur_limbs)`` uploads a key
  *compacted to the ciphertext's limb count* (``beta = ceil(cur_limbs/alpha)``
  digits, ``cur_limbs + K`` limbs each).
  ``Context.clear_cuda_rotation_cache(keep_rotations=...)`` drops the device
  copies and keeps the host master material.
* ``CKKSContextSpec.rotation_key_limb_limits`` generates a *trimmed* host key
  (``limit + K`` limbs, ``ceil(limit/alpha)`` digits) that rotates exactly
  like the full key up to ``limit`` limbs and raises ``ValueError`` above it
  Host memory per rank scales with the tier caps instead of the full chain.

Key generation is deterministic (fixed Blake2 seed, one PRNG stream per
rotation in ``parallel_deterministic`` mode): every rank that builds a
context from the same chain parameters gets the *same* secret key, public
key and rotation keys, regardless of which rotations or limb caps it asks
for.  Ciphertexts therefore move between ranks without any shared material.

Public surface
--------------
``KeyTier`` and ``KeyPlan`` describe rotations, caps, and residency.
``KeyLifecycle`` manages prefetch, eviction, and station leases.
``PinnedCompactKeys`` pre-compacts leased keys in pinned host memory.
``KeyTracer`` records rotations and cache activity during local execution.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Mapping

from ..memory import GIB

RESIDENT_POLICY = "resident"
LEASE_POLICY = "lease"
KEY_POLICIES = (RESIDENT_POLICY, LEASE_POLICY)

T0_TIERS = ("bootstrap", "rmsnorm")
T1_TIERS = ("linear",)
ATTN_INPUT_PHASE_TIERS = ("attn_input_lane", "attn_input_q", "attn_input_k", "attn_input_v")
ATTN_CORE_PHASE_TIERS = ("attn_qk", "attn_pv")
T2_TIERS = (
    ("attn_input",)
    + ATTN_INPUT_PHASE_TIERS
    + ("attn_core",)
    + ATTN_CORE_PHASE_TIERS
    + ("softmax", "pairfm_output")
)
TIER_ORDER = T0_TIERS + T1_TIERS + T2_TIERS


def normalize_rotation(rotation: int, *, ring_dim: int) -> int:
    """Signed slot offset -> the automorphism index EasyFHE keys its caches by.

    EasyFHE stores ``rot < 0`` as ``N/2 + rot`` (``_norm_rot_index``) and
    ``clear_cuda_rotation_cache`` normalises ``keep_rotations`` the same way.
    """

    rotation = int(rotation)
    half_ring = int(ring_dim) // 2
    if rotation < 0:
        return half_ring + rotation
    return rotation


def rotation_key_bytes(*, ring_dim: int, limbs: int, alpha: int, special_limbs: int) -> int:
    """Device bytes of one rotation key compacted to ``limbs`` Q limbs.

    ``bx`` and ``ax`` each hold ``ceil(limbs/alpha)`` digits of
    ``(limbs + special_limbs) * N`` u64 words.
    """

    limbs = int(limbs)
    if limbs <= 0:
        return 0
    beta = (limbs + int(alpha) - 1) // int(alpha)
    words = beta * (limbs + int(special_limbs)) * int(ring_dim)
    return 2 * 8 * words


# ------------------------------------------------------------------- tiers
@dataclass(frozen=True)
class KeyTier:
    """One residency group of rotation keys.

    ``level`` is 0 (layer resident), 1 (linear, policy-dependent) or 2
    (station lease).  ``limb_cap`` is the largest ciphertext limb count the
    tier's rotations are ever applied to; ``None`` means "full chain".  The
    cap is used twice: as ``rotation_key_limb_limits`` at key generation
    (trimmed host key) and as the ``cur_limbs`` handed to
    ``get_rotation_key_for_limbs`` when the tier is prefetched.
    """

    name: str
    level: int
    rotations: tuple[int, ...]
    limb_cap: int | None = None

    def __post_init__(self) -> None:
        if str(self.name) not in TIER_ORDER:
            raise ValueError(f"unknown tier {self.name!r}; choose from {TIER_ORDER}")
        if int(self.level) not in (0, 1, 2):
            raise ValueError("tier level must be 0, 1 or 2")
        if self.limb_cap is not None and int(self.limb_cap) <= 0:
            raise ValueError("limb_cap must be positive")
        object.__setattr__(self, "rotations", tuple(sorted({int(v) for v in self.rotations if int(v) != 0})))
        object.__setattr__(self, "level", int(self.level))
        object.__setattr__(self, "limb_cap", None if self.limb_cap is None else int(self.limb_cap))

    @property
    def rotation_set(self) -> frozenset[int]:
        return frozenset(self.rotations)

    def __len__(self) -> int:
        return len(self.rotations)


@dataclass(frozen=True)
class KeyPlan:
    """All tiers of one rank plus which of them stay resident under a policy."""

    tiers: tuple[KeyTier, ...]
    policy: str = LEASE_POLICY
    ring_dim: int = 1 << 16
    #: rotation keys hold ``L`` Q limbs when uncapped (``depth + 1``).
    full_limbs: int = 38
    #: non-T0 tiers a preset promotes to layer residency (e.g. ``("linear",)``).
    resident_extra: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if str(self.policy) not in KEY_POLICIES:
            raise ValueError(f"unknown key policy {self.policy!r}; choose from {KEY_POLICIES}")
        names = [tier.name for tier in self.tiers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tier names in plan: {names}")
        ordered = tuple(sorted(self.tiers, key=lambda tier: TIER_ORDER.index(tier.name)))
        object.__setattr__(self, "tiers", ordered)

    # -- lookup
    def tier(self, name: str) -> KeyTier:
        for tier in self.tiers:
            if tier.name == str(name):
                return tier
        raise KeyError(f"plan has no tier {name!r}; tiers={self.names}")

    def has_tier(self, name: str) -> bool:
        return any(tier.name == str(name) for tier in self.tiers)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tier.name for tier in self.tiers)

    # -- residency
    @property
    def resident_names(self) -> tuple[str, ...]:
        """Tiers that are never evicted between stations.

        ``resident`` (A100): everything.  ``lease`` (4090): T0 only; T1
        promotion is a preset decision recorded in :attr:`resident_extra`.
        """

        if self.policy == RESIDENT_POLICY:
            return self.names
        return tuple(tier.name for tier in self.tiers if tier.level == 0 or tier.name in self.resident_extra)

    @property
    def resident_rotations(self) -> frozenset[int]:
        keys: set[int] = set()
        for name in self.resident_names:
            keys.update(self.tier(name).rotations)
        return frozenset(keys)

    @property
    def union(self) -> tuple[int, ...]:
        keys: set[int] = set()
        for tier in self.tiers:
            keys.update(tier.rotations)
        return tuple(sorted(keys))

    def limb_limits(self) -> dict[int, int]:
        """``rotation -> Q limbs`` for ``CKKSContextSpec.rotation_key_limb_limits``.

        A rotation used by several tiers gets the largest cap; rotations in
        any uncapped tier are left out (full chain).
        """

        caps: dict[int, int] = {}
        uncapped: set[int] = set()
        for tier in self.tiers:
            if tier.limb_cap is None:
                uncapped.update(tier.rotations)
                continue
            for rotation in tier.rotations:
                caps[rotation] = max(int(tier.limb_cap), caps.get(rotation, 0))
        return {rotation: cap for rotation, cap in sorted(caps.items()) if rotation not in uncapped}

    def exclusive_rotations(self, name: str) -> tuple[int, ...]:
        """Rotations of ``name`` that no earlier tier (in :data:`TIER_ORDER`) already holds."""

        target = self.tier(name)
        earlier: set[int] = set()
        for tier in self.tiers:
            if tier.name == target.name:
                break
            earlier.update(tier.rotations)
        return tuple(sorted(target.rotation_set - earlier))

    def estimate_bytes(self, name: str, *, alpha: int, special_limbs: int, limbs: int | None = None) -> int:
        """Device bytes of the whole tier at ``limbs`` (default: its cap or the full chain)."""

        tier = self.tier(name)
        use = tier.limb_cap if limbs is None else int(limbs)
        if use is None:
            use = int(self.full_limbs)
        per_key = rotation_key_bytes(ring_dim=self.ring_dim, limbs=int(use), alpha=int(alpha), special_limbs=int(special_limbs))
        return per_key * len(tier)

    def summary(self, *, alpha: int | None = None, special_limbs: int | None = None) -> dict[str, Any]:
        rows = []
        for tier in self.tiers:
            row: dict[str, Any] = {
                "tier": tier.name,
                "level": tier.level,
                "keys": len(tier),
                "exclusive_keys": len(self.exclusive_rotations(tier.name)),
                "limb_cap": tier.limb_cap,
                "resident": tier.name in self.resident_names,
            }
            if alpha is not None and special_limbs is not None:
                row["est_gib"] = self.estimate_bytes(tier.name, alpha=alpha, special_limbs=special_limbs) / GIB
            rows.append(row)
        return {
            "policy": self.policy,
            "union_keys": len(self.union),
            "resident_keys": len(self.resident_rotations),
            "capped_keys": len(self.limb_limits()),
            "tiers": rows,
        }


def build_key_plan(
    preset,
    feature_layout,
    attention_layout,
    *,
    tiers: Iterable[str] = TIER_ORDER,
    schedule=None,
    limb_caps: Mapping[str, int] | None = None,
    resident_extra: Iterable[str] = (),
    policy: str | None = None,
    ownership=None,
    extra_rotations: Mapping[str, Iterable[int]] | None = None,
    emit_kv_cache: bool = False,
    prefix_k: int = 0,
) -> KeyPlan:
    """Ask the production rotation planners for every requested tier.

    ``limb_caps`` maps tier name -> Q-limb cap (host key trimming +
    prefetch limbs).  Nothing is capped by default: the first single-card
    probe runs with full keys and the tracer's ``max_limbs`` per tier is what
    the presets record afterwards.

    The ``attn_input_*`` phase tiers need ``ownership`` (a
    :class:`~.ownership.RankOwnership`): Q targets are its query pairs, the K
    pair / V carrier follow the GQA mapping and the V partial is restricted to
    its own lane sources.  ``extra_rotations`` (tier name -> signed offsets)
    is merged into the named tiers before normalisation; the single-card
    verification probe uses it to hold every rank's V-phase keys at once.

    ``emit_kv_cache`` plans the Decode-cache emission of the multi-card
    Prefill (:func:`~.stations.attention_input_exchange` with
    ``emit_kv_cache=True`` + :func:`~.kv_emit.prefill_kv_cache_station`): the
    ``K_C`` expand on ``attn_input_k``, the own-source ``V_C`` gather plan on
    ``attn_input_v`` and the PairFold conversion rotations (half payload,
    conjugate) on both.

    ``prefix_k`` adds the plaintext Prefix QK/PV and shared Softmax rotations
    to their respective tiers.  Prefix heads are sliced to this rank's KV
    pairs before the attention core.  Compact-specific Softmax rotations are
    added only when the rank-local compact Prefix rows fit.
    """

    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.operators.attention.audit import delta_pv_rotations, delta_qk_rotations
    from llama3fhe.operators.attention.distributed.distributed_input import attention_input_shard_rotations
    from llama3fhe.operators.attention.prefill.input import complex_token_attention_rotations
    from llama3fhe.operators.attention.prefill.operator import (
        delta_attention_rotations,
        delta_softmax_rotations,
        pairfm_output_rotations,
    )
    from llama3fhe.operators.attention.prefix.prefix_pc import (
        prefix_pc_pv_rotations,
        prefix_pc_qk_rotations,
        prefix_pc_rotations,
    )
    from llama3fhe.operators.attention.prefix.prefix_softmax import (
        prefix_softmax_rotations,
    )
    from llama3fhe.schedule import LayerSchedule

    from ..worker import resident_rotations

    wanted = tuple(str(name) for name in tiers)
    unknown = set(wanted) - set(TIER_ORDER)
    if unknown:
        raise ValueError(f"unknown tiers {sorted(unknown)}; choose from {TIER_ORDER}")
    caps = {str(k): int(v) for k, v in (limb_caps or {}).items()}
    extras = {str(k): tuple(int(v) for v in values) for k, values in (extra_rotations or {}).items()}
    unknown_extra = set(extras) - set(wanted)
    if unknown_extra:
        raise ValueError(f"extra_rotations name tiers not in the plan: {sorted(unknown_extra)}")
    schedule = LayerSchedule() if schedule is None else schedule
    ring_dim = 1 << int(preset.log_n)
    prefix_k = int(prefix_k)
    if prefix_k < 0:
        raise ValueError(f"prefix_k must be non-negative, got {prefix_k}")
    prefix_layout = attention_layout
    if prefix_k and ownership is not None:
        prefix_layout = replace(
            attention_layout,
            query_heads=2 * len(tuple(ownership.query_pair_indices)),
            key_value_heads=2
            * len(
                {
                    int(q) // int(attention_layout.gqa_ratio)
                    for q in ownership.query_pair_indices
                }
            ),
        )
    if prefix_k > int(attention_layout.head_dim):
        raise ValueError(
            f"prefix_k={prefix_k} exceeds head_dim={attention_layout.head_dim}"
        )
    compact_prefix_k = (
        prefix_k
        if int(prefix_layout.query_pair_count) * prefix_k <= int(prefix_layout.head_dim)
        else 0
    )

    families = []
    if "bootstrap" in wanted:
        families.append("bootstrap")
    if "rmsnorm" in wanted:
        families.append("rmsnorm")
    if "linear" in wanted:
        families.append("qkv")
    plan = resident_rotations(preset, feature_layout, attention_layout, families=tuple(families), schedule=schedule)

    def norm(values, name: str | None = None) -> tuple[int, ...]:
        merged = tuple(values) + (extras.get(name, ()) if name is not None else ())
        return tuple(sorted({normalize_rotation(int(v), ring_dim=ring_dim) for v in merged if int(v) != 0}))

    built: list[KeyTier] = []
    if "bootstrap" in wanted:
        built.append(KeyTier("bootstrap", 0, norm(plan.bootstrap, "bootstrap"), caps.get("bootstrap")))
    if "rmsnorm" in wanted:
        built.append(KeyTier("rmsnorm", 0, norm(plan.rmsnorm, "rmsnorm"), caps.get("rmsnorm")))
    if "linear" in wanted:
        # QKV, W_O and MLP request the same diagonal set (rot_tiers probe); the
        # planner is asked for QKV only and the tier is named for all three.
        built.append(KeyTier("linear", 1, norm(plan.qkv, "linear"), caps.get("linear")))
    if set(wanted) & set(T2_TIERS):
        carrier = LinearCarrierLayout(
            seq_len=int(feature_layout.seq_len),
            dimension=int(feature_layout.hidden_dim),
            slots=int(feature_layout.slots),
        )
        operator_config = AttentionOperatorConfig(
            sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
            shear_baby_steps=int(schedule.shear_baby_steps),
            pv_complex_delta_folding=bool(schedule.pv_complex_delta_folding),
            pv_prefold_after_softmax=bool(schedule.pv_prefold_after_softmax),
        )
        log_n = int(preset.log_n)
        if "attn_input" in wanted:
            built.append(
                KeyTier(
                    "attn_input",
                    2,
                    norm(
                        complex_token_attention_rotations(
                            carrier,
                            attention_layout,
                            sparse_max_baby_offsets=int(operator_config.sparse_max_baby_offsets),
                            shear_baby_steps=int(operator_config.shear_baby_steps),
                            log_n=log_n,
                        ),
                        "attn_input",
                    ),
                    caps.get("attn_input"),
                )
            )
        phase_names = tuple(name for name in ATTN_INPUT_PHASE_TIERS if name in wanted)
        if phase_names:
            if ownership is None:
                raise ValueError(f"tiers {phase_names} need the rank ownership")
            from .ownership import key_pairs_for_rank

            ratio = int(attention_layout.gqa_ratio)
            all_carriers = tuple(range(int(attention_layout.key_value_pair_count) // 2))
            phases = attention_input_shard_rotations(
                carrier,
                attention_layout,
                query_pair_indices=tuple(ownership.query_pair_indices),
                key_pair_indices=key_pairs_for_rank(ownership, gqa_ratio=ratio),
                value_carrier_indices=all_carriers,
                value_source_indices=tuple(ownership.cipher_indices),
                sparse_max_baby_offsets=int(operator_config.sparse_max_baby_offsets),
                shear_baby_steps=int(operator_config.shear_baby_steps),
                log_n=log_n,
                emit_cache_c=bool(emit_kv_cache),
            )
            if emit_kv_cache:
                from llama3fhe.layouts.pairfold128 import PairFold128Layout
                from llama3fhe.operators.attention.pairfold.pairfold_prefill import prefill_c_to_pairfold_rotations

                # prefill_kv_cache_station converts the gathered K_C / V_C to PairFold on
                # every rank: half-payload shift + conjugate (folded K_C -> real C).
                convert = tuple(prefill_c_to_pairfold_rotations(PairFold128Layout()))
                phases = dict(phases)
                phases["key"] = tuple(sorted(set(phases["key"]) | set(convert)))
                phases["value"] = tuple(sorted(set(phases["value"]) | set(convert)))
            phase_of = {"attn_input_lane": "lane", "attn_input_q": "query", "attn_input_k": "key", "attn_input_v": "value"}
            for name in phase_names:
                built.append(KeyTier(name, 2, norm(phases[phase_of[name]], name), caps.get(name)))
        if "attn_core" in wanted:
            built.append(
                KeyTier(
                    "attn_core",
                    2,
                    norm(
                        delta_attention_rotations(
                            attention_layout,
                            include_qk_feature_fold=bool(operator_config.qk_complex_feature_folding),
                            qk_key_features_pre_folded=bool(operator_config.qk_key_features_pre_folded),
                            qk_query_features_pre_folded=bool(operator_config.qk_query_features_pre_folded),
                            include_pv_delta_fold=bool(operator_config.pv_complex_delta_folding),
                            pv_probability_pre_folded=bool(operator_config.pv_prefold_after_softmax),
                            pv_value_pre_folded=bool(operator_config.pv_prefold_after_softmax),
                            log_n=log_n,
                        ) + (prefix_pc_rotations(prefix_layout, prefix_k=prefix_k) if prefix_k else ()),
                        "attn_core",
                    ),
                    caps.get("attn_core"),
                )
            )
        if "attn_qk" in wanted:
            built.append(
                KeyTier(
                    "attn_qk",
                    2,
                    norm(
                        delta_qk_rotations(
                            attention_layout,
                            complex_feature_folding=bool(operator_config.qk_complex_feature_folding),
                            key_features_pre_folded=bool(operator_config.qk_key_features_pre_folded),
                            query_features_pre_folded=bool(operator_config.qk_query_features_pre_folded),
                            log_n=log_n,
                        ) + (prefix_pc_qk_rotations(prefix_layout, prefix_k=prefix_k) if prefix_k else ()),
                        "attn_qk",
                    ),
                    caps.get("attn_qk"),
                )
            )
        if "attn_pv" in wanted:
            built.append(
                KeyTier(
                    "attn_pv",
                    2,
                    norm(
                        delta_pv_rotations(
                            attention_layout,
                            complex_delta_folding=bool(operator_config.pv_complex_delta_folding),
                            probability_pre_folded=bool(operator_config.pv_prefold_after_softmax),
                            value_pre_folded=bool(operator_config.pv_prefold_after_softmax),
                            log_n=log_n,
                        ) + (prefix_pc_pv_rotations(prefix_layout, prefix_k=prefix_k) if prefix_k else ()),
                        "attn_pv",
                    ),
                    caps.get("attn_pv"),
                )
            )
        if "softmax" in wanted:
            built.append(
                KeyTier(
                    "softmax", 2,
                    norm(
                        delta_softmax_rotations(attention_layout)
                        + (
                            prefix_softmax_rotations(
                                prefix_layout, prefix_k=compact_prefix_k
                            )
                            if prefix_k
                            else ()
                        ),
                        "softmax",
                    ),
                    caps.get("softmax"),
                )
            )
        if "pairfm_output" in wanted:
            built.append(
                KeyTier(
                    "pairfm_output",
                    2,
                    norm(
                        pairfm_output_rotations(
                            feature_layout,
                            attention_layout,
                            first_baby_steps=int(operator_config.pairfm_first_baby_steps),
                            block_baby_steps=int(operator_config.pairfm_block_baby_steps),
                            log_n=log_n,
                        ),
                        "pairfm_output",
                    ),
                    caps.get("pairfm_output"),
                )
            )
    return KeyPlan(
        tiers=tuple(built),
        policy=str(preset.key_policy if policy is None else policy),
        ring_dim=ring_dim,
        full_limbs=int(preset.depth) + 1,
        resident_extra=tuple(str(name) for name in resident_extra),
    )


# --------------------------------------------------------------- lifecycle
@dataclass
class LeaseRecord:
    tier: str
    prefetch_seconds: float = 0.0
    prefetched_keys: int = 0
    evict_seconds: float = 0.0
    evicted_entries: int = 0
    retained_entries: int = 0
    #: async path (:attr:`KeyLifecycle.prefetch_mode` ``"async"``): keys of
    #: this tier that were already in flight when the lease started, the host
    #: time spent waiting for that copy at lease entry, and what this lease
    #: issued for the *next* tier (``None`` = nothing issued, see ``skipped``).
    async_keys: int = 0
    async_gib: float = 0.0
    async_wait_seconds: float = 0.0
    async_issued_tier: str | None = None
    async_issued_keys: int = 0
    async_issued_gib: float = 0.0
    async_issue_seconds: float = 0.0
    async_skipped: str | None = None


@dataclass
class PrefetchRecord:
    """One in-flight asynchronous tier upload (side stream + completion event)."""

    tier: str
    keys: int
    nbytes: int
    limbs: dict[int, int]
    event: Any
    issued_at: float
    holders: list = field(default_factory=list)


class PinnedCompactKeys:
    """Leased rotation keys pre-compacted to their tier cap in pinned host memory.



    ``prepare`` builds the compact tensors once per process (default: every
    tier the plan does not keep resident, at the rotation's largest cap).
    ``install`` shadows ``native._load_rotation_key_to_cuda``: a rotation with
    a pinned copy is served from it -- exactly the requested ``(limbs, beta)``
    slice so EasyFHE's cache keys and eviction stay unchanged -- and any other
    rotation falls through to the original.  Install *before* a
    :class:`KeyTracer` so the tracer still books misses, bytes and seconds.
    
    """

    def __init__(
        self, native, plan: KeyPlan, *, device: str = "cuda", pin_memory: bool = True, drop_host_copy: bool = True
    ) -> None:
        self.native = native
        self.plan = plan
        self.device = str(device)
        self.pin_memory = bool(pin_memory)
        self.drop_host_copy = bool(drop_host_copy)
        self.host_bytes_dropped = 0
        self.source_entries_dropped = 0
        self.entries: dict[int, tuple[Any, Any, int, int]] = {}
        self.prepare_seconds = 0.0
        self.hits = 0
        self.misses = 0
        self.bytes_copied = 0
        self._installed = False
        self._original: tuple[bool, Any] | None = None

    # -- host side
    def default_tiers(self) -> tuple[str, ...]:
        resident = set(self.plan.resident_names)
        return tuple(t.name for t in self.plan.tiers if t.name not in resident and t.limb_cap is not None)

    @property
    def nbytes(self) -> int:
        return sum(
            int(bx.numel()) * int(bx.element_size()) + int(ax.numel()) * int(ax.element_size())
            for bx, ax, _, _ in self.entries.values()
        )

    @property
    def gib(self) -> float:
        return self.nbytes / GIB

    def prepare(self, tiers: Iterable[str] | None = None) -> dict[str, Any]:
        """Compact + pin every rotation of ``tiers`` (default :meth:`default_tiers`).

        The P-limb offset follows EasyFHE's ``_rotation_key_special_mod_start``
        (``min(limb_limit, host_q_limbs)``) so the pinned layout equals what
        ``_compact_rotation_key_component_to_cuda`` would produce per load.
        """

        native = self.native
        key_map = getattr(native, "left_rot_key_map", None)
        if key_map is None:
            return {"tiers": (), "keys": 0, "gib": 0.0, "seconds": 0.0}
        names = tuple(self.default_tiers() if tiers is None else (str(t) for t in tiers))
        caps = self.plan.limb_limits()
        alpha = int(native.alpha)
        special = int(native.K)
        started = time.perf_counter()
        wanted: set[int] = set()
        for name in names:
            wanted.update(int(r) for r in self.plan.tier(name).rotations)
        # A rotation a resident tier also holds is served from the resident
        # (full-chain) CUDA copy by EasyFHE's dominance rule; pinning it would
        # only add an uncapped host copy that is never uploaded.
        wanted.difference_update(int(r) for r in self.plan.resident_rotations)
        # ``Context.__init__`` keeps the runtime material as ``_source_material``
        # (for ``construct_copy``); its ``left_rot_key_map`` lists hold the very
        # same host tensors, so dropping the host copy means replacing both.
        source_map = getattr(getattr(native, "_source_material", None), "left_rot_key_map", None)
        for rotation in sorted(wanted):
            if rotation in self.entries or rotation not in key_map:
                continue
            bx, ax = key_map[rotation]
            available = int(bx.shape[1]) - special
            special_mod_start = getattr(native, "_rotation_key_special_mod_start", None)
            if special_mod_start is not None:
                available = int(special_mod_start(rotation))
            cap = min(int(caps.get(rotation, available)), available)
            beta = (cap + alpha - 1) // alpha
            entry = (
                self._compact(bx, cap, beta, available, special, self.pin_memory),
                self._compact(ax, cap, beta, available, special, self.pin_memory),
                cap,
                beta,
            )
            self.entries[rotation] = entry
            if self.drop_host_copy and not bool(bx.is_cuda):
                self.host_bytes_dropped += int(bx.numel()) * int(bx.element_size()) + int(ax.numel()) * int(ax.element_size())
                key_map[rotation] = [entry[0], entry[1]]
                if source_map is not None:
                    source_pair = source_map.get(rotation)
                    if source_pair is not None and (source_pair[0] is bx or source_pair[1] is ax):
                        source_map[rotation] = [entry[0], entry[1]]
                        self.source_entries_dropped += 1
            bx = ax = None
        self.prepare_seconds += time.perf_counter() - started
        return {
            "tiers": names,
            "keys": len(self.entries),
            "gib": self.gib,
            "seconds": self.prepare_seconds,
            "host_gib_dropped": self.host_bytes_dropped / GIB,
            "source_entries_dropped": self.source_entries_dropped,
        }

    @staticmethod
    def _compact(component, cap: int, beta: int, available: int, special: int, pin_memory: bool = True):
        import torch

        source = component[:beta]
        out = torch.empty(
            (int(beta), int(cap) + int(special), int(source.shape[2])), dtype=source.dtype, pin_memory=bool(pin_memory)
        )
        torch.cat((source[:, :cap, :], source[:, available:available + special, :]), dim=1, out=out)
        return out

    # -- device side
    def install(self) -> "PinnedCompactKeys":
        if self._installed:
            return self
        native = self.native
        original = native._load_rotation_key_to_cuda
        cache = self

        def pinned_load(rot_index, special_mod_start, beta, available_special_mod_start):
            entry = cache.entries.get(int(rot_index))
            if entry is None or int(special_mod_start) > entry[2] or int(beta) > entry[3]:
                cache.misses += 1
                return original(rot_index, special_mod_start, beta, available_special_mod_start)
            cache.hits += 1
            return [
                cache._upload(entry[0], int(special_mod_start), int(beta), entry[2]),
                cache._upload(entry[1], int(special_mod_start), int(beta), entry[2]),
            ]

        self._original = ("_load_rotation_key_to_cuda" in vars(native), vars(native).get("_load_rotation_key_to_cuda"))
        native._load_rotation_key_to_cuda = pinned_load
        self._installed = True
        return self

    #: set by :meth:`KeyLifecycle._prefetch_async` around its side-stream
    #: uploads: the copies are issued ``non_blocking`` on the current (side)
    #: stream into buffers allocated on the compute stream (see ``_upload``).
    async_copy: bool = False

    def _upload(self, pinned, limbs: int, beta: int, cap: int):
        """Device copy of ``pinned`` trimmed to ``(beta, limbs + K)``.

        Every ``copy_`` below is contiguous on both sides (whole rows of one
        digit), so torch issues plain ``cudaMemcpy`` from the pinned buffer.
        A strided view such as ``pinned[:beta, :limbs]`` would be staged into a
        pageable host temporary first, which is the slow path being avoided.

        The destination is always allocated on the *default* (compute)
        stream: the caching allocator pools blocks per stream, so a buffer
        allocated on the prefetch side stream could never be reused by the
        operators once the key is evicted.  With ``async_copy`` the copies are
        enqueued on the current stream (the lifecycle's side stream) and
        ``non_blocking``; the caller orders them with an event.
        """

        import torch

        special = int(pinned.shape[1]) - int(cap)
        non_blocking = bool(self.async_copy)
        if non_blocking:
            default = torch.cuda.default_stream(self.device)
            with torch.cuda.stream(default):
                device = torch.empty((beta, limbs + special, int(pinned.shape[2])), dtype=pinned.dtype, device=self.device)
        else:
            device = torch.empty((beta, limbs + special, int(pinned.shape[2])), dtype=pinned.dtype, device=self.device)
        if limbs == cap:
            device.copy_(pinned[:beta], non_blocking=non_blocking)
        else:
            for digit in range(int(beta)):
                device[digit, :limbs].copy_(pinned[digit, :limbs], non_blocking=non_blocking)
                device[digit, limbs:].copy_(pinned[digit, cap:cap + special], non_blocking=non_blocking)
        self.bytes_copied += int(device.numel()) * int(device.element_size())
        return device

    def uninstall(self) -> None:
        if not self._installed:
            return
        was_instance, previous = self._original or (False, None)
        if was_instance:
            setattr(self.native, "_load_rotation_key_to_cuda", previous)
        else:
            try:
                delattr(self.native, "_load_rotation_key_to_cuda")
            except AttributeError:
                pass
        self._installed = False

    def release(self) -> None:
        self.uninstall()
        self.entries.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "keys": len(self.entries),
            "gib": self.gib,
            "host_gib_dropped": self.host_bytes_dropped / GIB,
            "source_entries_dropped": self.source_entries_dropped,
            "prepare_seconds": self.prepare_seconds,
            "hits": int(self.hits),
            "misses": int(self.misses),
            "gib_copied": self.bytes_copied / GIB,
        }


#: Which leased tier a station will need next, keyed by the tier whose lease
#: is starting (:attr:`KeyLifecycle.prefetch_mode` ``"async"``): the T2 layer
#: is ``qkv (linear) -> attn_input lane -> V partial -> [exchange] -> V finish
#: -> Q -> K -> attn_qk -> softmax -> attn_pv -> pairfm_output -> w_o (linear)``
#: (:mod:`.stations`).  ``softmax`` has no exclusive keys (its 24 rotations are
#: all in ``attn_qk``), so ``attn_qk`` chains straight to ``attn_pv``.
#: ``linear`` is leased by qkv, w_o and mlp alike; only the qkv lease benefits
#: (the lane keys prefetched at w_o / mlp just sit through the MLP, which is
#: harmless: they are what the next layer's qkv->lane needs anyway).
PREFETCH_CHAIN: dict[str, str] = {
    "linear": "attn_input_lane",
    "attn_input_lane": "attn_input_v",
    "attn_input_v": "attn_input_q",
    "attn_input_q": "attn_input_k",
    "attn_input_k": "attn_qk",
    "attn_qk": "attn_pv",
    "attn_pv": "pairfm_output",
}

PREFETCH_MODES = ("lazy", "sync", "async")

#: How :meth:`KeyLifecycle.evict` orders the drop against in-flight GPU work.
#: ``"device"`` is EasyFHE's own ``clear_cuda_rotation_cache`` (a full
#: ``torch.cuda.synchronize`` + ``torch.cuda.empty_cache``); ``"stream"``
#: waits for the compute stream only, so a prefetch copy still running on the
#: side stream is not joined, and leaves the freed blocks in the caching
#: allocator (the caching allocator's stream-ordered frees make the wait
#: itself optional; it is kept so the evict cost is booked where it belongs).
EVICT_SYNC_MODES = ("device", "stream")


class KeyLifecycle:
    """Prefetch / evict / lease against one native EasyFHE context.

    ``native`` is ``EasyFHEContext.context``.  Under the ``resident`` policy
    ``lease`` is a no-op (every key stays once loaded), so the A100 preset
    runs the same station code with zero eviction traffic.

    ``prefetch_mode`` (``lease`` policy only):

    * ``"lazy"`` -- keys load on first use inside the operators (synchronous
      H2D on the compute stream, one ``cudaMemcpy`` per miss).
    * ``"sync"`` -- :meth:`lease` uploads the whole tier at entry when asked
      (``prefetch=True``); still on the compute stream, still blocking.
    * ``"async"`` -- when a lease starts, the *next* tier of
      :data:`PREFETCH_CHAIN` is uploaded on a side CUDA stream while this
      station computes; the next lease waits on the completion event instead
      of copying.  The upload is trimmed to the limbs each rotation was seen
      to need in an earlier lease of that tier (:attr:`profile`, learnt from
      the requests; the first pass is lazy) and is skipped when the device
      does not have ``nbytes + prefetch_margin_gib`` free at issue time.
    """

    def __init__(
        self,
        native,
        plan: KeyPlan,
        *,
        tracer: "KeyTracer | None" = None,
        synchronize=None,
        prefetch_mode: str = "lazy",
        prefetch_margin_gib: float = 1.5,
        evict_sync: str = "device",
        evict_empty_cache: bool | None = None,
        device: str | None = None,
    ):
        self.native = native
        self.plan = plan
        self.tracer = tracer
        self._active: list[str] = []
        self._synchronize = synchronize
        self.records: list[LeaseRecord] = []
        self.pinned: PinnedCompactKeys | None = None
        if str(prefetch_mode) not in PREFETCH_MODES:
            raise ValueError(f"unknown prefetch_mode {prefetch_mode!r}; choose from {PREFETCH_MODES}")
        self.prefetch_mode = str(prefetch_mode)
        self.prefetch_margin_gib = float(prefetch_margin_gib)
        if str(evict_sync) not in EVICT_SYNC_MODES:
            raise ValueError(f"unknown evict_sync {evict_sync!r}; choose from {EVICT_SYNC_MODES}")
        self.evict_sync = str(evict_sync)
        #: ``torch.cuda.empty_cache()`` after the drop (default: only under ``"device"``).
        self.evict_empty_cache = (self.evict_sync == "device") if evict_empty_cache is None else bool(evict_empty_cache)
        self.device = device
        #: tier -> rotation -> largest Q-limb count any request of that tier asked for.
        self.profile: dict[str, dict[int, int]] = {}
        self.pending: dict[str, PrefetchRecord] = {}
        self.prefetch_log: list[dict[str, Any]] = []
        self._side_stream = None
        self._profile_hook_installed = False
        self._profile_original = None

    def pin_leased_keys(self, tiers: Iterable[str] | None = None) -> dict[str, Any]:
        """Build and install the :class:`PinnedCompactKeys` cache (call before the tracer)."""

        if self.pinned is None:
            self.pinned = PinnedCompactKeys(self.native, self.plan)
        report = self.pinned.prepare(tiers)
        self.pinned.install()
        return report

    def release(self) -> None:
        self.drain()
        if self.pinned is not None:
            self.pinned.release()

    # -- state
    @property
    def active_tiers(self) -> tuple[str, ...]:
        return tuple(self._active)

    def keep_set(self) -> set[int]:
        """Rotations that must survive the next ``evict``: resident + active leases + in-flight prefetches."""

        keep = set(self.plan.resident_rotations)
        for name in self._active:
            keep.update(self.plan.tier(name).rotations)
        for name in self.pending:
            keep.update(self.plan.tier(name).rotations)
        return keep

    # -- primitives
    def prefetch(
        self,
        tier: str,
        *,
        limbs: int | None = None,
        rotations: Iterable[int] | None = None,
    ) -> tuple[int, float]:
        """Upload keys of ``tier`` compacted to ``limbs`` (default: cap or full).

        ``rotations`` optionally narrows the upload to a phase-specific subset
        of the tier.  The normal station path leaves it unset and retains the
        original all-tier behavior.  Combined Prefill + Decode workers use the
        subset form to avoid loading Decode-only rotations before Prefill has
        reached its peak working set.

        Returns ``(keys_touched, seconds)``.  Uses the same
        ``get_rotation_key_for_limbs`` path the operators use, so a later
        request at ``<= limbs`` is a cache hit.
        """

        record = self.plan.tier(tier)
        use = record.limb_cap if limbs is None else int(limbs)
        if use is None:
            use = int(self.plan.full_limbs)
        requested = (
            record.rotations
            if rotations is None
            else tuple(sorted({int(v) for v in rotations if int(v) != 0}))
        )
        getter = getattr(self.native, "get_rotation_key_for_limbs", None)
        if getter is None:
            return 0, 0.0
        started = time.perf_counter()
        count = 0
        for rotation in requested:
            getter(int(rotation), int(use))
            count += 1
        self._sync()
        seconds = time.perf_counter() - started
        if self.tracer is not None:
            self.tracer.note_prefetch(tier, count, seconds)
        return count, seconds

    def evict(self, *, keep_extra: Iterable[int] = ()) -> dict[str, Any]:
        """Drop device copies of everything outside ``keep_set() | keep_extra``."""

        keep = self.keep_set() | {int(v) for v in keep_extra}
        started = time.perf_counter()
        if self.evict_sync == "device":
            clear = getattr(self.native, "clear_cuda_rotation_cache", None)
            if clear is None:
                return {}
            stats = clear(keep_rotations=tuple(sorted(keep)), empty_allocator_cache=bool(self.evict_empty_cache)) or {}
        else:
            stats = self._evict_stream_ordered(keep)
        seconds = time.perf_counter() - started
        stats = dict(stats)
        stats["seconds"] = seconds
        if self.tracer is not None:
            self.tracer.note_evict(stats)
        return stats

    def _evict_stream_ordered(self, keep: set[int]) -> dict[str, Any]:
        """``clear_cuda_rotation_cache`` semantics, but joined to the compute stream only.

        Same caches, same ``keep_rotations`` normalisation, same counters.
        The device copies are dropped after ``default_stream.synchronize()``
        (the kernels that read them have all been issued on that stream), so a
        prefetch copy in flight on the side stream keeps running.  The blocks
        stay in the caching allocator unless :attr:`evict_empty_cache`.
        """

        native = self.native
        ring = int(getattr(native, "N", self.plan.ring_dim))
        keep_indices = {normalize_rotation(int(v), ring_dim=ring) for v in keep}
        try:
            import torch

            device = self._device()
            torch.cuda.default_stream(device).synchronize()
        except Exception:
            self._sync()

        def rotation_from_key(key):
            candidate = key[0] if isinstance(key, tuple) and key else key
            try:
                return normalize_rotation(int(candidate), ring_dim=ring)
            except (TypeError, ValueError):
                return None

        stats: dict[str, Any] = {}
        for name in ("_rotation_key_cuda_cache", "_precompute_auto_cuda_cache", "_inverse_precompute_auto_cuda_cache"):
            cache = getattr(native, name, None)
            before = len(cache) if isinstance(cache, dict) else 0
            if isinstance(cache, dict):
                for key in tuple(cache):
                    rotation = rotation_from_key(key)
                    if rotation is None or rotation not in keep_indices:
                        del cache[key]
            stats[f"{name}_entries"] = int(before)
            stats[f"{name}_retained_entries"] = int(len(cache)) if isinstance(cache, dict) else 0
        batch_cache = getattr(native, "precompute_auto_maps_cache", None)
        before = len(batch_cache) if isinstance(batch_cache, dict) else 0
        if isinstance(batch_cache, dict):
            for key in tuple(batch_cache):
                offsets = key[0] if isinstance(key, tuple) and key else ()
                if isinstance(offsets, int):
                    offsets = (int(offsets),)
                try:
                    drop = not offsets or any(normalize_rotation(int(o), ring_dim=ring) not in keep_indices for o in offsets)
                except (TypeError, ValueError):
                    drop = True
                if drop:
                    del batch_cache[key]
        stats["precompute_auto_maps_cache_entries"] = int(before)
        stats["precompute_auto_maps_cache_retained_entries"] = int(len(batch_cache)) if isinstance(batch_cache, dict) else 0
        if self.evict_empty_cache:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        stats["cpu_master_rotation_keys_retained"] = int(len(getattr(native, "left_rot_key_map", {})))
        stats["requested_rotation_indices_kept"] = int(len(keep_indices))
        stats["evict_sync"] = self.evict_sync
        return stats

    @contextlib.contextmanager
    def lease(self, tier: str, *, limbs: int | None = None, prefetch: bool = True, evict_after: bool | None = None) -> Iterator[LeaseRecord]:
        """Bracket one station (or chunk) that uses ``tier``.

        Under ``lease`` policy: prefetch on entry (unless the tier is
        resident or ``prefetch=False``), evict on exit unless the tier is
        resident or ``evict_after=False``.  Nested leases keep their outer
        tiers alive.  Under ``resident`` policy nothing is evicted.
        """

        name = str(tier)
        record = LeaseRecord(tier=name)
        resident = name in self.plan.resident_names
        do_evict = (self.plan.policy == LEASE_POLICY and not resident) if evict_after is None else bool(evict_after)
        self._active.append(name)
        try:
            if self.prefetch_mode == "async" and self.plan.policy == LEASE_POLICY:
                self._await_pending(name, record)
                self._issue_next(name, record)
            if prefetch and self.prefetch_mode != "async":
                record.prefetched_keys, record.prefetch_seconds = self.prefetch(name, limbs=limbs)
            yield record
        finally:
            self._active.remove(name)
            if do_evict:
                stats = self.evict()
                record.evict_seconds = float(stats.get("seconds", 0.0))
                record.evicted_entries = int(
                    stats.get("_rotation_key_cuda_cache_entries", 0)
                    - stats.get("_rotation_key_cuda_cache_retained_entries", 0)
                )
                record.retained_entries = int(stats.get("_rotation_key_cuda_cache_retained_entries", 0))
            self.records.append(record)

    def _sync(self) -> None:
        if self._synchronize is not None:
            self._synchronize()
            return
        try:
            # Loading a key is already ordered on the active stream.  This
            # sync exists only to make the tracer's ``load_seconds`` a device
            # time; honor the shared profile-only switch so ``T2_STAGE_SYNC=0``
            # does not reintroduce a hidden barrier on every lazy miss.
            from llama3fhe.backend import profile_sync_device, stage_sync_enabled

            if stage_sync_enabled():
                profile_sync_device(str(getattr(self.native, "device", "cuda")))
        except Exception:
            pass

    # -- async prefetch (prefetch_mode == "async")
    def install_profile_hook(self) -> "KeyLifecycle":
        """Learn ``profile`` from the operators' key requests while a lease is active.

        Wraps ``native.get_rotation_key_for_limbs``; install *after* the
        :class:`KeyTracer` so this wrapper stays outermost for the whole run.
        Idempotent.
        """

        if self._profile_hook_installed:
            return self
        native = self.native
        lifecycle = self
        original = native.get_rotation_key_for_limbs
        rotation_tiers: dict[int, tuple[str, ...]] = {}
        for tier in self.plan.tiers:
            if tier.name in self.plan.resident_names:
                continue
            for rotation in tier.rotations:
                rotation_tiers[int(rotation)] = rotation_tiers.get(int(rotation), ()) + (tier.name,)

        def profiled_get(rot_index, cur_limbs):
            rot = int(rot_index)
            names = rotation_tiers.get(rot)
            if names:
                limbs = int(cur_limbs)
                for name in names:
                    if name in lifecycle._active:
                        seen = lifecycle.profile.setdefault(name, {})
                        if seen.get(rot, 0) < limbs:
                            seen[rot] = limbs
            return original(rot_index, cur_limbs)

        self._profile_original = ("get_rotation_key_for_limbs" in vars(native), vars(native).get("get_rotation_key_for_limbs"))
        native.get_rotation_key_for_limbs = profiled_get
        self._profile_hook_installed = True
        return self

    def _device(self):
        import torch

        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda", torch.cuda.current_device())

    def _side(self):
        import torch

        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(device=self._device())
        return self._side_stream

    def _raw_loader(self):
        """The ``_load_rotation_key_to_cuda`` callable *beneath* the tracer.

        The tracer's wrapper ends every load with a full device synchronize
        (that is how it times lazy misses), which would serialise the side
        stream; the async path therefore calls whatever the tracer shadowed --
        the :class:`PinnedCompactKeys` loader when pinning is on, otherwise
        EasyFHE's own (pageable, synchronous) copy.
        """

        native = self.native
        tracer = self.tracer
        if tracer is not None and getattr(tracer, "_installed", False):
            was_instance, previous = tracer._originals.get("_load_rotation_key_to_cuda", (False, None))
            if was_instance and previous is not None:
                return previous
            return type(native)._load_rotation_key_to_cuda.__get__(native, type(native))
        return native._load_rotation_key_to_cuda

    def _key_nbytes(self, rotation: int, limbs: int) -> int:
        """Bytes of one compact rotation key ``(beta, limbs + K, N)`` x 2 components, int64."""

        alpha = int(getattr(self.native, "alpha", 1))
        special = int(getattr(self.native, "K", 0))
        ring = int(getattr(self.native, "N", self.plan.ring_dim))
        beta = (int(limbs) + alpha - 1) // alpha
        return 2 * beta * (int(limbs) + special) * ring * 8

    def prefetch_plan(self, tier: str) -> tuple[dict[int, int], int]:
        """``(rotation -> limbs, nbytes)`` the async path would upload for ``tier`` right now.

        Rotations without a learnt limb count are left out (they load lazily on
        first use); rotations already cached at >= the needed limbs are left out too.
        """

        record = self.plan.tier(tier)
        seen = self.profile.get(str(tier), {})
        cache = getattr(self.native, "_rotation_key_cuda_cache", {})
        # Rotations the *running* stations may request must not be touched
        # here: the async path publishes the cache entry before the copy has
        # landed, so a shared rotation would race on the compute stream.  They
        # load lazily (and the lease's evict keeps them for the next tier).
        busy: set[int] = set()
        for name in self._active:
            busy.update(int(v) for v in self.plan.tier(name).rotations)
        want: dict[int, int] = {}
        for rotation in record.rotations:
            if int(rotation) in busy:
                continue
            limbs = seen.get(int(rotation))
            if limbs is None:
                continue
            if record.limb_cap is not None:
                limbs = min(int(limbs), int(record.limb_cap))
            cached = any(
                int(key[0]) == int(rotation) and int(key[1]) >= int(limbs)
                for key in cache
                if isinstance(key, tuple) and len(key) >= 2
            )
            if cached:
                continue
            want[int(rotation)] = int(limbs)
        nbytes = sum(self._key_nbytes(rotation, limbs) for rotation, limbs in want.items())
        return want, nbytes

    def _issue_next(self, current: str, record: LeaseRecord) -> None:
        nxt = PREFETCH_CHAIN.get(str(current))
        if nxt is None or not self.plan.has_tier(nxt):
            return
        if nxt in self.plan.resident_names or nxt in self._active or nxt in self.pending:
            return
        want, nbytes = self.prefetch_plan(nxt)
        record.async_issued_tier = nxt
        if not want:
            record.async_skipped = "no_profile" if not self.profile.get(nxt) else "all_cached"
            return
        import torch

        device = self._device()
        free, _total = torch.cuda.mem_get_info(device)
        # blocks the caching allocator holds but has not handed out are free for us too
        free = int(free) + max(0, int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device)))
        margin = int(self.prefetch_margin_gib * GIB)
        if nbytes + margin > int(free):
            record.async_skipped = f"budget free={free / GIB:.2f} need={(nbytes + margin) / GIB:.2f}"
            self.prefetch_log.append({"tier": nxt, "issued": False, "reason": record.async_skipped, "keys": len(want), "gib": nbytes / GIB})
            return
        started = time.perf_counter()
        self.pending[nxt] = self._prefetch_async(nxt, want, nbytes)
        record.async_issued_keys = len(want)
        record.async_issued_gib = nbytes / GIB
        record.async_issue_seconds = time.perf_counter() - started
        self.prefetch_log.append({"tier": nxt, "issued": True, "keys": len(want), "gib": nbytes / GIB, "issue_s": record.async_issue_seconds, "free_gib": free / GIB})

    def _prefetch_async(self, tier: str, want: dict[int, int], nbytes: int) -> PrefetchRecord:
        """Upload ``want`` on the side stream and insert into EasyFHE's cache; returns the completion record."""

        import torch

        native = self.native
        device = self._device()
        side = self._side()
        default = torch.cuda.default_stream(device)
        cache = native._rotation_key_cuda_cache
        alpha = int(native.alpha)
        holders: list = []
        pinned = self.pinned
        # Order: the destination buffers are allocated on the compute stream
        # (see PinnedCompactKeys._upload); the side stream must not write into
        # them before the compute stream's already-queued work that might
        # still be reading a recycled block has passed.
        side.wait_stream(default)
        loader = self._raw_loader()
        if pinned is not None:
            pinned.async_copy = True
        try:
            with torch.cuda.stream(side):
                for rotation, limbs in want.items():
                    beta = (int(limbs) + alpha - 1) // alpha
                    available = int(native._rotation_key_special_mod_start(int(rotation)))
                    key = (int(rotation), int(limbs), int(beta))
                    native._evict_dominated_rotation_key_versions(key)
                    swk = loader(int(rotation), int(limbs), int(beta), available)
                    cached = [swk[0], swk[1]]
                    cache[key] = cached
                    holders.append(cached)
                event = torch.cuda.Event()
                event.record(side)
        finally:
            if pinned is not None:
                pinned.async_copy = False
        if self.tracer is not None:
            self.tracer.note_prefetch(str(tier), len(want), 0.0, nbytes=int(nbytes))
        return PrefetchRecord(tier=str(tier), keys=len(want), nbytes=int(nbytes), limbs=dict(want), event=event, issued_at=time.perf_counter(), holders=holders)

    def _await_pending(self, name: str, record: LeaseRecord) -> None:
        pending = self.pending.pop(str(name), None)
        if pending is None:
            return
        import torch

        started = time.perf_counter()
        # Make the compute stream wait for the copies (device-side ordering).
        # Do not call ``event.synchronize()`` here: that turns the side-stream
        # prefetch back into a host barrier immediately before the station and
        # removes the overlap this mode is meant to provide.  The rotation-key
        # entries are already published in the native cache, the active lease
        # keeps them in ``keep_set()``, and the stream dependency makes the
        # first consumer wait on the device.  ``drain()`` remains the teardown
        # barrier for an in-flight copy.
        pending.event.wait(torch.cuda.default_stream(self._device()))
        record.async_wait_seconds = time.perf_counter() - started
        record.async_keys = int(pending.keys)
        record.async_gib = float(pending.nbytes) / GIB
        self.prefetch_log.append({"tier": str(name), "awaited": True, "keys": pending.keys, "gib": pending.nbytes / GIB, "wait_s": record.async_wait_seconds, "in_flight_s": started - pending.issued_at})

    def drain(self) -> None:
        """Wait for every in-flight prefetch (call before tearing the context down)."""

        for pending in list(self.pending.values()):
            try:
                pending.event.synchronize()
            except Exception:
                pass
        self.pending.clear()


# ------------------------------------------------------------------ tracer
@dataclass
class StageKeyStats:
    stage: str
    requests: int = 0
    misses: int = 0
    loads: int = 0
    bytes_loaded: int = 0
    load_seconds: float = 0.0
    prefetch_keys: int = 0
    prefetch_seconds: float = 0.0
    prefetch_bytes: int = 0
    evictions: int = 0
    evict_seconds: float = 0.0
    rotations: set[int] = field(default_factory=set)
    max_limbs: dict[int, int] = field(default_factory=dict)
    #: first misses of the stage as ``(rotation, limbs, beta)`` (diagnostics for the prefetch planner).
    miss_log: list[tuple[int, int, int]] = field(default_factory=list)
    MISS_LOG_LIMIT = 512

    @property
    def cache_hits(self) -> int:
        return int(self.requests) - int(self.misses)

    def as_row(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "requests": int(self.requests),
            "hits": int(self.cache_hits),
            "misses": int(self.misses),
            "loads": int(self.loads),
            "bytes_loaded": int(self.bytes_loaded),
            "gib_loaded": self.bytes_loaded / GIB,
            "load_seconds": float(self.load_seconds),
            "prefetch_keys": int(self.prefetch_keys),
            "prefetch_seconds": float(self.prefetch_seconds),
            "prefetch_gib": self.prefetch_bytes / GIB,
            "evictions": int(self.evictions),
            "evict_seconds": float(self.evict_seconds),
            "distinct_rotations": len(self.rotations),
            "max_limbs": int(max(self.max_limbs.values(), default=0)),
        }


class KeyTracer:
    """Record, per stage, what the operators really asked the key cache for.

    Wraps ``get_rotation_key_for_limbs`` (request / hit / miss + limbs) and
    ``_load_rotation_key_to_cuda`` (bytes and seconds of the H2D copy) on
    one native context.  ``stage`` is switched by :meth:`stage`; anything
    outside a stage is booked under ``"(outside)"``.  Install with
    :meth:`install`, remove with :meth:`uninstall`; both are idempotent.
    """

    OUTSIDE = "(outside)"

    def __init__(self, native, *, synchronize=None):
        self.native = native
        self.stages: dict[str, StageKeyStats] = {}
        # The current stage is per thread: the single-card emulation
        # (:mod:`.emulation`) runs one rank per thread against one shared
        # tracer, and a rank parked at a collective must not relabel the
        # requests of the rank that holds the baton.
        self._local = threading.local()
        self._installed = False
        self._originals: dict[str, Any] = {}
        self._synchronize = synchronize

    @property
    def current(self) -> str:
        return getattr(self._local, "stage", self.OUTSIDE)

    @current.setter
    def current(self, value: str) -> None:
        self._local.stage = str(value)

    # -- bookkeeping
    def _stats(self, stage: str | None = None) -> StageKeyStats:
        name = self.current if stage is None else str(stage)
        row = self.stages.get(name)
        if row is None:
            row = StageKeyStats(stage=name)
            self.stages[name] = row
        return row

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[StageKeyStats]:
        previous = self.current
        self.current = str(name)
        try:
            yield self._stats()
        finally:
            self.current = previous

    def note_prefetch(self, tier: str, count: int, seconds: float, *, nbytes: int = 0) -> None:
        row = self._stats()
        row.prefetch_keys += int(count)
        row.prefetch_seconds += float(seconds)
        row.prefetch_bytes += int(nbytes)

    def note_evict(self, stats: Mapping[str, Any]) -> None:
        row = self._stats()
        row.evictions += 1
        row.evict_seconds += float(stats.get("seconds", 0.0))

    # -- wrapping
    def install(self) -> "KeyTracer":
        if self._installed:
            return self
        native = self.native
        tracer = self
        original_get = native.get_rotation_key_for_limbs
        original_load = native._load_rotation_key_to_cuda

        def traced_get(rot_index, cur_limbs):
            row = tracer._stats()
            rot = int(rot_index)
            limbs = int(cur_limbs)
            row.requests += 1
            row.rotations.add(rot)
            if row.max_limbs.get(rot, 0) < limbs:
                row.max_limbs[rot] = limbs
            return original_get(rot_index, cur_limbs)

        def traced_load(rot_index, special_mod_start, beta, available_special_mod_start):
            row = tracer._stats()
            row.misses += 1
            if len(row.miss_log) < StageKeyStats.MISS_LOG_LIMIT:
                row.miss_log.append((int(rot_index), int(special_mod_start), int(beta)))
            started = time.perf_counter()
            result = original_load(rot_index, special_mod_start, beta, available_special_mod_start)
            tracer._sync()
            row.load_seconds += time.perf_counter() - started
            row.loads += 1
            for component in result:
                try:
                    row.bytes_loaded += int(component.numel()) * int(component.element_size())
                except Exception:
                    pass
            return result

        # Remember whether the attributes were instance-level before we shadow them.
        instance_attrs = vars(native)
        self._originals = {
            name: (name in instance_attrs, instance_attrs.get(name))
            for name in ("get_rotation_key_for_limbs", "_load_rotation_key_to_cuda")
        }
        native.get_rotation_key_for_limbs = traced_get
        native._load_rotation_key_to_cuda = traced_load
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        for name, (was_instance, previous) in self._originals.items():
            if was_instance:
                setattr(self.native, name, previous)
            else:
                try:
                    delattr(self.native, name)
                except AttributeError:
                    pass
        self._originals = {}
        self._installed = False

    def __enter__(self) -> "KeyTracer":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.uninstall()

    def _sync(self) -> None:
        if self._synchronize is not None:
            self._synchronize()
            return
        try:
            # Key-load timing is a profiling boundary.  Respect the shared
            # stage-sync switch so lazy misses do not reintroduce a hidden
            # device-wide barrier when the pipeline runs with sync disabled.
            from llama3fhe.backend import profile_sync_device, stage_sync_enabled

            if stage_sync_enabled():
                profile_sync_device(str(getattr(self.native, "device", "cuda")))
        except Exception:
            pass

    # -- reporting
    def rows(self) -> list[dict[str, Any]]:
        return [self.stages[name].as_row() for name in self.stages]

    def rotations(self, stage: str) -> frozenset[int]:
        row = self.stages.get(str(stage))
        return frozenset() if row is None else frozenset(row.rotations)

    def max_limbs(self, stage: str) -> int:
        row = self.stages.get(str(stage))
        return 0 if row is None else int(max(row.max_limbs.values(), default=0))

    def coverage(self, stage: str, plan: KeyPlan, tier: str) -> dict[str, Any]:
        """How the rotations a stage really used compare with the planned tier."""

        used = self.rotations(stage)
        planned = plan.tier(tier).rotation_set
        return {
            "stage": str(stage),
            "tier": str(tier),
            "planned": len(planned),
            "used": len(used),
            "used_in_plan": len(used & planned),
            "used_outside_plan": sorted(used - planned)[:32],
            "unused_planned": len(planned - used),
            "max_limbs": self.max_limbs(stage),
        }


__all__ = [
    "ATTN_INPUT_PHASE_TIERS",
    "KEY_POLICIES",
    "LEASE_POLICY",
    "RESIDENT_POLICY",
    "T0_TIERS",
    "T1_TIERS",
    "T2_TIERS",
    "TIER_ORDER",
    "EVICT_SYNC_MODES",
    "KeyLifecycle",
    "KeyPlan",
    "KeyTier",
    "KeyTracer",
    "PinnedCompactKeys",
    "LeaseRecord",
    "StageKeyStats",
    "build_key_plan",
    "normalize_rotation",
    "rotation_key_bytes",
]
