"""Hardware presets for the T2 runtime.

Presets share the depth-37, dnum-5 context and a (5,5) bootstrap program.
They configure key residency, linear chunk size, MLP feature shards,
polynomial tables, refresh layouts, and transport according to device capacity.
The 16-ciphertext production layout is split into ``world_size`` ranks of
``ciphers_per_rank = 16 / world_size`` adjacent feature-major ciphertexts
(:mod:`.ownership`: one T2 *slot* = two ciphertexts = one PairFM source shard
= one W_O carrier).  The presets default to ``world_size = 8`` (one slot per
rank, one process per card on an 8-card host); :meth:`ParallelPreset.with_world_size`
derives the 4-card variant (``world_size = 4``, two slots = four ciphertexts
per rank, still one process per card: ``bench/t2_layer.py --world-size 4``).
The older way of using 4 cards -- eight one-slot ranks, two processes per card
(``--ranks-per-device 2``, :meth:`ParallelPreset.for_colocation`) -- keeps
working with a reduced per-rank memory budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BootstrapPreset:
    """Bootstrap program shape and constant-cache policy for one card."""

    level_budget: tuple[int, ...] = (5, 5)
    cache_mode: str = "middle"
    log_slots: int = 15
    iterations: int = 1
    precision_bits: int = 14
    mode: str = "modraise_first"
    strategy: str = "double_hoist"

    def __post_init__(self) -> None:
        if len(self.level_budget) != 2 or any(int(v) <= 0 for v in self.level_budget):
            raise ValueError("level_budget must be two positive integers")
        if self.cache_mode not in {"none", "middle", "plain"}:
            raise ValueError(f"unsupported cache_mode {self.cache_mode!r}")
        object.__setattr__(
            self, "level_budget", tuple(int(v) for v in self.level_budget)
        )

    @property
    def program_depth(self) -> int:
        """Levels consumed by the Meta-BTS program: 10 + b1 + b2."""

        return 10 + int(self.level_budget[0]) + int(self.level_budget[1])

    def post_levels(self, depth: int) -> int:
        """Levels left after a bootstrap in a ``depth``-level context."""

        post = int(depth) - self.program_depth
        if post <= 0:
            raise ValueError(
                f"depth {depth} is too shallow for level budget {self.level_budget}"
            )
        return post


@dataclass(frozen=True)
class ParallelPreset:
    """Everything a worker needs to size itself, in one place."""

    name: str
    world_size: int = 8
    ciphers_per_rank: int = 2
    depth: int = 37
    dnum: int = 5
    log_n: int = 16
    bootstrap: BootstrapPreset = field(default_factory=BootstrapPreset)
    #: GiB of device memory the launcher refuses to exceed per rank (soft).
    device_budget_gib: float = 23.0
    #: Whether the rotation-key CUDA cache may be filled lazily (EasyFHE
    #: ``auto_load_keys``).  Always ``True`` in T2: the key lifecycle
    #: (:mod:`.keys`) decides what stays resident, EasyFHE just caches.
    auto_load_keys: bool = True
    key_policy: str = "lease"
    #: Non-T0 tiers promoted to layer residency under ``"lease"`` (the
    #: ``linear`` tier is the candidate; decided by the lifecycle probe).
    key_resident_extra: tuple[str, ...] = ()
    #: Q-limb caps per key tier as ``((tier, limbs), ...)`` (host key trimming
    #: + prefetch limbs).  Empty means full-chain keys everywhere; filled in
    #: from the lifecycle probe's ``max_limbs`` per stage.
    key_limb_caps: tuple[tuple[str, int], ...] = ()
    #: Keep the leased (non-resident, capped) tiers pre-compacted in pinned
    #: host memory (:class:`~.keys.PinnedCompactKeys`) so a per-layer reload
    #: is a plain pinned H2D copy.  Costs ~40% of those tiers' host key bytes
    #: extra per rank; irrelevant under ``"resident"`` (nothing is reloaded).
    pin_leased_keys: bool = True
    linear_max_plaintext_rows: int = 512
    #: How the leased key tiers reach the device (:class:`~.keys.KeyLifecycle`):
    #: ``"lazy"`` loads each key on its first request inside the operators;
    #: ``"sync"`` uploads a whole tier at lease entry; ``"async"`` uploads the
    #: *next* tier of the station chain on a side CUDA stream while the current
    #: station computes (needs one lazy learning pass per tier).  Irrelevant
    #: under ``key_policy="resident"``.
    key_prefetch_mode: str = "lazy"
    #: Free device memory (GiB) an async prefetch must leave untouched at issue
    #: time; below that the tier falls back to lazy loading for this lease.
    key_prefetch_margin_gib: float = 1.5
    #: What a lease's exit evict waits for: ``"device"`` = EasyFHE's
    #: ``clear_cuda_rotation_cache`` (full ``torch.cuda.synchronize`` +
    #: ``empty_cache``, joins any prefetch copy in flight); ``"stream"`` =
    #: compute-stream sync only, blocks stay in the caching allocator.
    key_evict_sync: str = "device"
    #: MLP intermediate-page split (:mod:`llama3fhe.operators.linear.shard.mlp_feature_shard`,
    #: Megatron-style tensor parallelism of the SwiGLU hidden dimension): the
    #: ``k`` adjacent ranks of a shard group each own ``pages/k`` of the 4
    #: intermediate pages and run gate/up + SiLU + the matching down column
    #: block for the tokens of the whole group, so the plaintext encode (the one
    #: linear cost that does not shrink with the world size) drops by
    #: ``1 - 1/k`` at the price of ``k`` x the hoisted baby basis (~0.7 GiB per
    #: ciphertext at 11 limbs) and two small all-to-alls.  ``1`` = plain token
    #: sharding (:func:`~llama3fhe.parallel.prefill.stations.mlp_station` runs the
    #: unchanged kernel).  Must divide ``world_size``.
    mlp_feature_shard_k: int = 1
    #: BSGS hoisting of the PairFM mask transforms
    #: (:func:`llama3fhe.operators.transform.apply_transform_by_level`, env
    #: ``LLAMA_FHE_TRANSFORM_HOIST``): ``"ext_normal"`` keeps the mask
    #: plaintexts in the extended basis (``cur_limbs + K`` limbs, +73% bytes at
    #: L3 ``K=8``) and saves one moddown per baby step; ``"normal"`` encodes
    #: plain plaintexts and lets the hoisted MAC do the moddown.  The launcher
    #: exports it to every worker; a value already in the environment wins.
    transform_hoist_strategy: str = "ext_normal"
    transport: str = "host"
    pairfm_refresh_bootstrap_iterations: int = 2
    rmsnorm_exit_mode: str = "pair"
    #: Layers that take the ``"slim"`` exit when :attr:`rmsnorm_exit_mode` is
    #: ``"slim"``; the others keep ``"pair"`` (``"all"`` / ``"3-31"`` /
    #: ``"0,1,2"`` / ``"0-2,14,31"``).  Layers 0-2 must stay on ``pair``: their
    #: rsqrt row is 80-190x the normalised stream and the 14-bit slim BS loses
    #: it (prescale and a second Meta-BTS pass do not recover it).
    #: ``T2_RMS_SLIM_LAYERS`` in the environment overrides.
    rmsnorm_slim_exit_layers: str = "all"
    rmsnorm_slim_period_rows: int | None = None
    #: Periodic slim pack of the Prefill Softmax Alg2 row sums
    #: (``AttentionOperatorConfig.softmax_slim_periodic``): the rank's slim
    #: ciphertext repeats every ``pairs_per_rank * 2 * seq_len`` slots and its
    #: slim-input / lambda refreshes (1-2 per Alg2 round) take a sparse program
    #: of ``log2`` that period (:func:`~.worker.softmax_slim_sparse_log_slots`:
    #: 9 at W=8, 10 at W=4, 12 single-card).  ``--softmax-slim-periodic`` /
    #: ``--no-softmax-slim-periodic`` on the benches override.
    softmax_slim_periodic: bool = False
    #: Extra sparse-slot bootstrap programs (``log2`` slot counts) every rank
    #: prepares next to the dense one, on top of the slim RMSNorm program
    #: :func:`~.worker.sparse_bootstrap_log_slots` derives.  The Decode
    #: merged-page Softmax runs its ``filled`` / ``lambda`` refreshes (the
    #: cross-page ``sum(y^2)`` row, periodic every ``decode_page_merge_stride``
    #: slots) through the program of ``log2(decode_page_merge_stride) = 12``
    #: whenever it is present (``--softmax-sparse-bs`` on the Decode bench;
    #: ``LLAMA_FHE_SOFTMAX_SPARSE_BS=0`` forces dense).  Each extra program
    #: costs its CtoS/StoC constants and a few rotation keys.
    bootstrap_extra_sparse_log_slots: tuple[int, ...] = ()
    #: RMSNorm rsqrt Chebyshev table the benches load instead of the production
    #: ``assets/polynomials/rmsnorm_poly_coeffs.json`` (``None`` = production).
    #: A bare file name resolves under ``assets/polynomials/``.  The high tier
    #: runs ``rmsnorm_poly_coeffs_reltol5e-3_noinc.json``
    #: with per-layer degrees chosen for a relative tolerance of 5e-3.
    #: ``T2_RMS_COEFFS`` in the environment
    #: overrides (``T2_RMS_COEFFS=default`` forces the production table).
    rmsnorm_coeffs: str | None = None
    softmax_coeffs: str | None = None
    silu_coeffs: str | None = None
    softmax_coeffs_parts: str = "exp,j1,j2"
    attention_input_overlap: bool = False

    def __post_init__(self) -> None:
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be positive")
        if int(self.pairfm_refresh_bootstrap_iterations) not in (1, 2):
            raise ValueError("pairfm_refresh_bootstrap_iterations must be 1 or 2")
        if str(self.rmsnorm_exit_mode) not in {"pair", "slim"}:
            raise ValueError(f"unsupported rmsnorm_exit_mode {self.rmsnorm_exit_mode!r}; choose pair or slim")
        if not str(self.rmsnorm_slim_exit_layers).strip():
            raise ValueError("rmsnorm_slim_exit_layers must be 'all' or a layer spec such as '3-31'")
        if self.rmsnorm_slim_period_rows is not None:
            p = int(self.rmsnorm_slim_period_rows)
            if p <= 0 or p & (p - 1):
                raise ValueError(f"rmsnorm_slim_period_rows must be a power of two, got {p}")
        parts = {p.strip().lower() for p in str(self.softmax_coeffs_parts).split(",") if p.strip()}
        if not parts or parts - {"exp", "j1", "j2", "k"}:
            raise ValueError(
                f"softmax_coeffs_parts {self.softmax_coeffs_parts!r} must be a comma list drawn from exp, j1, j2, k"
            )
        object.__setattr__(self, "softmax_coeffs_parts", ",".join(p for p in ("exp", "j1", "j2", "k") if p in parts))
        for attr in ("rmsnorm_coeffs", "softmax_coeffs"):
            value = getattr(self, attr)
            if value is not None and not str(value).strip():
                raise ValueError(f"{attr} must be None or a non-empty path")
        from .prefill.transport import TRANSPORT_BACKENDS

        if str(self.transport) not in TRANSPORT_BACKENDS:
            raise ValueError(f"unsupported transport {self.transport!r}; choose from {TRANSPORT_BACKENDS}")
        if int(self.mlp_feature_shard_k) <= 0 or int(self.world_size) % int(self.mlp_feature_shard_k):
            raise ValueError(
                f"mlp_feature_shard_k {self.mlp_feature_shard_k} must be a positive divisor of "
                f"world_size {self.world_size}"
            )
        if int(self.ciphers_per_rank) <= 0 or int(self.ciphers_per_rank) % 2:
            raise ValueError("ciphers_per_rank must be a positive even number")
        if str(self.key_policy) not in {"lease", "resident"}:
            raise ValueError(f"unsupported key_policy {self.key_policy!r}")
        if str(self.transform_hoist_strategy) not in {"ext_normal", "normal"}:
            raise ValueError(
                f"unsupported transform_hoist_strategy {self.transform_hoist_strategy!r}"
            )
        object.__setattr__(self, "key_resident_extra", tuple(str(v) for v in self.key_resident_extra))
        object.__setattr__(
            self, "key_limb_caps", tuple((str(k), int(v)) for k, v in dict(self.key_limb_caps).items())
        )
        if int(self.linear_max_plaintext_rows) <= 0:
            raise ValueError("linear_max_plaintext_rows must be positive")
        if str(self.key_prefetch_mode) not in {"lazy", "sync", "async"}:
            raise ValueError(f"unsupported key_prefetch_mode {self.key_prefetch_mode!r}")
        if float(self.key_prefetch_margin_gib) < 0:
            raise ValueError("key_prefetch_margin_gib must be non-negative")
        if str(self.key_evict_sync) not in {"device", "stream"}:
            raise ValueError(f"unsupported key_evict_sync {self.key_evict_sync!r}")

    def layer_schedule(self, base=None):
        """The :class:`~llama3fhe.schedule.LayerSchedule` this preset executes.

        ``base`` defaults to the production ``LayerSchedule()`` with
        ``steady_state_attention_refresh=True``: the T2 stations always run
        the steady-layer attention chain (pre-exp Softmax refresh), the way
        ``run_model.py`` resolves it for every layer of a multi-layer run.
        The bare ``LayerSchedule()`` default (``False``) is the fused
        single-card *Layer0* graph, whose Softmax enters exp straight from
        the QK adapter level; under the T2 station chain that path runs out
        of limbs inside the layer-0 exp polynomial (k=4, ps_k=6), so it is
        not what a ``--coeff-layer 0`` T2 run wants.  The preset then
        overrides the linear ``max_plaintext_rows`` on all three boundaries.
        The key plan (:func:`~.keys.build_key_plan`) and the stations must be
        built from the same object, otherwise the chunk giant-base rotations
        of one do not exist in the other.
        """

        from dataclasses import replace

        from llama3fhe.schedule import LayerSchedule

        schedule = LayerSchedule(steady_state_attention_refresh=True) if base is None else base
        rows = int(self.linear_max_plaintext_rows)
        if (
            int(schedule.qkv_max_plaintext_rows) == rows
            and int(schedule.output_max_plaintext_rows) == rows
            and int(schedule.mlp_max_plaintext_rows) == rows
        ):
            return schedule
        return replace(
            schedule,
            qkv_max_plaintext_rows=rows,
            output_max_plaintext_rows=rows,
            mlp_max_plaintext_rows=rows,
        )

    @property
    def slots(self) -> int:
        return 1 << (int(self.log_n) - 1)

    @property
    def limb_caps(self) -> dict[str, int]:
        return {name: int(limbs) for name, limbs in self.key_limb_caps}

    @property
    def bootstrap_post_levels(self) -> int:
        return self.bootstrap.post_levels(self.depth)

    @property
    def cipher_count(self) -> int:
        return int(self.world_size) * int(self.ciphers_per_rank)

    @property
    def slots_per_rank(self) -> int:
        return int(self.ciphers_per_rank) // 2

    def with_world_size(self, world_size: int, *, mlp_feature_shard_k: int | None = None) -> "ParallelPreset":
        """The same preset re-cut for ``world_size`` ranks over the same 16 ciphertexts.

        ``ciphers_per_rank`` becomes ``cipher_count / world_size`` (must be a
        positive even number: 8 -> 2, 4 -> 4, 2 -> 8).  ``mlp_feature_shard_k``
        must divide the new world size; when it does not (``k = 4`` on
        ``world_size = 2``) it is clamped to the largest divisor of the new
        world size not above the old value unless an explicit ``k`` is given.
        Everything else (bootstrap program, key policy, hoist strategy,
        device budget) is unchanged -- those are per-card knobs and the card
        is the same.  The linear chunk rows stay too, except that a world
        with more than one slot per rank is capped at
        :data:`SHARED_CARD_MAX_PLAINTEXT_ROWS` (the 2048-row profile is
        configured for one slot per card).
        """

        from dataclasses import replace

        world_size = int(world_size)
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        total = int(self.cipher_count)
        if total % world_size:
            raise ValueError(f"world_size {world_size} does not divide the {total}-ciphertext layout")
        per_rank = total // world_size
        if per_rank % 2:
            raise ValueError(
                f"world_size {world_size} gives {per_rank} ciphertexts per rank; a rank owns whole slots (even count)"
            )
        if mlp_feature_shard_k is None:
            k = int(self.mlp_feature_shard_k)
            while world_size % k:
                k -= 1
        else:
            k = int(mlp_feature_shard_k)
        rows = int(self.linear_max_plaintext_rows)
        if per_rank > 2:
            # More than one slot per card: the 2048-row chunk of the one-slot
            # Use the shared-card chunk limit when a rank owns multiple slots.
            # ``--max-plaintext-rows`` on the benches overrides.
            rows = min(rows, SHARED_CARD_MAX_PLAINTEXT_ROWS)
        if (
            world_size == int(self.world_size)
            and k == int(self.mlp_feature_shard_k)
            and rows == int(self.linear_max_plaintext_rows)
        ):
            return self
        return replace(
            self,
            world_size=world_size,
            ciphers_per_rank=per_rank,
            mlp_feature_shard_k=k,
            linear_max_plaintext_rows=rows,
        )

    def for_colocation(self, ranks_per_device: int) -> "ParallelPreset":
        """The same preset with ``device_budget_gib`` split between the
        ``ranks_per_device`` ranks that share one physical card (a 4-card
        96 GB host runs the 8-rank layout as two processes per card).
        """

        from dataclasses import replace

        ranks_per_device = int(ranks_per_device)
        if ranks_per_device <= 0:
            raise ValueError("ranks_per_device must be positive")
        if ranks_per_device == 1:
            return self
        return replace(
            self,
            device_budget_gib=float(self.device_budget_gib) / ranks_per_device,
            linear_max_plaintext_rows=min(int(self.linear_max_plaintext_rows), SHARED_CARD_MAX_PLAINTEXT_ROWS),
        )


SHARED_CARD_MAX_PLAINTEXT_ROWS = 512


MEASURED_LIMB_CAPS = (
    ("linear", 14),
    ("attn_input", 13),
    ("attn_input_lane", 13),
    ("attn_input_q", 13),
    ("attn_input_k", 13),
    ("attn_input_v", 13),
    ("attn_core", 13),
    ("attn_qk", 13),
    ("attn_pv", 13),
    ("softmax", 16),
    ("pairfm_output", 13),
)

RTX4090_LOW = ParallelPreset(
    name="rtx4090-low",
    depth=37,
    dnum=5,
    bootstrap=BootstrapPreset(level_budget=(5, 5), cache_mode="middle"),
    device_budget_gib=23.0,
    key_policy="lease",
    key_resident_extra=("linear",),
    key_limb_caps=MEASURED_LIMB_CAPS,
    # Reserve memory for asynchronous key prefetch.
    linear_max_plaintext_rows=256,
    transport="pinned",
    key_prefetch_mode="async",
    key_prefetch_margin_gib=1.5,
    key_evict_sync="stream",
    mlp_feature_shard_k=2,
    transform_hoist_strategy="normal",
    rmsnorm_coeffs="rmsnorm_poly_coeffs_reltol5e-3_noinc.json",
    softmax_coeffs="softmax_poly_coeffs_joint_tol_v1.json",
    softmax_coeffs_parts="exp",
)

def _high_preset(
    name: str, device_budget_gib: float, *, linear_max_plaintext_rows: int = 512
) -> ParallelPreset:
    """The efficiency-first profile shared by every >= 80 GB card."""

    return ParallelPreset(
        name=str(name),
        depth=37,
        dnum=5,
        bootstrap=BootstrapPreset(level_budget=(5, 5), cache_mode="plain"),
        device_budget_gib=float(device_budget_gib),
        key_policy="resident",
        # Same caps as the 4090: they are a property of the operators' limb
        # schedule, not of the card.
        key_limb_caps=MEASURED_LIMB_CAPS,
        linear_max_plaintext_rows=int(linear_max_plaintext_rows),
        mlp_feature_shard_k=4,
        transform_hoist_strategy="ext_normal",
        attention_input_overlap=True,
        pairfm_refresh_bootstrap_iterations=1,
        rmsnorm_exit_mode="slim",
        rmsnorm_slim_exit_layers="3-31",
        rmsnorm_coeffs="rmsnorm_poly_coeffs_reltol5e-3_noinc.json",
        softmax_coeffs="softmax_poly_coeffs_joint_tol_v1.json",
        softmax_coeffs_parts="exp",
        silu_coeffs="silu_poly_coeffs_joint_tol_v1.json",
        rmsnorm_slim_period_rows=16,
        softmax_slim_periodic=True,
    )


#: Eight A100 80 GB GPUs, one worker per GPU.
A100_HIGH = _high_preset("a100-high", device_budget_gib=78.0)

RTX6000PRO_HIGH = _high_preset("rtx6000pro-high", device_budget_gib=92.0, linear_max_plaintext_rows=2048)

#: Single-device H100 emulation of eight ranks (:mod:`.emulation`).
H100_EMU = _high_preset("h100-emu", device_budget_gib=78.0)

A800_4CARD = _high_preset("a800-4card", device_budget_gib=78.0).with_world_size(4)

#: ``<repo>/assets/polynomials``: where a bare table name in a preset lives.
POLYNOMIALS_DIR = Path(__file__).resolve().parents[2] / "assets" / "polynomials"


def polynomial_table_path(spec: str | os.PathLike[str]) -> Path:
    """Resolve a preset / environment table spec to a file.

    A spec with a directory part (or one that exists as given) is taken as is;
    a bare file name (``rmsnorm_poly_coeffs_reltol5e-3_noinc.json``) resolves
    under :data:`POLYNOMIALS_DIR`.  The file must exist.
    """

    path = Path(spec).expanduser()
    if path.exists():
        return path
    if path.parent == Path(".") and (POLYNOMIALS_DIR / path.name).exists():
        return POLYNOMIALS_DIR / path.name
    raise FileNotFoundError(f"polynomial table {str(spec)!r} not found (looked in {POLYNOMIALS_DIR})")


PRESETS: dict[str, ParallelPreset] = {
    RTX4090_LOW.name: RTX4090_LOW,
    A100_HIGH.name: A100_HIGH,
    RTX6000PRO_HIGH.name: RTX6000PRO_HIGH,
    H100_EMU.name: H100_EMU,
    A800_4CARD.name: A800_4CARD,
}

#: Presets whose memory knobs assume an >= 80 GB card (all key tiers resident).
HIGH_PRESETS: tuple[str, ...] = (A100_HIGH.name, RTX6000PRO_HIGH.name, H100_EMU.name, A800_4CARD.name)


def preset_by_name(name: str) -> ParallelPreset:
    try:
        return PRESETS[str(name)]
    except KeyError as exc:
        raise KeyError(
            f"unknown preset {name!r}; choose from {sorted(PRESETS)}"
        ) from exc


__all__ = [
    "A100_HIGH",
    "A800_4CARD",
    "H100_EMU",
    "HIGH_PRESETS",
    "PRESETS",
    "RTX4090_LOW",
    "RTX6000PRO_HIGH",
    "SHARED_CARD_MAX_PLAINTEXT_ROWS",
    "BootstrapPreset",
    "POLYNOMIALS_DIR",
    "ParallelPreset",
    "polynomial_table_path",
    "preset_by_name",
]
