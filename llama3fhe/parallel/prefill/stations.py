"""Layer stations executed by one T2 rank on its own FM ciphertexts.

A rank owns ``ciphers_per_rank = 16 / world_size`` adjacent FM ciphertexts,
i.e. ``slots_per_rank = ciphers_per_rank / 2`` T2 *slots* (slot ``s`` = FM
pair ``2s, 2s+1`` = query pairs ``2s, 2s+1`` = W_O carrier ``s``; see
:mod:`llama3fhe.parallel.prefill.ownership`).  ``world_size == 8`` is one slot per
rank (the layout the kernels were written for); ``world_size == 4`` is two
slots per rank on four cards with one process each.  The dense linear
stations batch their slots in one kernel call (weights encoded once); the
PairFM source half runs once per slot and pre-adds the same-target partials.

A *station* is one device-local step of the layer; between stations the
ciphertexts stay resident on the card.  Only two stations talk to other
ranks, and both are pure data movement (no cross-card FHE arithmetic):

============================  ==========================================
station                       cross-card traffic
============================  ==========================================
``rmsnorm_station``           none  (1 pair-BS per slot for entry and exit)
``qkv_station``               none  (token shard, own FM -> own carriers)
``attention_input_exchange``  all-gather of the 16 lane ciphers + all-to-all
                              of the own-source V partials
``attention_core_station``    none  (own Q pairs x own K/V pairs; 3 BS per rank)
``pairfm_exchange``           all-to-all of the (rank x carrier) pre-rescale
                              partials (one BS per slot: O_C refresh)
``w_o_station`` / residual    none  (own carriers -> own FM ciphers)
``rmsnorm_station`` (2)       none
``mlp_station``               none
residual                      none
============================  ==========================================



* entry: ``x0 + i*x1`` (~2 limbs) -> 18 limbs; the unpack doubles both
  streams and ``rmsnorm_fhe`` absorbs the 2x through ``input_value_scale``;
* exit: ``y0*s + i*y1*s`` (9 limbs) -> 18 limbs; the unpack doubles again.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ownership import RankOwnership

ATTN_IN_OVERLAP_ENV = "T2_ATTN_IN_OVERLAP"


def attention_input_overlap_enabled(default: bool = False) -> bool:
    raw = os.environ.get(ATTN_IN_OVERLAP_ENV)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "off", "false", "no"}


def _collective_isend(transport, kind: str, tag: str, payload):
    """``transport.<kind>_isend(tag, payload)`` or a ready handle around the blocking call.

    Duck-typed transports (test doubles) that only implement the blocking
    collectives still work with the overlapped station order; they just do
    not overlap.
    """

    from .transport import ReadyCollective

    isend = getattr(transport, f"{kind}_isend", None)
    if isend is not None:
        return isend(tag, payload)
    return ReadyCollective(getattr(transport, kind)(tag, payload))


def _release(*values) -> None:
    from llama3fhe.backend import release_if_supported

    for value in values:
        if value is not None:
            release_if_supported(value)


def _sync(context) -> None:
    """Stage-timer device sync (no-op under ``T2_STAGE_SYNC=0``; the transport keeps its own syncs)."""

    from llama3fhe.backend import profile_sync_device

    profile_sync_device(context.device)


def _levels(context, ciphers) -> tuple[int, ...]:
    return tuple(int(context.level_for_cipher(cipher)) for cipher in ciphers)


def _limbs(ciphers) -> tuple[int, ...]:
    return tuple(int(cipher.state.cur_limbs) for cipher in ciphers)


# ------------------------------------------------------------------- pair BS
def pair_bootstrap(
    ciphers,
    *,
    context,
    bootstrap_operator,
    iterations: int = 1,
    precision_bits: int = 14,
    synchronize: bool = True,
):
    """Refresh an even list of real FM ciphers two at a time via ``a + i*b``.

    Returns ``(refreshed, value_scale)``: the conjugate unpack doubles every
    component, so ``value_scale`` is always 2.  Inputs are *not* consumed.
    """

    from llama3fhe.operators.norm.refresh import _pack_complex_pair, _unpack_complex_pair

    ciphers = tuple(ciphers)
    if len(ciphers) % 2:
        raise ValueError("pair_bootstrap needs an even number of ciphertexts")
    out: list[object] = []
    refresh_many = getattr(bootstrap_operator, "refresh_many", None)
    if refresh_many is not None:
        # all pairs at once: the operator batches them through one C2S / EvalMod /
        # S2C pass when a batch backend is installed (HOLMES_BATCH_BS=1), else loops
        packed_all: list[object] = []
        refreshed_all: list[object] = []
        try:
            for index in range(0, len(ciphers), 2):
                packed_all.append(
                    _pack_complex_pair(ciphers[index], ciphers[index + 1], crypto_context=context)
                )
            try:
                refreshed_all = list(
                    refresh_many(
                        packed_all,
                        iterations=int(iterations),
                        precision_bits=int(precision_bits),
                        synchronize=bool(synchronize),
                    )
                )
            finally:
                _release(*packed_all)
                packed_all = []
            for refreshed in refreshed_all:
                out.extend(_unpack_complex_pair(refreshed, crypto_context=context))
            return tuple(out), 2.0
        except Exception:
            _release(*out)
            raise
        finally:
            _release(*packed_all)
            _release(*refreshed_all)
    try:
        for index in range(0, len(ciphers), 2):
            packed = _pack_complex_pair(ciphers[index], ciphers[index + 1], crypto_context=context)
            try:
                refreshed = bootstrap_operator.refresh(
                    packed,
                    iterations=int(iterations),
                    precision_bits=int(precision_bits),
                    synchronize=bool(synchronize),
                )
            finally:
                _release(packed)
            try:
                out.extend(_unpack_complex_pair(refreshed, crypto_context=context))
            finally:
                _release(refreshed)
        return tuple(out), 2.0
    except Exception:
        _release(*out)
        raise


def halve(ciphers, *, context):
    """Public 0.5 multiplication (one rescale) undoing a pair-BS doubling."""

    from llama3fhe.operators.norm.refresh import _half_rescale

    return tuple(_half_rescale(cipher, crypto_context=context) for cipher in ciphers)


# ------------------------------------------------------- linear entry limbs
#: Q limbs the attention chain needs at the QKV input (graph.py
#: ``minimum_attention_rmsnorm_limbs``): QKV 1 + K adapter 4 + QK 2 + score
#: preprocessing 1 + bootstrap floor 2, plus 3 so the never-refreshed value
#: stream still reaches the PairFM pairing at >= 6 limbs.
QKV_INPUT_LIMBS = 13

#: Limbs left after the SiLU PS depth (graph.py ``mlp_suffix_limbs``):
#: gate(1) + up-mul(1) + down(1) + residual match + a two-limb floor so the
#: next layer's entry pair-BS still has something to refresh.
MLP_SUFFIX_LIMBS = 6


def mlp_input_limbs(silu_entry, *, fuse_silu_domain_map: bool = False) -> int:
    """graph.py ``required_mlp_input_limbs`` for one SiLU artifact entry."""

    from llama3fhe.approx.chebyshev import chebyshev_ps_mul_depth

    return (
        int(chebyshev_ps_mul_depth(silu_entry["coefficients"]))
        + MLP_SUFFIX_LIMBS
        - int(bool(fuse_silu_domain_map))
    )


def linear_entry(ciphers, *, input_limbs: int | None, context):
    """Modulus-drop the exit pair-BS output to what the next linear needs.

    The RMSNorm exit pair-BS returns 18 limbs on L3 while QKV only needs 13
    and the MLP ``d + 6``; running the dense BSGS at the surplus modulus
    inflates every baby rotation, plaintext diagonal and rotation key by
    ``surplus / needed`` for nothing.  graph.py does the same drop
    (``drop_levels`` before QKV / MLP, clamped to the consumer floor); it is
    modulus-only, so no depth or scale is spent.

    Returns ``(inputs, owned)``: ``owned`` is ``True`` when ``inputs`` are
    fresh views the caller must release, ``False`` when the ciphers already
    sit at or below ``input_limbs`` (or ``input_limbs`` is ``None``) and are
    returned untouched.
    """

    from llama3fhe.operators.primitives import drop_to_limbs

    ciphers = tuple(ciphers)
    if input_limbs is None or not ciphers:
        return ciphers, False
    target = int(input_limbs)
    current = min(int(cipher.state.cur_limbs) for cipher in ciphers)
    if current <= target:
        return ciphers, False
    return drop_to_limbs(ciphers, target_limbs=target, crypto_context=context), True


# ------------------------------------------------------------------- RMSNorm
@dataclass
class RMSNormStationResult:
    """Normalised pair + refreshed residual pair for one rank."""

    ciphers: tuple[object, ...]
    #: Extra factor carried by ``ciphers`` on top of what
    #: ``deferred_output_scale`` already compensates (2.0 after the exit
    #: pair-BS unpack, 1.0 without exit refresh).
    value_scale: float
    #: ``gamma / entry_scale`` as returned by ``rmsnorm_fhe``; identical to the
    #: single-card graph's ``input_norm.deferred_output_scale``.
    deferred_output_scale: Any
    #: Refreshed copies of the inputs (x at post-BS levels, exact 1x) when
    #: requested, otherwise ``()``.  These become the residual branch.
    residual_inputs: tuple[object, ...]
    output_levels: tuple[int, ...]
    output_limbs: tuple[int, ...]
    stage_seconds: dict[str, float]
    level_trace: dict[str, Any] = field(default_factory=dict)
    bootstrap_calls: int = 0

    @property
    def wall_seconds(self) -> float:
        return float(sum(self.stage_seconds.values()))

    def linear_input_column_scale(self):
        """``input_column_scale`` for the next linear operator.

        ``deferred_output_scale / value_scale``: what ``graph.py`` passes
        as ``input_column_scale`` divided by the exit-unpack doubling.
        """

        import numpy as np

        deferred = np.asarray(self.deferred_output_scale, dtype=np.float64)
        return (deferred / float(self.value_scale)).astype(np.float32)

    def release(self) -> None:
        _release(*self.ciphers, *self.residual_inputs)
        self.ciphers = ()
        self.residual_inputs = ()


def rmsnorm_station(
    ciphers,
    gamma,
    *,
    coeffs,
    ownership: RankOwnership,
    layout,
    context,
    bootstrap_operator,
    keep_residual: bool = True,
    exit_refresh: bool = True,
    bootstrap_iterations: int = 1,
    bootstrap_precision_bits: int = 14,
    slim_refresh_min_limbs: int | None = 2,
    slim_refresh_iterations: int = 1,
    slim_refresh_pre_scale: float = 1.0,
    slim_period_rows: int | None = None,
    synchronize: bool = True,
    function_prefix: str = "t2.rmsnorm",
) -> RMSNormStationResult:
    """Entry pair-BS -> RMSNorm (gamma deferred, adaptive slim refresh) -> exit pair-BS.

    ``ciphers`` are this rank's two FM ciphertexts (global indices
    ``ownership.cipher_indices``), typically at ~2 limbs after the previous
    residual add.  They are not consumed.

    ``slim_refresh_min_limbs`` arms the production ``SlimRefresh`` of
    ``rmsnorm_fhe``: when the rsqrt polynomial leaves fewer than
    ``min_limbs + 2`` limbs (extract + scale still to come), the slim
    mean-square row is bootstrapped once before ``extract``.  Under L3
    (18 limbs after the entry BS) that is only the Paterson-Stockmeyer
    depth-13 fits (4097 coefficients: layer 2 ``ffn``), whose exit pair-BS
    otherwise fails at 1 limb (plan §12.4); every other layer runs exactly
    as before (no extra BS).  ``None`` disables it.  ``slim_refresh_pre_scale``
    (0 < f <= 1) multiplies the rsqrt row by ``f`` before that BS (undone in
    the extract plaintext) so a large rsqrt (layers 0-2: 50-200) enters the
    bootstrap in range; ``exit_refresh=False`` relies on that slim BS for the
    output level instead of the exit pair-BS (``slim_refresh_min_limbs=16``).
    ``slim_period_rows`` (power of two) is forwarded to ``rmsnorm_fhe``: the
    periodic slim pack whose refresh takes the operator's sparse program when
    ``bootstrap_operator.sparse_slots == slim_period_rows * tokens_per_cipher``.
    """

    from llama3fhe.operators.norm.rmsnorm import SlimRefresh, rmsnorm_fhe

    ciphers = tuple(ciphers)
    if len(ciphers) != len(ownership.cipher_indices):
        raise ValueError(
            f"rank {ownership.rank} owns {len(ownership.cipher_indices)} ciphertexts, got {len(ciphers)}"
        )
    stage: dict[str, float] = {}
    trace: dict[str, Any] = {
        "input_levels": _levels(context, ciphers),
        "input_limbs": _limbs(ciphers),
    }
    calls = 0
    refreshed: tuple[object, ...] = ()
    residual: tuple[object, ...] = ()
    outputs: tuple[object, ...] = ()
    try:
        started = time.perf_counter()
        refreshed, entry_scale = pair_bootstrap(
            ciphers,
            context=context,
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            synchronize=synchronize,
        )
        calls += len(ciphers) // 2
        stage["entry_pair_bootstrap"] = time.perf_counter() - started
        trace["entry_bootstrap_levels"] = _levels(context, refreshed)
        trace["entry_bootstrap_limbs"] = _limbs(refreshed)

        if keep_residual:
            started = time.perf_counter()
            residual = halve(refreshed, context=context)
            stage["residual_half_rescale"] = time.perf_counter() - started

        started = time.perf_counter()
        norm = rmsnorm_fhe(
            refreshed,
            gamma,
            coeffs=coeffs,
            layout=layout,
            crypto_context=context,
            input_value_scale=float(entry_scale),
            defer_output_scale=True,
            slim_refresh=(
                SlimRefresh(
                    bootstrap_operator=bootstrap_operator,
                    minimum_output_limbs=int(slim_refresh_min_limbs),
                    pre_scale_factor=float(slim_refresh_pre_scale),
                    iterations=int(slim_refresh_iterations),
                    precision_bits=int(bootstrap_precision_bits),
                    synchronize=bool(synchronize),
                )
                if slim_refresh_min_limbs is not None
                else None
            ),
            verify=False,
            cipher_indices=tuple(ownership.cipher_indices),
            synchronize=bool(synchronize),
            function_prefix=str(function_prefix),
            slim_period_rows=(
                None if slim_period_rows is None else int(slim_period_rows)
            ),
        )
        stage["rmsnorm"] = time.perf_counter() - started
        if "rsqrt_slim_bootstrap" in dict(norm.stage_seconds):
            # the slim refresh is one BS on the (single) slim mean-square row of the whole rank
            calls += 1
        _release(*refreshed)
        refreshed = ()
        outputs = tuple(norm.ciphers)
        norm.ciphers = ()
        trace["rmsnorm_output_levels"] = _levels(context, outputs)
        trace["rmsnorm_output_limbs"] = _limbs(outputs)
        trace["rmsnorm_level_schedule"] = dict(norm.level_schedule)
        trace["rmsnorm_stage_seconds"] = {k: float(v) for k, v in dict(norm.stage_seconds).items()}
        # rmsnorm_fhe folded the entry 2x into deferred_output_scale
        # (= gamma / entry_scale); nothing extra is pending yet.
        value_scale = 1.0

        if exit_refresh:
            started = time.perf_counter()
            fresh, exit_scale = pair_bootstrap(
                outputs,
                context=context,
                bootstrap_operator=bootstrap_operator,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                synchronize=synchronize,
            )
            calls += len(outputs) // 2
            stage["exit_pair_bootstrap"] = time.perf_counter() - started
            _release(*outputs)
            outputs = fresh
            value_scale *= float(exit_scale)
            trace["exit_bootstrap_levels"] = _levels(context, outputs)
            trace["exit_bootstrap_limbs"] = _limbs(outputs)
        if synchronize:
            _sync(context)
        result = RMSNormStationResult(
            ciphers=outputs,
            value_scale=float(value_scale),
            deferred_output_scale=norm.deferred_output_scale,
            residual_inputs=residual,
            output_levels=_levels(context, outputs),
            output_limbs=_limbs(outputs),
            stage_seconds={key: float(value) for key, value in stage.items()},
            level_trace=trace,
            bootstrap_calls=calls,
        )
        outputs = ()
        residual = ()
        return result
    except Exception:
        _release(*refreshed, *residual, *outputs)
        raise


# ------------------------------------------------------------------ residual
def residual_station(left, right, *, ownership: RankOwnership, layout, context, synchronize=True):
    """``left + right`` on this rank's pair (levels aligned inside ``residual_add``)."""

    from llama3fhe.operators.norm.rmsnorm import residual_add

    return residual_add(
        tuple(left),
        tuple(right),
        layout=layout,
        crypto_context=context,
        cipher_indices=tuple(ownership.cipher_indices),
        synchronize=bool(synchronize),
    )


# ---------------------------------------------------------------------- QKV
def qkv_station(
    ciphers,
    hidden_states,
    q_weight,
    k_weight,
    v_weight,
    *,
    ownership: RankOwnership,
    layout,
    attention_layout,
    context,
    config=None,
    operator_config=None,
    input_column_scale=None,
    input_limbs: int | None = QKV_INPUT_LIMBS,
    function_prefix: str = "t2.qkv",
):
    """Own FM ciphers -> own QKV carriers (:func:`qkv_token_shard_fhe`).

    One batched kernel call for all owned slots (``ciphers_per_rank`` inputs,
    two carriers per slot); the weight plaintexts are encoded once per rank.

    ``input_column_scale`` is where the deferred RMSNorm gamma / value scale
    goes (:meth:`RMSNormStationResult.linear_input_column_scale`).
    ``input_limbs`` is the entry modulus (:func:`linear_entry`; ``None`` runs
    at whatever the ciphers carry).  The 18-limb exit pair-BS output run
    straight into the dense BSGS is what put this station over the 4090
    budget (22 GiB allocated, OOM in ``moddown_from_ext``); at the production
    floor of 13 every baby rotation, diagonal and key shrinks by ~30%.
    """

    from llama3fhe.operators.linear.shard.qkv_source_shard import (
        QKVTokenShardSpec,
        qkv_token_shard_fhe,
    )

    spec = QKVTokenShardSpec(
        rank=int(ownership.rank),
        shard_index=int(ownership.rank),
        input_indices=ownership.cipher_indices,
        output_indices=ownership.cipher_indices,
        token_start=int(ownership.token_start),
        token_count=int(ownership.token_count),
    )
    inputs, owned = linear_entry(ciphers, input_limbs=input_limbs, context=context)
    try:
        return qkv_token_shard_fhe(
            hidden_states,
            q_weight,
            k_weight,
            v_weight,
            spec=spec,
            attention_layout=attention_layout,
            layout=layout,
            config=config,
            operator_config=operator_config,
            crypto_context=context,
            input_ciphers=inputs,
            verify=False,
            function_prefix=str(function_prefix),
            input_column_scale=input_column_scale,
        )
    finally:
        if owned:
            _release(*inputs)


# ---------------------------------------------------------------------- MLP
def mlp_station(
    ciphers,
    hidden_states,
    gate_weight,
    up_weight,
    down_weight,
    *,
    silu_entry,
    ownership: RankOwnership,
    layout,
    context,
    config=None,
    operator_config=None,
    input_column_scale=None,
    fuse_silu_domain_map: bool = False,
    input_limbs: int | None = None,
    feature_shard_k: int = 1,
    transport=None,
    function_prefix: str = "t2.mlp",
):
    """Own FM ciphers -> own FM ciphers (SwiGLU MLP).

    ``feature_shard_k = 1`` runs the unchanged kernel on the own ciphers
    (:func:`~llama3fhe.operators.linear.shard.mlp_token_shard.mlp_token_shard_fhe`,
    all owned slots in one batched call).
    ``feature_shard_k > 1`` splits the intermediate pages over the ``k``
    adjacent ranks of this rank's group
    (:func:`~llama3fhe.operators.linear.shard.mlp_feature_shard.mlp_feature_shard_fhe`,
    preset ``mlp_feature_shard_k``); it needs ``transport``, ``k`` must divide
    ``world_size`` and every rank of the world must call it the same number
    of times.

    ``input_limbs`` defaults to :func:`mlp_input_limbs` of ``silu_entry``
    (graph.py ``required_mlp_input_limbs``); pass an int to override or
    ``0``/negative to skip the entry drop.
    """

    from llama3fhe.operators.linear.shard.mlp_token_shard import MLPTokenShardSpec, mlp_token_shard_fhe

    if input_limbs is None:
        input_limbs = mlp_input_limbs(silu_entry, fuse_silu_domain_map=fuse_silu_domain_map)
    if int(feature_shard_k) > 1:
        from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
        from llama3fhe.operators.linear.shard.mlp_feature_shard import mlp_feature_shard_fhe

        if transport is None:
            raise ValueError("mlp_station with feature_shard_k > 1 needs a transport")
        local_layout = FeatureMajorPrefillLayout(
            seq_len=int(ownership.token_count), hidden_dim=int(layout.hidden_dim), slots=int(layout.slots)
        )
        local_hidden = np.ascontiguousarray(
            np.asarray(hidden_states, dtype=np.float32)[int(ownership.token_start) : int(ownership.token_stop)]
        )
        inputs, owned = linear_entry(
            ciphers, input_limbs=int(input_limbs) if int(input_limbs) > 0 else None, context=context
        )
        try:
            return mlp_feature_shard_fhe(
                local_hidden,
                gate_weight,
                up_weight,
                down_weight,
                silu_entry=silu_entry,
                layout=local_layout,
                rank=int(ownership.rank),
                feature_shard_k=int(feature_shard_k),
                transport=transport,
                config=config,
                operator_config=operator_config,
                crypto_context=context,
                input_ciphers=inputs,
                input_column_scale=input_column_scale,
                fuse_silu_domain_map=bool(fuse_silu_domain_map),
                verify=False,
                # transport tags must be identical on every rank (the
                # collectives match on them); ``function_prefix`` carries
                # the rank, so it must not leak into the tag.
                tag="mlp",
            )
        finally:
            if owned:
                _release(*inputs)

    spec = MLPTokenShardSpec(
        rank=int(ownership.rank),
        shard_index=int(ownership.rank),
        input_indices=ownership.cipher_indices,
        output_indices=ownership.cipher_indices,
        token_start=int(ownership.token_start),
        token_count=int(ownership.token_count),
    )
    inputs, owned = linear_entry(
        ciphers, input_limbs=int(input_limbs) if int(input_limbs) > 0 else None, context=context
    )
    try:
        return mlp_token_shard_fhe(
            hidden_states,
            gate_weight,
            up_weight,
            down_weight,
            silu_entry=silu_entry,
            spec=spec,
            layout=layout,
            config=config,
            operator_config=operator_config,
            crypto_context=context,
            input_ciphers=inputs,
            input_column_scale=input_column_scale,
            fuse_silu_domain_map=bool(fuse_silu_domain_map),
            verify=False,
        )
    finally:
        if owned:
            _release(*inputs)


# -------------------------------------------------------- communication (2)
@dataclass
class AttentionInputExchangeResult:
    """Own Q pairs, own K pair and the own V carrier's two pairs after the exchange."""

    query_items: tuple[tuple[int, object], ...]
    key_items: tuple[tuple[int, object], ...]
    value_items: tuple[tuple[int, object], ...]
    stage_seconds: dict[str, float]
    #: bytes this rank put on the wire / took off it, per collective
    comm_bytes: dict[str, int]
    scheduler_stats: dict[str, float] = field(default_factory=dict)
    lease_records: tuple[Any, ...] = ()
    #: ``emit_kv_cache=True``: the Decode-cache snapshots of this rank --
    #: ``(key_pair, folded K_C)`` of the own key pair(s) and ``(value_pair,
    #: real V_C)`` of the own carrier's two pairs (same conventions as the
    #: monolithic adapter's ``key_c_ciphers`` / ``value_c_ciphers``); hand them
    #: to :func:`~.kv_emit.prefill_kv_cache_station`.
    key_c_items: tuple[tuple[int, object], ...] = ()
    value_c_items: tuple[tuple[int, object], ...] = ()

    @property
    def ciphers(self) -> tuple[object, ...]:
        return tuple(
            cipher
            for _, cipher in self.query_items + self.key_items + self.value_items + self.key_c_items + self.value_c_items
        )

    @property
    def wall_seconds(self) -> float:
        return float(sum(self.stage_seconds.values()))

    def take_kv_cache_items(self) -> tuple[tuple[tuple[int, object], ...], tuple[tuple[int, object], ...]]:
        """Detach ``(key_c_items, value_c_items)``; the caller owns them from here on."""

        items = (self.key_c_items, self.value_c_items)
        self.key_c_items = ()
        self.value_c_items = ()
        return items

    def release(self) -> None:
        owned = self.ciphers
        self.query_items = ()
        self.key_items = ()
        self.value_items = ()
        self.key_c_items = ()
        self.value_c_items = ()
        _release(*owned)


def _packet_bytes(packets) -> int:
    total = 0
    for packet in packets:
        if packet is None:
            continue
        if isinstance(packet, (list, tuple)):
            total += _packet_bytes(packet)
        else:
            total += int(getattr(packet, "nbytes", 0))
    return total


@contextlib.contextmanager
def _maybe_lease(keys, tier: str, *, prefetch: bool):
    if keys is None:
        yield None
        return
    with keys.lease(tier, prefetch=prefetch) as record:
        yield record


def attention_input_exchange(
    carrier_ciphers,
    *,
    ownership: RankOwnership,
    world,
    transport,
    carrier_layout,
    attention_layout,
    context,
    cos,
    sin,
    model_config=None,
    operator_config=None,
    keys=None,
    prefetch_keys: bool = False,
    function_prefix: str = "t2.attn_in",
    tag: str = "attn_in/lane",
    overlap: bool | None = None,
    emit_kv_cache: bool = False,
) -> AttentionInputExchangeResult:
    """Own carriers -> own lanes -> V partial -> exchange -> Q, K, finished V.

    Phase-1 schedule of the attention input on one T2 rank (docs §12.12):

    1. ``lane``   :func:`attention_input_lane_shard_fhe` on the own two
                  carriers (tier ``attn_input_lane``);
    2. ``value``  :func:`attention_input_target_shard_fhe` with
                  ``value_source_indices = own sources`` and
                  ``value_return_partial=True`` for *both* V carriers -- the
                  own-source partial of every carrier (tier ``attn_input_v``);
    3. one ``transport.all_gather`` of the two own lane ciphers and one
       ``transport.all_to_all`` that sends the partial of carrier ``c`` to
       every rank consuming ``c`` (:func:`~.ownership.value_group_ranks`);
    4. ``finish`` sums the eight partials of the own carrier and splits them
       into the two V pairs (:func:`attention_input_value_finish_fhe`);
    5. ``query``  Q-only target shard over the sixteen lanes (``attn_input_q``);
    6. ``key``    K-only target shard for the own key pair (``attn_input_k``,
                  duplicated on the two ranks that read the pair).

    ``keys`` is the rank's :class:`~.keys.KeyLifecycle` (``None`` = no
    leasing); with ``prefetch_keys=False`` the sub-tier keys load on first
    use. The value carrier
    is not repeated between ranks: each rank finishes only the carrier it
    consumes, the other partials leave the card as packets.

    ``overlap`` (default: :func:`attention_input_overlap_enabled`, env
    ``T2_ATTN_IN_OVERLAP``) reorders the schedule to hide communication
    behind GPU work without changing any result: the lane all-gather is
    *issued* right after step 1 and *waited for* after step 2 (the V
    partials only need the own lanes), the partial all-to-all is issued
    after step 2 and waited for after steps 5-6 (Q/K only need the lanes).
    Stage keys then read ``lane``, ``lane_isend``, ``value_partial``,
    ``exchange`` (lane wait + partial isend), ``unpack``, ``query``, ``key``,
    ``exchange_vpart`` (partial wait), ``unpack_vpart``, ``value_finish``.

    ``emit_kv_cache`` additionally materialises the Decode-cache snapshots
    (the monolithic adapter's ``emit_cache_c``): step 2 also produces the
    own-source compact ``V_C`` partial of every carrier, which rides in the
    same all-to-all as the Delta partials and is finished alongside them
    (``value_c_items``: pairs ``2c, 2c+1`` of the own carrier); step 6 also
    expands the folded ``K_C`` of the own key pair(s) (``key_c_items``).  The
    attention path itself is unchanged.
    """

    from llama3fhe.operators.attention.distributed.distributed_input import (
        attention_input_lane_shard_fhe,
        attention_input_target_shard_fhe,
        attention_input_value_finish_fhe,
    )
    from .ownership import key_pairs_for_rank, value_carriers_for_rank, value_group_ranks
    from .transport import ReadyCollective
    from ..wire import pack_cipher, unpack_cipher

    own = tuple(carrier_ciphers)
    own_sources = tuple(int(v) for v in ownership.cipher_indices)
    if len(own) != len(own_sources):
        raise ValueError(
            f"rank {ownership.rank} owns {len(own_sources)} carriers, got {len(own)}"
        )
    ratio = int(attention_layout.gqa_ratio)
    q_pairs = tuple(int(v) for v in ownership.query_pair_indices)
    k_pairs = key_pairs_for_rank(ownership, gqa_ratio=ratio)
    v_carriers = value_carriers_for_rank(ownership, gqa_ratio=ratio)
    all_carriers = tuple(range(int(attention_layout.key_value_pair_count) // 2))
    source_count = int(carrier_layout.cipher_count)
    world_size = int(world.world_size)
    rank = int(ownership.rank)
    stage: dict[str, float] = {}
    comm: dict[str, int] = {}
    scheduler: dict[str, float] = {}
    leases: list[Any] = []
    lane = partial = None
    lanes: list[object | None] = [None] * source_count
    partial_ciphers: list[object] = []
    value_items: tuple[tuple[int, object], ...] = ()
    query_items: tuple[tuple[int, object], ...] = ()
    key_items: tuple[tuple[int, object], ...] = ()
    key_c_items: tuple[tuple[int, object], ...] = ()
    value_c_items: tuple[tuple[int, object], ...] = ()
    emit_kv_cache = bool(emit_kv_cache)
    try:
        overlap_mode = attention_input_overlap_enabled() if overlap is None else bool(overlap)
        host = not getattr(transport, "device_payloads", False)
        local_only = world_size == 1
        lane_handle = vpart_handle = None
        received: list[Any] | None = None

        def finish_value(received_rows) -> None:
            """Unpack the own carrier's partials and finish the V pairs (steps 3b/4)."""

            nonlocal value_items, value_c_items
            started = time.perf_counter()
            summands: list[tuple[int, object]] = []
            summands_c: list[tuple[int, object]] = []
            foreign: list[object] = []
            try:
                for src in range(world_size):
                    for entry in received_rows[src]:
                        carrier, packet = int(entry[0]), entry[1]
                        packet_c = entry[2] if emit_kv_cache else None
                        if src == rank:
                            summands.append((carrier, partials_by_carrier[carrier]))
                            if emit_kv_cache:
                                summands_c.append((carrier, c_partials_by_carrier[carrier]))
                            continue
                        cipher = unpack_cipher(packet, device=context.device)
                        foreign.append(cipher)
                        summands.append((carrier, cipher))
                        if emit_kv_cache:
                            cipher_c = unpack_cipher(packet_c, device=context.device)
                            foreign.append(cipher_c)
                            summands_c.append((carrier, cipher_c))
                for carrier in v_carriers:
                    # every rank (its own FM sources) contributes one partial per carrier
                    contributors = sum(1 for c, _ in summands if c == carrier)
                    if contributors != world_size:
                        raise RuntimeError(
                            f"carrier {carrier}: {contributors} partials, expected {world_size}"
                        )
                if rank not in value_group_ranks(world, v_carriers[0], gqa_ratio=ratio):
                    raise RuntimeError("ownership: rank is not in its own value group")
                _sync(context)
                stage["unpack_vpart" if overlap_mode else "unpack"] = (
                    stage.get("unpack_vpart" if overlap_mode else "unpack", 0.0)
                    + time.perf_counter() - started
                )

                # 4. finish the own carrier(s)
                started = time.perf_counter()
                with _maybe_lease(keys, "attn_input_v", prefetch=prefetch_keys) as record:
                    value_items = attention_input_value_finish_fhe(
                        summands, attention_layout=attention_layout, crypto_context=context
                    )
                    if emit_kv_cache:
                        # same sum -> Re/Im split -> repeat as the Delta V; pairs (2c, 2c+1) of the own carrier
                        value_c_items = attention_input_value_finish_fhe(
                            summands_c, attention_layout=attention_layout, crypto_context=context
                        )
                leases.append(record)
                _sync(context)
                stage["value_finish"] = time.perf_counter() - started
            finally:
                _release(*foreign)

        # 1. own lanes
        started = time.perf_counter()
        with _maybe_lease(keys, "attn_input_lane", prefetch=prefetch_keys) as record:
            lane = attention_input_lane_shard_fhe(
                own,
                source_indices=own_sources,
                carrier_layout=carrier_layout,
                attention_layout=attention_layout,
                crypto_context=context,
                cos=cos,
                sin=sin,
                function_prefix=f"{function_prefix}.lane",
            )
        leases.append(record)
        own_lanes = tuple(lane.lane_ciphers)
        lane.lane_ciphers = ()
        for source, cipher in zip(own_sources, own_lanes, strict=True):
            lanes[source] = cipher
        if local_only:
            lane_packets: tuple[Any, ...] = ()
        else:
            lane_packets = tuple(  # packing (a no-op view with device payloads) is booked to ``lane``
                pack_cipher(cipher, tag=f"{tag}/{source}", host=host)
                for source, cipher in zip(own_sources, own_lanes, strict=True)
            )
        stage["lane"] = time.perf_counter() - started
        if overlap_mode:
            # issue the lane all-gather now; the V partials below only need the own lanes
            started = time.perf_counter()
            if local_only:
                lane_handle = ReadyCollective([lane_packets])
            else:
                lane_handle = _collective_isend(transport, "all_gather", tag, lane_packets)
            stage["lane_isend"] = time.perf_counter() - started

        # 2. own-source V partial of every carrier
        started = time.perf_counter()
        with _maybe_lease(keys, "attn_input_v", prefetch=prefetch_keys) as record:
            partial = attention_input_target_shard_fhe(
                tuple(lanes),
                query_pair_indices=(),
                key_pair_indices=(),
                value_carrier_indices=all_carriers,
                carrier_layout=carrier_layout,
                attention_layout=attention_layout,
                crypto_context=context,
                model_config=model_config,
                operator_config=operator_config,
                function_prefix=f"{function_prefix}.vpart",
                value_source_indices=own_sources,
                value_return_partial=True,
                emit_cache_c=emit_kv_cache,
            )
        leases.append(record)
        scheduler.update({f"value.{k}": float(v) for k, v in partial.scheduler_stats.items()})
        for sub, seconds in (getattr(partial, "stage_seconds", None) or {}).items():
            stage[f"value_partial.{sub}"] = float(seconds)
        partials_by_carrier = {int(c): cipher for c, cipher in partial.value_partials}
        partial.value_partials = ()
        c_partials_by_carrier: dict[int, object] = {}
        if emit_kv_cache:
            c_partials_by_carrier = {int(c): cipher for c, cipher in partial.value_c_partials}
            partial.value_c_partials = ()
        partial_ciphers = list(partials_by_carrier.values()) + list(c_partials_by_carrier.values())
        _sync(context)  # the partials' GPU tail belongs to this stage, not to the exchange
        stage["value_partial"] = time.perf_counter() - started

        # 3. communication: lanes (all-gather) + partials (all-to-all)
        started = time.perf_counter()
        if local_only:
            # the own row of the all-to-all is never unpacked (``finish_value`` uses the device partials)
            partial_packets = {carrier: None for carrier in partials_by_carrier}
            c_partial_packets = {carrier: None for carrier in c_partials_by_carrier}
        else:
            partial_packets = {
                carrier: pack_cipher(cipher, tag=f"attn_in/vpart/{carrier}/{rank}", host=host)
                for carrier, cipher in partials_by_carrier.items()
            }
            c_partial_packets = {
                carrier: pack_cipher(cipher, tag=f"attn_in/vcpart/{carrier}/{rank}", host=host)
                for carrier, cipher in c_partials_by_carrier.items()
            }
        payloads_by_dst: list[Any] = []
        for dst in range(world_size):
            wanted = value_carriers_for_rank(world[dst], gqa_ratio=ratio)
            if emit_kv_cache:
                payloads_by_dst.append(tuple((c, partial_packets[c], c_partial_packets[c]) for c in wanted))
            else:
                payloads_by_dst.append(tuple((c, partial_packets[c]) for c in wanted))
        if local_only:
            gathered = [lane_packets]
            if overlap_mode:
                vpart_handle = ReadyCollective([payloads_by_dst[0]])
            else:
                received = [payloads_by_dst[0]]
        elif overlap_mode:
            gathered = lane_handle.wait()  # one round in flight at a time: lanes first
            vpart_handle = _collective_isend(transport, "all_to_all", "attn_in/vpart", payloads_by_dst)
        else:
            gathered = transport.all_gather(tag, lane_packets)
            received = transport.all_to_all("attn_in/vpart", payloads_by_dst)
        comm["lane_sent"] = _packet_bytes(lane_packets) * (world_size - 1)
        comm["lane_received"] = sum(_packet_bytes(p) for r, p in enumerate(gathered) if r != rank)
        comm["vpart_sent"] = sum(
            _packet_bytes([entry[1:] for entry in payload]) for d, payload in enumerate(payloads_by_dst) if d != rank
        )
        stage["exchange"] = time.perf_counter() - started

        # unpack: foreign lanes into their global slots
        started = time.perf_counter()
        for src in range(world_size):
            if src == rank:
                continue
            for source, packet in zip(world[src].cipher_indices, gathered[src], strict=True):
                lanes[int(source)] = unpack_cipher(packet, device=context.device)
        if not overlap_mode:
            # baseline order: partials of the own carrier(s), then finish V before Q/K
            comm["vpart_received"] = sum(
                _packet_bytes([entry[1:] for entry in payload]) for r, payload in enumerate(received) if r != rank
            )
            stage["unpack"] = time.perf_counter() - started
            finish_value(received)
            _release(*partial_ciphers)
            partial_ciphers = []
        else:
            _sync(context)
            stage["unpack"] = time.perf_counter() - started

        # 5. Q-only target shard over all sixteen lanes
        started = time.perf_counter()
        with _maybe_lease(keys, "attn_input_q", prefetch=prefetch_keys) as record:
            q_shard = attention_input_target_shard_fhe(
                tuple(lanes),
                query_pair_indices=q_pairs,
                key_pair_indices=(),
                value_carrier_indices=(),
                carrier_layout=carrier_layout,
                attention_layout=attention_layout,
                crypto_context=context,
                model_config=model_config,
                operator_config=operator_config,
                function_prefix=f"{function_prefix}.q",
            )
        leases.append(record)
        query_items = tuple(q_shard.query_items)
        q_shard.query_items = ()
        scheduler.update({f"query.{k}": float(v) for k, v in q_shard.scheduler_stats.items()})
        for sub, seconds in (getattr(q_shard, "stage_seconds", None) or {}).items():
            stage[f"query.{sub}"] = float(seconds)
        stage["query"] = time.perf_counter() - started

        # 6. K-only target shard for the own key pair (duplicated on its two readers)
        started = time.perf_counter()
        with _maybe_lease(keys, "attn_input_k", prefetch=prefetch_keys) as record:
            k_shard = attention_input_target_shard_fhe(
                tuple(lanes),
                query_pair_indices=(),
                key_pair_indices=k_pairs,
                value_carrier_indices=(),
                carrier_layout=carrier_layout,
                attention_layout=attention_layout,
                crypto_context=context,
                model_config=model_config,
                operator_config=operator_config,
                function_prefix=f"{function_prefix}.k",
                emit_cache_c=emit_kv_cache,
            )
        leases.append(record)
        key_items = tuple(k_shard.key_items)
        k_shard.key_items = ()
        if emit_kv_cache:
            key_c_items = tuple(k_shard.key_c_items)
            k_shard.key_c_items = ()
        scheduler.update({f"key.{k}": float(v) for k, v in k_shard.scheduler_stats.items()})
        for sub, seconds in (getattr(k_shard, "stage_seconds", None) or {}).items():
            stage[f"key.{sub}"] = float(seconds)
        stage["key"] = time.perf_counter() - started

        if overlap_mode:
            # 3b. the partials arrived while Q/K ran; collect, then finish V
            started = time.perf_counter()
            received = vpart_handle.wait()
            comm["vpart_received"] = sum(
                _packet_bytes([entry[1:] for entry in payload]) for r, payload in enumerate(received) if r != rank
            )
            stage["exchange_vpart"] = time.perf_counter() - started
            finish_value(received)
            _release(*partial_ciphers)
            partial_ciphers = []

        result = AttentionInputExchangeResult(
            query_items=query_items,
            key_items=key_items,
            value_items=value_items,
            stage_seconds={k: float(v) for k, v in stage.items()},
            comm_bytes={k: int(v) for k, v in comm.items()},
            scheduler_stats=scheduler,
            lease_records=tuple(r for r in leases if r is not None),
            key_c_items=key_c_items,
            value_c_items=value_c_items,
        )
        query_items = key_items = value_items = key_c_items = value_c_items = ()
        return result
    except Exception:
        _release(*(c for _, c in query_items + key_items + value_items + key_c_items + value_c_items))
        raise
    finally:
        _release(*partial_ciphers)
        _release(*(c for c in lanes if c is not None))
        if lane is not None:
            lane.release()
        if partial is not None:
            partial.release()


@dataclass
class AttentionCoreStationResult:
    """Own ``O_C`` pairs (one per own query pair) after QK -> softmax -> PV."""

    output_items: tuple[tuple[int, object], ...]
    stage_seconds: dict[str, float]
    level_trace: dict[str, tuple[int, ...]]
    bootstrap_calls: int
    output_limbs: tuple[int, ...]
    lease_records: tuple[Any, ...] = ()
    #: the per-rank sub-layout the monolithic core ran on
    sub_layout: Any = None

    @property
    def ciphers(self) -> tuple[object, ...]:
        return tuple(cipher for _, cipher in self.output_items)

    @property
    def wall_seconds(self) -> float:
        return float(sum(v for k, v in self.stage_seconds.items() if not k.startswith("softmax.")))

    def release(self) -> None:
        owned = self.ciphers
        self.output_items = ()
        _release(*owned)


def attention_core_operator_config(schedule, *, layer_idx: int, base=None):
    """The steady-layer (``layer_idx > 0``) ``AttentionOperatorConfig`` of ``graph.py``.

    ``AttentionOperatorConfig()`` defaults differ from production in the
    fields that decide the bootstrap chain (``refresh_softmax_exp_inputs``,
    ``pair_softmax_pre_exp_bootstrap``, ``softmax_main_bootstrap_iterations``)
    and in ``rotation_chunk_size`` / ``qk_output_level``, so the station
    states every production value explicitly.  ``qk_output_level`` is
    ``None``: the inputs already sit at the adapter exit (Q 9 / K 8 limbs),
    below the layer-0 alignment target the schedule carries for the fused
    single-card graph.  ``qk_score_level_drop`` belongs to the attention
    *input* station and is left at its default here.  ``base`` (optional)
    supplies the BSGS / folding fields shared with the input station.
    """

    from dataclasses import replace

    from llama3fhe.operators.attention.config import AttentionOperatorConfig

    steady = int(layer_idx) > 0 or bool(schedule.steady_state_attention_refresh)
    refresh_inputs = schedule.refresh_softmax_exp_inputs
    if refresh_inputs is None:
        refresh_inputs = steady
    base = AttentionOperatorConfig() if base is None else base
    return replace(
        base,
        sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
        shear_baby_steps=int(schedule.shear_baby_steps),
        batched_sparse_input_bsgs=bool(schedule.batched_sparse_attention_input),
        rotation_mode="fast",
        rotation_chunk_size=int(schedule.attention_rotation_chunk_size),
        qk_output_level=None,
        qk_complex_feature_folding=True,
        qk_key_features_pre_folded=True,
        qk_query_features_pre_scaled=True,
        qk_query_features_pre_folded=True,
        pv_complex_delta_folding=bool(schedule.pv_complex_delta_folding),
        pv_prefold_after_softmax=bool(schedule.pv_prefold_after_softmax),
        refresh_softmax_exp_inputs=bool(refresh_inputs),
        refresh_softmax_exp_outputs=bool(schedule.refresh_softmax_exp_outputs),
        softmax_main_deep_bootstrap=bool(schedule.softmax_main_deep_bootstrap and steady),
        pair_softmax_pre_exp_bootstrap=bool(schedule.pair_softmax_pre_exp_bootstrap and refresh_inputs),
        reuse_softmax_round_square=bool(schedule.reuse_softmax_round_square),
        direct_softmax_slim_inactive_fill=bool(schedule.direct_softmax_slim_inactive_fill),
        fuse_softmax_slim_chebyshev_map=bool(schedule.fuse_softmax_slim_chebyshev_map),
        softmax_main_bootstrap_iterations=schedule.softmax_main_bootstrap_iterations,
        softmax_lambda_bootstrap_iterations=schedule.softmax_lambda_bootstrap_iterations,
        match_softmax_single_pass_bootstrap_level=False,
        refresh_output_before_pairfm=(str(schedule.pairfm_bootstrap_mode) == "always"),
        pairfm_refresh_min_input_limbs=(6 if str(schedule.pairfm_bootstrap_mode) == "auto" else None),
        pairfm_post_refresh_input_limbs=schedule.pairfm_post_refresh_input_limbs,
        checkpoint_bootstrap_precision_bits=int(schedule.residual_bootstrap_precision),
    )


def attention_sub_layout(attention_layout, *, query_pair_count: int, key_value_pair_count: int):
    """The ``AttentionPairLayout`` of one rank's Q pairs x K/V pairs.

    Two heads share one pair, so ``query_heads = 2 * query_pair_count``; the
    GQA ratio is preserved.  The QK / PV / softmax kernels only read pair
    counts, ``seq_len``, ``head_dim`` and ``slots`` off the layout, so the
    sub-layout runs the untouched single-card core on the rank's share.
    """

    from dataclasses import replace

    return replace(
        attention_layout,
        query_heads=2 * int(query_pair_count),
        key_value_heads=2 * int(key_value_pair_count),
    )


def attention_sub_prefix_bundle(prefix_bundle, *, key_pair_indices, attention_layout, sub_layout):
    """Keep Prefix K/V plaintext while selecting this rank's global KV heads."""

    if prefix_bundle is None:
        return None
    from llama3fhe.approx.prefix import PrefixKVBundle

    heads = np.asarray(
        [
            head
            for pair in key_pair_indices
            for head in (2 * int(pair), 2 * int(pair) + 1)
        ],
        dtype=np.int64,
    )
    expected = (
        int(prefix_bundle.prefix_k),
        int(attention_layout.key_value_heads),
        int(attention_layout.head_dim),
    )
    flat = (expected[0], expected[1] * expected[2])
    key = np.asarray(prefix_bundle.key)
    value = np.asarray(prefix_bundle.value)
    if key.shape == flat:
        key = key.reshape(expected)
    if value.shape == flat:
        value = value.reshape(expected)
    if key.shape != expected or value.shape != expected:
        raise ValueError(
            f"Prefix K/V must each have global shape {expected} or {flat}, "
            f"got {key.shape} and {value.shape}"
        )
    if np.any(heads < 0) or np.any(heads >= expected[1]):
        raise ValueError(
            f"KV pair indices {tuple(key_pair_indices)} exceed {expected[1]} Prefix heads"
        )
    if int(sub_layout.key_value_heads) == expected[1] and np.array_equal(
        heads, np.arange(expected[1])
    ):
        return prefix_bundle
    return PrefixKVBundle.from_logical(
        key[:, heads, :], value[:, heads, :], layout=sub_layout
    )


def attention_core_station(
    query_items,
    key_items,
    value_items,
    *,
    ownership: RankOwnership | None,
    attention_layout,
    context,
    softmax_config,
    operator_config,
    bootstrap_operator,
    layer_idx: int,
    keys=None,
    prefetch_keys: bool = False,
    token_valid_mask=None,
    prefix_bundle=None,
    function_prefix: str = "t2.attn_core",
):
    """QK -> Delta Alg2 softmax -> PV on this rank's own pairs, no traffic.

    ``query_items`` / ``key_items`` / ``value_items`` are ``(pair_index,
    cipher)`` lists as produced by :func:`attention_input_exchange`; V pairs
    are matched to K pairs by index (the exchange returns the own carrier's
    two pairs, one of which is the rank's K pair).  The single-card
    :func:`attention_fhe_delta_alg2` is run unchanged on the sub-layout
    (2 Q pairs x 1 KV pair for eight ranks), which keeps its softmax slim
    packing and paired checkpoints: one paired pre-exp BS, one lambda BS and
    one main checkpoint BS per rank without Prefix.  ``prefix_bundle`` is a
    global plaintext K/V page; this station selects this rank's KV heads and
    passes them to the plaintext-ciphertext Prefix QK/PV paths.

    Key lifecycle (``keys`` given): the QK-only tier is leased first; inside
    ``on_qk_last_use`` it is released and the ``softmax`` + ``attn_pv``
    tiers are leased *before* the QK evict runs, so the 64 rotations QK and
    PV share survive.  The QK / PV unions do not depend on the pair count,
    so the per-rank lease equals the full-layout lease (~7.8 + 7.9 GiB at
    cap 13) and only one of them is on the card at a time.  Inputs are not
    consumed.
    """

    from llama3fhe.operators.attention.prefill.operator import attention_fhe_delta_alg2

    query_items = tuple((int(i), c) for i, c in query_items)
    key_items = tuple((int(i), c) for i, c in key_items)
    value_by_pair = {int(i): c for i, c in value_items}
    if not query_items or not key_items:
        raise ValueError("attention_core_station needs at least one Q pair and one K pair")
    ratio = int(attention_layout.gqa_ratio)
    k_pairs = tuple(sorted(i for i, _ in key_items))
    q_pairs = tuple(sorted(i for i, _ in query_items))
    # every Q pair must read one of the given K pairs (full-layout GQA map)
    # and every K pair must serve the same number of Q pairs, so the
    # sub-layout ratio len(q)/len(k) is an integer (2 for a T2 rank, 4 full)
    groups: dict[int, int] = {k: 0 for k in k_pairs}
    for q in q_pairs:
        if q // ratio not in groups:
            raise ValueError(f"query pair {q} reads key pair {q // ratio}, not among {k_pairs}")
        groups[q // ratio] += 1
    if len(set(groups.values())) != 1 or 0 in groups.values():
        raise ValueError(f"query pairs {q_pairs} are not evenly grouped over key pairs {k_pairs}: {groups}")
    missing = [k for k in k_pairs if k not in value_by_pair]
    if missing:
        raise ValueError(f"value pairs {sorted(value_by_pair)} lack the K pairs {missing}")
    if ownership is not None:
        own_q = tuple(sorted(int(v) for v in ownership.query_pair_indices))
        if own_q != q_pairs:
            raise ValueError(f"rank {ownership.rank} owns query pairs {own_q}, got {q_pairs}")

    sub_layout = attention_sub_layout(attention_layout, query_pair_count=len(q_pairs), key_value_pair_count=len(k_pairs))
    local_prefix_bundle = attention_sub_prefix_bundle(
        prefix_bundle,
        key_pair_indices=k_pairs,
        attention_layout=attention_layout,
        sub_layout=sub_layout,
    )
    query_ciphers = tuple(dict(query_items)[i] for i in q_pairs)
    key_ciphers = tuple(dict(key_items)[i] for i in k_pairs)
    value_ciphers = tuple(value_by_pair[i] for i in k_pairs)

    leases: list[Any] = []
    qk_stack = contextlib.ExitStack()
    rest_stack = contextlib.ExitStack()
    calls_before = int(getattr(bootstrap_operator, "calls", 0))
    stage: dict[str, float] = {}

    def _lease(stack, tier):
        if keys is None:
            return None
        record = stack.enter_context(keys.lease(tier, prefetch=False))
        leases.append(record)
        return record

    def _prefetch(tier, record):
        if keys is not None and prefetch_keys and record is not None:
            count, seconds = keys.prefetch(tier)
            record.prefetched_keys = int(count)
            record.prefetch_seconds = float(seconds)

    def on_qk_last_use():
        started = time.perf_counter()
        # enter the downstream tiers first so the QK evict keeps their overlap
        sm = _lease(rest_stack, "softmax")
        pv = _lease(rest_stack, "attn_pv")
        qk_stack.close()
        _prefetch("softmax", sm)
        _prefetch("attn_pv", pv)
        stage["lease_qk_to_pv"] = time.perf_counter() - started

    def on_pv_last_use():
        started = time.perf_counter()
        rest_stack.close()
        stage["lease_pv_exit"] = time.perf_counter() - started

    result = None
    try:
        started = time.perf_counter()
        qk = _lease(qk_stack, "attn_qk")
        _prefetch("attn_qk", qk)
        stage["lease_qk_enter"] = time.perf_counter() - started
        result = attention_fhe_delta_alg2(
            query_ciphers,
            key_ciphers,
            value_ciphers,
            layout=sub_layout,
            crypto_context=context,
            softmax_config=softmax_config,
            operator_config=operator_config,
            layer_idx=int(layer_idx),
            softmax_bootstrap=bootstrap_operator,
            token_valid_mask=token_valid_mask,
            prefix_bundle=local_prefix_bundle,
            on_qk_last_use=on_qk_last_use,
            on_pv_last_use=on_pv_last_use,
        )
    finally:
        qk_stack.close()
        rest_stack.close()
    _sync(context)
    outputs = tuple(result.output_ciphers)
    result.output_ciphers = ()
    stage.update({k: float(v) for k, v in result.stage_seconds.items()})
    return AttentionCoreStationResult(
        output_items=tuple(zip(q_pairs, outputs, strict=True)),
        stage_seconds=stage,
        level_trace={k: tuple(int(x) for x in v) for k, v in result.level_trace.items()},
        bootstrap_calls=int(getattr(bootstrap_operator, "calls", 0)) - calls_before,
        output_limbs=_limbs(outputs),
        lease_records=tuple(r for r in leases if r is not None),
        sub_layout=sub_layout,
    )


@dataclass
class PairFMExchangeResult:
    """This rank's W_O carriers (``ownership.target_carrier_indices``) after the reduce-scatter."""

    carriers: tuple[object, ...]
    carrier_indices: tuple[int, ...]
    stage_seconds: dict[str, float]
    source_stage_seconds: dict[str, float]
    comm_bytes: dict[str, int]
    bootstrap_calls: int
    input_limbs: tuple[int, ...]
    partial_limbs: int
    output_limbs: tuple[int, ...]
    lease_records: tuple[Any, ...] = ()
    source_slot_seconds: tuple[float, ...] = ()

    @property
    def ciphers(self) -> tuple[object, ...]:
        return tuple(c for c in self.carriers if c is not None)

    @property
    def carrier(self) -> object:
        """The single carrier of a one-slot rank (``world_size == 8``)."""

        if len(self.carriers) != 1:
            raise ValueError(
                f"carrier requires exactly one owned carrier; this rank owns {self.carrier_indices} (use carriers)"
            )
        return self.carriers[0]

    @property
    def carrier_index(self) -> int:
        if len(self.carrier_indices) != 1:
            raise ValueError(
                f"carrier_index requires exactly one owned carrier; this rank owns {self.carrier_indices}"
            )
        return int(self.carrier_indices[0])

    @property
    def wall_seconds(self) -> float:
        return float(sum(self.stage_seconds.values()))

    def release(self) -> None:
        owned = self.ciphers
        self.carriers = ()
        _release(*owned)


def pairfm_exchange(
    output_items,
    *,
    ownership: RankOwnership,
    world,
    transport,
    feature_layout,
    attention_layout,
    context,
    model_config=None,
    operator_config=None,
    bootstrap_operator=None,
    keys=None,
    prefetch_keys: bool = False,
    function_prefix: str = "t2.pairfm",
    tag: str = "pairfm/partial",
    after_exchange=None,
) -> PairFMExchangeResult:
    """Own ``O_C`` -> 8 pre-rescale partials -> all-to-all -> own W_O carriers.

    ``output_items`` are the ``(global query pair, cipher)`` items of
    :class:`AttentionCoreStationResult` (this rank's query pairs, two per
    owned slot); they are not consumed.  Device-local halves are
    :func:`~llama3fhe.operators.attention.distributed.distributed_output.pairfm_rank_partials_fhe`
    (per slot: pair the two ``O_C`` -> refresh when below the pairing
    threshold (``operator_config.pairfm_refresh_min_input_limbs``, 6 in the
    production graph; ``O_C`` leaves the core at 4 limbs on L3 so every slot
    pays this one BS) -> two slot permutations -> 8 partials, one per target
    carrier; then the same-target partials of the rank's slots are pre-added
    on the device) and
    :func:`~llama3fhe.operators.attention.distributed.distributed_output.pairfm_target_reduce_fhe`
    (per owned carrier: sum the ``world_size`` rank partials and rescale
    once).  The middle is one ``transport.all_to_all`` of
    :class:`~llama3fhe.parallel.wire.CipherPacket` lists: rank ``r`` sends
    rank ``d`` one packet per carrier ``d`` owns (``slots_per_rank`` packets,
    ``world_size - 1`` destinations).  The source half runs under the
    ``pairfm_output`` lease.

    ``after_exchange`` is an optional zero-argument hook invoked immediately
    after this rank's PairFM all-to-all receive completes.  The Prefill KV
    side pipeline uses it to release its final all-gather, so that gather can
    overlap the local PairFM target reduction while the transport still has
    only one collective round in flight.
    """

    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.operators.attention.distributed.distributed_output import (
        pairfm_rank_partials_fhe,
        pairfm_target_reduce_fhe,
    )

    from ..wire import pack_cipher, unpack_cipher

    rank = int(ownership.rank)
    world_size = int(world.world_size)
    carrier_indices = tuple(int(v) for v in ownership.target_carrier_indices)
    slot_indices = tuple(int(v) for v in ownership.slot_indices)
    if carrier_indices != slot_indices:
        raise ValueError(
            f"rank {rank}: PairFM needs target carriers == owned slots, got {carrier_indices} vs {slot_indices}"
        )
    items = tuple((int(pair), cipher) for pair, cipher in output_items)
    expected_pairs = tuple(int(p) for p in ownership.query_pair_indices)
    if tuple(p for p, _ in items) != expected_pairs:
        raise ValueError(
            f"rank {rank}: pairfm_exchange expects the own query pairs {expected_pairs}, got {[p for p, _ in items]}"
        )
    slot_pairs = ownership.slot_query_pairs()  # ((2s, 2s+1), ...) per owned slot
    by_pair = dict(items)
    ciphers_by_slot = tuple(tuple(by_pair[p] for p in pair_group) for pair_group in slot_pairs)
    carrier_count = int(world.slot_count)
    if carrier_count != int(feature_layout.cipher_count) // 2:
        raise ValueError(
            f"world has {carrier_count} carriers but the layout has {int(feature_layout.cipher_count) // 2}"
        )
    operator_config = AttentionOperatorConfig() if operator_config is None else operator_config
    ciphers = tuple(cipher for _, cipher in items)

    stage: dict[str, float] = {}
    comm: dict[str, int] = {}
    leases: list[Any] = []
    calls_before = int(getattr(bootstrap_operator, "calls", 0)) if bootstrap_operator is not None else 0
    input_limbs = _limbs(ciphers)

    # 1. source half per slot, same-target partials pre-added on the device
    #    (8 partials, one per target carrier) under the pairfm_output lease
    started = time.perf_counter()
    with _maybe_lease(keys, "pairfm_output", prefetch=prefetch_keys) as record:
        source = pairfm_rank_partials_fhe(
            ciphers_by_slot,
            slot_indices=slot_indices,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            crypto_context=context,
            model_config=model_config,
            operator_config=operator_config,
            bootstrap_operator=bootstrap_operator,
            function_prefix=f"{function_prefix}.source",
        )
    leases.append(record)
    _sync(context)
    stage["source"] = time.perf_counter() - started
    for sub, seconds in (getattr(source, "stage_seconds", None) or {}).items():
        stage[f"source.{sub}"] = float(seconds)
    partials: list[object | None] = list(source.partial_ciphers)
    source.partial_ciphers = ()
    if len(partials) != carrier_count:
        _release(*partials)
        raise RuntimeError(f"rank {rank}: {len(partials)} partials for {carrier_count} target carriers")
    partial_limbs = int(partials[0].state.cur_limbs)

    carriers: list[object] = []
    summands_by_carrier: list[list[object]] = [[] for _ in carrier_indices]
    try:
        # 2. all-to-all: the partial for carrier c goes to the rank owning c;
        #    rank d receives one packet per carrier it owns
        started = time.perf_counter()
        host = not getattr(transport, "device_payloads", False)
        payloads_by_dst: list[Any] = []
        for dst in range(world_size):
            if dst == rank:
                payloads_by_dst.append(None)
                continue
            payloads_by_dst.append(
                tuple(
                    pack_cipher(partials[target], tag=f"{tag}/{target}/{rank}", host=host)
                    for target in world[dst].target_carrier_indices
                )
            )
        received = transport.all_to_all(tag, payloads_by_dst)
        # A side-stream Prefill KV emitter may be holding its final
        # all-gather until this round completes.  Signal immediately after
        # the transport receive half so its gather can overlap our local
        # unpack/target-reduce work without ever having two collectives in
        # flight on the same transport.
        if after_exchange is not None:
            after_exchange()
        comm["partial_sent"] = _packet_bytes(payloads_by_dst)
        comm["partial_received"] = sum(_packet_bytes([p]) for r, p in enumerate(received) if r != rank)
        stage["exchange"] = time.perf_counter() - started

        # 3. unpack the foreign partials of the own carriers, release the sent
        started = time.perf_counter()
        for src in range(world_size):
            if src == rank:
                for position, target in enumerate(carrier_indices):
                    summands_by_carrier[position].append(partials[target])
                    partials[target] = None
                continue
            packets = received[src]
            if packets is None or len(packets) != len(carrier_indices):
                raise RuntimeError(
                    f"rank {rank}: expected {len(carrier_indices)} partial packets from rank {src}, "
                    f"got {None if packets is None else len(packets)}"
                )
            for position, packet in enumerate(packets):
                cipher = unpack_cipher(packet, device=context.device)
                summands_by_carrier[position].append(cipher)
        _release(*(c for c in partials if c is not None))
        partials = []
        _sync(context)
        stage["unpack"] = time.perf_counter() - started

        # 4. per owned carrier: reduce + one rescale -> own W_O carrier
        #    (consumes the summands, also on failure -- hand them over first)
        started = time.perf_counter()
        for position in range(len(carrier_indices)):
            summands = summands_by_carrier[position]
            summands_by_carrier[position] = []
            carriers.append(pairfm_target_reduce_fhe(summands, crypto_context=context, synchronize=True))
        stage["reduce"] = time.perf_counter() - started
    except Exception:
        # everything still owned here: unsent/own partials, unpacked foreign
        # partials not yet handed to a reduce, and carriers already produced
        _release(*(c for c in partials if c is not None))
        _release(*(c for group in summands_by_carrier for c in group))
        _release(*carriers)
        raise

    calls_after = int(getattr(bootstrap_operator, "calls", 0)) if bootstrap_operator is not None else 0
    return PairFMExchangeResult(
        carriers=tuple(carriers),
        carrier_indices=carrier_indices,
        stage_seconds=stage,
        source_stage_seconds={str(k): float(v) for k, v in source.stage_seconds.items()},
        comm_bytes=comm,
        bootstrap_calls=calls_after - calls_before,
        input_limbs=input_limbs,
        partial_limbs=partial_limbs,
        output_limbs=tuple(int(c.state.cur_limbs) for c in carriers),
        lease_records=tuple(leases),
        source_slot_seconds=tuple(float(v) for v in source.slot_wall_seconds),
    )


# ---------------------------------------------------------------------- W_O
def w_o_station(
    carriers,
    o_weight,
    *,
    ownership: RankOwnership,
    layout,
    attention_layout,
    context,
    operator_config=None,
    profile: dict[str, float] | None = None,
):
    """Own W_O carriers -> own FM shards (:func:`w_o_carrier_shards_fhe`).

    Carrier ``c`` holds token pair ``(2c, 2c+1)`` = FM ciphers ``2c, 2c+1``;
    a rank's carriers are its own slots, so the projection lands on the
    residual stream without any communication.  ``carriers`` is one carrier
    (one-slot rank) or the tuple returned by :func:`pairfm_exchange`
    (``PairFMExchangeResult.carriers``), in ``target_carrier_indices`` order.
    Returns ``ciphers_per_rank`` FM shards in ``cipher_indices`` order.  The
    carriers are not consumed.
    """

    import numpy as np

    from llama3fhe.operators.linear.prefill.output import w_o_carrier_shards_fhe

    if isinstance(carriers, (tuple, list)):
        inputs = tuple(carriers)
    else:
        inputs = (carriers,)
    expected = tuple(int(v) for v in ownership.target_carrier_indices)
    if len(inputs) != len(expected):
        raise ValueError(
            f"rank {int(ownership.rank)}: w_o_station got {len(inputs)} carriers for carriers {expected}"
        )
    return w_o_carrier_shards_fhe(
        inputs,
        o_weight,
        carrier_indices=expected,
        feature_layout=layout,
        attention_layout=attention_layout,
        crypto_context=context,
        operator_config=operator_config,
        dtype=np.float64,
        profile=profile,
    )


__all__ = [
    "AttentionCoreStationResult",
    "AttentionInputExchangeResult",
    "MLP_SUFFIX_LIMBS",
    "QKV_INPUT_LIMBS",
    "RMSNormStationResult",
    "attention_core_operator_config",
    "attention_core_station",
    "attention_input_exchange",
    "attention_sub_layout",
    "PairFMExchangeResult",
    "halve",
    "linear_entry",
    "mlp_input_limbs",
    "mlp_station",
    "pair_bootstrap",
    "pairfm_exchange",
    "qkv_station",
    "residual_station",
    "rmsnorm_station",
    "w_o_station",
]
