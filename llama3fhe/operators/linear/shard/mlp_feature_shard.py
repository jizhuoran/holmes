from __future__ import annotations

"""Feature-sharded (page-split) feature-major MLP for the T2 parallel layer.

Token sharding (:mod:`.mlp_token_shard`) gives every rank its own two FM
ciphertexts and the *whole* gate/up/down weight: each rank encodes all
``4 pages x 16 chunks`` diagonal plaintexts itself. This encoding work is
repeated on every rank under pure token sharding.

Here the ranks of one *shard group* (``k`` adjacent ranks, ``rank // k``)
split the *intermediate pages* between them (Megatron-style tensor
parallelism of the SwiGLU hidden dimension): member ``p`` owns pages
``p*pages/k .. (p+1)*pages/k`` and, for the tokens of *every* member of the
group, computes gate/up of its pages, SiLU-gates them and multiplies them by
the matching column block of the down weight.  Two collectives per MLP move
the ``2`` FM inputs of every member in (``gu_in``) and the low-limb,
pre-rescale down partials back to their owner (``dn_acc``); the owner sums
them (``homo_add`` at scale degree 2) and rescales once.  The encode cost
drops by ``1 - 1/k``; the hoisted baby basis of a foreign input is recomputed
locally. Presets configure ``k`` according to device capacity.
``k = 1`` is exactly :func:`~.mlp.mlp_fhe`.

Page splitting uses two collectives to exchange inputs and return partials.

All ranks of the world must run this with the same ``feature_shard_k`` and
the same transport call order (the collectives are world-synchronous); ranks
outside a group exchange empty payloads with it.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_all, release_if_supported
from llama3fhe.backend import profile_sync_device as sync_device  # stage-timer sync, no-op under T2_STAGE_SYNC=0
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.nonlinear.elementwise import eval_silu_chebyshev_cipher

from ....approx.chebyshev import chebyshev_ps_mul_depth, eval_chebyshev_series_direct
from ...primitives import mul_cipher_rescale as _mul_cipher_rescale
from .. import (
    LLAMA3_LINEAR_SCHEDULES,
    LinearOperator,
    LinearOperatorConfig,
    pair_real_ciphers,
    split_complex_cipher_twice,
    structured_linear_weight,
    structured_weight_packing_enabled,
)
from ..prefill.mlp import FeatureMajorMLPComplexity, FeatureMajorMLPLayout, mlp_complexity


# ------------------------------------------------------------------ grouping
def feature_shard_group(rank: int, world_size: int, k: int) -> tuple[int, ...]:
    """Ranks of the shard group that contains ``rank`` (``k`` adjacent ranks)."""

    rank, world_size, k = int(rank), int(world_size), int(k)
    if k <= 0:
        raise ValueError("feature_shard_k must be positive")
    if world_size % k:
        raise ValueError(f"world_size {world_size} is not a multiple of feature_shard_k {k}")
    if rank < 0 or rank >= world_size:
        raise IndexError(f"rank {rank} outside world of size {world_size}")
    base = (rank // k) * k
    return tuple(range(base, base + k))


def feature_shard_pages(page_count: int, position: int, k: int) -> tuple[int, ...]:
    """Contiguous intermediate pages of group member ``position`` (``k`` members).

    Every page costs the same in FHE (the last one is zero padded to the page
    width), so a contiguous even split is balanced by construction.
    """

    page_count, position, k = int(page_count), int(position), int(k)
    if k <= 0 or position < 0 or position >= k:
        raise ValueError(f"position {position} outside a group of {k}")
    if page_count % k:
        raise ValueError(f"{page_count} intermediate pages do not split evenly over {k} ranks")
    width = page_count // k
    return tuple(range(position * width, (position + 1) * width))


@dataclass
class MLPFeatureShardResult:
    """Own FM outputs plus timing / accounting of the sharded MLP."""

    ciphers: tuple[object, ...]
    complexity: FeatureMajorMLPComplexity
    wall_seconds: float
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]
    #: bytes this rank put on the wire / took off it, per collective
    comm_bytes: dict[str, int]
    feature_shard_k: int
    group: tuple[int, ...]
    #: intermediate pages this rank owns (gate/up + SiLU + down column block)
    pages: tuple[int, ...]
    output: np.ndarray | None = None
    expected: np.ndarray | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        release_all(self.ciphers)
        self.ciphers = ()
        self._released = True


def _add_profile(destination: dict[str, float], prefix: str, profile: dict[str, float]) -> None:
    for name, seconds in profile.items():
        key = f"{prefix}_{name}"
        destination[key] = float(destination.get(key, 0.0) + float(seconds))


def _packet_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(_packet_bytes(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


def _exchange(
    transport,
    tag: str,
    *,
    rank: int,
    group: tuple[int, ...],
    payload_for: dict[int, list],
    comm: dict[str, int],
) -> dict[int, list]:
    """One world all-to-all whose only non-empty payloads go to the group peers.

    ``payload_for[peer]`` is the list of ``(key, CipherPacket)`` items for that
    peer; returns the items every group peer sent to us, keyed by source rank.
    """

    payloads_by_dst: list[Any] = [None] * int(transport.world_size)
    for peer in group:
        if peer != rank:
            payloads_by_dst[peer] = tuple(payload_for.get(peer, ()))
    received = transport.all_to_all(tag, payloads_by_dst)
    comm[f"{tag}_sent"] = sum(
        _packet_bytes([packet for _, packet in payload])
        for dst, payload in enumerate(payloads_by_dst)
        if payload is not None and dst != rank
    )
    comm[f"{tag}_received"] = sum(
        _packet_bytes([packet for _, packet in received[src]])
        for src in group
        if src != rank and received[src] is not None
    )
    return {src: list(received[src] or ()) for src in group if src != rank}


# ------------------------------------------------------------------- kernel
def mlp_feature_shard_fhe(
    hidden_states: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    silu_entry: dict,
    layout: FeatureMajorPrefillLayout,
    rank: int,
    feature_shard_k: int,
    transport,
    config: Llama3CKKSConfig | None = None,
    operator_config: LinearOperatorConfig | None = None,
    crypto_context=None,
    input_ciphers: tuple[object, ...] | list[object] | None = None,
    input_column_scale: np.ndarray | None = None,
    fuse_silu_domain_map: bool = False,
    verify: bool = False,
    tag: str = "mlp",
) -> MLPFeatureShardResult:
    """Run the feature-major SwiGLU MLP on this rank's FM ciphertexts with the
    intermediate pages split over the ``feature_shard_k`` ranks of the group
    (see the module docstring).

    ``layout`` / ``hidden_states`` describe *this rank's* tokens (the same
    local view :func:`~.mlp_token_shard.mlp_token_shard_fhe` builds); every
    rank of the world must call this the same number of times with the same
    ``tag`` (the collectives match on it -- never put the rank in it).
    """

    if crypto_context is None or input_ciphers is None:
        raise ValueError("mlp_feature_shard_fhe requires a context and encrypted inputs")
    if transport is None:
        raise ValueError("mlp_feature_shard_fhe requires a transport")
    rank = int(rank)
    k = int(feature_shard_k)
    world_size = int(transport.world_size)
    group = feature_shard_group(rank, world_size, k)
    position = group.index(rank)
    peers = tuple(peer for peer in group if peer != rank)

    operator_config = (
        LLAMA3_LINEAR_SCHEDULES.gate_up if operator_config is None else operator_config
    )
    if not bool(operator_config.reuse_baby_rotations):
        raise ValueError("the feature-major FHE operator requires reuse_baby_rotations=True.")
    config = Llama3CKKSConfig() if config is None else config
    config_slots = 1 << (int(config.simulator.logN) - 1)
    if int(layout.slots) != config_slots:
        raise ValueError(
            "feature-major MLP layout/config slot mismatch: "
            f"layout.slots={layout.slots}, config slots={config_slots}."
        )
    if int(crypto_context.max_slots) != int(layout.slots):
        raise ValueError("feature-major MLP layout/context slot mismatch.")

    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    gate_weight = np.asarray(gate_weight, dtype=np.float32)
    up_weight = np.asarray(up_weight, dtype=np.float32)
    down_weight = np.asarray(down_weight, dtype=np.float32)
    expected_hidden = (int(layout.seq_len), int(layout.hidden_dim))
    if hidden_states.shape != expected_hidden:
        raise ValueError(f"hidden_states must have shape {expected_hidden}, got {hidden_states.shape}.")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape.")
    intermediate_width, hidden_dim = gate_weight.shape
    if hidden_dim != int(layout.hidden_dim):
        raise ValueError("gate/up input width must match layout.hidden_dim.")
    if down_weight.shape != (hidden_dim, intermediate_width):
        raise ValueError(f"down_weight must have shape {(hidden_dim, intermediate_width)}, got {down_weight.shape}.")

    coefficients = np.asarray(silu_entry["coefficients"], dtype=np.float64)
    lower_bound, upper_bound = (float(value) for value in silu_entry["fit_interval"])
    column_scale = None
    if input_column_scale is not None:
        column_scale = np.asarray(input_column_scale, dtype=np.float32).reshape(-1)
        if column_scale.shape != (hidden_dim,):
            raise ValueError(f"MLP input_column_scale must have shape {(hidden_dim,)}, got {column_scale.shape}.")
        if not np.all(np.isfinite(column_scale)):
            raise ValueError("MLP input_column_scale must be finite.")
    gate_alpha = 2.0 / (upper_bound - lower_bound)
    gate_beta = -1.0 - 2.0 * lower_bound / (upper_bound - lower_bound)
    gate_scale = gate_alpha if bool(fuse_silu_domain_map) else 1.0

    mlp_layout = FeatureMajorMLPLayout(layout, intermediate_width)
    pages = int(mlp_layout.feature_page_count)
    pairs = int(mlp_layout.paired_token_cipher_count)
    page_width = int(mlp_layout.page_width)
    my_pages = feature_shard_pages(pages, position, k)
    complexity = mlp_complexity(
        layout,
        intermediate_width=intermediate_width,
        silu_polynomial_degree=max(0, int(coefficients.size) - 1),
        silu_multiplicative_depth=chebyshev_ps_mul_depth(coefficients),
        baby_steps=int(operator_config.baby_steps),
        max_plaintext_rows=int(operator_config.max_plaintext_rows),
        reuse_baby_rotations_across_outputs=True,
        fuse_silu_domain_map=bool(fuse_silu_domain_map),
    )

    input_ciphers = tuple(input_ciphers)
    if len(input_ciphers) != int(layout.cipher_count):
        raise ValueError(f"expected {layout.cipher_count} input ciphers, got {len(input_ciphers)}.")
    if len({int(crypto_context.level_for_cipher(c)) for c in input_ciphers}) != 1:
        raise ValueError("feature-major MLP inputs must enter at one common level.")

    linear_config = LinearOperatorConfig(
        baby_steps=int(operator_config.baby_steps),
        baby_anchor_step=int(operator_config.baby_anchor_step),
        max_plaintext_rows=int(operator_config.max_plaintext_rows),
        reuse_baby_rotations=True,
        hoist_strategy=str(operator_config.hoist_strategy),
    )
    gate_up_operator = LinearOperator(
        crypto_context=crypto_context, token_lanes=int(layout.tokens_per_cipher), config=linear_config
    )
    down_operator = LinearOperator(
        crypto_context=crypto_context, token_lanes=int(layout.tokens_per_cipher), config=linear_config
    )
    conjugation = int(crypto_context.context.M) - 1
    use_structured_weight = structured_weight_packing_enabled()

    from llama3fhe.parallel.wire import pack_cipher, unpack_cipher

    wall_start = time.perf_counter()
    stage: dict[str, float] = {}
    comm: dict[str, int] = {}
    # everything device-side this function owns, released in ``finally``
    paired: tuple[object, ...] = ()
    conjugated: tuple[object, ...] = ()
    foreign_inputs: list[object] = []
    group_inputs: list[object] = []  # ordered: member 0 (paired.., conj..), member 1, ...
    projected_page: list[object | None] = []
    gate_half: list[object] = []  # [(member, my page, pair)] flattened, member-major
    up_half: list[object] = []
    gated_complex: list[object] = []  # same order
    down_partials: list[object | None] = []  # [member * pairs + pair], pre-rescale
    down_accumulators: list[object | None] = [None] * pairs
    outputs: list[object] = []
    success = False

    def _sync() -> None:
        sync_device(crypto_context.device)

    def _homo_add_consume(current, term):
        """``current + term`` releasing both operands."""

        combined = crypto_context.fhe.homo_add(current, term, crypto_context.context)
        release_if_supported(current)
        release_if_supported(term)
        return combined

    def _gate_up_weight_page(page: int):
        active = mlp_layout.active_width(page)
        feature_start = page * page_width
        if use_structured_weight:
            output_rows = np.arange(page_width, dtype=np.int64)
            active_rows = output_rows < active
            source_rows = np.where(active_rows, output_rows, -1)
            return structured_linear_weight(
                (
                    gate_weight[feature_start : feature_start + active],
                    up_weight[feature_start : feature_start + active],
                ),
                dimension=page_width,
                real_source=np.where(active_rows, 0, -1),
                real_row=source_rows,
                real_scale=0.25 * gate_scale,
                imag_source=np.where(active_rows, 1, -1),
                imag_row=source_rows,
                imag_scale=0.25,
                column_scale=(1.0 if column_scale is None else column_scale),
            )
        weight = np.zeros((page_width, page_width), dtype=np.complex128)
        gate_page = gate_weight[feature_start : feature_start + active].astype(np.complex128)
        up_page = up_weight[feature_start : feature_start + active].astype(np.complex128)
        if column_scale is not None:
            gate_page *= column_scale[None, :]
            up_page *= column_scale[None, :]
        weight[:active] = 0.25 * (gate_scale * gate_page + 1j * up_page)
        return weight

    def _down_weight_page(page: int):
        active = mlp_layout.active_width(page)
        feature_start = page * page_width
        if use_structured_weight:
            columns = np.full((page_width,), -1, dtype=np.int64)
            columns[:active] = np.arange(active, dtype=np.int64)
            return structured_linear_weight(
                (down_weight[:, feature_start : feature_start + active],),
                dimension=page_width,
                real_source=np.zeros((page_width,), dtype=np.int64),
                real_row=np.arange(page_width, dtype=np.int64),
                real_scale=0.5,
                column=columns,
            )
        weight = np.zeros((page_width, page_width), dtype=np.float64)
        weight[:, :active] = 0.5 * down_weight[:, feature_start : feature_start + active]
        return weight

    try:
        # ------------------------------------------------ 1. pair + conjugate
        started = time.perf_counter()
        paired = pair_real_ciphers(input_ciphers, crypto_context=crypto_context)
        if len(paired) != pairs:
            raise RuntimeError(f"pairing produced {len(paired)} ciphers, layout expects {pairs}")
        conj_list: list[object] = []
        try:
            for cipher in paired:
                conj_list.append(crypto_context.fhe.homo_rotate(cipher, conjugation, crypto_context.context))
            conjugated = tuple(conj_list)
            conj_list = []
        finally:
            release_all(conj_list)
        _sync()
        stage["input_pair_and_conjugate"] = time.perf_counter() - started
        own_inputs = tuple(paired) + tuple(conjugated)  # 2 * pairs
        per_member = len(own_inputs)

        # ------------------------------------- 2. exchange gate/up inputs
        started = time.perf_counter()
        host = not getattr(transport, "device_payloads", False)
        own_packets = [pack_cipher(c, tag=f"{tag}/gu_in/{rank}/{j}", host=host) for j, c in enumerate(own_inputs)]
        payload_for = {peer: [(j, packet) for j, packet in enumerate(own_packets)] for peer in peers}
        received = _exchange(transport, f"{tag}/gu_in", rank=rank, group=group, payload_for=payload_for, comm=comm)
        own_packets = []
        for member in group:
            if member == rank:
                group_inputs.extend(own_inputs)
                continue
            items = sorted(received[member], key=lambda item: int(item[0]))
            if [int(j) for j, _ in items] != list(range(per_member)):
                raise RuntimeError(f"rank {rank}: incomplete gate/up inputs from {member}")
            for _, packet in items:
                cipher = unpack_cipher(packet, device=crypto_context.device)
                foreign_inputs.append(cipher)
                group_inputs.append(cipher)
        received = {}
        _sync()
        stage["gate_up_input_exchange"] = time.perf_counter() - started

        # ------------------- 3. gate/up of my pages for every member's tokens
        profile = gate_up_operator.prepare_baby_rotations(group_inputs, dimension=page_width)
        _add_profile(stage, "gate_up_projection", profile)
        for page in my_pages:
            weight_start = time.perf_counter()
            weight = _gate_up_weight_page(page)
            stage["gate_up_weight_transform_cpu"] = float(
                stage.get("gate_up_weight_transform_cpu", 0.0) + time.perf_counter() - weight_start
            )
            projected, profile = gate_up_operator.project(group_inputs, weight, dtype=np.dtype(np.complex128))
            _add_profile(stage, "gate_up_projection", profile)
            projected_page = list(projected)
            if len(projected_page) != k * per_member:
                raise RuntimeError(f"gate/up page {page} produced {len(projected_page)} ciphers, expected {k * per_member}")
            started = time.perf_counter()
            for member_index in range(k):
                base = member_index * per_member
                for j in range(pairs):
                    direct = projected_page[base + j]
                    conjugate_evaluation = projected_page[base + pairs + j]
                    conjugated_output = crypto_context.fhe.homo_rotate(
                        conjugate_evaluation, conjugation, crypto_context.context
                    )
                    try:
                        gate_half.append(crypto_context.fhe.homo_add(direct, conjugated_output, crypto_context.context))
                        delta = crypto_context.fhe.homo_sub(direct, conjugated_output, crypto_context.context)
                        try:
                            up_half.append(fhe.homo_mul_i(delta, crypto_context.context, negative=True))
                        finally:
                            release_if_supported(delta)
                    finally:
                        release_if_supported(conjugated_output)
            release_all(projected_page)
            projected_page = []
            _sync()
            stage["gate_up_combine"] = float(stage.get("gate_up_combine", 0.0) + time.perf_counter() - started)
        gate_up_operator.clear_baby_cache()
        release_all(foreign_inputs)
        foreign_inputs = []
        group_inputs = []
        # gate_half / up_half order: (my page, member, pair)

        # -------------------------------------------------------- 4. SiLU
        started = time.perf_counter()
        for gate_cipher, up_cipher in zip(gate_half, up_half, strict=True):
            gate_real = gate_imag = up_real = up_imag = None
            activated_real = activated_imag = gated_real = gated_imag = imaginary_i = None
            try:
                gate_real, gate_imag = split_complex_cipher_twice(gate_cipher, crypto_context=crypto_context)
                up_real, up_imag = split_complex_cipher_twice(up_cipher, crypto_context=crypto_context)
                if bool(fuse_silu_domain_map):
                    mapped = []
                    try:
                        for part in (gate_real, gate_imag):
                            beta = fhe.encode_scalar(
                                gate_beta,
                                cur_limbs=int(part.state.cur_limbs),
                                scale_degree=1,
                                scaling_factor=part.state.scaling_factor,
                                context=crypto_context.context,
                            )
                            mapped.append(fhe.homo_add_scalar(part, beta, crypto_context.context))
                    except Exception:
                        release_all(mapped)
                        raise
                    release_if_supported(gate_real)
                    release_if_supported(gate_imag)
                    gate_real, gate_imag = mapped
                activated_real = eval_silu_chebyshev_cipher(
                    gate_real, coefficients, lower_bound=lower_bound, upper_bound=upper_bound,
                    crypto_context=crypto_context, input_is_chebyshev_mapped=bool(fuse_silu_domain_map),
                )
                activated_imag = eval_silu_chebyshev_cipher(
                    gate_imag, coefficients, lower_bound=lower_bound, upper_bound=upper_bound,
                    crypto_context=crypto_context, input_is_chebyshev_mapped=bool(fuse_silu_domain_map),
                )
                gated_real = _mul_cipher_rescale(activated_real, up_real, crypto_context=crypto_context)
                gated_imag = _mul_cipher_rescale(activated_imag, up_imag, crypto_context=crypto_context)
                imaginary_i = fhe.homo_mul_i(gated_imag, crypto_context.context)
                gated_complex.append(crypto_context.fhe.homo_add(gated_real, imaginary_i, crypto_context.context))
            finally:
                release_all(
                    [c for c in (gate_real, gate_imag, up_real, up_imag, activated_real, activated_imag,
                                 gated_real, gated_imag, imaginary_i) if c is not None]
                )
        release_all(gate_half + up_half)
        gate_half = []
        up_half = []
        _sync()
        stage["silu_gated"] = time.perf_counter() - started
        if len(gated_complex) != len(my_pages) * k * pairs:
            raise RuntimeError(
                f"SiLU produced {len(gated_complex)} ciphers, expected {len(my_pages) * k * pairs}"
            )

        # ------------- 5. down column block of my pages, summed over my pages
        down_partials = [None] * (k * pairs)
        for page_index, page in enumerate(my_pages):
            weight_start = time.perf_counter()
            weight = _down_weight_page(page)
            stage["down_weight_transform_cpu"] = float(
                stage.get("down_weight_transform_cpu", 0.0) + time.perf_counter() - weight_start
            )
            page_start = page_index * k * pairs
            page_inputs = gated_complex[page_start : page_start + k * pairs]  # (member, pair) order
            projected, profile = down_operator.project(
                page_inputs, weight, dtype=np.dtype(np.float64), rescale=False
            )
            _add_profile(stage, "down_projection", profile)
            started = time.perf_counter()
            terms: list[object | None] = list(projected)
            try:
                for index, term in enumerate(terms):
                    terms[index] = None
                    current = down_partials[index]
                    down_partials[index] = term if current is None else _homo_add_consume(current, term)
            finally:
                release_all([t for t in terms if t is not None])
            # a page's babies never hit on the next page
            down_operator.clear_baby_cache()
            release_all(page_inputs)
            for index in range(page_start, page_start + k * pairs):
                gated_complex[index] = None
            _sync()
            stage["down_page_accumulate"] = float(
                stage.get("down_page_accumulate", 0.0) + time.perf_counter() - started
            )
        gated_complex = []

        # ----------------- 6. exchange down partials, sum, rescale once
        started = time.perf_counter()
        payload_for = {}
        for member_index, member in enumerate(group):
            if member == rank:
                continue
            payload_for[member] = [
                (j, pack_cipher(down_partials[member_index * pairs + j], tag=f"{tag}/dn_acc/{rank}/{j}", host=host))
                for j in range(pairs)
            ]
        received = _exchange(transport, f"{tag}/dn_acc", rank=rank, group=group, payload_for=payload_for, comm=comm)
        for index, cipher in enumerate(down_partials):
            if index // pairs != position:
                release_if_supported(cipher)
                down_partials[index] = None
        _sync()
        stage["down_partial_exchange"] = time.perf_counter() - started

        started = time.perf_counter()
        base = position * pairs
        for member in peers:
            for key, packet in received[member]:
                j = int(key)
                foreign = unpack_cipher(packet, device=crypto_context.device)
                current = down_partials[base + j]
                down_partials[base + j] = None
                down_partials[base + j] = _homo_add_consume(current, foreign)
        received = {}
        for j in range(pairs):
            accumulator = down_partials[base + j]
            down_partials[base + j] = None
            if accumulator is None:
                raise RuntimeError("down projection produced an empty accumulator.")
            try:
                down_accumulators[j] = fhe.rescale(accumulator, crypto_context.context)
            finally:
                release_if_supported(accumulator)
        down_partials = []
        _sync()
        stage["down_partial_reduce"] = time.perf_counter() - started

        # ------------------------------------------------ 7. final split
        started = time.perf_counter()
        for carrier in down_accumulators:
            real, imaginary = split_complex_cipher_twice(carrier, crypto_context=crypto_context)
            outputs.extend((real, imaginary))
        extra = outputs[int(layout.cipher_count):]
        outputs = outputs[: int(layout.cipher_count)]
        release_all(extra)
        _sync()
        stage["final_split"] = time.perf_counter() - started

        output = expected = None
        max_abs_diff = mean_abs_diff = None
        if verify:
            started = time.perf_counter()
            decrypted = np.stack(
                [np.asarray(crypto_context.decrypt(cipher), dtype=np.float64) for cipher in outputs]
            )
            output = layout.unpack(decrypted)
            scaled_hidden = hidden_states if column_scale is None else hidden_states * column_scale[None, :]
            gate_reference = scaled_hidden @ gate_weight.T
            up_reference = scaled_hidden @ up_weight.T
            activated_reference = eval_chebyshev_series_direct(
                gate_reference, coefficients, lower_bound, upper_bound
            ).astype(np.float32, copy=False)
            expected = ((activated_reference * up_reference) @ down_weight.T).astype(np.float32, copy=False)
            difference = output - expected
            max_abs_diff = float(np.max(np.abs(difference)))
            mean_abs_diff = float(np.mean(np.abs(difference)))
            _sync()
            stage["verify_decrypt"] = time.perf_counter() - started

        success = True
        return MLPFeatureShardResult(
            ciphers=tuple(outputs),
            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={key: float(value) for key, value in stage.items()},
            output_levels=tuple(sorted({int(crypto_context.level_for_cipher(c)) for c in outputs})),
            comm_bytes=dict(comm),
            feature_shard_k=k,
            group=group,
            pages=my_pages,
            output=output,
            expected=expected,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
        )
    finally:
        gate_up_operator.close()
        down_operator.close()
        release_all(list(paired) + list(conjugated) + foreign_inputs)
        release_all([c for c in projected_page if c is not None])
        release_all(gate_half + up_half)
        release_all([c for c in gated_complex if c is not None])
        release_all([c for c in down_partials if c is not None])
        release_all([c for c in down_accumulators if c is not None])
        if not success:
            release_all(outputs)


__all__ = [
    "MLPFeatureShardResult",
    "feature_shard_group",
    "feature_shard_pages",
    "mlp_feature_shard_fhe",
]
