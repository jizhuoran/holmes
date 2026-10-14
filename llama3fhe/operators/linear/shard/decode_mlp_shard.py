"""Page- and chunk-split SwiGLU MLP for the multi-card *decode* step (T2-Decode).

The decode hidden state is one feature-major ciphertext that every rank holds
(the station chain is redundant up to here), so unlike the prefill
:mod:`.mlp_feature_shard` nothing has to be exchanged *into* the MLP.  The
work is split along the two axes :class:`~llama3fhe.parallel.decode.decode_ownership.DecodeRankOwnership`
hands out:

* **intermediate pages** (4 x 4096 features with the production layout):
  with ``world_size <= pages`` a rank owns whole pages (gate/up of the page,
  SiLU-gate, down column block) and only the low-limb, pre-rescale down
  partials have to be summed over the ranks (one world ``all_reduce``, the
  caller-visible ``{tag}/down`` round);
* **BSGS diagonal chunks** inside a page when ``world_size > pages``: the
  ranks sharing a page (``ownership.mlp_partners``) split the *down* chunks
  of the page between them (free: the down partials go into the world sum
  anyway) and, with ``split_gate_up=True``, also the gate/up chunks -- then
  the pre-rescale gate/up partials are summed among the partners first
  (``group_all_reduce``, one ``all_to_all`` round per page, the
  ``{tag}/gu/<page>`` rounds) and every partner rescales the sum and runs the
  SiLU of the page redundantly.  With ``split_gate_up=False`` the partners
  evaluate the whole gate/up page redundantly and skip that round (A/B knob:
  one collective less against twice the gate/up MAC).

Every rank of the world must call this the same number of times with the same
``tag`` (the collectives match on it).  All ranks end with the same output
ciphertext (CKKS addition is exact, the sum is taken in rank order everywhere).
``world_size == 1`` is exactly :func:`~.mlp.mlp_fhe`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_all, release_if_supported
from llama3fhe.backend import synchronize_device as sync_device
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


@dataclass
class DecodeMLPShardResult:
    """The (identical on every rank) MLP output plus timing / accounting."""

    ciphers: tuple[object, ...]
    complexity: FeatureMajorMLPComplexity
    wall_seconds: float
    stage_seconds: dict[str, float]
    output_levels: tuple[int, ...]
    #: ``{round}_sent/_received/_seconds/_rounds`` per collective round
    comm: dict[str, Any]
    #: intermediate pages this rank evaluated
    pages: tuple[int, ...]
    #: ranks (incl. self) that shared those pages
    partners: tuple[int, ...]
    #: BSGS chunks of the page this rank MAC'd (``None`` = every chunk)
    gate_up_chunks: tuple[int, ...] | None
    down_chunks: tuple[int, ...] | None
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


def decode_mlp_shard_fhe(
    hidden_states: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    silu_entry: dict,
    layout: FeatureMajorPrefillLayout,
    ownership,
    transport,
    config: Llama3CKKSConfig | None = None,
    operator_config: LinearOperatorConfig | None = None,
    crypto_context=None,
    input_ciphers: tuple[object, ...] | list[object] | None = None,
    input_column_scale: np.ndarray | None = None,
    fuse_silu_domain_map: bool = False,
    verify: bool = False,
    tag: str = "decode_mlp",
    split_gate_up: bool = True,
) -> DecodeMLPShardResult:
    """Run the SwiGLU MLP of one decode token with the pages / chunks of
    ``ownership`` (a :class:`~llama3fhe.parallel.decode.decode_ownership.DecodeRankOwnership`)
    on this rank; see the module docstring.

    ``input_ciphers`` are the (identical on every rank) feature-major
    ciphertexts of ``layout``; they are borrowed, not consumed.
    """

    if crypto_context is None or input_ciphers is None:
        raise ValueError("decode_mlp_shard_fhe requires a context and encrypted inputs")
    if transport is None:
        raise ValueError("decode_mlp_shard_fhe requires a transport")
    world_size = int(transport.world_size)
    rank = int(transport.rank)
    if int(ownership.rank) != rank or int(ownership.world_size) != world_size:
        raise ValueError(
            f"ownership (rank {ownership.rank} of {ownership.world_size}) does not match the transport "
            f"(rank {rank} of {world_size})"
        )

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
    if int(ownership.mlp_pages) != pages:
        raise ValueError(
            f"ownership was built for {ownership.mlp_pages} intermediate pages, the weights have {pages}"
        )
    my_pages = tuple(int(page) for page in ownership.mlp_page_indices)
    partners = tuple(int(peer) for peer in ownership.mlp_partners)
    chunk_split = int(ownership.mlp_chunk_split)
    gate_up_split = bool(split_gate_up) and chunk_split > 1
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
        lane_shift=int(operator_config.lane_shift),
    )
    gate_up_operator = LinearOperator(
        crypto_context=crypto_context, token_lanes=int(layout.tokens_per_cipher), config=linear_config
    )
    down_operator = LinearOperator(
        crypto_context=crypto_context, token_lanes=int(layout.tokens_per_cipher), config=linear_config
    )
    gate_up_chunk_count = int(gate_up_operator.plan(page_width).plaintext_chunks)
    down_chunk_count = int(down_operator.plan(page_width).plaintext_chunks)
    gate_up_chunks = (
        tuple(ownership.mlp_chunk_indices(gate_up_chunk_count)) if gate_up_split else None
    )
    down_chunks = tuple(ownership.mlp_chunk_indices(down_chunk_count)) if chunk_split > 1 else None
    conjugation = int(crypto_context.context.M) - 1
    use_structured_weight = structured_weight_packing_enabled()

    from llama3fhe.parallel.decode.decode_collectives import all_reduce_ciphers, group_all_reduce_ciphers

    wall_start = time.perf_counter()
    stage: dict[str, float] = {}
    comm: dict[str, Any] = {}
    paired: tuple[object, ...] = ()
    conjugated: tuple[object, ...] = ()
    projected_page: list[object | None] = []
    gate_half: list[object] = []  # (my page, pair) order
    up_half: list[object] = []
    gated_complex: list[object | None] = []  # same order
    down_partials: list[object | None] = [None] * pairs
    down_accumulators: list[object | None] = [None] * pairs
    outputs: list[object] = []
    success = False

    def _sync() -> None:
        sync_device(crypto_context.device)

    def _homo_add_consume(current, term):
        combined = crypto_context.fhe.homo_add(current, term, crypto_context.context)
        release_if_supported(current)
        release_if_supported(term)
        return combined

    def _gate_up_weight_page(page: int, *, recovery_scale: float = 0.25):
        # ``recovery_scale``: 0.25 for the two-sided (x, conj x) kernel whose
        # two level-free recoveries each double the value; 0.5 for the
        # one-sided real-input kernel (one ``2*Re / 2*Im`` split).
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
                real_scale=recovery_scale * gate_scale,
                imag_source=np.where(active_rows, 1, -1),
                imag_row=source_rows,
                imag_scale=recovery_scale,
                column_scale=(1.0 if column_scale is None else column_scale),
            )
        weight = np.zeros((page_width, page_width), dtype=np.complex128)
        gate_page = gate_weight[feature_start : feature_start + active].astype(np.complex128)
        up_page = up_weight[feature_start : feature_start + active].astype(np.complex128)
        if column_scale is not None:
            gate_page *= column_scale[None, :]
            up_page *= column_scale[None, :]
        weight[:active] = recovery_scale * (gate_scale * gate_page + 1j * up_page)
        return weight

    def _down_weight_page(page: int, *, recovery_scale: float = 0.5):
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
                real_scale=recovery_scale,
                column=columns,
            )
        weight = np.zeros((page_width, page_width), dtype=np.float64)
        weight[:, :active] = recovery_scale * down_weight[:, feature_start : feature_start + active]
        return weight

    def _silu_gate_times_up(gate_cipher, up_cipher):
        """``silu(gate) * up`` of one real pair (both borrowed); owned output."""

        mapped = activated = None
        try:
            gate_input = gate_cipher
            if bool(fuse_silu_domain_map):
                beta = fhe.encode_scalar(
                    gate_beta,
                    cur_limbs=int(gate_cipher.state.cur_limbs),
                    scale_degree=1,
                    scaling_factor=gate_cipher.state.scaling_factor,
                    context=crypto_context.context,
                )
                mapped = fhe.homo_add_scalar(gate_cipher, beta, crypto_context.context)
                gate_input = mapped
            activated = eval_silu_chebyshev_cipher(
                gate_input, coefficients, lower_bound=lower_bound, upper_bound=upper_bound,
                crypto_context=crypto_context, input_is_chebyshev_mapped=bool(fuse_silu_domain_map),
            )
            return _mul_cipher_rescale(activated, up_cipher, crypto_context=crypto_context)
        finally:
            release_if_supported(mapped)
            release_if_supported(activated)

    if gate_up_operator.plan(page_width).shifted_lanes:
        return _decode_mlp_shard_shifted_lanes(
            gate_up_operator=gate_up_operator,
            down_operator=down_operator,
            input_ciphers=input_ciphers,
            layout=layout,
            mlp_layout=mlp_layout,
            my_pages=my_pages,
            partners=partners,
            gate_up_chunks=gate_up_chunks,
            down_chunks=down_chunks,
            gate_up_split=gate_up_split,
            gate_up_weight_page=_gate_up_weight_page,
            down_weight_page=_down_weight_page,
            silu_gate_times_up=_silu_gate_times_up,
            transport=transport,
            crypto_context=crypto_context,
            tag=tag,
            complexity=complexity,
            verify=bool(verify),
            verify_reference=(
                None
                if not verify
                else lambda: _mlp_reference(
                    hidden_states, gate_weight, up_weight, down_weight, column_scale=column_scale,
                    coefficients=coefficients, lower_bound=lower_bound, upper_bound=upper_bound,
                )
            ),
        )

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
        inputs = tuple(paired) + tuple(conjugated)  # 2 * pairs

        # ----------------------------- 2. gate/up of my pages (my chunks)
        profile = gate_up_operator.prepare_baby_rotations(inputs, dimension=page_width)
        _add_profile(stage, "gate_up_projection", profile)
        for page in my_pages:
            weight_start = time.perf_counter()
            weight = _gate_up_weight_page(page)
            stage["gate_up_weight_transform_cpu"] = float(
                stage.get("gate_up_weight_transform_cpu", 0.0) + time.perf_counter() - weight_start
            )
            projected, profile = gate_up_operator.project(
                inputs,
                weight,
                dtype=np.dtype(np.complex128),
                chunk_indices=gate_up_chunks,
                rescale=not gate_up_split,
            )
            _add_profile(stage, "gate_up_projection", profile)
            projected_page = list(projected)
            if len(projected_page) != 2 * pairs:
                raise RuntimeError(f"gate/up page {page} produced {len(projected_page)} ciphers, expected {2 * pairs}")
            if gate_up_split:
                # sum the pre-rescale chunk partials of the page partners, rescale once
                started = time.perf_counter()
                # one world-wide round: the tag must not depend on the page (rank)
                summed = group_all_reduce_ciphers(
                    transport,
                    f"{tag}/gu",
                    projected_page,
                    group=partners,
                    crypto_context=crypto_context,
                    comm=comm,
                )
                projected_page = []
                rescaled: list[object] = []
                try:
                    for cipher in summed:
                        rescaled.append(fhe.rescale(cipher, crypto_context.context))
                finally:
                    release_all(summed)
                projected_page = rescaled
                _sync()
                stage["gate_up_partner_reduce"] = float(
                    stage.get("gate_up_partner_reduce", 0.0) + time.perf_counter() - started
                )
            started = time.perf_counter()
            for j in range(pairs):
                direct = projected_page[j]
                conjugate_evaluation = projected_page[pairs + j]
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

        # -------------------------------------------------------- 3. SiLU
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
        if len(gated_complex) != len(my_pages) * pairs:
            raise RuntimeError(f"SiLU produced {len(gated_complex)} ciphers, expected {len(my_pages) * pairs}")

        # ------------- 4. down column block of my pages / my chunks, summed
        for page_index, page in enumerate(my_pages):
            weight_start = time.perf_counter()
            weight = _down_weight_page(page)
            stage["down_weight_transform_cpu"] = float(
                stage.get("down_weight_transform_cpu", 0.0) + time.perf_counter() - weight_start
            )
            page_start = page_index * pairs
            page_inputs = [c for c in gated_complex[page_start : page_start + pairs] if c is not None]
            projected, profile = down_operator.project(
                page_inputs, weight, dtype=np.dtype(np.float64), chunk_indices=down_chunks, rescale=False
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
            down_operator.clear_baby_cache()
            release_all(page_inputs)
            for index in range(page_start, page_start + pairs):
                gated_complex[index] = None
            _sync()
            stage["down_page_accumulate"] = float(
                stage.get("down_page_accumulate", 0.0) + time.perf_counter() - started
            )
        gated_complex = []

        # ----------------- 5. world sum of the down partials, rescale once
        started = time.perf_counter()
        if any(c is None for c in down_partials):
            raise RuntimeError("down projection produced an empty accumulator.")
        summed = all_reduce_ciphers(
            transport, f"{tag}/down", down_partials, crypto_context=crypto_context, comm=comm
        )
        down_partials = []
        _sync()
        stage["down_world_reduce"] = time.perf_counter() - started
        started = time.perf_counter()
        try:
            for j, accumulator in enumerate(summed):
                down_accumulators[j] = fhe.rescale(accumulator, crypto_context.context)
        finally:
            release_all(summed)
        _sync()
        stage["down_rescale"] = time.perf_counter() - started

        # ------------------------------------------------ 6. final split
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
        return DecodeMLPShardResult(
            ciphers=tuple(outputs),
            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={key: float(value) for key, value in stage.items()},
            output_levels=tuple(sorted({int(crypto_context.level_for_cipher(c)) for c in outputs})),
            comm=dict(comm),
            pages=my_pages,
            partners=partners,
            gate_up_chunks=gate_up_chunks,
            down_chunks=down_chunks,
            output=output,
            expected=expected,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
        )
    finally:
        gate_up_operator.close()
        down_operator.close()
        release_all(list(paired) + list(conjugated))
        release_all([c for c in projected_page if c is not None])
        release_all(gate_half + up_half)
        release_all([c for c in gated_complex if c is not None])
        release_all([c for c in down_partials if c is not None])
        release_all([c for c in down_accumulators if c is not None])
        if not success:
            release_all(outputs)


def _mlp_reference(
    hidden_states: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    column_scale: np.ndarray | None,
    coefficients: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    scaled_hidden = hidden_states if column_scale is None else hidden_states * column_scale[None, :]
    gate_reference = scaled_hidden @ gate_weight.T
    up_reference = scaled_hidden @ up_weight.T
    activated_reference = eval_chebyshev_series_direct(
        gate_reference, coefficients, lower_bound, upper_bound
    ).astype(np.float32, copy=False)
    return ((activated_reference * up_reference) @ down_weight.T).astype(np.float32, copy=False)


def _decode_mlp_shard_shifted_lanes(
    *,
    gate_up_operator: LinearOperator,
    down_operator: LinearOperator,
    input_ciphers: tuple[object, ...],
    layout: FeatureMajorPrefillLayout,
    mlp_layout: FeatureMajorMLPLayout,
    my_pages: tuple[int, ...],
    partners: tuple[int, ...],
    gate_up_chunks: tuple[int, ...] | None,
    down_chunks: tuple[int, ...] | None,
    gate_up_split: bool,
    gate_up_weight_page,
    down_weight_page,
    silu_gate_times_up,
    transport,
    crypto_context,
    tag: str,
    complexity: FeatureMajorMLPComplexity,
    verify: bool,
    verify_reference,
) -> DecodeMLPShardResult:
    """The shifted-lane (SL) body of :func:`decode_mlp_shard_fhe`.

    The DecodeFM input is real and lives in token lane 0, so the one-sided
    kernel is used throughout: no ``conj(x)`` companion, no padded imaginary
    stream, one SiLU per page.  Per page::

        x_sl = expand(x)                       # 7 hoisted rotations, no level
        c    = fold(sum_ranks MAC(x_sl, 0.5*(gate + i*up)))   # +1 level (MAC)
        c    = mask_lane0(c)                   # +1 level (lanes 1.. are garbage)
        g, u = 2*Re(c), 2*Im(c)                # conjugation, no level
        h    = silu(g) * u                     # SiLU depth + 1
        acc += MAC(expand(h), down_page)       # +1 level (rescaled after the sum)
        out  = mask_lane0(fold(sum_ranks acc)) # +1 level

    so the whole MLP costs two more levels than the lane-replicated path.
    Lane 0 of ``out`` is the real MLP output; lanes 1.. are exactly zero
    (up to CKKS noise), which keeps the residual stream lane-0 clean.
    """

    from llama3fhe.parallel.decode.decode_collectives import all_reduce_ciphers, group_all_reduce_ciphers

    from ..shifted_lane import expand_lane0_to_shifted_lanes, fold_shifted_lanes, mask_lane0

    page_width = int(mlp_layout.page_width)
    if int(layout.cipher_count) != 1 or len(input_ciphers) != 1:
        raise ValueError("the shifted-lane decode MLP needs the one-ciphertext DecodeFM stream.")
    plan = gate_up_operator.plan(page_width)
    if not plan.shifted_lanes or int(plan.token_lanes) != int(layout.tokens_per_cipher):
        raise ValueError("shifted-lane MLP: the operator plan does not carry the lane shift.")
    mask_input = os.environ.get("LLAMA_FHE_SL_MASK_INPUT", "0") == "1"

    wall_start = time.perf_counter()
    stage: dict[str, float] = {}
    comm: dict[str, Any] = {}
    sl_input = None
    projected_page: list[object] = []
    folded = masked = gate = up = None
    gated: list[object | None] = []
    gated_sl = None
    down_partial = None
    summed: list[object] = []
    outputs: list[object] = []
    success = False

    def _sync() -> None:
        sync_device(crypto_context.device)

    def _homo_add_consume(current, term):
        combined = crypto_context.fhe.homo_add(current, term, crypto_context.context)
        release_if_supported(current)
        release_if_supported(term)
        return combined

    def _fold_and_mask(cipher, *, stage_name: str):
        started = time.perf_counter()
        lane_sum = fold_shifted_lanes(cipher, plan=plan, crypto_context=crypto_context)
        try:
            clean = mask_lane0(lane_sum, plan=plan, crypto_context=crypto_context)
        finally:
            release_if_supported(lane_sum)
        _sync()
        stage[stage_name] = float(stage.get(stage_name, 0.0) + time.perf_counter() - started)
        return clean

    try:
        # ------------------------------------------------- 1. lane expansion
        started = time.perf_counter()
        sl_input = expand_lane0_to_shifted_lanes(
            input_ciphers[0], plan=plan, crypto_context=crypto_context, mask=mask_input
        )
        _sync()
        stage["input_lane_expand"] = time.perf_counter() - started

        # ----------------------------- 2. gate/up of my pages (my chunks)
        profile = gate_up_operator.prepare_baby_rotations((sl_input,), dimension=page_width)
        _add_profile(stage, "gate_up_projection", profile)
        for page in my_pages:
            weight_start = time.perf_counter()
            weight = gate_up_weight_page(page, recovery_scale=0.5)
            stage["gate_up_weight_transform_cpu"] = float(
                stage.get("gate_up_weight_transform_cpu", 0.0) + time.perf_counter() - weight_start
            )
            projected, profile = gate_up_operator.project(
                (sl_input,),
                weight,
                dtype=np.dtype(np.complex128),
                chunk_indices=gate_up_chunks,
                rescale=not gate_up_split,
            )
            _add_profile(stage, "gate_up_projection", profile)
            projected_page = list(projected)
            if len(projected_page) != 1:
                raise RuntimeError(f"gate/up page {page} produced {len(projected_page)} ciphers, expected 1")
            if gate_up_split:
                started = time.perf_counter()
                reduced = group_all_reduce_ciphers(
                    transport, f"{tag}/gu", projected_page, group=partners, crypto_context=crypto_context, comm=comm
                )
                projected_page = []
                try:
                    projected_page = [fhe.rescale(reduced[0], crypto_context.context)]
                finally:
                    release_all(reduced)
                _sync()
                stage["gate_up_partner_reduce"] = float(
                    stage.get("gate_up_partner_reduce", 0.0) + time.perf_counter() - started
                )
            masked = _fold_and_mask(projected_page[0], stage_name="gate_up_fold_mask")
            release_all(projected_page)
            projected_page = []
            # ----------------------------------------------------- 3. SiLU
            started = time.perf_counter()
            gate, up = split_complex_cipher_twice(masked, crypto_context=crypto_context)
            release_if_supported(masked)
            masked = None
            gated.append(silu_gate_times_up(gate, up))
            release_if_supported(gate)
            release_if_supported(up)
            gate = up = None
            _sync()
            stage["silu_gated"] = float(stage.get("silu_gated", 0.0) + time.perf_counter() - started)
        gate_up_operator.clear_baby_cache()
        release_if_supported(sl_input)
        sl_input = None

        # ------------- 4. down column block of my pages / my chunks, summed
        for page_index, page in enumerate(my_pages):
            started = time.perf_counter()
            gated_sl = expand_lane0_to_shifted_lanes(gated[page_index], plan=plan, crypto_context=crypto_context)
            release_if_supported(gated[page_index])
            gated[page_index] = None
            _sync()
            stage["down_lane_expand"] = float(stage.get("down_lane_expand", 0.0) + time.perf_counter() - started)
            weight_start = time.perf_counter()
            weight = down_weight_page(page, recovery_scale=1.0)
            stage["down_weight_transform_cpu"] = float(
                stage.get("down_weight_transform_cpu", 0.0) + time.perf_counter() - weight_start
            )
            projected, profile = down_operator.project(
                (gated_sl,), weight, dtype=np.dtype(np.float64), chunk_indices=down_chunks, rescale=False
            )
            _add_profile(stage, "down_projection", profile)
            down_operator.clear_baby_cache()
            release_if_supported(gated_sl)
            gated_sl = None
            started = time.perf_counter()
            (term,) = projected
            down_partial = term if down_partial is None else _homo_add_consume(down_partial, term)
            _sync()
            stage["down_page_accumulate"] = float(
                stage.get("down_page_accumulate", 0.0) + time.perf_counter() - started
            )
        gated = []

        # ----------------- 5. world sum of the down partials, rescale once
        started = time.perf_counter()
        if down_partial is None:
            raise RuntimeError("down projection produced an empty accumulator.")
        summed = list(
            all_reduce_ciphers(transport, f"{tag}/down", [down_partial], crypto_context=crypto_context, comm=comm)
        )
        down_partial = None
        _sync()
        stage["down_world_reduce"] = time.perf_counter() - started
        started = time.perf_counter()
        rescaled = fhe.rescale(summed[0], crypto_context.context)
        release_all(summed)
        summed = []
        _sync()
        stage["down_rescale"] = time.perf_counter() - started
        try:
            folded = _fold_and_mask(rescaled, stage_name="down_fold_mask")
        finally:
            release_if_supported(rescaled)
        outputs = [folded]
        folded = None

        output = expected = None
        max_abs_diff = mean_abs_diff = None
        if verify:
            started = time.perf_counter()
            decrypted = np.stack([np.real(np.asarray(crypto_context.decrypt(cipher))) for cipher in outputs])
            output = layout.unpack(decrypted.astype(np.float64))
            expected = verify_reference()
            difference = output - expected
            max_abs_diff = float(np.max(np.abs(difference)))
            mean_abs_diff = float(np.mean(np.abs(difference)))
            _sync()
            stage["verify_decrypt"] = time.perf_counter() - started

        success = True
        return DecodeMLPShardResult(
            ciphers=tuple(outputs),
            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={key: float(value) for key, value in stage.items()},
            output_levels=tuple(sorted({int(crypto_context.level_for_cipher(c)) for c in outputs})),
            comm=dict(comm),
            pages=my_pages,
            partners=partners,
            gate_up_chunks=gate_up_chunks,
            down_chunks=down_chunks,
            output=output,
            expected=expected,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
        )
    finally:
        gate_up_operator.close()
        down_operator.close()
        release_if_supported(sl_input)
        release_all(projected_page)
        for cipher in (folded, masked, gate, up, gated_sl, down_partial):
            release_if_supported(cipher)
        release_all([c for c in gated if c is not None])
        release_all(summed)
        if not success:
            release_all(outputs)


__all__ = ["DecodeMLPShardResult", "decode_mlp_shard_fhe"]
