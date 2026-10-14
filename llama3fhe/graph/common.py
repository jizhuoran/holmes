"""Shared pieces of the layer graph: weight/approximation/result dataclasses, residual
audits, the attention-cache rotation helpers, ``required_rotations`` and
``operation_audit`` (moved verbatim out of the former ``llama3fhe/graph.py``)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from ..backend import release_all, synchronize_device
from ..decode_packing import (
    DEFAULT_DECODE_PACKING,
    DecodePacking,
    validate_decode_packing,
)
from ..layouts.attention import AttentionPairLayout
from ..layouts.feature_major import FeatureMajorPrefillLayout
from ..layouts.linear import LinearCarrierLayout
from ..operators.attention.audit import (
    delta_pv_operation_counts,
    delta_pv_rotations,
    delta_qk_operation_counts,
    delta_softmax_alg2_complexity,
    delta_softmax_rotations,
    pipeline_static_audit,
)
from ..operators.attention.config import AttentionOperatorConfig
from ..operators.attention.prefill.operator import attention_to_w_o_rotations
from ..operators.attention.prefill.output import (
    pairfm_output_complexity,
    pairfm_output_rotations,
)
from ..operators.linear import (
    LinearOperatorConfig,
    linear_rotations,
)
from ..operators.linear.prefill.mlp import mlp_complexity, mlp_rotations
from ..operators.linear.prefill.qkv import qkv_rotations
from ..operators.nonlinear.bootstrap import BootstrapOperator

try:
    from ..operators.norm.qk_norm import qk_norm_fhe, qk_norm_rotations
except ImportError:
    def qk_norm_rotations(*, token_lanes: int) -> tuple[int, ...]:
        return ()
    def qk_norm_fhe(*args, **kwargs):
        raise RuntimeError("QK-Norm is not available in this runtime.")

try:
    from ..operators.norm.refresh import refresh_complex_ciphers
except ImportError:
    def refresh_complex_ciphers(*args, **kwargs):
        raise RuntimeError(
            "refresh_complex_ciphers is not available in this runtime."
        )

from ..operators.attention.prefill.input import DeltaKVPage
from ..operators.norm.rmsnorm import rmsnorm_complexity, rmsnorm_rotations
from ..operators.pairfold_kv_cache import PairFoldLayerCache
from ..schedule import LayerSchedule


@dataclass(frozen=True)
class LayerWeights:
    """One layer's plaintext weights, exactly as the checkpoint stores them."""

    input_layernorm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    post_attention_layernorm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray
    q_norm: np.ndarray | None = None
    k_norm: np.ndarray | None = None


@dataclass(frozen=True)
class LayerApproximations:
    """The polynomial approximations selected for one layer."""

    input_norm: dict
    post_attention_norm: dict
    silu: dict
    softmax: object
    qk_norm: dict | None = None


@dataclass
class LayerResult:
    ciphers: tuple[object, ...]
    wall_seconds: float
    stage_seconds: dict[str, float]
    nested_stage_seconds: dict[str, object]
    softmax_bootstrap_calls: int
    residual_bootstrap_calls: int
    softmax_native_bootstrap_calls: int
    residual_native_bootstrap_calls: int
    output_levels: tuple[int, ...]
    output: np.ndarray | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    key_c_ciphers: tuple[object, ...] = ()
    value_c_ciphers: tuple[object, ...] = ()
    pairfold_kv_cache: PairFoldLayerCache | None = None
    delta_kv_page: DeltaKVPage | None = None

    def release(self) -> None:
        release_all(self.ciphers)
        release_all(self.key_c_ciphers)
        release_all(self.value_c_ciphers)
        pairfold_kv_cache = self.pairfold_kv_cache
        delta_kv_page = self.delta_kv_page
        self.ciphers = ()
        self.key_c_ciphers = ()
        self.value_c_ciphers = ()
        self.pairfold_kv_cache = None
        self.delta_kv_page = None
        if pairfold_kv_cache is not None:
            pairfold_kv_cache.release()
        if delta_kv_page is not None:
            delta_kv_page.release()


def _cipher_levels(ciphers, *, crypto_context) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(crypto_context.level_for_cipher(cipher))
                for cipher in tuple(ciphers)
            }
        )
    )


def _cipher_runtime_state(cipher, *, crypto_context) -> dict[str, object]:
    """Return the complete metadata relevant to a ciphertext addition."""

    scaling_factor = cipher.state.scaling_factor
    return {
        "level": int(crypto_context.level_for_cipher(cipher)),
        "cur_limbs": int(cipher.state.cur_limbs),
        "scale_degree": int(cipher.state.scale_degree),
        "scaling_factor": (
            None if scaling_factor is None else float(scaling_factor)
        ),
        "slots": int(cipher.slots),
        "is_ext": bool(cipher.is_ext),
        "batch_size": int(getattr(cipher, "batch_size", 1)),
        "component_count": int(len(cipher.cv)),
    }


def _cipher_stream_state_audit(ciphers, *, crypto_context) -> dict[str, object]:
    """Compact identical full-state rows without losing their multiplicity."""

    groups: list[dict[str, object]] = []
    for cipher in tuple(ciphers):
        state = _cipher_runtime_state(cipher, crypto_context=crypto_context)
        for group in groups:
            comparable = {
                key: value
                for key, value in group.items()
                if key != "count"
            }
            if comparable == state:
                group["count"] = int(group["count"]) + 1
                break
        else:
            groups.append({**state, "count": 1})
    return {
        "cipher_count": int(sum(int(group["count"]) for group in groups)),
        "unique_states": groups,
    }


def _residual_state_audit(
    left_ciphers,
    right_ciphers,
    *,
    crypto_context,
) -> dict[str, object]:
    """Validate structural ownership boundaries and report add alignment.

    Different levels/scales are expected here: ``residual_add``
    explicitly aligns the higher-limb operand to the lower-limb operand.  The
    fields that cannot be repaired by that alignment must already match.
    """

    left_ciphers = tuple(left_ciphers)
    right_ciphers = tuple(right_ciphers)
    if len(left_ciphers) != len(right_ciphers):
        raise ValueError(
            "residual state audit requires equal streams, got "
            f"{len(left_ciphers)} and {len(right_ciphers)}."
        )
    alignment_pairs = 0
    for index, (left, right) in enumerate(
        zip(left_ciphers, right_ciphers, strict=True)
    ):
        for field in ("slots", "is_ext"):
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value != right_value:
                raise ValueError(
                    f"residual pair {index}: {field} mismatch: "
                    f"{left_value} != {right_value}."
                )
        alignment_pairs += int(left.state != right.state)
    return {
        "left": _cipher_stream_state_audit(
            left_ciphers, crypto_context=crypto_context
        ),
        "right": _cipher_stream_state_audit(
            right_ciphers, crypto_context=crypto_context
        ),
        "state_alignment_pairs": int(alignment_pairs),
    }


def _select_residual_identity(
    original_ciphers,
    refreshed_ciphers,
    *,
    preserve_prebootstrap: bool,
    refresh_was_run: bool,
) -> tuple[tuple[object, ...], str]:
    """Borrow, rather than copy, the selected residual identity stream."""

    original_ciphers = tuple(original_ciphers)
    refreshed_ciphers = tuple(refreshed_ciphers)
    if bool(preserve_prebootstrap) or not bool(refresh_was_run):
        return original_ciphers, "prebootstrap"
    if len(refreshed_ciphers) != len(original_ciphers):
        raise ValueError(
            "bootstrap-refreshed residual stream has the wrong size: "
            f"{len(refreshed_ciphers)} != {len(original_ciphers)}."
        )
    return refreshed_ciphers, "bootstrap_refreshed"


def _linear_config(
    schedule: LayerSchedule,
    max_plaintext_rows: int,
    hoist_strategy: str,
) -> LinearOperatorConfig:
    """The linear execution policy for one projection boundary."""

    return LinearOperatorConfig(
        baby_steps=int(schedule.linear_baby_steps),
        baby_anchor_step=int(schedule.linear_anchor_step),
        max_plaintext_rows=int(max_plaintext_rows),
        hoist_strategy=str(hoist_strategy),
    )


def _decode_linear_config(
    schedule: LayerSchedule,
    max_plaintext_rows: int,
    hoist_strategy: str,
    *,
    dimension: int = 4096,
    token_lanes: int = 8,
) -> LinearOperatorConfig:
    """The policy of one *decode* linear boundary.

    ``schedule.decode_shifted_lanes=False`` is :func:`_linear_config`.  With
    the shifted-lane packing on, the block has ``dimension / token_lanes``
    diagonals, the baby count is :attr:`LayerSchedule.decode_sl_baby_steps`
    and the plaintext chunk is ``lane_shift / token_lanes`` rows (64 for the
    production layout) so every 4096-block has exactly ``token_lanes`` (8)
    chunks -- the axis the decode ranks split (``W | 8``).  The key plan and
    the stations must both go through here.
    """

    if not bool(getattr(schedule, "decode_shifted_lanes", False)):
        return _linear_config(schedule, max_plaintext_rows, hoist_strategy)
    dimension = int(dimension)
    token_lanes = int(token_lanes)
    if dimension % token_lanes:
        raise ValueError(f"shifted lanes need token_lanes | dimension, got {token_lanes} / {dimension}.")
    lane_shift = dimension // token_lanes
    rows = max(int(schedule.decode_sl_baby_steps), lane_shift // token_lanes)
    return LinearOperatorConfig(
        baby_steps=int(schedule.decode_sl_baby_steps),
        baby_anchor_step=int(schedule.linear_anchor_step),
        max_plaintext_rows=int(rows),
        hoist_strategy=str(hoist_strategy),
        lane_shift=int(lane_shift),
    )


def _attention_exclusive_cache_keep_rotations(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    schedule: LayerSchedule,
    operator_config: AttentionOperatorConfig,
    log_n: int,
    prefix_k: int = 0,
    emit_kv_cache: bool = False,
    decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
) -> dict[str, tuple[int, ...]]:
    """Return exact downstream keep-sets at the QK and PV boundaries."""

    output_linear = _linear_config(
        schedule, schedule.output_max_plaintext_rows, schedule.output_hoist_strategy
    )
    mlp_linear = _linear_config(
        schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy
    )
    post_pv = set(
        pairfm_output_rotations(
            feature_layout,
            attention_layout,
            first_baby_steps=int(operator_config.pairfm_first_baby_steps),
            block_baby_steps=int(operator_config.pairfm_block_baby_steps),
            log_n=int(log_n),
        )
    )
    post_pv.update(
        linear_rotations(
            dimension=int(feature_layout.hidden_dim),
            slots=int(feature_layout.slots),
            token_lanes=int(feature_layout.tokens_per_cipher),
            include_conjugation=True,
            operator_config=output_linear,
        )
    )
    post_pv.update(rmsnorm_rotations(feature_layout))
    post_pv.update(mlp_rotations(feature_layout, operator_config=mlp_linear))
    post_pv.update(
        BootstrapOperator.required_rotations(
            log_n=int(log_n),
            log_slots=int(schedule.bootstrap_log_slots),
            level_budget=tuple(int(v) for v in schedule.bootstrap_level_budget),
            deep_level_budget=(
                (3, 3) if operator_config.softmax_main_deep_bootstrap else None
            ),
            sparse_log_slots=(
                tuple(int(k) for k in schedule.bootstrap_sparse_log_slots) or None
            ),
        )
    )

    if bool(emit_kv_cache):
        # Legacy emitted K_C uses the conjugation key.  PairFold128 adds its
        # half-page permutation; retain both until the post-W_O conversion.
        post_pv.add(4 * int(feature_layout.slots) - 1)
        if validate_decode_packing(decode_packing) == "pairfold128":
            from ..layouts.pairfold128 import PairFold128Layout
            from ..operators.attention.pairfold.pairfold_prefill import (
                prefill_c_to_pairfold_rotations,
            )

            post_pv.update(
                prefill_c_to_pairfold_rotations(PairFold128Layout())
            )

    qk_keep = set(post_pv)
    qk_keep.update(delta_softmax_rotations(attention_layout))
    qk_keep.update(
        delta_pv_rotations(
            attention_layout,
            complex_delta_folding=bool(
                operator_config.pv_complex_delta_folding
            ),
            # Even when the PV kernel consumes pre-folded inputs, the fold
            # helpers run after this QK boundary and still need their three
            # half-shift rotations.  The non-pre-folded stage union contains
            # exactly those helpers plus the folded kernel's rotations.
            probability_pre_folded=False,
            value_pre_folded=False,
            log_n=int(log_n),
        )
    )
    if int(prefix_k):
        from ..operators.attention.prefix.prefix_pc import prefix_pc_pv_rotations
        from ..operators.attention.prefix.prefix_softmax import prefix_softmax_rotations

        qk_keep.update(
            prefix_softmax_rotations(attention_layout, prefix_k=int(prefix_k))
        )
        qk_keep.update(
            prefix_pc_pv_rotations(attention_layout, prefix_k=int(prefix_k))
        )

    return {
        "qk_to_softmax": tuple(sorted(int(v) for v in qk_keep if int(v))),
        "pv_to_pairfm": tuple(sorted(int(v) for v in post_pv if int(v))),
    }


def _evict_attention_exclusive_cuda_cache(
    crypto_context, *, keep_rotations: tuple[int, ...]
) -> float:
    """Synchronize, prune all non-downstream CUDA key material, and trim."""

    clear = getattr(crypto_context.context, "clear_cuda_rotation_cache", None)
    if not callable(clear):
        raise RuntimeError(
            "EasyFHE Context.clear_cuda_rotation_cache is required for "
            "attention exclusive-cache eviction"
        )
    start = time.perf_counter()
    synchronize_device(crypto_context.device)
    clear(
        keep_rotations=tuple(int(v) for v in keep_rotations),
        empty_allocator_cache=True,
    )
    return float(time.perf_counter() - start)


def _prefill_delta_pairfold_v_enabled(
    emit_kv_cache: bool, decode_packing: DecodePacking
) -> bool:
    return (
        bool(emit_kv_cache)
        and decode_packing == "pairfold128"
        and os.environ.get("LLAMA_FHE_PREFILL_DELTA_PAIRFOLD_V", "")
        .strip()
        .lower()
        not in {"", "0", "false", "off", "no"}
    )


def _pairfold_prefill_cache_from_raw_c(
    folded_key_ciphers: tuple[object, ...] | list[object],
    real_value_ciphers: tuple[object, ...] | list[object],
    *,
    prepacked_value_ciphers: tuple[object, ...] | list[object] = (),
    attention_layout: AttentionPairLayout,
    crypto_context,
    target_limbs: int | None,
    bootstrap_operator,
    bootstrap_iterations: int,
    occupied: int,
    emitted_key_c: bool = False,
    profile: dict[str, object] | None = None,
    value_target_limbs: int | None = None,
) -> PairFoldLayerCache:
    """Consume raw Prefill C streams and return one owned PairFold page.

    The converter masks the suffix of a partially occupied page before it
    becomes a Decode cache.  ``profile`` is populated only on success.
    """

    from ..layouts.pairfold128 import PairFold128Layout
    from ..operators.attention.pairfold.pairfold_prefill import (
        prefill_c_to_pairfold_fhe,
        prefill_emitted_key_c_direct_to_pairfold_fhe,
        prefill_emitted_key_c_to_pairfold_fhe,
        prefill_real_value_c_to_pairfold_fhe,
    )
    from ..operators.pairfold_kv_cache import (
        PairFoldCipherPage,
        PairFoldTokenWrite,
        drop_pairfold_token_to_limbs,
    )

    key_inputs = tuple(folded_key_ciphers)
    value_inputs = tuple(real_value_ciphers)
    prepacked_values = tuple(prepacked_value_ciphers)
    pairfold_layout = PairFold128Layout()
    occupied = int(occupied)
    direct_key = bool(emitted_key_c) and os.environ.get(
        "LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", ""
    ).strip().lower() not in {"", "0", "false", "off", "no"}
    key_transferred = False
    value_transferred = False
    prepacked_transferred = False
    pairfold_keys: tuple[object, ...] = ()
    pairfold_values: tuple[object, ...] = ()
    token = None
    page = None
    cache = None
    try:
        if value_target_limbs is not None and target_limbs is None:
            raise ValueError("PairFold V limb target requires a K limb target.")
        if not 1 <= occupied <= int(pairfold_layout.seq_len):
            raise ValueError(
                f"prefill_occupied must be in [1, {pairfold_layout.seq_len}], "
                f"got {occupied}."
            )
        if prepacked_values and (value_inputs or len(prepacked_values) != pairfold_layout.key_value_groups):
            raise ValueError("prepacked PairFold V replaces exactly four raw V_C rows.")

        if profile is not None:
            raw_key_limbs = tuple(
                int(cipher.state.cur_limbs) for cipher in key_inputs
            )
            raw_value_limbs = tuple(
                int(cipher.state.cur_limbs)
                for cipher in (prepacked_values or value_inputs)
            )
            synchronize_device(crypto_context.device)
            convert_start = time.perf_counter()
        if prepacked_values:
            key_transferred = True
            converter = (
                prefill_emitted_key_c_direct_to_pairfold_fhe
                if direct_key
                else prefill_emitted_key_c_to_pairfold_fhe
            )
            pairfold_keys = converter(
                key_inputs,
                attention_layout=attention_layout,
                pairfold_layout=pairfold_layout,
                crypto_context=crypto_context,
                occupied=occupied,
            )
            pairfold_values = prepacked_values
            prepacked_transferred = True
        elif direct_key:
            # Each converter owns its own raw stream from the call boundary.
            key_transferred = True
            pairfold_keys = prefill_emitted_key_c_direct_to_pairfold_fhe(
                key_inputs,
                attention_layout=attention_layout,
                pairfold_layout=pairfold_layout,
                crypto_context=crypto_context,
                occupied=occupied,
            )
            value_transferred = True
            pairfold_values = prefill_real_value_c_to_pairfold_fhe(
                value_inputs,
                attention_layout=attention_layout,
                pairfold_layout=pairfold_layout,
                crypto_context=crypto_context,
                occupied=occupied,
            )
        else:
            # The combined converter owns both streams even if it raises.
            key_transferred = value_transferred = True
            pairfold_keys, pairfold_values = prefill_c_to_pairfold_fhe(
                key_inputs,
                value_inputs,
                attention_layout=attention_layout,
                pairfold_layout=pairfold_layout,
                crypto_context=crypto_context,
                emitted_key_c=bool(emitted_key_c),
                occupied=occupied,
            )
        if profile is not None:
            synchronize_device(crypto_context.device)
            convert_seconds = time.perf_counter() - convert_start
            converted_key_limbs = tuple(
                int(cipher.state.cur_limbs) for cipher in pairfold_keys
            )
            converted_value_limbs = tuple(
                int(cipher.state.cur_limbs) for cipher in pairfold_values
            )
            value_limb_target = (
                target_limbs if value_target_limbs is None else value_target_limbs
            )
            refresh_bootstrap_calls = (
                0
                if target_limbs is None
                else (
                    sum(
                        int(cipher.state.cur_limbs) < int(target_limbs)
                        for cipher in pairfold_keys
                    )
                    + sum(
                        int(cipher.state.cur_limbs) < int(value_limb_target)
                        for cipher in pairfold_values
                    )
                )
            )
        token = PairFoldTokenWrite(
            key_groups=pairfold_keys,
            value_groups=pairfold_values,
            column=0,
        )
        pairfold_keys = ()
        pairfold_values = ()
        if profile is not None:
            refresh_start = time.perf_counter()
        if target_limbs is not None:
            token = drop_pairfold_token_to_limbs(
                token,
                target_limbs=int(target_limbs),
                crypto_context=crypto_context,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=int(bootstrap_iterations),
                value_target_limbs=value_target_limbs,
            )
        if profile is not None:
            synchronize_device(crypto_context.device)
            refresh_drop_seconds = time.perf_counter() - refresh_start
            page_start = time.perf_counter()
        page = PairFoldCipherPage.from_token_write(
            token,
            occupied=occupied,
        )
        token = None
        cache = PairFoldLayerCache(
            pages=(page,),
            content_tokens=occupied,
        )
        page = None
        if profile is not None:
            synchronize_device(crypto_context.device)
            profile.update(
                {
                    "direct_key": bool(direct_key),
                    "prepacked_value": bool(prepacked_values),
                    "convert_seconds": float(convert_seconds),
                    "refresh_drop_seconds": float(refresh_drop_seconds),
                    "page_seconds": float(time.perf_counter() - page_start),
                    "raw_key_limbs": raw_key_limbs,
                    "raw_value_limbs": raw_value_limbs,
                    "converted_key_limbs": converted_key_limbs,
                    "converted_value_limbs": converted_value_limbs,
                    "cache_key_limbs": tuple(
                        int(cipher.state.cur_limbs)
                        for cipher in cache.pages[0].key_groups
                    ),
                    "cache_value_limbs": tuple(
                        int(cipher.state.cur_limbs)
                        for cipher in cache.pages[0].value_groups
                    ),
                    "refresh_bootstrap_calls": int(refresh_bootstrap_calls),
                }
            )
        result = cache
        cache = None
        return result
    finally:
        if not key_transferred:
            release_all(key_inputs)
        if not value_transferred:
            release_all(value_inputs)
        if not prepacked_transferred:
            release_all(prepacked_values)
        release_all(pairfold_keys)
        release_all(pairfold_values)
        if token is not None:
            token.release()
        if page is not None:
            page.release()
        if cache is not None:
            cache.release()


def required_rotations(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    schedule: LayerSchedule | None = None,
    log_n: int = 16,
    bootstrap_deep_level_budget: tuple[int, int] | None = None,
    emit_kv_cache: bool = False,
    prefix_k: int = 0,
    decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
) -> tuple[int, ...]:
    """Exact rotation-key union for the complete feature-major Layer0.

    ``bootstrap_deep_level_budget`` stays a separate parameter: key
    planning must include the deep budget's CtoS/StoC union when the deep
    main bootstrap is enabled, while a layer's own rotation accounting
    never does.
    """

    decode_packing = validate_decode_packing(decode_packing)
    schedule = LayerSchedule() if schedule is None else schedule
    # The stages below read the schedule's decisions as plain locals;
    # binding them here keeps the object at the boundary only.
    qkv_hoist_strategy = schedule.qkv_hoist_strategy
    output_hoist_strategy = schedule.output_hoist_strategy
    mlp_hoist_strategy = schedule.mlp_hoist_strategy
    qkv_max_plaintext_rows = schedule.qkv_max_plaintext_rows
    output_max_plaintext_rows = schedule.output_max_plaintext_rows
    mlp_max_plaintext_rows = schedule.mlp_max_plaintext_rows
    sparse_max_baby_offsets = schedule.sparse_max_baby_offsets
    shear_baby_steps = schedule.shear_baby_steps
    bootstrap_log_slots = schedule.bootstrap_log_slots
    bootstrap_level_budget = schedule.bootstrap_level_budget
    pv_complex_delta_folding = schedule.pv_complex_delta_folding
    pv_prefold_after_softmax = schedule.pv_prefold_after_softmax

    # One config per linear boundary, assembled from the schedule exactly
    # once. The key planner and the kernel must read the same object: a
    # partially rebuilt config plans a different key union and the missing
    # key only surfaces as a crash deep inside a rotation.
    qkv_linear = _linear_config(
        schedule, qkv_max_plaintext_rows, qkv_hoist_strategy
    )
    output_linear = _linear_config(
        schedule, output_max_plaintext_rows, output_hoist_strategy
    )
    mlp_linear = _linear_config(
        schedule, mlp_max_plaintext_rows, mlp_hoist_strategy
    )

    rotations = set(
        qkv_rotations(feature_layout, operator_config=qkv_linear)
    )
    rotations.update(
        attention_to_w_o_rotations(
            LinearCarrierLayout(
                seq_len=int(feature_layout.seq_len),
                dimension=int(feature_layout.hidden_dim),
                slots=int(feature_layout.slots),
            ),
            feature_layout,
            attention_layout,
            operator_config=AttentionOperatorConfig(
                sparse_max_baby_offsets=int(sparse_max_baby_offsets),
                shear_baby_steps=int(shear_baby_steps),
                pv_complex_delta_folding=bool(pv_complex_delta_folding),
                pv_prefold_after_softmax=bool(pv_prefold_after_softmax),
            ),
            log_n=int(log_n),
            prefix_k=int(prefix_k),
        )
    )
    rotations.update(
        linear_rotations(
            dimension=int(feature_layout.hidden_dim),
            slots=int(feature_layout.slots),
            token_lanes=int(feature_layout.tokens_per_cipher),
            include_conjugation=True,
            operator_config=output_linear,
        )
    )
    rotations.update(rmsnorm_rotations(feature_layout))
    rotations.update(
        qk_norm_rotations(token_lanes=int(feature_layout.tokens_per_cipher))
    )
    rotations.update(
        mlp_rotations(feature_layout, operator_config=mlp_linear)
    )
    if bool(emit_kv_cache):
        from ..operators.attention.prefill.input import emit_cache_c_rotations

        rotations.update(
            emit_cache_c_rotations(
                LinearCarrierLayout(
                    seq_len=int(feature_layout.seq_len),
                    dimension=int(feature_layout.hidden_dim),
                    slots=int(feature_layout.slots),
                ),
                attention_layout,
                sparse_max_baby_offsets=int(sparse_max_baby_offsets),
                direct_pairfold_v=_prefill_delta_pairfold_v_enabled(
                    emit_kv_cache, decode_packing
                ),
            )
        )
        if decode_packing == "pairfold128":
            from ..layouts.pairfold128 import PairFold128Layout
            from ..operators.attention.pairfold.pairfold_prefill import (
                prefill_c_to_pairfold_rotations,
            )

            rotations.update(
                prefill_c_to_pairfold_rotations(PairFold128Layout())
            )
    rotations.update(
        BootstrapOperator.required_rotations(
            log_n=int(log_n),
            log_slots=int(bootstrap_log_slots),
            level_budget=tuple(int(value) for value in bootstrap_level_budget),
            deep_level_budget=(
                None
                if bootstrap_deep_level_budget is None
                else tuple(int(v) for v in bootstrap_deep_level_budget)
            ),
            sparse_log_slots=(
                tuple(int(k) for k in schedule.bootstrap_sparse_log_slots) or None
            ),
        )
    )
    return tuple(sorted(int(rotation) for rotation in rotations if int(rotation)))


def operation_audit(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    schedule: LayerSchedule | None = None,
    intermediate_width: int = 14336,
    input_rmsnorm_degree: int = 16,
    post_attention_rmsnorm_degree: int = 32,
    silu_degree: int = 48,
    silu_multiplicative_depth: int = 6,
) -> dict[str, object]:
    """Static component audit without pretending polynomial internals are free."""

    schedule = LayerSchedule() if schedule is None else schedule
    # The stages below read the schedule's decisions as plain locals;
    # binding them here keeps the object at the boundary only.
    linear_baby_steps = schedule.linear_baby_steps
    qkv_hoist_strategy = schedule.qkv_hoist_strategy
    output_hoist_strategy = schedule.output_hoist_strategy
    mlp_hoist_strategy = schedule.mlp_hoist_strategy
    qkv_max_plaintext_rows = schedule.qkv_max_plaintext_rows
    output_max_plaintext_rows = schedule.output_max_plaintext_rows
    mlp_max_plaintext_rows = schedule.mlp_max_plaintext_rows
    sparse_max_baby_offsets = schedule.sparse_max_baby_offsets
    shear_baby_steps = schedule.shear_baby_steps
    pv_complex_delta_folding = schedule.pv_complex_delta_folding
    pv_prefold_after_softmax = schedule.pv_prefold_after_softmax
    pair_softmax_pre_exp_bootstrap = schedule.pair_softmax_pre_exp_bootstrap
    reuse_softmax_round_square = schedule.reuse_softmax_round_square
    direct_softmax_slim_inactive_fill = schedule.direct_softmax_slim_inactive_fill
    fuse_softmax_slim_chebyshev_map = schedule.fuse_softmax_slim_chebyshev_map
    adaptive_rmsnorm_slim_refresh = schedule.adaptive_rmsnorm_slim_refresh
    rmsnorm2_slim_log_slots = schedule.rmsnorm2_slim_log_slots
    batched_sparse_attention_input = schedule.batched_sparse_attention_input
    fuse_rmsnorm_output_scale = schedule.fuse_rmsnorm_output_scale
    fuse_silu_domain_map = schedule.fuse_silu_domain_map
    preserve_prebootstrap_residual = schedule.preserve_prebootstrap_residual
    softmax_main_bootstrap_iterations = schedule.softmax_main_bootstrap_iterations
    softmax_main_deep_bootstrap = schedule.softmax_main_deep_bootstrap
    softmax_lambda_bootstrap_iterations = schedule.softmax_lambda_bootstrap_iterations
    match_softmax_single_pass_bootstrap_level = schedule.match_softmax_single_pass_bootstrap_level

    attention_input = pipeline_static_audit(
        feature_layout,
        attention_layout,
        qkv_operator_config=_linear_config(
            schedule, qkv_max_plaintext_rows, qkv_hoist_strategy
        ),
        sparse_max_baby_offsets=int(sparse_max_baby_offsets),
        shear_baby_steps=int(shear_baby_steps),
        include_scores=False,
    )
    qk = delta_qk_operation_counts(
        attention_layout,
        complex_feature_folding=True,
        key_features_pre_folded=True,
        query_features_pre_scaled=True,
        query_features_pre_folded=True,
    )
    pv = delta_pv_operation_counts(
        attention_layout,
        complex_delta_folding=bool(pv_complex_delta_folding),
        probability_pre_folded=bool(pv_prefold_after_softmax),
        value_pre_folded=bool(pv_prefold_after_softmax),
    )
    pairfm = pairfm_output_complexity(
        feature_layout,
        attention_layout,
    )
    input_rms = rmsnorm_complexity(
        feature_layout,
        polynomial_degree=int(input_rmsnorm_degree),
        apply_gamma=not bool(fuse_rmsnorm_output_scale),
    )
    post_rms = rmsnorm_complexity(
        feature_layout,
        polynomial_degree=int(post_attention_rmsnorm_degree),
        apply_gamma=not bool(fuse_rmsnorm_output_scale),
    )
    softmax = delta_softmax_alg2_complexity(
        attention_layout,
        reuse_round_square=bool(reuse_softmax_round_square),
        direct_slim_inactive_fill=bool(
            direct_softmax_slim_inactive_fill
        ),
        fuse_slim_chebyshev_map=bool(
            fuse_softmax_slim_chebyshev_map
        ),
    )
    mlp = mlp_complexity(
        feature_layout,
        intermediate_width=int(intermediate_width),
        silu_polynomial_degree=int(silu_degree),
        silu_multiplicative_depth=int(silu_multiplicative_depth),
        baby_steps=int(linear_baby_steps),
        max_plaintext_rows=int(mlp_max_plaintext_rows),
        reuse_baby_rotations_across_outputs=True,
        fuse_silu_domain_map=bool(fuse_silu_domain_map),
    )
    w_rows = int(feature_layout.hidden_dim)
    w_ptct = w_rows * (int(feature_layout.cipher_count) // 2)
    residual_bootstrap_copies = (
        0
        if bool(preserve_prebootstrap_residual)
        else 2 * int(feature_layout.cipher_count)
    )
    return {
        "shape": {
            "seq_len": int(feature_layout.seq_len),
            "hidden_dim": int(feature_layout.hidden_dim),
            "intermediate_width": int(intermediate_width),
            "canonical_cipher_count": int(feature_layout.cipher_count),
        },
        "stages": {
            "input_rmsnorm": input_rms.to_dict(),
            "feature_major_qkv_to_attention": attention_input.to_dict(),
            "qk": qk,
            "softmax": softmax.to_dict(),
            "pv": pv,
            "pairfm_output_conversion": pairfm.to_dict(),
            "w_o": {
                "encoded_weight_rows": w_rows,
                "pt_ct_multiplications": w_ptct,
                "multiplicative_depth": 1,
                "max_plaintext_rows": int(output_max_plaintext_rows),
            },
            "post_attention_rmsnorm": post_rms.to_dict(),
            "mlp": mlp.to_dict(),
            "residuals": {
                "ciphertext_additions": 2 * int(feature_layout.cipher_count),
                "bootstrap_identity_scalar_multiplications": int(
                    residual_bootstrap_copies
                ),
                "bootstrap_identity_rescales": int(
                    residual_bootstrap_copies
                ),
            },
        },
        "settings": {
            "pv_complex_delta_folding": bool(pv_complex_delta_folding),
            "pv_prefold_after_softmax": bool(pv_prefold_after_softmax),
            "pair_softmax_pre_exp_bootstrap": bool(
                pair_softmax_pre_exp_bootstrap
            ),
            "reuse_softmax_round_square": bool(
                reuse_softmax_round_square
            ),
            "direct_softmax_slim_inactive_fill": bool(
                direct_softmax_slim_inactive_fill
            ),
            "fuse_softmax_slim_chebyshev_map": bool(
                fuse_softmax_slim_chebyshev_map
            ),
            "softmax_main_bootstrap_iterations": (
                None
                if softmax_main_bootstrap_iterations is None
                else int(softmax_main_bootstrap_iterations)
            ),
            "softmax_main_deep_bootstrap": bool(
                softmax_main_deep_bootstrap
            ),
            "softmax_lambda_bootstrap_iterations": (
                None
                if softmax_lambda_bootstrap_iterations is None
                else int(softmax_lambda_bootstrap_iterations)
            ),
            "match_softmax_single_pass_bootstrap_level": bool(
                match_softmax_single_pass_bootstrap_level
            ),
            "adaptive_rmsnorm_slim_refresh": bool(
                adaptive_rmsnorm_slim_refresh
            ),
            "rmsnorm2_slim_log_slots": (
                None
                if rmsnorm2_slim_log_slots is None
                else int(rmsnorm2_slim_log_slots)
            ),
            "batched_sparse_attention_input": bool(
                batched_sparse_attention_input
            ),
            "fuse_rmsnorm_output_scale": bool(
                fuse_rmsnorm_output_scale
            ),
            "fuse_silu_domain_map": bool(fuse_silu_domain_map),
            "preserve_prebootstrap_residual": bool(
                preserve_prebootstrap_residual
            ),
            "qkv_max_plaintext_rows": int(qkv_max_plaintext_rows),
            "output_max_plaintext_rows": int(output_max_plaintext_rows),
            "mlp_max_plaintext_rows": int(mlp_max_plaintext_rows),
            "qkv_hoist_strategy": str(qkv_hoist_strategy),
            "output_hoist_strategy": str(output_hoist_strategy),
            "mlp_hoist_strategy": str(mlp_hoist_strategy),
        },
    }
