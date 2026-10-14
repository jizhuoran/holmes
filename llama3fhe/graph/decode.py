"""Decode layer graph: :func:`evaluate_decode`, :func:`decode_layer_rotations` and
:func:`inspect_decode_layer_input` (moved verbatim out of the former ``llama3fhe/graph.py``)."""

from __future__ import annotations

import os
import time

import numpy as np

from ..approx.chebyshev import chebyshev_ps_mul_depth
from ..backend import release_all, trim_device_allocator_cache
from ..config import Llama3CKKSConfig
from ..decode_packing import (
    DEFAULT_DECODE_PACKING,
    DecodePacking,
    validate_decode_packing,
)
from ..layouts.attention import AttentionPairLayout
from ..layouts.feature_major import FeatureMajorPrefillLayout
from ..layouts.linear import LinearCarrierLayout
from ..operators.linear.prefill.mlp import mlp_fhe, mlp_rotations
from ..operators.linear.prefill.qkv import qkv_fhe, qkv_rotations
from ..operators.nonlinear.bootstrap import BootstrapOperator
from ..operators.norm.refresh import (
    bootstrap_rmsnorm_fhe,
    refresh_ciphers,
)
from ..operators.norm.rmsnorm import residual_add, rmsnorm_fhe, rmsnorm_rotations
from ..operators.pairfold_kv_cache import PairFoldLayerCache
from ..operators.primitives import drop_levels
from ..schedule import LayerSchedule
from .common import (
    _decode_linear_config,
    LayerApproximations,
    LayerResult,
    LayerWeights,
    _cipher_levels,
    _linear_config,
)


def inspect_decode_layer_input(
    input_ciphers: tuple[object, ...] | list[object],
    *,
    feature_layout: FeatureMajorPrefillLayout,
    crypto_context,
) -> dict[str, object]:
    """Validate DecodeFM × 1 and report the entry level / limb budget.

    Lane 0 holds the new token; lanes 1..T-1 are zero.  ``cur_limbs`` is
    ``L - input_level``.  Fresh client encrypt uses ``input_level=0``
    (full tower).  A previous Decode layer's FFN residual may arrive with
    only a few limbs and then RMSN1 needs ``refresh_input=True``.
    """

    if int(feature_layout.seq_len) != 1:
        raise ValueError(
            "Decode layer input requires DecodeFM seq_len=1, "
            f"got {feature_layout.seq_len}."
        )
    if int(feature_layout.cipher_count) != 1:
        raise ValueError(
            "Decode layer input requires one FM ciphertext, "
            f"got cipher_count={feature_layout.cipher_count}."
        )
    ciphers = tuple(input_ciphers)
    if len(ciphers) != 1:
        raise ValueError(
            f"Decode layer expects 1 DecodeFM ciphertext, got {len(ciphers)}."
        )
    cipher = ciphers[0]
    slots = int(getattr(cipher, "slots", feature_layout.slots))
    if int(slots) != int(feature_layout.slots):
        raise ValueError(
            "DecodeFM slot count must match the layout, "
            f"got {slots} vs {feature_layout.slots}."
        )
    if int(crypto_context.max_slots) != int(feature_layout.slots):
        raise ValueError(
            "DecodeFM layout/context slot mismatch: "
            f"{feature_layout.slots} vs {crypto_context.max_slots}."
        )
    return {
        "cipher_count": 1,
        "seq_len": 1,
        "hidden_dim": int(feature_layout.hidden_dim),
        "slots": int(slots),
        "tokens_per_cipher": int(feature_layout.tokens_per_cipher),
        "live_lane": 0,
        "level": int(crypto_context.level_for_cipher(cipher)),
        "cur_limbs": int(cipher.state.cur_limbs),
        "scale_degree": int(cipher.state.scale_degree),
    }


def decode_sparse_bootstrap_log_slots(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout | None = None,
    *,
    slim_period_rows: int | None = 1,
    merged_page_softmax: bool = True,
) -> tuple[int, ...]:
    """``log2`` slot counts of the sparse bootstrap programs the Decode graph can use.

    * the period-``slim_period_rows`` slim RMSNorm row of the DecodeFM layout
      (``slim_period_slots`` -> 8 slots -> 3 for one cipher / period 1), the
      program :func:`~llama3fhe.operators.norm.rmsnorm.bootstrap_rmsnorm_fhe`
      routes the rsqrt refresh through when the operator carries it;
    * the merged-page Softmax ``total`` / ``filled`` / ``lambda`` rows, which
      are periodic every ``attention_layout.decode_page_merge_stride`` slots
      (4096 -> 12; :func:`~llama3fhe.operators.attention.decode.decode_softmax.merged_page_sparse_period`).

    Same derivation as the multi-card :func:`llama3fhe.parallel.worker.sparse_bootstrap_log_slots`
    with ``rmsnorm_slim_period_rows=1`` and the preset's
    ``bootstrap_extra_sparse_log_slots``; the result is what
    :attr:`LayerSchedule.bootstrap_sparse_log_slots` should carry.
    """

    from ..operators.norm.rmsnorm import slim_period_slots

    ks: set[int] = set()
    full = int(feature_layout.slots)
    if slim_period_rows is not None and int(slim_period_rows) > 0:
        period = int(slim_period_slots(feature_layout, int(slim_period_rows)))
        if 1 < period < full and period & (period - 1) == 0:
            ks.add(period.bit_length() - 1)
    if bool(merged_page_softmax) and attention_layout is not None:
        stride = int(attention_layout.decode_page_merge_stride)
        groups = int(attention_layout.decode_page_merge_groups)
        if (
            1 < stride < full
            and stride & (stride - 1) == 0
            and int(attention_layout.slots) % (stride * groups) == 0
        ):
            ks.add(stride.bit_length() - 1)
    return tuple(sorted(ks))


def decode_layer_rotations(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    schedule: LayerSchedule | None = None,
    skip_attention: bool = True,
    include_bootstrap: bool = False,
    log_n: int = 16,
    decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
) -> tuple[int, ...]:
    """Rotation union for the Decode layer graph (not Prefill Delta)."""

    decode_packing = validate_decode_packing(decode_packing)
    schedule = LayerSchedule() if schedule is None else schedule
    # SL (schedule.decode_shifted_lanes) changes the diagonal set of QKV and
    # the MLP; W_O stays lane-replicated (shifted lanes).
    mlp_linear = _decode_linear_config(
        schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy
    )
    rotations: set[int] = set(rmsnorm_rotations(feature_layout))
    rotations.update(mlp_rotations(feature_layout, operator_config=mlp_linear))
    if not skip_attention:
        qkv_linear = _decode_linear_config(
            schedule,
            schedule.qkv_max_plaintext_rows,
            schedule.qkv_hoist_strategy,
        )
        output_linear = _linear_config(
            schedule,
            schedule.output_max_plaintext_rows,
            schedule.output_hoist_strategy,
        )
        if decode_packing == "pairfold128":
            from ..layouts.pairfold128 import PairFold128Layout
            from ..operators.attention.pairfold.pairfold_decode import (
                pairfold_decode_rotations,
            )
            from ..operators.attention.pairfold.pairfold_route import (
                pairfold_route_rotations,
            )
            from ..operators.linear.pairfold.pairfold_output import pairfold_w_o_rotations
            from ..operators.linear.pairfold.pairfold_qkv import pairfold_qkv_rotations

            pairfold_layout = PairFold128Layout()
            rotations.update(
                pairfold_qkv_rotations(
                    pairfold_layout, operator_config=qkv_linear
                )
            )
            rotations.update(
                pairfold_route_rotations(
                    pairfold_layout,
                    token_columns=None,
                    max_baby_offsets=int(schedule.sparse_max_baby_offsets),
                )
            )
            rotations.update(pairfold_decode_rotations(pairfold_layout))
            rotations.update(
                pairfold_w_o_rotations(
                    pairfold_layout, operator_config=output_linear
                )
            )
        else:
            from ..layouts.linear import LinearCarrierLayout
            from ..operators.attention.decode.decode import (
                decode_attention_rotations,
                decode_output_c_rotations,
            )
            from ..operators.attention.decode.decode_adapter import (
                decode_adapter_rotations,
            )
            from ..operators.linear.prefill.output import decode_w_o_rotations

            rotations.update(
                qkv_rotations(feature_layout, operator_config=qkv_linear)
            )
            rotations.update(
                decode_adapter_rotations(
                    LinearCarrierLayout(
                        seq_len=1,
                        dimension=int(feature_layout.hidden_dim),
                        slots=int(feature_layout.slots),
                    ),
                    attention_layout,
                )
            )
            rotations.update(decode_attention_rotations(attention_layout))
            rotations.update(
                decode_output_c_rotations(attention_layout, feature_layout)
            )
            rotations.update(
                decode_w_o_rotations(
                    feature_layout, operator_config=output_linear
                )
            )
    if bool(include_bootstrap):
        rotations.update(
            BootstrapOperator.required_rotations(
                log_n=int(log_n),
                log_slots=int(schedule.bootstrap_log_slots),
                level_budget=tuple(
                    int(value) for value in schedule.bootstrap_level_budget
                ),
                sparse_log_slots=(
                    tuple(int(k) for k in schedule.bootstrap_sparse_log_slots)
                    or None
                ),
            )
        )
    return tuple(sorted(value for value in rotations if value))


def evaluate_decode(
    hidden_states: np.ndarray,
    *,
    layer_idx: int,
    weights: LayerWeights,
    approximations: LayerApproximations,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    config: Llama3CKKSConfig,
    crypto_context,
    input_ciphers: tuple[object, ...] | list[object],
    bootstrap_operator=None,
    schedule: LayerSchedule | None = None,
    refresh_input: bool = False,
    skip_attention: bool = True,
    kv_cache_k_ciphers: tuple[object, ...] | list[object] | None = None,
    kv_cache_v_ciphers: tuple[object, ...] | list[object] | None = None,
    occupied: int = 0,
    position: int = 0,
    rope_theta: float = 500000.0,
    verify: bool = False,
    expected_output: np.ndarray | None = None,
    max_kv_pages: int = 1,
    prefix_bundle=None,
    consume_kv_cache: bool = False,
    decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
    pairfold_kv_cache: PairFoldLayerCache | None = None,
) -> LayerResult:
    """One Decode layer from DecodeFM × 1.  Prefill ``evaluate`` is unchanged.

    Entry: 1 ciphertext, lane 0 live.  Fresh encrypt uses ``input_level=0``
    and ``cur_limbs = L``.  ``skip_attention=True`` walks RMSN1 then identity
    residual, RMSN2, and FFN.  With the archived explicit
    ``decode_packing="legacy_c"``, the full path is RMSN1 → QKV → C adapter →
    Decode Alg2 attention → short W_O → residual → RMSN2 → FFN.
    ``decode_packing="pairfold128"``
    replaces that attention span with one sparse QKV carrier, transactional
    PairFold KV append, streaming-Q QK / the same Alg2 Softmax / PairFold PV,
    and folded W_O.  It never materializes the legacy adapter or O_C→FM step.
    A prepared ``bootstrap_operator`` refreshes Alg2 ``y`` after exp, each
    λ round, and fused RMSN1/RMSN2 Meta-BTS (with the same slim-refresh /
    MLP limb floor as Prefill).  That is what lets Decode reuse Prefill's
    fitted degrees on the production depth-37 tower.  ``refresh_input``
    still covers RMSN1 for later layers.  For either selected cache format,
    ``consume_kv_cache=True`` explicitly transfers ownership into this call;
    ``False`` (default) clones history.  PairFold occupancy comes solely from
    ``PairFoldLayerCache.content_tokens`` and must equal external ``occupied``.
    """

    decode_packing = validate_decode_packing(decode_packing)
    if bool(verify) and expected_output is None:
        raise ValueError("verify=True requires expected_output.")
    if not skip_attention and approximations.softmax is None:
        raise ValueError("skip_attention=False requires approximations.softmax.")
    if bool(refresh_input) and bootstrap_operator is None:
        raise ValueError("refresh_input=True requires a bootstrap operator.")
    occupied = int(occupied)
    position = int(position)
    max_kv_pages = int(max_kv_pages)
    if max_kv_pages <= 0:
        raise ValueError("max_kv_pages must be positive.")
    pairfold_layout = None
    if decode_packing == "legacy_c":
        if pairfold_kv_cache is not None:
            raise ValueError(
                "legacy_c Decode does not accept pairfold_kv_cache."
            )
        if (
            not skip_attention
            and occupied >= int(attention_layout.seq_len) * max_kv_pages
        ):
            raise ValueError(
                "Decode KV pages are full: "
                f"occupied={occupied}, seq_len={attention_layout.seq_len}, "
                f"max_kv_pages={max_kv_pages}."
            )
    else:
        from ..layouts.pairfold128 import PairFold128Layout

        pairfold_layout = PairFold128Layout()
        if kv_cache_k_ciphers is not None or kv_cache_v_ciphers is not None:
            raise ValueError(
                "pairfold128 Decode does not accept legacy K/V cache "
                "cipher tuples."
            )
        if pairfold_kv_cache is not None and not isinstance(
            pairfold_kv_cache, PairFoldLayerCache
        ):
            raise TypeError(
                "pairfold_kv_cache must be a PairFoldLayerCache or None."
            )
        if pairfold_kv_cache is not None and pairfold_kv_cache.is_released:
            raise ValueError("pairfold_kv_cache has already been released.")
        if bool(skip_attention) and pairfold_kv_cache is not None:
            raise ValueError(
                "skip_attention=True does not consume or return a PairFold "
                "KV cache."
            )
        if not skip_attention:
            from ..operators.attention.pairfold.pairfold_operator import (
                pairfold_prefix_token_count,
            )

            cache_tokens = (
                0
                if pairfold_kv_cache is None
                else int(pairfold_kv_cache.content_tokens)
            )
            if occupied != cache_tokens:
                raise ValueError(
                    "PairFold cache occupancy mismatch: "
                    f"occupied={occupied}, content_tokens={cache_tokens}."
                )
            capacity = int(pairfold_layout.seq_len) * max_kv_pages
            if occupied >= capacity:
                raise ValueError(
                    "Decode PairFold KV pages are full: "
                    f"content_tokens={occupied}, page_tokens="
                    f"{pairfold_layout.seq_len}, max_kv_pages={max_kv_pages}."
                )
            prefix_tokens = pairfold_prefix_token_count(
                prefix_bundle,
                page_size=int(pairfold_layout.seq_len),
                key_value_heads=int(pairfold_layout.key_value_heads),
                head_dim=int(pairfold_layout.head_dim),
            )
            expected_position = prefix_tokens + cache_tokens
            if position != expected_position:
                raise ValueError(
                    "PairFold Decode position must be contiguous with Prefix "
                    "and content occupancy: "
                    f"position={position}, prefix_tokens={prefix_tokens}, "
                    f"content_tokens={cache_tokens}, "
                    f"expected={expected_position}."
                )
            feature_contract = (
                int(feature_layout.seq_len),
                int(feature_layout.hidden_dim),
                int(feature_layout.slots),
            )
            expected_feature_contract = (
                1,
                int(pairfold_layout.carrier_dimension),
                int(pairfold_layout.slots),
            )
            if feature_contract != expected_feature_contract:
                raise ValueError(
                    "pairfold128 Decode requires production DecodeFM layout "
                    f"{expected_feature_contract}, got {feature_contract}."
                )
            attention_contract = (
                int(attention_layout.seq_len),
                int(attention_layout.head_dim),
                int(attention_layout.query_heads),
                int(attention_layout.key_value_heads),
                int(attention_layout.slots),
            )
            expected_attention_contract = (
                int(pairfold_layout.seq_len),
                int(pairfold_layout.head_dim),
                int(pairfold_layout.query_heads),
                int(pairfold_layout.key_value_heads),
                int(pairfold_layout.slots),
            )
            if attention_contract != expected_attention_contract:
                raise ValueError(
                    "pairfold128 Decode requires production attention layout "
                    f"{expected_attention_contract}, got "
                    f"{attention_contract}."
                )

    schedule = LayerSchedule() if schedule is None else schedule
    if (
        not skip_attention
        and decode_packing == "pairfold128"
        and schedule.kv_value_limbs is not None
        and schedule.kv_limbs is None
    ):
        raise ValueError("PairFold V limb target requires a K limb target.")
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    expected_shape = (
        int(feature_layout.seq_len),
        int(feature_layout.hidden_dim),
    )
    if hidden_states.shape != expected_shape:
        raise ValueError(
            f"Decode hidden stub must have shape {expected_shape}, "
            f"got {hidden_states.shape}."
        )
    if int(attention_layout.slots) != int(feature_layout.slots):
        raise ValueError("Decode FM and attention slot counts must match.")

    input_ciphers = tuple(input_ciphers)
    entry = inspect_decode_layer_input(
        input_ciphers,
        feature_layout=feature_layout,
        crypto_context=crypto_context,
    )

    fuse_rmsnorm_output_scale = bool(schedule.fuse_rmsnorm_output_scale)
    fuse_silu_domain_map = bool(schedule.fuse_silu_domain_map)
    # Periodic slim pack of both RMSNorms (period-1 row on the one DecodeFM
    # cipher -> log_slots=3 sparse rsqrt refresh when the operator has it).
    decode_rmsnorm_slim_period_rows = (
        None
        if schedule.decode_rmsnorm_slim_period_rows is None
        or int(schedule.decode_rmsnorm_slim_period_rows) <= 0
        else int(schedule.decode_rmsnorm_slim_period_rows)
    )
    wall_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    nested: dict[str, object] = {
        "decode_input": entry,
        "output_levels": {},
    }
    input_norm = first_residual = post_norm = mlp = final_residual = None
    qkv = adapter = attention = None
    skip_zeros: tuple[object, ...] = ()
    wo_ciphers: tuple[object, ...] = ()
    fm_ciphers: tuple[object, ...] = ()
    raised_oc: tuple[object, ...] = ()
    cache_k: tuple[object, ...] = ()
    cache_v: tuple[object, ...] = ()
    # Page-level ownership is tracked until the flat cache tuple is built.
    # This matters for destructive ``consume_kv_cache`` calls: an exception
    # while appending K/V must still release the transferred old pages.
    owned_k_pages: tuple[tuple[object, ...], ...] = ()
    owned_v_pages: tuple[tuple[object, ...], ...] = ()
    pairfold_working_cache: PairFoldLayerCache | None = None
    pairfold_token = None
    pairfold_attention = None
    pairfold_qkv = None
    pairfold_refreshed_o: tuple[object, ...] = ()

    mlp_inputs: tuple[object, ...] = ()
    owns_mlp_inputs = False
    owns_cache = False
    softmax_bootstrap_calls = 0
    softmax_native_bootstrap_calls = 0
    pairfold_bootstrap_calls = {
        "kv_limb_drop": 0,
        "attention_softmax": 0,
        "folded_o_refresh": 0,
    }
    pairfold_native_bootstrap_calls = {
        "kv_limb_drop": 0,
        "attention_softmax": 0,
        "folded_o_refresh": 0,
    }
    layer_bootstrap_calls_before = 0
    layer_native_bootstrap_calls_before = 0
    if bootstrap_operator is not None:
        layer_bootstrap_calls_before = int(bootstrap_operator.calls)
        layer_native_bootstrap_calls_before = int(
            bootstrap_operator.native_calls
        )

    def current_bootstrap_counters() -> tuple[int, int]:
        if bootstrap_operator is None:
            return 0, 0
        return (
            int(bootstrap_operator.calls),
            int(bootstrap_operator.native_calls),
        )

    silu_entry = approximations.silu
    mlp_suffix_limbs = int(os.environ.get("LLAMA_FHE_MLP_SUFFIX_LIMBS", "6"))
    if mlp_suffix_limbs < 3:
        raise ValueError("LLAMA_FHE_MLP_SUFFIX_LIMBS must be at least 3.")
    shifted_lanes = bool(getattr(schedule, "decode_shifted_lanes", False))
    if shifted_lanes:
        # the SL MLP masks lane 0 at the gate/up exit and at the down exit (+2 levels)
        mlp_suffix_limbs += 2
    required_mlp_input_limbs = (
        chebyshev_ps_mul_depth(silu_entry["coefficients"])
        + mlp_suffix_limbs
        - int(bool(fuse_silu_domain_map))
    )
    softmax_bootstrap_iterations = (
        schedule.residual_bootstrap_iterations
        if schedule.softmax_main_bootstrap_iterations is None
        else int(schedule.softmax_main_bootstrap_iterations)
    )
    try:
        if decode_packing == "pairfold128" and not skip_attention:
            cache_start = time.perf_counter()
            if pairfold_kv_cache is None:
                pairfold_working_cache = PairFoldLayerCache()
            elif bool(consume_kv_cache):
                # From this point onward the graph owns the input object.
                # Direct callers must treat consume=True as a move; model
                # integration clears its source slot before making the call.
                pairfold_working_cache = pairfold_kv_cache
            else:
                pairfold_working_cache = pairfold_kv_cache.clone()
            stage_seconds["pairfold_cache_acquire"] = (
                time.perf_counter() - cache_start
            )

        start = time.perf_counter()
        if bool(refresh_input):
            minimum_attention_rmsnorm_limbs = 14 + int(
                os.environ.get("LLAMA_FHE_ATTN_EXTRA_LIMBS", "0")
            )
            input_norm = bootstrap_rmsnorm_fhe(
                input_ciphers,
                weights.input_layernorm,
                coeffs=approximations.input_norm,
                layout=feature_layout,
                crypto_context=crypto_context,
                bootstrap_operator=bootstrap_operator,
                pair_inputs=False,
                bootstrap_iterations=int(schedule.residual_bootstrap_iterations),
                bootstrap_precision_bits=int(
                    schedule.residual_bootstrap_precision
                ),
                bootstrap_output_level_drop=int(
                    schedule.residual_bootstrap_level_drop
                ),
                minimum_rmsnorm_output_limbs=minimum_attention_rmsnorm_limbs,
                adaptive_slim_refresh=bool(
                    schedule.adaptive_rmsnorm_slim_refresh
                ),
                defer_output_scale=fuse_rmsnorm_output_scale,
                slim_period_rows=decode_rmsnorm_slim_period_rows,
                function_prefix=(
                    f"model.layers.{layer_idx}.input_layernorm.decode"
                ),
            )
        else:
            input_norm = rmsnorm_fhe(
                input_ciphers,
                weights.input_layernorm,
                coeffs=approximations.input_norm,
                layout=feature_layout,
                crypto_context=crypto_context,
                verify=False,
                defer_output_scale=fuse_rmsnorm_output_scale,
                function_prefix=(
                    f"model.layers.{layer_idx}.input_layernorm.decode"
                ),
            )
        stage_seconds["input_rmsnorm"] = time.perf_counter() - start
        nested["input_rmsnorm"] = input_norm.stage_seconds
        nested["output_levels"]["input_rmsnorm"] = input_norm.output_levels
        nested["output_levels"]["input_rmsnorm_limbs"] = tuple(
            sorted({int(cipher.state.cur_limbs) for cipher in input_norm.ciphers})
        )
        after_input_norm_calls = layer_bootstrap_calls_before
        after_input_norm_native = layer_native_bootstrap_calls_before
        if bootstrap_operator is not None:
            after_input_norm_calls = int(bootstrap_operator.calls)
            after_input_norm_native = int(bootstrap_operator.native_calls)

        if skip_attention:
            start = time.perf_counter()
            skip_zeros = tuple(
                crypto_context.fhe.homo_sub(
                    cipher, cipher, crypto_context.context
                )
                for cipher in input_ciphers
            )
            first_residual = residual_add(
                input_ciphers,
                skip_zeros,
                layout=feature_layout,
                crypto_context=crypto_context,
                verify=False,
            )
            release_all(skip_zeros)
            skip_zeros = ()
            stage_seconds["attention_residual"] = time.perf_counter() - start
            nested["output_levels"]["attention_residual"] = (
                first_residual.output_levels
            )
            input_norm.release()
            input_norm = None
        elif decode_packing == "pairfold128":
            from ..operators.attention.pairfold.pairfold_operator import (
                pairfold_attention_fhe,
            )
            from ..operators.attention.pairfold.pairfold_route import (
                pairfold_kv_from_carrier_fhe,
                pairfold_query_from_carrier_fhe,
            )
            from ..operators.linear.pairfold.pairfold_output import (
                pairfold_w_o_fhe,
                pairfold_w_o_needs_refresh,
            )
            from ..operators.linear.pairfold.pairfold_qkv import pairfold_qkv_fhe
            from ..operators.pairfold_kv_cache import (
                drop_pairfold_token_to_limbs,
            )

            if pairfold_layout is None or pairfold_working_cache is None:
                raise RuntimeError(
                    "pairfold128 Decode cache/layout ownership was not "
                    "initialized."
                )
            attention_path_start = time.perf_counter()
            position = int(position)
            token_column = int(pairfold_working_cache.content_tokens) % int(
                pairfold_layout.seq_len
            )

            start = time.perf_counter()
            pairfold_qkv = pairfold_qkv_fhe(
                hidden_states,
                weights.q_proj,
                weights.k_proj,
                weights.v_proj,
                layout=pairfold_layout,
                config=config,
                # SL or lane-replicated per schedule.decode_shifted_lanes;
                # single rank: fold inside the operator (default fold=True)
                operator_config=_decode_linear_config(
                    schedule,
                    schedule.qkv_max_plaintext_rows,
                    schedule.qkv_hoist_strategy,
                ),
                crypto_context=crypto_context,
                input_ciphers=input_norm.ciphers,
                verify=False,
                input_column_scale=input_norm.deferred_output_scale,
            )
            stage_seconds["qkv_projection"] = time.perf_counter() - start
            nested["qkv_projection"] = dict(pairfold_qkv.stage_seconds)
            nested["output_levels"]["qkv_carriers"] = (
                pairfold_qkv.output_levels
            )
            input_norm.release()
            input_norm = None

            kv_route_profile: dict[str, float] = {}
            start = time.perf_counter()
            pairfold_token = pairfold_kv_from_carrier_fhe(
                pairfold_qkv.carrier_cipher,
                carrier_layout=pairfold_qkv.output_layout,
                layout=pairfold_layout,
                crypto_context=crypto_context,
                position=position,
                token_column=token_column,
                rope_theta=float(rope_theta),
                max_baby_offsets=int(schedule.sparse_max_baby_offsets),
                profile=kv_route_profile,
            )
            stage_seconds["kv_extract"] = time.perf_counter() - start
            nested["pairfold_kv_extract"] = dict(kv_route_profile)

            kv_drop_calls_before, kv_drop_native_before = (
                current_bootstrap_counters()
            )
            if schedule.kv_limbs is not None:
                start = time.perf_counter()
                pairfold_token = drop_pairfold_token_to_limbs(
                    pairfold_token,
                    target_limbs=int(schedule.kv_limbs),
                    crypto_context=crypto_context,
                    bootstrap_operator=bootstrap_operator,
                    bootstrap_iterations=int(
                        schedule.residual_bootstrap_iterations
                    ),
                    value_target_limbs=schedule.kv_value_limbs,
                )
                stage_seconds["kv_limb_drop"] = time.perf_counter() - start
            else:
                stage_seconds["kv_limb_drop"] = 0.0
            kv_drop_calls_after, kv_drop_native_after = (
                current_bootstrap_counters()
            )
            pairfold_bootstrap_calls["kv_limb_drop"] = (
                kv_drop_calls_after - kv_drop_calls_before
            )
            pairfold_native_bootstrap_calls["kv_limb_drop"] = (
                kv_drop_native_after - kv_drop_native_before
            )

            start = time.perf_counter()
            page_index, appended_column = pairfold_working_cache.append(
                pairfold_token, crypto_context=crypto_context
            )
            pairfold_token = None
            stage_seconds["kv_append"] = time.perf_counter() - start
            nested["pairfold_kv_append"] = {
                "page_index": int(page_index),
                "token_column": int(appended_column),
                "content_tokens": int(pairfold_working_cache.content_tokens),
                "page_count": int(len(pairfold_working_cache.pages)),
            }

            key_pages = tuple(
                page.key_groups for page in pairfold_working_cache.pages
            )
            value_pages = tuple(
                page.value_groups for page in pairfold_working_cache.pages
            )
            occupied_pages = tuple(
                int(page.occupied) for page in pairfold_working_cache.pages
            )
            query_route_profiles: list[dict[str, float]] = []

            def extract_streaming_query(
                carrier_cipher,
                *,
                group: int,
                query_offset: int,
            ):
                route_profile: dict[str, float] = {}
                query = pairfold_query_from_carrier_fhe(
                    carrier_cipher,
                    carrier_layout=pairfold_qkv.output_layout,
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

            attention_calls_before, attention_native_before = (
                current_bootstrap_counters()
            )
            start = time.perf_counter()
            pairfold_attention = pairfold_attention_fhe(
                pairfold_qkv.carrier_cipher,
                key_pages,
                value_pages,
                query_extractor=extract_streaming_query,
                pairfold_layout=pairfold_layout,
                score_layout=attention_layout,
                crypto_context=crypto_context,
                softmax_config=approximations.softmax,
                occupied_pages=occupied_pages,
                layer_idx=int(layer_idx),
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=softmax_bootstrap_iterations,
                bootstrap_precision_bits=int(
                    schedule.residual_bootstrap_precision
                ),
                prefix_bundle=prefix_bundle,
            )
            stage_seconds["decode_attention"] = time.perf_counter() - start
            attention_stages = dict(pairfold_attention.stage_seconds)
            nested["decode_attention"] = attention_stages
            nested["decode_attention_profile"] = dict(
                pairfold_attention.profile
            )
            nested["pairfold_query_routes"] = tuple(query_route_profiles)
            for name in (
                "q_extract_and_broadcast",
                "qk_prefix",
                "qk_content",
                "softmax",
                "probability_refresh",
                "pv_prefix",
                "pv_content",
                "pv",
                "folded_o_assemble",
            ):
                stage_seconds[name] = float(attention_stages.get(name, 0.0))
            # The attention orchestrator no longer needs the carrier after it
            # has returned its sole owned folded-O ciphertext.
            pairfold_qkv.release()
            pairfold_qkv = None
            attention_calls_after, attention_native_after = (
                current_bootstrap_counters()
            )
            pairfold_bootstrap_calls["attention_softmax"] = (
                attention_calls_after - attention_calls_before
            )
            pairfold_native_bootstrap_calls["attention_softmax"] = (
                attention_native_after - attention_native_before
            )
            softmax_bootstrap_calls = pairfold_bootstrap_calls[
                "attention_softmax"
            ]
            softmax_native_bootstrap_calls = (
                pairfold_native_bootstrap_calls["attention_softmax"]
            )

            folded_o_cipher = pairfold_attention.folded_o_cipher
            if folded_o_cipher is None:
                raise RuntimeError(
                    "PairFold attention returned no folded-O ciphertext."
            )
            folded_o_inputs = (folded_o_cipher,)
            folded_o_calls_before, folded_o_native_before = (
                current_bootstrap_counters()
            )
            start = time.perf_counter()
            needs_output_refresh = pairfold_w_o_needs_refresh(
                folded_o_inputs,
                crypto_context=crypto_context,
            )
            if needs_output_refresh:
                if bootstrap_operator is None:
                    raise ValueError(
                        "PairFold folded-O has too few limbs for W_O and no "
                        "bootstrap operator is available."
                    )
                refreshed = bootstrap_operator.refresh(
                    folded_o_cipher,
                    iterations=int(schedule.residual_bootstrap_iterations),
                    precision_bits=int(schedule.residual_bootstrap_precision),
                )
                forbidden_refresh_ids = {
                    id(folded_o_cipher),
                    *(id(cipher) for cipher in input_ciphers),
                    *(
                        id(cipher)
                        for page in pairfold_working_cache.pages
                        for cipher in page.handles
                    ),
                }
                if pairfold_kv_cache is not None:
                    forbidden_refresh_ids.update(
                        id(cipher)
                        for page in pairfold_kv_cache.pages
                        for cipher in page.handles
                    )
                if refreshed is None:
                    raise RuntimeError(
                        "PairFold folded-O refresh returned no ciphertext."
                    )
                if id(refreshed) in forbidden_refresh_ids:
                    raise RuntimeError(
                        "PairFold folded-O refresh must return a distinct "
                        "owned ciphertext and may not alias folded-O, Decode "
                        "input, or KV cache storage."
                    )
                pairfold_refreshed_o = (refreshed,)
                folded_o_inputs = pairfold_refreshed_o
            stage_seconds["folded_o_refresh"] = time.perf_counter() - start
            folded_o_calls_after, folded_o_native_after = (
                current_bootstrap_counters()
            )
            pairfold_bootstrap_calls["folded_o_refresh"] = (
                folded_o_calls_after - folded_o_calls_before
            )
            pairfold_native_bootstrap_calls["folded_o_refresh"] = (
                folded_o_native_after - folded_o_native_before
            )
            nested["folded_o_refresh"] = {
                "applied": bool(needs_output_refresh),
            }

            clear_start = time.perf_counter()
            if bool(schedule.evict_rotation_cuda_cache_before_output_projection):
                clear = getattr(
                    crypto_context.context, "clear_cuda_rotation_cache", None
                )
                if callable(clear):
                    clear()
            stage_seconds["decode_rotation_cache_clear"] = (
                time.perf_counter() - clear_start
            )
            trim_start = time.perf_counter()
            trim_device_allocator_cache(crypto_context.device)
            stage_seconds["decode_allocator_trim"] = (
                time.perf_counter() - trim_start
            )

            folded_w_o_profile: dict[str, float] = {}
            start = time.perf_counter()
            wo_ciphers = pairfold_w_o_fhe(
                folded_o_inputs,
                weights.o_proj,
                layout=pairfold_layout,
                crypto_context=crypto_context,
                operator_config=_linear_config(
                    schedule,
                    schedule.output_max_plaintext_rows,
                    schedule.output_hoist_strategy,
                ),
                profile=folded_w_o_profile,
            )
            stage_seconds["folded_w_o"] = time.perf_counter() - start
            nested["folded_w_o"] = dict(folded_w_o_profile)
            owned_refreshed_o = pairfold_refreshed_o
            pairfold_refreshed_o = ()
            release_all(owned_refreshed_o)
            pairfold_attention.release()
            pairfold_attention = None

            start = time.perf_counter()
            first_residual = residual_add(
                input_ciphers,
                wo_ciphers,
                layout=feature_layout,
                crypto_context=crypto_context,
                verify=False,
            )
            residual_add_seconds = time.perf_counter() - start
            stage_seconds["attention_residual_add"] = residual_add_seconds
            # Keep the historical name as a compatibility alias, but give it
            # one unambiguous meaning: the residual add itself.
            stage_seconds["attention_residual"] = residual_add_seconds
            release_all(wo_ciphers)
            wo_ciphers = ()
            stage_seconds["attention_path_total"] = (
                time.perf_counter() - attention_path_start
            )
            nested["decode_output"] = {
                name: stage_seconds[name]
                for name in (
                    "folded_o_refresh",
                    "decode_rotation_cache_clear",
                    "decode_allocator_trim",
                    "folded_w_o",
                    "attention_residual_add",
                )
            }
            nested["output_levels"]["attention_residual"] = (
                first_residual.output_levels
            )
        else:
            from ..operators.attention.decode.decode import (
                decode_attention_pages_fhe,
                output_c_to_decode_fm_fhe,
            )
            from ..operators.attention.decode.decode_adapter import decode_prepare_qkv_fhe
            from ..operators.kv_cache import (
                drop_c_ciphers_to_limbs,
                extend_kv_c_pages,
                flatten_c_pages,
                split_c_pages,
            )
            from ..operators.linear.prefill.output import decode_w_o_fhe

            occupied = int(occupied)
            position = int(position)
            seq_len = int(attention_layout.seq_len)
            pair_count = int(attention_layout.key_value_pair_count)
            local_index = occupied % seq_len
            start = time.perf_counter()
            qkv = qkv_fhe(
                hidden_states,
                weights.q_proj,
                weights.k_proj,
                weights.v_proj,
                attention_layout=attention_layout,
                layout=feature_layout,
                config=config,
                operator_config=_linear_config(
                    schedule,
                    schedule.qkv_max_plaintext_rows,
                    schedule.qkv_hoist_strategy,
                ),
                crypto_context=crypto_context,
                input_ciphers=input_norm.ciphers,
                verify=False,
                function_prefix=(
                    f"model.layers.{layer_idx}.self_attn.feature_major_qkv.decode"
                ),
                input_column_scale=input_norm.deferred_output_scale,
            )
            stage_seconds["qkv_projection"] = time.perf_counter() - start
            nested["qkv_projection"] = qkv.stage_seconds
            nested["output_levels"]["qkv_carriers"] = qkv.output_levels
            input_norm.release()
            input_norm = None

            start = time.perf_counter()
            adapter = decode_prepare_qkv_fhe(
                qkv.carrier_ciphers,
                carrier_layout=qkv.output_layout,
                attention_layout=attention_layout,
                crypto_context=crypto_context,
                position=position,
                token_index=local_index,
                rope_theta=float(rope_theta),
            )
            qkv.release()
            qkv = None
            stage_seconds["decode_adapter"] = time.perf_counter() - start
            if adapter.stage_seconds:
                nested["decode_adapter"] = dict(adapter.stage_seconds)
            if adapter.route_stats:
                nested["decode_adapter_route_stats"] = dict(adapter.route_stats)

            if schedule.kv_limbs is not None:
                kv_drop_start = time.perf_counter()
                kv_drop_k_start = time.perf_counter()
                adapter.key_ciphers = drop_c_ciphers_to_limbs(
                    adapter.key_ciphers,
                    target_limbs=int(schedule.kv_limbs),
                    crypto_context=crypto_context,
                    bootstrap_operator=bootstrap_operator,
                    bootstrap_iterations=int(
                        schedule.residual_bootstrap_iterations
                    ),
                )
                kv_drop_k = time.perf_counter() - kv_drop_k_start
                kv_drop_v_start = time.perf_counter()
                adapter.value_ciphers = drop_c_ciphers_to_limbs(
                    adapter.value_ciphers,
                    target_limbs=int(schedule.kv_limbs),
                    crypto_context=crypto_context,
                    bootstrap_operator=bootstrap_operator,
                    bootstrap_iterations=int(
                        schedule.residual_bootstrap_iterations
                    ),
                )
                kv_drop_v = time.perf_counter() - kv_drop_v_start
                stage_seconds["kv_limb_drop"] = time.perf_counter() - kv_drop_start
                nested["kv_limb_drop"] = {
                    "key": kv_drop_k,
                    "value": kv_drop_v,
                    "total": stage_seconds["kv_limb_drop"],
                }
            else:
                stage_seconds["kv_limb_drop"] = 0.0

            split_start = time.perf_counter()
            if occupied == 0:
                old_k_pages = ()
                old_v_pages = ()
            else:
                if kv_cache_k_ciphers is None or kv_cache_v_ciphers is None:
                    raise ValueError(
                        "occupied>0 requires kv_cache_k_ciphers and "
                        "kv_cache_v_ciphers."
                    )
                old_k_pages = split_c_pages(
                    kv_cache_k_ciphers, pair_count=pair_count
                )
                old_v_pages = split_c_pages(
                    kv_cache_v_ciphers, pair_count=pair_count
                )
                expected_pages = (
                    occupied + seq_len - 1
                ) // seq_len
                if len(old_k_pages) != expected_pages or len(old_v_pages) != expected_pages:
                    raise ValueError(
                        "cached C pages do not match occupied tokens: "
                        f"occupied={occupied}, pages={len(old_k_pages)}."
                    )
            stage_seconds["kv_split"] = time.perf_counter() - split_start
            # Complete historical pages transfer when consume_kv_cache=True;
            # only the live tail is homo_add'd.  consume=False keeps the
            # original deep-copy snapshot path.
            extend_start = time.perf_counter()
            owned_k_pages, owned_v_pages = extend_kv_c_pages(
                old_k_pages,
                old_v_pages,
                adapter.key_ciphers,
                adapter.value_ciphers,
                crypto_context=crypto_context,
                consume=bool(consume_kv_cache) and occupied > 0,
                open_new_page=occupied == 0 or local_index == 0,
            )
            stage_seconds["kv_extend"] = time.perf_counter() - extend_start
            adapter.key_ciphers = ()
            adapter.value_ciphers = ()
            owns_cache = True
            occupied_after = occupied + 1

            flatten_start = time.perf_counter()
            cache_k = flatten_c_pages(owned_k_pages)
            cache_v = flatten_c_pages(owned_v_pages)
            stage_seconds["kv_flatten"] = time.perf_counter() - flatten_start
            remaining = occupied_after
            occupied_pages = []
            for _ in owned_k_pages:
                take = min(seq_len, remaining)
                occupied_pages.append(take)
                remaining -= take

            start = time.perf_counter()
            attention = decode_attention_pages_fhe(
                adapter.query_ciphers,
                owned_k_pages,
                owned_v_pages,
                layout=attention_layout,
                crypto_context=crypto_context,
                softmax_config=approximations.softmax,
                occupied_pages=occupied_pages,
                layer_idx=int(layer_idx),
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=softmax_bootstrap_iterations,
                bootstrap_precision_bits=int(
                    schedule.residual_bootstrap_precision
                ),
                prefix_bundle=prefix_bundle,
            )
            stage_seconds["decode_attention"] = time.perf_counter() - start
            nested["decode_attention"] = attention.stage_seconds
            for name, seconds in attention.stage_seconds.items():
                stage_seconds[f"decode_attention.{name}"] = float(seconds)
            if bootstrap_operator is not None:
                softmax_bootstrap_calls = (
                    int(bootstrap_operator.calls) - after_input_norm_calls
                )
                softmax_native_bootstrap_calls = (
                    int(bootstrap_operator.native_calls)
                    - after_input_norm_native
                )

            attention_output_start = time.perf_counter()
            oc_inputs = attention.output_ciphers
            output_c_start = attention_output_start
            if bootstrap_operator is not None:
                oc_limbs = min(
                    int(cipher.state.cur_limbs) for cipher in oc_inputs
                )
                # Sparse O_C→FM is a PT×CT rescale.  Layer 31 Alg2 k=5
                # leaves 1 limb; Prefill PairFM would have refreshed here.
                if oc_limbs < 3:
                    raised_oc = refresh_ciphers(
                        oc_inputs,
                        crypto_context=crypto_context,
                        bootstrap_operator=bootstrap_operator,
                        iterations=int(
                            schedule.residual_bootstrap_iterations
                        ),
                        precision_bits=int(
                            schedule.residual_bootstrap_precision
                        ),
                        function_prefix=(
                            f"model.layers.{layer_idx}.decode.output_c"
                        ),
                    )
                    oc_inputs = raised_oc
                    nested["output_levels"]["decode_output_c_refresh"] = (
                        oc_limbs
                    )
            stage_seconds["decode_output_c_refresh"] = time.perf_counter() - output_c_start
            conversion_start = time.perf_counter()
            fm_ciphers = output_c_to_decode_fm_fhe(
                oc_inputs,
                attention_layout=attention_layout,
                feature_layout=feature_layout,
                crypto_context=crypto_context,
            )
            stage_seconds["decode_output_c_to_fm"] = time.perf_counter() - conversion_start
            if raised_oc:
                release_all(raised_oc)
                raised_oc = ()
            attention.release()
            attention = None
            clear_start = time.perf_counter()
            if bool(schedule.evict_rotation_cuda_cache_before_output_projection):
                clear = getattr(
                    crypto_context.context, "clear_cuda_rotation_cache", None
                )
                if callable(clear):
                    clear()
            stage_seconds["decode_rotation_cache_clear"] = time.perf_counter() - clear_start
            trim_start = time.perf_counter()
            trim_device_allocator_cache(crypto_context.device)
            stage_seconds["decode_allocator_trim"] = time.perf_counter() - trim_start
            wo_start = time.perf_counter()
            wo_ciphers = decode_w_o_fhe(
                fm_ciphers,
                weights.o_proj,
                feature_layout=feature_layout,
                crypto_context=crypto_context,
                operator_config=_linear_config(
                    schedule,
                    schedule.output_max_plaintext_rows,
                    schedule.output_hoist_strategy,
                ),
            )
            stage_seconds["decode_w_o"] = time.perf_counter() - wo_start
            release_all(fm_ciphers)
            fm_ciphers = ()
            residual_start = time.perf_counter()
            first_residual = residual_add(
                input_ciphers,
                wo_ciphers,
                layout=feature_layout,
                crypto_context=crypto_context,
                verify=False,
            )
            residual_add_seconds = time.perf_counter() - residual_start
            stage_seconds["attention_residual_add"] = residual_add_seconds
            stage_seconds["attention_residual"] = residual_add_seconds
            release_all(wo_ciphers)
            wo_ciphers = ()
            adapter.release()
            adapter = None
            stage_seconds["attention_output_total"] = (
                time.perf_counter() - attention_output_start
            )
            nested["decode_output"] = {
                name: stage_seconds[name]
                for name in (
                    "decode_output_c_refresh",
                    "decode_output_c_to_fm",
                    "decode_rotation_cache_clear",
                    "decode_allocator_trim",
                    "decode_w_o",
                    "attention_residual_add",
                )
                if name in stage_seconds
            }
            nested["output_levels"]["attention_residual"] = (
                first_residual.output_levels
            )

        start = time.perf_counter()
        if bootstrap_operator is not None:
            post_norm = bootstrap_rmsnorm_fhe(
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
                bootstrap_precision_bits=int(
                    schedule.residual_bootstrap_precision
                ),
                bootstrap_output_level_drop=int(
                    schedule.residual_bootstrap_level_drop
                ),
                minimum_rmsnorm_output_limbs=required_mlp_input_limbs,
                return_refreshed_inputs=False,
                defer_output_scale=fuse_rmsnorm_output_scale,
                adaptive_slim_refresh=bool(
                    schedule.adaptive_rmsnorm_slim_refresh
                ),
                slim_period_rows=decode_rmsnorm_slim_period_rows,
                use_slim_program=schedule.rmsnorm2_slim_log_slots is not None,
                function_prefix=(
                    f"model.layers.{layer_idx}.post_attention_layernorm.decode"
                ),
            )
            stage_seconds["post_attention_bootstrap_rmsnorm"] = (
                time.perf_counter() - start
            )
            nested["post_attention_bootstrap_rmsnorm"] = post_norm.stage_seconds
            nested["post_attention_rmsnorm_level_schedule"] = dict(
                post_norm.level_schedule
            )
            nested["output_levels"]["post_attention_bootstrap_raw"] = tuple(
                post_norm.bootstrap_output_levels
            )
        else:
            post_norm = rmsnorm_fhe(
                first_residual.ciphers,
                weights.post_attention_layernorm,
                coeffs=approximations.post_attention_norm,
                layout=feature_layout,
                crypto_context=crypto_context,
                verify=False,
                defer_output_scale=fuse_rmsnorm_output_scale,
                function_prefix=(
                    f"model.layers.{layer_idx}.post_attention_layernorm.decode"
                ),
            )
            stage_seconds["post_attention_rmsnorm"] = (
                time.perf_counter() - start
            )
            nested["post_attention_rmsnorm"] = post_norm.stage_seconds
        nested["output_levels"]["post_attention_rmsnorm"] = (
            post_norm.output_levels
        )

        if bool(schedule.evict_rotation_cuda_cache_before_mlp):
            start = time.perf_counter()
            clear = getattr(
                crypto_context.context, "clear_cuda_rotation_cache", None
            )
            if callable(clear):
                clear()
            stage_seconds["pre_mlp_rotation_cache_evict"] = (
                time.perf_counter() - start
            )
        trim_device_allocator_cache(crypto_context.device)

        start = time.perf_counter()
        if bootstrap_operator is not None:
            # DecodeFM is one real CT, so Prefill's pair-pack refresh_ciphers
            # cannot run.  Unpaired refresh + the same suffix-6 floor leaves
            # the next layer enough limbs to Meta-BTS RMSN1.
            minimum_mlp_input_limbs = min(
                int(cipher.state.cur_limbs) for cipher in post_norm.ciphers
            )
            refreshed_mlp_input = (
                minimum_mlp_input_limbs < required_mlp_input_limbs
            )
            requested_fusion_mlp_drop = (
                int(bool(fuse_rmsnorm_output_scale))
                + int(bool(fuse_silu_domain_map))
            )
            requested_mlp_input_drop = (
                int(schedule.mlp_input_level_drop) + requested_fusion_mlp_drop
            )
            maximum_safe_mlp_input_drop = max(
                0, minimum_mlp_input_limbs - required_mlp_input_limbs
            )
            scheduled_mlp_input_drop = min(
                requested_mlp_input_drop, maximum_safe_mlp_input_drop
            )
            applied_mlp_input_drop = 0
            if refreshed_mlp_input:
                raised_mlp_inputs: list[object] = []
                try:
                    for cipher in post_norm.ciphers:
                        raised_mlp_inputs.append(
                            bootstrap_operator.refresh(
                                cipher,
                                iterations=int(
                                    schedule.residual_bootstrap_iterations
                                ),
                                precision_bits=int(
                                    schedule.residual_bootstrap_precision
                                ),
                            )
                        )
                    raised_limbs = min(
                        int(cipher.state.cur_limbs)
                        for cipher in raised_mlp_inputs
                    )
                    applied_mlp_input_drop = max(
                        0, raised_limbs - required_mlp_input_limbs
                    )
                    mlp_inputs = drop_levels(
                        tuple(raised_mlp_inputs),
                        applied_mlp_input_drop,
                        crypto_context=crypto_context,
                    )
                finally:
                    release_all(raised_mlp_inputs)
            else:
                mlp_inputs = drop_levels(
                    post_norm.ciphers,
                    scheduled_mlp_input_drop,
                    crypto_context=crypto_context,
                )
                applied_mlp_input_drop = scheduled_mlp_input_drop
            owns_mlp_inputs = True
            nested["output_levels"]["mlp_input"] = _cipher_levels(
                mlp_inputs, crypto_context=crypto_context
            )
            minimum_scheduled_mlp_limbs = min(
                int(cipher.state.cur_limbs) for cipher in mlp_inputs
            )
            if minimum_scheduled_mlp_limbs < required_mlp_input_limbs:
                raise RuntimeError(
                    "Decode MLP input scheduling violated its consumer budget: "
                    f"{minimum_scheduled_mlp_limbs} < "
                    f"{required_mlp_input_limbs}."
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
            stage_seconds[
                "pre_mlp_bootstrap"
                if refreshed_mlp_input
                else "mlp_input_modulus_drop"
            ] = time.perf_counter() - start
        else:
            mlp_inputs = post_norm.ciphers

        start = time.perf_counter()
        if shifted_lanes:
            # SL MLP: the one-sided shifted-lane kernel of the multi-card
            # station, run as the single rank of a world of one (the
            # collectives are no-ops at world_size == 1).
            from ..operators.linear.shard.decode_mlp_shard import decode_mlp_shard_fhe
            from ..parallel.decode.decode_ownership import DecodeRankOwnership
            from ..parallel.decode.single_rank import SingleRankTransport

            mlp = decode_mlp_shard_fhe(
                hidden_states,
                weights.gate_proj,
                weights.up_proj,
                weights.down_proj,
                silu_entry=approximations.silu,
                layout=feature_layout,
                ownership=DecodeRankOwnership(rank=0, world_size=1),
                transport=SingleRankTransport(),
                config=config,
                operator_config=_decode_linear_config(
                    schedule,
                    schedule.mlp_max_plaintext_rows,
                    schedule.mlp_hoist_strategy,
                ),
                crypto_context=crypto_context,
                input_ciphers=mlp_inputs,
                input_column_scale=post_norm.deferred_output_scale,
                fuse_silu_domain_map=fuse_silu_domain_map,
                verify=False,
                tag="mlp",
                split_gate_up=False,
            )
        else:
            mlp = mlp_fhe(
                hidden_states,
                weights.gate_proj,
                weights.up_proj,
                weights.down_proj,
                silu_entry=approximations.silu,
                layout=feature_layout,
                config=config,
                operator_config=_linear_config(
                    schedule,
                    schedule.mlp_max_plaintext_rows,
                    schedule.mlp_hoist_strategy,
                ),
                crypto_context=crypto_context,
                input_ciphers=mlp_inputs,
                verify=False,
                input_column_scale=post_norm.deferred_output_scale,
                fuse_silu_domain_map=fuse_silu_domain_map,
            )
        stage_seconds["mlp"] = time.perf_counter() - start
        nested["mlp"] = dict(mlp.stage_seconds)
        nested["output_levels"]["mlp"] = mlp.output_levels
        if owns_mlp_inputs:
            release_all(mlp_inputs)
            mlp_inputs = ()
            owns_mlp_inputs = False
        post_norm.release()
        post_norm = None

        start = time.perf_counter()
        final_residual = residual_add(
            first_residual.ciphers,
            mlp.ciphers,
            layout=feature_layout,
            crypto_context=crypto_context,
            verify=False,
        )
        stage_seconds["mlp_residual"] = time.perf_counter() - start
        nested["output_levels"]["layer_output"] = final_residual.output_levels
        mlp.release()
        mlp = None
        first_residual.release()
        first_residual = None

        output = None
        max_abs_diff = mean_abs_diff = None
        if bool(verify):
            packed = np.stack(
                [
                    np.asarray(crypto_context.decrypt(cipher), dtype=np.float64)
                    for cipher in final_residual.ciphers
                ]
            )
            output = feature_layout.unpack(packed)
            difference = output - np.asarray(expected_output, dtype=np.float32)
            max_abs_diff = float(np.max(np.abs(difference)))
            mean_abs_diff = float(np.mean(np.abs(difference)))

        output_ciphers = final_residual.ciphers
        final_residual.ciphers = ()
        result_k, result_v = cache_k, cache_v
        cache_k = ()
        cache_v = ()
        owns_cache = False
        if decode_packing == "pairfold128" and not skip_attention:
            layer_calls_after, layer_native_after = (
                current_bootstrap_counters()
            )
            total_calls = layer_calls_after - layer_bootstrap_calls_before
            total_native_calls = (
                layer_native_after - layer_native_bootstrap_calls_before
            )
            named_calls = sum(pairfold_bootstrap_calls.values())
            named_native_calls = sum(
                pairfold_native_bootstrap_calls.values()
            )
            other_residual_calls = total_calls - named_calls
            other_residual_native_calls = (
                total_native_calls - named_native_calls
            )
            if other_residual_calls < 0 or other_residual_native_calls < 0:
                raise RuntimeError(
                    "PairFold bootstrap counters are not monotonic across "
                    "the Decode layer."
                )
            nested["bootstrap_calls"] = {
                name: {
                    "calls": int(pairfold_bootstrap_calls[name]),
                    "native_calls": int(
                        pairfold_native_bootstrap_calls[name]
                    ),
                }
                for name in (
                    "kv_limb_drop",
                    "attention_softmax",
                    "folded_o_refresh",
                )
            }
            nested["bootstrap_calls"]["other_residual"] = {
                "calls": int(other_residual_calls),
                "native_calls": int(other_residual_native_calls),
            }
            nested["bootstrap_calls"]["total"] = {
                "calls": int(total_calls),
                "native_calls": int(total_native_calls),
            }
        result = LayerResult(
            ciphers=tuple(output_ciphers),
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={str(key): float(value) for key, value in stage_seconds.items()},
            nested_stage_seconds=nested,
            softmax_bootstrap_calls=int(softmax_bootstrap_calls),
            residual_bootstrap_calls=(
                0
                if bootstrap_operator is None
                else int(bootstrap_operator.calls)
                - layer_bootstrap_calls_before
                - int(softmax_bootstrap_calls)
            ),
            softmax_native_bootstrap_calls=int(softmax_native_bootstrap_calls),
            residual_native_bootstrap_calls=(
                0
                if bootstrap_operator is None
                else int(bootstrap_operator.native_calls)
                - layer_native_bootstrap_calls_before
                - int(softmax_native_bootstrap_calls)
            ),
            output_levels=_cipher_levels(
                output_ciphers, crypto_context=crypto_context
            ),
            output=output,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            key_c_ciphers=tuple(result_k),
            value_c_ciphers=tuple(result_v),
            pairfold_kv_cache=pairfold_working_cache,
        )
        pairfold_working_cache = None
        return result
    finally:
        release_all(skip_zeros)
        release_all(wo_ciphers)
        release_all(fm_ciphers)
        release_all(raised_oc)
        owned_refreshed_o = pairfold_refreshed_o
        pairfold_refreshed_o = ()
        release_all(owned_refreshed_o)
        if pairfold_token is not None:
            pairfold_token.release()
        if pairfold_attention is not None:
            pairfold_attention.release()
        if pairfold_qkv is not None:
            pairfold_qkv.release()
        if pairfold_working_cache is not None:
            pairfold_working_cache.release()
        if owns_mlp_inputs:
            release_all(mlp_inputs)
        if owns_cache:
            if cache_k:
                release_all(cache_k)
            else:
                for page in owned_k_pages:
                    release_all(page)
            if cache_v:
                release_all(cache_v)
            else:
                for page in owned_v_pages:
                    release_all(page)
        if attention is not None:
            attention.release()
        if adapter is not None:
            adapter.release()
        if qkv is not None:
            qkv.release()
        if final_residual is not None:
            final_residual.release()
        if mlp is not None:
            mlp.release()
        if post_norm is not None:
            post_norm.release()
        if first_residual is not None:
            first_residual.release()
        if input_norm is not None:
            input_norm.release()
