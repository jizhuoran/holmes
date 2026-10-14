from __future__ import annotations

"""End-to-end attention boundary: QKV carriers in, W_O carriers out."""

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from llama3fhe.backend import (
    profile_sync_device as sync_device,  # stage-timer sync, no-op under T2_STAGE_SYNC=0
)
from llama3fhe.backend import release_all, release_if_supported, synchronize_device
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

from ..audit import delta_attention_rotations, delta_softmax_rotations
from ..config import AttentionOperatorConfig
from ..prefix.prefix_pc import pv_prefix_pc_fhe, qk_prefix_pc_fhe
from ..prefix.prefix_softmax import (
    _add_cipher_pairs,
    prefix_softmax_rotations,
    softmax_alg2_prefix_partitions_fhe,
)
from .input import (
    DeltaKVPage,
    complex_token_attention_rotations,
    prepare_attention_inputs_fhe,
)
from .output import pairfm_output_rotations, prepare_w_o_inputs_fhe
from .pv import (
    fold_probability_delta_halves_fhe,
    fold_value_delta_halves_fhe,
    pv_delta_fhe,
)
from .qk import qk_delta_fhe


def _diagnostic_cipher_range(
    label: str,
    ciphers,
    *,
    crypto_context,
    scale: float | None = None,
) -> None:
    if os.environ.get("LLAMA_FHE_STAGE_DIAGNOSTICS", "") != "1":
        return
    values = np.stack(
        [np.asarray(crypto_context.decrypt(cipher)) for cipher in ciphers]
    )
    extra = ""
    if scale is not None:
        extra = (
            f" scaled_max={float(np.max(np.abs(values.real))) * float(scale)}"
            f" scale={float(scale)}"
        )
    print(
        f"[attention-diagnostic] stage={label} "
        f"real_max={float(np.max(np.abs(values.real)))} "
        f"imag_max={float(np.max(np.abs(values.imag)))} "
        f"real_mean={float(np.mean(np.abs(values.real)))} "
        f"imag_mean={float(np.mean(np.abs(values.imag)))} "
        f"levels={sorted({int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers})}"
        f"{extra}",
        flush=True,
    )


@dataclass
class AttentionToWOResult:
    """Own the complex feature-major ciphertexts consumed directly by W_O."""

    w_o_input_ciphers: tuple[object, ...]
    stage_seconds: dict[str, float]
    input_scheduler_stats: dict[str, float]
    output_levels: tuple[int, ...]
    level_trace: dict[str, tuple[int, ...]]
    key_c_ciphers: tuple[object, ...] = ()
    value_c_ciphers: tuple[object, ...] = ()
    pairfold_value_ciphers: tuple[object, ...] = ()
    delta_kv_page: DeltaKVPage | None = None

    def release(self) -> None:
        release_all(self.w_o_input_ciphers)
        release_all(self.key_c_ciphers)
        release_all(self.value_c_ciphers)
        release_all(self.pairfold_value_ciphers)
        if self.delta_kv_page is not None:
            self.delta_kv_page.release()
            self.delta_kv_page = None


@dataclass
class DeltaAttentionAlg2FHEResult:
    output_ciphers: tuple[object, ...]
    stage_seconds: dict[str, float]
    level_trace: dict[str, tuple[int, ...]]

    def release(self) -> None:
        for cipher in self.output_ciphers:
            release_if_supported(cipher)


def attention_fhe_delta_alg2(
    query_ciphers: tuple[object, ...],
    key_delta_ciphers: tuple[object, ...],
    value_delta_ciphers: tuple[object, ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    softmax_config,
    operator_config: AttentionOperatorConfig | None = None,
    layer_idx: int = 0,
    softmax_bootstrap=None,
    token_valid_mask: np.ndarray | None = None,
    prefix_bundle=None,
    historical_delta_pages: tuple[DeltaKVPage, ...] = (),
    on_qk_last_use: Callable[[], None] | None = None,
    on_value_last_use: Callable[[], None] | None = None,
    on_pv_last_use: Callable[[], None] | None = None,
) -> DeltaAttentionAlg2FHEResult:
    """Ciphertext-only QK → Delta Alg2 → PV with no repack boundary.

    Every schedule decision is read off ``operator_config`` — the same
    object the surrounding operator validates — so this core cannot state
    defaults of its own.
    """

    from .multi_page import prefill_delta_page_masks, softmax_alg2_fhe_delta_pages
    from .softmax import softmax_alg2_fhe_delta

    operator_config = (
        AttentionOperatorConfig()
        if operator_config is None
        else operator_config
    )
    pv_prefold_after_softmax = operator_config.pv_prefold_after_softmax
    refresh_softmax_exp_inputs = operator_config.refresh_softmax_exp_inputs
    refresh_softmax_exp_outputs = operator_config.refresh_softmax_exp_outputs
    historical_delta_pages = tuple(historical_delta_pages)
    if historical_delta_pages and prefix_bundle is not None and int(prefix_bundle.prefix_k):
        raise ValueError(
            "plaintext Prefix and historical Delta pages are not yet supported together."
        )
    for index, page in enumerate(historical_delta_pages):
        if len(page.key_ciphers) != int(layout.key_value_pair_count):
            raise ValueError(
                f"historical Delta page {index} needs {layout.key_value_pair_count} K ciphertexts."
            )
        if len(page.value_ciphers) != int(layout.key_value_pair_count):
            raise ValueError(
                f"historical Delta page {index} needs {layout.key_value_pair_count} V ciphertexts."
            )

    stage_seconds: dict[str, float] = {}
    scores: tuple[object, ...] = ()
    historical_scores: list[tuple[object, ...]] = []
    scores_p: tuple[object, ...] = ()
    probabilities: tuple[object, ...] = ()
    historical_probabilities: tuple[tuple[object, ...], ...] = ()
    probabilities_p: tuple[object, ...] = ()
    folded_probabilities: tuple[object, ...] = ()
    folded_values: tuple[object, ...] = ()
    outputs: tuple[object, ...] = ()
    outputs_p: tuple[object, ...] = ()
    softmax_result = None
    success = False
    level_trace: dict[str, tuple[int, ...]] = {
        name: tuple(
            sorted(
                {
                    int(crypto_context.level_for_cipher(cipher))
                    for cipher in ciphers
                }
            )
        )
        for name, ciphers in (
            ("query", query_ciphers),
            ("key_delta", key_delta_ciphers),
            ("value_delta", value_delta_ciphers),
        )
    }
    try:
        start = time.perf_counter()
        scores = qk_delta_fhe(
            query_ciphers,
            key_delta_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            operator_config=operator_config,
        )
        for page in historical_delta_pages:
            historical_scores.append(
                qk_delta_fhe(
                    query_ciphers,
                    page.key_ciphers,
                    layout=layout,
                    crypto_context=crypto_context,
                    operator_config=operator_config,
                )
            )
        sync_device(crypto_context.device)
        stage_seconds["qk"] = time.perf_counter() - start
        level_trace["scores"] = tuple(
            sorted(
                {
                    int(crypto_context.level_for_cipher(cipher))
                    for cipher in scores
                }
            )
        )

        score_scale = 1.0 / math.sqrt(float(layout.head_dim))
        _diagnostic_cipher_range(
            "scores",
            scores,
            crypto_context=crypto_context,
            scale=score_scale,
        )

        prefix_k = 0
        if prefix_bundle is not None:
            prefix_k = int(prefix_bundle.prefix_k)
            start = time.perf_counter()
            scores_p = qk_prefix_pc_fhe(
                query_ciphers,
                prefix_bundle.key,
                layout=layout,
                crypto_context=crypto_context,
                prefix_k=prefix_k,
                complex_feature_folding=bool(
                    operator_config.qk_complex_feature_folding
                ),
            )
            sync_device(crypto_context.device)
            stage_seconds["qk_prefix"] = time.perf_counter() - start
            # Prefix QK is a separate plaintext-K path from the Delta QK
            # kernel.  Keep its decrypted range visible in an explicitly
            # requested diagnostic run; otherwise a bad prefix score is only
            # observable much later as a Softmax probability mismatch.
            _diagnostic_cipher_range(
                "scores_prefix",
                scores_p,
                crypto_context=crypto_context,
                scale=score_scale,
            )

        # Both Delta and optional prefix QK have crossed an explicit device
        # barrier.  The adapter owner can now drop Q/K without extending their
        # lifetime through Softmax and PV.
        if on_qk_last_use is not None:
            on_qk_last_use()
        query_ciphers = ()
        key_delta_ciphers = ()

        start = time.perf_counter()
        if prefix_k:
            softmax_stages: dict[str, float] = {}
            probabilities_p, probabilities = softmax_alg2_prefix_partitions_fhe(
                scores_p,
                scores,
                layout=layout,
                crypto_context=crypto_context,
                config=softmax_config,
                prefix_k=prefix_k,
                layer_idx=int(layer_idx),
                score_scale=score_scale,
                content_form="delta",
                token_valid_mask=token_valid_mask,
                bootstrap_operator=softmax_bootstrap,
                refresh_before_exp=bool(refresh_softmax_exp_inputs),
                refresh_after_exp=bool(refresh_softmax_exp_outputs),
                operator_config=operator_config,
                stage_seconds=softmax_stages,
            )
            level_trace["probabilities"] = tuple(
                sorted(
                    {
                        int(crypto_context.level_for_cipher(cipher))
                        for cipher in probabilities
                    }
                )
            )
            stage_seconds.update(
                {
                    f"softmax.{name}": float(seconds)
                    for name, seconds in softmax_stages.items()
                }
            )
        elif historical_delta_pages:
            softmax_result = softmax_alg2_fhe_delta_pages(
                (*historical_scores, scores),
                prefill_delta_page_masks(
                    layout,
                    historical_key_valid_masks=tuple(
                        page.token_valid_mask for page in historical_delta_pages
                    ),
                    current_valid_mask=token_valid_mask,
                ),
                layout=layout,
                layer_idx=int(layer_idx),
                config=softmax_config,
                crypto_context=crypto_context,
                operator_config=operator_config,
                score_scale=score_scale,
                bootstrap_operator=softmax_bootstrap,
                refresh_before_exp=bool(refresh_softmax_exp_inputs),
                refresh_after_exp=bool(refresh_softmax_exp_outputs),
                current_valid_mask=token_valid_mask,
            )
            historical_probabilities = softmax_result.probability_pages[:-1]
            probabilities = softmax_result.probability_pages[-1]
            softmax_result.probability_pages = ()
            level_trace["probabilities"] = tuple(softmax_result.output_levels)
            level_trace["softmax_slim_sparse"] = (
                int(softmax_result.slim_period_rows or 0),
                int(softmax_result.slim_sparse_bootstraps),
            )
        else:
            softmax_result = softmax_alg2_fhe_delta(
                scores,
                layout=layout,
                layer_idx=int(layer_idx),
                config=softmax_config,
                crypto_context=crypto_context,
                operator_config=operator_config,
                score_scale=score_scale,
                bootstrap_operator=softmax_bootstrap,
                refresh_before_exp=bool(refresh_softmax_exp_inputs),
                refresh_after_exp=bool(refresh_softmax_exp_outputs),
                token_valid_mask=token_valid_mask,
            )
            probabilities = softmax_result.probability_ciphers
            softmax_result.probability_ciphers = ()
            level_trace["probabilities"] = tuple(softmax_result.output_levels)
            # (slim pack period in Delta rows or 0, Alg2 slim refreshes that took the sparse program)
            level_trace["softmax_slim_sparse"] = (
                int(softmax_result.slim_period_rows or 0),
                int(softmax_result.slim_sparse_bootstraps),
            )
        sync_device(crypto_context.device)
        stage_seconds["softmax"] = time.perf_counter() - start
        if softmax_result is not None:
            stage_seconds.update(
                {
                    f"softmax.{name}": float(seconds)
                    for name, seconds in softmax_result.stage_seconds.items()
                }
            )
        _diagnostic_cipher_range(
            "probabilities", probabilities, crypto_context=crypto_context
        )
        for cipher in scores + scores_p + tuple(
            cipher for page in historical_scores for cipher in page
        ):
            release_if_supported(cipher)
        scores = ()
        scores_p = ()
        historical_scores = []

        if bool(pv_prefold_after_softmax):
            start = time.perf_counter()
            folded_probabilities = fold_probability_delta_halves_fhe(
                probabilities,
                layout=layout,
                crypto_context=crypto_context,
            )
            folded_values = fold_value_delta_halves_fhe(
                value_delta_ciphers,
                layout=layout,
                crypto_context=crypto_context,
            )
            sync_device(crypto_context.device)
            stage_seconds["pv_fold_inputs"] = time.perf_counter() - start

        start = time.perf_counter()
        pv_inputs_pre_folded = bool(pv_prefold_after_softmax)
        outputs = pv_delta_fhe(
            folded_probabilities if bool(pv_prefold_after_softmax) else probabilities,
            folded_values if bool(pv_prefold_after_softmax) else value_delta_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            operator_config=operator_config,
            inputs_pre_folded=pv_inputs_pre_folded,
        )
        for page_probabilities, page in zip(
            historical_probabilities, historical_delta_pages, strict=True
        ):
            page_folded_probabilities: tuple[object, ...] = ()
            page_folded_values: tuple[object, ...] = ()
            page_outputs: tuple[object, ...] = ()
            try:
                if bool(pv_prefold_after_softmax):
                    page_folded_probabilities = fold_probability_delta_halves_fhe(
                        page_probabilities,
                        layout=layout,
                        crypto_context=crypto_context,
                    )
                    page_folded_values = fold_value_delta_halves_fhe(
                        page.value_ciphers,
                        layout=layout,
                        crypto_context=crypto_context,
                    )
                page_outputs = pv_delta_fhe(
                    page_folded_probabilities
                    if bool(pv_prefold_after_softmax)
                    else page_probabilities,
                    page_folded_values
                    if bool(pv_prefold_after_softmax)
                    else page.value_ciphers,
                    layout=layout,
                    crypto_context=crypto_context,
                    operator_config=operator_config,
                    inputs_pre_folded=pv_inputs_pre_folded,
                )
                combined = _add_cipher_pairs(
                    outputs, page_outputs, crypto_context=crypto_context
                )
                release_all(outputs)
                outputs = combined
            finally:
                release_all(page_folded_probabilities)
                release_all(page_folded_values)
                release_all(page_outputs)
        sync_device(crypto_context.device)
        stage_seconds["pv"] = time.perf_counter() - start
        for cipher in folded_probabilities + folded_values + probabilities:
            release_if_supported(cipher)
        folded_probabilities = ()
        folded_values = ()
        probabilities = ()
        for page in historical_probabilities:
            release_all(page)
        historical_probabilities = ()
        if on_value_last_use is not None:
            on_value_last_use()
        value_delta_ciphers = ()
        if prefix_k:
            start = time.perf_counter()
            outputs_p = pv_prefix_pc_fhe(
                probabilities_p,
                prefix_bundle.value,
                layout=layout,
                crypto_context=crypto_context,
                prefix_k=prefix_k,
            )
            combined = _add_cipher_pairs(
                outputs, outputs_p, crypto_context=crypto_context
            )
            for cipher in outputs + outputs_p:
                release_if_supported(cipher)
            outputs = combined
            outputs_p = ()
            sync_device(crypto_context.device)
            stage_seconds["pv_prefix"] = time.perf_counter() - start
        for cipher in probabilities_p:
            release_if_supported(cipher)
        probabilities_p = ()
        if on_pv_last_use is not None:
            # Delta and optional Prefix PV are both complete.  Cross one final
            # device barrier before a caller evicts PV-only CUDA key material.
            synchronize_device(crypto_context.device)
            on_pv_last_use()
        _diagnostic_cipher_range(
            "output_c",
            outputs,
            crypto_context=crypto_context,
        )

        success = True
        return DeltaAttentionAlg2FHEResult(
            output_ciphers=outputs,
            stage_seconds={name: float(seconds) for name, seconds in stage_seconds.items()},
            level_trace={
                **level_trace,
                "output_c": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in outputs
                        }
                    )
                ),
            },
        )
    finally:
        if not success:
            # A kernel can fail after enqueueing work but before its ordinary
            # stage barrier.  Attempt the barrier before releasing any input
            # or temporary handle, while preserving the original exception if
            # synchronization itself also fails.
            try:
                synchronize_device(crypto_context.device)
            except Exception:
                pass
        if softmax_result is not None:
            softmax_result.release()
        for cipher in (
            scores
            + scores_p
            + tuple(cipher for page in historical_scores for cipher in page)
            + probabilities
            + probabilities_p
            + tuple(cipher for page in historical_probabilities for cipher in page)
            + folded_probabilities
            + folded_values
            + outputs_p
        ):
            release_if_supported(cipher)
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)


def attention_to_w_o_rotations(
    carrier_layout: LinearCarrierLayout,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    operator_config: AttentionOperatorConfig | None = None,
    log_n: int = 16,
    prefix_k: int = 0,
) -> tuple[int, ...]:
    """Exact rotation-key union excluding QKV and W_O linear projections."""

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    rotations = set(
        complex_token_attention_rotations(
            carrier_layout,
            attention_layout,
            sparse_max_baby_offsets=int(
                operator_config.sparse_max_baby_offsets
            ),
            shear_baby_steps=int(operator_config.shear_baby_steps),
            log_n=int(log_n),
        )
    )
    rotations.update(
        delta_attention_rotations(
            attention_layout,
            include_qk_feature_fold=bool(
                operator_config.qk_complex_feature_folding
            ),
            qk_key_features_pre_folded=bool(
                operator_config.qk_key_features_pre_folded
            ),
            qk_query_features_pre_folded=bool(
                operator_config.qk_query_features_pre_folded
            ),
            include_pv_delta_fold=bool(
                operator_config.pv_complex_delta_folding
            ),
            pv_probability_pre_folded=bool(
                operator_config.pv_prefold_after_softmax
            ),
            pv_value_pre_folded=bool(
                operator_config.pv_prefold_after_softmax
            ),
            log_n=int(log_n),
        )
    )
    rotations.update(delta_softmax_rotations(attention_layout))
    prefix_k = int(prefix_k)
    if prefix_k:
        from ..prefix.prefix_pc import prefix_pc_rotations

        rotations.update(prefix_pc_rotations(attention_layout, prefix_k=prefix_k))
        rotations.update(
            prefix_softmax_rotations(attention_layout, prefix_k=prefix_k)
        )
    rotations.update(
        pairfm_output_rotations(
            feature_layout,
            attention_layout,
            first_baby_steps=int(operator_config.pairfm_first_baby_steps),
            block_baby_steps=int(operator_config.pairfm_block_baby_steps),
            log_n=int(log_n),
        )
    )
    return tuple(sorted(int(value) for value in rotations if int(value)))


def attention_fhe_from_qkv_carriers(
    carrier_ciphers: tuple[object, ...] | list[object],
    *,
    carrier_layout: LinearCarrierLayout,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    softmax_config,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    layer_idx: int = 0,
    softmax_bootstrap=None,
    expected_query: np.ndarray | None = None,
    expected_key: np.ndarray | None = None,
    expected_value: np.ndarray | None = None,
    expected_output_c: np.ndarray | None = None,
    token_valid_mask: np.ndarray | None = None,
    function_prefix: str = "attention",
    prefix_bundle=None,
    historical_delta_pages: tuple[DeltaKVPage, ...] = (),
    emit_delta_kv_page: bool = False,
    direct_pairfold_v: bool = False,
    pairfold_occupied: int = 128,
    on_carriers_last_use: Callable[[], None] | None = None,
    on_qk_last_use: Callable[[], None] | None = None,
    on_pv_last_use: Callable[[], None] | None = None,
) -> AttentionToWOResult:
    """Run RoPE/layout, QK, Softmax, PV, and PairFM output conversion.

    ``carrier_ciphers`` remain caller-owned.  When ``on_carriers_last_use``
    is supplied, it is called only after the adapter has completed and an
    explicit device barrier has made releasing those caller-owned handles
    safe.  The returned eight ciphertexts are the exact complex token-pair
    feature-major boundary consumed by W_O.
    """

    carrier_ciphers = tuple(carrier_ciphers)
    carrier_levels = tuple(
        sorted(
            {
                int(crypto_context.level_for_cipher(cipher))
                for cipher in carrier_ciphers
            }
        )
    )
    model_config = Llama3CKKSConfig() if model_config is None else model_config
    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    expected_inputs = (expected_query, expected_key, expected_value)
    verify_inputs = any(value is not None for value in expected_inputs)
    if verify_inputs and not all(value is not None for value in expected_inputs):
        raise ValueError(
            "expected_query/key/value must either all be supplied or all omitted."
        )

    attention_inputs = None
    attention_core = None
    output = None
    delta_kv_page = None
    transferred: tuple[object, ...] = ()
    key_c: tuple[object, ...] = ()
    value_c: tuple[object, ...] = ()
    pairfold_v: tuple[object, ...] = ()
    success = False
    stages: dict[str, float] = {}
    try:
        start = time.perf_counter()
        attention_inputs = prepare_attention_inputs_fhe(
            carrier_ciphers,
            carrier_layout=carrier_layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            model_config=model_config,
            operator_config=operator_config,
            verify=bool(verify_inputs),
            expected_query=expected_query,
            expected_key=expected_key,
            expected_value=expected_value,
            function_prefix=f"{function_prefix}.input",
            direct_pairfold_v=bool(direct_pairfold_v),
            pairfold_occupied=int(pairfold_occupied),
        )
        if on_carriers_last_use is not None:
            synchronize_device(crypto_context.device)
            on_carriers_last_use()
            carrier_ciphers = ()
        stages["input"] = time.perf_counter() - start
        stages.update(
            {
                f"input.{name}": float(seconds)
                for name, seconds in attention_inputs.stage_seconds.items()
            }
        )
        _diagnostic_cipher_range(
            "query_c", attention_inputs.query_ciphers,
            crypto_context=crypto_context,
        )
        _diagnostic_cipher_range(
            "key_delta", attention_inputs.key_ciphers,
            crypto_context=crypto_context,
        )
        _diagnostic_cipher_range(
            "value_delta", attention_inputs.value_ciphers,
            crypto_context=crypto_context,
        )
        if emit_delta_kv_page:
            delta_kv_page = DeltaKVPage(
                key_ciphers=(),
                value_ciphers=(),
                token_valid_mask=(
                    None
                    if token_valid_mask is None
                    else np.asarray(token_valid_mask, dtype=bool).copy()
                ),
            )

        start = time.perf_counter()
        def release_qk_inputs() -> None:
            if delta_kv_page is not None:
                delta_kv_page.key_ciphers = attention_inputs.key_ciphers
                attention_inputs.key_ciphers = ()
            attention_inputs.release_qk_inputs()
            if on_qk_last_use is not None:
                on_qk_last_use()

        def release_value_input() -> None:
            if delta_kv_page is not None:
                delta_kv_page.value_ciphers = attention_inputs.value_ciphers
                attention_inputs.value_ciphers = ()
            attention_inputs.release_value_input()

        attention_core = attention_fhe_delta_alg2(
            attention_inputs.query_ciphers,
            attention_inputs.key_ciphers,
            attention_inputs.value_ciphers,
            layout=attention_layout,
            crypto_context=crypto_context,
            softmax_config=softmax_config,
            operator_config=operator_config,
            layer_idx=int(layer_idx),
            softmax_bootstrap=softmax_bootstrap,
            token_valid_mask=token_valid_mask,
            prefix_bundle=prefix_bundle,
            historical_delta_pages=historical_delta_pages,
            on_qk_last_use=release_qk_inputs,
            on_value_last_use=release_value_input,
            on_pv_last_use=on_pv_last_use,
        )
        if delta_kv_page is not None and (
            len(delta_kv_page.key_ciphers) != int(attention_layout.key_value_pair_count)
            or len(delta_kv_page.value_ciphers)
            != int(attention_layout.key_value_pair_count)
        ):
            raise RuntimeError("attention core did not return a complete Delta K/V page.")
        stages["qk_softmax_pv"] = time.perf_counter() - start
        stages.update(
            {
                f"qk_softmax_pv.{name}": float(seconds)
                for name, seconds in attention_core.stage_seconds.items()
            }
        )
        start = time.perf_counter()
        output = prepare_w_o_inputs_fhe(
            attention_core.output_ciphers,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            model_config=model_config,
            operator_config=operator_config,
            bootstrap_operator=softmax_bootstrap,
            verify=expected_output_c is not None,
            expected_output_c=expected_output_c,
            function_prefix=f"{function_prefix}.output",
        )
        stages["output"] = time.perf_counter() - start
        stages.update(
            {
                f"output.{name}": float(seconds)
                for name, seconds in output.stage_seconds.items()
            }
        )
        _diagnostic_cipher_range(
            "pairfm_w_o_input", output.carrier_ciphers,
            crypto_context=crypto_context,
        )

        release_all(output.real_ciphers)
        output.real_ciphers = ()
        transferred = output.carrier_ciphers
        output.carrier_ciphers = ()
        key_c = attention_inputs.key_c_ciphers
        value_c = attention_inputs.value_c_ciphers
        pairfold_v = attention_inputs.pairfold_value_ciphers
        attention_inputs.key_c_ciphers = ()
        attention_inputs.value_c_ciphers = ()
        attention_inputs.pairfold_value_ciphers = ()
        result = AttentionToWOResult(
            w_o_input_ciphers=transferred,
            stage_seconds={str(key): float(value) for key, value in stages.items()},
            input_scheduler_stats={
                str(key): float(value)
                for key, value in attention_inputs.scheduler_stats.items()
            },
            output_levels=tuple(
                sorted(
                    {
                        int(crypto_context.level_for_cipher(cipher))
                        for cipher in transferred
                    }
                )
            ),
            level_trace={
                "linear_carriers": carrier_levels,
                "query": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in attention_inputs.query_ciphers
                        }
                    )
                ),
                "key_delta": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in attention_inputs.key_ciphers
                        }
                    )
                ),
                "value_delta": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in attention_inputs.value_ciphers
                        }
                    )
                ),
                **attention_core.level_trace,
                "w_o_input": tuple(
                    sorted(
                        {
                            int(crypto_context.level_for_cipher(cipher))
                            for cipher in transferred
                        }
                    )
                ),
            },
            key_c_ciphers=key_c,
            value_c_ciphers=value_c,
            pairfold_value_ciphers=pairfold_v,
            delta_kv_page=delta_kv_page,
        )
        transferred = ()
        key_c = value_c = pairfold_v = ()
        delta_kv_page = None
        success = True
        return result
    finally:
        if output is not None:
            output.release()
        if attention_core is not None:
            attention_core.release()
        if attention_inputs is not None:
            attention_inputs.release()
        if not success:
            release_all(transferred)
            release_all(key_c)
            release_all(value_c)
            release_all(pairfold_v)
            if delta_kv_page is not None:
                delta_kv_page.release()


__all__ = [
    "AttentionToWOResult",
    "attention_fhe_from_qkv_carriers",
    "attention_to_w_o_rotations",
]
