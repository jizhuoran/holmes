from __future__ import annotations

import os

"""Transformer-layer definition and encrypted model execution."""

import time
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from . import graph
from .backend import plaintext_cache_snapshot, release_all
from .config import Llama3CKKSConfig
from .decode_packing import (
    DEFAULT_DECODE_PACKING,
    DecodePacking,
    validate_decode_packing,
)

# Re-exported: the layer contract types live beside evaluate() in
# graph.py, but callers have always imported them from here.
from .graph import LayerApproximations, LayerWeights
from .layouts.attention import AttentionPairLayout
from .layouts.feature_major import FeatureMajorPrefillLayout
from .operators.nonlinear.bootstrap import BootstrapConfig, BootstrapOperator
from .operators.attention.prefill.input import DeltaKVPage
from .operators.pairfold_kv_cache import PairFoldLayerCache
from .positions import validate_prefill_cache_page
from .schedule import LayerSchedule


def prepare_bootstrap_operator(crypto_context, schedule: LayerSchedule):
    """Prepare the one bootstrap program a run shares across layers.

    This is the single place a BootstrapConfig is built from a schedule;
    the application and the in-forward fallback both use it, so every
    bootstrap knob on LayerSchedule is honored on every path.
    """

    depth = int(crypto_context.context.L) - 1
    return BootstrapOperator.prepare(
        crypto_context,
        BootstrapConfig(
            log_slots=int(schedule.bootstrap_log_slots),
            level_budget=tuple(
                int(value) for value in schedule.bootstrap_level_budget
            ),
            output_levels=int(schedule.bootstrap_post_levels),
            iterations=int(schedule.softmax_bootstrap_iterations),
            precision_bits=int(schedule.softmax_bootstrap_precision),
            mode=os.environ.get("EASYFHE_BOOTSTRAP_MODE", "modraise_first"),
            evict_rotation_cache=bool(
                schedule.evict_rotation_cuda_cache_before_softmax_bootstrap
            ),
            deep_level_budget=(
                (3, 3) if schedule.softmax_main_deep_bootstrap else None
            ),
            deep_output_levels=(
                # Program depth = approx(10) + b1 + b2, so a (3,3) budget
                # fits two more output levels than (4,4) in the same chain.
                depth - 16 if schedule.softmax_main_deep_bootstrap else None
            ),
            # Decode sparse programs (period-1 slim RMSNorm row, merged-page
            # Softmax rows); empty on Prefill-only schedules.
            sparse_log_slots=(
                tuple(int(k) for k in schedule.bootstrap_sparse_log_slots) or None
            ),
        ),
    )


def with_decode_sparse_bootstrap(
    schedule: LayerSchedule,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout | None = None,
    *,
    slim_period_rows: int | None = 1,
    merged_page_softmax: bool = True,
) -> LayerSchedule:
    """``schedule`` with the Decode sparse bootstrap programs switched on.

    Sets :attr:`LayerSchedule.decode_rmsnorm_slim_period_rows` (period-1 slim
    row on the one DecodeFM cipher) and merges the programs
    :func:`llama3fhe.graph.decode_sparse_bootstrap_log_slots` derives
    (``(3, 12)`` for the production layouts) into
    :attr:`LayerSchedule.bootstrap_sparse_log_slots`, so
    :func:`prepare_bootstrap_operator` builds them and every rotation planner
    (Prefill and Decode) keys them.  Mirrors the multi-card bench
    (``rmsnorm_slim_period_rows=1`` + ``bootstrap_extra_sparse_log_slots``).
    """

    ks = set(int(k) for k in schedule.bootstrap_sparse_log_slots)
    ks.update(
        graph.decode_sparse_bootstrap_log_slots(
            feature_layout,
            attention_layout,
            slim_period_rows=slim_period_rows,
            merged_page_softmax=merged_page_softmax,
        )
    )
    return replace(
        schedule,
        decode_rmsnorm_slim_period_rows=(
            None if slim_period_rows is None else int(slim_period_rows)
        ),
        bootstrap_sparse_log_slots=tuple(sorted(ks)),
    )


@dataclass(frozen=True)
class TransformerLayer:
    """One indexed layer with no context, keygen, encryption or I/O effects."""

    index: int
    weights: LayerWeights
    approximations: LayerApproximations
    config: Llama3CKKSConfig
    feature_layout: FeatureMajorPrefillLayout = field(
        default_factory=FeatureMajorPrefillLayout
    )
    attention_layout: AttentionPairLayout = field(
        default_factory=AttentionPairLayout
    )
    rope_theta: float = 500000.0

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("layer index must be non-negative.")
        if not float(self.rope_theta) > 0:
            raise ValueError("rope_theta must be positive.")

    def forward(
        self,
        hidden_states: np.ndarray,
        *,
        crypto_context,
        input_ciphers: tuple[object, ...] | list[object],
        bootstrap_operator=None,
        schedule: LayerSchedule | None = None,
        refresh_input: bool = False,
        attention_token_valid_mask: np.ndarray | None = None,
        verify: bool = False,
        expected_output: np.ndarray | None = None,
        skip_attention: bool = False,
        emit_kv_cache: bool = False,
        prefix_bundle=None,
        historical_delta_pages: tuple[DeltaKVPage, ...] = (),
        emit_delta_kv_page: bool = False,
        content_offset: int = 0,
        decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
        prefill_occupied: int | None = None,
    ) -> graph.LayerResult:
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        expected_shape = (
            int(self.feature_layout.seq_len),
            int(self.feature_layout.hidden_dim),
        )
        if hidden_states.shape != expected_shape:
            raise ValueError(
                f"layer hidden-state stub must have shape {expected_shape}, "
                f"got {hidden_states.shape}."
            )
        return graph.evaluate(
            hidden_states,
            layer_idx=int(self.index),
            weights=self.weights,
            approximations=self.approximations,
            attention_layout=self.attention_layout,
            config=self.config,
            crypto_context=crypto_context,
            input_ciphers=input_ciphers,
            bootstrap_operator=bootstrap_operator,
            schedule=schedule,
            refresh_input=bool(refresh_input),
            attention_token_valid_mask=attention_token_valid_mask,
            rope_theta=float(self.rope_theta),
            verify=bool(verify),
            expected_output=expected_output,
            skip_attention=bool(skip_attention),
            emit_kv_cache=bool(emit_kv_cache),
            prefix_bundle=prefix_bundle,
            historical_delta_pages=historical_delta_pages,
            emit_delta_kv_page=bool(emit_delta_kv_page),
            content_offset=int(content_offset),
            decode_packing=decode_packing,
            prefill_occupied=prefill_occupied,
        )

    def forward_decode(
        self,
        hidden_states: np.ndarray,
        *,
        crypto_context,
        input_ciphers: tuple[object, ...] | list[object],
        bootstrap_operator=None,
        schedule: LayerSchedule | None = None,
        refresh_input: bool = False,
        skip_attention: bool = False,
        kv_cache_k_ciphers: tuple[object, ...] | list[object] | None = None,
        kv_cache_v_ciphers: tuple[object, ...] | list[object] | None = None,
        occupied: int = 0,
        position: int = 0,
        verify: bool = False,
        expected_output: np.ndarray | None = None,
        max_kv_pages: int = 1,
        prefix_bundle=None,
        consume_kv_cache: bool = False,
        decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
        pairfold_kv_cache: PairFoldLayerCache | None = None,
    ) -> graph.LayerResult:
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        expected_shape = (
            int(self.feature_layout.seq_len),
            int(self.feature_layout.hidden_dim),
        )
        if hidden_states.shape != expected_shape:
            raise ValueError(
                f"layer hidden-state stub must have shape {expected_shape}, "
                f"got {hidden_states.shape}."
            )
        return graph.evaluate_decode(
            hidden_states,
            layer_idx=int(self.index),
            weights=self.weights,
            approximations=self.approximations,
            feature_layout=self.feature_layout,
            attention_layout=self.attention_layout,
            config=self.config,
            crypto_context=crypto_context,
            input_ciphers=input_ciphers,
            bootstrap_operator=bootstrap_operator,
            schedule=schedule,
            refresh_input=bool(refresh_input),
            skip_attention=bool(skip_attention),
            kv_cache_k_ciphers=kv_cache_k_ciphers,
            kv_cache_v_ciphers=kv_cache_v_ciphers,
            occupied=int(occupied),
            position=int(position),
            rope_theta=float(self.rope_theta),
            verify=bool(verify),
            expected_output=expected_output,
            max_kv_pages=int(max_kv_pages),
            prefix_bundle=prefix_bundle,
            consume_kv_cache=bool(consume_kv_cache),
            decode_packing=decode_packing,
            pairfold_kv_cache=pairfold_kv_cache,
        )


@dataclass
class ModelResult:
    ciphers: tuple[object, ...]
    wall_seconds: float
    bootstrap_setup_seconds: float
    bootstrap_calls: int
    native_bootstrap_calls: int
    per_layer: list[dict[str, object]]
    kv_caches: list[tuple[tuple[object, ...], tuple[object, ...]]] = field(
        default_factory=list
    )
    pairfold_kv_caches: list[PairFoldLayerCache] = field(default_factory=list)
    delta_kv_pages: list[DeltaKVPage] = field(default_factory=list)

    def release(self) -> None:
        release_all(self.ciphers)
        self.ciphers = ()
        for key_ciphers, value_ciphers in self.kv_caches:
            release_all(key_ciphers)
            release_all(value_ciphers)
        self.kv_caches = []
        pairfold_kv_caches = self.pairfold_kv_caches
        self.pairfold_kv_caches = []
        for cache in pairfold_kv_caches:
            cache.release()
        delta_kv_pages = self.delta_kv_pages
        self.delta_kv_pages = []
        for page in delta_kv_pages:
            page.release()


@dataclass
class DecodeModelResult:
    """One next-token Decode step plus one selected per-layer KV format."""

    ciphers: tuple[object, ...]
    kv_caches: list[tuple[tuple[object, ...], tuple[object, ...]]]
    occupied: int
    position: int
    wall_seconds: float
    per_layer: list[dict[str, object]]
    pairfold_kv_caches: list[PairFoldLayerCache] = field(default_factory=list)

    def release(self) -> None:
        release_all(self.ciphers)
        self.ciphers = ()
        for key_ciphers, value_ciphers in self.kv_caches:
            release_all(key_ciphers)
            release_all(value_ciphers)
        self.kv_caches = []
        pairfold_kv_caches = self.pairfold_kv_caches
        self.pairfold_kv_caches = []
        for cache in pairfold_kv_caches:
            cache.release()


@dataclass(frozen=True)
class Llama3Model:
    """A fixed-size model that consumes a streaming iterable of layer objects."""

    config: Llama3CKKSConfig
    num_layers: int = 32
    feature_layout: FeatureMajorPrefillLayout = field(
        default_factory=FeatureMajorPrefillLayout
    )
    attention_layout: AttentionPairLayout = field(
        default_factory=AttentionPairLayout
    )

    def __post_init__(self) -> None:
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive.")

    def required_rotations(
        self,
        schedule: LayerSchedule | None = None,
        *,
        bootstrap_deep_level_budget: tuple[int, int] | None = None,
        emit_kv_cache: bool = False,
        prefix_k: int = 0,
        decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
    ) -> tuple[int, ...]:
        return graph.required_rotations(
            self.feature_layout,
            self.attention_layout,
            schedule=schedule,
            log_n=int(self.config.simulator.logN),
            bootstrap_deep_level_budget=bootstrap_deep_level_budget,
            emit_kv_cache=bool(emit_kv_cache),
            prefix_k=int(prefix_k),
            decode_packing=decode_packing,
        )

    def required_decode_rotations(
        self,
        schedule: LayerSchedule | None = None,
        *,
        skip_attention: bool = False,
        include_bootstrap: bool = False,
        decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
    ) -> tuple[int, ...]:
        return graph.decode_layer_rotations(
            self.feature_layout,
            self.attention_layout,
            schedule=schedule,
            skip_attention=bool(skip_attention),
            include_bootstrap=bool(include_bootstrap),
            log_n=int(self.config.simulator.logN),
            decode_packing=decode_packing,
        )

    def decode(
        self,
        hidden_states: np.ndarray,
        *,
        crypto_context,
        input_ciphers: tuple[object, ...] | list[object],
        layers: Iterable[TransformerLayer],
        schedule: LayerSchedule | None = None,
        bootstrap_operator: BootstrapOperator | None = None,
        refresh_first_layer: bool = False,
        refresh_post_attention: bool = False,
        skip_attention: bool = False,
        kv_caches: list[tuple[tuple[object, ...], tuple[object, ...]]] | None = None,
        occupied: int = 0,
        position: int = 0,
        verify: bool = False,
        expected_output: np.ndarray | None = None,
        progress_callback=None,
        max_kv_pages: int = 1,
        prefix_bundles=None,
        consume_kv_cache: bool = False,
        decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
        pairfold_kv_caches: list[PairFoldLayerCache | None] | None = None,
    ) -> DecodeModelResult:
        """One next-token step on DecodeFM with an explicit KV layout.

        Layer zero consumes the caller's encrypted token.  Later layers
        refresh the previous DecodeFM output into RMSN1, same as Prefill.
        ``refresh_post_attention`` prepares the residual Meta-BTS program so
        RMSN2 (and Decode Alg2 after exp) can run on the production depth-37
        tower.  ``kv_caches`` is the legacy C-form input and
        ``pairfold_kv_caches`` is its independent PairFold128 replacement;
        exactly the field selected by ``decode_packing`` may be supplied.
        ``consume_kv_cache=True`` transfers ownership of the caller's cache
        into the returned result and avoids cloning historical pages.  The
        default keeps the original non-consuming, deep-copy semantics.  The
        consuming mode is destructive after a layer accepts ownership.  A
        failure before that handoff restores the current PairFold source slot;
        an accepted cache is released by the failing layer graph instead.
        """

        decode_packing = validate_decode_packing(decode_packing)
        if int(self.feature_layout.seq_len) != 1:
            raise ValueError(
                "decode requires a DecodeFM feature layout with seq_len=1, "
                f"got seq_len={self.feature_layout.seq_len}."
            )
        schedule = LayerSchedule() if schedule is None else schedule
        layer_schedule = replace(schedule, qk_output_level=None)
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        expected_shape = (
            int(self.feature_layout.seq_len),
            int(self.feature_layout.hidden_dim),
        )
        if hidden_states.shape != expected_shape:
            raise ValueError(
                f"model hidden input must have shape {expected_shape}, "
                f"got {hidden_states.shape}."
            )
        current = tuple(input_ciphers)
        if len(current) != int(self.feature_layout.cipher_count):
            raise ValueError(
                f"model expects {self.feature_layout.cipher_count} input "
                f"ciphers, got {len(current)}."
            )
        occupied = int(occupied)
        position = int(position)
        skip_attention = bool(skip_attention)
        if occupied < 0 or position < 0:
            raise ValueError("occupied and position must be non-negative.")
        max_kv_pages = int(max_kv_pages)
        if max_kv_pages <= 0:
            raise ValueError("max_kv_pages must be positive.")
        if prefix_bundles is not None:
            prefix_bundles = tuple(prefix_bundles)
            if len(prefix_bundles) != int(self.num_layers):
                raise ValueError(
                    "prefix_bundles must have one entry per layer, got "
                    f"{len(prefix_bundles)} for {self.num_layers} layers."
                )
        if (
            not skip_attention
            and occupied >= int(self.attention_layout.seq_len) * max_kv_pages
        ):
            raise ValueError(
                "Decode KV pages are full: "
                f"occupied={occupied}, seq_len={self.attention_layout.seq_len}, "
                f"max_kv_pages={max_kv_pages}."
            )
        if decode_packing == "legacy_c":
            if pairfold_kv_caches is not None:
                raise ValueError(
                    "legacy_c Decode does not accept pairfold_kv_caches."
                )
            if occupied > 0 and not skip_attention:
                if kv_caches is None or len(kv_caches) != int(self.num_layers):
                    raise ValueError(
                        "occupied>0 requires kv_caches with one (K_C, V_C) pair "
                        "per layer, got "
                        f"{None if kv_caches is None else len(kv_caches)}."
                    )
                if bool(consume_kv_cache) and not isinstance(kv_caches, list):
                    raise ValueError(
                        "consume_kv_cache=True requires kv_caches to be a "
                        "mutable list."
                    )
        else:
            from .operators.attention.pairfold.pairfold_operator import (
                pairfold_prefix_token_count,
            )

            if kv_caches is not None:
                raise ValueError(
                    "pairfold128 Decode does not accept legacy kv_caches."
                )
            if skip_attention:
                if pairfold_kv_caches is not None:
                    raise ValueError(
                        "skip_attention=True does not accept PairFold KV caches."
                    )
            elif occupied > 0 and pairfold_kv_caches is None:
                raise ValueError(
                    "occupied>0 requires pairfold_kv_caches with one cache "
                    "per layer."
                )
            if pairfold_kv_caches is not None:
                if len(pairfold_kv_caches) != int(self.num_layers):
                    raise ValueError(
                        "pairfold_kv_caches must have one entry per layer, got "
                        f"{len(pairfold_kv_caches)} for {self.num_layers} layers."
                    )
                if bool(consume_kv_cache) and not isinstance(
                    pairfold_kv_caches, list
                ):
                    raise ValueError(
                        "consume_kv_cache=True requires pairfold_kv_caches to "
                        "be a mutable list."
                    )
                cache_ids: set[int] = set()
                cache_handle_layers: dict[int, int] = {}
                for layer_index, cache in enumerate(pairfold_kv_caches):
                    if not isinstance(cache, PairFoldLayerCache):
                        raise TypeError(
                            "pairfold_kv_caches entries must be live "
                            "PairFoldLayerCache objects; layer "
                            f"{layer_index} has {type(cache).__name__}."
                        )
                    if cache.is_released:
                        raise ValueError(
                            f"PairFold cache for layer {layer_index} is released."
                        )
                    cache._validate_invariants()
                    if id(cache) in cache_ids:
                        raise ValueError(
                            "pairfold_kv_caches must not alias across layers."
                        )
                    cache_ids.add(id(cache))
                    if int(cache.content_tokens) != occupied:
                        raise ValueError(
                            "PairFold cache occupancy mismatch at layer "
                            f"{layer_index}: occupied={occupied}, "
                            f"content_tokens={cache.content_tokens}."
                        )
                    for handle in cache.handles:
                        identity = id(handle)
                        previous_layer = cache_handle_layers.get(identity)
                        if previous_layer is not None:
                            raise ValueError(
                                "PairFold ciphertext handles must not alias "
                                "across layers: "
                                f"layers {previous_layer} and {layer_index}."
                            )
                        cache_handle_layers[identity] = layer_index
            if not skip_attention:
                prefix_counts = tuple(
                    pairfold_prefix_token_count(
                        None if prefix_bundles is None else bundle,
                        page_size=int(self.attention_layout.seq_len),
                        key_value_heads=int(
                            self.attention_layout.key_value_heads
                        ),
                        head_dim=int(self.attention_layout.head_dim),
                    )
                    for bundle in (
                        (None,) * int(self.num_layers)
                        if prefix_bundles is None
                        else prefix_bundles
                    )
                )
                if len(set(prefix_counts)) != 1:
                    raise ValueError(
                        "PairFold prefix_bundles must have the same verified "
                        f"token occupancy across layers, got {prefix_counts}."
                    )
                prefix_tokens = prefix_counts[0]
                expected_position = prefix_tokens + occupied
                if position != expected_position:
                    raise ValueError(
                        "PairFold Decode position must be contiguous with "
                        "Prefix and content occupancy: "
                        f"position={position}, prefix_tokens={prefix_tokens}, "
                        f"content_tokens={occupied}, "
                        f"expected={expected_position}."
                    )

        owns_current = False
        needs_bootstrap = (
            int(self.num_layers) > 1
            or bool(refresh_first_layer)
            or bool(refresh_post_attention)
        )
        owns_bootstrap = False
        iterator = iter(layers)
        per_layer: list[dict[str, object]] = []
        new_caches: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
        new_pairfold_caches: list[PairFoldLayerCache] = []
        total_start = time.perf_counter()
        shape_stub = np.zeros(expected_shape, dtype=np.float32)
        try:
            if needs_bootstrap:
                owns_bootstrap = bootstrap_operator is None
                if bootstrap_operator is None:
                    bootstrap_operator = prepare_bootstrap_operator(
                        crypto_context, schedule
                    )

            first_layer_index: int | None = None
            for expected_index in range(int(self.num_layers)):
                try:
                    layer = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"layer stream ended at {expected_index}; expected "
                        f"{self.num_layers} layers."
                    ) from exc
                if first_layer_index is None:
                    first_layer_index = int(layer.index)
                elif int(layer.index) != first_layer_index + expected_index:
                    raise ValueError(
                        "layer stream must be consecutive, expected index "
                        f"{first_layer_index + expected_index}, got "
                        f"{layer.index}."
                    )
                cache_k = cache_v = None
                pairfold_cache = None
                moved_pairfold_cache = None
                if not skip_attention:
                    if decode_packing == "legacy_c" and occupied > 0:
                        assert kv_caches is not None
                        cache_k, cache_v = kv_caches[expected_index]
                    elif (
                        decode_packing == "pairfold128"
                        and pairfold_kv_caches is not None
                    ):
                        pairfold_cache = pairfold_kv_caches[expected_index]
                        assert isinstance(pairfold_cache, PairFoldLayerCache)
                        if bool(consume_kv_cache):
                            moved_pairfold_cache = pairfold_cache
                            pairfold_kv_caches[expected_index] = None
                start = time.perf_counter()
                layer_result = None
                try:
                    is_last = expected_index + 1 == int(self.num_layers)
                    layer_result = layer.forward_decode(
                        hidden_states if expected_index == 0 else shape_stub,
                        crypto_context=crypto_context,
                        input_ciphers=current,
                        bootstrap_operator=bootstrap_operator,
                        schedule=layer_schedule,
                        refresh_input=(
                            expected_index > 0 or bool(refresh_first_layer)
                        ),
                        skip_attention=skip_attention,
                        kv_cache_k_ciphers=cache_k,
                        kv_cache_v_ciphers=cache_v,
                        occupied=occupied,
                        position=position,
                        verify=bool(verify) and is_last,
                        expected_output=(
                            expected_output if bool(verify) and is_last else None
                        ),
                        max_kv_pages=max_kv_pages,
                        prefix_bundle=(
                            None
                            if prefix_bundles is None
                            else prefix_bundles[expected_index]
                        ),
                        consume_kv_cache=bool(consume_kv_cache),
                        decode_packing=decode_packing,
                        pairfold_kv_cache=pairfold_cache,
                    )
                    compute_finished = time.perf_counter()
                    next_ciphers = tuple(layer_result.ciphers)
                    if len(next_ciphers) != int(
                        self.feature_layout.cipher_count
                    ):
                        raise RuntimeError(
                            f"layer {expected_index} returned {len(next_ciphers)} "
                            "ciphertexts; DecodeFM requires "
                            f"{self.feature_layout.cipher_count}."
                        )
                    if skip_attention:
                        if (
                            layer_result.key_c_ciphers
                            or layer_result.value_c_ciphers
                            or getattr(
                                layer_result, "pairfold_kv_cache", None
                            )
                            is not None
                        ):
                            raise RuntimeError(
                                "skip_attention Decode returned an unexpected KV cache."
                            )
                    elif decode_packing == "legacy_c":
                        if (
                            getattr(layer_result, "pairfold_kv_cache", None)
                            is not None
                        ):
                            raise RuntimeError(
                                "legacy_c Decode returned a PairFold KV cache."
                            )
                        next_k = tuple(layer_result.key_c_ciphers)
                        next_v = tuple(layer_result.value_c_ciphers)
                        layer_result.key_c_ciphers = ()
                        layer_result.value_c_ciphers = ()
                        new_caches.append((next_k, next_v))
                        if bool(consume_kv_cache) and kv_caches is not None:
                            kv_caches[expected_index] = ((), ())
                    else:
                        if (
                            layer_result.key_c_ciphers
                            or layer_result.value_c_ciphers
                        ):
                            raise RuntimeError(
                                "pairfold128 Decode returned legacy C-form caches."
                            )
                        next_pairfold_cache = layer_result.pairfold_kv_cache
                        if next_pairfold_cache is None:
                            raise RuntimeError(
                                "pairfold128 Decode returned no PairFold KV cache."
                            )
                        if next_pairfold_cache.is_released:
                            raise RuntimeError(
                                "pairfold128 Decode returned a released KV cache."
                            )
                        if int(next_pairfold_cache.content_tokens) != occupied + 1:
                            raise RuntimeError(
                                "pairfold128 Decode returned the wrong occupancy: "
                                f"expected {occupied + 1}, got "
                                f"{next_pairfold_cache.content_tokens}."
                            )
                        if moved_pairfold_cache is not None:
                            if next_pairfold_cache is not moved_pairfold_cache:
                                raise RuntimeError(
                                    "consume_kv_cache=True must return the moved "
                                    "PairFold cache object."
                                )
                        elif (
                            pairfold_cache is not None
                            and next_pairfold_cache is pairfold_cache
                        ):
                            # A non-consuming graph must never return the
                            # borrowed object as an owned result.  Detach it
                            # before raising so LayerResult.release cannot
                            # destroy caller ownership while reporting the
                            # contract violation.
                            layer_result.pairfold_kv_cache = None
                            raise RuntimeError(
                                "consume_kv_cache=False must return a cloned "
                                "PairFold cache object."
                            )
                        layer_result.pairfold_kv_cache = None
                        new_pairfold_caches.append(next_pairfold_cache)
                        moved_pairfold_cache = None

                    layer_result.ciphers = ()
                    previous = current
                    release_previous = owns_current
                    current = next_ciphers
                    owns_current = True
                    if release_previous:
                        release_all(previous)
                    per_layer.append(
                        {
                            "layer": expected_index,
                            "wall_seconds": float(compute_finished - start),
                            "kernel_wall_seconds": float(
                                layer_result.wall_seconds
                            ),
                            "stage_seconds": dict(layer_result.stage_seconds),
                            "nested_stage_seconds": dict(
                                layer_result.nested_stage_seconds
                            ),
                            "output_levels": list(layer_result.output_levels),
                            "max_abs_diff": layer_result.max_abs_diff,
                            "softmax_bootstrap_calls": int(
                                layer_result.softmax_bootstrap_calls
                            ),
                            "residual_bootstrap_calls": int(
                                layer_result.residual_bootstrap_calls
                            ),
                        }
                    )
                    if callable(progress_callback):
                        progress_callback(dict(per_layer[-1]))
                finally:
                    try:
                        if layer_result is not None:
                            layer_result.release()
                    finally:
                        if (
                            moved_pairfold_cache is not None
                            and not moved_pairfold_cache.is_released
                        ):
                            assert pairfold_kv_caches is not None
                            assert pairfold_kv_caches[expected_index] is None
                            pairfold_kv_caches[expected_index] = (
                                moved_pairfold_cache
                            )

            output = current
            owns_current = False
            result_caches = new_caches
            result_pairfold_caches = new_pairfold_caches
            new_caches = []
            new_pairfold_caches = []
            return DecodeModelResult(
                ciphers=output,
                kv_caches=result_caches,
                occupied=occupied + (0 if skip_attention else 1),
                position=position + 1,
                wall_seconds=float(time.perf_counter() - total_start),
                per_layer=per_layer,
                pairfold_kv_caches=result_pairfold_caches,
            )
        finally:
            close = getattr(iterator, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                try:
                    if owns_current:
                        release_all(current)
                finally:
                    for key_ciphers, value_ciphers in new_caches:
                        release_all(key_ciphers)
                        release_all(value_ciphers)
                    for cache in new_pairfold_caches:
                        cache.release()
                    if owns_bootstrap and bootstrap_operator is not None:
                        bootstrap_operator.release()

    def forward(
        self,
        hidden_states: np.ndarray,
        *,
        crypto_context,
        input_ciphers: tuple[object, ...] | list[object],
        layers: Iterable[TransformerLayer],
        bootstrap_operator: BootstrapOperator | None = None,
        schedule: LayerSchedule | None = None,
        refresh_first_layer: bool = False,
        attention_token_valid_mask: np.ndarray | None = None,
        progress_callback=None,
        diagnostic_callback=None,
        skip_attention: bool = False,
        emit_kv_cache: bool = False,
        prefix_bundles=None,
        historical_delta_pages=None,
        emit_delta_kv_page: bool = False,
        content_offset: int = 0,
        decode_packing: DecodePacking = DEFAULT_DECODE_PACKING,
        prefill_occupied: int | None = None,
    ) -> ModelResult:
        """Evaluate all layers without decrypting an inter-layer boundary.

        Layer zero consumes the caller's encrypted embedding.  Every later
        layer refreshes the previous FM output and fuses that refresh into its
        input RMSNorm.  The supplied layer iterable may therefore stream one
        checkpoint layer at a time.

        ``emit_kv_cache=True`` returns either legacy C-form pairs or independent
        PairFold128 caches according to ``decode_packing``.  The legacy layout
        remains the default, so ordinary Prefill memory use is unchanged.
        """

        decode_packing = validate_decode_packing(decode_packing)
        if prefill_occupied is not None:
            prefill_occupied = int(prefill_occupied)
        if decode_packing == "pairfold128" and bool(emit_kv_cache):
            validate_prefill_cache_page(
                occupied=prefill_occupied,
                token_valid_mask=attention_token_valid_mask,
                seq_len=int(self.feature_layout.seq_len),
            )
        if int(content_offset) < 0:
            raise ValueError("content_offset must be non-negative.")
        if skip_attention and emit_delta_kv_page:
            raise ValueError("emit_delta_kv_page requires the attention path.")
        schedule = LayerSchedule() if schedule is None else schedule
        # An absolute Layer0 alignment target cannot be reused after a
        # bootstrap has raised the modulus chain.
        layer_schedule = replace(schedule, qk_output_level=None)
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        expected_shape = (
            int(self.feature_layout.seq_len),
            int(self.feature_layout.hidden_dim),
        )
        if hidden_states.shape != expected_shape:
            raise ValueError(
                f"model hidden input must have shape {expected_shape}, "
                f"got {hidden_states.shape}."
            )
        current = tuple(input_ciphers)
        if len(current) != int(self.feature_layout.cipher_count):
            raise ValueError(
                f"model expects {self.feature_layout.cipher_count} input "
                f"ciphers, got {len(current)}."
            )

        owns_current = False
        owns_bootstrap = bootstrap_operator is None
        iterator = iter(layers)
        per_layer: list[dict[str, object]] = []
        emitted_caches: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
        emitted_pairfold_caches: list[PairFoldLayerCache] = []
        emitted_delta_pages: list[DeltaKVPage] = []
        setup_seconds = 0.0
        total_start = time.perf_counter()
        shape_stub = np.zeros(expected_shape, dtype=np.float32)
        emit_kv_cache = bool(emit_kv_cache)
        if prefix_bundles is not None:
            prefix_bundles = tuple(prefix_bundles)
            if len(prefix_bundles) != int(self.num_layers):
                raise ValueError(
                    "prefix_bundles must have one entry per layer, got "
                    f"{len(prefix_bundles)} for {self.num_layers} layers."
                )
        if historical_delta_pages is not None:
            historical_delta_pages = tuple(
                tuple(layer_pages) for layer_pages in historical_delta_pages
            )
            if len(historical_delta_pages) != int(self.num_layers):
                raise ValueError(
                    "historical_delta_pages must have one page sequence per layer, got "
                    f"{len(historical_delta_pages)} for {self.num_layers} layers."
                )
            expected_history_tokens: int | None = None
            for layer_index, layer_pages in enumerate(historical_delta_pages):
                history_tokens = 0
                for page_index, page in enumerate(layer_pages):
                    if not isinstance(page, DeltaKVPage):
                        raise TypeError(
                            f"historical_delta_pages[{layer_index}][{page_index}] "
                            "must be a DeltaKVPage."
                        )
                    mask = page.token_valid_mask
                    if mask is None:
                        history_tokens += int(self.feature_layout.seq_len)
                    else:
                        valid = np.asarray(mask, dtype=bool)
                        if valid.shape != (int(self.feature_layout.seq_len),):
                            raise ValueError(
                                f"historical Delta page mask must have shape "
                                f"({self.feature_layout.seq_len},), got {valid.shape}."
                            )
                        history_tokens += int(np.count_nonzero(valid))
                if expected_history_tokens is None:
                    expected_history_tokens = history_tokens
                elif history_tokens != expected_history_tokens:
                    raise ValueError(
                        "historical Delta pages must cover the same number of "
                        "tokens at every layer."
                    )
            if historical_delta_pages and historical_delta_pages[0] and int(
                content_offset
            ) != expected_history_tokens:
                raise ValueError(
                    "content_offset must equal the number of valid tokens in "
                    f"historical Delta pages ({expected_history_tokens})."
                )
            if prefix_bundles is not None and any(
                historical_delta_pages[index]
                and getattr(prefix_bundles[index], "prefix_k", 0)
                for index in range(int(self.num_layers))
            ):
                raise ValueError(
                    "plaintext Prefix and historical Delta pages are not yet supported together."
                )
        try:
            if bootstrap_operator is None:
                bootstrap_operator = prepare_bootstrap_operator(
                    crypto_context, schedule
                )
                setup_seconds = float(bootstrap_operator.setup_seconds)
            bootstrap_calls_before = int(bootstrap_operator.calls)
            native_bootstrap_calls_before = int(
                bootstrap_operator.native_calls
            )

            first_layer_index: int | None = None
            for expected_index in range(int(self.num_layers)):
                try:
                    layer = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"layer stream ended at {expected_index}; expected "
                        f"{self.num_layers} layers."
                    ) from exc
                if first_layer_index is None:
                    first_layer_index = int(layer.index)
                elif int(layer.index) != first_layer_index + expected_index:
                    raise ValueError(
                        "layer stream must be consecutive, expected index "
                        f"{first_layer_index + expected_index}, got "
                        f"{layer.index}."
                    )
                start = time.perf_counter()
                cache_before = plaintext_cache_snapshot(crypto_context)
                layer_result = None
                try:
                    layer_result = layer.forward(
                        hidden_states if expected_index == 0 else shape_stub,
                        crypto_context=crypto_context,
                        input_ciphers=current,
                        bootstrap_operator=bootstrap_operator,
                        schedule=layer_schedule,
                        refresh_input=(
                            expected_index > 0 or bool(refresh_first_layer)
                        ),
                        attention_token_valid_mask=attention_token_valid_mask,
                        verify=False,
                        skip_attention=bool(skip_attention),
                        emit_kv_cache=emit_kv_cache,
                        prefix_bundle=(
                            None
                            if prefix_bundles is None
                            else prefix_bundles[expected_index]
                        ),
                        historical_delta_pages=(
                            ()
                            if historical_delta_pages is None
                            else historical_delta_pages[expected_index]
                        ),
                        emit_delta_kv_page=bool(emit_delta_kv_page),
                        content_offset=int(content_offset),
                        decode_packing=decode_packing,
                        prefill_occupied=prefill_occupied,
                    )
                    compute_finished = time.perf_counter()
                    record_compute = getattr(
                        iterator, "record_compute_window", None
                    )
                    if callable(record_compute):
                        record_compute(expected_index, start, compute_finished)
                    next_ciphers = tuple(layer_result.ciphers)
                    if len(next_ciphers) != int(
                        self.feature_layout.cipher_count
                    ):
                        raise RuntimeError(
                            f"layer {expected_index} returned {len(next_ciphers)} "
                            "ciphertexts; the canonical feature-major boundary "
                            f"requires {self.feature_layout.cipher_count}."
                        )
                    if emit_kv_cache:
                        if decode_packing == "legacy_c":
                            if (
                                getattr(
                                    layer_result, "pairfold_kv_cache", None
                                )
                                is not None
                            ):
                                raise RuntimeError(
                                    "legacy_c Prefill returned a PairFold KV cache."
                                )
                            next_k = tuple(layer_result.key_c_ciphers)
                            next_v = tuple(layer_result.value_c_ciphers)
                            layer_result.key_c_ciphers = ()
                            layer_result.value_c_ciphers = ()
                            emitted_caches.append((next_k, next_v))
                        else:
                            if (
                                layer_result.key_c_ciphers
                                or layer_result.value_c_ciphers
                            ):
                                raise RuntimeError(
                                    "pairfold128 Prefill returned legacy C-form "
                                    "caches."
                                )
                            next_pairfold_cache = layer_result.pairfold_kv_cache
                            if next_pairfold_cache is None:
                                raise RuntimeError(
                                    "pairfold128 Prefill returned no PairFold KV "
                                    "cache."
                                )
                            expected_occupied = (
                                int(self.feature_layout.seq_len)
                                if prefill_occupied is None
                                else int(prefill_occupied)
                            )
                            if next_pairfold_cache.is_released:
                                raise RuntimeError(
                                    "pairfold128 Prefill returned a released KV "
                                    "cache."
                                )
                            if (
                                int(next_pairfold_cache.content_tokens)
                                != expected_occupied
                            ):
                                raise RuntimeError(
                                    "pairfold128 Prefill returned the wrong "
                                    "occupancy: expected "
                                    f"{expected_occupied}, got "
                                    f"{next_pairfold_cache.content_tokens}."
                                )
                            if any(
                                next_pairfold_cache is cache
                                for cache in emitted_pairfold_caches
                            ):
                                raise RuntimeError(
                                    "pairfold128 Prefill caches must not alias "
                                    "across layers."
                                )
                            layer_result.pairfold_kv_cache = None
                            emitted_pairfold_caches.append(next_pairfold_cache)
                    if emit_delta_kv_page:
                        next_delta_page = layer_result.delta_kv_page
                        if next_delta_page is None:
                            raise RuntimeError(
                                "Prefill layer returned no Delta K/V page."
                            )
                        layer_result.delta_kv_page = None
                        emitted_delta_pages.append(next_delta_page)
                    layer_result.ciphers = ()
                    previous = current
                    release_previous = owns_current
                    current = next_ciphers
                    owns_current = True
                    if release_previous:
                        release_all(previous)
                    cache_after = plaintext_cache_snapshot(crypto_context)
                    per_layer.append(
                        {
                            "layer": expected_index,
                            "wall_seconds": float(compute_finished - start),
                            "kernel_wall_seconds": float(
                                layer_result.wall_seconds
                            ),
                            "stage_seconds": dict(layer_result.stage_seconds),
                            "nested_stage_seconds": dict(
                                layer_result.nested_stage_seconds
                            ),
                            "output_levels": list(layer_result.output_levels),
                            "softmax_bootstrap_calls": int(
                                layer_result.softmax_bootstrap_calls
                            ),
                            "residual_bootstrap_calls": int(
                                layer_result.residual_bootstrap_calls
                            ),
                            "softmax_native_bootstrap_calls": int(
                                layer_result.softmax_native_bootstrap_calls
                            ),
                            "residual_native_bootstrap_calls": int(
                                layer_result.residual_native_bootstrap_calls
                            ),
                            "plaintext_cache": {
                                "before": cache_before,
                                "after": cache_after,
                                "delta": {
                                    name: int(cache_after[name])
                                    - int(cache_before[name])
                                    for name in (
                                        "entries",
                                        "bytes",
                                        "hits",
                                        "misses",
                                        "skips",
                                    )
                                },
                            },
                        }
                    )
                    if callable(diagnostic_callback):
                        per_layer[-1]["value_diagnostic"] = dict(
                            diagnostic_callback(expected_index, current, layer)
                        )
                    if callable(progress_callback):
                        progress_callback(dict(per_layer[-1]))
                finally:
                    if layer_result is not None:
                        layer_result.release()

            output = current
            owns_current = False
            result_caches = emitted_caches
            result_pairfold_caches = emitted_pairfold_caches
            result_delta_pages = emitted_delta_pages
            emitted_caches = []
            emitted_pairfold_caches = []
            emitted_delta_pages = []
            return ModelResult(
                ciphers=output,
                wall_seconds=float(time.perf_counter() - total_start),
                bootstrap_setup_seconds=float(setup_seconds),
                bootstrap_calls=(
                    int(bootstrap_operator.calls) - bootstrap_calls_before
                ),
                native_bootstrap_calls=(
                    int(bootstrap_operator.native_calls)
                    - native_bootstrap_calls_before
                ),
                per_layer=per_layer,
                kv_caches=result_caches,
                pairfold_kv_caches=result_pairfold_caches,
                delta_kv_pages=result_delta_pages,
            )
        finally:
            close = getattr(iterator, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                try:
                    if owns_current:
                        release_all(current)
                finally:
                    for key_ciphers, value_ciphers in emitted_caches:
                        release_all(key_ciphers)
                        release_all(value_ciphers)
                    for cache in emitted_pairfold_caches:
                        cache.release()
                    for page in emitted_delta_pages:
                        page.release()
                    if owns_bootstrap and bootstrap_operator is not None:
                        bootstrap_operator.release()


__all__ = [
    "DecodeModelResult",
    "LayerApproximations",
    "LayerSchedule",
    "LayerWeights",
    "Llama3Model",
    "ModelResult",
    "TransformerLayer",
    "prepare_bootstrap_operator",
    "with_decode_sparse_bootstrap",
]
