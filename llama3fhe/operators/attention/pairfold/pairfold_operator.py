from __future__ import annotations

"""End-to-end PairFold128 Decode attention core.

This module is the ownership boundary between the Decode graph and the stable
PairFold QK/PV kernels.  It deliberately stops at the single folded-O carrier:
the graph may borrow that ciphertext for a conditional refresh and folded
``W_O``, while :class:`PairFoldAttentionFHEResult` remains its owner.

The input carrier, K/V cache pages, and plaintext Prefix bundle are borrowed.
Every Q route, score page, probability page, and folded-O temporary created by
this operator is owned locally until it is either released or transferred in
the result.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import profile_sync_device as sync_device
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout

from ..decode import decode_softmax as _decode_softmax
from .pairfold_decode import (
    pv_pairfold_pages_fhe,
    qk_pairfold_pages_fhe,
    query_selection_head_rows,
    validate_pairfold_score_compatibility,
)


@dataclass
class PairFoldAttentionFHEResult:
    """Own exactly one folded-O ciphertext returned by PairFold attention."""

    folded_o_cipher: object | None
    wall_seconds: float
    stage_seconds: dict[str, float]
    profile: dict[str, float]
    score_occupied: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.folded_o_cipher is None:
            raise ValueError("PairFold attention result requires one folded-O cipher.")
        self.wall_seconds = float(self.wall_seconds)
        self.stage_seconds = {
            str(name): float(seconds) for name, seconds in self.stage_seconds.items()
        }
        self.profile = {str(name): float(value) for name, value in self.profile.items()}
        self.score_occupied = tuple(int(value) for value in self.score_occupied)

    @property
    def output_ciphers(self) -> tuple[object, ...]:
        """Tuple compatibility for borrowed-input folded-W_O call sites."""

        if self.folded_o_cipher is None:
            return ()
        return (self.folded_o_cipher,)

    def release(self) -> None:
        """Release the owned carrier once; repeated calls are harmless."""

        cipher = self.folded_o_cipher
        self.folded_o_cipher = None
        release_if_supported(cipher)


def _normalize_pages(
    pages: Iterable[Iterable[object]],
    *,
    expected_groups: int,
    name: str,
) -> tuple[tuple[object, ...], ...]:
    result = tuple(tuple(page) for page in pages)
    for page_index, page in enumerate(result):
        if len(page) != int(expected_groups):
            raise ValueError(
                f"PairFold {name} page {page_index} expects "
                f"{expected_groups} group ciphers, got {len(page)}."
            )
        if any(cipher is None for cipher in page):
            raise ValueError(
                f"PairFold {name} page {page_index} must not contain None."
            )
    return result


def _normalize_occupied(
    occupied_pages: Iterable[int],
    *,
    page_count: int,
    page_size: int,
) -> tuple[int, ...]:
    occupied = tuple(int(value) for value in occupied_pages)
    if len(occupied) != int(page_count):
        raise ValueError(
            "PairFold key/value/occupied page counts must match: "
            f"{page_count}/{page_count}/{len(occupied)}."
        )
    for page_index, value in enumerate(occupied):
        if not 1 <= value <= int(page_size):
            raise ValueError(
                f"occupied_pages[{page_index}]={value} outside [1, {page_size}]."
            )
    return occupied


def pairfold_prefix_token_count(
    prefix_bundle,
    *,
    page_size: int,
    key_value_heads: int,
    head_dim: int,
) -> int:
    """Return a Prefix bundle's verified logical token occupancy.

    PairFold Decode consumes the logical K/V payload, not the padded packed
    C page.  Consequently ``prefix_k`` is metadata rather than an independent
    source of truth: it must agree with both logical payload token axes.
    """

    if prefix_bundle is None:
        return 0
    if not hasattr(prefix_bundle, "prefix_k"):
        raise TypeError("prefix_bundle must provide prefix_k metadata.")
    if not hasattr(prefix_bundle, "key") or not hasattr(prefix_bundle, "value"):
        raise TypeError("prefix_bundle must provide logical key and value payloads.")
    prefix_k = int(prefix_bundle.prefix_k)
    if not 1 <= prefix_k <= int(page_size):
        raise ValueError(f"prefix_bundle.prefix_k={prefix_k} outside [1, {page_size}].")
    expected_tail = (int(key_value_heads), int(head_dim))
    expected_flat = int(key_value_heads) * int(head_dim)
    token_counts: dict[str, int] = {}
    for name in ("key", "value"):
        payload = np.asarray(getattr(prefix_bundle, name))
        if payload.ndim == 3 and tuple(payload.shape[1:]) == expected_tail:
            token_counts[name] = int(payload.shape[0])
        elif payload.ndim == 2 and int(payload.shape[1]) == expected_flat:
            token_counts[name] = int(payload.shape[0])
        else:
            raise ValueError(
                f"prefix_bundle.{name} must be logical "
                f"[tokens,{expected_tail[0]},{expected_tail[1]}] or "
                f"[tokens,{expected_flat}], got {payload.shape}."
            )
    if token_counts["key"] != token_counts["value"]:
        raise ValueError(
            "prefix_bundle logical K/V token counts differ: "
            f"{token_counts['key']} != {token_counts['value']}."
        )
    if token_counts["key"] != prefix_k:
        raise ValueError(
            "prefix_bundle.prefix_k must equal its logical payload occupancy: "
            f"prefix_k={prefix_k}, tokens={token_counts['key']}."
        )
    return prefix_k


def _prefix_payload(prefix_bundle, *, layout: PairFold128Layout):
    prefix_k = pairfold_prefix_token_count(
        prefix_bundle,
        page_size=int(layout.seq_len),
        key_value_heads=int(layout.key_value_heads),
        head_dim=int(layout.head_dim),
    )
    if prefix_bundle is None:
        return 0, None, None
    return prefix_k, prefix_bundle.key, prefix_bundle.value


def _borrowed_cipher_ids(
    carrier_cipher,
    key_pages: tuple[tuple[object, ...], ...],
    value_pages: tuple[tuple[object, ...], ...],
) -> set[int]:
    return {
        id(cipher)
        for cipher in (
            carrier_cipher,
            *(cipher for page in key_pages for cipher in page),
            *(cipher for page in value_pages for cipher in page),
        )
    }


def _release_values(
    values: Iterable[object],
    *,
    borrowed_ids: set[int],
    released_ids: set[int],
) -> None:
    first_error: BaseException | None = None
    for value in tuple(values):
        if value is None:
            continue
        identity = id(value)
        if identity in borrowed_ids or identity in released_ids:
            continue
        # Mark before calling the backend so an exceptional release is never
        # retried by the outer cleanup path.
        released_ids.add(identity)
        try:
            release_if_supported(value)
        except BaseException as error:  # cleanup must continue for all pages
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _release_pages(
    pages: Iterable[Iterable[object]],
    *,
    borrowed_ids: set[int],
    released_ids: set[int],
) -> None:
    _release_values(
        (cipher for page in tuple(pages) for cipher in tuple(page)),
        borrowed_ids=borrowed_ids,
        released_ids=released_ids,
    )


def _validate_single_cipher_pages(
    pages: Iterable[Iterable[object]],
    *,
    expected_pages: int,
    forbidden_ids: set[int],
    name: str,
) -> tuple[tuple[object], ...]:
    normalized = tuple(tuple(page) for page in pages)
    if len(normalized) != int(expected_pages):
        raise RuntimeError(
            f"{name} returned {len(normalized)} pages, expected {expected_pages}."
        )
    handles: list[object] = []
    for page_index, page in enumerate(normalized):
        if len(page) != 1 or page[0] is None:
            raise RuntimeError(
                f"{name} page {page_index} must contain exactly one cipher."
            )
        handles.append(page[0])
    if len({id(cipher) for cipher in handles}) != len(handles):
        raise RuntimeError(f"{name} pages must own distinct ciphertexts.")
    if any(id(cipher) in forbidden_ids for cipher in handles):
        raise RuntimeError(f"{name} output aliases a forbidden ciphertext.")
    return normalized  # type: ignore[return-value]


def _sync_if_available(crypto_context) -> None:
    device = getattr(crypto_context, "device", None)
    if device is not None:
        sync_device(device)


def _refresh_probability_pages_owned(
    probability_pages: tuple[tuple[object, ...], ...],
    *,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    borrowed_ids: set[int],
    released_ids: set[int],
) -> tuple[tuple[object, ...], ...]:
    """Consume an owned page tuple and return the refreshed ownership set."""

    pending = [tuple(page) for page in probability_pages]
    refreshed: list[tuple[object, ...]] = []
    success = False
    try:
        for page_index, page in enumerate(pending):
            output = _decode_softmax._refresh_row_ciphers_if_short(
                page,
                bootstrap_operator=bootstrap_operator,
                need_limbs=6,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
            )
            # The refresh helper consumes ``page`` only when it actually
            # refreshes.  In both branches ownership is transferred to output.
            pending[page_index] = ()
            refreshed.append(tuple(output))
        result = tuple(refreshed)
        refreshed = []
        success = True
        return result
    finally:
        if not success:
            _release_pages(
                pending,
                borrowed_ids=borrowed_ids,
                released_ids=released_ids,
            )
            _release_pages(
                refreshed,
                borrowed_ids=borrowed_ids,
                released_ids=released_ids,
            )


def _copy_profile(
    destination: dict[str, float], prefix: str, source: dict[str, float]
) -> None:
    destination.update(
        {f"{prefix}.{name}": float(value) for name, value in source.items()}
    )


def pairfold_attention_fhe(
    carrier_cipher,
    key_pages: Iterable[Iterable[object]],
    value_pages: Iterable[Iterable[object]],
    *,
    query_extractor: Callable[..., object],
    pairfold_layout: PairFold128Layout,
    score_layout: AttentionPairLayout,
    crypto_context,
    softmax_config,
    occupied_pages: Iterable[int],
    layer_idx: int = 0,
    rounds: int | None = None,
    score_scale: float | None = None,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    prefix_bundle=None,
    query_selection: Iterable[tuple[int, int]] | None = None,
    query_group_extractor: Callable[..., Iterable[object]] | None = None,
) -> PairFoldAttentionFHEResult:
    """Run streaming PairFold QK -> existing Decode Alg2 -> PairFold PV.

    ``carrier_cipher``, every K/V cache ciphertext, and ``prefix_bundle`` are
    borrowed.  Prefix scores occupy page zero; content page order is unchanged.
    The returned result owns one folded-O ciphertext.  Bootstrap iteration and
    precision overrides are passed unchanged to both existing Softmax and the
    legacy-equivalent post-Softmax ``need_limbs=6`` probability refresh.

    ``qk_prefix``/``qk_content`` and ``pv_prefix``/``pv_content`` are direct
    P4 profiling probes.  Shared work remains in the corresponding ``qk`` or
    ``pv`` total and is not estimated or double-assigned to a partition.

    ``query_selection`` (multi-card decode) restricts QK and PV to this rank's
    ``(group, query_offset)`` streams; the softmax then runs on score pages
    whose other segments are zero (segments are independent under the
    token-axis reductions) and the returned folded-O is this rank's partial.

    ``query_group_extractor`` is used by serial Decode to extract all query
    offsets of one GQA group with a shared sparse baby basis.  It is ignored
    when ``query_selection`` selects a multi-card subset.
    """

    if carrier_cipher is None:
        raise ValueError("carrier_cipher must not be None.")
    if not callable(query_extractor):
        raise TypeError("query_extractor must be callable.")
    validate_pairfold_score_compatibility(pairfold_layout, score_layout)

    groups = int(pairfold_layout.key_value_groups)
    normalized_keys = _normalize_pages(key_pages, expected_groups=groups, name="K")
    normalized_values = _normalize_pages(value_pages, expected_groups=groups, name="V")
    if len(normalized_keys) != len(normalized_values):
        raise ValueError(
            "PairFold key/value page counts must match: "
            f"{len(normalized_keys)} != {len(normalized_values)}."
        )
    occupied = _normalize_occupied(
        occupied_pages,
        page_count=len(normalized_keys),
        page_size=int(pairfold_layout.seq_len),
    )
    prefix_k, prefix_key, prefix_value = _prefix_payload(
        prefix_bundle, layout=pairfold_layout
    )
    score_occupied = ((prefix_k,) if prefix_k else ()) + occupied

    borrowed_ids = _borrowed_cipher_ids(
        carrier_cipher, normalized_keys, normalized_values
    )
    released_ids: set[int] = set()
    score_pages: tuple[tuple[object, ...], ...] = ()
    probability_pages: tuple[tuple[object, ...], ...] = ()
    folded_outputs: tuple[object, ...] = ()
    folded_o = None
    qk_profile: dict[str, float] = {}
    pv_profile: dict[str, float] = {}
    stage_seconds = {
        "q_extract_and_broadcast": 0.0,
        "qk_prefix": 0.0,
        "qk_content": 0.0,
        "qk": 0.0,
        "softmax": 0.0,
        "probability_refresh": 0.0,
        "pv_prefix": 0.0,
        "pv_content": 0.0,
        "pv": 0.0,
        "folded_o_assemble": 0.0,
        "wall": 0.0,
    }
    profile: dict[str, float] = {
        "content_pages": float(len(normalized_keys)),
        "prefix_pages": float(bool(prefix_k)),
        "score_pages": float(len(score_occupied)),
    }

    # Only forwarded when set so the default call is unchanged.
    stream_kwargs = (
        {} if query_selection is None else {"query_selection": tuple(query_selection)}
    )
    softmax_kwargs: dict[str, object] = {}
    if stream_kwargs:
        profile["query_streams"] = float(len(stream_kwargs["query_selection"]))
        # The other streams' head rows are exactly zero on this rank's score
        # pages; the softmax must treat them as invalid, not as real rows.
        softmax_kwargs["head_rows"] = query_selection_head_rows(
            stream_kwargs["query_selection"], layout=pairfold_layout
        )

    _sync_if_available(crypto_context)
    wall_started = time.perf_counter()
    try:
        qk_started = time.perf_counter()
        qk_kwargs = {
            "query_extractor": query_extractor,
            "layout": pairfold_layout,
            "crypto_context": crypto_context,
            "occupied_pages": occupied,
            "key_prefix": prefix_key,
            "prefix_k": prefix_k or None,
            "profile": qk_profile,
            **stream_kwargs,
        }
        if query_group_extractor is not None:
            qk_kwargs["query_group_extractor"] = query_group_extractor
        score_pages = qk_pairfold_pages_fhe(
            carrier_cipher,
            normalized_keys,
            **qk_kwargs,
        )
        _sync_if_available(crypto_context)
        stage_seconds["qk"] = time.perf_counter() - qk_started
        score_pages = _validate_single_cipher_pages(
            score_pages,
            expected_pages=len(score_occupied),
            forbidden_ids=borrowed_ids,
            name="PairFold QK",
        )
        stage_seconds["q_extract_and_broadcast"] = float(
            qk_profile.get("query_extract", 0.0)
            + qk_profile.get("query_broadcast", 0.0)
        )
        stage_seconds["qk_prefix"] = float(qk_profile.get("prefix_compute", 0.0))
        stage_seconds["qk_content"] = float(qk_profile.get("content_compute", 0.0))

        softmax_started = time.perf_counter()
        probability_pages = _decode_softmax.softmax_alg2_row_pages_fhe(
            score_pages,
            layout=score_layout,
            crypto_context=crypto_context,
            config=softmax_config,
            occupied_pages=score_occupied,
            layer_idx=int(layer_idx),
            rounds=rounds,
            score_scale=score_scale,
            bootstrap_operator=bootstrap_operator,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_precision_bits=bootstrap_precision_bits,
            **softmax_kwargs,
        )
        _sync_if_available(crypto_context)
        stage_seconds["softmax"] = time.perf_counter() - softmax_started
        probability_pages = _validate_single_cipher_pages(
            probability_pages,
            expected_pages=len(score_occupied),
            forbidden_ids=borrowed_ids
            | {id(cipher) for page in score_pages for cipher in page},
            name="Decode Alg2 Softmax",
        )

        _release_pages(
            score_pages,
            borrowed_ids=borrowed_ids,
            released_ids=released_ids,
        )
        score_pages = ()

        refresh_started = time.perf_counter()
        owned_before_refresh = probability_pages
        probability_pages = ()
        probability_pages = _refresh_probability_pages_owned(
            owned_before_refresh,
            bootstrap_operator=bootstrap_operator,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_precision_bits=bootstrap_precision_bits,
            borrowed_ids=borrowed_ids,
            released_ids=released_ids,
        )
        probability_pages = _validate_single_cipher_pages(
            probability_pages,
            expected_pages=len(score_occupied),
            forbidden_ids=borrowed_ids,
            name="post-Softmax probability refresh",
        )
        _sync_if_available(crypto_context)
        stage_seconds["probability_refresh"] = time.perf_counter() - refresh_started

        content_offset = int(bool(prefix_k))
        prefix_probabilities = probability_pages[0] if prefix_k else None
        content_probabilities = probability_pages[content_offset:]
        pv_started = time.perf_counter()
        folded_outputs = tuple(
            pv_pairfold_pages_fhe(
                content_probabilities,
                normalized_values,
                layout=pairfold_layout,
                crypto_context=crypto_context,
                occupied_pages=occupied,
                prefix_probability_ciphers=prefix_probabilities,
                value_prefix=prefix_value,
                prefix_k=prefix_k or None,
                profile=pv_profile,
                **stream_kwargs,
            )
        )
        _sync_if_available(crypto_context)
        stage_seconds["pv"] = time.perf_counter() - pv_started
        if len(folded_outputs) != 1 or folded_outputs[0] is None:
            raise RuntimeError(
                "PairFold PV must return exactly one folded-O ciphertext."
            )
        folded_o = folded_outputs[0]
        if id(folded_o) in borrowed_ids:
            raise RuntimeError("PairFold PV output must not alias a borrowed input.")
        if any(folded_o is cipher for page in probability_pages for cipher in page):
            raise RuntimeError("PairFold PV output must not alias a probability page.")
        stage_seconds["pv_prefix"] = float(pv_profile.get("prefix_compute", 0.0))
        stage_seconds["pv_content"] = float(pv_profile.get("content_compute", 0.0))
        stage_seconds["folded_o_assemble"] = float(
            pv_profile.get("output_route", 0.0)
            + pv_profile.get("folded_o_accumulate", 0.0)
        )

        _release_pages(
            probability_pages,
            borrowed_ids=borrowed_ids,
            released_ids=released_ids,
        )
        probability_pages = ()

        stage_seconds["wall"] = time.perf_counter() - wall_started
        _copy_profile(profile, "qk", qk_profile)
        _copy_profile(profile, "pv", pv_profile)
        _copy_profile(profile, "stage", stage_seconds)
        result = PairFoldAttentionFHEResult(
            folded_o_cipher=folded_o,
            wall_seconds=stage_seconds["wall"],
            stage_seconds=stage_seconds,
            profile=profile,
            score_occupied=score_occupied,
        )
        folded_outputs = ()
        folded_o = None
        return result
    finally:
        cleanup_error: BaseException | None = None
        for pages in (score_pages, probability_pages):
            try:
                _release_pages(
                    pages,
                    borrowed_ids=borrowed_ids,
                    released_ids=released_ids,
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        try:
            _release_values(
                folded_outputs + (folded_o,),
                borrowed_ids=borrowed_ids,
                released_ids=released_ids,
            )
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error


__all__ = [
    "PairFoldAttentionFHEResult",
    "pairfold_attention_fhe",
    "pairfold_prefix_token_count",
]
