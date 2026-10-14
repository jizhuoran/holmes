from __future__ import annotations

"""Decode-native PairFold128 QK/PV kernels.

The kernels in this module deliberately stop at the two stable boundaries of
the Decode attention graph:

* :func:`qk_pairfold_pages_fhe` returns one legacy-compatible compact score
  ciphertext per 128-token page.  The existing Decode Alg2 implementation can
  consume those pages without a packing conversion.
* :func:`pv_pairfold_pages_fhe` consumes the compact probability pages and
  returns one folded-O carrier, ready for the PairFold W_O kernel.

Inputs are borrowed and outputs are owned by the caller.  All ciphertexts
allocated by a failed call are released before the exception is propagated.
The query contract is the output of the PairFold carrier route: one clean
token-column-zero ciphertext for each ``(KV group, GQA query offset)``.
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout

from ...primitives import (
    accumulate_owned,
    encode_mask,
    mul_cipher_rescale,
    mul_plain_rescale,
)
from ..decode.decode import _rotate_add_tree
from ..decode.decode_softmax import _align_cipher_pair, _sum_aligned_ciphers


@dataclass(frozen=True)
class PairFoldDecodeComplexity:
    """Operation counts that distinguish the new schedule from legacy C-form.

    ``prefix_plaintext`` adds one Prefix page evaluated with CTxPT.  The
    content counts never include Prefix work, which keeps warm profiles easy
    to compare with ciphertext-cache-only runs.
    """

    content_pages: int
    prefix_plaintext: bool
    qk_cipher_multiplications: int
    pv_cipher_multiplications: int
    prefix_plain_multiplications: int
    query_broadcasts: int
    query_peak_live_ciphers: int
    qk_feature_reductions: int
    pv_feature_broadcasts: int
    pv_token_reductions: int
    qk_conjugations: int


def pairfold_decode_complexity(
    content_pages: int, *, prefix_plaintext: bool = False
) -> PairFoldDecodeComplexity:
    """Return the core arithmetic count for a PairFold Decode attention."""

    content_pages = int(content_pages)
    if content_pages < 0:
        raise ValueError(
            f"content_pages must be non-negative, got {content_pages}."
        )
    ways = 8
    prefix = int(bool(prefix_plaintext))
    return PairFoldDecodeComplexity(
        content_pages=content_pages,
        prefix_plaintext=bool(prefix_plaintext),
        qk_cipher_multiplications=ways * content_pages,
        pv_cipher_multiplications=ways * content_pages,
        # Eight QxK and eight PxV plaintext products.
        prefix_plain_multiplications=2 * ways * prefix,
        # Each Q is broadcast once and reused by Prefix and all content pages.
        query_broadcasts=ways,
        # The public complexity summary keeps the legacy one-stream contract.
        # The optional serial grouped route reports its larger live set in the
        # runtime profile (``query_peak_live_ciphers``).
        query_peak_live_ciphers=1,
        qk_feature_reductions=ways * (content_pages + prefix),
        pv_feature_broadcasts=ways * (content_pages + prefix),
        # Products are accumulated across pages before this reduction.
        pv_token_reductions=ways,
        qk_conjugations=ways * (content_pages + prefix),
    )


def qk_pairfold_rotations(layout: PairFold128Layout) -> tuple[int, ...]:
    """Exact rotation-key union for PairFold content/Prefix QK."""

    rotations: list[int] = []
    rotations.extend(-step for step in layout.token_rotations())
    rotations.extend(layout.feature_rotations())
    # CKKS ring M=4*slots at logN=16; EasyFHE represents conjugation as M-1.
    rotations.append(4 * int(layout.slots) - 1)
    for group in range(int(layout.key_value_groups)):
        for pair in range(int(layout.pair_blocks)):
            for query_offset in range(int(layout.gqa_ratio)):
                rotation = layout.page_to_score_rotation(
                    group, pair, query_offset
                )
                if rotation:
                    rotations.append(rotation)
    return tuple(dict.fromkeys(rotations))


def pv_pairfold_rotations(layout: PairFold128Layout) -> tuple[int, ...]:
    """Exact rotation-key union for PairFold content/Prefix PV."""

    rotations: list[int] = []
    for group in range(int(layout.key_value_groups)):
        for pair in range(int(layout.pair_blocks)):
            for query_offset in range(int(layout.gqa_ratio)):
                rotation = layout.score_to_page_rotation(
                    group, pair, query_offset
                )
                if rotation:
                    rotations.append(rotation)
    rotations.extend(-step for step in layout.feature_rotations())
    rotations.extend(layout.token_rotations())
    for group in range(int(layout.key_value_groups)):
        for query_offset in range(int(layout.gqa_ratio)):
            for lane in range(int(layout.head_lanes)):
                rotation = layout.page_to_output_rotation(
                    group, query_offset, lane
                )
                if rotation:
                    rotations.append(rotation)
    return tuple(dict.fromkeys(rotations))


def pv_probability_route_rotations(layout: PairFold128Layout) -> tuple[int, ...]:
    """Rotations applied to the high-limb probability before the PV product."""

    rotations = [
        layout.score_to_page_rotation(group, pair, query_offset)
        for group in range(int(layout.key_value_groups))
        for pair in range(int(layout.pair_blocks))
        for query_offset in range(int(layout.gqa_ratio))
    ]
    rotations.extend(-step for step in layout.feature_rotations())
    return tuple(dict.fromkeys(rotation for rotation in rotations if rotation))


def pairfold_decode_rotations(layout: PairFold128Layout) -> tuple[int, ...]:
    """Rotation-key union for QK, existing compact Softmax, and PV.

    The compact Softmax uses exactly the same token stride as PairFold128, so
    its ``+/- token_rotations`` are already present in this union.
    """

    from ..decode.decode_softmax import softmax_page_merge_rotations

    return tuple(
        dict.fromkeys(
            qk_pairfold_rotations(layout)
            + pv_pairfold_rotations(layout)
            + softmax_page_merge_rotations(layout)
        )
    )


def validate_pairfold_score_compatibility(
    layout: PairFold128Layout,
    score_layout: AttentionPairLayout,
) -> None:
    """Fail if PairFold scores are not slot-identical to compact Decode C."""

    fields = ("seq_len", "head_dim", "query_heads", "key_value_heads", "slots")
    for field in fields:
        actual = int(getattr(layout, field))
        expected = int(getattr(score_layout, field))
        if actual != expected:
            raise ValueError(
                f"PairFold/compact-score {field} mismatch: {actual} != {expected}."
            )
    probe = np.arange(
        int(layout.query_heads) * int(layout.seq_len), dtype=np.float64
    ).reshape(int(layout.query_heads), int(layout.seq_len))
    if not np.array_equal(
        layout.pack_scores(probe),
        score_layout.pack_compact_decode_scores(probe),
    ):
        raise ValueError("PairFold scores are not compact Decode slot-compatible.")


def _profile_add(profile: dict[str, float] | None, name: str, value: float) -> None:
    if profile is not None:
        profile[name] = float(profile.get(name, 0.0)) + float(value)


def _timed(
    profile: dict[str, float] | None,
    name: str,
    function: Callable[[], object],
):
    started = time.perf_counter()
    result = function()
    _profile_add(profile, name, time.perf_counter() - started)
    return result


def _release_all(values: Iterable[object | None]) -> None:
    for value in values:
        release_if_supported(value)


def _require_query_ciphers(
    query_ciphers: Iterable[object], *, layout: PairFold128Layout
) -> tuple[object, ...]:
    values = tuple(query_ciphers)
    if len(values) != int(layout.query_residue_count):
        raise ValueError(
            f"PairFold QK expects {layout.query_residue_count} Q ciphers, "
            f"got {len(values)}."
        )
    return values


def _require_pages(
    pages: Iterable[Iterable[object]],
    *,
    layout: PairFold128Layout,
    name: str,
) -> tuple[tuple[object, ...], ...]:
    result = tuple(tuple(page) for page in pages)
    expected = int(layout.key_value_groups)
    for index, page in enumerate(result):
        if len(page) != expected:
            raise ValueError(
                f"PairFold {name} page {index} expects {expected} group "
                f"ciphers, got {len(page)}."
            )
    return result


def _require_occupied_pages(
    occupied_pages: Iterable[int],
    *,
    page_count: int,
    layout: PairFold128Layout,
) -> tuple[int, ...]:
    occupied = tuple(int(value) for value in occupied_pages)
    if len(occupied) != int(page_count):
        raise ValueError(
            "PairFold pages and occupied_pages must have the same length."
        )
    for index, value in enumerate(occupied):
        if not 0 <= value <= int(layout.seq_len):
            raise ValueError(
                f"occupied_pages[{index}]={value} outside [0, {layout.seq_len}]."
            )
    return occupied


def _require_probability_pages(
    probability_pages: Iterable[Iterable[object]],
) -> tuple[tuple[object], ...]:
    result: list[tuple[object]] = []
    for index, page in enumerate(probability_pages):
        page = tuple(page)
        if len(page) != 1:
            raise ValueError(
                f"PairFold probability page {index} expects 1 compact cipher, "
                f"got {len(page)}."
            )
        result.append((page[0],))
    return tuple(result)


def _rotate(cipher, rotation: int, *, crypto_context):
    rotation = int(rotation)
    if rotation == 0:
        return cipher.deep_copy()
    return crypto_context.fhe.homo_rotate(
        cipher, rotation, crypto_context.context
    )


def _mul_cipher_aligned(left, right, *, crypto_context):
    aligned_left, aligned_right, owned_left, owned_right = _align_cipher_pair(
        left, right, crypto_context=crypto_context
    )
    try:
        return mul_cipher_rescale(
            aligned_left, aligned_right, crypto_context=crypto_context
        )
    finally:
        if owned_left:
            release_if_supported(aligned_left)
        if owned_right:
            release_if_supported(aligned_right)


def _sum_borrowed_aligned(ciphers: Iterable[object], *, crypto_context):
    ciphers = tuple(ciphers)
    if not ciphers:
        raise ValueError("cannot sum an empty ciphertext sequence.")
    return _sum_aligned_ciphers(ciphers, crypto_context=crypto_context)


def _try_inplace_owned_add(current, term, *, crypto_context):
    """Return ``(used, result)`` for an exact-metadata in-place owned add."""

    if current is None:
        return True, term
    inplace_add = getattr(crypto_context.fhe, "homo_add_inplace", None)
    same_metadata = (
        callable(inplace_add)
        and not bool(getattr(current, "is_ext", False))
        and not bool(getattr(term, "is_ext", False))
        and int(current.slots) == int(term.slots)
        and all(
            getattr(current.state, field, None) == getattr(term.state, field, None)
            for field in ("cur_limbs", "scale_degree", "scaling_factor")
        )
    )
    if not same_metadata:
        return False, None
    try:
        combined = inplace_add(current, term, crypto_context.context)
    except Exception:
        release_if_supported(term)
        raise
    # Backends commonly return ``None`` for a mutating ``*_inplace`` call;
    # retain the owned accumulator in that case.
    if combined is None:
        combined = current
    if combined is not current:
        release_if_supported(current)
    release_if_supported(term)
    return True, combined


def _accumulate_owned_fast(current, term, *, crypto_context):
    """Consume two owned ciphertexts, using in-place add when metadata matches."""

    used, result = _try_inplace_owned_add(current, term, crypto_context=crypto_context)
    if used:
        return result
    return accumulate_owned(current, term, crypto_context=crypto_context)


def _accumulate_aligned_owned(current, term, *, crypto_context):
    """Consume ``term`` and, after a successful add, ``current``."""

    if current is None:
        return term
    # Folded-O accumulation is a hot serial-Decode boundary.  After the
    # first stream both operands normally have identical metadata, so the
    # backend can add into the owned accumulator without allocating another
    # ciphertext.  Keep the aligned allocating path for mixed levels and for
    # test/fallback backends that do not expose the in-place primitive.
    used, fast = _try_inplace_owned_add(current, term, crypto_context=crypto_context)
    if used:
        return fast
    try:
        combined = _sum_borrowed_aligned(
            (current, term), crypto_context=crypto_context
        )
    except Exception:
        release_if_supported(term)
        raise
    release_if_supported(current)
    release_if_supported(term)
    return combined


def _encode_for_cipher(
    values: np.ndarray,
    cipher,
    *,
    name: str,
    crypto_context,
):
    values = np.asarray(values).reshape(-1)
    slots = int(getattr(cipher, "slots", values.size))
    if values.size != slots:
        raise ValueError(
            f"{name} plaintext has {values.size} slots, cipher has {slots}."
        )
    return encode_mask(
        values,
        name=name,
        level=int(crypto_context.level_for_cipher(cipher)),
        crypto_context=crypto_context,
        dtype=values.dtype,
    )


def _mul_mask_rotate(
    cipher,
    mask: np.ndarray,
    rotation: int,
    *,
    name: str,
    crypto_context,
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
    cache_key: tuple[object, ...] | None = None,
):
    plain = masked = None
    cached_plain = False
    try:
        if plaintext_cache is not None and cache_key is not None:
            cache_id = (
                *cache_key,
                int(getattr(cipher, "slots", 0)),
                int(crypto_context.level_for_cipher(cipher)),
                int(getattr(cipher.state, "scale_degree", 1)),
                repr(getattr(cipher.state, "scaling_factor", None)),
            )
            plain = plaintext_cache.get(cache_id)
            if plain is None:
                plain = _encode_for_cipher(
                    mask, cipher, name=name, crypto_context=crypto_context
                )
                plaintext_cache[cache_id] = plain
                cached_plain = True
            else:
                cached_plain = True
        else:
            plain = _encode_for_cipher(
                mask, cipher, name=name, crypto_context=crypto_context
            )
        masked = mul_plain_rescale(cipher, plain, crypto_context=crypto_context)
        if int(rotation) == 0:
            result = masked
            masked = None
            return result
        return crypto_context.fhe.homo_rotate(
            masked, int(rotation), crypto_context.context
        )
    finally:
        release_if_supported(masked)
        if not cached_plain:
            release_if_supported(plain)


def _qk_source_mask(
    layout: PairFold128Layout, *, pair: int, occupied: int
) -> np.ndarray:
    mask = np.zeros((int(layout.slots),), dtype=np.float64)
    for token in range(int(occupied)):
        for lane in range(int(layout.head_lanes)):
            mask[layout.page_slot(pair, 0, token, lane)] = 1.0
    return mask


def _probability_source_mask(
    layout: PairFold128Layout,
    *,
    group: int,
    pair: int,
    query_offset: int,
    occupied: int,
) -> np.ndarray:
    mask = np.zeros((int(layout.slots),), dtype=np.float64)
    for token in range(int(occupied)):
        for lane in range(int(layout.head_lanes)):
            mask[
                layout.score_slot(group, pair, query_offset, token, lane)
            ] = 1.0
    return mask


def _output_lane_mask(
    layout: PairFold128Layout, *, lane: int
) -> np.ndarray:
    mask = np.zeros((int(layout.slots),), dtype=np.float64)
    for pair in range(int(layout.pair_blocks)):
        for feature in range(int(layout.fold_dim)):
            mask[layout.page_slot(pair, feature, 0, lane)] = 1.0
    return mask


def _broadcast_queries(
    query_ciphers: tuple[object, ...],
    *,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None,
) -> tuple[object, ...]:
    outputs: list[object] = []
    try:
        for query in query_ciphers:
            outputs.append(
                _timed(
                    profile,
                    "query_broadcast",
                    lambda query=query: _rotate_add_tree(
                        query,
                        stride=-2,
                        count=int(layout.seq_len),
                        crypto_context=crypto_context,
                    ),
                )
            )
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        _release_all(outputs)


def _extract_real(cipher, *, crypto_context):
    rotate_add = getattr(crypto_context.fhe, "homo_rotate_add", None)
    if callable(rotate_add) and os.environ.get("LLAMA_FHE_ROTATE_ADD", "1") != "0":
        return rotate_add(
            cipher,
            int(crypto_context.context.M) - 1,
            crypto_context.context,
            addend=cipher,
        )
    conjugated = None
    try:
        conjugated = crypto_context.fhe.homo_rotate(
            cipher,
            int(crypto_context.context.M) - 1,
            crypto_context.context,
        )
        return crypto_context.fhe.homo_add(
            cipher, conjugated, crypto_context.context
        )
    finally:
        release_if_supported(conjugated)


def _route_qk_score(
    reduced_real,
    *,
    group: int,
    query_offset: int,
    occupied: int,
    layout: PairFold128Layout,
    crypto_context,
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
):
    accumulator = None
    placed = None
    try:
        for pair in range(int(layout.pair_blocks)):
            placed = _mul_mask_rotate(
                reduced_real,
                _qk_source_mask(layout, pair=pair, occupied=occupied),
                layout.page_to_score_rotation(group, pair, query_offset),
                name=(
                    "pairfold.qk.score."
                    f"o{occupied}.g{group}.r{query_offset}.p{pair}"
                ),
                crypto_context=crypto_context,
                plaintext_cache=plaintext_cache,
                cache_key=("qk.score", int(occupied), int(pair)),
            )
            pending = placed
            placed = None
            accumulator = _accumulate_owned_fast(
                accumulator, pending, crypto_context=crypto_context
            )
        if accumulator is None:
            raise RuntimeError("PairFold QK score route produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(placed)
        release_if_supported(accumulator)


def _qk_product_to_score(
    product,
    *,
    group: int,
    query_offset: int,
    occupied: int,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None,
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
):
    reduced = real = None
    try:
        reduced = _timed(
            profile,
            "feature_reduce",
            lambda: _rotate_add_tree(
                product,
                stride=2 * int(layout.seq_len),
                count=int(layout.fold_dim),
                crypto_context=crypto_context,
            ),
        )
        # Q=.5*(Qlo+iQhi), K=Klo-iKhi, hence z+conj(z) is already
        # the real dot product.  There is intentionally no second factor 2.
        real = _timed(
            profile,
            "real_extract",
            lambda: _extract_real(reduced, crypto_context=crypto_context),
        )
        return _timed(
            profile,
            "score_route",
            lambda: _route_qk_score(
                real,
                group=group,
                query_offset=query_offset,
                occupied=occupied,
                layout=layout,
                crypto_context=crypto_context,
                plaintext_cache=plaintext_cache,
            ),
        )
    finally:
        release_if_supported(real)
        release_if_supported(reduced)


def _qk_cipher_page_from_broadcasts(
    broadcasts: tuple[object, ...],
    key_groups: tuple[object, ...],
    *,
    occupied: int,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None,
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
):
    accumulator = term = None
    try:
        for group in range(int(layout.key_value_groups)):
            for query_offset in range(int(layout.gqa_ratio)):
                query_index = layout.query_cipher_index(group, query_offset)
                product = None
                try:
                    product = _timed(
                        profile,
                        "cipher_multiply",
                        lambda group=group, query_index=query_index: (
                            _mul_cipher_aligned(
                                broadcasts[query_index],
                                key_groups[group],
                                crypto_context=crypto_context,
                            )
                        ),
                    )
                    term = _qk_product_to_score(
                        product,
                        group=group,
                        query_offset=query_offset,
                        occupied=occupied,
                        layout=layout,
                        crypto_context=crypto_context,
                        profile=profile,
                        plaintext_cache=plaintext_cache,
                    )
                finally:
                    release_if_supported(product)
                pending = term
                term = None
                accumulator = _accumulate_owned_fast(
                    accumulator, pending, crypto_context=crypto_context
                )
        if accumulator is None:
            raise RuntimeError("PairFold QK page produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(term)
        release_if_supported(accumulator)


def _qk_plain_page_from_broadcasts(
    broadcasts: tuple[object, ...],
    packed_key: np.ndarray,
    *,
    occupied: int,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None,
    plains: dict[tuple[int, int], object],
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
):
    accumulator = term = None
    try:
        for group in range(int(layout.key_value_groups)):
            for query_offset in range(int(layout.gqa_ratio)):
                query_index = layout.query_cipher_index(group, query_offset)
                query = broadcasts[query_index]
                key = (group, int(query.state.cur_limbs))
                plain = plains.get(key)
                if plain is None:
                    plain = _encode_for_cipher(
                        packed_key[group],
                        query,
                        name=f"pairfold.prefix.qk.k{group}",
                        crypto_context=crypto_context,
                    )
                    plains[key] = plain
                product = None
                try:
                    product = _timed(
                        profile,
                        "plain_multiply",
                        lambda query=query, plain=plain: mul_plain_rescale(
                            query, plain, crypto_context=crypto_context
                        ),
                    )
                    term = _qk_product_to_score(
                        product,
                        group=group,
                        query_offset=query_offset,
                        occupied=occupied,
                        layout=layout,
                        crypto_context=crypto_context,
                        profile=profile,
                        plaintext_cache=plaintext_cache,
                    )
                finally:
                    release_if_supported(product)
                pending = term
                term = None
                accumulator = _accumulate_owned_fast(
                    accumulator, pending, crypto_context=crypto_context
                )
        if accumulator is None:
            raise RuntimeError("PairFold Prefix QK produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(term)
        release_if_supported(accumulator)


def qk_pairfold_pages_from_queries_fhe(
    query_ciphers: Iterable[object],
    key_pages: Iterable[Iterable[object]],
    *,
    layout: PairFold128Layout,
    crypto_context,
    occupied_pages: Iterable[int],
    key_prefix: np.ndarray | None = None,
    prefix_k: int | None = None,
    profile: dict[str, float] | None = None,
) -> tuple[tuple[object], ...]:
    """Low-level borrowed-query QK helper for tests and isolated kernels.

    Unlike the production carrier API, this helper requires eight Q
    ciphertexts to exist at once.  It remains useful as an A/B oracle.  The
    broadcasts are shared by the optional plaintext Prefix page and every
    content page.  Inputs are borrowed.
    """

    query_ciphers = _require_query_ciphers(query_ciphers, layout=layout)
    key_pages = _require_pages(key_pages, layout=layout, name="K")
    occupied = _require_occupied_pages(
        occupied_pages, page_count=len(key_pages), layout=layout
    )
    if (key_prefix is None) != (prefix_k is None):
        raise ValueError("key_prefix and prefix_k must be provided together.")
    packed_prefix = None
    if key_prefix is not None:
        prefix_k = int(prefix_k)  # type: ignore[arg-type]
        if not 1 <= prefix_k <= int(layout.seq_len):
            raise ValueError(
                f"prefix_k={prefix_k} outside [1, {layout.seq_len}]."
            )
        packed_prefix = _pack_plain_prefix(
            key_prefix,
            layout=layout,
            prefix_k=prefix_k,
            key=True,
        )
    if not key_pages and packed_prefix is None:
        raise ValueError("PairFold QK requires Prefix or content pages.")
    wall_started = time.perf_counter()
    broadcasts: tuple[object, ...] = ()
    output_pages: list[tuple[object]] = []
    plains: dict[tuple[int, int], object] = {}
    mask_plaintexts: dict[tuple[object, ...], object] = {}
    try:
        broadcasts = _broadcast_queries(
            query_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            profile=profile,
        )
        if packed_prefix is not None:
            output_pages.append(
                (
                    _qk_plain_page_from_broadcasts(
                        broadcasts,
                        packed_prefix,
                        occupied=int(prefix_k),
                        layout=layout,
                        crypto_context=crypto_context,
                        profile=profile,
                        plains=plains,
                        plaintext_cache=mask_plaintexts,
                    ),
                )
            )
        for key_groups, page_occupied in zip(
            key_pages, occupied, strict=True
        ):
            output_pages.append(
                (
                    _qk_cipher_page_from_broadcasts(
                        broadcasts,
                        key_groups,
                        occupied=page_occupied,
                        layout=layout,
                        crypto_context=crypto_context,
                        profile=profile,
                        plaintext_cache=mask_plaintexts,
                    ),
                )
            )
        result = tuple(output_pages)
        output_pages = []
        if profile is not None:
            profile["content_pages"] = float(len(key_pages))
            profile["prefix_pages"] = float(packed_prefix is not None)
            profile["ct_ct"] = float(8 * len(key_pages))
            profile["data_ct_pt"] = float(8 * (packed_prefix is not None))
            profile["ct_pt"] = profile["data_ct_pt"]
            profile["wall"] = float(time.perf_counter() - wall_started)
        return result
    finally:
        _release_all(broadcasts)
        _release_all(plains.values())
        _release_all(mask_plaintexts.values())
        for page in output_pages:
            _release_all(page)


def _pack_plain_prefix(
    values: np.ndarray,
    *,
    layout: PairFold128Layout,
    prefix_k: int,
    key: bool,
) -> np.ndarray:
    values = np.asarray(values)
    expected_packed = (int(layout.key_value_groups), int(layout.slots))
    if values.shape == expected_packed:
        if not np.iscomplexobj(values):
            raise TypeError("packed PairFold Prefix K/V must be complex.")
        return values
    if values.ndim == 2 and values.shape[1] == (
        int(layout.key_value_heads) * int(layout.head_dim)
    ):
        values = values.reshape(
            values.shape[0],
            int(layout.key_value_heads),
            int(layout.head_dim),
        )
    expected_tail = (int(layout.key_value_heads), int(layout.head_dim))
    if values.ndim != 3 or values.shape[1:] != expected_tail:
        raise ValueError(
            "PairFold Prefix K/V must be packed [2,32768] or logical "
            f"[tokens,{expected_tail[0]},{expected_tail[1]}], got {values.shape}."
        )
    if int(values.shape[0]) < int(prefix_k):
        raise ValueError(
            f"Prefix provides {values.shape[0]} tokens but prefix_k={prefix_k}."
        )
    pack = layout.pack_key if key else layout.pack_value
    return pack(values[:prefix_k], dtype=np.complex128)


def _query_selection(
    selection: Iterable[tuple[int, int]] | None, *, layout: PairFold128Layout
) -> tuple[tuple[int, int], ...]:
    """Validated ``(group, query_offset)`` streams; default: all of them in kernel order."""

    groups = int(layout.key_value_groups)
    ratio = int(layout.gqa_ratio)
    if selection is None:
        return tuple((g, r) for g in range(groups) for r in range(ratio))
    streams = tuple((int(g), int(r)) for g, r in selection)
    if not streams:
        raise ValueError("query_selection must name at least one (group, query_offset).")
    if len(set(streams)) != len(streams):
        raise ValueError("query_selection must not repeat a (group, query_offset).")
    for group, query_offset in streams:
        if not 0 <= group < groups:
            raise IndexError(f"group={group} outside [0, {groups}).")
        if not 0 <= query_offset < ratio:
            raise IndexError(f"query_offset={query_offset} outside [0, {ratio}).")
    return streams


def query_selection_head_rows(
    selection: Iterable[tuple[int, int]] | None, *, layout: PairFold128Layout
) -> tuple[int, ...]:
    """Query-head rows (compact score packing) that ``selection`` streams own.

    Each ``(group, query_offset)`` stream covers ``pair_blocks * head_lanes``
    query heads; the Decode softmax treats every other head row as invalid
    (``head_rows=``) so the zero rows of a multi-card partial score page never
    reach the ``1/sqrt`` polynomial.  ``None`` selects every head.
    """

    streams = _query_selection(selection, layout=layout)
    rows = {
        int(layout.query_head(group, pair, lane, query_offset))
        for group, query_offset in streams
        for pair in range(int(layout.pair_blocks))
        for lane in range(int(layout.head_lanes))
    }
    return tuple(sorted(rows))


def qk_pairfold_pages_fhe(
    carrier_cipher,
    key_pages: Iterable[Iterable[object]],
    *,
    query_extractor: Callable[..., object],
    layout: PairFold128Layout,
    crypto_context,
    occupied_pages: Iterable[int],
    key_prefix: np.ndarray | None = None,
    prefix_k: int | None = None,
    profile: dict[str, float] | None = None,
    query_selection: Iterable[tuple[int, int]] | None = None,
    query_group_extractor: Callable[..., Iterable[object]] | None = None,
) -> tuple[tuple[object], ...]:
    """Production streaming PairFold QK from one QKV carrier.

    ``query_extractor`` is called as
    ``extractor(carrier, group=g, query_offset=r)`` and must return one owned,
    clean token-column-zero Q ciphertext.  The kernel broadcasts that one Q,
    reuses it for the optional plaintext Prefix and every content page, and
    releases it before asking for the next ``(g,r)``.  With the optional group
    callback, up to one GQA group's outputs are materialized (four Q ciphers
    for the production layout), while QK still consumes and releases one at a
    time.  The carrier, K pages, and Prefix array are borrowed.

    If a Prefix is present its compact score page is returned first, matching
    the existing multi-page Decode Alg2 page order.

    ``query_group_extractor`` is an optional serial-Decode callback receiving
    ``(carrier, group=..., query_offsets=...)`` and returning one owned query
    cipher per offset.  It lets all offsets in a GQA group share their sparse
    baby rotation basis.  ``query_selection`` (multi-card decode) restricts the ``(group,
    query_offset)`` streams evaluated by this call; the compact score pages
    then only carry those streams' segments (the other segments stay zero).
    Default: all ``key_value_groups * gqa_ratio`` streams.
    """

    if carrier_cipher is None:
        raise ValueError("carrier_cipher must not be None.")
    if not callable(query_extractor):
        raise TypeError("query_extractor must be callable.")
    if query_group_extractor is not None and not callable(query_group_extractor):
        raise TypeError("query_group_extractor must be callable when provided.")
    streams = _query_selection(query_selection, layout=layout)
    key_pages = _require_pages(key_pages, layout=layout, name="K")
    occupied = _require_occupied_pages(
        occupied_pages, page_count=len(key_pages), layout=layout
    )
    if (key_prefix is None) != (prefix_k is None):
        raise ValueError("key_prefix and prefix_k must be provided together.")
    packed_prefix = None
    if key_prefix is not None:
        prefix_k = int(prefix_k)  # type: ignore[arg-type]
        if not 1 <= prefix_k <= int(layout.seq_len):
            raise ValueError(
                f"prefix_k={prefix_k} outside [1, {layout.seq_len}]."
            )
        packed_prefix = _pack_plain_prefix(
            key_prefix,
            layout=layout,
            prefix_k=prefix_k,
            key=True,
        )
    if not key_pages and packed_prefix is None:
        raise ValueError("PairFold QK requires Prefix or content pages.")

    page_offset = int(packed_prefix is not None)
    accumulators: list[object | None] = [
        None
    ] * (len(key_pages) + page_offset)
    plains: dict[tuple[int, int], object] = {}
    mask_plaintexts: dict[tuple[object, ...], object] = {}
    borrowed_ids = {id(carrier_cipher)} | {
        id(cipher) for page in key_pages for cipher in page
    }
    query = broadcast = term = None
    pending_queries: list[object | None] = []
    query_peak = 0
    query_batch_count = 0
    wall_started = time.perf_counter()
    try:
        def _release_query_batch(values: Iterable[object]) -> None:
            for value in values:
                if value is not None and id(value) not in borrowed_ids:
                    release_if_supported(value)

        # Serial Decode can route one whole GQA group with one shared baby
        # rotation batch.  Multi-card callers keep the legacy one-stream
        # callback because each rank owns only a subset of streams.
        grouped = query_group_extractor is not None and query_selection is None
        if grouped:
            grouped_streams = tuple(
                tuple((group, query_offset) for query_offset in range(int(layout.gqa_ratio)))
                for group in range(int(layout.key_value_groups))
            )
        else:
            grouped_streams = tuple((stream,) for stream in streams)

        for batch_streams in grouped_streams:
            query_batch_count += 1
            extract_started = time.perf_counter()
            if grouped:
                group = int(batch_streams[0][0])
                query_batch = tuple(
                    query_group_extractor(
                        carrier_cipher,
                        group=group,
                        query_offsets=tuple(offset for _, offset in batch_streams),
                    )
                )
            else:
                query_batch = tuple(
                    query_extractor(
                        carrier_cipher,
                        group=int(group),
                        query_offset=int(query_offset),
                    )
                    for group, query_offset in batch_streams
                )
            pending_queries = list(query_batch)
            _profile_add(profile, "query_extract", time.perf_counter() - extract_started)
            if len(query_batch) != len(batch_streams):
                _release_query_batch(pending_queries)
                pending_queries = []
                raise RuntimeError(
                    "query extractor returned an unexpected batch length: "
                    f"{len(query_batch)} != {len(batch_streams)}"
                )
            if any(query is None for query in query_batch):
                _release_query_batch(pending_queries)
                pending_queries = []
                raise RuntimeError("query extractor returned None in a query batch.")
            if any(id(query) in borrowed_ids for query in query_batch):
                _release_query_batch(pending_queries)
                pending_queries = []
                raise ValueError("query extractor must return newly owned ciphertexts.")

            for query_index, ((group, query_offset), query) in enumerate(
                zip(batch_streams, query_batch, strict=True)
            ):
                query_peak = max(query_peak, len(query_batch))
                try:
                    broadcast = _timed(
                        profile,
                        "query_broadcast",
                        lambda: _rotate_add_tree(
                            query,
                            stride=-2,
                            count=int(layout.seq_len),
                            crypto_context=crypto_context,
                        ),
                    )
                    if packed_prefix is not None:
                        prefix_compute_started = time.perf_counter()
                        key = (group, int(broadcast.state.cur_limbs))
                        plain = plains.get(key)
                        if plain is None:
                            plain = _encode_for_cipher(
                                packed_prefix[group],
                                broadcast,
                                name=f"pairfold.prefix.qk.k{group}",
                                crypto_context=crypto_context,
                            )
                            plains[key] = plain
                        product = None
                        try:
                            product = _timed(
                                profile,
                                "plain_multiply",
                                lambda broadcast=broadcast, plain=plain: (
                                    mul_plain_rescale(
                                        broadcast,
                                        plain,
                                        crypto_context=crypto_context,
                                    )
                                ),
                            )
                            term = _qk_product_to_score(
                                product,
                                group=group,
                                query_offset=query_offset,
                                occupied=int(prefix_k),
                                layout=layout,
                                crypto_context=crypto_context,
                                profile=profile,
                                plaintext_cache=mask_plaintexts,
                            )
                        finally:
                            release_if_supported(product)
                        pending = term
                        term = None
                        accumulators[0] = _accumulate_owned_fast(
                            accumulators[0],
                            pending,
                            crypto_context=crypto_context,
                        )
                        _profile_add(
                            profile,
                            "prefix_compute",
                            time.perf_counter() - prefix_compute_started,
                        )

                    for page_index, (key_groups, page_occupied) in enumerate(
                        zip(key_pages, occupied, strict=True)
                    ):
                        content_compute_started = time.perf_counter()
                        product = None
                        try:
                            product = _timed(
                                profile,
                                "cipher_multiply",
                                lambda key_groups=key_groups, group=group: (
                                    _mul_cipher_aligned(
                                        broadcast,
                                        key_groups[group],
                                        crypto_context=crypto_context,
                                    )
                                ),
                            )
                            term = _qk_product_to_score(
                                product,
                                group=group,
                                query_offset=query_offset,
                                occupied=page_occupied,
                                layout=layout,
                                crypto_context=crypto_context,
                                profile=profile,
                                plaintext_cache=mask_plaintexts,
                            )
                        finally:
                            release_if_supported(product)
                        pending = term
                        term = None
                        output_index = page_offset + page_index
                        accumulators[output_index] = _accumulate_owned_fast(
                            accumulators[output_index],
                            pending,
                            crypto_context=crypto_context,
                        )
                        _profile_add(
                            profile,
                            "content_compute",
                            time.perf_counter() - content_compute_started,
                        )
                finally:
                    release_if_supported(term)
                    term = None
                    release_if_supported(broadcast)
                    broadcast = None
                    release_if_supported(query)
                    query = None
                    pending_queries[query_index] = None
            pending_queries = []


        if any(cipher is None for cipher in accumulators):
            raise RuntimeError("streaming PairFold QK produced an empty page.")
        result = tuple((cipher,) for cipher in accumulators)  # type: ignore[misc]
        accumulators = [None] * len(accumulators)
        if profile is not None:
            profile["content_pages"] = float(len(key_pages))
            profile["prefix_pages"] = float(packed_prefix is not None)
            profile["ct_ct"] = float(len(streams) * len(key_pages))
            profile["data_ct_pt"] = float(len(streams) * (packed_prefix is not None))
            profile["ct_pt"] = profile["data_ct_pt"]
            profile["query_extract_count"] = float(len(streams))
            profile["query_streams"] = float(len(streams))
            profile["query_peak_live"] = float(query_peak)
            profile["query_peak_live_ciphers"] = float(query_peak)
            profile["query_batch_count"] = float(query_batch_count)
            profile["query_group_batched"] = float(bool(grouped))
            profile["wall"] = float(time.perf_counter() - wall_started)
        return result
    finally:
        # The per-query finally normally owns these; these guards cover an
        # extractor or setup exception before that block is entered.
        release_if_supported(term)
        release_if_supported(broadcast)
        release_if_supported(query)
        for pending in pending_queries:
            if pending is not None and id(pending) not in borrowed_ids:
                release_if_supported(pending)
        _release_all(plains.values())
        _release_all(mask_plaintexts.values())
        _release_all(accumulators)


def qk_pairfold_prefix_from_queries_fhe(
    query_ciphers: Iterable[object],
    key_prefix: np.ndarray,
    *,
    prefix_k: int,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None = None,
) -> tuple[object]:
    """Evaluate a plaintext Prefix K page into one compact score ciphertext.

    This convenience wrapper is the Prefix-only form of
    :func:`qk_pairfold_pages_from_queries_fhe`.
    """

    pages = qk_pairfold_pages_from_queries_fhe(
        query_ciphers,
        (),
        layout=layout,
        crypto_context=crypto_context,
        occupied_pages=(),
        key_prefix=key_prefix,
        prefix_k=prefix_k,
        profile=profile,
    )
    return pages[0]


def qk_pairfold_prefix_fhe(
    carrier_cipher,
    key_prefix: np.ndarray,
    *,
    query_extractor: Callable[..., object],
    prefix_k: int,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None = None,
) -> tuple[object]:
    """Prefix-only convenience wrapper for streaming carrier QK."""

    pages = qk_pairfold_pages_fhe(
        carrier_cipher,
        (),
        query_extractor=query_extractor,
        layout=layout,
        crypto_context=crypto_context,
        occupied_pages=(),
        key_prefix=key_prefix,
        prefix_k=prefix_k,
        profile=profile,
    )
    return pages[0]


def _route_probability_to_page(
    probability,
    *,
    group: int,
    query_offset: int,
    occupied: int,
    layout: PairFold128Layout,
    crypto_context,
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
):
    accumulator = placed = None
    try:
        for pair in range(int(layout.pair_blocks)):
            placed = _mul_mask_rotate(
                probability,
                _probability_source_mask(
                    layout,
                    group=group,
                    pair=pair,
                    query_offset=query_offset,
                    occupied=occupied,
                ),
                layout.score_to_page_rotation(group, pair, query_offset),
                name=(
                    "pairfold.pv.probability."
                    f"o{occupied}.g{group}.r{query_offset}.p{pair}"
                ),
                crypto_context=crypto_context,
                plaintext_cache=plaintext_cache,
                cache_key=(
                    "pv.probability",
                    int(group),
                    int(pair),
                    int(query_offset),
                    int(occupied),
                ),
            )
            pending = placed
            placed = None
            accumulator = _accumulate_owned_fast(
                accumulator, pending, crypto_context=crypto_context
            )
        if accumulator is None:
            raise RuntimeError("PairFold probability route produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(placed)
        release_if_supported(accumulator)


def _broadcast_probability(
    probability,
    *,
    group: int,
    query_offset: int,
    occupied: int,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None,
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
):
    routed = None
    try:
        routed = _timed(
            profile,
            "probability_route",
            lambda: _route_probability_to_page(
                probability,
                group=group,
                query_offset=query_offset,
                occupied=occupied,
                layout=layout,
                crypto_context=crypto_context,
                plaintext_cache=plaintext_cache,
            ),
        )
        return _timed(
            profile,
            "feature_broadcast",
            lambda: _rotate_add_tree(
                routed,
                stride=-(2 * int(layout.seq_len)),
                count=int(layout.fold_dim),
                crypto_context=crypto_context,
            ),
        )
    finally:
        release_if_supported(routed)


def _route_output_lanes(
    reduced,
    *,
    group: int,
    query_offset: int,
    layout: PairFold128Layout,
    crypto_context,
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
):
    accumulator = placed = None
    try:
        for lane in range(int(layout.head_lanes)):
            placed = _mul_mask_rotate(
                reduced,
                _output_lane_mask(layout, lane=lane),
                layout.page_to_output_rotation(group, query_offset, lane),
                name=f"pairfold.pv.output.g{group}.r{query_offset}.l{lane}",
                crypto_context=crypto_context,
                plaintext_cache=plaintext_cache,
                cache_key=("pv.output", int(lane)),
            )
            pending = placed
            placed = None
            accumulator = _accumulate_owned_fast(
                accumulator, pending, crypto_context=crypto_context
            )
        if accumulator is None:
            raise RuntimeError("PairFold folded-O route produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(placed)
        release_if_supported(accumulator)


def _finish_pv_products(
    products: list[object],
    *,
    group: int,
    query_offset: int,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None,
    plaintext_cache: dict[tuple[object, ...], object] | None = None,
):
    summed = reduced = None
    try:
        if len(products) == 1:
            # ``products`` owns each page product.  The usual borrowed sum
            # deliberately deep-copies a one-item sequence, but the serial
            # Decode steady state has one content page and no Prefix, so that
            # copy only adds a ciphertext allocation.  Transfer the sole
            # product out of the caller's cleanup list before reducing it.
            summed = products.pop()
            _profile_add(profile, "page_accumulate", 0.0)
        else:
            # Products are owned by this call, so consume them into the
            # accumulator.  Equal-level pages use the in-place add fast path;
            # mixed-level pages retain the aligned allocating fallback.
            accumulate_started = time.perf_counter()
            summed = products[0]
            products[0] = None  # type: ignore[assignment]
            try:
                for index in range(1, len(products)):
                    term = products[index]
                    products[index] = None  # type: ignore[assignment]
                    summed = _accumulate_aligned_owned(
                        summed,
                        term,
                        crypto_context=crypto_context,
                    )
            except BaseException:
                release_if_supported(summed)
                summed = None
                raise
            _profile_add(
                profile,
                "page_accumulate",
                time.perf_counter() - accumulate_started,
            )
        reduced = _timed(
            profile,
            "token_reduce",
            lambda: _rotate_add_tree(
                summed,
                stride=2,
                count=int(layout.seq_len),
                crypto_context=crypto_context,
            ),
        )
        return _timed(
            profile,
            "output_route",
            lambda: _route_output_lanes(
                reduced,
                group=group,
                query_offset=query_offset,
                layout=layout,
                crypto_context=crypto_context,
                plaintext_cache=plaintext_cache,
            ),
        )
    finally:
        release_if_supported(reduced)
        release_if_supported(summed)


def pv_pairfold_pages_fhe(
    probability_pages: Iterable[Iterable[object]],
    value_pages: Iterable[Iterable[object]],
    *,
    layout: PairFold128Layout,
    crypto_context,
    occupied_pages: Iterable[int],
    prefix_probability_ciphers: Iterable[object] | None = None,
    value_prefix: np.ndarray | None = None,
    prefix_k: int | None = None,
    profile: dict[str, float] | None = None,
    query_selection: Iterable[tuple[int, int]] | None = None,
) -> tuple[object]:
    """Evaluate compact probabilities x Prefix/content PairFold V pages.

    For each ``(group, query_offset)``, the optional plaintext Prefix product
    and every ciphertext content-page product are level-aligned and accumulated
    *before* the one token-axis reduction.  The eight reduced results are
    routed into disjoint Q residues of one folded-O carrier.  Inputs are
    borrowed.

    ``query_selection`` (multi-card decode) restricts the streams to this
    rank's ``(group, query_offset)`` pairs; the folded-O carrier is then the
    partial of those residues and the ranks add their partials up
    (:func:`add_pairfold_output_contributions_fhe`).  Default: all streams.
    """

    streams = _query_selection(query_selection, layout=layout)
    probability_pages = _require_probability_pages(probability_pages)
    value_pages = _require_pages(value_pages, layout=layout, name="V")
    if len(probability_pages) != len(value_pages):
        raise ValueError(
            "PairFold probability and value page counts must match."
        )
    occupied = _require_occupied_pages(
        occupied_pages, page_count=len(value_pages), layout=layout
    )
    prefix_arguments = (
        prefix_probability_ciphers is not None,
        value_prefix is not None,
        prefix_k is not None,
    )
    if any(prefix_arguments) and not all(prefix_arguments):
        raise ValueError(
            "prefix_probability_ciphers, value_prefix, and prefix_k must be "
            "provided together."
        )
    prefix_probability = None
    packed_prefix = None
    if all(prefix_arguments):
        prefix_page = _require_probability_pages(
            (tuple(prefix_probability_ciphers),)  # type: ignore[arg-type]
        )
        prefix_probability = prefix_page[0][0]
        prefix_k = int(prefix_k)  # type: ignore[arg-type]
        if not 1 <= prefix_k <= int(layout.seq_len):
            raise ValueError(
                f"prefix_k={prefix_k} outside [1, {layout.seq_len}]."
            )
        packed_prefix = _pack_plain_prefix(
            value_prefix,  # type: ignore[arg-type]
            layout=layout,
            prefix_k=prefix_k,
            key=False,
        )
    if not value_pages and packed_prefix is None:
        raise ValueError("PairFold PV requires Prefix or content pages.")
    wall_started = time.perf_counter()
    output = contribution = None
    products: list[object] = []
    plains: dict[tuple[int, int], object] = {}
    mask_plaintexts: dict[tuple[object, ...], object] = {}
    try:
        for group, query_offset in streams:
            products = []
            try:
                for page_index, page_occupied in enumerate(occupied):
                    content_compute_started = time.perf_counter()
                    broadcast = product = None
                    try:
                        broadcast = _broadcast_probability(
                            probability_pages[page_index][0],
                            group=group,
                            query_offset=query_offset,
                            occupied=page_occupied,
                            layout=layout,
                            crypto_context=crypto_context,
                            profile=profile,
                            plaintext_cache=mask_plaintexts,
                        )
                        product = _timed(
                            profile,
                            "cipher_multiply",
                            lambda broadcast=broadcast, page_index=page_index, group=group: (
                                _mul_cipher_aligned(
                                    broadcast,
                                    value_pages[page_index][group],
                                    crypto_context=crypto_context,
                                )
                            ),
                        )
                        products.append(product)
                        product = None
                    finally:
                        release_if_supported(product)
                        release_if_supported(broadcast)
                    _profile_add(
                        profile,
                        "content_compute",
                        time.perf_counter() - content_compute_started,
                    )
                if packed_prefix is not None:
                    prefix_compute_started = time.perf_counter()
                    broadcast = product = None
                    try:
                        broadcast = _broadcast_probability(
                            prefix_probability,
                            group=group,
                            query_offset=query_offset,
                            occupied=int(prefix_k),
                            layout=layout,
                            crypto_context=crypto_context,
                            profile=profile,
                            plaintext_cache=mask_plaintexts,
                        )
                        key = (group, int(broadcast.state.cur_limbs))
                        plain = plains.get(key)
                        if plain is None:
                            plain = _encode_for_cipher(
                                packed_prefix[group],
                                broadcast,
                                name=f"pairfold.prefix.pv.v{group}",
                                crypto_context=crypto_context,
                            )
                            plains[key] = plain
                        product = _timed(
                            profile,
                            "plain_multiply",
                            lambda broadcast=broadcast, plain=plain: (
                                mul_plain_rescale(
                                    broadcast,
                                    plain,
                                    crypto_context=crypto_context,
                                )
                            ),
                        )
                        products.append(product)
                        product = None
                    finally:
                        release_if_supported(product)
                        release_if_supported(broadcast)
                    _profile_add(
                        profile,
                        "prefix_compute",
                        time.perf_counter() - prefix_compute_started,
                    )
                contribution = _finish_pv_products(
                    products,
                    group=group,
                    query_offset=query_offset,
                    layout=layout,
                    crypto_context=crypto_context,
                    profile=profile,
                    plaintext_cache=mask_plaintexts,
                )
            finally:
                _release_all(products)
                products = []
            pending = contribution
            contribution = None
            folded_o_accumulate_started = time.perf_counter()
            output = _accumulate_aligned_owned(
                output, pending, crypto_context=crypto_context
            )
            _profile_add(
                profile,
                "folded_o_accumulate",
                time.perf_counter() - folded_o_accumulate_started,
            )
        if output is None:
            raise RuntimeError("PairFold PV produced no folded-O carrier.")
        result = output
        output = None
        if profile is not None:
            profile["content_pages"] = float(len(value_pages))
            profile["prefix_pages"] = float(packed_prefix is not None)
            profile["ct_ct"] = float(len(streams) * len(value_pages))
            profile["data_ct_pt"] = float(len(streams) * (packed_prefix is not None))
            profile["ct_pt"] = profile["data_ct_pt"]
            profile["token_reductions"] = float(len(streams))
            profile["query_streams"] = float(len(streams))
            profile["wall"] = float(time.perf_counter() - wall_started)
        return (result,)
    finally:
        _release_all(products)
        release_if_supported(contribution)
        release_if_supported(output)
        _release_all(plains.values())
        _release_all(mask_plaintexts.values())


def pv_pairfold_prefix_fhe(
    probability_ciphers: Iterable[object],
    value_prefix: np.ndarray,
    *,
    prefix_k: int,
    layout: PairFold128Layout,
    crypto_context,
    profile: dict[str, float] | None = None,
) -> tuple[object]:
    """Evaluate one compact Prefix probability page x plaintext Prefix V.

    This convenience wrapper is the Prefix-only form of
    :func:`pv_pairfold_pages_fhe`.  A graph evaluating Prefix plus content
    should pass both to that function, which accumulates every product before
    the shared eight token reductions.  Inputs are borrowed.
    """

    return pv_pairfold_pages_fhe(
        (),
        (),
        layout=layout,
        crypto_context=crypto_context,
        occupied_pages=(),
        prefix_probability_ciphers=probability_ciphers,
        value_prefix=value_prefix,
        prefix_k=prefix_k,
        profile=profile,
    )


def add_pairfold_output_contributions_fhe(
    contributions: Iterable[object], *, crypto_context
) -> tuple[object]:
    """Level-align and add borrowed folded-O contributions."""

    contributions = tuple(contributions)
    if not contributions:
        raise ValueError("at least one folded-O contribution is required.")
    return (
        _sum_borrowed_aligned(contributions, crypto_context=crypto_context),
    )


__all__ = [
    "query_selection_head_rows",
    "PairFoldDecodeComplexity",
    "add_pairfold_output_contributions_fhe",
    "pairfold_decode_complexity",
    "pairfold_decode_rotations",
    "pv_pairfold_pages_fhe",
    "pv_pairfold_prefix_fhe",
    "pv_pairfold_rotations",
    "pv_probability_route_rotations",
    "qk_pairfold_pages_fhe",
    "qk_pairfold_pages_from_queries_fhe",
    "qk_pairfold_prefix_fhe",
    "qk_pairfold_prefix_from_queries_fhe",
    "qk_pairfold_rotations",
    "validate_pairfold_score_compatibility",
]
