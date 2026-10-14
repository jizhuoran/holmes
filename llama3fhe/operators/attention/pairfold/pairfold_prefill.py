from __future__ import annotations

"""Convert legacy Prefill C pages into Decode-native PairFold128 pages.

The Prefill attention-input adapter emits four two-head ciphertexts for each
of K and V.  K is already feature-folded as
``0.5 * (K_low + i*K_high)`` (on all feature rows), while V is real C-form.
PairFold128 stores four heads per ciphertext, so two legacy head pairs are
merged into each of two K/V ciphertexts:

``K_pairfold = K_low - i*K_high`` and
``V_pairfold = V_low + i*V_high``.

The FHE entry point owns and consumes all eight input ciphertexts.  On an
exception it also releases every output constructed before the failure.
"""

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.pairfold_kv_cache import PairFoldCipherPage

from ...primitives import accumulate_owned, bundle_plain, mul_plain_rescale


def encrypt_pairfold_plaintext_page(
    key: np.ndarray,
    value: np.ndarray,
    *,
    pairfold_layout: PairFold128Layout,
    crypto_context,
    level: int = 0,
) -> PairFoldCipherPage:
    """Pack and encrypt a plaintext Prefill/content page directly.

    This is the boundary used by plaintext-Prefill/encrypted-Decode
    experiments.  It deliberately bypasses legacy C form: two K payloads and
    two V payloads are encrypted in their persistent PairFold layout and
    returned under one owned page.  On any encryption or page-construction
    failure, every ciphertext created by this call is released.
    """

    level = int(level)
    if level < 0:
        raise ValueError(f"encryption level must be non-negative, got {level}.")
    # Normalize through the layout contract before allocating a ciphertext;
    # this also accepts the supported flattened [tokens, 1024] spelling.
    logical_key = pairfold_layout._kv_page(key, name="key")
    logical_value = pairfold_layout._kv_page(value, name="value")
    occupied = int(logical_key.shape[0])
    if int(logical_value.shape[0]) != occupied:
        raise ValueError(
            "plaintext PairFold K/V token counts must match, got "
            f"{occupied}/{logical_value.shape[0]}."
        )
    if occupied < 1:
        raise ValueError("a plaintext PairFold page must contain at least one token.")

    packed_key = pairfold_layout.pack_key(logical_key)
    packed_value = pairfold_layout.pack_value(logical_value)
    outputs: list[object] = []
    try:
        for payload in (*packed_key, *packed_value):
            outputs.append(
                crypto_context.encrypt(
                    payload,
                    level=level,
                    slots=int(pairfold_layout.slots),
                )
            )
        page = PairFoldCipherPage(
            key_groups=tuple(outputs[: pairfold_layout.key_value_groups]),
            value_groups=tuple(outputs[pairfold_layout.key_value_groups :]),
            occupied=occupied,
        )
        outputs = []
        return page
    finally:
        for cipher in outputs:
            release_if_supported(cipher)


def _validate_layouts(
    attention_layout: AttentionPairLayout,
    pairfold_layout: PairFold128Layout,
) -> None:
    for name in ("seq_len", "head_dim", "query_heads", "key_value_heads", "slots"):
        legacy = int(getattr(attention_layout, name))
        pairfold = int(getattr(pairfold_layout, name))
        if legacy != pairfold:
            raise ValueError(
                f"Prefill/PairFold {name} mismatch: {legacy} != {pairfold}."
            )
    expected_pairs = int(pairfold_layout.key_value_groups) * int(
        pairfold_layout.pair_blocks
    )
    if int(attention_layout.key_value_pair_count) != expected_pairs:
        raise ValueError(
            "Prefill/PairFold KV pair mismatch: "
            f"{attention_layout.key_value_pair_count} != {expected_pairs}."
        )
    if int(attention_layout.payload_slots) != int(pairfold_layout.slots):
        raise ValueError(
            "PairFold Prefill conversion requires one unrepeated C payload per "
            "ciphertext."
        )


def _require_numpy_rows(
    rows: np.ndarray,
    *,
    count: int,
    slots: int,
    name: str,
) -> np.ndarray:
    rows = np.asarray(rows)
    expected = (int(count), int(slots))
    if rows.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {rows.shape}.")
    return rows


def prefill_c_to_pairfold_numpy(
    folded_key_ciphers: np.ndarray,
    real_value_ciphers: np.ndarray,
    *,
    attention_layout: AttentionPairLayout,
    pairfold_layout: PairFold128Layout,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw Prefill ``K_C``/``V_C`` payloads to PairFold128.

    The returned arrays each have shape ``[2, 32768]`` and are slot-for-slot
    equal to :meth:`PairFold128Layout.pack_key` and ``pack_value`` for the
    same logical page.  V intentionally has full scale; only the incoming K
    uses the Prefill adapter's ``0.5`` fold.
    """

    _validate_layouts(attention_layout, pairfold_layout)
    pair_count = int(attention_layout.key_value_pair_count)
    slots = int(pairfold_layout.slots)
    folded_key = _require_numpy_rows(
        folded_key_ciphers,
        count=pair_count,
        slots=slots,
        name="raw folded Prefill K_C",
    )
    real_value = _require_numpy_rows(
        real_value_ciphers,
        count=pair_count,
        slots=slots,
        name="real Prefill V_C",
    )
    if not np.iscomplexobj(folded_key):
        raise TypeError("raw folded Prefill K_C must be complex.")
    if np.iscomplexobj(real_value):
        raise TypeError("real Prefill V_C must not be complex.")

    head_dim = int(pairfold_layout.head_dim)
    fold_dim = int(pairfold_layout.fold_dim)
    seq_len = int(pairfold_layout.seq_len)
    lanes = int(pairfold_layout.head_lanes)
    groups = int(pairfold_layout.key_value_groups)
    pairs = int(pairfold_layout.pair_blocks)
    key_grid = folded_key.reshape(pair_count, head_dim, seq_len, lanes)
    value_grid = real_value.reshape(pair_count, head_dim, seq_len, lanes)
    dtype = np.result_type(folded_key.dtype, real_value.dtype, np.complex64)
    pairfold_key = np.empty(
        (groups, pairs, fold_dim, seq_len, lanes), dtype=dtype
    )
    pairfold_value = np.empty_like(pairfold_key)

    for group in range(groups):
        for pair in range(pairs):
            source = group * pairs + pair
            # Raw K carries 0.5*(lo+i*hi); PairFold K is lo-i*hi.
            pairfold_key[group, pair] = 2.0 * np.conjugate(
                key_grid[source, :fold_dim]
            )
            # PairFold V is full-scale lo+i*hi (there is no 0.5 here).
            pairfold_value[group, pair] = (
                value_grid[source, :fold_dim]
                + 1j * value_grid[source, fold_dim:]
            )

    return pairfold_key.reshape(groups, slots), pairfold_value.reshape(
        groups, slots
    )


def prefill_c_to_pairfold_rotations(
    pairfold_layout: PairFold128Layout,
) -> tuple[int, ...]:
    """Rotation keys required by :func:`prefill_c_to_pairfold_fhe`."""

    slots = int(pairfold_layout.slots)
    # CKKS exposes N/2 complex slots and uses cyclotomic order M=2N, hence
    # conjugation is the automorphism M-1 = 4*slots-1.  Do not derive this as
    # 2*slots-1: that only matches a reduced fake context and omits the real
    # GPU conjugation key.
    return (slots // 2, 4 * slots - 1)


def _pair_mask_bundle(
    pairfold_layout: PairFold128Layout,
    *,
    crypto_context,
    occupied: int = 128,
):
    occupied = int(occupied)
    if not 1 <= occupied <= int(pairfold_layout.seq_len):
        raise ValueError(f"PairFold occupied must be in [1, 128], got {occupied}.")
    slots = int(pairfold_layout.slots)
    pair_width = slots // int(pairfold_layout.pair_blocks)
    vectors: dict[str, np.ndarray] = {}
    for pair in range(int(pairfold_layout.pair_blocks)):
        start = pair * pair_width
        stop = start + pair_width
        value = np.zeros((slots,), dtype=np.float64)
        value[start:stop] = 1.0
        value.reshape(2, 64, int(pairfold_layout.seq_len), 2)[
            :, :, occupied:, :
        ] = 0.0
        vectors[f"value.pair{pair}"] = value
        vectors[f"key.pair{pair}"] = 2.0 * value
    return crypto_context.constant_bundle(
        vectors,
        cache_key=(
            "attention.pairfold_prefill_masks",
            int(pairfold_layout.seq_len),
            int(pairfold_layout.head_dim),
            slots,
            occupied,
        ),
    )


def _masked_pair(
    cipher,
    *,
    mask_bundle,
    mask_name: str,
    crypto_context,
):
    plain = None
    try:
        plain = bundle_plain(
            mask_bundle,
            mask_name,
            cipher,
            crypto_context=crypto_context,
            cache=True,
        )
        return mul_plain_rescale(cipher, plain, crypto_context=crypto_context)
    finally:
        release_if_supported(plain)


def _multiply_i(cipher, *, crypto_context):
    return fhe.homo_mul_i(cipher, crypto_context.context, negative=False)


def _convert_key_group(
    sources: tuple[object, object],
    *,
    half_rotation: int,
    conjugate_rotation: int,
    mask_bundle,
    crypto_context,
):
    accumulator = None
    try:
        for pair, source in enumerate(sources):
            conjugated = shifted = term = None
            try:
                conjugated = crypto_context.fhe.homo_rotate(
                    source, conjugate_rotation, crypto_context.context
                )
                routed = conjugated
                if pair:
                    shifted = crypto_context.fhe.homo_rotate(
                        conjugated, half_rotation, crypto_context.context
                    )
                    routed = shifted
                term = _masked_pair(
                    routed,
                    mask_bundle=mask_bundle,
                    mask_name=f"key.pair{pair}",
                    crypto_context=crypto_context,
                )
                pending = term
                term = None
                accumulator = accumulate_owned(
                    accumulator, pending, crypto_context=crypto_context
                )
            finally:
                release_if_supported(term)
                release_if_supported(shifted)
                release_if_supported(conjugated)
        if accumulator is None:
            raise RuntimeError("PairFold K conversion produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(accumulator)


def _convert_value_group(
    sources: tuple[object, object],
    *,
    half_rotation: int,
    mask_bundle,
    crypto_context,
):
    accumulator = None
    try:
        for pair, source in enumerate(sources):
            rotated = imaginary = combined = term = None
            try:
                rotated = crypto_context.fhe.homo_rotate(
                    source, half_rotation, crypto_context.context
                )
                imaginary_source = rotated if pair == 0 else source
                imaginary = _multiply_i(
                    imaginary_source,
                    crypto_context=crypto_context,
                )
                real_source = source if pair == 0 else rotated
                combined = crypto_context.fhe.homo_add(
                    real_source, imaginary, crypto_context.context
                )
                term = _masked_pair(
                    combined,
                    mask_bundle=mask_bundle,
                    mask_name=f"value.pair{pair}",
                    crypto_context=crypto_context,
                )
                pending = term
                term = None
                accumulator = accumulate_owned(
                    accumulator, pending, crypto_context=crypto_context
                )
            finally:
                release_if_supported(term)
                release_if_supported(combined)
                release_if_supported(imaginary)
                release_if_supported(rotated)
        if accumulator is None:
            raise RuntimeError("PairFold V conversion produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(accumulator)


def _convert_real_key_group(
    sources: tuple[object, object],
    *,
    half_rotation: int,
    mask_bundle,
    crypto_context,
):
    """Pack two real C-form K rows as ``K_lo - i*K_hi``.

    ``sources`` are already Decode-ready ``pack_key_c`` rows (one row per
    adjacent KV-head pair).  PairFold stores the lower 64 feature rows in the
    real part and the upper 64 rows in the imaginary part.  The negative
    imaginary sign is intentional: the QK kernel consumes the conjugated K
    convention.  This path is kept separate from :func:`_convert_key_group`,
    whose input contract is the ideal half-scaled complex folded payload.
    """

    accumulator = None
    try:
        for pair, source in enumerate(sources):
            rotated = imaginary = combined = term = None
            try:
                rotated = crypto_context.fhe.homo_rotate(
                    source, half_rotation, crypto_context.context
                )
                imaginary_source = rotated if pair == 0 else source
                imaginary = fhe.homo_mul_i(
                    imaginary_source,
                    crypto_context.context,
                    negative=True,
                )
                real_source = source if pair == 0 else rotated
                combined = crypto_context.fhe.homo_add(
                    real_source, imaginary, crypto_context.context
                )
                term = _masked_pair(
                    combined,
                    mask_bundle=mask_bundle,
                    mask_name=f"value.pair{pair}",
                    crypto_context=crypto_context,
                )
                pending = term
                term = None
                accumulator = accumulate_owned(
                    accumulator, pending, crypto_context=crypto_context
                )
            finally:
                release_if_supported(term)
                release_if_supported(combined)
                release_if_supported(imaginary)
                release_if_supported(rotated)
        if accumulator is None:
            raise RuntimeError("PairFold real-K conversion produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(accumulator)


def _convert_emitted_key_group_direct(
    sources: tuple[object, object],
    *,
    half_rotation: int,
    mask_bundle,
    crypto_context,
):
    """Select the emitted lower feature half directly into PairFold K."""

    accumulator = None
    try:
        for pair, source in enumerate(sources):
            shifted = term = None
            try:
                routed = source
                if pair:
                    shifted = crypto_context.fhe.homo_rotate(
                        source, half_rotation, crypto_context.context
                    )
                    routed = shifted
                term = _masked_pair(
                    routed,
                    mask_bundle=mask_bundle,
                    mask_name=f"value.pair{pair}",
                    crypto_context=crypto_context,
                )
                pending = term
                term = None
                accumulator = accumulate_owned(
                    accumulator, pending, crypto_context=crypto_context
                )
            finally:
                release_if_supported(term)
                release_if_supported(shifted)
        if accumulator is None:
            raise RuntimeError("Direct emitted-K conversion produced no output.")
        result = accumulator
        accumulator = None
        return result
    finally:
        release_if_supported(accumulator)


def prefill_emitted_key_c_to_pairfold_fhe(
    folded_key_ciphers: tuple[object, ...] | list[object],
    *,
    attention_layout: AttentionPairLayout,
    pairfold_layout: PairFold128Layout,
    crypto_context,
    occupied: int = 128,
) -> tuple[object, ...]:
    """Convert the graph's emitted Prefill K_C stream to PairFold K.

    The graph emits the historical expanded/folded K_C contract, not the
    idealized ``pack_key_c_feature_folded`` test payload.  Reuse the legacy
    correction (folded-to-real followed by the expand/unfold sign mask) before
    packing the two real K rows into each PairFold ciphertext.  All input rows
    are consumed on success and on failure.
    """

    from ..decode.decode import (
        correct_prefill_emitted_key_c_fhe,
        folded_key_c_to_real_c_fhe,
    )

    key_inputs = tuple(folded_key_ciphers)
    corrected: tuple[object, ...] = ()
    folded: tuple[object, ...] = ()
    raw_transferred = False
    folded_transferred = False
    outputs: list[object] = []
    success = False
    try:
        _validate_layouts(attention_layout, pairfold_layout)
        expected = int(attention_layout.key_value_pair_count)
        if len(key_inputs) != expected:
            raise ValueError(
                f"PairFold emitted-K converter expects {expected} K_C ciphers, "
                f"got {len(key_inputs)}."
            )
        raw_transferred = True
        folded = folded_key_c_to_real_c_fhe(
            key_inputs, crypto_context=crypto_context
        )
        folded_transferred = True
        corrected = correct_prefill_emitted_key_c_fhe(
            folded,
            layout=attention_layout,
            crypto_context=crypto_context,
        )
        folded = ()
        masks = _pair_mask_bundle(
            pairfold_layout, crypto_context=crypto_context, occupied=occupied
        )
        half_rotation, _ = prefill_c_to_pairfold_rotations(pairfold_layout)
        pairs = int(pairfold_layout.pair_blocks)
        for group in range(int(pairfold_layout.key_value_groups)):
            start = group * pairs
            outputs.append(
                _convert_real_key_group(
                    corrected[start : start + pairs],
                    half_rotation=half_rotation,
                    mask_bundle=masks,
                    crypto_context=crypto_context,
                )
            )
        result = tuple(outputs)
        outputs = []
        success = True
        return result
    finally:
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)
        if not raw_transferred:
            for cipher in key_inputs:
                release_if_supported(cipher)
        if not folded_transferred:
            for cipher in folded:
                release_if_supported(cipher)
        # If correction failed before returning, it has consumed ``folded``;
        # otherwise the successful call above cleared it.
        for cipher in corrected:
            release_if_supported(cipher)


def prefill_emitted_key_c_direct_to_pairfold_fhe(
    folded_key_ciphers: tuple[object, ...] | list[object],
    *,
    attention_layout: AttentionPairLayout,
    pairfold_layout: PairFold128Layout,
    crypto_context,
    occupied: int = 128,
) -> tuple[object, ...]:
    """Pack emitted folded ``K_C`` directly into PairFold K ciphertexts.

    The graph's expanded K_C has lower-half slots ``S = K_low - i*K_high``
    and upper-half slots ``i*conj(S)``.  The legacy folded-to-real, correction,
    and real-K pack reduce to exactly ``S`` in the PairFold feature slots.
    Select those slots without the intermediate real-K materialization.
    """

    key_inputs = tuple(folded_key_ciphers)
    outputs: list[object] = []
    success = False
    try:
        _validate_layouts(attention_layout, pairfold_layout)
        expected = int(attention_layout.key_value_pair_count)
        if len(key_inputs) != expected:
            raise ValueError(
                f"PairFold direct emitted-K converter expects {expected} K_C ciphers, "
                f"got {len(key_inputs)}."
            )
        masks = _pair_mask_bundle(
            pairfold_layout, crypto_context=crypto_context, occupied=occupied
        )
        half_rotation, _ = prefill_c_to_pairfold_rotations(pairfold_layout)
        pairs = int(pairfold_layout.pair_blocks)
        for group in range(int(pairfold_layout.key_value_groups)):
            start = group * pairs
            outputs.append(
                _convert_emitted_key_group_direct(
                    tuple(key_inputs[start : start + pairs]),
                    half_rotation=half_rotation,
                    mask_bundle=masks,
                    crypto_context=crypto_context,
                )
            )
        result = tuple(outputs)
        outputs = []
        success = True
        return result
    finally:
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)
        for cipher in key_inputs:
            release_if_supported(cipher)


def prefill_emitted_key_c_direct_group_to_pairfold_fhe(
    folded_key_ciphers: tuple[object, ...] | list[object],
    *,
    pairfold_layout: PairFold128Layout,
    crypto_context,
    occupied: int = 128,
) -> object:
    """Directly pack one two-row emitted-K PairFold group."""

    sources = tuple(folded_key_ciphers)
    try:
        if len(sources) != int(pairfold_layout.pair_blocks):
            raise ValueError(
                "a PairFold K group requires "
                f"{pairfold_layout.pair_blocks} raw pairs, got {len(sources)}."
            )
        masks = _pair_mask_bundle(
            pairfold_layout, crypto_context=crypto_context, occupied=occupied
        )
        half_rotation, _ = prefill_c_to_pairfold_rotations(pairfold_layout)
        result = _convert_emitted_key_group_direct(
            sources,
            half_rotation=half_rotation,
            mask_bundle=masks,
            crypto_context=crypto_context,
        )
        return result
    finally:
        # Group helpers own their raw inputs on entry, just like the
        # all-groups converter.  Release them after a successful conversion as
        # well; otherwise owner-directed routing would retain every raw pair
        # until the layer cache is destroyed.
        for cipher in sources:
            release_if_supported(cipher)


def prefill_emitted_key_c_group_to_pairfold_fhe(
    folded_key_ciphers: tuple[object, ...] | list[object],
    *,
    attention_layout: AttentionPairLayout,
    pairfold_layout: PairFold128Layout,
    group: int,
    crypto_context,
    occupied: int = 128,
) -> object:
    """Convert one emitted Prefill K group to a PairFold ciphertext.

    This is the grouped equivalent of
    :func:`prefill_emitted_key_c_to_pairfold_fhe`.  It is used by the
    owner-directed multi-rank KV emitter so a rank only receives and converts
    the two raw K pairs needed for its output group.  The legacy emitted-K
    correction mask is sliced to the selected group before the PairFold pack.
    """

    sources = tuple(folded_key_ciphers)
    corrected: tuple[object, ...] = ()
    real: tuple[object, ...] = ()
    outputs: list[object] = []
    raw_transferred = False
    success = False
    try:
        _validate_layouts(attention_layout, pairfold_layout)
        group = int(group)
        groups = int(pairfold_layout.key_value_groups)
        pairs = int(pairfold_layout.pair_blocks)
        if not 0 <= group < groups:
            raise IndexError(f"group={group} outside [0, {groups}).")
        if len(sources) != pairs:
            raise ValueError(
                "a PairFold emitted-K group requires "
                f"{pairs} raw pairs, got {len(sources)}."
            )

        # folded_key_c_to_real_c_fhe consumes ``sources``.  Apply only the
        # correction rows belonging to this group, since the helper normally
        # receives all groups at once.
        from ..decode.decode import (
            _mul_packed_rows,
            folded_key_c_to_real_c_fhe,
            prefill_key_c_emit_correction_mask,
        )

        raw_transferred = True
        real = folded_key_c_to_real_c_fhe(sources, crypto_context=crypto_context)
        mask = prefill_key_c_emit_correction_mask(attention_layout)
        start = group * pairs
        try:
            corrected = _mul_packed_rows(
                real,
                mask[start : start + pairs],
                name=f"prefill.key_c.correct.group{group}",
                crypto_context=crypto_context,
            )
        finally:
            for cipher in real:
                release_if_supported(cipher)
            real = ()

        masks = _pair_mask_bundle(
            pairfold_layout, crypto_context=crypto_context, occupied=occupied
        )
        half_rotation, _ = prefill_c_to_pairfold_rotations(pairfold_layout)
        result = _convert_real_key_group(
            corrected,
            half_rotation=half_rotation,
            mask_bundle=masks,
            crypto_context=crypto_context,
        )
        for cipher in corrected:
            release_if_supported(cipher)
        corrected = ()
        success = True
        return result
    finally:
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)
        for cipher in corrected:
            release_if_supported(cipher)
        for cipher in real:
            release_if_supported(cipher)
        if not raw_transferred:
            for cipher in sources:
                release_if_supported(cipher)


def prefill_c_to_pairfold_fhe(
    folded_key_ciphers: tuple[object, ...] | list[object],
    real_value_ciphers: tuple[object, ...] | list[object],
    *,
    attention_layout: AttentionPairLayout,
    pairfold_layout: PairFold128Layout,
    crypto_context,
    emitted_key_c: bool = False,
    occupied: int = 128,
) -> tuple[tuple[object, object], tuple[object, object]]:
    """Consume raw Prefill C ciphers and return two PairFold K plus two V.

    Ownership transfers at function entry: all input ciphertexts are released
    on success and on every exception path.  Returned ciphertexts are owned by
    the caller.  Each output consumes one plaintext-multiplication level.
    ``emitted_key_c`` selects the graph-emitted K_C contract, applying the
    legacy folded-to-real/sign correction before PairFold packing.  The
    default remains the idealized half-scaled complex payload used by the
    standalone converter tests.
    """

    key_inputs = tuple(folded_key_ciphers)
    value_inputs = tuple(real_value_ciphers)
    key_outputs: list[object] = []
    value_outputs: list[object] = []
    key_transferred = False
    success = False
    try:
        _validate_layouts(attention_layout, pairfold_layout)
        expected = int(attention_layout.key_value_pair_count)
        if len(key_inputs) != expected:
            raise ValueError(
                f"PairFold Prefill converter expects {expected} K_C ciphers, "
                f"got {len(key_inputs)}."
            )
        if len(value_inputs) != expected:
            raise ValueError(
                f"PairFold Prefill converter expects {expected} V_C ciphers, "
                f"got {len(value_inputs)}."
            )

        masks = _pair_mask_bundle(
            pairfold_layout, crypto_context=crypto_context, occupied=occupied
        )
        half_rotation, conjugate_rotation = prefill_c_to_pairfold_rotations(
            pairfold_layout
        )
        pairs = int(pairfold_layout.pair_blocks)
        if bool(emitted_key_c):
            # The emitted-K helper consumes every key input itself.  Mark the
            # transfer before entering it so the outer cleanup cannot release
            # the same handles a second time.
            key_transferred = True
            key_outputs.extend(
                prefill_emitted_key_c_to_pairfold_fhe(
                    key_inputs,
                    attention_layout=attention_layout,
                    pairfold_layout=pairfold_layout,
                    crypto_context=crypto_context,
                    occupied=occupied,
                )
            )
        for group in range(int(pairfold_layout.key_value_groups)):
            start = group * pairs
            if not emitted_key_c:
                key_outputs.append(
                    _convert_key_group(
                        key_inputs[start : start + pairs],
                        half_rotation=half_rotation,
                        conjugate_rotation=conjugate_rotation,
                        mask_bundle=masks,
                        crypto_context=crypto_context,
                    )
                )
            value_outputs.append(
                _convert_value_group(
                    value_inputs[start : start + pairs],
                    half_rotation=half_rotation,
                    mask_bundle=masks,
                    crypto_context=crypto_context,
                )
            )

        result = (tuple(key_outputs), tuple(value_outputs))
        key_outputs = []
        value_outputs = []
        success = True
        return result  # type: ignore[return-value]
    finally:
        if not success:
            for cipher in key_outputs + value_outputs:
                release_if_supported(cipher)
        seen: set[int] = set()
        if not key_transferred:
            cleanup_inputs = key_inputs + value_inputs
        else:
            cleanup_inputs = value_inputs
        for cipher in cleanup_inputs:
            identity = id(cipher)
            if identity not in seen:
                release_if_supported(cipher)
                seen.add(identity)


def prefill_real_value_c_to_pairfold_fhe(
    real_value_ciphers: tuple[object, ...] | list[object],
    *,
    attention_layout: AttentionPairLayout,
    pairfold_layout: PairFold128Layout,
    crypto_context,
    occupied: int = 128,
) -> tuple[object, ...]:
    """Convert only real Prefill V_C rows into PairFold V groups."""

    value_inputs = tuple(real_value_ciphers)
    outputs: list[object] = []
    success = False
    try:
        _validate_layouts(attention_layout, pairfold_layout)
        expected = int(attention_layout.key_value_pair_count)
        if len(value_inputs) != expected:
            raise ValueError(
                f"PairFold V converter expects {expected} V_C ciphers, got {len(value_inputs)}."
            )
        masks = _pair_mask_bundle(
            pairfold_layout, crypto_context=crypto_context, occupied=occupied
        )
        half_rotation, _ = prefill_c_to_pairfold_rotations(pairfold_layout)
        pairs = int(pairfold_layout.pair_blocks)
        for group in range(int(pairfold_layout.key_value_groups)):
            start = group * pairs
            outputs.append(
                _convert_value_group(
                    tuple(value_inputs[start : start + pairs]),
                    half_rotation=half_rotation,
                    mask_bundle=masks,
                    crypto_context=crypto_context,
                )
            )
        result = tuple(outputs)
        outputs = []
        success = True
        return result
    finally:
        if not success:
            for cipher in outputs:
                release_if_supported(cipher)
        for cipher in value_inputs:
            release_if_supported(cipher)


def prefill_real_value_c_group_to_pairfold_fhe(
    real_value_ciphers: tuple[object, ...] | list[object],
    *,
    pairfold_layout: PairFold128Layout,
    crypto_context,
    occupied: int = 128,
) -> object:
    """Convert one two-row real V_C group into a PairFold V ciphertext."""

    sources = tuple(real_value_ciphers)
    try:
        if len(sources) != int(pairfold_layout.pair_blocks):
            raise ValueError(
                "a PairFold V group requires "
                f"{pairfold_layout.pair_blocks} raw pairs, got {len(sources)}."
            )
        masks = _pair_mask_bundle(
            pairfold_layout, crypto_context=crypto_context, occupied=occupied
        )
        half_rotation, _ = prefill_c_to_pairfold_rotations(pairfold_layout)
        result = _convert_value_group(
            sources,
            half_rotation=half_rotation,
            mask_bundle=masks,
            crypto_context=crypto_context,
        )
        return result
    finally:
        for cipher in sources:
            release_if_supported(cipher)


__all__ = [
    "encrypt_pairfold_plaintext_page",
    "prefill_emitted_key_c_to_pairfold_fhe",
    "prefill_emitted_key_c_group_to_pairfold_fhe",
    "prefill_emitted_key_c_direct_to_pairfold_fhe",
    "prefill_emitted_key_c_direct_group_to_pairfold_fhe",
    "prefill_real_value_c_to_pairfold_fhe",
    "prefill_real_value_c_group_to_pairfold_fhe",
    "prefill_c_to_pairfold_fhe",
    "prefill_c_to_pairfold_numpy",
    "prefill_c_to_pairfold_rotations",
]
