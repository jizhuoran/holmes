"""Prefill PC attention: ciphertext ``Q_C`` / ``P_p`` × plaintext prefix K/V.

k is 2–4, so each prefix column is a C-form token broadcast encoded as a
plaintext and multiplied with ``homo_mul_pt``.  Feature reduce / broadcast
reuse the Decode C-grid butterflies (stride ``2S``).  EasyFHE is unchanged.
"""

from __future__ import annotations

import numpy as np

from llama3fhe.approx.prefix import broadcast_prefix_token_c
from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout

from ...primitives import accumulate_owned, encode_mask, mul_plain_rescale
from ..decode.decode import (
    _feature_broadcast_cipher,
    _feature_reduce_cipher,
    _homo_rotate,
    _mul_packed_rows,
    _pad_mask_row,
)


def prefix_pc_qk_rotations(layout: AttentionPairLayout, *, prefix_k: int) -> tuple[int, ...]:
    prefix_k = int(prefix_k)
    stride = 2 * int(layout.seq_len)
    place = tuple(-stride * index for index in range(1, prefix_k))
    return tuple(
        dict.fromkeys(
            layout.decode_qk_feature_rotations()
            + place
            + (4 * int(layout.slots) - 1,)
        )
    )


def prefix_pc_pv_rotations(layout: AttentionPairLayout, *, prefix_k: int) -> tuple[int, ...]:
    prefix_k = int(prefix_k)
    stride = 2 * int(layout.seq_len)
    shift = tuple(stride * index for index in range(1, prefix_k))
    return tuple(
        dict.fromkeys(layout.decode_feature_broadcast_rotations() + shift)
    )


def prefix_pc_rotations(layout: AttentionPairLayout, *, prefix_k: int) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            prefix_pc_qk_rotations(layout, prefix_k=prefix_k)
            + prefix_pc_pv_rotations(layout, prefix_k=prefix_k)
        )
    )


def _release_tuple(ciphers) -> None:
    for cipher in tuple(ciphers or ()):
        release_if_supported(cipher)


def _mul_plain_row(
    cipher,
    row: np.ndarray,
    *,
    name: str,
    crypto_context,
    plaintext_cache: dict[tuple[object, int], object] | None = None,
    cache_key: object | None = None,
):
    slots = int(getattr(cipher, "slots", np.asarray(row).size))
    padded = _pad_mask_row(row, slots=slots)
    level = int(crypto_context.level_for_cipher(cipher))
    key = (cache_key if cache_key is not None else name, level)
    plain = None
    if plaintext_cache is not None:
        plain = plaintext_cache.get(key)
    if plain is None:
        plain = encode_mask(
            padded, name=name, level=level, crypto_context=crypto_context
        )
        if plaintext_cache is not None:
            plaintext_cache[key] = plain
    try:
        return mul_plain_rescale(cipher, plain, crypto_context=crypto_context)
    finally:
        if plaintext_cache is None:
            release_if_supported(plain)


def _kv_token(values: np.ndarray, prefix_j: int, *, layout: AttentionPairLayout, name: str):
    values = np.asarray(values, dtype=np.float32)
    heads = int(layout.key_value_heads)
    dim = int(layout.head_dim)
    prefix_j = int(prefix_j)
    if values.ndim == 2 and values.shape[1] == heads * dim:
        return values[prefix_j]
    if values.ndim == 3 and values.shape[1:] == (heads, dim):
        return values[prefix_j]
    raise ValueError(f"{name} must be [k, Hkv*d] or [k, Hkv, d], got {values.shape}.")


def qk_prefix_pc_fhe(
    query_ciphers: tuple[object, ...] | list[object],
    key_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    prefix_k: int | None = None,
    complex_feature_folding: bool = False,
) -> tuple[object, ...]:
    """``S_p`` in C form: feature ``j`` holds ``⟨Q, K_p[j]⟩`` for every query token.

    Does not consume ``query_ciphers``.
    """

    query_ciphers = tuple(query_ciphers)
    n_pair = int(layout.query_pair_count)
    if len(query_ciphers) != n_pair:
        raise ValueError(
            f"PC QK expects {n_pair} Q_C ciphers, got {len(query_ciphers)}."
        )
    key_prefix = np.asarray(key_prefix, dtype=np.float32)
    prefix_k = int(key_prefix.shape[0] if prefix_k is None else prefix_k)
    if prefix_k < 1 or prefix_k > int(layout.head_dim):
        raise ValueError(f"prefix_k={prefix_k} outside [1, {layout.head_dim}].")
    ratio = int(layout.gqa_ratio)
    stride = 2 * int(layout.seq_len)
    row0 = layout.c_feature_row_mask(0)
    accumulators: list[object | None] = [None] * n_pair
    # One K prefix column is shared by ``gqa_ratio`` query pairs.  Reusing the
    # encoded plaintext across those products avoids a 4x duplicate CKKS
    # encode on the production 32:8 head geometry.  Plaintexts are released
    # once after all pairs have consumed them.
    plaintext_cache: dict[tuple[object, int], object] = {}
    try:
        for prefix_j in range(prefix_k):
            token = _kv_token(
                key_prefix, prefix_j, layout=layout, name="key_prefix"
            )
            k_plain = broadcast_prefix_token_c(token, layout=layout)
            for pair_index, query in enumerate(query_ciphers):
                kv_pair = pair_index // ratio
                product = reduced = conjugated = real_reduced = isolated = placed = None
                try:
                    product = _mul_plain_row(
                        query,
                        k_plain[kv_pair],
                        name=f"prefix.pc.qk.j{prefix_j}.p{pair_index}",
                        crypto_context=crypto_context,
                        plaintext_cache=plaintext_cache,
                        cache_key=("qk", prefix_j, kv_pair),
                    )
                    reduced = _feature_reduce_cipher(
                        product, layout=layout, crypto_context=crypto_context
                    )
                    if complex_feature_folding:
                        conjugated = crypto_context.fhe.homo_rotate(
                            reduced,
                            int(crypto_context.context.M) - 1,
                            crypto_context.context,
                        )
                        real_reduced = crypto_context.fhe.homo_add(
                            reduced, conjugated, crypto_context.context
                        )
                        release_if_supported(reduced)
                        reduced = real_reduced
                        real_reduced = None
                    isolated = _mul_packed_rows(
                        (reduced,),
                        row0,
                        name=f"prefix.pc.qk.row0.j{prefix_j}.p{pair_index}",
                        crypto_context=crypto_context,
                    )[0]
                    placed = _homo_rotate(
                        isolated,
                        -stride * int(prefix_j),
                        crypto_context=crypto_context,
                    )
                    accumulators[pair_index] = accumulate_owned(
                        accumulators[pair_index],
                        placed,
                        crypto_context=crypto_context,
                    )
                    placed = None
                finally:
                    release_if_supported(product)
                    release_if_supported(reduced)
                    release_if_supported(conjugated)
                    release_if_supported(real_reduced)
                    release_if_supported(isolated)
                    release_if_supported(placed)
        if any(item is None for item in accumulators):
            raise RuntimeError("PC QK produced an empty pair ciphertext.")
        result = tuple(accumulators)
        accumulators = []
        return result
    finally:
        _release_tuple(accumulators)
        for plain in plaintext_cache.values():
            release_if_supported(plain)


def pv_prefix_pc_fhe(
    probability_ciphers: tuple[object, ...] | list[object],
    value_prefix: np.ndarray,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    prefix_k: int | None = None,
) -> tuple[object, ...]:
    """``O_p`` in Q_C form: ``Σ_j broadcast(P_p[j]) ⊙ broadcast(V_p[j])``.

    Does not consume ``probability_ciphers``.
    """

    probability_ciphers = tuple(probability_ciphers)
    n_pair = int(layout.query_pair_count)
    if len(probability_ciphers) != n_pair:
        raise ValueError(
            f"PC PV expects {n_pair} P_p ciphers, got {len(probability_ciphers)}."
        )
    value_prefix = np.asarray(value_prefix, dtype=np.float32)
    prefix_k = int(value_prefix.shape[0] if prefix_k is None else prefix_k)
    if prefix_k < 1 or prefix_k > int(layout.head_dim):
        raise ValueError(f"prefix_k={prefix_k} outside [1, {layout.head_dim}].")
    ratio = int(layout.gqa_ratio)
    stride = 2 * int(layout.seq_len)
    accumulators: list[object | None] = [None] * n_pair
    plaintext_cache: dict[tuple[object, int], object] = {}
    try:
        for prefix_j in range(prefix_k):
            token = _kv_token(
                value_prefix, prefix_j, layout=layout, name="value_prefix"
            )
            v_plain = broadcast_prefix_token_c(token, layout=layout)
            row_mask = layout.c_feature_row_mask(prefix_j)
            for pair_index, probability in enumerate(probability_ciphers):
                kv_pair = pair_index // ratio
                isolated = shifted = broadcast = product = None
                try:
                    isolated = _mul_packed_rows(
                        (probability,),
                        row_mask,
                        name=f"prefix.pc.pv.row{prefix_j}.p{pair_index}",
                        crypto_context=crypto_context,
                    )[0]
                    shifted = _homo_rotate(
                        isolated,
                        stride * int(prefix_j),
                        crypto_context=crypto_context,
                    )
                    broadcast = _feature_broadcast_cipher(
                        shifted, layout=layout, crypto_context=crypto_context
                    )
                    product = _mul_plain_row(
                        broadcast,
                        v_plain[kv_pair],
                        name=f"prefix.pc.pv.v{prefix_j}.p{pair_index}",
                        crypto_context=crypto_context,
                        plaintext_cache=plaintext_cache,
                        cache_key=("pv", prefix_j, kv_pair),
                    )
                    accumulators[pair_index] = accumulate_owned(
                        accumulators[pair_index],
                        product,
                        crypto_context=crypto_context,
                    )
                    product = None
                finally:
                    release_if_supported(isolated)
                    release_if_supported(shifted)
                    release_if_supported(broadcast)
                    release_if_supported(product)
        if any(item is None for item in accumulators):
            raise RuntimeError("PC PV produced an empty pair ciphertext.")
        result = tuple(accumulators)
        accumulators = []
        return result
    finally:
        _release_tuple(accumulators)
        for plain in plaintext_cache.values():
            release_if_supported(plain)


__all__ = [
    "prefix_pc_pv_rotations",
    "prefix_pc_qk_rotations",
    "prefix_pc_rotations",
    "pv_prefix_pc_fhe",
    "qk_prefix_pc_fhe",
]
