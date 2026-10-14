from __future__ import annotations

"""Packed NumPy oracles for the Decode-only PairFold128 contract."""

import numpy as np

from llama3fhe.layouts.pairfold128 import PairFold128Layout


def _occupied(layout: PairFold128Layout, occupied: int | None) -> int:
    occupied = int(layout.seq_len if occupied is None else occupied)
    if not 0 <= occupied <= int(layout.seq_len):
        raise ValueError(f"occupied={occupied} outside [0, {layout.seq_len}].")
    return occupied


def _require_packed(
    packed: np.ndarray, *, count: int, layout: PairFold128Layout, name: str
) -> np.ndarray:
    packed = np.asarray(packed)
    expected = (int(count), int(layout.slots))
    if packed.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {packed.shape}.")
    return packed


def qk_pairfold128_numpy(
    query_ciphers: np.ndarray,
    key_ciphers: np.ndarray,
    *,
    layout: PairFold128Layout,
    occupied: int | None = None,
) -> np.ndarray:
    """Folded Q x K, returning the legacy-compatible compact score payload."""

    query_ciphers = _require_packed(
        query_ciphers,
        count=layout.query_residue_count,
        layout=layout,
        name="PairFold128 query",
    )
    key_ciphers = _require_packed(
        key_ciphers,
        count=layout.key_value_groups,
        layout=layout,
        name="PairFold128 key",
    )
    occupied = _occupied(layout, occupied)
    real_dtype = np.result_type(query_ciphers.real.dtype, key_ciphers.real.dtype)
    packed = np.zeros((1, int(layout.slots)), dtype=real_dtype)
    features = range(layout.fold_dim)
    tokens = np.arange(occupied, dtype=np.int64)

    for group in range(int(layout.key_value_groups)):
        for query_offset in range(layout.gqa_ratio):
            query_cipher = layout.query_cipher_index(group, query_offset)
            for pair in range(layout.pair_blocks):
                for lane in range(layout.head_lanes):
                    query_slots = np.fromiter(
                        (
                            layout.page_slot(pair, feature, 0, lane)
                            for feature in features
                        ),
                        dtype=np.int64,
                        count=layout.fold_dim,
                    )
                    key_slots = np.asarray(
                        [
                            [
                                layout.page_slot(pair, feature, int(token), lane)
                                for token in tokens
                            ]
                            for feature in range(layout.fold_dim)
                        ],
                        dtype=np.int64,
                    )
                    folded_query = query_ciphers[query_cipher, query_slots]
                    folded_key = key_ciphers[group, key_slots]
                    scores = 2.0 * np.real(
                        np.einsum(
                            "f,ft->t", folded_query, folded_key, optimize=True
                        )
                    )
                    score_slots = np.fromiter(
                        (
                            layout.score_slot(
                                group,
                                pair,
                                query_offset,
                                int(token),
                                lane,
                            )
                            for token in tokens
                        ),
                        dtype=np.int64,
                        count=occupied,
                    )
                    packed[0, score_slots] = scores
    return packed


def pv_pairfold128_numpy(
    probability_ciphers: np.ndarray,
    value_ciphers: np.ndarray,
    *,
    layout: PairFold128Layout,
    occupied: int | None = None,
) -> np.ndarray:
    """Compact probabilities x folded V, returning one folded-O carrier."""

    probability_ciphers = _require_packed(
        probability_ciphers,
        count=1,
        layout=layout,
        name="PairFold128 probability",
    )
    value_ciphers = _require_packed(
        value_ciphers,
        count=layout.key_value_groups,
        layout=layout,
        name="PairFold128 value",
    )
    occupied = _occupied(layout, occupied)
    complex_dtype = np.result_type(value_ciphers.dtype, np.complex64)
    packed = np.zeros((1, int(layout.slots)), dtype=complex_dtype)
    tokens = np.arange(occupied, dtype=np.int64)

    for group in range(int(layout.key_value_groups)):
        for query_offset in range(layout.gqa_ratio):
            for pair in range(layout.pair_blocks):
                for lane in range(layout.head_lanes):
                    probability_slots = np.fromiter(
                        (
                            layout.score_slot(
                                group,
                                pair,
                                query_offset,
                                int(token),
                                lane,
                            )
                            for token in tokens
                        ),
                        dtype=np.int64,
                        count=occupied,
                    )
                    probabilities = probability_ciphers[
                        0, probability_slots
                    ].real
                    value_slots = np.asarray(
                        [
                            [
                                layout.page_slot(pair, feature, int(token), lane)
                                for token in tokens
                            ]
                            for feature in range(layout.fold_dim)
                        ],
                        dtype=np.int64,
                    )
                    folded_value = value_ciphers[group, value_slots]
                    folded_output = np.einsum(
                        "t,ft->f", probabilities, folded_value, optimize=True
                    )
                    for feature, value in enumerate(folded_output):
                        packed[
                            0,
                            layout.output_slot(
                                group,
                                pair,
                                query_offset,
                                feature,
                                lane,
                            ),
                        ] = value
    return packed


def folded_w_o_pairfold128_numpy(
    folded_output: np.ndarray,
    weight: np.ndarray,
    *,
    layout: PairFold128Layout,
) -> np.ndarray:
    """Apply dense W_O through the folded complex identity.

    For ``z=x_low+i*x_high`` and
    ``A=.5*(W_low-i*W_high)``, ``A*z+conj(A*z)`` equals the real dense
    projection. The returned shape is one logical DecodeFM row ``[1,4096]``.
    """

    logical = layout.unpack_output(folded_output)
    weight = np.asarray(weight)
    dimension = int(layout.query_heads) * int(layout.head_dim)
    if weight.shape != (dimension, dimension):
        raise ValueError(
            f"W_O must have shape {(dimension, dimension)}, got {weight.shape}."
        )
    if np.iscomplexobj(weight):
        raise TypeError(f"W_O must be real, got {weight.dtype}.")
    folded = logical[:, : layout.fold_dim] + 1j * logical[:, layout.fold_dim :]
    weight_heads = weight.reshape(
        dimension, int(layout.query_heads), int(layout.head_dim)
    )
    low_projection = np.einsum(
        "ohf,hf->o",
        weight_heads[:, :, : layout.fold_dim],
        folded,
        optimize=True,
    )
    high_projection = np.einsum(
        "ohf,hf->o",
        weight_heads[:, :, layout.fold_dim :],
        folded,
        optimize=True,
    )
    projected = 0.5 * (low_projection - 1j * high_projection)
    return np.asarray((projected + np.conjugate(projected)).real).reshape(1, -1)


__all__ = [
    "folded_w_o_pairfold128_numpy",
    "pv_pairfold128_numpy",
    "qk_pairfold128_numpy",
]
