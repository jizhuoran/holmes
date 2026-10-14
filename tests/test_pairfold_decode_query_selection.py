"""CPU tests: ``query_selection`` partials of the PairFold QK/PV kernels add up to the full result.

The multi-card decode step gives every rank a subset of the eight
``(group, query_offset)`` streams; the compact score pages and the folded-O
carrier it produces are partials whose sum over the ranks must equal the
single-card result exactly (disjoint slots / disjoint residues).
"""

from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.pairfold.pairfold_decode import (
    pv_pairfold_pages_fhe,
    qk_pairfold_pages_fhe,
    qk_pairfold_pages_from_queries_fhe,
)
from llama3fhe.parallel.decode.decode_ownership import decode_world
from test_pairfold_decode import _FakeContext


def _partition(world_size):
    return [own.query_streams for own in decode_world(world_size)]


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_qk_partials_over_query_selection_sum_to_full(world_size) -> None:
    rng = np.random.default_rng(20260930 + world_size)
    layout = PairFold128Layout()
    occupied_pages = (128, 5)
    packed_query = layout.pack_query(rng.normal(scale=0.2, size=(layout.query_heads, layout.head_dim)))
    packed_keys = [
        layout.pack_key(rng.normal(scale=0.2, size=(layout.seq_len, layout.key_value_heads, layout.head_dim)))
        for _ in occupied_pages
    ]
    context = _FakeContext(slots=layout.slots)
    carrier = context.cipher(np.zeros(layout.slots), label="carrier")
    key_pages = tuple(
        tuple(context.cipher(row, label=f"k{page}.{group}") for group, row in enumerate(packed))
        for page, packed in enumerate(packed_keys)
    )
    query_ciphers = tuple(context.cipher(row, label=f"q{index}") for index, row in enumerate(packed_query))
    full = qk_pairfold_pages_from_queries_fhe(
        query_ciphers, key_pages, layout=layout, crypto_context=context, occupied_pages=occupied_pages
    )
    expected = [page[0].values.copy() for page in full]

    def extractor(carrier_cipher, *, group, query_offset):
        assert carrier_cipher is carrier
        row = packed_query[layout.query_cipher_index(group, query_offset)]
        return context.cipher(row, label=f"extract{group}{query_offset}")

    total = [np.zeros_like(page) for page in expected]
    extract_calls = 0
    for streams in _partition(world_size):
        profile: dict[str, float] = {}
        pages = qk_pairfold_pages_fhe(
            carrier,
            key_pages,
            query_extractor=extractor,
            layout=layout,
            crypto_context=context,
            occupied_pages=occupied_pages,
            profile=profile,
            query_selection=streams,
        )
        assert profile["query_extract_count"] == len(streams) == 8 // world_size
        assert profile["ct_ct"] == len(streams) * len(occupied_pages)
        extract_calls += len(streams)
        for index, page in enumerate(pages):
            total[index] += page[0].values
            release_if_supported(page[0])
    assert extract_calls == 8
    for got, want in zip(total, expected, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("world_size", [2, 8])
def test_pv_partials_over_query_selection_sum_to_full(world_size) -> None:
    rng = np.random.default_rng(20260931 + world_size)
    layout = PairFold128Layout()
    occupied_pages = (128, 3)
    packed_probabilities = [
        layout.pack_scores(rng.uniform(0.0, 0.02, size=(layout.query_heads, layout.seq_len)))
        for _ in occupied_pages
    ]
    packed_values = [
        layout.pack_value(rng.normal(scale=0.2, size=(layout.seq_len, layout.key_value_heads, layout.head_dim)))
        for _ in occupied_pages
    ]
    context = _FakeContext(slots=layout.slots)
    probability_pages = tuple(
        (context.cipher(row[0], cur_limbs=10, label=f"p{index}"),) for index, row in enumerate(packed_probabilities)
    )
    value_pages = tuple(
        tuple(context.cipher(row, cur_limbs=9, label=f"v{page}.{group}") for group, row in enumerate(packed))
        for page, packed in enumerate(packed_values)
    )
    (full,) = pv_pairfold_pages_fhe(
        probability_pages, value_pages, layout=layout, crypto_context=context, occupied_pages=occupied_pages
    )
    expected = full.values.copy()
    release_if_supported(full)

    total = np.zeros_like(expected)
    for streams in _partition(world_size):
        profile: dict[str, float] = {}
        (partial,) = pv_pairfold_pages_fhe(
            probability_pages,
            value_pages,
            layout=layout,
            crypto_context=context,
            occupied_pages=occupied_pages,
            profile=profile,
            query_selection=streams,
        )
        assert profile["token_reductions"] == len(streams)
        total += partial.values
        release_if_supported(partial)
    np.testing.assert_allclose(total, expected, atol=1e-12, rtol=1e-12)


def test_query_selection_validation() -> None:
    layout = PairFold128Layout()
    context = _FakeContext(slots=layout.slots)
    carrier = context.cipher(np.zeros(layout.slots), label="carrier")
    key_page = tuple(context.cipher(np.zeros(layout.slots), label=f"k{g}") for g in range(layout.key_value_groups))
    for bad, error in (((), ValueError), (((0, 0), (0, 0)), ValueError), (((2, 0),), IndexError), (((0, 4),), IndexError)):
        with pytest.raises(error):
            qk_pairfold_pages_fhe(
                carrier,
                (key_page,),
                query_extractor=lambda c, group, query_offset: context.cipher(np.zeros(layout.slots), label="q"),
                layout=layout,
                crypto_context=context,
                occupied_pages=(1,),
                query_selection=bad,
            )
