"""CPU tests: multi-card decode softmax must not treat the other ranks' zero head rows as real rows.

With ``query_selection`` a rank's compact score pages carry only its own
``(group, query_offset)`` streams; the other query-head rows are exactly zero.
A zero row is a *uniform* softmax row whose ``sum(y^2)`` leaves the per-layer
``1/sqrt`` fit interval (production layer 2 / 23 blow up to 1e100+), and the
garbage then poisons every slot of the ciphertext.  ``head_rows`` tells the
Alg2 kernel which rows are real; the others are forced to zero probability and
their ``1/sqrt`` input is the safe fill constant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from llama3fhe.application.args import DEFAULT_SOFTMAX
from llama3fhe.approx.softmax import SoftmaxPolyConfig, load_softmax_poly_coeffs
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.decode import decode_softmax
from llama3fhe.operators.attention.decode.decode_softmax import (
    apply_decode_alg2_public_map_numpy,
    softmax_alg2_poly_numpy,
)
from llama3fhe.operators.attention.pairfold.pairfold_decode import query_selection_head_rows
from llama3fhe.parallel.decode.decode_ownership import decode_world


@pytest.fixture(scope="module")
def production_softmax():
    return SoftmaxPolyConfig.from_artifact(load_softmax_poly_coeffs(DEFAULT_SOFTMAX))


def _in_range_scores(config, *, layer_idx: int, seed: int) -> np.ndarray:
    """[32, 128] raw scores whose full Alg2 stays inside every fit interval."""

    rng = np.random.default_rng(seed)
    u_hat = float(config.U_hat_for(layer_idx))
    stats_min = float(config.stats_min_for(layer_idx))
    scaled = np.clip(rng.normal(loc=u_hat - 14.0, scale=3.0, size=(32, 128)), stats_min, u_hat)
    scaled[:, :2] = u_hat - rng.uniform(0.0, 1.0, size=(32, 2))
    return scaled * math.sqrt(128.0)


def _owned_mask(rows) -> np.ndarray:
    mask = np.zeros((32,), dtype=bool)
    mask[list(rows)] = True
    return mask


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_query_selection_head_rows_partition_the_32_heads(world_size) -> None:
    layout = PairFold128Layout()
    rows = [query_selection_head_rows(own.query_streams, layout=layout) for own in decode_world(world_size)]
    assert all(len(r) == 32 // world_size for r in rows)
    flat = sorted(sum(rows, ()))
    assert flat == list(range(32))
    # rank 0 of an 8-card world owns stream (0, 0) = query heads 0, 4, 8, 12
    if world_size == 8:
        assert rows[0] == (0, 4, 8, 12)
    assert query_selection_head_rows(None, layout=layout) == tuple(range(32))


def test_zero_head_rows_without_head_rows_blow_up_at_layer_2(production_softmax) -> None:
    """Documents the production failure: the uniform zero rows leave the j1 interval."""

    layout = AttentionPairLayout()
    scores = _in_range_scores(production_softmax, layer_idx=2, seed=2)
    rows = query_selection_head_rows(decode_world(8)[0].query_streams, layout=PairFold128Layout())
    partial = np.where(_owned_mask(rows)[:, None], scores, 0.0)
    _, report = softmax_alg2_poly_numpy(partial, layout=layout, config=production_softmax, layer_idx=2, occupied=128)
    assert not report["invsqrt_rounds"][0]["in_range"]
    assert report["invsqrt_rounds"][0]["max"] > report["invsqrt_rounds"][0]["interval"][1]


@pytest.mark.parametrize("layer_idx", [1, 2, 3, 14, 31])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_head_rows_partials_match_full_softmax_rows(production_softmax, layer_idx, world_size) -> None:
    layout = AttentionPairLayout()
    pairfold = PairFold128Layout()
    scores = _in_range_scores(production_softmax, layer_idx=layer_idx, seed=layer_idx)
    full, report = softmax_alg2_poly_numpy(
        scores, layout=layout, config=production_softmax, layer_idx=layer_idx, occupied=128
    )
    assert report["exp_in_range"] and all(r["in_range"] for r in report["invsqrt_rounds"])
    for own in decode_world(world_size):
        rows = query_selection_head_rows(own.query_streams, layout=pairfold)
        owned = _owned_mask(rows)
        partial = np.where(owned[:, None], scores, 0.0)
        probabilities, partial_report = softmax_alg2_poly_numpy(
            partial,
            layout=layout,
            config=production_softmax,
            layer_idx=layer_idx,
            occupied=128,
            head_rows=rows,
        )
        assert np.all(np.isfinite(probabilities))
        assert all(r["in_range"] for r in partial_report["invsqrt_rounds"])
        assert partial_report["exp_in_range"]
        np.testing.assert_array_equal(probabilities[~owned], 0.0)
        np.testing.assert_array_equal(probabilities[owned], full[owned])


def test_public_map_numpy_marks_other_head_rows_invalid(production_softmax) -> None:
    layout = AttentionPairLayout()
    scores = _in_range_scores(production_softmax, layer_idx=2, seed=5)
    rows = (0, 4, 8, 12)
    full = apply_decode_alg2_public_map_numpy(scores, layout=layout, config=production_softmax, layer_idx=2, occupied=100)
    masked = apply_decode_alg2_public_map_numpy(
        scores, layout=layout, config=production_softmax, layer_idx=2, occupied=100, head_rows=rows
    )
    owned = _owned_mask(rows)
    np.testing.assert_array_equal(masked[owned], full[owned])
    k = production_softmax.k_for(2)
    invalid = (production_softmax.stats_min_for(2) - production_softmax.U_hat_for(2)) / 2**k
    np.testing.assert_allclose(masked[~owned], np.float32(invalid), rtol=1e-6)
    # the invalid columns of owned rows are unchanged as well
    np.testing.assert_allclose(masked[owned][:, 100:], np.float32(invalid), rtol=1e-6)


def test_packed_masks_restrict_to_owned_rows_and_fill_their_column_zero(production_softmax) -> None:
    layout = AttentionPairLayout()
    rows = (1, 5, 9, 13)
    head_mask = decode_softmax._head_rows_mask(layout, rows)
    unpacked = layout.unpack_compact_decode_scores(head_mask)
    owned = _owned_mask(rows)
    np.testing.assert_array_equal(unpacked[owned], 1.0)
    np.testing.assert_array_equal(unpacked[~owned], 0.0)

    prefix = decode_softmax._with_head_rows(layout.compact_decode_prefix_mask(7), head_mask)
    prefix_rows = layout.unpack_compact_decode_scores(prefix)
    assert prefix_rows[owned, :7].all() and not prefix_rows[owned, 7:].any()
    assert not prefix_rows[~owned].any()

    poly = production_softmax.poly_for_layer(production_softmax.invsqrt_alg2["j1_rough"], 2)
    fill = decode_softmax._safe_invsqrt_fill(poly)
    with_rows = layout.unpack_compact_decode_scores(
        decode_softmax._invsqrt_fill_mask(layout, poly, head_mask=head_mask)
    )
    without_rows = layout.unpack_compact_decode_scores(
        decode_softmax._invsqrt_fill_mask(layout, poly, head_mask=None)
    )
    # owned rows: column 0 keeps the real row sum (no fill), other columns filled
    np.testing.assert_array_equal(with_rows[owned, 0], 0.0)
    np.testing.assert_allclose(with_rows[owned, 1:], fill)
    # other rows: every column (including column 0, whose sum is zero) is filled
    np.testing.assert_allclose(with_rows[~owned], fill)
    # the single-card mask is unchanged
    np.testing.assert_array_equal(without_rows[:, 0], 0.0)
    np.testing.assert_allclose(without_rows[:, 1:], fill)
    assert decode_softmax._head_rows_mask(layout, None) is None
    assert decode_softmax._with_head_rows(prefix, None) is prefix


def test_head_rows_validation() -> None:
    layout = AttentionPairLayout()
    with pytest.raises(ValueError):
        decode_softmax._head_rows_mask(layout, ())
    with pytest.raises(IndexError):
        decode_softmax._head_rows_mask(layout, (0, 32))
