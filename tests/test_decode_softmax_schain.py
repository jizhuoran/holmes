"""``k = 2`` merged-page Decode Softmax as the S-chain (one aux bootstrap).

Same probabilities as the Alg2 merged kernel (``P = (y0 lam1)^4 lam2^2`` is
``y0^4 * (lam1^2 lam2)^2``), two bootstraps instead of three (post-exp plus
the co-packed ``T`` / ``lam1`` refresh), and the probability leaves with
enough limbs for PV.
"""

from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.backend import release_if_supported
from llama3fhe.operators.attention.decode.decode_softmax import (
    SCHAIN_PROBABILITY_MIN_LIMBS,
    SOFTMAX_MERGE_PAGES_ENV,
    SOFTMAX_SCHAIN_ENV,
    DecodeAlg2Poly,
    schain_applicable,
    schain_limb_schedule,
    softmax_alg2_row_pages_fhe,
    softmax_schain_enabled,
)

from test_decode_softmax_merged_pages import (
    _context,
    _CountingRefresh,
    _decrypt,
    _layout,
)


def test_schain_env_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SOFTMAX_SCHAIN_ENV, raising=False)
    assert not softmax_schain_enabled()
    monkeypatch.setenv(SOFTMAX_SCHAIN_ENV, "1")
    assert softmax_schain_enabled()


def test_schain_limb_schedule_and_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SOFTMAX_SCHAIN_ENV, "1")
    config = DecodeAlg2Poly.synthetic(k=2, degree=8)  # PS depth 4 (+1 for the unfused map)
    plan = schain_limb_schedule(config, 0, exit_limbs=18)
    assert plan == {"lam1": 18 - 2 - 5, "pack": 18 - 2 - 5 - 3, "probability": 18 - 4 - 4}
    assert schain_applicable(config, 0, 2, exit_limbs=18)
    assert not schain_applicable(config, 0, 1, exit_limbs=18)  # k != 2
    assert not schain_applicable(config, 0, 3, exit_limbs=18)
    # too shallow an exit level: the probability would need a refresh -> keep Alg2
    assert not schain_applicable(config, 0, 2, exit_limbs=4 + 4 + SCHAIN_PROBABILITY_MIN_LIMBS - 1)
    monkeypatch.setenv(SOFTMAX_SCHAIN_ENV, "0")
    assert not schain_applicable(config, 0, 2, exit_limbs=18)


def _run(context, layout, pages, config, *, occupied, head_rows, bootstrap):
    probs = softmax_alg2_row_pages_fhe(
        pages,
        layout=layout,
        crypto_context=context,
        config=config,
        occupied_pages=occupied,
        layer_idx=0,
        rounds=None,
        bootstrap_operator=bootstrap,
        bootstrap_iterations=1,
        bootstrap_precision_bits=0,
        head_rows=head_rows,
    )
    try:
        limbs = tuple(int(c.state.cur_limbs) for page in probs for c in page)
        return tuple(_decrypt(page, layout=layout, context=context) for page in probs), limbs
    finally:
        for page in probs:
            for cipher in page:
                release_if_supported(cipher)


@pytest.mark.gpu
@pytest.mark.parametrize("head_rows", [None, (0, 2)])
def test_schain_matches_alg2_merged_with_one_fewer_bootstrap(
    monkeypatch: pytest.MonkeyPatch, head_rows
) -> None:
    rng = np.random.default_rng(26100401)
    layout = _layout()
    occupied = (4, 2)
    scores = [rng.normal(scale=0.3, size=(4, 4)).astype(np.float32) for _ in occupied]
    if head_rows is not None:
        for page in scores:
            page[[1, 3], :] = 0.0
    config = DecodeAlg2Poly.synthetic(k=2, degree=8)
    context = _context(layout, depth=40)
    ciphers = tuple(
        tuple(context.encrypt(row, level=0, slots=layout.slots) for row in layout.pack_compact_decode_scores(s))
        for s in scores
    )
    monkeypatch.setenv(SOFTMAX_MERGE_PAGES_ENV, "1")
    try:
        monkeypatch.setenv(SOFTMAX_SCHAIN_ENV, "0")
        counter = _CountingRefresh(context, limbs=18, slots=layout.slots)
        alg2, _ = _run(context, layout, ciphers, config, occupied=occupied, head_rows=head_rows, bootstrap=counter)
        alg2_calls = counter.calls
        monkeypatch.setenv(SOFTMAX_SCHAIN_ENV, "1")
        counter = _CountingRefresh(context, limbs=18, slots=layout.slots)
        schain, limbs = _run(
            context, layout, ciphers, config, occupied=occupied, head_rows=head_rows, bootstrap=counter
        )
        schain_calls = counter.calls
    finally:
        for page in ciphers:
            for cipher in page:
                release_if_supported(cipher)
    assert alg2_calls == 3  # post_exp + r1.lam + r2.lam
    assert schain_calls == 2  # post_exp + co-packed pack
    plan = schain_limb_schedule(config, 0, exit_limbs=18)
    assert all(limb == plan["probability"] for limb in limbs), (limbs, plan)
    assert plan["probability"] >= SCHAIN_PROBABILITY_MIN_LIMBS
    assert len(schain) == len(alg2) == 2
    for got, want, occ in zip(schain, alg2, occupied, strict=True):
        np.testing.assert_allclose(got, want, atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(got[:, occ:], 0.0, atol=1e-3)
        if head_rows is not None:
            np.testing.assert_allclose(got[[1, 3], :], 0.0, atol=1e-3)
    rows = np.concatenate([page[:, :occ] for page, occ in zip(schain, occupied, strict=True)], axis=1)
    owned = list(head_rows) if head_rows is not None else list(range(4))
    np.testing.assert_allclose(rows[owned].sum(axis=1), 1.0, atol=0.25)  # synthetic degree-8 polys
