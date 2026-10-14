"""GQA shift split: partial QK / PV accumulations over complementary shift ranges
add up (slot-wise, before relinearisation) to the full kernel's output.

This is the correctness contract behind splitting the K-/V-side rotation work
of one K pair between the two ranks that share it (W=8: ranks 2g, 2g+1).  The
kernels are linear in their triplet accumulators, so any payload will do; the
test uses random real payloads and compares decrypted outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from llama3fhe import Llama3CKKSConfig
from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.attention.audit import delta_attention_rotations
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.operators.attention.prefill.pv import (
    finalize_pv_accumulators,
    pv_delta_fhe,
    pv_shift_count,
)
from llama3fhe.operators.attention.prefill.qk import (
    finalize_qk_accumulators,
    qk_delta_fhe,
    qk_shift_count,
)

LOG_N = 14


def _layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=32, head_dim=32, query_heads=8, key_value_heads=2, slots=1 << (LOG_N - 1)
    )


def _config() -> AttentionOperatorConfig:
    # the production shape: folded, pre-folded K / pre-scaled+pre-folded Q, complex PV fold
    return AttentionOperatorConfig(
        qk_output_level=None,
        qk_complex_feature_folding=True,
        qk_key_features_pre_folded=True,
        qk_query_features_pre_scaled=True,
        qk_query_features_pre_folded=True,
        pv_complex_delta_folding=True,
    )


def _context(layout: AttentionPairLayout, *, depth: int = 12):
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = LOG_N
    cfg.simulator.maxLevelsRemaining = int(depth)
    cfg.simulator.device = "cuda"
    rotations = delta_attention_rotations(
        layout,
        include_qk_feature_fold=True,
        qk_key_features_pre_folded=True,
        qk_query_features_pre_folded=True,
        include_pv_delta_fold=True,
        pv_probability_pre_folded=True,
        pv_value_pre_folded=True,
        log_n=LOG_N,
    )
    return create_sim_context(
        cfg.simulator,
        input_dim=int(layout.head_dim),
        block_out=int(layout.seq_len),
        rotations=rotations,
        include_default_rotations=False,
    )


def _encrypt_many(context, layout, count: int, rng, *, level: int) -> tuple[object, ...]:
    return tuple(
        context.encrypt(rng.uniform(-1.0, 1.0, size=int(layout.slots)), level=int(level), slots=int(layout.slots))
        for _ in range(int(count))
    )


def _decrypt(context, ciphers) -> np.ndarray:
    return np.stack([np.asarray(context.decrypt(c), dtype=np.complex128).real for c in ciphers])


def _split_matches_full(kernel, finalize, shift_count: int, cuts: tuple[int, ...], *, lhs, rhs, layout, context, cfg, **kw):
    full = kernel(lhs, rhs, layout=layout, crypto_context=context, operator_config=cfg, **kw)
    bounds = (0,) + tuple(cuts) + (shift_count,)
    partials = []
    try:
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            partials.append(
                kernel(lhs, rhs, layout=layout, crypto_context=context, operator_config=cfg,
                       shift_range=(lo, hi), finalize=False, **kw)
            )
        summed = []
        for index in range(len(full)):
            acc = partials[0][index]
            partials[0] = tuple(p if i != index else None for i, p in enumerate(partials[0]))
            for other in partials[1:]:
                nxt = context.fhe.homo_add(acc, other[index], context.context)
                release_if_supported(acc)
                acc = nxt
            summed.append(acc)
        merged = finalize(summed, crypto_context=context, **({"operator_config": cfg} if finalize is finalize_qk_accumulators else {}))
        a = _decrypt(context, full)
        b = _decrypt(context, merged)
        for c in merged:
            release_if_supported(c)
        assert [int(c.state.cur_limbs) for c in full] == [int(c.state.cur_limbs) for c in merged]
        scale = float(np.abs(a).max())
        assert scale > 1e-3, "degenerate output; the comparison would be vacuous"
        rel = float(np.abs(a - b).max()) / scale
        print(f"shift split {kernel.__name__} cuts={cuts}: |full|max={scale:.3g} rel_diff={rel:.3g}")
        return rel
    finally:
        for c in full:
            release_if_supported(c)
        for group in partials:
            for c in group:
                if c is not None:
                    release_if_supported(c)


@pytest.mark.gpu
@pytest.mark.parametrize("cuts", [(8,), (5,), (4, 11)])
def test_qk_shift_split_sums_to_full_kernel(cuts) -> None:
    layout = _layout()
    cfg = _config()
    context = _context(layout)
    rng = np.random.default_rng(26100201)
    queries = _encrypt_many(context, layout, layout.query_pair_count, rng, level=2)
    keys = _encrypt_many(context, layout, layout.key_value_pair_count, rng, level=2)
    try:
        count = qk_shift_count(layout, operator_config=cfg)
        assert count == layout.head_dim // 2
        rel = _split_matches_full(
            qk_delta_fhe, finalize_qk_accumulators, count, cuts,
            lhs=queries, rhs=keys, layout=layout, context=context, cfg=cfg,
        )
        assert rel < 1e-5, rel
    finally:
        for c in queries + keys:
            release_if_supported(c)


@pytest.mark.gpu
@pytest.mark.parametrize("cuts", [(8,), (3,), (6, 13)])
def test_pv_shift_split_sums_to_full_kernel(cuts) -> None:
    layout = _layout()
    cfg = _config()
    context = _context(layout)
    rng = np.random.default_rng(26100202)
    probabilities = _encrypt_many(context, layout, layout.query_pair_count, rng, level=2)
    values = _encrypt_many(context, layout, layout.key_value_pair_count, rng, level=2)
    try:
        count = pv_shift_count(layout)
        assert count == layout.seq_len // 2
        rel = _split_matches_full(
            pv_delta_fhe, finalize_pv_accumulators, count, cuts,
            lhs=probabilities, rhs=values, layout=layout, context=context, cfg=cfg,
            inputs_pre_folded=True,
        )
        assert rel < 1e-5, rel
    finally:
        for c in probabilities + values:
            release_if_supported(c)


def test_shift_range_validation() -> None:
    from llama3fhe.operators.attention.prefill.qk import _resolve_shift_range

    assert _resolve_shift_range(None, 64) == (0, 64)
    assert _resolve_shift_range((32, 64), 64) == (32, 64)
    for bad in ((0, 0), (-1, 4), (4, 65), (10, 5)):
        with pytest.raises(ValueError):
            _resolve_shift_range(bad, 64)
