from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.attention.prefix.fused_delta import (
    FusedPrefixDeltaPlan,
    _minimum_no_refresh_input_limbs,
    fused_prefix_rotations,
)


@pytest.mark.parametrize(
    ("rounds", "checkpoint_after_round", "required_limbs"),
    [(2, None, 21), (4, None, 25), (4, 4, 22), (5, 0, 14)],
)
def test_no_pre_exp_refresh_budget_follows_main_checkpoint(
    rounds: int, checkpoint_after_round: int | None, required_limbs: int
) -> None:
    assert _minimum_no_refresh_input_limbs(
        exp_depth=5, rounds=rounds,
        checkpoint_after_round=checkpoint_after_round,
    ) == required_limbs


@pytest.mark.parametrize("seq", [8, 128])
def test_fused_delta_routing_matches_dual_alg2(seq: int) -> None:
    layout = AttentionPairLayout(
        seq_len=seq,
        head_dim=seq,
        query_heads=4,
        key_value_heads=2,
        slots=2 * seq * seq,
    )
    plan = FusedPrefixDeltaPlan.build(layout, prefix_k=2)
    rng = np.random.default_rng(20261006 + seq)
    prefix_scores = rng.normal(scale=0.3, size=(4, seq, 2)).astype(np.float32)
    content_scores = rng.normal(scale=0.3, size=(4, seq, seq)).astype(np.float32)
    prefix_c = layout.pack_prefix_scores_prefill(prefix_scores)
    content_delta = layout.pack_score_delta(content_scores)
    regular = np.roll(
        prefix_c * plan.prefix_source_regular, 2 * seq, axis=-1
    )
    donor = np.roll(
        prefix_c * plan.prefix_source_donor, 4 * seq + 4, axis=-1
    )
    y = np.exp(content_delta * plan.content_mask + regular + donor)
    y *= plan.active_mask

    expected_prefix = np.exp(prefix_scores.astype(np.float64))
    expected_content = np.exp(content_scores.astype(np.float64))
    expected_content *= np.tri(seq, dtype=bool)
    for _ in range(4):
        squared = y * y
        donor_squared = squared * plan.donor_mask
        row_sum_input = (
            squared * (1.0 - plan.donor_mask)
            + np.roll(donor_squared, -(2 * seq + 4), axis=-1)
        )
        row_sums = row_sum_input.reshape(2, seq, seq, 2).sum(axis=1)
        lambda_squared = np.broadcast_to(
            (1.0 / row_sums)[:, None, :, :], (2, seq, seq, 2)
        ).reshape(2, layout.slots)
        adjusted_lambda_squared = (
            lambda_squared * (1.0 - plan.donor_mask)
            + np.roll(lambda_squared, 2 * seq + 4, axis=-1)
            * plan.donor_mask
        )
        y = squared * adjusted_lambda_squared

        reference_prefix_squared = expected_prefix**2
        reference_content_squared = expected_content**2
        denominator = (
            reference_prefix_squared.sum(axis=-1)
            + reference_content_squared.sum(axis=-1)
        )
        expected_prefix = reference_prefix_squared / denominator[..., None]
        expected_content = reference_content_squared / denominator[..., None]

        prefix_c = (
            np.roll(y * plan.regular_mask, -2 * seq, axis=-1)
            + np.roll(y * plan.donor_mask, -(4 * seq + 4), axis=-1)
        )
        content_delta = y * plan.content_mask
        np.testing.assert_allclose(
            layout.unpack_prefix_scores_prefill(prefix_c, prefix_k=2),
            expected_prefix,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            layout.unpack_score_delta(content_delta),
            expected_content,
            atol=1e-7,
        )

    if seq == 128:
        assert set(fused_prefix_rotations(layout)) == {
            -516, -260, -256, 256, 260, 516
        }


def test_fused_delta_rejects_unsupported_geometry() -> None:
    layout = AttentionPairLayout(
        seq_len=8, head_dim=8, query_heads=4, key_value_heads=2, slots=128
    )
    with pytest.raises(ValueError, match="exactly two"):
        FusedPrefixDeltaPlan.build(layout, prefix_k=3)


@pytest.mark.gpu
@pytest.mark.parametrize("seq", [8, 128])
def test_fused_delta_fhe_matches_dual_alg2(seq: int, monkeypatch) -> None:
    from llama3fhe.approx.prefix import dual_softmax_alg2_numpy
    from llama3fhe.backend import create_sim_context, release_if_supported
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.operators.attention.decode.decode_softmax import DecodeAlg2Poly
    from llama3fhe.operators.attention.prefix.prefix_softmax import (
        prefix_softmax_rotations,
        softmax_alg2_prefix_partitions_fhe,
    )
    from llama3fhe.operators.nonlinear.bootstrap import (
        BootstrapConfig,
        BootstrapOperator,
    )

    layout = AttentionPairLayout(
        seq_len=seq,
        head_dim=seq,
        query_heads=4,
        key_value_heads=2,
        slots=8192 if seq == 8 else 32768,
    )
    rng = np.random.default_rng(2601006 + seq)
    if seq == 8:
        prefix_scores = rng.normal(scale=0.02, size=(4, seq, 2)).astype(np.float32)
        content_scores = rng.normal(scale=0.02, size=(4, seq, seq)).astype(np.float32)
        config = DecodeAlg2Poly.synthetic(k=1, degree=8)
    else:
        from llama3fhe.approx.softmax import (
            SoftmaxPolyConfig,
            load_softmax_poly_coeffs,
        )

        artifact = Path(__file__).resolve().parents[1] / (
            "assets/polynomials/softmax_poly_coeffs_prefix_k2.json"
        )
        config = SoftmaxPolyConfig.from_artifact(load_softmax_poly_coeffs(artifact))
        prefix_scores = np.empty((4, seq, 2), dtype=np.float32)
        prefix_scores[:, :, 0] = 2.2
        prefix_scores[:, :, 1] = 2.0
        content_scores = np.full((4, seq, seq), -2.0, dtype=np.float32)
        for query in range(seq):
            content_scores[:, query, query] = 2.2
    score_scale = None if seq == 8 else 10.0
    oracle_factor = 1.0 if seq == 8 else 10.0 * np.sqrt(float(seq))
    expected_p, expected_c, _ = dual_softmax_alg2_numpy(
        prefix_scores * oracle_factor,
        content_scores * oracle_factor,
        layout=layout,
        config=config,
        prefix_k=2,
    )
    cfg = Llama3CKKSConfig()
    cfg.simulator.logN = 14 if seq == 8 else 16
    cfg.simulator.maxLevelsRemaining = 32 if seq == 8 else 37
    cfg.simulator.device = "cuda"
    monkeypatch.setenv(
        "LLAMA_FHE_PREFIX_PIPELINE",
        os.environ.get("LLAMA_FHE_PREFIX_TEST_PIPELINE", "fused"),
    )
    rotations = set(prefix_softmax_rotations(layout, prefix_k=2))
    if seq == 128:
        rotations.update(BootstrapOperator.required_rotations(log_n=16))
    context = create_sim_context(
        cfg.simulator,
        input_dim=seq,
        block_out=seq,
        rotations=tuple(sorted(rotations)),
        include_default_rotations=True,
    )
    bootstrap = (
        None
        if seq == 8
        else BootstrapOperator.prepare(
            context,
            BootstrapConfig(
                log_slots=15,
                level_budget=(4, 4),
                output_levels=19,
                iterations=1,
                precision_bits=14,
            ),
        )
    )
    def encrypt_scores(rows):
        return tuple(
            context.encrypt(row, level=0, slots=layout.slots)
            for row in rows
        )

    p_ciphers = encrypt_scores(layout.pack_prefix_scores_prefill(prefix_scores))
    c_ciphers = encrypt_scores(layout.pack_score_delta(content_scores))
    out_p = out_c = None
    try:
        out_p, out_c = softmax_alg2_prefix_partitions_fhe(
            p_ciphers,
            c_ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            prefix_k=2,
            rounds=config.k_for(0),
            content_form="delta",
            score_scale=score_scale,
            bootstrap_operator=bootstrap,
            operator_config=AttentionOperatorConfig(
                reuse_softmax_round_square=True,
                fuse_softmax_slim_chebyshev_map=True,
            ),
        )
        packed_p = np.stack([np.asarray(context.decrypt(ct)).real for ct in out_p])
        packed_c = np.stack([np.asarray(context.decrypt(ct)).real for ct in out_c])
        packed_p = np.tile(packed_p[:, : layout.payload_slots],
                           (1, layout.payload_repetitions))
        packed_c = np.tile(packed_c[:, : layout.payload_slots],
                           (1, layout.payload_repetitions))
        actual_p = layout.unpack_prefix_scores_prefill(packed_p, prefix_k=2)
        actual_c = layout.unpack_score_delta(packed_c)
        np.testing.assert_allclose(actual_p, expected_p, atol=8e-2, rtol=8e-2)
        np.testing.assert_allclose(actual_c, expected_c, atol=8e-2, rtol=8e-2)
        np.testing.assert_allclose(actual_p[:, -2:], expected_p[:, -2:], atol=8e-2)
        limbs = min(int(ct.state.cur_limbs) for ct in out_p + out_c)
        print(f"fused_prefix_fhe seq={seq} output_min_limbs={limbs}")
        assert limbs >= (6 if seq == 128 else 2)
    finally:
        for cipher in p_ciphers + c_ciphers + tuple(out_p or ()) + tuple(out_c or ()):
            release_if_supported(cipher)
        if bootstrap is not None:
            bootstrap.release()


@pytest.mark.gpu
@pytest.mark.parametrize(
    (
        "layer_idx", "refresh_before_exp", "prefix_limbs", "content_limbs",
        "expected_auto_refresh", "pipeline",
    ),
    [
        (0, False, 9, 8, True, "fused"),
        (10, True, 9, 8, False, "fused"),
        (31, True, 9, 8, False, "fused"),
        (10, False, 21, 21, False, "fused"),
        (10, False, 20, 20, True, "fused"),
        (0, False, 9, 8, True, "dual"),
        (10, False, 24, 8, True, "dual"),
        (10, False, 8, 24, True, "dual"),
        (31, True, 28, 28, False, "dual"),
        (31, True, 28, 28, False, "fused"),
    ],
)
def test_prefix_softmax_entry_limbs(
    layer_idx: int,
    refresh_before_exp: bool,
    prefix_limbs: int,
    content_limbs: int,
    expected_auto_refresh: bool,
    pipeline: str,
    monkeypatch,
) -> None:
    from llama3fhe.approx.prefix import dual_softmax_alg2_numpy
    from llama3fhe.approx.softmax import SoftmaxPolyConfig, load_softmax_poly_coeffs
    from llama3fhe.backend import create_sim_context, release_if_supported
    from llama3fhe.config import Llama3CKKSConfig
    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.operators.attention.prefix.prefix_softmax import (
        prefix_softmax_rotations,
        softmax_alg2_prefix_partitions_fhe,
    )
    from llama3fhe.operators.nonlinear.bootstrap import (
        BootstrapConfig,
        BootstrapOperator,
    )

    layout = AttentionPairLayout(
        seq_len=128, head_dim=128, query_heads=4, key_value_heads=2, slots=32768
    )
    artifact = Path(__file__).resolve().parents[1] / (
        "assets/polynomials/softmax_poly_coeffs_prefix_k2.json"
    )
    config = SoftmaxPolyConfig.from_artifact(load_softmax_poly_coeffs(artifact))
    score_scale = max(10.0, config.U_hat_for(layer_idx) / 3.5)
    high = (config.U_hat_for(layer_idx) - 1.0) / score_scale
    next_high = (config.U_hat_for(layer_idx) - 3.0) / score_scale
    low = (config.U_hat_for(layer_idx) - 25.0) / score_scale
    prefix_scores = np.empty((4, 128, 2), dtype=np.float32)
    prefix_scores[:, :, 0] = high
    prefix_scores[:, :, 1] = next_high
    content_scores = np.full((4, 128, 128), low, dtype=np.float32)
    for query in range(128):
        content_scores[:, query, query] = high
    oracle_factor = score_scale * np.sqrt(128.0)
    expected_p, expected_c, _ = dual_softmax_alg2_numpy(
        prefix_scores * oracle_factor,
        content_scores * oracle_factor,
        layout=layout,
        config=config,
        layer_idx=layer_idx,
        prefix_k=2,
    )

    cfg = Llama3CKKSConfig()
    cfg.simulator.maxLevelsRemaining = 37
    monkeypatch.setenv("LLAMA_FHE_PREFIX_PIPELINE", pipeline)
    monkeypatch.setenv("LLAMA_FHE_PREFIX_ROUND_REFRESH", "native")
    if pipeline == "dual":
        monkeypatch.setenv("LLAMA_FHE_PREFIX_PACKING", "compact")
    rotations = set(prefix_softmax_rotations(layout, prefix_k=2))
    rotations.update(BootstrapOperator.required_rotations(log_n=16))
    context = create_sim_context(
        cfg.simulator,
        input_dim=128,
        block_out=128,
        rotations=tuple(sorted(rotations)),
        include_default_rotations=True,
    )
    bootstrap = BootstrapOperator.prepare(
        context,
        BootstrapConfig(
            log_slots=15,
            level_budget=(4, 4),
            output_levels=19,
            iterations=1,
            precision_bits=14,
        ),
    )
    p_ciphers = tuple(
        context.encrypt(
            row, level=context.context.L - prefix_limbs, slots=layout.slots
        )
        for row in layout.pack_prefix_scores_prefill(prefix_scores)
    )
    c_ciphers = tuple(
        context.encrypt(
            row, level=context.context.L - content_limbs, slots=layout.slots
        )
        for row in layout.pack_score_delta(content_scores)
    )
    out_p = out_c = None
    stages: dict[str, float] = {}
    calls_before = bootstrap.calls
    native_before = bootstrap.native_calls
    try:
        out_p, out_c = softmax_alg2_prefix_partitions_fhe(
            p_ciphers,
            c_ciphers,
            layout=layout,
            crypto_context=context,
            config=config,
            prefix_k=2,
            layer_idx=layer_idx,
            content_form="delta",
            score_scale=score_scale,
            bootstrap_operator=bootstrap,
            refresh_before_exp=refresh_before_exp,
            stage_seconds=stages,
            operator_config=AttentionOperatorConfig(
                refresh_softmax_exp_inputs=refresh_before_exp,
                pair_softmax_pre_exp_bootstrap=refresh_before_exp,
                reuse_softmax_round_square=True,
                fuse_softmax_slim_chebyshev_map=True,
                softmax_main_bootstrap_iterations=2 if layer_idx in (0, 31) else 1,
                softmax_lambda_bootstrap_iterations=2 if layer_idx in (0, 31) else 1,
            ),
        )
        packed_p = np.stack([np.asarray(context.decrypt(ct)).real for ct in out_p])
        packed_c = np.stack([np.asarray(context.decrypt(ct)).real for ct in out_c])
        actual_p = layout.unpack_prefix_scores_prefill(packed_p, prefix_k=2)
        actual_c = layout.unpack_score_delta(packed_c)
        prefix_error = float(np.max(np.abs(actual_p - expected_p)))
        content_error = float(np.max(np.abs(actual_c - expected_c)))
        limbs_p = min(int(ct.state.cur_limbs) for ct in out_p)
        limbs_c = min(int(ct.state.cur_limbs) for ct in out_c)
        print(
            f"prefix_entry pipeline={pipeline} layer={layer_idx} "
            f"prefix_error={prefix_error:.4g} "
            f"content_error={content_error:.4g} limbs_p={limbs_p} limbs_c={limbs_c} "
            f"pre_exp_bootstrap={'pre_exp_bootstrap' in stages}"
        )
        assert ("pre_exp_bootstrap" in stages) == bool(
            refresh_before_exp or expected_auto_refresh
        )
        if layer_idx == 31:
            assert max(prefix_error, content_error) < 1e-3
            assert bootstrap.native_calls - native_before == 2 * (
                bootstrap.calls - calls_before
            )
        np.testing.assert_allclose(actual_p, expected_p, atol=8e-2, rtol=8e-2)
        np.testing.assert_allclose(actual_c, expected_c, atol=8e-2, rtol=8e-2)
        assert limbs_p >= 6 and limbs_c >= 6
    finally:
        for cipher in p_ciphers + c_ciphers + tuple(out_p or ()) + tuple(out_c or ()):
            release_if_supported(cipher)
        bootstrap.release()
