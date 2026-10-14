from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import llama3fhe.operators.linear.prefill.mlp as mlp_module
from llama3fhe import Llama3CKKSConfig
from llama3fhe.approx.chebyshev import (
    chebyshev_ps_mul_depth,
    eval_chebyshev_coefficients,
)
from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.layouts.feature_major import (
    FeatureMajorPrefillLayout,
)
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.prefill.mlp import (
    FeatureMajorMLPLayout,
    mlp_complexity,
    mlp_fhe,
    mlp_packed_numpy,
    mlp_rotations,
)


def test_feature_major_mlp_layout_pages_roundtrip() -> None:
    canonical = FeatureMajorPrefillLayout(
        seq_len=8, hidden_dim=8, slots=32
    )
    layout = FeatureMajorMLPLayout(canonical, intermediate_width=12)
    values = np.arange(8 * 12, dtype=np.float64).reshape(8, 12)

    packed = layout.pack_intermediate(values)

    assert canonical.tokens_per_cipher == 4
    assert canonical.cipher_count == 2
    assert layout.feature_page_count == 2
    assert layout.active_width(0) == 8
    assert layout.active_width(1) == 4
    assert layout.paired_token_cipher_count == 1
    assert layout.intermediate_real_cipher_count == 4
    assert layout.intermediate_complex_cipher_count == 2
    assert packed.shape == (4, 32)
    np.testing.assert_array_equal(layout.unpack_intermediate(packed), values)


def test_feature_major_mlp_carrier_algebra_matches_direct_swiglu() -> None:
    rng = np.random.default_rng(26081431)
    canonical = FeatureMajorPrefillLayout(
        seq_len=8, hidden_dim=8, slots=32
    )
    hidden = rng.normal(scale=0.1, size=(8, 8))
    gate = rng.normal(scale=0.1, size=(12, 8))
    up = rng.normal(scale=0.1, size=(12, 8))
    down = rng.normal(scale=0.1, size=(8, 12))
    silu = lambda value: value / (1.0 + np.exp(-value))

    packed = mlp_packed_numpy(
        hidden,
        gate,
        up,
        down,
        layout=canonical,
        silu=silu,
    )
    actual = canonical.unpack(packed)
    gate_values = hidden @ gate.T
    up_values = hidden @ up.T
    expected = (silu(gate_values) * up_values) @ down.T

    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)


def test_feature_major_mlp_fused_input_scale_and_silu_map_are_equivalent() -> None:
    rng = np.random.default_rng(26081501)
    canonical = FeatureMajorPrefillLayout(
        seq_len=8, hidden_dim=8, slots=32
    )
    logical_hidden = rng.normal(scale=0.1, size=(8, 8))
    physical_scale = rng.uniform(0.75, 1.25, size=(8,))
    physical_hidden = logical_hidden / physical_scale[None, :]
    gate = rng.normal(scale=0.1, size=(12, 8))
    up = rng.normal(scale=0.1, size=(12, 8))
    down = rng.normal(scale=0.1, size=(8, 12))
    silu = lambda value: value / (1.0 + np.exp(-value))
    fit_interval = (-0.8, 1.2)

    baseline = mlp_packed_numpy(
        logical_hidden,
        gate,
        up,
        down,
        layout=canonical,
        silu=silu,
    )
    fused = mlp_packed_numpy(
        physical_hidden,
        gate,
        up,
        down,
        layout=canonical,
        silu=silu,
        input_column_scale=physical_scale,
        fuse_silu_domain_map=True,
        silu_fit_interval=fit_interval,
    )

    np.testing.assert_allclose(fused, baseline, atol=2e-8, rtol=2e-6)


def test_feature_major_mlp_production_static_cost() -> None:
    cost = mlp_complexity()

    assert cost.feature_pages == 4
    assert cost.canonical_real_ciphers == 16
    assert cost.paired_token_ciphers == 8
    assert cost.intermediate_complex_ciphers == 32
    assert cost.baby_steps == 64
    assert cost.plaintext_chunks == 8
    assert cost.reuse_baby_rotations_across_outputs is True

    assert cost.gate_up_encoded_weight_plaintexts == 16384
    assert cost.gate_up_pt_ct_multiplications == 262144
    assert cost.gate_up_baby_rotations == 1008
    assert cost.gate_up_giant_rotations == 3584
    assert cost.gate_up_chunk_alignment_rotations == 448
    assert cost.gate_up_input_conjugations == 8
    assert cost.gate_up_recovery_conjugations == 32

    assert cost.silu_polynomial_evaluations == 64
    assert cost.silu_polynomial_degree == 48
    assert cost.silu_split_conjugations == 32
    assert cost.up_split_conjugations == 32
    assert cost.gated_ct_ct_multiplications == 64
    assert cost.gated_relinearizations == 64

    assert cost.down_encoded_weight_plaintexts == 16384
    assert cost.down_pt_ct_multiplications == 131072
    assert cost.down_baby_rotations == 2016
    assert cost.down_giant_rotations == 1792
    assert cost.down_chunk_alignment_rotations == 224
    assert cost.down_accumulation_additions == 24
    assert cost.final_split_conjugations == 8

    assert cost.encoded_weight_plaintexts == 32768
    assert cost.pt_ct_multiplications == 393216
    assert cost.rotations_including_conjugations == 9184
    assert cost.multiplicative_depth == 10
    assert mlp_complexity(
        fuse_silu_domain_map=True
    ).multiplicative_depth == 9

    rotations = mlp_rotations()
    assert 8 in rotations
    assert 31 * 8 in rotations
    assert 32 * 8 in rotations
    # Normal hoisting needs only the global 512-row chunk alignments; the
    # per-giant extended-double-hoist keys are intentionally absent.
    assert 16 * 32 * 8 in rotations
    assert 112 * 32 * 8 in rotations
    assert 4 * 32768 - 1 in rotations


def test_mlp_phase_rotation_key_lifecycle_is_strictly_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "LLAMA_FHE_MLP_PHASE_ROTATION_KEY_LIFECYCLE"
    monkeypatch.delenv(name, raising=False)
    assert mlp_module._mlp_phase_rotation_key_lifecycle_enabled() is False

    monkeypatch.setenv(name, "0")
    assert mlp_module._mlp_phase_rotation_key_lifecycle_enabled() is False
    monkeypatch.setenv(name, "1")
    assert mlp_module._mlp_phase_rotation_key_lifecycle_enabled() is True
    monkeypatch.setenv(name, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        mlp_module._mlp_phase_rotation_key_lifecycle_enabled()


def test_mlp_down_rotation_preload_plan_is_exact_for_production() -> None:
    layout = FeatureMajorPrefillLayout()
    operator = LinearOperatorConfig(
        baby_steps=64,
        baby_anchor_step=64,
        max_plaintext_rows=512,
        hoist_strategy="normal",
    )

    plan = mlp_module._mlp_down_rotation_preload_plan(layout, operator)
    conjugation = 4 * int(layout.slots) - 1
    expected = tuple(
        rotation
        for rotation in mlp_rotations(layout, operator_config=operator)
        if int(rotation) != conjugation
    )

    assert plan.rotation_keys == expected
    assert len(plan.rotation_keys) == 71
    assert len(plan.forward_maps) == 63
    assert plan.forward_maps == tuple(range(8, 512, 8))
    assert plan.inverse_maps == (
        512,
        4096,
        8192,
        12288,
        16384,
        20480,
        24576,
        28672,
    )
    assert set(plan.rotation_keys) == (
        set(plan.forward_maps) | set(plan.inverse_maps)
    )


def test_mlp_phase_rotation_key_transitions_use_supported_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Native:
        N = 65536
        auto_load_keys_resolved = True

        def __init__(self) -> None:
            self._rotation_key_cuda_cache = {"old": object()}
            self._precompute_auto_cuda_cache = {"old": object()}
            self._inverse_precompute_auto_cuda_cache = {"old": object()}

        def clear_cuda_rotation_cache(self, **kwargs):
            events.append(("clear", kwargs))
            self._rotation_key_cuda_cache.clear()
            self._precompute_auto_cuda_cache.clear()
            self._inverse_precompute_auto_cuda_cache.clear()
            return {
                "_rotation_key_cuda_cache_entries": 1,
                "_rotation_key_cuda_cache_retained_entries": 0,
            }

        def get_rotation_key_for_limbs(self, rotation, cur_limbs):
            events.append(("key", rotation, cur_limbs))
            self._rotation_key_cuda_cache[(rotation, cur_limbs)] = object()
            return ((object(), object()), cur_limbs)

        def get_precompute_auto(self, rotation):
            events.append(("forward", rotation))
            self._precompute_auto_cuda_cache[rotation] = object()
            return object()

        def get_inverse_precompute_auto(self, rotation):
            events.append(("inverse", rotation))
            self._inverse_precompute_auto_cuda_cache[rotation] = object()
            return object()

    native = Native()
    context = SimpleNamespace(context=native, device="cuda:3")
    monkeypatch.setattr(
        mlp_module,
        "sync_device",
        lambda device: events.append(("sync", device)),
    )

    evict_s, evict_stats = mlp_module._evict_linear_keys_for_mlp_silu(
        context, conjugation_rotation=131071
    )
    plan = mlp_module._MLPDownRotationPreloadPlan(
        rotation_keys=(-1, 8),
        forward_maps=(-1,),
        inverse_maps=(8,),
    )
    preload_s, preload_stats = mlp_module._preload_mlp_down_rotation_keys(
        context, plan=plan, cur_limbs=9
    )

    assert evict_s >= 0.0
    assert preload_s >= 0.0
    assert events == [
        (
            "clear",
            {
                "keep_rotations": (131071,),
                "empty_allocator_cache": False,
            },
        ),
        ("key", 32767, 9),
        ("key", 8, 9),
        ("forward", 32767),
        ("inverse", 8),
        ("sync", "cuda:3"),
    ]
    assert evict_stats["conjugation_rotations_kept"] == 1.0
    assert evict_stats["rotation_key_cuda_cache_entries"] == 1.0
    assert evict_stats["rotation_key_cuda_cache_retained_entries"] == 0.0
    assert preload_stats["rotation_keys_requested"] == 2.0
    assert preload_stats["forward_maps_requested"] == 1.0
    assert preload_stats["inverse_maps_requested"] == 1.0
    assert preload_stats["rotation_key_cuda_cache_entries_before"] == 0.0
    assert preload_stats["rotation_key_cuda_cache_entries_after"] == 2.0

    stage: dict[str, float] = {}
    mlp_module._record_rotation_key_transition(
        stage, "gate_up_to_silu", evict_s, evict_stats
    )
    mlp_module._record_rotation_key_transition(
        stage, "silu_to_down", preload_s, preload_stats
    )
    assert "rotation_key_lifecycle_gate_up_to_silu_seconds" in stage
    assert stage[
        "rotation_key_lifecycle_silu_to_down_rotation_keys_requested"
    ] == 2.0


def test_mlp_phase_rotation_key_lifecycle_rejects_host_streaming() -> None:
    native = SimpleNamespace(auto_load_keys_resolved=False)
    context = SimpleNamespace(context=native)

    with pytest.raises(RuntimeError, match="requires eager cached keys"):
        mlp_module._require_phase_rotation_key_lifecycle_context(context)


def test_mlp_fine_profile_has_separate_silu_polynomial_boundary() -> None:
    labels: list[str] = []

    class FineAddVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_fine_add"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                labels.append(str(node.args[0].value))
            self.generic_visit(node)

    FineAddVisitor().visit(
        ast.parse(inspect.getsource(mlp_module.mlp_fhe))
    )

    assert labels == [
        "silu_gated.split_complex",
        "silu_gated.domain_map",
        "silu_gated.silu_polynomial",
        "silu_gated.gate_mul_up",
        "silu_gated.repack_complex",
    ]


def test_feature_major_mlp_rejects_layout_config_slot_mismatch() -> None:
    layout = FeatureMajorPrefillLayout(seq_len=8, hidden_dim=8, slots=32)
    config = Llama3CKKSConfig()
    config.simulator.logN = 14
    hidden = np.zeros((8, 8), dtype=np.float32)
    gate = np.zeros((12, 8), dtype=np.float32)
    up = np.zeros((12, 8), dtype=np.float32)
    down = np.zeros((8, 12), dtype=np.float32)

    with pytest.raises(ValueError, match="layout/config slot mismatch"):
        mlp_fhe(
            hidden,
            gate,
            up,
            down,
            silu_entry={
                "coefficients": [0.0, 1.0],
                "fit_interval": [-1.0, 1.0],
            },
            layout=layout,
            config=config,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("fuse_silu_domain_map", [False, True])
def test_feature_major_mlp_fhe_small_matches_polynomial_reference(
    fuse_silu_domain_map: bool,
) -> None:
    rng = np.random.default_rng(26081432)
    canonical = FeatureMajorPrefillLayout(
        seq_len=32, hidden_dim=512, slots=8192
    )
    hidden = rng.normal(scale=0.01, size=(32, 512)).astype(np.float32)
    gate = rng.normal(scale=0.01, size=(768, 512)).astype(np.float32)
    up = rng.normal(scale=0.01, size=(768, 512)).astype(np.float32)
    down = rng.normal(scale=0.01, size=(512, 768)).astype(np.float32)
    config = Llama3CKKSConfig()
    config.simulator.logN = 14
    config.simulator.maxLevelsRemaining = 16
    config.simulator.device = "cuda"
    config.qkv_linear.input_level = 0
    fit_interval = (-0.1, 0.1)
    silu_entry = {
        "coefficients": eval_chebyshev_coefficients(
            lambda value: value / (1.0 + np.exp(-value)),
            fit_interval[0],
            fit_interval[1],
            16,
        ),
        "fit_interval": fit_interval,
    }
    # One policy object, shared by the key planner and the kernel: a second
    # config here could silently plan a different rotation union.
    small_linear = LinearOperatorConfig(
        baby_steps=8,
        baby_anchor_step=8,
        max_plaintext_rows=64,
        hoist_strategy="ext_normal",
    )
    context = create_sim_context(
        config.simulator,
        input_dim=canonical.hidden_dim,
        block_out=canonical.tokens_per_cipher,
        rotations=mlp_rotations(
            canonical,
            operator_config=small_linear,
        ),
        include_default_rotations=False,
    )
    inputs = tuple(
        context.encrypt(row, level=0, slots=canonical.slots)
        for row in canonical.pack(hidden)
    )

    result = mlp_fhe(
        hidden,
        gate,
        up,
        down,
        silu_entry=silu_entry,
        layout=canonical,
        config=config,
        crypto_context=context,
        input_ciphers=inputs,
        verify=True,
        fuse_silu_domain_map=bool(fuse_silu_domain_map),
        operator_config=small_linear,
    )
    try:
        assert len(result.ciphers) == canonical.cipher_count
        assert result.output is not None
        assert result.expected is not None
        assert result.max_abs_diff is not None
        assert result.max_abs_diff < 2e-5
        assert result.complexity.feature_pages == 2
        assert result.complexity.reuse_baby_rotations_across_outputs is True
        assert result.complexity.gate_up_baby_rotations == 14
        assert result.complexity.gate_up_pt_ct_multiplications == 2048
        assert result.complexity.down_pt_ct_multiplications == 1024
        assert result.complexity.multiplicative_depth == (
            chebyshev_ps_mul_depth(silu_entry["coefficients"])
            + 4
            - int(fuse_silu_domain_map)
        )
    finally:
        result.release()
        for cipher in inputs:
            release_if_supported(cipher)
