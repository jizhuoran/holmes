from __future__ import annotations

from dataclasses import replace

import pytest

from llama3fhe.parallel.presets import (
    A100_HIGH,
    A800_4CARD,
    H100_EMU,
    HIGH_PRESETS,
    PRESETS,
    RTX4090_LOW,
    RTX6000PRO_HIGH,
    SHARED_CARD_MAX_PLAINTEXT_ROWS,
    BootstrapPreset,
    ParallelPreset,
    preset_by_name,
)


def test_l3_low_preset_matches_measured_configuration() -> None:
    assert RTX4090_LOW.world_size == 8
    assert RTX4090_LOW.ciphers_per_rank == 2
    assert RTX4090_LOW.cipher_count == 16
    assert RTX4090_LOW.dnum == 5
    assert RTX4090_LOW.depth == 37
    assert RTX4090_LOW.slots == 32768
    assert RTX4090_LOW.bootstrap.level_budget == (5, 5)
    assert RTX4090_LOW.bootstrap.cache_mode == "middle"
    assert RTX4090_LOW.key_prefetch_mode == "async" and RTX4090_LOW.key_evict_sync == "stream"
    assert A100_HIGH.key_prefetch_mode == "lazy" and A100_HIGH.key_evict_sync == "device"
    # 10 + 5 + 5 = 20 program levels -> 17 post-bootstrap levels at depth 37.
    assert RTX4090_LOW.bootstrap.program_depth == 20
    assert RTX4090_LOW.bootstrap_post_levels == 17


def test_high_presets_share_the_l3_program_and_the_same_code_path() -> None:
    assert set(HIGH_PRESETS) == {"a100-high", "rtx6000pro-high", "h100-emu", "a800-4card"}
    for name in HIGH_PRESETS:
        preset = preset_by_name(name)
        assert preset.bootstrap.level_budget == RTX4090_LOW.bootstrap.level_budget
        assert preset.bootstrap.program_depth == RTX4090_LOW.bootstrap.program_depth
        assert preset.bootstrap.cache_mode == "plain"
        assert preset.bootstrap_post_levels == 17
        if name == "a800-4card":
            # the only preset cut for four cards out of the box: two slots per rank
            assert preset.world_size == 4 and preset.ciphers_per_rank == 4
        else:
            assert preset.world_size == RTX4090_LOW.world_size
            assert preset.ciphers_per_rank == RTX4090_LOW.ciphers_per_rank
        assert preset.dnum == RTX4090_LOW.dnum and preset.depth == RTX4090_LOW.depth
        assert preset.key_policy == "resident"
        assert preset.limb_caps == RTX4090_LOW.limb_caps
        assert preset.mlp_feature_shard_k == 4
        assert preset.transform_hoist_strategy == "ext_normal"
        assert preset.attention_input_overlap is True
    assert RTX6000PRO_HIGH.linear_max_plaintext_rows == 2048
    assert A100_HIGH.linear_max_plaintext_rows == 512 and H100_EMU.linear_max_plaintext_rows == 512
    assert A100_HIGH.device_budget_gib == 78.0
    assert H100_EMU.device_budget_gib == 78.0
    # A800: the A100 profile at world_size 4; recutting back to 8 gives the A100 one-slot layout
    assert A800_4CARD.device_budget_gib == 78.0 and A800_4CARD.linear_max_plaintext_rows == 512
    assert A800_4CARD.mlp_feature_shard_k == 4 and A800_4CARD.rmsnorm_slim_period_rows == 16
    eight = A800_4CARD.with_world_size(8)
    assert eight.world_size == 8 and eight.ciphers_per_rank == 2 and eight.cipher_count == 16
    assert RTX6000PRO_HIGH.device_budget_gib == 92.0
    # The low-memory preset retains the baseline order and cache policy.
    assert RTX4090_LOW.attention_input_overlap is False and RTX4090_LOW.bootstrap.cache_mode == "middle"
    assert ParallelPreset(name="x").attention_input_overlap is False


def test_for_colocation_only_splits_the_recorded_budget() -> None:
    two_per_card = RTX6000PRO_HIGH.for_colocation(2)
    assert two_per_card.device_budget_gib == 46.0
    assert two_per_card.world_size == 8 and two_per_card.key_policy == "resident"
    assert two_per_card.bootstrap == RTX6000PRO_HIGH.bootstrap
    # two one-slot ranks on one card cannot both hold the 2048-row chunk (2 x 74.5 GiB)
    assert two_per_card.linear_max_plaintext_rows == SHARED_CARD_MAX_PLAINTEXT_ROWS == 512
    assert A100_HIGH.for_colocation(2).linear_max_plaintext_rows == 512
    assert RTX4090_LOW.for_colocation(2).linear_max_plaintext_rows == 256  # never raised
    assert RTX6000PRO_HIGH.for_colocation(1) is RTX6000PRO_HIGH
    with pytest.raises(ValueError):
        RTX6000PRO_HIGH.for_colocation(0)


def test_with_world_size_recuts_the_16_ciphertext_layout() -> None:
    four = RTX6000PRO_HIGH.with_world_size(4)
    assert (four.world_size, four.ciphers_per_rank, four.slots_per_rank, four.cipher_count) == (4, 4, 2, 16)
    # k = 4 divides 4: the MLP feature shard is kept
    assert four.mlp_feature_shard_k == 4
    # everything per-card is unchanged ...
    assert four.device_budget_gib == RTX6000PRO_HIGH.device_budget_gib
    assert four.key_policy == "resident" and four.bootstrap == RTX6000PRO_HIGH.bootstrap
    assert four.transform_hoist_strategy == "ext_normal" and four.attention_input_overlap is True
    assert four.name == RTX6000PRO_HIGH.name
    # Two slots per card use the shared-card chunk limit.
    assert four.linear_max_plaintext_rows == SHARED_CARD_MAX_PLAINTEXT_ROWS == 512
    assert A100_HIGH.with_world_size(4).linear_max_plaintext_rows == 512
    assert RTX4090_LOW.with_world_size(4).linear_max_plaintext_rows == 256  # never raised
    two = RTX6000PRO_HIGH.with_world_size(2)
    assert (two.world_size, two.ciphers_per_rank, two.mlp_feature_shard_k) == (2, 8, 2)  # k clamped 4 -> 2
    assert RTX4090_LOW.with_world_size(4).mlp_feature_shard_k == 2
    assert RTX6000PRO_HIGH.with_world_size(4, mlp_feature_shard_k=1).mlp_feature_shard_k == 1
    assert RTX6000PRO_HIGH.with_world_size(8) is RTX6000PRO_HIGH
    # the re-cut back to 8 keeps the 512 (the fallback is not undone); everything else round-trips
    assert four.with_world_size(8) == replace(RTX6000PRO_HIGH, linear_max_plaintext_rows=512)
    assert A100_HIGH.with_world_size(4).with_world_size(8) == A100_HIGH
    for bad in (0, 3, 5, 16, 32):
        with pytest.raises(ValueError):
            RTX6000PRO_HIGH.with_world_size(bad)
    with pytest.raises(ValueError):
        RTX6000PRO_HIGH.with_world_size(4, mlp_feature_shard_k=8)  # does not divide 4


def test_preset_lookup_and_validation() -> None:
    assert preset_by_name("rtx4090-low") is RTX4090_LOW
    assert set(PRESETS) == {"rtx4090-low", "a100-high", "rtx6000pro-high", "h100-emu", "a800-4card"}
    assert preset_by_name("a100-high") is A100_HIGH
    assert preset_by_name("rtx6000pro-high") is RTX6000PRO_HIGH
    with pytest.raises(KeyError):
        preset_by_name("h100")
    with pytest.raises(ValueError):
        BootstrapPreset(level_budget=(5,))
    with pytest.raises(ValueError):
        BootstrapPreset(cache_mode="raw")
    with pytest.raises(ValueError):
        BootstrapPreset(level_budget=(20, 20)).post_levels(37)
    with pytest.raises(ValueError):
        ParallelPreset(name="odd", ciphers_per_rank=3)


def test_key_policy_fields_validate_and_normalise() -> None:
    bare = ParallelPreset(name="x")
    assert bare.key_policy == "lease"
    assert bare.key_resident_extra == ()
    assert bare.limb_caps == {}
    capped = ParallelPreset(name="x", key_limb_caps={"linear": 18, "attn_input": 6})
    assert capped.key_limb_caps == (("linear", 18), ("attn_input", 6))
    assert capped.limb_caps == {"linear": 18, "attn_input": 6}
    with pytest.raises(ValueError):
        ParallelPreset(name="x", key_policy="sometimes")


def test_rtx4090_low_keeps_linear_resident_with_measured_caps() -> None:
    from llama3fhe.parallel.presets import MEASURED_LIMB_CAPS

    assert RTX4090_LOW.key_policy == "lease"
    assert RTX4090_LOW.key_resident_extra == ("linear",)
    assert RTX4090_LOW.limb_caps == dict(MEASURED_LIMB_CAPS)
    assert RTX4090_LOW.limb_caps["linear"] <= 16  # beta stays 2 (alpha 8)
    assert A100_HIGH.key_policy == "resident"
    assert A100_HIGH.key_resident_extra == ()
    assert A100_HIGH.limb_caps == RTX4090_LOW.limb_caps



def test_transform_hoist_strategy_is_a_preset_field_exported_to_workers(monkeypatch) -> None:
    from dataclasses import replace

    from llama3fhe.operators.transform import transform_hoist_strategy

    assert RTX4090_LOW.transform_hoist_strategy == "normal"
    assert A100_HIGH.transform_hoist_strategy == "ext_normal"
    assert ParallelPreset(name="x").transform_hoist_strategy == "ext_normal"
    with pytest.raises(ValueError):
        replace(RTX4090_LOW, transform_hoist_strategy="fast")

    monkeypatch.delenv("LLAMA_FHE_TRANSFORM_HOIST", raising=False)
    assert transform_hoist_strategy() == "ext_normal"
    monkeypatch.setenv("LLAMA_FHE_TRANSFORM_HOIST", "normal")
    assert transform_hoist_strategy() == "normal"
    monkeypatch.setenv("LLAMA_FHE_TRANSFORM_HOIST", "bogus")
    with pytest.raises(ValueError):
        transform_hoist_strategy()


def test_transport_is_a_preset_field_with_the_pinned_arena_on_the_4090() -> None:
    from dataclasses import replace

    from llama3fhe.parallel.prefill.transport import TRANSPORT_BACKENDS

    assert RTX4090_LOW.transport == "pinned"
    assert A100_HIGH.transport == "host" and ParallelPreset(name="x").transport == "host"
    assert all(preset.transport in TRANSPORT_BACKENDS for preset in PRESETS.values())
    with pytest.raises(ValueError):
        replace(RTX4090_LOW, transport="nccl")


def test_pairfm_refresh_passes_are_a_preset_field_one_on_the_high_tier() -> None:
    from dataclasses import replace

    from llama3fhe.parallel.presets import RTX6000PRO_HIGH

    assert RTX6000PRO_HIGH.pairfm_refresh_bootstrap_iterations == 1
    assert A100_HIGH.pairfm_refresh_bootstrap_iterations == 1
    # the 4090 keeps the inherited two passes until validated there
    assert RTX4090_LOW.pairfm_refresh_bootstrap_iterations == 2
    assert ParallelPreset(name="x").pairfm_refresh_bootstrap_iterations == 2
    # survives the 4-card / colocation re-cuts
    assert RTX6000PRO_HIGH.with_world_size(4).pairfm_refresh_bootstrap_iterations == 1
    assert RTX6000PRO_HIGH.for_colocation(2).pairfm_refresh_bootstrap_iterations == 1
    with pytest.raises(ValueError):
        replace(RTX4090_LOW, pairfm_refresh_bootstrap_iterations=3)


def test_numeric_knobs_are_preset_fields_validated_on_the_high_tier() -> None:
    from dataclasses import replace

    from llama3fhe.parallel.presets import POLYNOMIALS_DIR, polynomial_table_path

    for preset in (RTX6000PRO_HIGH, A100_HIGH, H100_EMU):
        assert preset.rmsnorm_exit_mode == "slim"
        assert preset.rmsnorm_slim_exit_layers == "3-31"
        assert preset.rmsnorm_coeffs == "rmsnorm_poly_coeffs_reltol5e-3_noinc.json"
        assert preset.softmax_coeffs == "softmax_poly_coeffs_joint_tol_v1.json"
        assert preset.softmax_coeffs_parts == "exp"
        # bare names resolve under assets/polynomials and the files ship with the repo
        assert polynomial_table_path(preset.rmsnorm_coeffs) == POLYNOMIALS_DIR / preset.rmsnorm_coeffs
        assert polynomial_table_path(preset.softmax_coeffs).is_file()
    assert RTX4090_LOW.rmsnorm_exit_mode == "pair"
    assert RTX4090_LOW.rmsnorm_coeffs == "rmsnorm_poly_coeffs_reltol5e-3_noinc.json"
    assert RTX4090_LOW.softmax_coeffs == "softmax_poly_coeffs_joint_tol_v1.json"
    assert RTX4090_LOW.softmax_coeffs_parts == "exp"
    assert ParallelPreset(name="x").rmsnorm_coeffs is None
    assert ParallelPreset(name="x").softmax_coeffs_parts == "exp,j1,j2"
    assert ParallelPreset(name="x").rmsnorm_exit_mode == "pair"
    # survive the 4-card / colocation re-cuts
    four = RTX6000PRO_HIGH.with_world_size(4)
    assert (four.rmsnorm_exit_mode, four.rmsnorm_slim_exit_layers, four.softmax_coeffs_parts) == ("slim", "3-31", "exp")
    assert RTX6000PRO_HIGH.for_colocation(2).rmsnorm_coeffs == RTX6000PRO_HIGH.rmsnorm_coeffs
    # parts are normalised to canonical order; junk is rejected
    assert ParallelPreset(name="x", softmax_coeffs_parts="j2, exp").softmax_coeffs_parts == "exp,j2"
    with pytest.raises(ValueError):
        replace(RTX6000PRO_HIGH, rmsnorm_exit_mode="sparse")
    with pytest.raises(ValueError):
        replace(RTX6000PRO_HIGH, softmax_coeffs_parts="exp,invsqrt")
    with pytest.raises(ValueError):
        replace(RTX6000PRO_HIGH, softmax_coeffs_parts="")
    with pytest.raises(ValueError):
        replace(RTX6000PRO_HIGH, rmsnorm_slim_exit_layers=" ")
    with pytest.raises(ValueError):
        replace(RTX6000PRO_HIGH, rmsnorm_coeffs="")
    with pytest.raises(FileNotFoundError):
        polynomial_table_path("no_such_table.json")


def test_benches_take_the_preset_numerics_and_let_the_environment_override(monkeypatch) -> None:
    from bench.t2_layer import (
        rmsnorm_coeffs_table,
        rmsnorm_exit_description,
        rmsnorm_exit_kwargs,
        softmax_coeffs_table,
        softmax_table_degrees,
    )
    from llama3fhe.application.args import DEFAULT_RMS, DEFAULT_SOFTMAX
    from llama3fhe.approx.rmsnorm import get_layer_coeffs

    for name in ("T2_RMS_EXIT", "T2_RMS_SLIM_LAYERS", "T2_RMS_COEFFS", "T2_RMS_COEFFS_LAYERS",
                 "T2_SOFTMAX_COEFFS", "T2_SOFTMAX_PARTS", "T2_SOFTMAX_LAYERS"):
        monkeypatch.delenv(name, raising=False)

    # no preset: production numerics, exactly what the benches did before
    bare = ParallelPreset(name="x")
    assert rmsnorm_exit_kwargs(10, None) == {"exit_refresh": True}
    assert rmsnorm_exit_kwargs(10, None, bare) == {"exit_refresh": True}
    assert rmsnorm_exit_description(bare)["mode"] == "pair"
    _, info = rmsnorm_coeffs_table(DEFAULT_RMS, bare)
    assert info == {"source": str(DEFAULT_RMS), "patched_layers": []}
    _, info = softmax_coeffs_table(DEFAULT_SOFTMAX, bare)
    assert info == {"source": str(DEFAULT_SOFTMAX), "patched_layers": []}
    # the 4090 keeps the pair exit but takes the two tables (dq4)
    assert rmsnorm_exit_kwargs(10, None, RTX4090_LOW) == {"exit_refresh": True}
    _, info = rmsnorm_coeffs_table(DEFAULT_RMS, RTX4090_LOW)
    assert info["patched_layers"] == "all"

    high = RTX6000PRO_HIGH.with_world_size(4)
    assert rmsnorm_exit_kwargs(2, None, high) == {"exit_refresh": True, "slim_period_rows": 16}
    assert rmsnorm_exit_kwargs(3, None, high) == {"exit_refresh": False, "slim_refresh_min_limbs": 16,
                                                  "slim_period_rows": 16}
    assert rmsnorm_exit_kwargs(31, None, high)["exit_refresh"] is False
    desc = rmsnorm_exit_description(high)
    assert desc["mode"] == "slim" and desc["slim_layers"] == list(range(3, 32))
    rms_high, info = rmsnorm_coeffs_table(DEFAULT_RMS, high)
    assert info["patched_layers"] == "all" and info["source"].endswith("rmsnorm_poly_coeffs_reltol5e-3_noinc.json")
    rms_prod, _ = rmsnorm_coeffs_table(DEFAULT_RMS, None)
    # the refit never raises a degree, and lowers at least one
    prod_deg = [int(get_layer_coeffs(rms_prod, b, i)["degree"]) for b in ("attn", "ffn") for i in range(32)]
    high_deg = [int(get_layer_coeffs(rms_high, b, i)["degree"]) for b in ("attn", "ffn") for i in range(32)]
    assert all(h <= p for h, p in zip(high_deg, prod_deg)) and sum(high_deg) < sum(prod_deg)
    sm_high, info = softmax_coeffs_table(DEFAULT_SOFTMAX, high)
    assert info["parts"] == ["exp"] and info["patched_layers"] == "all"
    assert info["alt"].endswith("softmax_poly_coeffs_joint_tol_v1.json")
    sm_prod, _ = softmax_coeffs_table(DEFAULT_SOFTMAX, None)
    deg_high, deg_prod = softmax_table_degrees(sm_high), softmax_table_degrees(sm_prod)
    assert all(deg_high[k]["exp"] <= deg_prod[k]["exp"] for k in deg_prod)
    assert any(deg_high[k]["exp"] < deg_prod[k]["exp"] for k in deg_prod)
    # j1 / j2 / k untouched when only exp is taken
    assert all((deg_high[k]["j1"], deg_high[k]["j2"], deg_high[k]["k"]) ==
               (deg_prod[k]["j1"], deg_prod[k]["j2"], deg_prod[k]["k"]) for k in deg_prod)

    # the environment still wins for A/B, and "default" forces the production table
    monkeypatch.setenv("T2_RMS_EXIT", "pair")
    assert rmsnorm_exit_kwargs(10, None, high) == {"exit_refresh": True, "slim_period_rows": 16}
    monkeypatch.setenv("T2_RMS_EXIT", "slim")
    monkeypatch.setenv("T2_RMS_SLIM_LAYERS", "10")
    assert rmsnorm_exit_kwargs(10, None, high)["exit_refresh"] is False
    assert rmsnorm_exit_kwargs(11, None, high)["exit_refresh"] is True
    monkeypatch.setenv("T2_RMS_COEFFS", "default")
    _, info = rmsnorm_coeffs_table(DEFAULT_RMS, high)
    assert info["patched_layers"] == []
    monkeypatch.setenv("T2_SOFTMAX_COEFFS", "none")
    _, info = softmax_coeffs_table(DEFAULT_SOFTMAX, high)
    assert info["patched_layers"] == []
    monkeypatch.setenv("T2_SOFTMAX_COEFFS", "softmax_poly_coeffs_joint_tol_v1.json")
    monkeypatch.setenv("T2_SOFTMAX_PARTS", "exp,j1,j2")
    _, info = softmax_coeffs_table(DEFAULT_SOFTMAX, high)
    assert info["parts"] == ["exp", "j1", "j2"]
    monkeypatch.setenv("T2_RMS_EXIT", "sparse")
    with pytest.raises(ValueError):
        rmsnorm_exit_kwargs(10, None, high)


def test_sparse_bootstrap_log_slots_combines_slim_row_and_preset_extras() -> None:
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.parallel.worker import slim_sparse_log_slots, sparse_bootstrap_log_slots

    preset = replace(preset_by_name("rtx6000pro-high"), rmsnorm_slim_period_rows=None)
    prefill = FeatureMajorPrefillLayout(seq_len=128, hidden_dim=4096, slots=32768)
    decode = FeatureMajorPrefillLayout(seq_len=1, hidden_dim=4096, slots=32768)
    assert slim_sparse_log_slots(preset, prefill) is None
    assert sparse_bootstrap_log_slots(preset, prefill) is None
    assert slim_sparse_log_slots(preset_by_name("rtx6000pro-high"), prefill) == 7

    slim16 = replace(preset, rmsnorm_slim_period_rows=16)
    assert slim_sparse_log_slots(slim16, prefill) == 7  # 16 rows * 8 tokens/cipher = 128 slots
    assert slim_sparse_log_slots(slim16, decode) == 7
    assert sparse_bootstrap_log_slots(slim16, prefill) == (7,)
    slim1 = replace(preset, rmsnorm_slim_period_rows=1)
    assert sparse_bootstrap_log_slots(slim1, decode) == (3,)  # one DecodeFM cipher: period 8 slots

    softmax = replace(slim1, bootstrap_extra_sparse_log_slots=(12,))
    assert sparse_bootstrap_log_slots(softmax, decode) == (3, 12)
    # duplicates / out-of-range extras are dropped
    assert sparse_bootstrap_log_slots(replace(slim16, bootstrap_extra_sparse_log_slots=(7, 15, 0)), prefill) == (7,)
    assert sparse_bootstrap_log_slots(replace(preset, bootstrap_extra_sparse_log_slots=(12,)), decode) == (12,)
