from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bench import prefill_decode as experiment
from llama3fhe.layouts import PairFold128Layout
from llama3fhe.parallel.prefill.keys import KeyPlan, KeyTier


class _FakeCipher:
    def __init__(self, values: np.ndarray, *, limbs: int) -> None:
        self.values = np.asarray(values)
        self.state = SimpleNamespace(cur_limbs=int(limbs))
        self.release_calls = 0
        self.clones: list[_FakeCipher] = []

    def deep_copy(self) -> _FakeCipher:
        clone = _FakeCipher(self.values.copy(), limbs=self.state.cur_limbs)
        self.clones.append(clone)
        return clone

    def release(self) -> None:
        if self.release_calls:
            raise AssertionError("cipher released twice")
        self.release_calls += 1


class _FakeContext:
    def __init__(self, *, levels: int = 37, fail_at: int | None = None) -> None:
        self.context = SimpleNamespace(L=int(levels))
        self.fail_at = fail_at
        self.encrypt_calls: list[tuple[int, int]] = []
        self.ciphers: list[_FakeCipher] = []

    def encrypt(
        self,
        values: np.ndarray,
        *,
        level: int,
        slots: int,
    ) -> _FakeCipher:
        call_index = len(self.encrypt_calls)
        self.encrypt_calls.append((int(level), int(slots)))
        if call_index == self.fail_at:
            raise RuntimeError(f"injected encryption failure at {call_index}")
        cipher = _FakeCipher(
            values,
            limbs=int(self.context.L) - int(level),
        )
        self.ciphers.append(cipher)
        return cipher


def _logical_kv(
    *,
    layers: int = 2,
    tokens: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (int(layers), int(tokens), 8, 128)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return values, -values


def test_decode_packing_cli_defaults_to_pairfold() -> None:
    parser = experiment._build_experiment_parser()

    default, passthrough = parser.parse_known_args([])
    pairfold, pairfold_passthrough = parser.parse_known_args(
        ["--decode-packing", "pairfold128", "--warmup", "2"]
    )

    assert default.decode_packing == "pairfold128"
    assert default.no_prefill_kv_cache is False
    assert default.match_prefill_decode_key_plan is False
    assert default.fixed_decode_benchmark is False
    assert default.benchmark_warmups == 2
    assert default.benchmark_repetitions == 5
    assert default.benchmark_order_label is None
    assert passthrough == []
    assert pairfold.decode_packing == "pairfold128"
    assert pairfold_passthrough == ["--warmup", "2"]
    with pytest.raises(SystemExit):
        parser.parse_known_args(["--decode-packing", "pairfold"])


def test_prefill_only_baseline_mode_accepts_one_page_without_decode() -> None:
    extra, passthrough = experiment._build_experiment_parser().parse_known_args(
        ["--no-prefill-kv-cache", "--num-tokens", "1"]
    )

    assert passthrough == []
    experiment._validate_no_prefill_kv_cache_mode(
        extra, continuation_count=0, prefill_page_count=1
    )


def test_matched_no_kv_baseline_uses_candidate_rotation_modes() -> None:
    baseline, _ = experiment._build_experiment_parser().parse_known_args(
        [
            "--no-prefill-kv-cache",
            "--match-prefill-decode-key-plan",
            "--num-tokens",
            "1",
        ]
    )
    experiment._validate_no_prefill_kv_cache_mode(baseline)
    baseline_modes = experiment._rotation_planning_modes(
        emit_prefill_kv_cache=False,
        decode_steps_requested=False,
        match_key_plan=baseline.match_prefill_decode_key_plan,
    )
    candidate_modes = experiment._rotation_planning_modes(
        emit_prefill_kv_cache=True,
        decode_steps_requested=True,
        match_key_plan=False,
    )
    assert baseline_modes == candidate_modes == (True, True)

    planned_prefill_emit: list[bool] = []
    def required_prefill_rotations(_schedule, *, emit_kv_cache, **_kwargs):
        planned_prefill_emit.append(bool(emit_kv_cache))
        return (1, 2) if emit_kv_cache else (1,)

    prefill_model = SimpleNamespace(
        required_rotations=required_prefill_rotations
    )
    decode_model = SimpleNamespace(
        required_decode_rotations=lambda _schedule, **_kwargs: (3,)
    )
    def plan(modes: tuple[bool, bool]):
        return experiment._planned_rotation_keys(
            prefill_model=prefill_model,
            decode_model=decode_model,
            schedule=object(),
            plan_prefill_kv_cache=modes[0],
            plan_decode_keys=modes[1],
            prefix_k=0,
            decode_packing="pairfold128",
        )

    assert plan(baseline_modes) == plan(candidate_modes) == ((1, 2, 3), (3,))
    assert plan((False, False)) == ((1,), ())
    assert planned_prefill_emit == [True, True, False]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--match-prefill-decode-key-plan"],
        ["--no-prefill-kv-cache", "--match-prefill-decode-key-plan", "--decode-packing", "legacy_c"],
    ],
)
def test_matched_key_plan_rejects_incompatible_modes(arguments: list[str]) -> None:
    extra, _ = experiment._build_experiment_parser().parse_known_args(arguments)
    with pytest.raises(ValueError, match="--match-prefill-decode-key-plan requires"):
        experiment._validate_no_prefill_kv_cache_mode(extra)


def test_rotation_key_plan_summary_detects_index_and_cap_changes() -> None:
    options = {
        "ring_dim": 64,
        "full_limbs": 38,
        "rotation_key_limb_cap": None,
        "rotation_key_limb_limits": {3: 12, 31: 20},
    }
    planned = experiment._rotation_key_plan_summary((3, -1, 3), **options)
    assert planned["rotation_count"] == 2
    assert planned["capped_rotation_count"] == 2
    assert planned == experiment._rotation_key_plan_summary((-1, 3), **options)
    assert planned["rotations_sha256"] != experiment._rotation_key_plan_summary(
        (3, 4), **options
    )["rotations_sha256"]
    changed_caps = {**options, "rotation_key_limb_limits": {3: 13, 31: 20}}
    assert planned["effective_limb_caps_sha256"] != experiment._rotation_key_plan_summary(
        (3, -1), **changed_caps
    )["effective_limb_caps_sha256"]
    global_cap = {**options, "rotation_key_limb_cap": 8}
    assert experiment._rotation_key_plan_summary(
        (3, -1), **global_cap
    )["capped_rotation_count"] == 2
    assert experiment._rotation_key_plan_summary(
        (3, -1), **global_cap
    )["effective_limb_caps_sha256"] == experiment._rotation_key_plan_summary(
        (3, -1), **{**options, "rotation_key_limb_limits": {3: 8, 31: 8}}
    )["effective_limb_caps_sha256"]


def test_joint_rotation_key_limits_cover_both_phases_and_full_keys() -> None:
    prefill = KeyPlan(
        (
            KeyTier("bootstrap", 0, (27,), None),
            KeyTier("linear", 1, (24, 25, 26), 14),
        )
    )
    decode = KeyPlan(
        (
            KeyTier("rmsnorm", 0, (29,), None),
            KeyTier("linear", 1, (24,), 16),
            KeyTier("attn_core", 2, (25, 28, 29), 13),
        )
    )

    limits = experiment._joint_rotation_key_limb_limits(
        (prefill, decode),
        rotations=(24, 25, 26, 27, 28, 29, 30),
        ring_dim=1 << 16,
        uncapped_rotations=(26,),
    )

    assert limits == {24: 16, 25: 14, 28: 13}
    assert experiment._joint_rotation_key_limb_limits(
        (prefill,), rotations=(24, 25), ring_dim=1 << 16
    ) == {24: 14, 25: 14}


def test_tiered_decode_requires_steady_first_layer() -> None:
    extra, passthrough = experiment._build_experiment_parser().parse_known_args(
        ["--tiered-rotation-key-limits", "--steady-state-layer", "--num-tokens", "2"]
    )
    assert extra.tiered_rotation_key_limits is True
    assert passthrough == ["--steady-state-layer"]
    experiment._validate_tiered_decode_mode(
        tiered=True, decode_requested=True, steady_state_layer=True
    )
    experiment._validate_tiered_decode_mode(
        tiered=True, decode_requested=False, steady_state_layer=False
    )
    with pytest.raises(ValueError, match="requires --steady-state-layer"):
        experiment._validate_tiered_decode_mode(
            tiered=True, decode_requested=True, steady_state_layer=False
        )


@pytest.mark.parametrize(
    ("arguments", "continuation_count", "prefill_page_count"),
    [
        (["--num-tokens", "2"], 0, 1),
        (["--skip-prefill"], 0, 1),
        (["--oracle-only"], 0, 1),
        (["--fixed-decode-benchmark"], 0, 1),
        ([], 1, 1),
        ([], 0, 2),
    ],
)
def test_prefill_only_baseline_rejects_cache_consumers(
    arguments: list[str], continuation_count: int, prefill_page_count: int
) -> None:
    extra, _ = experiment._build_experiment_parser().parse_known_args(
        ["--no-prefill-kv-cache", "--num-tokens", "1", *arguments]
    )

    with pytest.raises(ValueError, match="--no-prefill-kv-cache requires"):
        experiment._validate_no_prefill_kv_cache_mode(
            extra,
            continuation_count=continuation_count,
            prefill_page_count=prefill_page_count,
        )


def test_variable_prompt_tokenization_keeps_all_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(
        encode=lambda *_args, **_kwargs: list(range(130))
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda _path: tokenizer
            )
        ),
    )
    args, passthrough = experiment._build_experiment_parser().parse_known_args(
        ["--prompt", "long prompt"]
    )
    ids, loaded_tokenizer = experiment._load_experiment_input_ids(args)
    assert passthrough == []
    assert loaded_tokenizer is tokenizer
    np.testing.assert_array_equal(ids, np.arange(130))


def test_fixed_decode_benchmark_cli_is_explicit_and_configurable() -> None:
    parser = experiment._build_experiment_parser()

    parsed, passthrough = parser.parse_known_args(
        [
            "--fixed-decode-benchmark",
            "--benchmark-warmups",
            "3",
            "--benchmark-repetitions",
            "7",
            "--benchmark-order-label",
            "B2",
        ]
    )

    assert parsed.fixed_decode_benchmark is True
    assert parsed.benchmark_warmups == 3
    assert parsed.benchmark_repetitions == 7
    assert parsed.benchmark_order_label == "B2"
    assert passthrough == []


def test_skip_prefill_encrypts_logical_kv_directly_as_pairfold() -> None:
    layout = PairFold128Layout()
    keys, values = _logical_kv()
    context = _FakeContext()

    caches = experiment._encrypt_poly_pairfold_caches(
        keys,
        values,
        context=context,
        pairfold=layout,
        kv_limbs=12,
    )

    assert len(caches) == 2
    assert context.encrypt_calls == [(25, layout.slots)] * 8
    assert [cache.content_tokens for cache in caches] == [3, 3]
    assert [len(cache.pages) for cache in caches] == [1, 1]
    assert all(len(cache.pages[0].handles) == 4 for cache in caches)
    np.testing.assert_array_equal(
        np.stack([cipher.values for cipher in caches[0].pages[0].key_groups]),
        layout.pack_key(keys[0]),
    )
    np.testing.assert_array_equal(
        np.stack([cipher.values for cipher in caches[0].pages[0].value_groups]),
        layout.pack_value(values[0]),
    )

    inventory = experiment._cache_inventory(
        "pairfold128",
        legacy_caches=[],
        pairfold_caches=caches,
        pair_count=4,
    )
    assert inventory["pages"] == 2
    assert inventory["ciphertexts"] == 8
    assert inventory["layers"][0]["content_tokens"] == 3

    handles = tuple(context.ciphers)
    experiment._release_pairfold_caches([caches[0], None, caches[1]])
    assert all(cipher.release_calls == 1 for cipher in handles)


def test_pairfold_cache_limb_validation_checks_key_and_value_separately() -> None:
    layout = PairFold128Layout()
    keys, values = _logical_kv(layers=1)
    context = _FakeContext()
    caches = experiment._encrypt_poly_pairfold_caches(
        keys,
        values,
        context=context,
        pairfold=layout,
        kv_limbs=12,
    )
    try:
        page = caches[0].pages[0]
        for cipher in page.value_groups:
            cipher.state.cur_limbs = 8
        assert experiment._assert_pairfold_cache_limbs(
            caches[0], layer_index=0, key_limbs=12, value_limbs=8
        ) == ({12}, {8})
        with pytest.raises(RuntimeError, match="K=\\[12\\] V=\\[8\\]"):
            experiment._assert_pairfold_cache_limbs(
                caches[0], layer_index=0, key_limbs=12, value_limbs=12
            )
    finally:
        experiment._release_pairfold_caches(caches)


def test_pairfold_plaintext_cache_failure_releases_every_created_cipher() -> None:
    layout = PairFold128Layout()
    keys, values = _logical_kv()
    context = _FakeContext(fail_at=6)

    with pytest.raises(RuntimeError, match="encryption failure at 6"):
        experiment._encrypt_poly_pairfold_caches(
            keys,
            values,
            context=context,
            pairfold=layout,
            kv_limbs=12,
        )

    assert len(context.ciphers) == 6
    assert all(cipher.release_calls == 1 for cipher in context.ciphers)


def test_cache_inventory_and_stage_totals_remain_packing_specific() -> None:
    legacy = [
        ((object(),) * 8, (object(),) * 8),
        ((object(),) * 4, (object(),) * 4),
    ]
    inventory = experiment._cache_inventory(
        "legacy_c",
        legacy_caches=legacy,
        pairfold_caches=[],
        pair_count=4,
    )

    assert inventory["packing"] == "legacy_c"
    assert inventory["pages"] == 3
    assert inventory["ciphertexts"] == 24
    stage_totals = experiment._stage_totals(
        [
            {"stage_seconds": {"qkv": 0.5, "route": 0.2}},
            {"stage_seconds": {"qkv": 0.4, "route.detail": 0.1}},
        ]
    )
    assert stage_totals == {"qkv": 0.9, "route": 0.2, "route.detail": 0.1}
    json.dumps({"cache_samples": [inventory], "stage_totals": stage_totals})

    with pytest.raises(ValueError, match="received PairFold"):
        experiment._cache_inventory(
            "legacy_c",
            legacy_caches=legacy,
            pairfold_caches=[object()],
            pair_count=4,
        )


def test_fixed_decode_trial_summary_reports_median_p90_and_stages() -> None:
    trials = [
        {
            "wall_seconds": float(value),
            "model_wall_seconds": float(value) - 0.1,
            "stage_seconds": {"qkv": float(value) / 10.0},
            "cuda_allocator_peak_bytes": 1000 + value,
            "cuda_allocator_incremental_peak_bytes": 100 + value,
        }
        for value in range(1, 6)
    ]

    summary = experiment._summarize_fixed_decode_trials(trials)

    assert summary["wall_seconds"] == {
        "count": 5,
        "min": 1.0,
        "median": 3.0,
        "p90": pytest.approx(4.6),
        "max": 5.0,
    }
    assert summary["stage_seconds"]["qkv"]["median"] == pytest.approx(0.3)
    assert summary["cuda_allocator_peak_bytes"]["max"] == 1005.0
    json.dumps(summary)


def test_fixed_decode_benchmark_reuses_layers_and_cleans_each_trial(
    monkeypatch,
) -> None:
    layers = (object(), object())
    factory_calls = 0
    stream_close_calls = 0
    decode_first_layer_refreshes: list[bool] = []
    decoded_layer_ids: list[tuple[int, ...]] = []
    results = []
    cleanup_snapshots: list[tuple[str, int, int]] = []
    seed = _FakeCipher(np.ones(2), limbs=12)

    class Stream:
        def __iter__(self):
            return iter(layers)

        def close(self) -> None:
            nonlocal stream_close_calls
            stream_close_calls += 1

    def layer_factory():
        nonlocal factory_calls
        factory_calls += 1
        return Stream()

    class Result:
        def __init__(self) -> None:
            self.kv_caches = []
            self.pairfold_kv_caches = []
            self.wall_seconds = 0.25
            self.per_layer = [{"stage_seconds": {"qkv": 0.1}}]
            self.release_calls = 0

        def release(self) -> None:
            self.release_calls += 1

    class Model:
        num_layers = len(layers)

        def decode(self, _hidden, *, layers, **kwargs):
            decoded_layer_ids.append(tuple(id(layer) for layer in layers))
            decode_first_layer_refreshes.append(bool(kwargs["refresh_first_layer"]))
            result = Result()
            results.append(result)
            return result

    def cleanup(device: str) -> None:
        cleanup_snapshots.append(
            (
                str(device),
                results[-1].release_calls,
                seed.clones[-1].release_calls,
            )
        )

    monkeypatch.setattr(experiment, "synchronize_device", lambda _device: None)
    monkeypatch.setattr(
        experiment, "begin_cuda_allocator_sample", lambda _device: 100
    )
    monkeypatch.setattr(
        experiment, "end_cuda_allocator_sample", lambda _device: 200
    )
    monkeypatch.setattr(
        experiment,
        "cuda_memory_snapshot",
        lambda device: {"device": str(device)},
    )
    monkeypatch.setattr(
        experiment, "_release_transient_device_memory", cleanup
    )

    report = experiment._run_fixed_decode_benchmark(
        decode_model=Model(),
        hidden=np.zeros((1, 2), dtype=np.float32),
        seed_input_ciphers=(seed,),
        decode_layers=layer_factory,
        context=SimpleNamespace(device="cpu"),
        schedule=object(),
        bootstrap=object(),
        refresh_first_layer=True,
        decode_packing="legacy_c",
        legacy_caches=[],
        pairfold_caches=[],
        prefix_bundles=None,
        occupied=0,
        position=0,
        max_kv_pages=1,
        pair_count=4,
        warmup_runs=2,
        measured_runs=3,
        order_label="A1",
    )

    assert factory_calls == 1
    assert stream_close_calls == 1
    assert decoded_layer_ids == [tuple(map(id, layers))] * 5
    assert decode_first_layer_refreshes == [True] * 5
    assert len(results) == 5
    assert all(result.release_calls == 1 for result in results)
    assert len(seed.clones) == 5
    assert all(clone.release_calls == 1 for clone in seed.clones)
    assert cleanup_snapshots == [("cpu", 1, 1)] * 5
    assert report["protocol"]["name"] == "fixed_decode_state_v2"
    assert report["protocol"][
        "layers_materialized_once_outside_timed_region"
    ] is True
    assert report["protocol"][
        "stable_layer_weight_approximation_identity"
    ] is True
    assert report["protocol"][
        "transient_device_cleanup_after_each_trial"
    ] is True
    assert report["protocol"][
        "session_plaintext_cache_preserved_between_trials"
    ] is True


def test_fixed_decode_cache_clones_survive_consuming_source_mutation() -> None:
    source_key = (_FakeCipher(np.ones(2), limbs=12),)
    source_value = (_FakeCipher(-np.ones(2), limbs=12),)
    source = [(source_key, source_value)]

    trial_legacy, trial_pairfold = experiment._clone_decode_cache_state(
        "legacy_c",
        legacy_caches=source,
        pairfold_caches=[],
    )
    moved_key, moved_value = trial_legacy[0]
    trial_legacy[0] = ((), ())
    experiment._release_legacy_caches(trial_legacy)

    assert trial_pairfold == []
    assert all(cipher.release_calls == 0 for cipher in (*source_key, *source_value))
    assert all(cipher.release_calls == 0 for cipher in (*moved_key, *moved_value))
    experiment._release_legacy_caches([(moved_key, moved_value)])
    assert all(cipher.release_calls == 1 for cipher in (*moved_key, *moved_value))
    assert all(cipher.release_calls == 0 for cipher in (*source_key, *source_value))
    experiment._release_legacy_caches(source)


def test_pairfold_benchmark_clone_tolerates_none_after_consume() -> None:
    layout = PairFold128Layout()
    keys, values = _logical_kv(layers=1)
    context = _FakeContext()
    source = experiment._encrypt_poly_pairfold_caches(
        keys,
        values,
        context=context,
        pairfold=layout,
        kv_limbs=12,
    )

    trial_legacy, trial_pairfold = experiment._clone_decode_cache_state(
        "pairfold128",
        legacy_caches=[],
        pairfold_caches=source,
    )
    moved = trial_pairfold[0]
    trial_pairfold[0] = None
    experiment._release_pairfold_caches(trial_pairfold)

    assert trial_legacy == []
    assert moved is not None
    assert moved is not source[0]
    assert not source[0].is_released
    moved.release()
    assert not source[0].is_released
    experiment._release_pairfold_caches(source)


def test_manifest_free_poly_cache_only_reuses_complete_no_prefix_stack(
    tmp_path,
) -> None:
    cache_path = tmp_path / "poly_cache.npz"
    assert not experiment._poly_cache_is_reusable(
        cache_path,
        prefix_k=0,
        layer_start=0,
        num_layers=32,
        checkpoint_layers=32,
    )
    cache_path.touch()

    assert experiment._poly_cache_is_reusable(
        cache_path,
        prefix_k=0,
        layer_start=0,
        num_layers=32,
        checkpoint_layers=32,
    )
    assert not experiment._poly_cache_is_reusable(
        cache_path,
        prefix_k=2,
        layer_start=0,
        num_layers=32,
        checkpoint_layers=32,
    )
    assert not experiment._poly_cache_is_reusable(
        cache_path,
        prefix_k=0,
        layer_start=8,
        num_layers=1,
        checkpoint_layers=32,
    )
    assert not experiment._poly_cache_is_reusable(
        cache_path,
        prefix_k=0,
        layer_start=0,
        num_layers=1,
        checkpoint_layers=32,
    )


def test_nonfinite_poly_state_fails_before_gpu_with_subset_hint() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"before GPU setup.*layer_start=8.*not a semantic model oracle",
    ):
        experiment._require_finite_poly(
            "Prefill hidden",
            np.asarray([0.0, np.nan], dtype=np.float32),
            layer_start=8,
        )


def test_compare_full_logits_reports_vocab_error_and_token_agreement() -> None:
    report = experiment._compare_full_logits(
        np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        np.asarray([1.5, 2.75, 2.0], dtype=np.float32),
    )

    assert report == {
        "max_abs": 0.5,
        "mean_abs": 0.25,
        "fhe_argmax_token_id": 1,
        "poly_argmax_token_id": 1,
        "argmax_match": True,
        "fhe_top_logit": 3.0,
        "poly_top_logit": 2.75,
        "top_logit_abs_diff": 0.25,
    }
    json.dumps(report)


def test_compare_full_logits_reports_argmax_mismatch() -> None:
    report = experiment._compare_full_logits(
        np.asarray([4.0, 3.0], dtype=np.float32),
        np.asarray([2.0, 5.0], dtype=np.float32),
    )

    assert report["fhe_argmax_token_id"] == 0
    assert report["poly_argmax_token_id"] == 1
    assert report["argmax_match"] is False
    assert report["top_logit_abs_diff"] == 1.0


@pytest.mark.parametrize(
    ("fhe_logits", "poly_logits", "label"),
    [
        ([0.0, np.nan], [0.0, 1.0], "FHE"),
        ([0.0, 1.0], [0.0, np.inf], "poly"),
    ],
)
def test_compare_full_logits_rejects_nonfinite_vocab(
    fhe_logits,
    poly_logits,
    label: str,
) -> None:
    with pytest.raises(RuntimeError, match=rf"{label} full logits contain 1/2"):
        experiment._compare_full_logits(fhe_logits, poly_logits)


def test_poly_decode_token_uses_checkpoint_layer_offset(monkeypatch) -> None:
    selected: list[int] = []
    calls: list[tuple[int, object]] = []

    class ApproxConfig:
        def for_layer(self, layer_idx: int):
            selected.append(int(layer_idx))
            return f"approx-{int(layer_idx)}"

    def fake_decode(
        hidden,
        *,
        cache_key,
        cache_value,
        layer_idx,
        prefix_bundle,
        **_kwargs,
    ):
        calls.append((int(layer_idx), prefix_bundle))
        return SimpleNamespace(
            hidden=np.asarray(hidden, dtype=np.float32) + 1.0,
            key=np.full_like(cache_key[:1], int(layer_idx)),
            value=np.full_like(cache_value[:1], -int(layer_idx)),
        )

    monkeypatch.setattr(experiment, "poly_decode_layer_numpy", fake_decode)
    prefixes = [object(), object()]
    hidden, keys, values = experiment._poly_decode_token(
        np.zeros((1, 2), dtype=np.float32),
        [np.zeros((1, 2), dtype=np.float32) for _ in range(2)],
        [np.zeros((1, 2), dtype=np.float32) for _ in range(2)],
        layer_weights=[object(), object()],
        approx_config=ApproxConfig(),
        attention=object(),
        position=128,
        rope_theta=500_000.0,
        layer_start=8,
        prefix_bundles=prefixes,
    )

    assert selected == [8, 9]
    assert calls == [(8, prefixes[0]), (9, prefixes[1])]
    np.testing.assert_array_equal(hidden, [[2.0, 2.0]])
    np.testing.assert_array_equal(keys[0][-1], [8.0, 8.0])
    np.testing.assert_array_equal(keys[1][-1], [9.0, 9.0])
    np.testing.assert_array_equal(values[0][-1], [-8.0, -8.0])
    np.testing.assert_array_equal(values[1][-1], [-9.0, -9.0])
