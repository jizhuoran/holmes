"""CPU ownership tests for the canonical QK -> Softmax -> PV core."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe import graph
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.graph import LayerApproximations, LayerWeights
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.attention.prefill import operator, softmax
from llama3fhe.operators.attention.prefill.input import ComplexTokenAttentionResult
from llama3fhe.operators.attention.prefill.softmax import DeltaSoftmaxAlg2Result
from llama3fhe.schedule import LayerSchedule


class _Cipher:
    def __init__(self, label: str, events: list[str]) -> None:
        self.label = str(label)
        self.events = events
        self.release_count = 0

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"released more than once: {self.label}")
        self.release_count = 1
        self.events.append(f"release:{self.label}")


def _layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=64,
    )


def _owner(
    query: _Cipher, key: _Cipher, value: _Cipher
) -> ComplexTokenAttentionResult:
    return ComplexTokenAttentionResult(
        query_ciphers=(query,),
        key_ciphers=(key,),
        value_ciphers=(value,),
        wall_seconds=0.0,
        stage_seconds={},
        scheduler_stats={},
    )


def _context():
    return SimpleNamespace(
        device="cpu",
        level_for_cipher=lambda _cipher: 0,
    )


def test_attention_core_releases_each_stream_at_its_synchronized_last_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    query = _Cipher("query", events)
    key = _Cipher("key", events)
    value = _Cipher("value", events)
    score = _Cipher("score", events)
    probability = _Cipher("probability", events)
    output = _Cipher("output", events)
    owner = _owner(query, key, value)

    def fake_qk(*_args, **_kwargs):  # noqa: ANN002, ANN003
        events.append("qk")
        assert query.release_count == key.release_count == 0
        return (score,)

    def fake_softmax(*_args, **_kwargs):  # noqa: ANN002, ANN003
        events.append("softmax")
        assert owner.query_ciphers == owner.key_ciphers == ()
        assert query.release_count == key.release_count == 1
        assert score.release_count == 0
        return DeltaSoftmaxAlg2Result(
            probability_ciphers=(probability,),
            stage_seconds={},
            output_levels=(0,),
        )

    def fake_pv(probabilities, values, **_kwargs):  # noqa: ANN001, ANN003
        events.append("pv")
        assert tuple(probabilities) == (probability,)
        assert tuple(values) == (value,)
        assert score.release_count == 1
        assert probability.release_count == value.release_count == 0
        return (output,)

    monkeypatch.delenv("LLAMA_FHE_STAGE_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(operator, "qk_delta_fhe", fake_qk)
    monkeypatch.setattr(softmax, "softmax_alg2_fhe_delta", fake_softmax)
    monkeypatch.setattr(operator, "pv_delta_fhe", fake_pv)
    # stage-timer syncs (profile_sync_device) and the semantic last-use
    # barriers (synchronize_device) both show up as "sync" events
    monkeypatch.setattr(
        operator, "sync_device", lambda _device: events.append("sync")
    )
    monkeypatch.setattr(
        operator, "synchronize_device", lambda _device: events.append("sync")
    )

    result = operator.attention_fhe_delta_alg2(
        owner.query_ciphers,
        owner.key_ciphers,
        owner.value_ciphers,
        layout=_layout(),
        crypto_context=_context(),
        softmax_config=object(),
        on_qk_last_use=owner.release_qk_inputs,
        on_value_last_use=owner.release_value_input,
        on_pv_last_use=lambda: events.append("pv-last-use"),
    )

    assert owner.query_ciphers == ()
    assert owner.key_ciphers == ()
    assert owner.value_ciphers == ()
    assert [
        query.release_count,
        key.release_count,
        score.release_count,
        probability.release_count,
        value.release_count,
        output.release_count,
    ] == [1, 1, 1, 1, 1, 0]
    assert events == [
        "qk",
        "sync",
        "release:query",
        "release:key",
        "softmax",
        "sync",
        "release:score",
        "pv",
        "sync",
        "release:probability",
        "release:value",
        "sync",
        "pv-last-use",
    ]

    owner.release()
    owner.release()
    result.release()
    assert output.release_count == 1


def test_attention_core_failure_synchronizes_then_releases_remaining_inputs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    query = _Cipher("query", events)
    key = _Cipher("key", events)
    value = _Cipher("value", events)
    score = _Cipher("score", events)
    probability = _Cipher("probability", events)
    owner = _owner(query, key, value)

    monkeypatch.delenv("LLAMA_FHE_STAGE_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(
        operator, "qk_delta_fhe", lambda *_args, **_kwargs: (score,)
    )
    monkeypatch.setattr(
        softmax,
        "softmax_alg2_fhe_delta",
        lambda *_args, **_kwargs: DeltaSoftmaxAlg2Result(
            probability_ciphers=(probability,),
            stage_seconds={},
            output_levels=(0,),
        ),
    )

    def fail_pv(*_args, **_kwargs):  # noqa: ANN002, ANN003
        events.append("pv-failed")
        raise RuntimeError("injected PV failure")

    monkeypatch.setattr(operator, "pv_delta_fhe", fail_pv)
    # stage-timer syncs (profile_sync_device) and the semantic last-use
    # barriers (synchronize_device) both show up as "sync" events
    monkeypatch.setattr(
        operator, "sync_device", lambda _device: events.append("sync")
    )
    monkeypatch.setattr(
        operator, "synchronize_device", lambda _device: events.append("sync")
    )

    with pytest.raises(RuntimeError, match="injected PV failure"):
        try:
            operator.attention_fhe_delta_alg2(
                owner.query_ciphers,
                owner.key_ciphers,
                owner.value_ciphers,
                layout=_layout(),
                crypto_context=_context(),
                softmax_config=object(),
                on_qk_last_use=owner.release_qk_inputs,
                on_value_last_use=owner.release_value_input,
            )
        finally:
            # Mirrors attention_fhe_from_qkv_carriers' outer owner cleanup.
            owner.release()

    assert [
        query.release_count,
        key.release_count,
        score.release_count,
        probability.release_count,
        value.release_count,
    ] == [1, 1, 1, 1, 1]
    assert events[-3:] == [
        "sync",
        "release:probability",
        "release:value",
    ]


def test_attention_adapter_syncs_before_carrier_last_use_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    carrier = _Cipher("carrier", events)
    query = _Cipher("query", events)
    key = _Cipher("key", events)
    value = _Cipher("value", events)
    adapter = _owner(query, key, value)

    def fake_prepare(*_args, **_kwargs):  # noqa: ANN002, ANN003
        events.append("adapter")
        return adapter

    def release_carrier() -> None:
        events.append("carrier-last-use")
        carrier.release()

    def fail_core(*_args, **_kwargs):  # noqa: ANN002, ANN003
        events.append("core")
        raise RuntimeError("stop after adapter boundary")

    monkeypatch.delenv("LLAMA_FHE_STAGE_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(operator, "prepare_attention_inputs_fhe", fake_prepare)
    monkeypatch.setattr(operator, "attention_fhe_delta_alg2", fail_core)
    # stage-timer syncs (profile_sync_device) and the semantic last-use
    # barriers (synchronize_device) both show up as "sync" events
    monkeypatch.setattr(
        operator, "sync_device", lambda _device: events.append("sync")
    )
    monkeypatch.setattr(
        operator, "synchronize_device", lambda _device: events.append("sync")
    )

    with pytest.raises(RuntimeError, match="stop after adapter boundary"):
        operator.attention_fhe_from_qkv_carriers(
            (carrier,),
            carrier_layout=object(),
            feature_layout=object(),
            attention_layout=object(),
            crypto_context=_context(),
            softmax_config=object(),
            on_carriers_last_use=release_carrier,
        )

    assert events[:5] == [
        "adapter",
        "sync",
        "carrier-last-use",
        "release:carrier",
        "core",
    ]
    assert carrier.release_count == 1
    assert query.release_count == key.release_count == value.release_count == 1


def test_attention_exclusive_cache_helper_syncs_then_prunes_exact_keep_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class _NativeContext:
        def clear_cuda_rotation_cache(self, **kwargs) -> None:
            events.append(("clear", kwargs))

    context = SimpleNamespace(
        device="cuda:0", context=_NativeContext()
    )
    monkeypatch.setattr(
        graph.common,
        "synchronize_device",
        lambda device: events.append(("sync", device)),
    )

    elapsed = graph._evict_attention_exclusive_cuda_cache(
        context, keep_rotations=(-128, 128, 16384)
    )

    assert elapsed >= 0.0
    assert events == [
        ("sync", "cuda:0"),
        (
            "clear",
            {
                "keep_rotations": (-128, 128, 16384),
                "empty_allocator_cache": True,
            },
        ),
    ]


class _GraphCipher:
    def __init__(self, label: str, *, limbs: int = 20) -> None:
        self.label = str(label)
        self.state = SimpleNamespace(
            cur_limbs=int(limbs),
            scale_degree=1,
            scaling_factor=1.0,
        )
        self.slots = 32768
        self.is_ext = False
        self.batch_size = 1
        self.cv = (object(), object())
        self.release_count = 0

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"released more than once: {self.label}")
        self.release_count = 1


class _GraphOwnedResult:
    def __init__(self, ciphers: tuple[_GraphCipher, ...]) -> None:
        self.ciphers = tuple(ciphers)
        self.stage_seconds = {}
        self.level_schedule = {}
        self.output_levels = (0,)
        self.slim_bootstrap_output_levels = ()
        self.bootstrap_output_levels = ()
        self.refreshed_input_ciphers = ()
        self.deferred_output_scale = None

    def release(self) -> None:
        owned = self.ciphers
        self.ciphers = ()
        for cipher in owned:
            cipher.release()


class _GraphQKVResult:
    def __init__(self, carrier: _GraphCipher) -> None:
        self.carrier_ciphers = (carrier,)
        self.output_layout = object()
        self.stage_seconds = {}
        self.output_levels = (0,)

    def release(self) -> None:
        owned = self.carrier_ciphers
        self.carrier_ciphers = ()
        for cipher in owned:
            cipher.release()


class _GraphQKNormResult(_GraphOwnedResult):
    pass


class _GraphContext:
    device = "cpu"

    @staticmethod
    def level_for_cipher(cipher: _GraphCipher) -> int:
        return 40 - int(cipher.state.cur_limbs)


class _GraphBootstrap:
    calls = 0
    native_calls = 0
    cache_eviction_seconds = 0.0


def _graph_weights(*, qk_norm: bool) -> LayerWeights:
    scalar = np.ones((1,), dtype=np.float32)
    norm_weight = scalar if qk_norm else None
    return LayerWeights(
        input_layernorm=scalar,
        q_proj=scalar,
        k_proj=scalar,
        v_proj=scalar,
        o_proj=scalar,
        post_attention_layernorm=scalar,
        gate_proj=scalar,
        up_proj=scalar,
        down_proj=scalar,
        q_norm=norm_weight,
        k_norm=norm_weight,
    )


def _graph_approximations(*, qk_norm: bool) -> LayerApproximations:
    norm = {
        "coefficients": np.asarray((1.0, 1.0)),
        "fit_interval": (0.0, 1.0),
    }
    return LayerApproximations(
        input_norm=norm,
        post_attention_norm=norm,
        silu={
            "coefficients": np.asarray((0.0, 1.0)),
            "fit_interval": (-1.0, 1.0),
        },
        softmax=object(),
        qk_norm=(norm if qk_norm else None),
    )


@pytest.mark.parametrize("with_qk_norm", (False, True))
@pytest.mark.parametrize("evict_exclusive", (False, True))
def test_graph_releases_carrier_owners_once_when_attention_fails_after_adapter(
    monkeypatch: pytest.MonkeyPatch,
    with_qk_norm: bool,
    evict_exclusive: bool,
) -> None:
    feature_layout = FeatureMajorPrefillLayout()
    qkv_carrier = _GraphCipher("qkv-carrier")
    qk_norm_carrier = _GraphCipher("qk-norm-carrier")
    qkv_result = _GraphQKVResult(qkv_carrier)
    qk_norm_result = _GraphQKNormResult((qk_norm_carrier,))
    seen: dict[str, object] = {}
    evictions: list[tuple[int, ...]] = []

    monkeypatch.setattr(graph.prefill, "required_rotations", lambda *_a, **_k: ())
    monkeypatch.setattr(
        graph.prefill,
        "rmsnorm_fhe",
        lambda *_a, **_k: _GraphOwnedResult(
            tuple(
                _GraphCipher(f"rms-{index}", limbs=13)
                for index in range(feature_layout.cipher_count)
            )
        ),
    )
    monkeypatch.setattr(graph.prefill, "qkv_fhe", lambda *_a, **_k: qkv_result)
    monkeypatch.setattr(
        graph.prefill,
        "_attention_exclusive_cache_keep_rotations",
        lambda *_a, **_k: {
            "qk_to_softmax": (11, 12),
            "pv_to_pairfm": (21, 22),
        },
    )
    monkeypatch.setattr(
        graph.prefill,
        "_evict_attention_exclusive_cuda_cache",
        lambda _context, *, keep_rotations: (
            evictions.append(tuple(keep_rotations)) or 0.0
        ),
    )
    if with_qk_norm:
        monkeypatch.setattr(
            graph.prefill, "qk_norm_fhe", lambda *_a, **_k: qk_norm_result
        )

    def fail_after_adapter(carriers, **kwargs):  # noqa: ANN001, ANN003
        seen["carriers"] = tuple(carriers)
        assert qkv_carrier.release_count == 0
        assert qk_norm_carrier.release_count == 0
        kwargs["on_carriers_last_use"]()
        assert qkv_carrier.release_count == 1
        assert qk_norm_carrier.release_count == int(with_qk_norm)
        assert callable(kwargs["on_qk_last_use"]) is evict_exclusive
        assert callable(kwargs["on_pv_last_use"]) is evict_exclusive
        if evict_exclusive:
            kwargs["on_qk_last_use"]()
            kwargs["on_pv_last_use"]()
        raise RuntimeError("injected post-adapter failure")

    monkeypatch.setattr(
        graph.prefill, "attention_fhe_from_qkv_carriers", fail_after_adapter
    )
    inputs = tuple(
        _GraphCipher(f"input-{index}")
        for index in range(feature_layout.cipher_count)
    )

    with pytest.raises(RuntimeError, match="post-adapter failure"):
        graph.evaluate(
            np.zeros((128, 4096), dtype=np.float32),
            layer_idx=0,
            weights=_graph_weights(qk_norm=with_qk_norm),
            approximations=_graph_approximations(qk_norm=with_qk_norm),
            attention_layout=AttentionPairLayout(),
            config=Llama3CKKSConfig(),
            crypto_context=_GraphContext(),
            input_ciphers=inputs,
            bootstrap_operator=_GraphBootstrap(),
            schedule=LayerSchedule(
                adaptive_rmsnorm_slim_refresh=False,
                fuse_rmsnorm_output_scale=False,
                evict_attention_exclusive_cuda_cache=evict_exclusive,
            ),
        )

    expected = (qk_norm_carrier,) if with_qk_norm else (qkv_carrier,)
    assert seen["carriers"] == expected
    assert qkv_carrier.release_count == 1
    assert qk_norm_carrier.release_count == int(with_qk_norm)
    assert all(cipher.release_count == 0 for cipher in inputs)
    assert evictions == ([(11, 12), (21, 22)] if evict_exclusive else [])
