"""CPU-only ownership gates for PairFold128 Prefill cache emission."""

from __future__ import annotations

from dataclasses import dataclass, replace
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe import graph
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.graph import LayerApproximations, LayerWeights
from llama3fhe.layouts import AttentionPairLayout, FeatureMajorPrefillLayout
from llama3fhe.operators.attention.decode import decode
from llama3fhe.operators.attention.pairfold import pairfold_prefill
from llama3fhe.operators.pairfold_kv_cache import (
    PairFoldCipherPage,
    PairFoldLayerCache,
)
from llama3fhe.schedule import LayerSchedule


@dataclass(frozen=True)
class _State:
    cur_limbs: int = 20
    scale_degree: int = 1
    scaling_factor: float | None = 1.0

    def replace(self, **changes) -> _State:  # noqa: ANN003
        return replace(self, **changes)


class _Cipher:
    def __init__(self, label: str, *, limbs: int = 20) -> None:
        self.label = str(label)
        self.state = _State(cur_limbs=int(limbs))
        self.slots = 32768
        self.is_ext = False
        self.batch_size = 1
        self.cv = (object(), object())
        self.release_count = 0

    def deep_copy(self) -> _Cipher:
        return _Cipher(f"{self.label}.copy", limbs=self.state.cur_limbs)

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"cipher released more than once: {self.label}")
        self.release_count = 1


class _Fhe:
    def align_to(self, cipher: _Cipher, state: _State, _context) -> _Cipher:
        output = _Cipher(f"{cipher.label}.drop", limbs=state.cur_limbs)
        output.state = state
        return output


class _Context:
    def __init__(self) -> None:
        self.context = SimpleNamespace()
        self.fhe = _Fhe()
        self.device = "cpu"

    @staticmethod
    def level_for_cipher(cipher: _Cipher) -> int:
        return 40 - int(cipher.state.cur_limbs)


def _raw_stream(prefix: str, *, limbs: int = 9) -> tuple[_Cipher, ...]:
    return tuple(_Cipher(f"{prefix}{index}", limbs=limbs) for index in range(4))


def test_prefill_evaluate_signature_selects_pairfold_default() -> None:
    parameters = signature(graph.evaluate).parameters
    assert parameters["decode_packing"].default == "pairfold128"
    assert parameters["prefill_occupied"].default is None


def test_pairfold_prefill_cache_converts_then_drops_transactionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", "1")
    raw_key = _raw_stream("raw-k")
    raw_value = _raw_stream("raw-v")
    converted = tuple(_Cipher(f"converted-{index}", limbs=9) for index in range(4))
    calls: list[tuple[tuple[object, ...], tuple[object, ...]]] = []

    def fake_convert(keys, values, **_kwargs):  # noqa: ANN001, ANN003
        keys = tuple(keys)
        values = tuple(values)
        calls.append((keys, values))
        for cipher in keys + values:
            cipher.release()
        return converted[:2], converted[2:]

    monkeypatch.setattr(
        pairfold_prefill, "prefill_c_to_pairfold_fhe", fake_convert
    )
    context = _Context()

    cache = graph._pairfold_prefill_cache_from_raw_c(
        raw_key,
        raw_value,
        attention_layout=AttentionPairLayout(),
        crypto_context=context,
        target_limbs=6,
        bootstrap_operator=None,
        bootstrap_iterations=1,
        occupied=128,
    )

    assert calls == [(raw_key, raw_value)]
    assert all(cipher.release_count == 1 for cipher in raw_key + raw_value)
    assert all(cipher.release_count == 1 for cipher in converted)
    assert cache.content_tokens == 128
    assert len(cache.pages) == 1
    assert cache.pages[0].occupied == 128
    assert len(cache.pages[0].handles) == 4
    assert {
        int(cipher.state.cur_limbs) for cipher in cache.pages[0].handles
    } == {6}
    cache_handles = cache.pages[0].handles
    cache.release()
    assert all(cipher.release_count == 1 for cipher in cache_handles)


def test_pairfold_prefill_emits_partial_page_and_consumes_raw_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_key = _raw_stream("raw-k")
    raw_value = _raw_stream("raw-v")
    converted = tuple(_Cipher(f"partial-{index}", limbs=9) for index in range(4))
    seen = {}

    def fake_convert(keys, values, **kwargs):  # noqa: ANN001, ANN003
        seen.update(kwargs)
        for cipher in tuple(keys) + tuple(values):
            cipher.release()
        return converted[:2], converted[2:]

    monkeypatch.setattr(
        pairfold_prefill, "prefill_c_to_pairfold_fhe", fake_convert
    )

    cache = graph._pairfold_prefill_cache_from_raw_c(
        raw_key,
        raw_value,
        attention_layout=AttentionPairLayout(),
        crypto_context=_Context(),
        target_limbs=None,
        bootstrap_operator=None,
        bootstrap_iterations=1,
        occupied=127,
    )

    assert seen["occupied"] == 127
    assert cache.content_tokens == 127
    assert cache.pages[0].occupied == 127
    assert all(cipher.release_count == 1 for cipher in raw_key + raw_value)
    cache.release()


def test_pairfold_prefill_direct_key_profiles_conversion_and_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_key = _raw_stream("raw-k", limbs=8)
    raw_value = _raw_stream("raw-v", limbs=10)
    converted_key = tuple(_Cipher(f"key-{index}", limbs=5) for index in range(2))
    converted_value = tuple(_Cipher(f"value-{index}", limbs=9) for index in range(2))
    calls: list[tuple[str, int]] = []
    refreshed: list[_Cipher] = []

    def convert_key(keys, **kwargs):  # noqa: ANN001, ANN003
        calls.append(("key", kwargs["occupied"]))
        for cipher in keys:
            cipher.release()
        return converted_key

    def convert_value(values, **kwargs):  # noqa: ANN001, ANN003
        calls.append(("value", kwargs["occupied"]))
        for cipher in values:
            cipher.release()
        return converted_value

    def forbidden_legacy(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("direct-K flag entered legacy converter")

    class RefreshBootstrap:
        def refresh(self, cipher: _Cipher, *, iterations: int) -> _Cipher:
            assert iterations == 2
            output = _Cipher(f"{cipher.label}.refresh", limbs=9)
            refreshed.append(output)
            return output

    monkeypatch.setenv("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", "1")
    monkeypatch.setattr(
        pairfold_prefill, "prefill_emitted_key_c_direct_to_pairfold_fhe", convert_key
    )
    monkeypatch.setattr(
        pairfold_prefill, "prefill_real_value_c_to_pairfold_fhe", convert_value
    )
    monkeypatch.setattr(pairfold_prefill, "prefill_c_to_pairfold_fhe", forbidden_legacy)
    profile: dict[str, object] = {}
    cache = graph._pairfold_prefill_cache_from_raw_c(
        raw_key,
        raw_value,
        attention_layout=AttentionPairLayout(),
        crypto_context=_Context(),
        target_limbs=6,
        bootstrap_operator=RefreshBootstrap(),
        bootstrap_iterations=2,
        occupied=127,
        emitted_key_c=True,
        profile=profile,
    )

    assert calls == [("key", 127), ("value", 127)]
    assert cache.content_tokens == 127
    assert cache.pages[0].occupied == 127
    assert profile["direct_key"] is True
    assert profile["raw_key_limbs"] == (8, 8, 8, 8)
    assert profile["raw_value_limbs"] == (10, 10, 10, 10)
    assert profile["converted_key_limbs"] == (5, 5)
    assert profile["converted_value_limbs"] == (9, 9)
    assert profile["cache_key_limbs"] == (6, 6)
    assert profile["cache_value_limbs"] == (6, 6)
    assert profile["refresh_bootstrap_calls"] == 2
    assert all(
        float(profile[name]) >= 0
        for name in ("convert_seconds", "refresh_drop_seconds", "page_seconds")
    )
    assert all(
        cipher.release_count == 1
        for cipher in (
            raw_key + raw_value + converted_key + converted_value + tuple(refreshed)
        )
    )
    cache_handles = cache.pages[0].handles
    cache.release()
    assert all(cipher.release_count == 1 for cipher in cache_handles)


def test_pairfold_prefill_split_target_profiles_only_key_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_key = _raw_stream("raw-k", limbs=8)
    prepacked_value = tuple(_Cipher(f"packed-v-{i}", limbs=8) for i in range(2))
    converted_key = tuple(_Cipher(f"key-{i}", limbs=5) for i in range(2))
    refreshed: list[_Cipher] = []

    def convert_key(keys, **_kwargs):  # noqa: ANN001, ANN003
        for cipher in keys:
            cipher.release()
        return converted_key

    class RefreshBootstrap:
        def refresh(self, cipher: _Cipher, *, iterations: int) -> _Cipher:
            assert iterations == 1
            output = _Cipher(f"{cipher.label}.refresh", limbs=16)
            refreshed.append(output)
            return output

    monkeypatch.setenv("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", "1")
    monkeypatch.setattr(
        pairfold_prefill, "prefill_emitted_key_c_direct_to_pairfold_fhe", convert_key
    )
    profile: dict[str, object] = {}
    cache = graph._pairfold_prefill_cache_from_raw_c(
        raw_key,
        (),
        prepacked_value_ciphers=prepacked_value,
        attention_layout=AttentionPairLayout(),
        crypto_context=_Context(),
        target_limbs=12,
        value_target_limbs=8,
        bootstrap_operator=RefreshBootstrap(),
        bootstrap_iterations=1,
        occupied=128,
        emitted_key_c=True,
        profile=profile,
    )

    assert profile["converted_key_limbs"] == (5, 5)
    assert profile["converted_value_limbs"] == (8, 8)
    assert profile["refresh_bootstrap_calls"] == 2
    assert profile["cache_key_limbs"] == (12, 12)
    assert profile["cache_value_limbs"] == (8, 8)
    assert len(refreshed) == 2
    assert all(
        cipher.release_count == 1
        for cipher in raw_key + prepacked_value + converted_key + tuple(refreshed)
    )
    cache.release()


def test_pairfold_prefill_rejects_value_target_without_key_target() -> None:
    raw_key = _raw_stream("raw-k")
    raw_value = _raw_stream("raw-v")

    with pytest.raises(ValueError, match="requires a K limb target"):
        graph._pairfold_prefill_cache_from_raw_c(
            raw_key,
            raw_value,
            attention_layout=AttentionPairLayout(),
            crypto_context=_Context(),
            target_limbs=None,
            value_target_limbs=8,
            bootstrap_operator=None,
            bootstrap_iterations=1,
            occupied=128,
        )

    assert all(cipher.release_count == 1 for cipher in raw_key + raw_value)


@pytest.mark.parametrize("fail_stream", ("key", "value"))
def test_pairfold_prefill_direct_key_failure_releases_each_owner_once(
    monkeypatch: pytest.MonkeyPatch,
    fail_stream: str,
) -> None:
    raw_key = _raw_stream("raw-k")
    raw_value = _raw_stream("raw-v")
    converted_key = tuple(_Cipher(f"key-{index}") for index in range(2))

    def convert_key(keys, **_kwargs):  # noqa: ANN001, ANN003
        for cipher in keys:
            cipher.release()
        if fail_stream == "key":
            raise RuntimeError("injected K conversion failure")
        return converted_key

    def convert_value(values, **_kwargs):  # noqa: ANN001, ANN003
        for cipher in values:
            cipher.release()
        raise RuntimeError("injected V conversion failure")

    monkeypatch.setenv("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", "1")
    monkeypatch.setattr(
        pairfold_prefill, "prefill_emitted_key_c_direct_to_pairfold_fhe", convert_key
    )
    monkeypatch.setattr(
        pairfold_prefill, "prefill_real_value_c_to_pairfold_fhe", convert_value
    )
    profile: dict[str, object] = {}
    with pytest.raises(
        RuntimeError,
        match=f"{dict(key='K', value='V')[fail_stream]} conversion failure",
    ):
        graph._pairfold_prefill_cache_from_raw_c(
            raw_key,
            raw_value,
            attention_layout=AttentionPairLayout(),
            crypto_context=_Context(),
            target_limbs=None,
            bootstrap_operator=None,
            bootstrap_iterations=1,
            occupied=128,
            emitted_key_c=True,
            profile=profile,
        )

    assert profile == {}
    assert all(cipher.release_count == 1 for cipher in raw_key + raw_value)
    assert all(
        cipher.release_count == int(fail_stream == "value")
        for cipher in converted_key
    )


@pytest.mark.parametrize("fail_key", (False, True))
def test_pairfold_prefill_prepacked_value_ownership(
    monkeypatch: pytest.MonkeyPatch,
    fail_key: bool,
) -> None:
    raw_key = _raw_stream("raw-k", limbs=8)
    prepacked_value = tuple(_Cipher(f"packed-v-{index}", limbs=9) for index in range(2))
    converted_key = tuple(_Cipher(f"packed-k-{index}", limbs=7) for index in range(2))

    def convert_key(keys, **_kwargs):  # noqa: ANN001, ANN003
        assert tuple(keys) == raw_key
        for cipher in keys:
            cipher.release()
        if fail_key:
            raise RuntimeError("injected K conversion failure")
        return converted_key

    monkeypatch.setenv("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", "0")
    monkeypatch.setattr(
        pairfold_prefill, "prefill_emitted_key_c_to_pairfold_fhe", convert_key
    )
    profile: dict[str, object] = {}
    if fail_key:
        with pytest.raises(RuntimeError, match="K conversion failure"):
            graph._pairfold_prefill_cache_from_raw_c(
                raw_key,
                (),
                prepacked_value_ciphers=prepacked_value,
                attention_layout=AttentionPairLayout(),
                crypto_context=_Context(),
                target_limbs=None,
                bootstrap_operator=None,
                bootstrap_iterations=1,
                occupied=127,
                emitted_key_c=True,
                profile=profile,
            )
        assert profile == {}
    else:
        cache = graph._pairfold_prefill_cache_from_raw_c(
            raw_key,
            (),
            prepacked_value_ciphers=prepacked_value,
            attention_layout=AttentionPairLayout(),
            crypto_context=_Context(),
            target_limbs=None,
            bootstrap_operator=None,
            bootstrap_iterations=1,
            occupied=127,
            emitted_key_c=True,
            profile=profile,
        )
        assert cache.pages[0].value_groups == prepacked_value
        assert profile["prepacked_value"] is True
        assert profile["raw_value_limbs"] == (9, 9)
        assert all(cipher.release_count == 0 for cipher in prepacked_value)
        cache.release()

    assert all(cipher.release_count == 1 for cipher in raw_key + prepacked_value)
    assert all(
        cipher.release_count == int(not fail_key) for cipher in converted_key
    )


def test_evaluate_rejects_partial_pairfold_prefill_before_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"attention": 0, "converter": 0}

    def forbidden_attention(*_args, **_kwargs):  # noqa: ANN002, ANN003
        calls["attention"] += 1
        raise AssertionError("partial PairFold Prefill reached attention")

    def forbidden_converter(*_args, **_kwargs):  # noqa: ANN002, ANN003
        calls["converter"] += 1
        raise AssertionError("partial PairFold Prefill reached converter")

    monkeypatch.setattr(
        graph.prefill, "attention_fhe_from_qkv_carriers", forbidden_attention
    )
    monkeypatch.setattr(
        pairfold_prefill, "prefill_c_to_pairfold_fhe", forbidden_converter
    )

    with pytest.raises(ValueError, match="right-padded token_valid_mask"):
        graph.evaluate(
            np.zeros((1, 1), dtype=np.float32),
            layer_idx=0,
            weights=_weights(),
            approximations=_approximations(),
            attention_layout=AttentionPairLayout(),
            config=Llama3CKKSConfig(),
            crypto_context=None,
            input_ciphers=(),
            bootstrap_operator=None,
            emit_kv_cache=True,
            decode_packing="pairfold128",
            prefill_occupied=127,
        )

    assert calls == {"attention": 0, "converter": 0}


@pytest.mark.parametrize(
    ("valid_mask", "message"),
    (
        (np.ones((127,), dtype=bool), "must have shape"),
        (
            np.concatenate(
                (np.ones((127,), dtype=bool), np.zeros((1,), dtype=bool))
            ),
            "requires valid tokens",
        ),
    ),
)
def test_evaluate_rejects_pairfold_padded_mask_before_attention(
    monkeypatch: pytest.MonkeyPatch,
    valid_mask: np.ndarray,
    message: str,
) -> None:
    calls = {"attention": 0, "converter": 0}

    def forbidden_attention(*_args, **_kwargs):  # noqa: ANN002, ANN003
        calls["attention"] += 1
        raise AssertionError("padded PairFold Prefill reached attention")

    def forbidden_converter(*_args, **_kwargs):  # noqa: ANN002, ANN003
        calls["converter"] += 1
        raise AssertionError("padded PairFold Prefill reached converter")

    monkeypatch.setattr(
        graph.prefill, "attention_fhe_from_qkv_carriers", forbidden_attention
    )
    monkeypatch.setattr(
        pairfold_prefill, "prefill_c_to_pairfold_fhe", forbidden_converter
    )

    with pytest.raises(ValueError, match=message):
        graph.evaluate(
            np.zeros((1, 1), dtype=np.float32),
            layer_idx=0,
            weights=_weights(),
            approximations=_approximations(),
            attention_layout=AttentionPairLayout(),
            config=Llama3CKKSConfig(),
            crypto_context=None,
            input_ciphers=(),
            bootstrap_operator=None,
            attention_token_valid_mask=valid_mask,
            emit_kv_cache=True,
            decode_packing="pairfold128",
        )

    assert calls == {"attention": 0, "converter": 0}


def test_pairfold_prefill_drop_failure_releases_converted_outputs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama3fhe.operators import pairfold_kv_cache

    raw_key = _raw_stream("raw-k")
    raw_value = _raw_stream("raw-v")
    converted = tuple(_Cipher(f"converted-{index}", limbs=9) for index in range(4))

    def fake_convert(keys, values, **_kwargs):  # noqa: ANN001, ANN003
        for cipher in tuple(keys) + tuple(values):
            cipher.release()
        return converted[:2], converted[2:]

    def fail_drop(_token, **_kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("injected four-way drop failure")

    monkeypatch.setattr(
        pairfold_prefill, "prefill_c_to_pairfold_fhe", fake_convert
    )
    monkeypatch.setattr(
        pairfold_kv_cache, "drop_pairfold_token_to_limbs", fail_drop
    )

    with pytest.raises(RuntimeError, match="four-way drop failure"):
        graph._pairfold_prefill_cache_from_raw_c(
            raw_key,
            raw_value,
            attention_layout=AttentionPairLayout(),
            crypto_context=_Context(),
            target_limbs=6,
            bootstrap_operator=None,
            bootstrap_iterations=1,
            occupied=128,
        )

    assert all(cipher.release_count == 1 for cipher in raw_key + raw_value)
    assert all(cipher.release_count == 1 for cipher in converted)


class _OwnedResult:
    def __init__(self, ciphers: tuple[_Cipher, ...]) -> None:
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


class _QKVResult(_OwnedResult):
    def __init__(self) -> None:
        super().__init__((_Cipher("qkv-carrier"),))
        self.carrier_ciphers = self.ciphers
        self.output_layout = object()

    def release(self) -> None:
        self.ciphers = self.carrier_ciphers
        self.carrier_ciphers = ()
        super().release()


class _AttentionResult:
    def __init__(self, *, direct_value: bool = False) -> None:
        self.w_o_input_ciphers = (_Cipher("attention-o"),)
        self.key_c_ciphers = _raw_stream("attention-raw-k")
        self.value_c_ciphers = () if direct_value else _raw_stream("attention-raw-v")
        self.pairfold_value_ciphers = (
            tuple(_Cipher(f"attention-pairfold-v-{index}") for index in range(2))
            if direct_value
            else ()
        )
        self.delta_kv_page = None
        self.stage_seconds = {}
        self.input_scheduler_stats = {}
        self.level_trace = {}
        self.output_levels = (0,)

    def release(self) -> None:
        owned = (
            tuple(self.w_o_input_ciphers)
            + tuple(self.key_c_ciphers)
            + tuple(self.value_c_ciphers)
            + tuple(self.pairfold_value_ciphers)
        )
        self.w_o_input_ciphers = ()
        self.key_c_ciphers = ()
        self.value_c_ciphers = ()
        self.pairfold_value_ciphers = ()
        for cipher in owned:
            cipher.release()


class _Bootstrap:
    calls = 0
    native_calls = 0
    cache_eviction_seconds = 0.0


def _weights() -> LayerWeights:
    scalar = np.ones((1,), dtype=np.float32)
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
    )


def _approximations() -> LayerApproximations:
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
    )


@pytest.mark.parametrize("direct_value", (False, True))
def test_evaluate_pairfold_emit_bypasses_legacy_key_repair(
    monkeypatch: pytest.MonkeyPatch,
    direct_value: bool,
) -> None:
    config = Llama3CKKSConfig()
    attention_layout = AttentionPairLayout()
    feature_layout = FeatureMajorPrefillLayout()
    context = _Context()
    input_ciphers = tuple(
        _Cipher(f"input-{index}") for index in range(feature_layout.cipher_count)
    )
    captured: dict[str, object] = {}
    monkeypatch.setenv("LLAMA_FHE_PREFILL_DELTA_PAIRFOLD_V", str(int(direct_value)))
    attention_result = _AttentionResult(direct_value=direct_value)
    raw_key = attention_result.key_c_ciphers
    raw_value = attention_result.value_c_ciphers
    prepacked_value = attention_result.pairfold_value_ciphers

    def fake_required_rotations(*_args, **kwargs):  # noqa: ANN003
        captured["rotation_packing"] = kwargs["decode_packing"]
        return ()

    def fake_rmsnorm(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return _OwnedResult(
            tuple(
                _Cipher(f"rms1-{index}", limbs=13)
                for index in range(feature_layout.cipher_count)
            )
        )

    def fake_post_norm(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return _OwnedResult(
            tuple(
                _Cipher(f"rms2-{index}")
                for index in range(feature_layout.cipher_count)
            )
        )

    def fake_w_o(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return tuple(
            _Cipher(f"w-o-{index}")
            for index in range(feature_layout.cipher_count)
        )

    def fake_residual(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return _OwnedResult(
            tuple(
                _Cipher(f"residual-{index}")
                for index in range(feature_layout.cipher_count)
            )
        )

    def fake_mlp(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return _OwnedResult(
            tuple(
                _Cipher(f"mlp-{index}")
                for index in range(feature_layout.cipher_count)
            )
        )

    def fake_pairfold_cache(keys, values, **kwargs):  # noqa: ANN001, ANN003
        keys = tuple(keys)
        values = tuple(values)
        prepacked = tuple(kwargs["prepacked_value_ciphers"])
        captured["raw_keys"] = keys
        captured["raw_values"] = values
        captured["prepacked_values"] = prepacked
        captured["occupied"] = kwargs["occupied"]
        captured["target_limbs"] = kwargs["target_limbs"]
        captured["value_target_limbs"] = kwargs["value_target_limbs"]
        kwargs["profile"]["convert_seconds"] = 0.25
        for cipher in keys + values + prepacked:
            cipher.release()
        handles = tuple(_Cipher(f"pairfold-{index}") for index in range(4))
        page = PairFoldCipherPage(
            key_groups=handles[:2], value_groups=handles[2:], occupied=128
        )
        return PairFoldLayerCache(pages=(page,), content_tokens=128)

    def forbidden_legacy(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("PairFold emit entered legacy K repair")

    monkeypatch.setattr(graph.prefill, "required_rotations", fake_required_rotations)
    monkeypatch.setattr(graph.prefill, "rmsnorm_fhe", fake_rmsnorm)
    monkeypatch.setattr(graph.prefill, "qkv_fhe", lambda *_args, **_kwargs: _QKVResult())
    def fake_attention(*_args, **kwargs):  # noqa: ANN002, ANN003
        captured["direct_value"] = kwargs["direct_pairfold_v"]
        return attention_result

    monkeypatch.setattr(
        graph.prefill, "attention_fhe_from_qkv_carriers", fake_attention
    )
    monkeypatch.setattr(graph.prefill, "w_o_fhe", fake_w_o)
    monkeypatch.setattr(graph.prefill, "residual_add", fake_residual)
    monkeypatch.setattr(graph.prefill, "bootstrap_rmsnorm_fhe", fake_post_norm)
    monkeypatch.setattr(graph.prefill, "mlp_fhe", fake_mlp)
    monkeypatch.setattr(
        graph.prefill, "_pairfold_prefill_cache_from_raw_c", fake_pairfold_cache
    )
    monkeypatch.setattr(
        graph.prefill,
        "drop_levels",
        lambda ciphers, _levels, **_kwargs: tuple(
            cipher.deep_copy() for cipher in ciphers
        ),
    )
    monkeypatch.setattr(graph.prefill, "trim_device_allocator_cache", lambda _device: None)
    monkeypatch.setattr(
        decode, "folded_key_c_to_real_c_fhe", forbidden_legacy
    )
    monkeypatch.setattr(
        decode, "correct_prefill_emitted_key_c_fhe", forbidden_legacy
    )

    result = graph.evaluate(
        np.zeros((128, 4096), dtype=np.float32),
        layer_idx=0,
        weights=_weights(),
        approximations=_approximations(),
        attention_layout=attention_layout,
        config=config,
        crypto_context=context,
        input_ciphers=input_ciphers,
        bootstrap_operator=_Bootstrap(),
        schedule=LayerSchedule(kv_limbs=12, kv_value_limbs=8),
        emit_kv_cache=True,
        decode_packing="pairfold128",
    )

    assert captured["rotation_packing"] == "pairfold128"
    assert captured["raw_keys"] == raw_key
    assert captured["raw_values"] == raw_value
    assert captured["prepacked_values"] == prepacked_value
    assert captured["direct_value"] == direct_value
    assert captured["occupied"] == 128
    assert captured["target_limbs"] == 12
    assert captured["value_target_limbs"] == 8
    assert result.key_c_ciphers == ()
    assert result.value_c_ciphers == ()
    assert result.pairfold_kv_cache is not None
    assert result.pairfold_kv_cache.content_tokens == 128
    assert "prefill_to_pairfold_once" in result.stage_seconds
    assert result.nested_stage_seconds["kv_cache"]["profile"] == {
        "convert_seconds": 0.25
    }
    assert all(
        cipher.release_count == 1
        for cipher in raw_key + raw_value + prepacked_value
    )
    assert all(cipher.release_count == 0 for cipher in input_ciphers)
    result.release()
