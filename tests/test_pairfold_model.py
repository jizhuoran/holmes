"""CPU-only ownership gates for model-level PairFold128 integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import llama3fhe.model as model_module
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.graph import LayerResult
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.model import (
    DecodeModelResult,
    Llama3Model,
    ModelResult,
    TransformerLayer,
)
from llama3fhe.operators.pairfold_kv_cache import (
    PairFoldCipherPage,
    PairFoldLayerCache,
)


class _Owned:
    def __init__(self, label: str) -> None:
        self.label = str(label)
        self.release_count = 0
        self.copies: list[_Owned] = []

    def deep_copy(self) -> _Owned:
        copied = _Owned(f"{self.label}.copy")
        self.copies.append(copied)
        return copied

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"released twice: {self.label}")
        self.release_count += 1


class _Bootstrap:
    calls = 0
    native_calls = 0
    setup_seconds = 0.0


@dataclass
class _Layer:
    index: int
    decode_impl: object | None = None
    forward_impl: object | None = None

    def forward_decode(self, hidden_states, **kwargs):  # noqa: ANN001, ANN003
        assert callable(self.decode_impl)
        return self.decode_impl(hidden_states, **kwargs)

    def forward(self, hidden_states, **kwargs):  # noqa: ANN001, ANN003
        assert callable(self.forward_impl)
        return self.forward_impl(hidden_states, **kwargs)


def _cache(tokens: int, label: str) -> PairFoldLayerCache:
    tokens = int(tokens)
    if tokens == 0:
        return PairFoldLayerCache()
    pages = []
    remaining = tokens
    page_index = 0
    while remaining:
        occupied = min(128, remaining)
        handles = tuple(
            _Owned(f"{label}.p{page_index}.{kind}")
            for kind in ("K0", "K1", "V0", "V1")
        )
        pages.append(
            PairFoldCipherPage(
                key_groups=handles[:2],
                value_groups=handles[2:],
                occupied=occupied,
            )
        )
        remaining -= occupied
        page_index += 1
    return PairFoldLayerCache(pages=tuple(pages), content_tokens=tokens)


def _advance(cache: PairFoldLayerCache) -> None:
    """Fake one committed append without invoking an FHE backend."""

    assert cache.pages
    assert cache.pages[-1].occupied < 128
    cache.pages[-1].occupied += 1
    cache.content_tokens += 1
    cache._validate_invariants()


def _result(
    output_ciphers: tuple[object, ...],
    *,
    pairfold_cache: PairFoldLayerCache | None = None,
    legacy_key: tuple[object, ...] = (),
    legacy_value: tuple[object, ...] = (),
) -> LayerResult:
    return LayerResult(
        ciphers=output_ciphers,
        wall_seconds=0.01,
        stage_seconds={},
        nested_stage_seconds={},
        softmax_bootstrap_calls=0,
        residual_bootstrap_calls=0,
        softmax_native_bootstrap_calls=0,
        residual_native_bootstrap_calls=0,
        output_levels=(),
        key_c_ciphers=legacy_key,
        value_c_ciphers=legacy_value,
        pairfold_kv_cache=pairfold_cache,
    )


def _decode_model(num_layers: int = 1) -> tuple[Llama3Model, np.ndarray]:
    feature = FeatureMajorPrefillLayout(seq_len=1)
    model = Llama3Model(
        config=Llama3CKKSConfig(),
        num_layers=num_layers,
        feature_layout=feature,
        attention_layout=AttentionPairLayout(),
    )
    return model, np.zeros((1, feature.hidden_dim), dtype=np.float32)


def _prefill_model(num_layers: int = 1) -> tuple[Llama3Model, np.ndarray]:
    feature = FeatureMajorPrefillLayout()
    model = Llama3Model(
        config=Llama3CKKSConfig(),
        num_layers=num_layers,
        feature_layout=feature,
        attention_layout=AttentionPairLayout(),
    )
    return model, np.zeros(
        (feature.seq_len, feature.hidden_dim), dtype=np.float32
    )


def _outputs(count: int, label: str) -> tuple[_Owned, ...]:
    return tuple(_Owned(f"{label}.{index}") for index in range(int(count)))


def test_transformer_layer_passes_pairfold_flags_to_both_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = FeatureMajorPrefillLayout(seq_len=2, hidden_dim=4, slots=8)
    layer = TransformerLayer(
        index=3,
        weights=object(),  # type: ignore[arg-type]
        approximations=object(),  # type: ignore[arg-type]
        config=Llama3CKKSConfig(),
        feature_layout=feature,
        attention_layout=AttentionPairLayout(),
    )
    cache = _cache(0, "pass")
    calls: dict[str, dict[str, object]] = {}
    prefill_sentinel = object()
    decode_sentinel = object()

    def fake_prefill(*_args, **kwargs):  # noqa: ANN002, ANN003
        calls["prefill"] = kwargs
        return prefill_sentinel

    def fake_decode(*_args, **kwargs):  # noqa: ANN002, ANN003
        calls["decode"] = kwargs
        return decode_sentinel

    monkeypatch.setattr(model_module.graph, "evaluate", fake_prefill)
    monkeypatch.setattr(model_module.graph, "evaluate_decode", fake_decode)
    hidden = np.zeros((2, 4), dtype=np.float32)

    assert (
        layer.forward(
            hidden,
            crypto_context=object(),
            input_ciphers=(_Owned("in"),),
            decode_packing="pairfold128",
            prefill_occupied=128,
        )
        is prefill_sentinel
    )
    assert (
        layer.forward_decode(
            hidden,
            crypto_context=object(),
            input_ciphers=(_Owned("decode.in"),),
            decode_packing="pairfold128",
            pairfold_kv_cache=cache,
        )
        is decode_sentinel
    )
    assert calls["prefill"]["decode_packing"] == "pairfold128"
    assert calls["prefill"]["prefill_occupied"] == 128
    assert calls["decode"]["decode_packing"] == "pairfold128"
    assert calls["decode"]["pairfold_kv_cache"] is cache
    cache.release()


@pytest.mark.parametrize("result_type", (ModelResult, DecodeModelResult))
def test_model_results_release_pairfold_field_exactly_once(result_type) -> None:  # noqa: ANN001
    output = _Owned("output")
    cache = _cache(1, "result")
    handles = cache.pages[0].handles
    common = {
        "ciphers": (output,),
        "kv_caches": [],
        "wall_seconds": 0.0,
        "per_layer": [],
        "pairfold_kv_caches": [cache],
    }
    if result_type is ModelResult:
        result = result_type(
            **common,
            bootstrap_setup_seconds=0.0,
            bootstrap_calls=0,
            native_bootstrap_calls=0,
        )
    else:
        result = result_type(**common, occupied=1, position=1)

    result.release()
    result.release()

    assert result.ciphers == ()
    assert result.kv_caches == []
    assert result.pairfold_kv_caches == []
    assert output.release_count == 1
    assert cache.is_released
    assert all(handle.release_count == 1 for handle in handles)


@pytest.mark.parametrize("case", ("wrong_family", "occupancy", "alias"))
def test_pairfold_decode_preflight_is_atomic(case: str) -> None:
    num_layers = 2 if case == "alias" else 1
    model, hidden = _decode_model(num_layers)
    input_cipher = _Owned("input")
    cache = _cache(1 if case != "occupancy" else 2, case)
    caches = [cache, cache] if case == "alias" else [cache]
    kwargs = {
        "crypto_context": object(),
        "input_ciphers": (input_cipher,),
        "layers": (),
        "bootstrap_operator": _Bootstrap(),
        "decode_packing": "pairfold128",
        "pairfold_kv_caches": caches,
        "occupied": 1,
        "position": 1,
    }
    if case == "wrong_family":
        kwargs["kv_caches"] = []
        match = "does not accept legacy"
    elif case == "occupancy":
        match = "occupancy mismatch"
    else:
        match = "must not alias"

    with pytest.raises((TypeError, ValueError), match=match):
        model.decode(hidden, **kwargs)

    assert caches[0] is cache
    assert not cache.is_released
    assert input_cipher.release_count == 0
    cache.release()


def test_pairfold_decode_rejects_cipher_handle_alias_across_layers() -> None:
    model, hidden = _decode_model(2)
    first = _cache(1, "first")
    second = _cache(1, "second")
    displaced = second.pages[0].key_groups[0]
    second.pages[0].key_groups = (
        first.pages[0].key_groups[0],
        second.pages[0].key_groups[1],
    )
    caches: list[PairFoldLayerCache | None] = [first, second]
    try:
        with pytest.raises(ValueError, match="handles must not alias across layers"):
            model.decode(
                hidden,
                crypto_context=object(),
                input_ciphers=(_Owned("input"),),
                layers=(),
                bootstrap_operator=_Bootstrap(),
                decode_packing="pairfold128",
                pairfold_kv_caches=caches,
                occupied=1,
                position=1,
            )
        assert not first.is_released
        assert not second.is_released
    finally:
        second.pages[0].key_groups = (
            displaced,
            second.pages[0].key_groups[1],
        )
        first.release()
        second.release()


def test_pairfold_decode_consume_moves_each_layer_slot_and_result_owns_caches() -> None:
    model, hidden = _decode_model(2)
    caches: list[PairFoldLayerCache | None] = [
        _cache(1, "layer0"),
        _cache(1, "layer1"),
    ]
    originals = tuple(caches)
    seen: list[int] = []

    def implementation(index: int):
        def run(_hidden, **kwargs):  # noqa: ANN001, ANN003
            cache = kwargs["pairfold_kv_cache"]
            assert caches[index] is None
            assert cache is originals[index]
            assert kwargs["decode_packing"] == "pairfold128"
            assert kwargs["consume_kv_cache"] is True
            _advance(cache)
            seen.append(index)
            return _result((_Owned(f"out{index}"),), pairfold_cache=cache)

        return run

    layers = tuple(
        _Layer(index=index, decode_impl=implementation(index))
        for index in range(2)
    )
    result = model.decode(
        hidden,
        crypto_context=object(),
        input_ciphers=(_Owned("input"),),
        layers=layers,
        bootstrap_operator=_Bootstrap(),
        decode_packing="pairfold128",
        pairfold_kv_caches=caches,
        occupied=1,
        position=1,
        consume_kv_cache=True,
    )

    assert seen == [0, 1]
    assert caches == [None, None]
    assert result.kv_caches == []
    assert tuple(result.pairfold_kv_caches) == originals
    assert all(cache.content_tokens == 2 for cache in result.pairfold_kv_caches)
    assert result.occupied == 2
    assert result.position == 2
    result.release()
    assert all(cache.is_released for cache in originals)


def test_pairfold_decode_pre_acquire_failure_restores_only_current_slot() -> None:
    model, hidden = _decode_model(2)
    first = _cache(1, "first")
    second = _cache(1, "second")
    caches: list[PairFoldLayerCache | None] = [first, second]

    def layer0(_hidden, **kwargs):  # noqa: ANN001, ANN003
        assert caches[0] is None
        cache = kwargs["pairfold_kv_cache"]
        _advance(cache)
        return _result((_Owned("layer0.out"),), pairfold_cache=cache)

    def layer1(_hidden, **kwargs):  # noqa: ANN001, ANN003
        assert caches[1] is None
        assert kwargs["pairfold_kv_cache"] is second
        raise RuntimeError("before graph ownership acquire")

    with pytest.raises(RuntimeError, match="before graph ownership"):
        model.decode(
            hidden,
            crypto_context=object(),
            input_ciphers=(_Owned("input"),),
            layers=(
                _Layer(0, decode_impl=layer0),
                _Layer(1, decode_impl=layer1),
            ),
            bootstrap_operator=_Bootstrap(),
            decode_packing="pairfold128",
            pairfold_kv_caches=caches,
            occupied=1,
            position=1,
            consume_kv_cache=True,
        )

    assert caches == [None, second]
    assert first.is_released
    assert not second.is_released
    assert second.content_tokens == 1
    second.release()


def test_pairfold_decode_post_acquire_failure_keeps_slot_empty() -> None:
    model, hidden = _decode_model()
    cache = _cache(1, "accepted")
    caches: list[PairFoldLayerCache | None] = [cache]

    def fail_after_acquire(_hidden, **kwargs):  # noqa: ANN001, ANN003
        assert caches[0] is None
        accepted = kwargs["pairfold_kv_cache"]
        accepted.release()
        raise RuntimeError("after graph ownership acquire")

    with pytest.raises(RuntimeError, match="after graph ownership"):
        model.decode(
            hidden,
            crypto_context=object(),
            input_ciphers=(_Owned("input"),),
            layers=(_Layer(0, decode_impl=fail_after_acquire),),
            decode_packing="pairfold128",
            pairfold_kv_caches=caches,
            occupied=1,
            position=1,
            consume_kv_cache=True,
        )

    assert caches == [None]
    assert cache.is_released


def test_pairfold_decode_non_consuming_returns_clone_and_preserves_input() -> None:
    model, hidden = _decode_model()
    original = _cache(1, "borrowed")
    caches: list[PairFoldLayerCache | None] = [original]

    def clone_then_append(_hidden, **kwargs):  # noqa: ANN001, ANN003
        borrowed = kwargs["pairfold_kv_cache"]
        clone = borrowed.clone()
        _advance(clone)
        return _result((_Owned("output"),), pairfold_cache=clone)

    result = model.decode(
        hidden,
        crypto_context=object(),
        input_ciphers=(_Owned("input"),),
        layers=(_Layer(0, decode_impl=clone_then_append),),
        decode_packing="pairfold128",
        pairfold_kv_caches=caches,
        occupied=1,
        position=1,
        consume_kv_cache=False,
    )

    returned = result.pairfold_kv_caches[0]
    assert caches == [original]
    assert returned is not original
    assert original.content_tokens == 1
    assert returned.content_tokens == 2
    result.release()
    assert returned.is_released
    assert not original.is_released
    original.release()


def test_pairfold_prefill_model_collects_only_pairfold_result_field() -> None:
    model, hidden = _prefill_model()
    cache = _cache(128, "prefill")
    observed: dict[str, object] = {}

    def forward(_hidden, **kwargs):  # noqa: ANN001, ANN003
        observed.update(kwargs)
        return _result(
            _outputs(model.feature_layout.cipher_count, "prefill.out"),
            pairfold_cache=cache,
        )

    result = model.forward(
        hidden,
        crypto_context=object(),
        input_ciphers=_outputs(model.feature_layout.cipher_count, "input"),
        layers=(_Layer(0, forward_impl=forward),),
        bootstrap_operator=_Bootstrap(),
        emit_kv_cache=True,
        decode_packing="pairfold128",
        prefill_occupied=128,
    )

    assert observed["decode_packing"] == "pairfold128"
    assert observed["prefill_occupied"] == 128
    assert result.kv_caches == []
    assert result.pairfold_kv_caches == [cache]
    result.release()
    assert cache.is_released


@pytest.mark.parametrize("invalid_kind", ("partial_occupied", "padded_mask"))
def test_pairfold_prefill_rejects_partial_page_before_running_layers(
    invalid_kind: str,
) -> None:
    model, hidden = _prefill_model()
    calls = 0

    def forbidden(_hidden, **_kwargs):  # noqa: ANN001, ANN003
        nonlocal calls
        calls += 1
        raise AssertionError("PairFold partial-page preflight ran a layer")

    kwargs: dict[str, object] = {}
    if invalid_kind == "partial_occupied":
        kwargs["prefill_occupied"] = 127
    else:
        mask = np.ones(model.feature_layout.seq_len, dtype=bool)
        mask[-1] = False
        kwargs["attention_token_valid_mask"] = mask

    with pytest.raises(ValueError, match="right-padded|requires valid tokens"):
        model.forward(
            hidden,
            crypto_context=object(),
            input_ciphers=_outputs(model.feature_layout.cipher_count, "input"),
            layers=(_Layer(0, forward_impl=forbidden),),
            bootstrap_operator=_Bootstrap(),
            emit_kv_cache=True,
            decode_packing="pairfold128",
            **kwargs,
        )

    assert calls == 0


def test_pairfold_prefill_later_failure_releases_earlier_cache() -> None:
    model, hidden = _prefill_model(2)
    cache = _cache(128, "prefill.first")

    def first(_hidden, **_kwargs):  # noqa: ANN001, ANN003
        return _result(
            _outputs(model.feature_layout.cipher_count, "first.out"),
            pairfold_cache=cache,
        )

    def second(_hidden, **_kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("second Prefill layer failed")

    with pytest.raises(RuntimeError, match="second Prefill"):
        model.forward(
            hidden,
            crypto_context=object(),
            input_ciphers=_outputs(model.feature_layout.cipher_count, "input"),
            layers=(
                _Layer(0, forward_impl=first),
                _Layer(1, forward_impl=second),
            ),
            bootstrap_operator=_Bootstrap(),
            emit_kv_cache=True,
            decode_packing="pairfold128",
        )

    assert cache.is_released
