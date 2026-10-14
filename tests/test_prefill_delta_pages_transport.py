"""Ownership and routing tests for encrypted Prefill history pages."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.graph import LayerResult
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.model import Llama3Model
from llama3fhe.operators.attention.prefill import multi_page, operator
from llama3fhe.operators.attention.prefill.input import (
    ComplexTokenAttentionResult,
    DeltaKVPage,
)


class _Cipher:
    def __init__(self, label: str, value: float = 0.0) -> None:
        self.label = label
        self.value = value
        self.release_count = 0
        self.state = SimpleNamespace(
            cur_limbs=8, scale_degree=1, scaling_factor=float(2**40)
        )

    def deep_copy(self):
        return _Cipher(f"{self.label}.copy", self.value)

    def release(self) -> None:
        self.release_count += 1
        assert self.release_count == 1, f"double release: {self.label}"


def _layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4, head_dim=4, query_heads=4, key_value_heads=2, slots=64
    )


def _context():
    fhe = SimpleNamespace(
        homo_add=lambda left, right, _context: _Cipher(
            "combined", left.value + right.value
        )
    )
    return SimpleNamespace(
        device="cpu",
        fhe=fhe,
        context=object(),
        level_for_cipher=lambda _cipher: 0,
    )


def test_snapshot_delta_page_is_independent_of_attention_inputs() -> None:
    key = _Cipher("key", 2.0)
    value = _Cipher("value", 3.0)
    owner = ComplexTokenAttentionResult(
        query_ciphers=(_Cipher("query"),),
        key_ciphers=(key,),
        value_ciphers=(value,),
        wall_seconds=0.0,
        stage_seconds={},
        scheduler_stats={},
    )
    mask = np.array([True, True, False, False])
    page = owner.snapshot_delta_page(token_valid_mask=mask)
    mask[:] = False
    owner.release()
    assert page.key_ciphers[0] is not key
    assert page.value_ciphers[0] is not value
    np.testing.assert_array_equal(
        page.token_valid_mask, [True, True, False, False]
    )
    assert page.key_ciphers[0].release_count == 0
    assert page.value_ciphers[0].release_count == 0
    page.release()
    page.release()
    assert key.release_count == value.release_count == 1


@pytest.mark.parametrize("fail_at", [None, "after_qk", "after_pv"])
def test_adapter_moves_current_page_at_last_use_without_copy(
    monkeypatch: pytest.MonkeyPatch, fail_at: str | None
) -> None:
    class NoCopyCipher(_Cipher):
        def deep_copy(self):
            raise AssertionError("Delta page emission must move ciphertexts")

    query = NoCopyCipher("query")
    key = NoCopyCipher("key")
    value = NoCopyCipher("value")
    carrier = _Cipher("carrier")
    core_output = _Cipher("core_output")
    w_input = _Cipher("w_input")
    adapter = ComplexTokenAttentionResult(
        query_ciphers=(query,),
        key_ciphers=(key,),
        value_ciphers=(value,),
        wall_seconds=0.0,
        stage_seconds={},
        scheduler_stats={},
    )
    mask = np.array([True, True, False, False])

    def fake_core(_query, _key, _value, **kwargs):
        assert _query == (query,) and _key == (key,) and _value == (value,)
        kwargs["on_qk_last_use"]()
        if fail_at == "after_qk":
            raise RuntimeError("after_qk")
        kwargs["on_value_last_use"]()
        if fail_at == "after_pv":
            raise RuntimeError("after_pv")
        return operator.DeltaAttentionAlg2FHEResult(
            output_ciphers=(core_output,), stage_seconds={}, level_trace={}
        )

    class FakeOutput:
        real_ciphers = ()
        carrier_ciphers = (w_input,)
        stage_seconds = {}

        def release(self):
            assert self.carrier_ciphers == ()

    monkeypatch.setattr(operator, "prepare_attention_inputs_fhe", lambda *_a, **_k: adapter)
    monkeypatch.setattr(operator, "attention_fhe_delta_alg2", fake_core)
    monkeypatch.setattr(operator, "prepare_w_o_inputs_fhe", lambda *_a, **_k: FakeOutput())
    monkeypatch.setattr(operator, "sync_device", lambda _device: None)
    monkeypatch.setattr(operator, "synchronize_device", lambda _device: None)

    result = None
    try:
        if fail_at is None:
            result = operator.attention_fhe_from_qkv_carriers(
                (carrier,),
                carrier_layout=object(),
                feature_layout=object(),
                attention_layout=_layout(),
                crypto_context=_context(),
                softmax_config=object(),
                emit_delta_kv_page=True,
                token_valid_mask=mask,
            )
            page = result.delta_kv_page
            assert page is not None
            assert page.key_ciphers == (key,)
            assert page.value_ciphers == (value,)
            mask[:] = False
            np.testing.assert_array_equal(page.token_valid_mask, [True, True, False, False])
            assert key.release_count == value.release_count == 0
        else:
            with pytest.raises(RuntimeError, match=fail_at):
                operator.attention_fhe_from_qkv_carriers(
                    (carrier,),
                    carrier_layout=object(),
                    feature_layout=object(),
                    attention_layout=_layout(),
                    crypto_context=_context(),
                    softmax_config=object(),
                    emit_delta_kv_page=True,
                    token_valid_mask=mask,
                )
            assert key.release_count == value.release_count == 1
    finally:
        if result is not None:
            result.release()
        carrier.release()
    assert query.release_count == 1
    assert key.release_count == value.release_count == 1


@pytest.mark.parametrize("fail_history_pv", [False, True])
def test_history_uses_shared_softmax_and_keeps_borrowed_page(
    monkeypatch: pytest.MonkeyPatch, fail_history_pv: bool
) -> None:
    layout = _layout()
    query = _Cipher("query")
    key = _Cipher("current_key")
    value = _Cipher("current_value")
    history_key = _Cipher("history_key")
    history_value = _Cipher("history_value")
    history = DeltaKVPage(
        key_ciphers=(history_key,),
        value_ciphers=(history_value,),
        token_valid_mask=np.array([True, True, False, False]),
    )
    current_score = _Cipher("current_score")
    history_score = _Cipher("history_score")
    current_probability = _Cipher("current_probability")
    history_probability = _Cipher("history_probability")
    current_output = _Cipher("current_output", 7.0)
    history_output = _Cipher("history_output", 5.0)
    seen: list[str] = []

    def fake_qk(_query, keys, **_kwargs):
        assert _query == (query,)
        if keys == (key,):
            return (current_score,)
        assert keys == (history_key,)
        return (history_score,)

    def fake_softmax(score_pages, masks, **_kwargs):
        assert score_pages == ((history_score,), (current_score,))
        expected = multi_page.prefill_delta_page_masks(
            layout,
            historical_key_valid_masks=(history.token_valid_mask,),
            current_valid_mask=np.array([True, True, True, False]),
        )
        for actual, reference in zip(masks, expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
        seen.append("softmax")
        return multi_page.DeltaSoftmaxAlg2PagesResult(
            probability_pages=(
                (history_probability,),
                (current_probability,),
            ),
            stage_seconds={},
            output_levels=(0,),
            slim_period_rows=None,
            slim_sparse_bootstraps=0,
        )

    def fake_pv(probabilities, values, **_kwargs):
        if probabilities == (current_probability,):
            assert values == (value,)
            return (current_output,)
        assert probabilities == (history_probability,)
        assert values == (history_value,)
        if fail_history_pv:
            raise RuntimeError("history PV failed")
        return (history_output,)

    monkeypatch.setattr(operator, "qk_delta_fhe", fake_qk)
    monkeypatch.setattr(multi_page, "softmax_alg2_fhe_delta_pages", fake_softmax)
    monkeypatch.setattr(operator, "pv_delta_fhe", fake_pv)
    monkeypatch.setattr(operator, "sync_device", lambda _device: None)
    monkeypatch.setattr(operator, "synchronize_device", lambda _device: None)

    result = None
    try:
        if fail_history_pv:
            with pytest.raises(RuntimeError, match="history PV failed"):
                operator.attention_fhe_delta_alg2(
                    (query,), (key,), (value,),
                    layout=layout,
                    crypto_context=_context(),
                    softmax_config=object(),
                    historical_delta_pages=(history,),
                    token_valid_mask=np.array([True, True, True, False]),
                )
        else:
            result = operator.attention_fhe_delta_alg2(
                (query,), (key,), (value,),
                layout=layout,
                crypto_context=_context(),
                softmax_config=object(),
                historical_delta_pages=(history,),
                token_valid_mask=np.array([True, True, True, False]),
            )
            assert result.output_ciphers[0].value == 12.0
            result.release()
            assert current_output.release_count == 1
            assert history_output.release_count == 1
        assert seen == ["softmax"]
        assert current_score.release_count == history_score.release_count == 1
        assert current_probability.release_count == history_probability.release_count == 1
        assert history_key.release_count == history_value.release_count == 0
    finally:
        history.release()
        query.release()
        key.release()
        value.release()


def test_prefix_and_history_are_rejected_before_qk() -> None:
    history = DeltaKVPage(key_ciphers=(_Cipher("key"),), value_ciphers=(_Cipher("value"),))
    try:
        with pytest.raises(ValueError, match="not yet supported together"):
            operator.attention_fhe_delta_alg2(
                (_Cipher("query"),), (_Cipher("current_key"),),
                (_Cipher("current_value"),),
                layout=_layout(),
                crypto_context=_context(),
                softmax_config=object(),
                historical_delta_pages=(history,),
                prefix_bundle=SimpleNamespace(prefix_k=1),
            )
    finally:
        history.release()


def _model() -> tuple[Llama3Model, np.ndarray]:
    feature = FeatureMajorPrefillLayout()
    return (
        Llama3Model(
            config=Llama3CKKSConfig(),
            num_layers=2,
            feature_layout=feature,
            attention_layout=AttentionPairLayout(),
        ),
        np.zeros((feature.seq_len, feature.hidden_dim), dtype=np.float32),
    )


def _page(label: str, mask: np.ndarray | None = None) -> DeltaKVPage:
    return DeltaKVPage(
        key_ciphers=(_Cipher(f"{label}.key"),),
        value_ciphers=(_Cipher(f"{label}.value"),),
        token_valid_mask=mask,
    )


def _layer_result(outputs: tuple[_Cipher, ...], page: DeltaKVPage) -> LayerResult:
    return LayerResult(
        ciphers=outputs,
        wall_seconds=0.0,
        stage_seconds={},
        nested_stage_seconds={},
        softmax_bootstrap_calls=0,
        residual_bootstrap_calls=0,
        softmax_native_bootstrap_calls=0,
        residual_native_bootstrap_calls=0,
        output_levels=(0,),
        delta_kv_page=page,
    )


def test_model_routes_each_history_layer_and_transfers_emitted_pages() -> None:
    model, hidden = _model()
    inputs = tuple(_Cipher(f"input.{index}") for index in range(model.feature_layout.cipher_count))
    histories = (_page("history.0"), _page("history.1"))
    emitted = (_page("emitted.0"), _page("emitted.1"))
    seen: list[int] = []

    def make_layer(index: int):
        def forward(_hidden, **kwargs):
            assert kwargs["historical_delta_pages"] == (histories[index],)
            assert kwargs["emit_delta_kv_page"] is True
            assert kwargs["content_offset"] == 128
            seen.append(index)
            outputs = tuple(
                _Cipher(f"output.{index}.{i}")
                for i in range(model.feature_layout.cipher_count)
            )
            return _layer_result(outputs, emitted[index])

        return SimpleNamespace(index=index, forward=forward)

    result = model.forward(
        hidden,
        crypto_context=object(),
        input_ciphers=inputs,
        layers=(make_layer(0), make_layer(1)),
        bootstrap_operator=SimpleNamespace(calls=0, native_calls=0),
        historical_delta_pages=((histories[0],), (histories[1],)),
        emit_delta_kv_page=True,
        content_offset=128,
    )
    try:
        assert seen == [0, 1]
        assert result.delta_kv_pages == list(emitted)
        assert all(cipher.release_count == 0 for page in histories for cipher in page.key_ciphers + page.value_ciphers)
    finally:
        result.release()
        for page in histories:
            page.release()
        for cipher in inputs:
            cipher.release()
    assert all(not page.key_ciphers and not page.value_ciphers for page in emitted)


@pytest.mark.parametrize("invalid", ["offset", "mask", "layer_count"])
def test_model_rejects_invalid_history_before_layer_work(invalid: str) -> None:
    model, hidden = _model()
    inputs = tuple(_Cipher(f"input.{index}") for index in range(model.feature_layout.cipher_count))
    mask = np.ones(128, dtype=bool)
    if invalid == "mask":
        mask = np.ones(127, dtype=bool)
    histories = (_page("history.0", mask), _page("history.1"))
    per_layer = ((histories[0],), (histories[1],))
    offset = 127 if invalid == "offset" else 128
    if invalid == "layer_count":
        per_layer = per_layer[:1]
    match = {
        "offset": "content_offset must equal",
        "mask": "historical Delta page mask must have shape",
        "layer_count": "one page sequence per layer",
    }[invalid]
    try:
        with pytest.raises(ValueError, match=match):
            model.forward(
                hidden,
                crypto_context=object(),
                input_ciphers=inputs,
                layers=(),
                bootstrap_operator=SimpleNamespace(calls=0, native_calls=0),
                historical_delta_pages=per_layer,
                content_offset=offset,
            )
        assert all(cipher.release_count == 0 for cipher in inputs)
    finally:
        for page in histories:
            page.release()
        for cipher in inputs:
            cipher.release()
