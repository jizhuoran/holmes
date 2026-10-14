from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.pairfold import pairfold_operator as operator


class _FakeState:
    def __init__(self, cur_limbs: int = 8) -> None:
        self.cur_limbs = int(cur_limbs)


class _FakeCipher:
    def __init__(self, label: str, *, cur_limbs: int = 8) -> None:
        self.label = str(label)
        self.state = _FakeState(cur_limbs)
        self.release_count = 0

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"cipher released twice: {self.label}")
        self.release_count = 1

    def __repr__(self) -> str:
        return f"_FakeCipher({self.label!r})"


class _Harness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        fail_stage: str | None = None,
        refresh: bool = False,
        softmax_alias_score: bool = False,
        refresh_alias_borrowed: bool = False,
    ) -> None:
        self.fail_stage = fail_stage
        self.refresh = bool(refresh)
        self.softmax_alias_score = bool(softmax_alias_score)
        self.refresh_alias_borrowed = bool(refresh_alias_borrowed)
        self.generated: list[_FakeCipher] = []
        self.calls: Counter[str] = Counter()
        self.arguments: dict[str, object] = {}
        self.borrowed_alias: _FakeCipher | None = None
        monkeypatch.setattr(operator, "qk_pairfold_pages_fhe", self.qk)
        monkeypatch.setattr(
            operator._decode_softmax,
            "softmax_alg2_row_pages_fhe",
            self.softmax,
        )
        monkeypatch.setattr(
            operator._decode_softmax,
            "_refresh_row_ciphers_if_short",
            self.refresh_if_short,
        )
        monkeypatch.setattr(operator, "pv_pairfold_pages_fhe", self.pv)

    def cipher(self, label: str) -> _FakeCipher:
        cipher = _FakeCipher(label)
        self.generated.append(cipher)
        return cipher

    def qk(
        self,
        carrier_cipher,
        key_pages,
        *,
        query_extractor,
        layout,
        crypto_context,
        occupied_pages,
        key_prefix,
        prefix_k,
        profile,
    ):
        del layout, crypto_context
        self.calls["qk"] += 1
        self.arguments["qk"] = {
            "carrier": carrier_cipher,
            "key_pages": key_pages,
            "query_extractor": query_extractor,
            "occupied_pages": occupied_pages,
            "key_prefix": key_prefix,
            "prefix_k": prefix_k,
        }
        if self.fail_stage == "qk":
            raise RuntimeError("injected qk failure")
        profile.update(
            {
                "query_extract": 0.01,
                "query_broadcast": 0.02,
                "prefix_compute": 0.03 if prefix_k else 0.0,
                "content_compute": 0.04 * len(key_pages),
            }
        )
        count = len(key_pages) + int(prefix_k is not None)
        return tuple((self.cipher(f"score.{index}"),) for index in range(count))

    def softmax(
        self,
        score_pages,
        *,
        layout,
        crypto_context,
        config,
        occupied_pages,
        layer_idx,
        rounds,
        score_scale,
        bootstrap_operator,
        bootstrap_iterations,
        bootstrap_precision_bits,
    ):
        del layout, crypto_context, config
        self.calls["softmax"] += 1
        self.arguments["softmax"] = {
            "score_pages": score_pages,
            "occupied_pages": occupied_pages,
            "layer_idx": layer_idx,
            "rounds": rounds,
            "score_scale": score_scale,
            "bootstrap_operator": bootstrap_operator,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_precision_bits": bootstrap_precision_bits,
        }
        if self.fail_stage == "softmax":
            raise RuntimeError("injected softmax failure")
        if self.softmax_alias_score:
            return (score_pages[0],) + tuple(
                (self.cipher(f"probability.{index}"),)
                for index in range(1, len(score_pages))
            )
        return tuple(
            (self.cipher(f"probability.{index}"),) for index in range(len(score_pages))
        )

    def refresh_if_short(
        self,
        page,
        *,
        bootstrap_operator,
        need_limbs,
        iterations,
        precision_bits,
    ):
        call = self.calls["refresh"]
        self.calls["refresh"] += 1
        self.arguments[f"refresh.{call}"] = {
            "bootstrap_operator": bootstrap_operator,
            "need_limbs": need_limbs,
            "iterations": iterations,
            "precision_bits": precision_bits,
        }
        if self.fail_stage == "refresh" and call == 1:
            raise RuntimeError("injected refresh failure")
        if self.refresh_alias_borrowed and call == 0:
            assert self.borrowed_alias is not None
            for cipher in page:
                cipher.release()
            return (self.borrowed_alias,)
        if not self.refresh:
            return tuple(page)
        output = self.cipher(f"refreshed.{call}")
        for cipher in page:
            cipher.release()
        return (output,)

    def pv(
        self,
        probability_pages,
        value_pages,
        *,
        layout,
        crypto_context,
        occupied_pages,
        prefix_probability_ciphers,
        value_prefix,
        prefix_k,
        profile,
    ):
        del layout, crypto_context
        self.calls["pv"] += 1
        self.arguments["pv"] = {
            "probability_pages": probability_pages,
            "value_pages": value_pages,
            "occupied_pages": occupied_pages,
            "prefix_probability_ciphers": prefix_probability_ciphers,
            "value_prefix": value_prefix,
            "prefix_k": prefix_k,
        }
        if self.fail_stage == "pv":
            raise RuntimeError("injected pv failure")
        profile.update(
            {
                "prefix_compute": 0.05 if prefix_k else 0.0,
                "content_compute": 0.06 * len(value_pages),
                "output_route": 0.07,
                "folded_o_accumulate": 0.08,
            }
        )
        return (self.cipher("folded_o"),)


def _borrowed_pages(count: int = 2):
    carrier = _FakeCipher("carrier")
    key_pages = tuple(
        (_FakeCipher(f"k{page}.0"), _FakeCipher(f"k{page}.1")) for page in range(count)
    )
    value_pages = tuple(
        (_FakeCipher(f"v{page}.0"), _FakeCipher(f"v{page}.1")) for page in range(count)
    )
    borrowed = (carrier,) + tuple(
        cipher for page in key_pages + value_pages for cipher in page
    )
    return carrier, key_pages, value_pages, borrowed


def _run(
    harness: _Harness,
    *,
    prefix_bundle=None,
    bootstrap_operator=None,
):
    carrier, key_pages, value_pages, borrowed = _borrowed_pages()
    harness.borrowed_alias = carrier
    query_extractor = object()
    # The fake QK does not invoke it, but the orchestrator requires the real
    # production callback contract to be callable.
    query_extractor = lambda *_args, **_kwargs: None
    result = operator.pairfold_attention_fhe(
        carrier,
        key_pages,
        value_pages,
        query_extractor=query_extractor,
        pairfold_layout=PairFold128Layout(),
        score_layout=AttentionPairLayout(),
        crypto_context=SimpleNamespace(),
        softmax_config=object(),
        occupied_pages=(128, 7),
        layer_idx=8,
        rounds=3,
        score_scale=0.125,
        bootstrap_operator=bootstrap_operator,
        bootstrap_iterations=2,
        bootstrap_precision_bits=17,
        prefix_bundle=prefix_bundle,
    )
    return result, borrowed, query_extractor


def _assert_borrowed_live(borrowed) -> None:
    assert all(cipher.release_count == 0 for cipher in borrowed)


def _assert_generated_released_once(
    harness: _Harness, *, except_for: tuple[_FakeCipher, ...] = ()
) -> None:
    exceptions = {id(cipher) for cipher in except_for}
    for cipher in harness.generated:
        expected = 0 if id(cipher) in exceptions else 1
        assert cipher.release_count == expected, cipher.label


def test_pairfold_attention_no_prefix_success_and_idempotent_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)

    result, borrowed, query_extractor = _run(harness)

    output = result.folded_o_cipher
    assert output is not None
    assert result.output_ciphers == (output,)
    assert result.score_occupied == (128, 7)
    assert harness.arguments["qk"]["query_extractor"] is query_extractor
    assert harness.arguments["qk"]["key_prefix"] is None
    assert harness.arguments["softmax"]["occupied_pages"] == (128, 7)
    assert harness.arguments["pv"]["prefix_probability_ciphers"] is None
    assert result.stage_seconds["q_extract_and_broadcast"] == pytest.approx(0.03)
    assert result.stage_seconds["qk_prefix"] == 0.0
    assert result.stage_seconds["qk_content"] == pytest.approx(0.08)
    assert result.stage_seconds["pv_prefix"] == 0.0
    assert result.stage_seconds["pv_content"] == pytest.approx(0.12)
    assert result.stage_seconds["folded_o_assemble"] == pytest.approx(0.15)
    assert "qk.prefix_compute" in result.profile
    assert "pv.folded_o_accumulate" in result.profile
    _assert_borrowed_live(borrowed)
    _assert_generated_released_once(harness, except_for=(output,))

    result.release()
    result.release()
    assert result.folded_o_cipher is None
    assert result.output_ciphers == ()
    assert output.release_count == 1
    _assert_borrowed_live(borrowed)


def test_pairfold_attention_prefix_order_bootstrap_forwarding_and_softmax_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, refresh=True)
    bootstrap = SimpleNamespace(name="bootstrap")
    key = np.zeros((3, 8, 128), dtype=np.float32)
    value = np.ones((3, 8, 128), dtype=np.float32)
    prefix_bundle = SimpleNamespace(
        prefix_k=3,
        key=key,
        value=value,
    )

    result, borrowed, _ = _run(
        harness,
        prefix_bundle=prefix_bundle,
        bootstrap_operator=bootstrap,
    )

    assert harness.calls["softmax"] == 1
    assert result.score_occupied == (3, 128, 7)
    assert harness.arguments["qk"]["key_prefix"] is key
    assert harness.arguments["qk"]["prefix_k"] == 3
    softmax_args = harness.arguments["softmax"]
    assert softmax_args["occupied_pages"] == (3, 128, 7)
    assert softmax_args["bootstrap_operator"] is bootstrap
    assert softmax_args["bootstrap_iterations"] == 2
    assert softmax_args["bootstrap_precision_bits"] == 17
    assert harness.calls["refresh"] == 3
    for page_index in range(3):
        refresh_args = harness.arguments[f"refresh.{page_index}"]
        assert refresh_args == {
            "bootstrap_operator": bootstrap,
            "need_limbs": 6,
            "iterations": 2,
            "precision_bits": 17,
        }
    pv_args = harness.arguments["pv"]
    prefix_probability = pv_args["prefix_probability_ciphers"]
    content_probabilities = pv_args["probability_pages"]
    assert len(prefix_probability) == 1
    assert len(content_probabilities) == 2
    assert prefix_probability[0].label == "refreshed.0"
    assert [page[0].label for page in content_probabilities] == [
        "refreshed.1",
        "refreshed.2",
    ]
    assert pv_args["value_prefix"] is value
    assert pv_args["prefix_k"] == 3
    assert result.stage_seconds["qk_prefix"] == pytest.approx(0.03)
    assert result.stage_seconds["pv_prefix"] == pytest.approx(0.05)
    _assert_borrowed_live(borrowed)
    output = result.folded_o_cipher
    assert output is not None
    _assert_generated_released_once(harness, except_for=(output,))
    result.release()


@pytest.mark.parametrize("fail_stage", ("qk", "softmax", "refresh", "pv"))
def test_pairfold_attention_stage_failure_cleans_owned_and_borrows_inputs(
    monkeypatch: pytest.MonkeyPatch,
    fail_stage: str,
) -> None:
    harness = _Harness(
        monkeypatch,
        fail_stage=fail_stage,
        refresh=True,
    )
    carrier, key_pages, value_pages, borrowed = _borrowed_pages()
    harness.borrowed_alias = carrier

    with pytest.raises(RuntimeError, match=f"injected {fail_stage} failure"):
        operator.pairfold_attention_fhe(
            carrier,
            key_pages,
            value_pages,
            query_extractor=lambda *_args, **_kwargs: None,
            pairfold_layout=PairFold128Layout(),
            score_layout=AttentionPairLayout(),
            crypto_context=SimpleNamespace(),
            softmax_config=object(),
            occupied_pages=(128, 7),
            bootstrap_operator=object(),
        )

    _assert_borrowed_live(borrowed)
    _assert_generated_released_once(harness)


def test_softmax_score_alias_is_rejected_without_double_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, softmax_alias_score=True)
    carrier, key_pages, value_pages, borrowed = _borrowed_pages()

    with pytest.raises(RuntimeError, match="forbidden ciphertext"):
        operator.pairfold_attention_fhe(
            carrier,
            key_pages,
            value_pages,
            query_extractor=lambda *_args, **_kwargs: None,
            pairfold_layout=PairFold128Layout(),
            score_layout=AttentionPairLayout(),
            crypto_context=SimpleNamespace(),
            softmax_config=object(),
            occupied_pages=(128, 7),
        )

    _assert_borrowed_live(borrowed)
    _assert_generated_released_once(harness)


def test_refresh_borrowed_alias_is_rejected_without_consuming_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(
        monkeypatch,
        refresh_alias_borrowed=True,
    )
    carrier, key_pages, value_pages, borrowed = _borrowed_pages()
    harness.borrowed_alias = carrier

    with pytest.raises(RuntimeError, match="forbidden ciphertext"):
        operator.pairfold_attention_fhe(
            carrier,
            key_pages,
            value_pages,
            query_extractor=lambda *_args, **_kwargs: None,
            pairfold_layout=PairFold128Layout(),
            score_layout=AttentionPairLayout(),
            crypto_context=SimpleNamespace(),
            softmax_config=object(),
            occupied_pages=(128, 7),
        )

    _assert_borrowed_live(borrowed)
    _assert_generated_released_once(harness)


def test_zero_occupied_content_page_is_rejected_before_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    carrier, key_pages, value_pages, borrowed = _borrowed_pages(count=1)

    with pytest.raises(ValueError, match=r"outside \[1, 128\]"):
        operator.pairfold_attention_fhe(
            carrier,
            key_pages,
            value_pages,
            query_extractor=lambda *_args, **_kwargs: None,
            pairfold_layout=PairFold128Layout(),
            score_layout=AttentionPairLayout(),
            crypto_context=SimpleNamespace(),
            softmax_config=object(),
            occupied_pages=(0,),
        )

    assert harness.calls == Counter()
    _assert_borrowed_live(borrowed)
