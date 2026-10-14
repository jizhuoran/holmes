from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.application.inputs import (
    _load_prompt_ids,
    split_prefill_pages,
    split_variable_prefill_ids,
)
from llama3fhe.approx import layer as poly_layer
from llama3fhe.approx.attention import apply_rope_heads, rope_cos_sin
from llama3fhe.positions import (
    content_rope_start_pos,
    validate_prefill_cache_page,
)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 127, 128])
def test_left_padding_keeps_prefix_relative_positions(length: int) -> None:
    valid = np.zeros(128, dtype=bool)
    valid[-length:] = True
    start = content_rope_start_pos(
        prefix_k=2, token_valid_mask=valid, seq_len=128
    )
    assert start + np.flatnonzero(valid)[0] == 2
    assert start + np.flatnonzero(valid)[-1] == 2 + length - 1


@pytest.mark.parametrize("length", [1, 7, 127])
def test_left_padded_rope_matches_unpadded_content(length: int) -> None:
    rng = np.random.default_rng(length)
    values = rng.normal(size=(length, 128)).astype(np.float32)
    padded = np.zeros((128, 128), dtype=np.float32)
    padded[-length:] = values
    valid = np.arange(128) >= 128 - length
    start = content_rope_start_pos(
        prefix_k=2, token_valid_mask=valid, seq_len=128
    )
    cos, sin = rope_cos_sin(seq_len=128, head_dim=128, start_pos=start)
    got = apply_rope_heads(
        padded, cos, sin, num_heads=1, head_dim=128
    )[-length:]
    cos_ref, sin_ref = rope_cos_sin(
        seq_len=length, head_dim=128, start_pos=2
    )
    expected = apply_rope_heads(
        values, cos_ref, sin_ref, num_heads=1, head_dim=128
    )
    np.testing.assert_allclose(got, expected, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("length", [1, 127, 128])
def test_partial_cache_accepts_only_right_padding(length: int) -> None:
    valid = np.arange(128) < length
    assert validate_prefill_cache_page(
        occupied=length, token_valid_mask=valid, seq_len=128
    ) == length
    if length < 128:
        with pytest.raises(ValueError, match="requires valid tokens"):
            validate_prefill_cache_page(
                occupied=length,
                token_valid_mask=valid[::-1],
                seq_len=128,
            )


@pytest.mark.parametrize("length", [1, 7, 8, 9, 127, 128, 129, 130, 256])
def test_variable_prefill_keeps_every_input_token(length: int) -> None:
    ids = np.arange(length, dtype=np.int64) + 11
    page, valid, occupied, tail = split_variable_prefill_ids(ids)
    assert page.shape == (128,)
    assert occupied == min(length, 128)
    assert valid.sum() == occupied
    np.testing.assert_array_equal(page[:occupied], ids[:occupied])
    np.testing.assert_array_equal(tail, ids[occupied:])
    np.testing.assert_array_equal(page[occupied:], ids[occupied - 1])


@pytest.mark.parametrize("length", [1, 127, 128, 129, 130, 255, 256, 257])
def test_prefill_pages_preserve_order_and_padding(length: int) -> None:
    ids = np.arange(length, dtype=np.int64) + 11
    pages = split_prefill_pages(ids)
    assert len(pages) == (length + 127) // 128
    assert sum(occupied for _, _, occupied in pages) == length
    np.testing.assert_array_equal(
        np.concatenate([page[:occupied] for page, _, occupied in pages]), ids
    )
    for index, (page, valid, occupied) in enumerate(pages):
        assert page.shape == valid.shape == (128,)
        assert valid.tolist() == [True] * occupied + [False] * (128 - occupied)
        np.testing.assert_array_equal(page[occupied:], ids[index * 128 + occupied - 1])


def test_prefill_pages_reject_empty_input_and_bad_width() -> None:
    with pytest.raises(ValueError, match="at least one"):
        split_prefill_pages(np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="page_size must be positive"):
        split_prefill_pages(np.array([1]), page_size=0)


def test_single_page_prompt_rejects_long_input_without_truncating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(
        encode=lambda *_args, **_kwargs: list(range(129)),
        pad_token_id=0,
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
    with pytest.raises(ValueError, match="129 tokens.*128-token"):
        _load_prompt_ids(
            "long prompt",
            model_dir="unused",
            metadata=SimpleNamespace(pad_token_id=0),
            seq_len=128,
            pad_side="left",
            add_special_tokens=False,
        )


def test_poly_prefill_stack_passes_valid_mask_to_every_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    valid = np.array([True, True, False, False])

    def fake_layer(hidden, **kwargs):
        seen.append(kwargs["token_valid_mask"])
        return SimpleNamespace(
            hidden=np.asarray(hidden) + 1,
            key=np.zeros((4, 2), dtype=np.float32),
            value=np.zeros((4, 2), dtype=np.float32),
        )

    monkeypatch.setattr(poly_layer, "poly_prefill_layer_numpy", fake_layer)
    output, keys, values = poly_layer.poly_prefill_stack_numpy(
        np.zeros((4, 2), dtype=np.float32),
        [object(), object()],
        approx_config=SimpleNamespace(for_layer=lambda _index: object()),
        attention_layout=object(),
        token_valid_mask=valid,
    )
    assert seen == [valid, valid]
    assert np.all(output == 2)
    assert keys.shape == values.shape == (2, 4, 2)
