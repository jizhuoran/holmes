from __future__ import annotations

import pytest

from llama3fhe.operators.kv_cache import (
    copy_c_pages,
    extend_c_pages,
    extend_kv_c_pages,
    flatten_c_pages,
    split_c_pages,
)


class _FakeCipher:
    def __init__(self, value: int, *, label: str = "") -> None:
        self.value = int(value)
        self.label = str(label)
        self.released = False

    def deep_copy(self) -> "_FakeCipher":
        clone = _FakeCipher(self.value, label=f"{self.label}.copy")
        return clone

    def release(self) -> None:
        self.released = True


class _FakeFhe:
    def homo_add(self, left, right, context):  # noqa: ANN001
        return _FakeCipher(left.value + right.value, label="sum")


class _FakeContext:
    def __init__(self) -> None:
        self.fhe = _FakeFhe()
        self.context = None


def _page(*values: int, label: str) -> tuple[_FakeCipher, ...]:
    return tuple(_FakeCipher(value, label=f"{label}.{index}") for index, value in enumerate(values))


def test_split_flatten_round_trip() -> None:
    flat = _page(1, 2, 3, 4, label="flat")
    pages = split_c_pages(flat, pair_count=2)
    assert len(pages) == 2
    assert flatten_c_pages(pages) == flat


def test_copy_c_pages_is_deep() -> None:
    original = (_page(1, 2, label="p0"),)
    copied = copy_c_pages(original)
    assert copied[0][0] is not original[0][0]
    assert copied[0][0].value == original[0][0].value
    assert copied[0][0].label.endswith(".copy")


def test_extend_opens_first_page_without_copy() -> None:
    token = _page(7, 8, label="tok")
    pages = extend_c_pages((), token, crypto_context=_FakeContext())
    assert pages == (token,)
    assert pages[0][0] is token[0]


def test_extend_transfers_complete_pages_when_consuming() -> None:
    history = _page(1, 2, label="h")
    token = _page(3, 4, label="tok")
    pages = extend_c_pages(
        (history,),
        token,
        crypto_context=_FakeContext(),
        consume=True,
        open_new_page=True,
    )
    assert len(pages) == 2
    assert pages[0][0] is history[0]
    assert pages[1][0] is token[0]


def test_extend_copies_complete_pages_when_not_consuming() -> None:
    history = _page(1, 2, label="h")
    token = _page(3, 4, label="tok")
    pages = extend_c_pages(
        (history,),
        token,
        crypto_context=_FakeContext(),
        consume=False,
        open_new_page=True,
    )
    assert pages[0][0] is not history[0]
    assert pages[1][0] is token[0]


def test_extend_adds_only_the_live_tail() -> None:
    frozen = _page(1, 2, label="frozen")
    tail = _page(10, 20, label="tail")
    token = _page(3, 4, label="tok")
    pages = extend_c_pages(
        (frozen, tail),
        token,
        crypto_context=_FakeContext(),
        consume=True,
        open_new_page=False,
    )
    assert len(pages) == 2
    assert pages[0][0] is frozen[0]
    assert pages[1][0].value == 13
    assert pages[1][1].value == 24
    assert tail[0].released
    assert tail[1].released
    assert not frozen[0].released


def test_extend_kv_rolls_back_value_pages_if_value_append_fails() -> None:
    class _BoomContext(_FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            inner = self.fhe.homo_add

            def _counted(left, right, context):  # noqa: ANN001
                self.calls += 1
                if self.calls > 2:
                    raise RuntimeError("value add failed")
                return inner(left, right, context)

            self.fhe.homo_add = _counted

    key_hist = _page(1, 2, label="kh")
    key_tail = _page(10, 20, label="kt")
    val_hist = _page(3, 4, label="vh")
    val_tail = _page(30, 40, label="vt")
    with pytest.raises(RuntimeError, match="value add failed"):
        extend_kv_c_pages(
            (key_hist, key_tail),
            (val_hist, val_tail),
            _page(1, 1, label="k"),
            _page(2, 2, label="v"),
            crypto_context=_BoomContext(),
            consume=True,
            open_new_page=False,
        )
    assert key_hist[0].released
    assert key_tail[0].released
    assert val_hist[0].released
    assert val_tail[0].released
