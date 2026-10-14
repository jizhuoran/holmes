from __future__ import annotations

from collections.abc import Iterable

import pytest

from llama3fhe.operators.pairfold_kv_cache import (
    PairFoldCipherPage,
    PairFoldLayerCache,
    PairFoldTokenWrite,
)


class _CopyController:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.calls = 0
        self.outputs: list[_FakeCipher] = []

    def copy(self, source: _FakeCipher) -> _FakeCipher:
        call_index = self.calls
        self.calls += 1
        source.copy_count += 1
        if call_index == self.fail_at:
            raise RuntimeError(f"copy failed at {call_index}")
        output = _FakeCipher(
            source.value,
            label=f"{source.label}.copy",
            copy_controller=self,
        )
        self.outputs.append(output)
        return output


class _FakeCipher:
    def __init__(
        self,
        value: int,
        *,
        label: str,
        copy_controller: _CopyController | None = None,
    ) -> None:
        self.value = int(value)
        self.label = str(label)
        self.copy_controller = copy_controller
        self.copy_count = 0
        self.release_count = 0

    def deep_copy(self) -> _FakeCipher:
        if self.copy_controller is not None:
            return self.copy_controller.copy(self)
        self.copy_count += 1
        return _FakeCipher(self.value, label=f"{self.label}.copy")

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"cipher released twice: {self.label}")
        self.release_count = 1


class _FakeFhe:
    def __init__(self, native_context: object, *, fail_at: int | None = None) -> None:
        self.native_context = native_context
        self.fail_at = fail_at
        self.calls: list[tuple[_FakeCipher, _FakeCipher]] = []
        self.outputs: list[_FakeCipher] = []

    def homo_add(
        self,
        left: _FakeCipher,
        right: _FakeCipher,
        context: object,
    ) -> _FakeCipher:
        assert context is self.native_context
        call_index = len(self.calls)
        self.calls.append((left, right))
        if call_index == self.fail_at:
            raise RuntimeError(f"add failed at {call_index}")
        output = _FakeCipher(
            left.value + right.value,
            label=f"add.{call_index}",
        )
        self.outputs.append(output)
        return output


class _FakeContext:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.context = object()
        self.fhe = _FakeFhe(self.context, fail_at=fail_at)


def _groups(
    label: str,
    start: int,
    *,
    controller: _CopyController | None = None,
) -> tuple[_FakeCipher, _FakeCipher]:
    return (
        _FakeCipher(start, label=f"{label}.0", copy_controller=controller),
        _FakeCipher(start + 1, label=f"{label}.1", copy_controller=controller),
    )


def _token(
    column: int,
    *,
    label: str = "token",
    start: int = 100,
    controller: _CopyController | None = None,
) -> PairFoldTokenWrite:
    return PairFoldTokenWrite(
        key_groups=_groups(f"{label}.k", start, controller=controller),
        value_groups=_groups(f"{label}.v", start + 10, controller=controller),
        column=column,
    )


def _page(
    occupied: int,
    *,
    label: str = "page",
    start: int = 10,
    controller: _CopyController | None = None,
) -> PairFoldCipherPage:
    return PairFoldCipherPage(
        key_groups=_groups(f"{label}.k", start, controller=controller),
        value_groups=_groups(f"{label}.v", start + 10, controller=controller),
        occupied=occupied,
    )


def _handles(owner: object) -> tuple[_FakeCipher, ...]:
    return tuple(owner.key_groups) + tuple(owner.value_groups)


def _assert_same_handles(
    actual: Iterable[_FakeCipher], expected: Iterable[_FakeCipher]
) -> None:
    actual_tuple = tuple(actual)
    expected_tuple = tuple(expected)
    assert len(actual_tuple) == len(expected_tuple)
    assert all(got is wanted for got, wanted in zip(actual_tuple, expected_tuple))


def _assert_live(handles: Iterable[_FakeCipher]) -> None:
    assert all(cipher.release_count == 0 for cipher in handles)


def _assert_released_once(handles: Iterable[_FakeCipher]) -> None:
    assert all(cipher.release_count == 1 for cipher in handles)


def test_empty_cache_append_moves_token_handles_without_work() -> None:
    cache = PairFoldLayerCache()
    token = _token(0)
    token_handles = _handles(token)
    context = _FakeContext()

    cache.append(token, crypto_context=context)

    assert cache.layout_tag == "pairfold128_v1"
    assert cache.content_tokens == 1
    assert len(cache.pages) == 1
    assert cache.pages[0].occupied == 1
    _assert_same_handles(_handles(cache.pages[0]), token_handles)
    assert not token.key_groups
    assert not token.value_groups
    assert context.fhe.calls == []
    _assert_live(token_handles)
    assert all(cipher.copy_count == 0 for cipher in token_handles)


@pytest.mark.parametrize("occupied", (1, 128))
def test_page_from_token_write_moves_handles_and_releases_exactly_once(
    occupied: int,
) -> None:
    token = _token(0, label=f"prefill.{occupied}")
    token_handles = _handles(token)

    page = PairFoldCipherPage.from_token_write(token, occupied)

    assert token.is_consumed
    assert not token.key_groups
    assert not token.value_groups
    assert page.occupied == occupied
    _assert_same_handles(_handles(page), token_handles)
    _assert_live(token_handles)
    assert all(cipher.copy_count == 0 for cipher in token_handles)

    page.release()
    page.release()
    _assert_released_once(token_handles)


@pytest.mark.parametrize("occupied", (0, 129))
def test_page_from_token_write_invalid_occupancy_is_atomic(occupied: int) -> None:
    token = _token(0, label=f"invalid.{occupied}")
    token_handles = _handles(token)

    with pytest.raises(ValueError, match="occupied"):
        PairFoldCipherPage.from_token_write(token, occupied)

    assert not token.is_consumed
    _assert_same_handles(_handles(token), token_handles)
    _assert_live(token_handles)
    assert all(cipher.copy_count == 0 for cipher in token_handles)


def test_page_from_token_write_constructor_failure_is_atomic() -> None:
    class _RejectingPage(PairFoldCipherPage):
        def __post_init__(self) -> None:
            super().__post_init__()
            raise RuntimeError("injected page construction failure")

    token = _token(0, label="constructor-failure")
    token_handles = _handles(token)

    with pytest.raises(RuntimeError, match="injected page construction failure"):
        _RejectingPage.from_token_write(token, 128)

    assert not token.is_consumed
    _assert_same_handles(_handles(token), token_handles)
    _assert_live(token_handles)
    assert all(cipher.copy_count == 0 for cipher in token_handles)


def test_full_page_append_opens_new_page_and_keeps_history_zero_copy() -> None:
    history = _page(128, label="history")
    history_handles = _handles(history)
    cache = PairFoldLayerCache(pages=(history,), content_tokens=128)
    token = _token(0)
    token_handles = _handles(token)
    context = _FakeContext()

    cache.append(token, crypto_context=context)

    assert cache.content_tokens == 129
    assert len(cache.pages) == 2
    assert cache.pages[0] is history
    _assert_same_handles(_handles(cache.pages[0]), history_handles)
    assert cache.pages[1].occupied == 1
    _assert_same_handles(_handles(cache.pages[1]), token_handles)
    assert not token.key_groups
    assert not token.value_groups
    assert context.fhe.calls == []
    _assert_live((*history_handles, *token_handles))
    assert all(cipher.copy_count == 0 for cipher in (*history_handles, *token_handles))


def test_extend_pages_moves_prefill_pages_without_copying() -> None:
    history_page = _page(128, label="history")
    new_page = _page(3, label="new", start=20)
    history = PairFoldLayerCache(pages=(history_page,), content_tokens=128)
    incoming = PairFoldLayerCache(pages=(new_page,), content_tokens=3)
    handles = history.handles + incoming.handles

    history.extend_pages(incoming)

    assert history.pages == (history_page, new_page)
    assert history.content_tokens == 131
    assert incoming.is_released
    _assert_live(handles)
    history.release()
    _assert_released_once(handles)


def test_extend_pages_rejects_partial_history_without_moving_owners() -> None:
    history_page = _page(127, label="history")
    new_page = _page(1, label="new", start=20)
    history = PairFoldLayerCache(pages=(history_page,), content_tokens=127)
    incoming = PairFoldLayerCache(pages=(new_page,), content_tokens=1)
    handles = history.handles + incoming.handles

    with pytest.raises(ValueError, match="tail must be full"):
        history.extend_pages(incoming)

    assert history.pages == (history_page,)
    assert incoming.pages == (new_page,)
    _assert_live(handles)
    history.release()
    incoming.release()
    _assert_released_once(handles)


def test_partial_append_adds_four_routes_and_commits_ownership() -> None:
    frozen = _page(128, label="frozen", start=0)
    tail = _page(5, label="tail", start=20)
    frozen_handles = _handles(frozen)
    tail_handles = _handles(tail)
    cache = PairFoldLayerCache(
        pages=(frozen, tail),
        content_tokens=133,
    )
    token = _token(5, start=200)
    token_handles = _handles(token)
    context = _FakeContext()

    cache.append(token, crypto_context=context)

    assert cache.content_tokens == 134
    assert cache.pages[0] is frozen
    assert cache.pages[1] is not tail
    assert cache.pages[1].occupied == 6
    assert context.fhe.calls == [
        (tail_handles[0], token_handles[0]),
        (tail_handles[1], token_handles[1]),
        (tail_handles[2], token_handles[2]),
        (tail_handles[3], token_handles[3]),
    ]
    _assert_same_handles(_handles(cache.pages[1]), context.fhe.outputs)
    assert [cipher.value for cipher in context.fhe.outputs] == [
        left.value + right.value
        for left, right in zip(tail_handles, token_handles)
    ]
    _assert_live(frozen_handles)
    _assert_released_once((*tail_handles, *token_handles))
    assert not token.key_groups
    assert not token.value_groups
    assert all(cipher.copy_count == 0 for cipher in frozen_handles)


@pytest.mark.parametrize("fail_at", range(4), ids=("K0", "K1", "V0", "V1"))
def test_partial_append_rolls_back_each_add_failure(fail_at: int) -> None:
    frozen = _page(128, label="frozen", start=0)
    tail = _page(11, label="tail", start=20)
    cache = PairFoldLayerCache(
        pages=(frozen, tail),
        content_tokens=139,
    )
    pages_before = tuple(cache.pages)
    frozen_handles = _handles(frozen)
    tail_handles = _handles(tail)
    token = _token(11, start=200)
    token_handles = _handles(token)
    context = _FakeContext(fail_at=fail_at)

    with pytest.raises(RuntimeError, match=f"add failed at {fail_at}"):
        cache.append(token, crypto_context=context)

    assert cache.content_tokens == 139
    assert len(cache.pages) == 2
    assert all(
        actual is expected for actual, expected in zip(cache.pages, pages_before)
    )
    assert cache.pages[0] is frozen
    assert cache.pages[1] is tail
    assert tail.occupied == 11
    _assert_same_handles(_handles(tail), tail_handles)
    _assert_same_handles(_handles(token), token_handles)
    _assert_live((*frozen_handles, *tail_handles, *token_handles))
    assert len(context.fhe.calls) == fail_at + 1
    assert len(context.fhe.outputs) == fail_at
    _assert_released_once(context.fhe.outputs)


@pytest.mark.parametrize(
    ("scenario", "wrong_column"),
    (("empty", 1), ("partial", 6), ("full", 127)),
)
def test_wrong_column_is_rejected_before_add_or_release(
    scenario: str,
    wrong_column: int,
) -> None:
    if scenario == "empty":
        cache = PairFoldLayerCache()
    elif scenario == "partial":
        cache = PairFoldLayerCache(
            pages=(_page(7, label="partial"),),
            content_tokens=7,
        )
    else:
        cache = PairFoldLayerCache(
            pages=(_page(128, label="full"),),
            content_tokens=128,
        )
    pages_before = tuple(cache.pages)
    cache_handles = tuple(
        cipher for page in cache.pages for cipher in _handles(page)
    )
    content_before = cache.content_tokens
    token = _token(wrong_column)
    token_handles = _handles(token)
    context = _FakeContext()

    with pytest.raises(ValueError):
        cache.append(token, crypto_context=context)

    assert cache.content_tokens == content_before
    assert len(cache.pages) == len(pages_before)
    assert all(
        actual is expected for actual, expected in zip(cache.pages, pages_before)
    )
    _assert_same_handles(_handles(token), token_handles)
    assert context.fhe.calls == []
    _assert_live((*cache_handles, *token_handles))


def test_column_127_fills_tail_then_column_zero_opens_a_new_page() -> None:
    tail = _page(127, label="tail")
    cache = PairFoldLayerCache(pages=(tail,), content_tokens=127)
    last_token = _token(127, label="last", start=200)
    context = _FakeContext()

    cache.append(last_token, crypto_context=context)

    assert cache.content_tokens == 128
    assert len(cache.pages) == 1
    full_page = cache.pages[0]
    assert full_page.occupied == 128
    full_page_handles = _handles(full_page)
    assert len(context.fhe.calls) == 4

    next_token = _token(0, label="next", start=300)
    next_handles = _handles(next_token)
    cache.append(next_token, crypto_context=context)

    assert cache.content_tokens == 129
    assert len(cache.pages) == 2
    assert cache.pages[0] is full_page
    _assert_same_handles(_handles(cache.pages[0]), full_page_handles)
    assert cache.pages[1].occupied == 1
    _assert_same_handles(_handles(cache.pages[1]), next_handles)
    assert len(context.fhe.calls) == 4
    _assert_live((*full_page_handles, *next_handles))


def test_token_page_and_cache_clone_are_deep() -> None:
    token = _token(37, label="token")
    token_handles = _handles(token)
    token_clone = token.clone()

    assert token_clone is not token
    assert token_clone.column == token.column
    for original, copied in zip(token_handles, _handles(token_clone)):
        assert copied is not original
        assert copied.value == original.value
        assert original.copy_count == 1
        assert original.release_count == 0

    page = _page(53, label="page")
    page_handles = _handles(page)
    page_clone = page.clone()

    assert page_clone is not page
    assert page_clone.occupied == page.occupied
    for original, copied in zip(page_handles, _handles(page_clone)):
        assert copied is not original
        assert copied.value == original.value
        assert original.copy_count == 1
        assert original.release_count == 0

    first = _page(128, label="first", start=0)
    second = _page(3, label="second", start=20)
    cache = PairFoldLayerCache(
        pages=(first, second),
        content_tokens=131,
    )
    cache_clone = cache.clone()

    assert cache_clone is not cache
    assert cache_clone.layout_tag == cache.layout_tag
    assert cache_clone.content_tokens == 131
    assert len(cache_clone.pages) == 2
    for original_page, copied_page in zip(cache.pages, cache_clone.pages):
        assert copied_page is not original_page
        assert copied_page.occupied == original_page.occupied
        for original, copied in zip(_handles(original_page), _handles(copied_page)):
            assert copied is not original
            assert copied.value == original.value
            assert original.copy_count == 1
            assert original.release_count == 0


@pytest.mark.parametrize(
    ("owner_kind", "fail_at"),
    (("token", 2), ("page", 2), ("cache", 5)),
)
def test_clone_failure_releases_only_partial_copies(
    owner_kind: str,
    fail_at: int,
) -> None:
    controller = _CopyController(fail_at=fail_at)
    if owner_kind == "token":
        owner = _token(4, controller=controller)
        pages_before: tuple[PairFoldCipherPage, ...] = ()
    elif owner_kind == "page":
        owner = _page(4, controller=controller)
        pages_before = ()
    else:
        owner = PairFoldLayerCache(
            pages=(
                _page(128, label="first", controller=controller),
                _page(4, label="second", start=20, controller=controller),
            ),
            content_tokens=132,
        )
        pages_before = tuple(owner.pages)
    source_handles = (
        _handles(owner)
        if owner_kind != "cache"
        else tuple(cipher for page in owner.pages for cipher in _handles(page))
    )

    with pytest.raises(RuntimeError, match=f"copy failed at {fail_at}"):
        owner.clone()

    assert controller.calls == fail_at + 1
    assert len(controller.outputs) == fail_at
    _assert_released_once(controller.outputs)
    _assert_live(source_handles)
    if owner_kind == "cache":
        assert owner.content_tokens == 132
        assert all(
            actual is expected
            for actual, expected in zip(owner.pages, pages_before)
        )
    else:
        _assert_same_handles(_handles(owner), source_handles)


def test_clone_rejects_a_deep_copy_returning_any_source_handle() -> None:
    page = _page(4, label="source-alias")
    source_handles = _handles(page)
    source_handles[0].deep_copy = lambda: source_handles[1]  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="source handle"):
        page.clone()

    _assert_live(source_handles)
    page.release()


def test_clone_rejects_duplicate_clone_handle_and_releases_it_once() -> None:
    page = _page(4, label="clone-alias")
    source_handles = _handles(page)
    shared_clone = _FakeCipher(99, label="shared-clone")
    source_handles[0].deep_copy = lambda: shared_clone  # type: ignore[method-assign]
    source_handles[1].deep_copy = lambda: shared_clone  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="must not alias"):
        page.clone()

    assert shared_clone.release_count == 1
    _assert_live(source_handles)
    page.release()


def test_cache_clone_checks_source_aliases_across_page_boundaries() -> None:
    controller = _CopyController()
    first = _page(128, label="first", controller=controller)
    second = _page(3, label="second", start=20, controller=controller)
    source_handles = (*_handles(first), *_handles(second))
    second.key_groups[0].deep_copy = lambda: first.key_groups[0]  # type: ignore[method-assign]
    cache = PairFoldLayerCache(pages=(first, second), content_tokens=131)

    with pytest.raises(RuntimeError, match="source handle"):
        cache.clone()

    _assert_live(source_handles)
    _assert_released_once(controller.outputs)
    cache.release()


def test_token_page_and_cache_release_are_idempotent() -> None:
    token = _token(0, label="token")
    token_handles = _handles(token)
    token.release()
    token.release()
    _assert_released_once(token_handles)
    assert not token.key_groups
    assert not token.value_groups

    page = _page(7, label="page")
    page_handles = _handles(page)
    page.release()
    page.release()
    _assert_released_once(page_handles)
    assert not page.key_groups
    assert not page.value_groups

    first = _page(128, label="first", start=0)
    second = _page(7, label="second", start=20)
    cache_handles = (*_handles(first), *_handles(second))
    cache = PairFoldLayerCache(
        pages=(first, second),
        content_tokens=135,
    )
    cache.release()
    cache.release()
    _assert_released_once(cache_handles)
    assert not cache.pages


def test_released_cache_rejects_clone_and_append_without_consuming_token() -> None:
    page = _page(3)
    page_handles = _handles(page)
    cache = PairFoldLayerCache(pages=(page,), content_tokens=3)
    cache.release()
    token = _token(3)
    token_handles = _handles(token)
    context = _FakeContext()

    with pytest.raises(RuntimeError, match="released"):
        cache.clone()
    with pytest.raises(RuntimeError, match="released"):
        cache.append(token, crypto_context=context)

    assert not cache.pages
    _assert_released_once(page_handles)
    _assert_same_handles(_handles(token), token_handles)
    _assert_live(token_handles)
    assert context.fhe.calls == []


@pytest.mark.parametrize("column", (-1, 128))
def test_token_rejects_column_outside_page(column: int) -> None:
    with pytest.raises(ValueError):
        _token(column)


@pytest.mark.parametrize("occupied", (0, 129))
def test_page_rejects_invalid_occupied_count(occupied: int) -> None:
    with pytest.raises(ValueError):
        _page(occupied)


def test_token_and_page_reject_group_width_other_than_two() -> None:
    one = (_FakeCipher(1, label="one"),)
    two = _groups("two", 2)
    three = (*_groups("three", 4), _FakeCipher(6, label="three.2"))

    with pytest.raises(ValueError):
        PairFoldTokenWrite(key_groups=one, value_groups=two, column=0)
    with pytest.raises(ValueError):
        PairFoldTokenWrite(key_groups=two, value_groups=three, column=0)
    with pytest.raises(ValueError):
        PairFoldCipherPage(key_groups=one, value_groups=two, occupied=1)
    with pytest.raises(ValueError):
        PairFoldCipherPage(key_groups=two, value_groups=three, occupied=1)


@pytest.mark.parametrize("owner_kind", ("token", "page"))
def test_token_and_page_reject_aliased_handles(owner_kind: str) -> None:
    shared = _FakeCipher(1, label="shared")
    key_groups = (shared, _FakeCipher(2, label="key.1"))
    value_groups = (_FakeCipher(3, label="value.0"), shared)

    with pytest.raises(ValueError, match="alias"):
        if owner_kind == "token":
            PairFoldTokenWrite(
                key_groups=key_groups,
                value_groups=value_groups,
                column=0,
            )
        else:
            PairFoldCipherPage(
                key_groups=key_groups,
                value_groups=value_groups,
                occupied=1,
            )

    _assert_live((shared, key_groups[1], value_groups[0]))


def test_cache_rejects_layout_content_and_page_invariant_violations() -> None:
    with pytest.raises(ValueError, match="layout"):
        PairFoldLayerCache(layout_tag="some_other_layout")
    with pytest.raises(ValueError):
        PairFoldLayerCache(content_tokens=1)

    page = _page(3)
    with pytest.raises(ValueError):
        PairFoldLayerCache(pages=(page,), content_tokens=2)

    partial = _page(2, label="partial")
    full = _page(128, label="full", start=20)
    with pytest.raises(ValueError):
        PairFoldLayerCache(
            pages=(partial, full),
            content_tokens=130,
        )


def test_cache_rejects_handle_alias_across_pages() -> None:
    first = _page(128, label="first")
    shared = first.key_groups[0]
    second = PairFoldCipherPage(
        key_groups=(shared, _FakeCipher(30, label="second.k.1")),
        value_groups=_groups("second.v", 40),
        occupied=1,
    )

    with pytest.raises(ValueError, match="alias"):
        PairFoldLayerCache(
            pages=(first, second),
            content_tokens=129,
        )

    _assert_live((*_handles(first), second.key_groups[1], *second.value_groups))
