from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from llama3fhe.operators.pairfold_kv_cache import (
    PairFoldTokenWrite,
    drop_pairfold_token_to_limbs,
)


@dataclass(frozen=True)
class _FakeState:
    cur_limbs: int
    scale_degree: int = 1
    scaling_factor: float | None = 1.0

    def replace(self, **changes) -> _FakeState:  # noqa: ANN003
        return replace(self, **changes)


class _Tracker:
    def __init__(self) -> None:
        self.copy_outputs: list[_FakeCipher] = []


class _FakeCipher:
    def __init__(
        self,
        label: str,
        limbs: int,
        *,
        tracker: _Tracker,
        scale_degree: int = 1,
        is_ext: bool = False,
    ) -> None:
        self.label = str(label)
        self.state = _FakeState(
            cur_limbs=int(limbs), scale_degree=int(scale_degree)
        )
        self.is_ext = bool(is_ext)
        self.tracker = tracker
        self.release_count = 0
        self.copy_count = 0

    def deep_copy(self) -> _FakeCipher:
        self.copy_count += 1
        output = _FakeCipher(
            f"{self.label}.copy",
            self.state.cur_limbs,
            tracker=self.tracker,
            scale_degree=self.state.scale_degree,
            is_ext=self.is_ext,
        )
        self.tracker.copy_outputs.append(output)
        return output

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"cipher released twice: {self.label}")
        self.release_count = 1


class _FakeFhe:
    def __init__(self, native_context: object, *, fail_at: int | None = None) -> None:
        self.native_context = native_context
        self.fail_at = fail_at
        self.calls: list[tuple[_FakeCipher, _FakeState]] = []
        self.outputs: list[_FakeCipher] = []

    def align_to(
        self,
        cipher: _FakeCipher,
        target_state: _FakeState,
        context: object,
    ) -> _FakeCipher:
        assert context is self.native_context
        call_index = len(self.calls)
        self.calls.append((cipher, target_state))
        if call_index == self.fail_at:
            raise RuntimeError(f"drop failed at {call_index}")
        if (
            cipher.state.cur_limbs == target_state.cur_limbs
            and cipher.state.scale_degree == target_state.scale_degree
        ):
            return cipher
        output = _FakeCipher(
            f"{cipher.label}.drop",
            target_state.cur_limbs,
            tracker=cipher.tracker,
            scale_degree=target_state.scale_degree,
        )
        output.state = target_state
        self.outputs.append(output)
        return output


class _FakeContext:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.context = object()
        self.fhe = _FakeFhe(self.context, fail_at=fail_at)


class _FakeBootstrap:
    def __init__(
        self,
        *,
        refreshed_limbs: int = 10,
        fail_at: int | None = None,
        alias_input: bool = False,
    ) -> None:
        self.refreshed_limbs = int(refreshed_limbs)
        self.fail_at = fail_at
        self.alias_input = bool(alias_input)
        self.calls: list[tuple[_FakeCipher, int]] = []
        self.outputs: list[_FakeCipher] = []

    def refresh(self, cipher: _FakeCipher, *, iterations: int) -> _FakeCipher:
        call_index = len(self.calls)
        self.calls.append((cipher, int(iterations)))
        if call_index == self.fail_at:
            raise RuntimeError(f"refresh failed at {call_index}")
        if self.alias_input:
            return cipher
        output = _FakeCipher(
            f"{cipher.label}.refresh",
            self.refreshed_limbs,
            tracker=cipher.tracker,
        )
        self.outputs.append(output)
        return output


def _token(
    limbs: tuple[int, int, int, int],
    *,
    column: int = 23,
    scale_degree: int = 1,
    is_ext: bool = False,
) -> tuple[PairFoldTokenWrite, _Tracker]:
    tracker = _Tracker()
    handles = tuple(
        _FakeCipher(
            label,
            have,
            tracker=tracker,
            scale_degree=scale_degree,
            is_ext=is_ext,
        )
        for label, have in zip(("K0", "K1", "V0", "V1"), limbs, strict=True)
    )
    return (
        PairFoldTokenWrite(
            key_groups=handles[:2], value_groups=handles[2:], column=column
        ),
        tracker,
    )


def _assert_same_handles(
    actual: tuple[object, ...], expected: tuple[object, ...]
) -> None:
    assert len(actual) == len(expected)
    assert all(left is right for left, right in zip(actual, expected, strict=True))


def _assert_live(handles: tuple[_FakeCipher, ...]) -> None:
    assert all(cipher.release_count == 0 for cipher in handles)


def _assert_released_once(handles: tuple[_FakeCipher, ...]) -> None:
    assert all(cipher.release_count == 1 for cipher in handles)


def test_target_limb_fast_path_moves_without_copy_or_backend_work() -> None:
    token, tracker = _token((6, 6, 6, 6), column=41)
    original_handles = token.handles
    context = _FakeContext()

    result = drop_pairfold_token_to_limbs(
        token, target_limbs=6, crypto_context=context
    )

    assert token.is_consumed
    assert token.handles == ()
    assert not result.is_consumed
    assert result.column == 41
    _assert_same_handles(result.handles, original_handles)
    _assert_live(original_handles)
    assert context.fhe.calls == []
    assert tracker.copy_outputs == []

    result.release()
    _assert_released_once(original_handles)


def test_split_target_fast_path_moves_without_backend_work() -> None:
    token, tracker = _token((12, 12, 8, 8))
    original_handles = token.handles
    context = _FakeContext()

    result = drop_pairfold_token_to_limbs(
        token, target_limbs=12, value_target_limbs=8, crypto_context=context
    )

    assert token.is_consumed
    _assert_same_handles(result.handles, original_handles)
    assert context.fhe.calls == []
    assert tracker.copy_outputs == []
    result.release()
    _assert_released_once(original_handles)


def test_split_target_refreshes_only_shallow_keys() -> None:
    token, _tracker = _token((4, 4, 8, 8))
    original_handles = token.handles
    context = _FakeContext()
    bootstrap = _FakeBootstrap(refreshed_limbs=16)

    result = drop_pairfold_token_to_limbs(
        token,
        target_limbs=12,
        value_target_limbs=8,
        crypto_context=context,
        bootstrap_operator=bootstrap,
    )

    assert token.is_consumed
    assert tuple(cipher.state.cur_limbs for cipher in result.handles) == (
        12, 12, 8, 8
    )
    assert [cipher.label for cipher, _iterations in bootstrap.calls] == [
        "K0",
        "K1",
    ]
    _assert_released_once(original_handles)
    _assert_released_once(tuple(bootstrap.outputs))
    result_handles = result.handles
    result.release()
    _assert_released_once(result_handles)


def test_split_target_value_drop_failure_preserves_token() -> None:
    token, _tracker = _token((4, 4, 8, 8))
    original_handles = token.handles
    context = _FakeContext(fail_at=2)
    bootstrap = _FakeBootstrap(refreshed_limbs=16)

    with pytest.raises(RuntimeError, match="drop failed at 2"):
        drop_pairfold_token_to_limbs(
            token,
            target_limbs=12,
            value_target_limbs=8,
            crypto_context=context,
            bootstrap_operator=bootstrap,
        )

    assert not token.is_consumed
    _assert_same_handles(token.handles, original_handles)
    _assert_live(original_handles)
    assert len(bootstrap.calls) == 2
    _assert_released_once(tuple(bootstrap.outputs))
    _assert_released_once(tuple(context.fhe.outputs))
    token.release()


@pytest.mark.parametrize("fail_at", range(4), ids=("K0", "K1", "V0", "V1"))
def test_each_drop_failure_rolls_back_all_token_ownership(fail_at: int) -> None:
    token, _tracker = _token((9, 9, 9, 9))
    original_handles = token.handles
    original_states = tuple(cipher.state for cipher in original_handles)
    context = _FakeContext(fail_at=fail_at)

    with pytest.raises(RuntimeError, match=f"drop failed at {fail_at}"):
        drop_pairfold_token_to_limbs(
            token, target_limbs=6, crypto_context=context
        )

    assert not token.is_consumed
    _assert_same_handles(token.handles, original_handles)
    assert tuple(cipher.state for cipher in original_handles) == original_states
    _assert_live(original_handles)
    assert len(context.fhe.calls) == fail_at + 1
    assert len(context.fhe.outputs) == fail_at
    _assert_released_once(tuple(context.fhe.outputs))


@pytest.mark.parametrize("fail_at", range(4), ids=("K0", "K1", "V0", "V1"))
def test_each_refresh_failure_rolls_back_all_token_ownership(fail_at: int) -> None:
    token, _tracker = _token((4, 4, 4, 4))
    original_handles = token.handles
    original_states = tuple(cipher.state for cipher in original_handles)
    context = _FakeContext()
    bootstrap = _FakeBootstrap(fail_at=fail_at)

    with pytest.raises(RuntimeError, match=f"refresh failed at {fail_at}"):
        drop_pairfold_token_to_limbs(
            token,
            target_limbs=6,
            crypto_context=context,
            bootstrap_operator=bootstrap,
            bootstrap_iterations=2,
        )

    assert not token.is_consumed
    _assert_same_handles(token.handles, original_handles)
    assert tuple(cipher.state for cipher in original_handles) == original_states
    _assert_live(original_handles)
    assert len(bootstrap.calls) == fail_at + 1
    assert all(iterations == 2 for _cipher, iterations in bootstrap.calls)
    assert len(bootstrap.outputs) == fail_at
    assert len(context.fhe.outputs) == fail_at
    _assert_released_once(tuple(bootstrap.outputs))
    _assert_released_once(tuple(context.fhe.outputs))


def test_mixed_limb_success_commits_only_after_all_four_outputs_exist() -> None:
    token, tracker = _token((6, 9, 4, 7), column=127)
    original_handles = token.handles
    context = _FakeContext()
    bootstrap = _FakeBootstrap()

    result = drop_pairfold_token_to_limbs(
        token,
        target_limbs=6,
        crypto_context=context,
        bootstrap_operator=bootstrap,
        bootstrap_iterations=2,
    )

    assert token.is_consumed
    assert token.handles == ()
    assert result.column == 127
    assert len(result.handles) == 4
    assert all(cipher.state.cur_limbs == 6 for cipher in result.handles)
    assert not {id(cipher) for cipher in result.handles} & {
        id(cipher) for cipher in original_handles
    }
    _assert_released_once(original_handles)
    assert [cipher.label for cipher, _state in context.fhe.calls] == [
        "K0",
        "K1",
        "V0.refresh",
        "V1",
    ]
    assert len(tracker.copy_outputs) == 1
    assert len(bootstrap.outputs) == 1
    _assert_released_once(tuple(bootstrap.outputs))
    _assert_live(result.handles)

    result_handles = result.handles
    result.release()
    _assert_released_once(result_handles)


def test_missing_bootstrap_and_invalid_state_leave_token_unchanged() -> None:
    token, _tracker = _token((4, 4, 4, 4))
    original_handles = token.handles
    context = _FakeContext()

    with pytest.raises(ValueError, match="too few limbs"):
        drop_pairfold_token_to_limbs(
            token, target_limbs=6, crypto_context=context
        )
    _assert_same_handles(token.handles, original_handles)
    _assert_live(original_handles)
    assert context.fhe.calls == []

    invalid, _tracker = _token((6, 6, 6, 6), scale_degree=2)
    invalid_handles = invalid.handles
    with pytest.raises(ValueError, match="degree-1"):
        drop_pairfold_token_to_limbs(
            invalid, target_limbs=6, crypto_context=context
        )
    _assert_same_handles(invalid.handles, invalid_handles)
    _assert_live(invalid_handles)


def test_refresh_alias_is_rejected_without_releasing_borrowed_input() -> None:
    token, _tracker = _token((4, 4, 4, 4))
    original_handles = token.handles
    context = _FakeContext()
    bootstrap = _FakeBootstrap(alias_input=True)

    with pytest.raises(RuntimeError, match="new owned ciphertext"):
        drop_pairfold_token_to_limbs(
            token,
            target_limbs=6,
            crypto_context=context,
            bootstrap_operator=bootstrap,
        )

    _assert_same_handles(token.handles, original_handles)
    _assert_live(original_handles)
    assert context.fhe.calls == []
