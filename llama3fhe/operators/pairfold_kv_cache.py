from __future__ import annotations

"""Owned ciphertext pages for the PairFold128 Decode KV cache.

The cache is deliberately independent from the legacy C-form cache.  A token
write owns exactly four ciphertexts until :meth:`PairFoldLayerCache.append`
either moves them into a new page or consumes them after a transactional tail
update.  Complete historical pages are never copied by ``append``.
"""

from dataclasses import dataclass, field
from typing import Iterable

from llama3fhe.backend import release_if_supported
from llama3fhe.operators.primitives import drop_to_limbs

PAIRFOLD_LAYOUT_TAG = "pairfold128_v1"
PAIRFOLD_PAGE_SIZE = 128
PAIRFOLD_GROUP_COUNT = 2


def _as_group(values: Iterable[object], *, name: str) -> tuple[object, ...]:
    group = tuple(values)
    if len(group) != PAIRFOLD_GROUP_COUNT:
        raise ValueError(
            f"{name} must contain exactly {PAIRFOLD_GROUP_COUNT} ciphertexts, "
            f"got {len(group)}."
        )
    if any(value is None for value in group):
        raise ValueError(f"{name} must not contain None.")
    return group


def _validate_unique_handles(handles: Iterable[object], *, owner: str) -> None:
    handles = tuple(handles)
    if len({id(handle) for handle in handles}) != len(handles):
        raise ValueError(f"{owner} ciphertext handles must not alias.")


def _release_handles(handles: Iterable[object]) -> BaseException | None:
    """Best-effort release, returning the first error after trying every handle."""

    first_error: BaseException | None = None
    seen: set[int] = set()
    for handle in tuple(handles):
        identity = id(handle)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            release_if_supported(handle)
        except BaseException as error:  # resource cleanup must continue
            if first_error is None:
                first_error = error
    return first_error


def _clone_handles(handles: Iterable[object]) -> tuple[object, ...]:
    sources = tuple(handles)
    _validate_unique_handles(sources, owner="PairFold clone sources")
    source_ids = {id(handle) for handle in sources}
    clones: list[object] = []
    clone_ids: set[int] = set()
    try:
        for handle in sources:
            clone = handle.deep_copy()
            if clone is None:
                raise RuntimeError("ciphertext deep_copy returned None.")
            identity = id(clone)
            if identity in source_ids:
                raise RuntimeError(
                    "ciphertext deep_copy must not return any source handle."
                )
            if identity in clone_ids:
                raise RuntimeError(
                    "ciphertext deep_copy outputs must not alias each other."
                )
            clone_ids.add(identity)
            clones.append(clone)
    except BaseException:
        _release_handles(
            clone for clone in clones if id(clone) not in source_ids
        )
        raise
    return tuple(clones)


@dataclass
class PairFoldTokenWrite:
    """Owned K/V payload for one PairFold128 token column."""

    key_groups: tuple[object, ...] | list[object]
    value_groups: tuple[object, ...] | list[object]
    column: int
    _released: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.key_groups = _as_group(self.key_groups, name="key_groups")
        self.value_groups = _as_group(self.value_groups, name="value_groups")
        self.column = int(self.column)
        if not 0 <= self.column < PAIRFOLD_PAGE_SIZE:
            raise ValueError(
                f"column must be in [0, {PAIRFOLD_PAGE_SIZE - 1}], "
                f"got {self.column}."
            )
        _validate_unique_handles(self.handles, owner="token write")

    @property
    def handles(self) -> tuple[object, ...]:
        return tuple(self.key_groups) + tuple(self.value_groups)

    @property
    def is_released(self) -> bool:
        return bool(self._released)

    @property
    def is_consumed(self) -> bool:
        return bool(self._released)

    def _assert_live(self) -> None:
        if self._released:
            raise RuntimeError("PairFold token write has already been consumed.")

    def _detach(self) -> tuple[object, ...]:
        """Drop ownership without releasing; the caller must own the handles."""

        self._assert_live()
        owned = self.handles
        self.key_groups = ()
        self.value_groups = ()
        self._released = True
        return owned

    def clone(self) -> PairFoldTokenWrite:
        self._assert_live()
        clones = _clone_handles(self.handles)
        try:
            return PairFoldTokenWrite(
                key_groups=clones[:PAIRFOLD_GROUP_COUNT],
                value_groups=clones[PAIRFOLD_GROUP_COUNT:],
                column=self.column,
            )
        except BaseException:
            _release_handles(clones)
            raise

    def move(self) -> PairFoldTokenWrite:
        """Move the four handles into a new live token wrapper.

        The returned token is the sole owner of the same ciphertext handles;
        this wrapper becomes consumed without releasing them.  Constructing
        the destination before detaching makes validation failure atomic.
        """

        self._assert_live()
        moved = PairFoldTokenWrite(
            key_groups=self.key_groups,
            value_groups=self.value_groups,
            column=self.column,
        )
        self._detach()
        return moved

    def release(self) -> None:
        if self._released:
            return
        owned = self._detach()
        error = _release_handles(owned)
        if error is not None:
            raise error


@dataclass
class PairFoldCipherPage:
    """One owned PairFold128 page: two K and two V ciphertexts."""

    key_groups: tuple[object, ...] | list[object]
    value_groups: tuple[object, ...] | list[object]
    occupied: int
    _released: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.key_groups = _as_group(self.key_groups, name="key_groups")
        self.value_groups = _as_group(self.value_groups, name="value_groups")
        self.occupied = int(self.occupied)
        if not 1 <= self.occupied <= PAIRFOLD_PAGE_SIZE:
            raise ValueError(
                f"occupied must be in [1, {PAIRFOLD_PAGE_SIZE}], "
                f"got {self.occupied}."
            )
        _validate_unique_handles(self.handles, owner="PairFold page")

    @property
    def handles(self) -> tuple[object, ...]:
        return tuple(self.key_groups) + tuple(self.value_groups)

    @property
    def remaining(self) -> int:
        return PAIRFOLD_PAGE_SIZE - int(self.occupied)

    @property
    def is_released(self) -> bool:
        return bool(self._released)

    def _assert_live(self) -> None:
        if self._released:
            raise RuntimeError("PairFold page has already been released.")

    def _detach(self) -> tuple[object, ...]:
        self._assert_live()
        owned = self.handles
        self.key_groups = ()
        self.value_groups = ()
        self._released = True
        return owned

    @classmethod
    def from_token_write(
        cls,
        token: PairFoldTokenWrite,
        occupied: int,
    ) -> PairFoldCipherPage:
        """Zero-copy a token payload into a page after full validation.

        ``token`` remains the sole owner while the destination constructor
        validates its groups and occupancy.  Only after construction succeeds
        is the token consumed, leaving the returned page as the sole owner.
        """

        if not isinstance(token, PairFoldTokenWrite):
            raise TypeError("token must be a PairFoldTokenWrite.")
        token._assert_live()
        page = cls(
            key_groups=token.key_groups,
            value_groups=token.value_groups,
            occupied=occupied,
        )
        token._detach()
        return page

    def clone(self) -> PairFoldCipherPage:
        self._assert_live()
        clones = _clone_handles(self.handles)
        try:
            return PairFoldCipherPage(
                key_groups=clones[:PAIRFOLD_GROUP_COUNT],
                value_groups=clones[PAIRFOLD_GROUP_COUNT:],
                occupied=self.occupied,
            )
        except BaseException:
            _release_handles(clones)
            raise

    def release(self) -> None:
        if self._released:
            return
        owned = self._detach()
        error = _release_handles(owned)
        if error is not None:
            raise error


@dataclass
class PairFoldLayerCache:
    """Append-only owned PairFold128 ciphertext cache for one model layer."""

    pages: tuple[PairFoldCipherPage, ...] | list[PairFoldCipherPage] = ()
    content_tokens: int = 0
    layout_tag: str = PAIRFOLD_LAYOUT_TAG
    _released: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.pages = tuple(self.pages)
        self.content_tokens = int(self.content_tokens)
        self.layout_tag = str(self.layout_tag)
        self._validate_invariants()

    @property
    def is_released(self) -> bool:
        return bool(self._released)

    @property
    def occupied(self) -> int:
        """Compatibility spelling for the number of resident content tokens."""

        return int(self.content_tokens)

    @property
    def handles(self) -> tuple[object, ...]:
        """All owned ciphertext handles in page and K/K/V/V order."""

        self._assert_live()
        return tuple(handle for page in self.pages for handle in page.handles)

    def _assert_live(self) -> None:
        if self._released:
            raise RuntimeError("PairFold layer cache has already been released.")

    def _validate_invariants(self) -> None:
        if self.layout_tag != PAIRFOLD_LAYOUT_TAG:
            raise ValueError(
                f"layout_tag must be {PAIRFOLD_LAYOUT_TAG!r}, "
                f"got {self.layout_tag!r}."
            )
        if self.content_tokens < 0:
            raise ValueError("content_tokens must be non-negative.")
        if not self.pages:
            if self.content_tokens != 0:
                raise ValueError("an empty PairFold cache must have zero tokens.")
            return

        all_handles: list[object] = []
        total = 0
        for index, page in enumerate(self.pages):
            if not isinstance(page, PairFoldCipherPage):
                raise TypeError("pages must contain PairFoldCipherPage objects.")
            page._assert_live()
            if index < len(self.pages) - 1 and page.occupied != PAIRFOLD_PAGE_SIZE:
                raise ValueError("every non-tail PairFold page must be full.")
            total += int(page.occupied)
            all_handles.extend(page.handles)
        if total != self.content_tokens:
            raise ValueError(
                "content_tokens must equal the sum of page occupancy, "
                f"got {self.content_tokens} and {total}."
            )
        _validate_unique_handles(all_handles, owner="PairFold cache")

    def clone(self) -> PairFoldLayerCache:
        self._assert_live()
        self._validate_invariants()
        clones = _clone_handles(self.handles)
        copied: list[PairFoldCipherPage] = []
        try:
            for page_index, page in enumerate(self.pages):
                offset = page_index * 2 * PAIRFOLD_GROUP_COUNT
                page_handles = clones[
                    offset : offset + 2 * PAIRFOLD_GROUP_COUNT
                ]
                copied.append(
                    PairFoldCipherPage(
                        key_groups=page_handles[:PAIRFOLD_GROUP_COUNT],
                        value_groups=page_handles[PAIRFOLD_GROUP_COUNT:],
                        occupied=page.occupied,
                    )
                )
            result = PairFoldLayerCache(
                pages=tuple(copied),
                content_tokens=self.content_tokens,
                layout_tag=self.layout_tag,
            )
            clones = ()
            return result
        except BaseException:
            _release_handles(clones)
            raise

    def extend_pages(self, other: PairFoldLayerCache) -> None:
        """Move complete new Prefill pages into this cache without copying CTs.

        The existing tail must be full; only the final incoming page may be
        partial.  Validation happens before either owner changes state.
        """

        self._assert_live()
        if not isinstance(other, PairFoldLayerCache):
            raise TypeError("other must be a PairFoldLayerCache.")
        other._assert_live()
        if self is other:
            raise ValueError("cannot extend a cache with itself.")
        self._validate_invariants()
        other._validate_invariants()
        if not other.pages:
            return
        if self.pages and self.pages[-1].occupied != PAIRFOLD_PAGE_SIZE:
            raise ValueError("existing PairFold tail must be full before page extension.")
        _validate_unique_handles(self.handles + other.handles, owner="combined PairFold cache")
        self.pages = self.pages + other.pages
        self.content_tokens += other.content_tokens
        other.pages = ()
        other.content_tokens = 0
        other._released = True

    def append(self, token: PairFoldTokenWrite, *, crypto_context) -> tuple[int, int]:
        """Consume ``token`` and append it with an all-or-nothing tail update.

        A new page takes the four token handles by move and performs no FHE
        operation.  A partial page creates K0, K1, V0, and V1 sums in that
        order, commits only after all four exist, then releases the old tail
        and token inputs.
        """

        self._assert_live()
        self._validate_invariants()
        if not isinstance(token, PairFoldTokenWrite):
            raise TypeError("token must be a PairFoldTokenWrite.")
        token._assert_live()

        cache_handles = tuple(
            handle for page in self.pages for handle in page.handles
        )
        _validate_unique_handles(
            cache_handles + token.handles, owner="cache and token write"
        )

        opens_page = not self.pages or self.pages[-1].occupied == PAIRFOLD_PAGE_SIZE
        expected_column = 0 if opens_page else int(self.pages[-1].occupied)
        if token.column != expected_column:
            raise ValueError(
                f"token column must be {expected_column}, got {token.column}."
            )

        page_index = len(self.pages) if opens_page else len(self.pages) - 1
        if opens_page:
            new_page = PairFoldCipherPage.from_token_write(token, occupied=1)
            self.pages = self.pages + (new_page,)
            self.content_tokens += 1
            return page_index, 0

        old_tail = self.pages[-1]
        outputs: list[object] = []
        input_ids = {id(handle) for handle in cache_handles + token.handles}
        try:
            for left, right in zip(
                old_tail.handles, token.handles, strict=True
            ):
                outputs.append(
                    crypto_context.fhe.homo_add(
                        left, right, crypto_context.context
                    )
                )
            _validate_unique_handles(outputs, owner="PairFold append outputs")
            if any(id(output) in input_ids for output in outputs):
                raise ValueError("homo_add outputs must not alias their inputs.")
            new_page = PairFoldCipherPage(
                key_groups=tuple(outputs[:PAIRFOLD_GROUP_COUNT]),
                value_groups=tuple(outputs[PAIRFOLD_GROUP_COUNT:]),
                occupied=old_tail.occupied + 1,
            )
        except BaseException:
            _release_handles(
                output for output in outputs if id(output) not in input_ids
            )
            raise

        # Commit ownership before any backend release can fail.  At this point
        # the old cache and token are still intact, so every computation error
        # above leaves both owners unchanged.
        self.pages = self.pages[:-1] + (new_page,)
        self.content_tokens += 1
        old_handles = old_tail._detach()
        token_handles = token._detach()
        release_error = _release_handles(old_handles + token_handles)
        if release_error is not None:
            raise release_error
        return page_index, expected_column

    def release(self) -> None:
        if self._released:
            return
        owned_pages = self.pages
        self.pages = ()
        self.content_tokens = 0
        self._released = True
        first_error: BaseException | None = None
        for page in owned_pages:
            try:
                page.release()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def drop_pairfold_token_to_limbs(
    token: PairFoldTokenWrite,
    *,
    target_limbs: int,
    crypto_context,
    bootstrap_operator=None,
    bootstrap_iterations: int = 1,
    value_target_limbs: int | None = None,
) -> PairFoldTokenWrite:
    """Transactionally consume ``token`` at K and optional V limb targets.

    K0, K1, V0, and V1 are borrowed until all four replacement ciphertexts
    have been produced.  A drop or refresh failure releases only the new
    temporaries and leaves ``token`` live with the same handles.  On success,
    the old token is consumed and the returned token is the sole owner.

    If all inputs already have their targets, ownership is moved into a fresh
    wrapper without a copy or backend call. Otherwise even an input already
    at its target is copied by :func:`drop_to_limbs`, so the commit never
    aliases the old token.
    """

    if not isinstance(token, PairFoldTokenWrite):
        raise TypeError("token must be a PairFoldTokenWrite.")
    token._assert_live()
    target_limbs = int(target_limbs)
    if target_limbs <= 1:
        raise ValueError(
            f"modulus drop must leave more than one limb, got {target_limbs}."
        )
    value_target_limbs = (
        target_limbs if value_target_limbs is None else int(value_target_limbs)
    )
    if value_target_limbs <= 1:
        raise ValueError(
            "value modulus drop must leave more than one limb, got "
            f"{value_target_limbs}."
        )
    bootstrap_iterations = int(bootstrap_iterations)
    if bootstrap_iterations <= 0:
        raise ValueError(
            "bootstrap_iterations must be positive, got "
            f"{bootstrap_iterations}."
        )

    inputs = token.handles
    targets = (target_limbs,) * PAIRFOLD_GROUP_COUNT + (
        value_target_limbs,
    ) * PAIRFOLD_GROUP_COUNT
    for cipher in inputs:
        if int(cipher.state.scale_degree) != 1 or bool(cipher.is_ext):
            raise ValueError(
                "modulus-only scheduling requires non-extended degree-1 "
                "ciphertexts."
            )
    current_limbs = tuple(int(cipher.state.cur_limbs) for cipher in inputs)
    if any(have < target for have, target in zip(current_limbs, targets, strict=True)):
        if bootstrap_operator is None:
            have, target = next(
                (have, target)
                for have, target in zip(current_limbs, targets, strict=True)
                if have < target
            )
            raise ValueError(
                f"too few limbs for the drop: {have} < {target}."
            )

    if all(have == target for have, target in zip(current_limbs, targets, strict=True)):
        return token.move()

    input_ids = {id(cipher) for cipher in inputs}
    outputs: list[object] = []
    replacement: PairFoldTokenWrite | None = None
    try:
        for cipher, have, target in zip(inputs, current_limbs, targets, strict=True):
            working = cipher
            refreshed = None
            if have < target:
                refreshed = bootstrap_operator.refresh(
                    cipher, iterations=bootstrap_iterations
                )
                if id(refreshed) in input_ids:
                    raise RuntimeError(
                        "bootstrap refresh must return a new owned ciphertext."
                    )
                working = refreshed
            try:
                dropped = drop_to_limbs(
                    (working,),
                    target_limbs=target,
                    crypto_context=crypto_context,
                )[0]
                outputs.append(dropped)
            finally:
                if refreshed is not None:
                    release_if_supported(refreshed)

        _validate_unique_handles(outputs, owner="PairFold limb-drop outputs")
        if any(id(output) in input_ids for output in outputs):
            raise RuntimeError(
                "PairFold limb-drop outputs must not alias token inputs."
            )
        replacement = PairFoldTokenWrite(
            key_groups=tuple(outputs[:PAIRFOLD_GROUP_COUNT]),
            value_groups=tuple(outputs[PAIRFOLD_GROUP_COUNT:]),
            column=token.column,
        )
        outputs = []  # ownership transferred to ``replacement``

        try:
            token.release()
        except BaseException:
            try:
                replacement.release()
            except BaseException:
                pass
            replacement = None
            raise
        result = replacement
        replacement = None
        return result
    except BaseException:
        cleanup_error = _release_handles(
            handle for handle in outputs if id(handle) not in input_ids
        )
        if replacement is not None:
            try:
                replacement.release()
            except BaseException:
                pass
        if cleanup_error is not None:
            # Preserve the computational failure; cleanup is best effort on
            # the rollback path just as it is for cache append.
            pass
        raise


__all__ = [
    "PAIRFOLD_GROUP_COUNT",
    "PAIRFOLD_LAYOUT_TAG",
    "PAIRFOLD_PAGE_SIZE",
    "PairFoldCipherPage",
    "PairFoldLayerCache",
    "PairFoldTokenWrite",
    "drop_pairfold_token_to_limbs",
]
