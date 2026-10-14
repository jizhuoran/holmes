from __future__ import annotations

"""Ciphertext helpers for C-form KV cache pages.

v1 does not change EasyFHE.  Appending a token is ``homo_add`` of a
single-column C payload into the live page.  Callers pack that column with
:meth:`AttentionPairLayout.pack_token_c` (or encrypt an already-rotated
column-0 payload after ``homo_rotate`` by ``2 * token_index``).
"""

from llama3fhe.backend import release_if_supported
from llama3fhe.operators.primitives import drop_to_limbs


def add_c_pages(
    page_ciphers: tuple[object, ...] | list[object],
    token_ciphers: tuple[object, ...] | list[object],
    *,
    crypto_context,
) -> tuple[object, ...]:
    """Return ``page + token`` without consuming the inputs."""

    page_ciphers = tuple(page_ciphers)
    token_ciphers = tuple(token_ciphers)
    if len(page_ciphers) != len(token_ciphers):
        raise ValueError(
            "C-page add expects matching cipher counts, got "
            f"{len(page_ciphers)} and {len(token_ciphers)}."
        )
    outputs: list[object] = []
    try:
        for left, right in zip(page_ciphers, token_ciphers, strict=True):
            outputs.append(
                crypto_context.fhe.homo_add(left, right, crypto_context.context)
            )
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for output in outputs:
            release_if_supported(output)


def split_c_pages(
    ciphers: tuple[object, ...] | list[object] | None,
    *,
    pair_count: int,
) -> tuple[tuple[object, ...], ...]:
    """Split a flat K_C / V_C stream into pair-cipher pages."""

    ciphers = tuple(ciphers or ())
    width = int(pair_count)
    if width <= 0:
        raise ValueError("pair_count must be positive.")
    if not ciphers:
        return ()
    if len(ciphers) % width:
        raise ValueError(
            "C-page stream length must be a multiple of the pair count, "
            f"got {len(ciphers)} and pair_count={width}."
        )
    return tuple(
        ciphers[index : index + width]
        for index in range(0, len(ciphers), width)
    )


def flatten_c_pages(
    pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]],
) -> tuple[object, ...]:
    return tuple(cipher for page in pages for cipher in page)


def _release_c_pages(
    pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] | None,
) -> None:
    for page in tuple(pages or ()):
        for cipher in page:
            release_if_supported(cipher)


def copy_c_pages(
    pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]],
) -> tuple[tuple[object, ...], ...]:
    """Deep-copy pages so a LayerResult can own a full cache snapshot."""

    copied: list[tuple[object, ...]] = []
    pending: list[object] = []
    try:
        for page in pages:
            owned: list[object] = []
            for cipher in page:
                clone = cipher.deep_copy()
                pending.append(clone)
                owned.append(clone)
            copied.append(tuple(owned))
            pending = []
        result = tuple(copied)
        copied = []
        return result
    finally:
        for cipher in pending:
            release_if_supported(cipher)
        for page in copied:
            for cipher in page:
                release_if_supported(cipher)


def extend_c_pages(
    pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] | None,
    token_ciphers: tuple[object, ...] | list[object],
    *,
    crypto_context,
    consume: bool = False,
    open_new_page: bool = False,
) -> tuple[tuple[object, ...], ...]:
    """Append one token column onto C-form pages.

    Complete historical pages are ownership-transferred when *consume* is
    true; otherwise they are deep-copied via :func:`copy_c_pages`.  The live
    tail is always a fresh ``homo_add`` (CKKS cannot update in place).
    *consume* then releases the old tail.  An empty *pages* tuple starts the
    first page from *token_ciphers* without copying.  *token_ciphers* are
    never consumed.
    """

    pages = tuple(tuple(page) for page in (pages or ()))
    token = tuple(token_ciphers)
    if not token:
        raise ValueError("token_ciphers must be non-empty.")
    if pages:
        width = len(pages[0])
        if width <= 0 or any(len(page) != width for page in pages):
            raise ValueError("C pages must share a positive cipher count.")
        if len(token) != width:
            raise ValueError(
                "token cipher count must match a C page, got "
                f"{len(token)} and {width}."
            )
    if not pages:
        return (token,)
    if open_new_page:
        if consume:
            return pages + (token,)
        copied = copy_c_pages(pages)
        return copied + (token,)

    prefix_src = pages[:-1]
    old_tail = pages[-1]
    owned_prefix: tuple[tuple[object, ...], ...] = ()
    appended: tuple[object, ...] | None = None
    try:
        owned_prefix = prefix_src if consume else copy_c_pages(prefix_src)
        appended = add_c_pages(
            old_tail, token, crypto_context=crypto_context
        )
    except Exception:
        if consume:
            _release_c_pages(pages)
        else:
            _release_c_pages(owned_prefix)
        raise
    if consume:
        for cipher in old_tail:
            release_if_supported(cipher)
    return owned_prefix + (appended,)


def extend_kv_c_pages(
    key_pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] | None,
    value_pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] | None,
    key_token: tuple[object, ...] | list[object],
    value_token: tuple[object, ...] | list[object],
    *,
    crypto_context,
    consume: bool = False,
    open_new_page: bool = False,
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    """Append one token onto K and V C-pages with joint rollback.

    Either both streams are extended or, on failure, every page this call
    took ownership of is released.  When *consume* is true and the K
    stream succeeds but V fails, the original V pages are released too so
    a Decode ``consume_kv_cache`` caller is not left holding a half-live
    cache.
    """

    key_pages = tuple(tuple(page) for page in (key_pages or ()))
    value_pages = tuple(tuple(page) for page in (value_pages or ()))
    owned_k: tuple[tuple[object, ...], ...] = ()
    owned_v: tuple[tuple[object, ...], ...] = ()
    value_started = False
    try:
        owned_k = extend_c_pages(
            key_pages,
            key_token,
            crypto_context=crypto_context,
            consume=consume,
            open_new_page=open_new_page,
        )
        value_started = True
        owned_v = extend_c_pages(
            value_pages,
            value_token,
            crypto_context=crypto_context,
            consume=consume,
            open_new_page=open_new_page,
        )
        return owned_k, owned_v
    except Exception:
        _release_c_pages(owned_k)
        _release_c_pages(owned_v)
        if consume and not value_started:
            _release_c_pages(value_pages)
        raise


def drop_c_ciphers_to_limbs(
    ciphers: tuple[object, ...] | list[object],
    *,
    target_limbs: int,
    crypto_context,
    bootstrap_operator=None,
    bootstrap_iterations: int = 1,
) -> tuple[object, ...]:
    """Consume ``ciphers`` and return owned copies at ``target_limbs``.

    K_C and V_C leave the adapter at different limb counts, so callers drop
    the two streams separately.  ``drop_to_limbs`` itself does not consume
    its inputs; this wrapper does, so the high-limb originals do not stay
    resident next to the cache.

    Later Prefill layers exit attention below the resident limb count.
    A one-shot write-path refresh raises them before the drop; Decode
    does not bootstrap cache on the hot path.
    """

    ciphers = tuple(ciphers)
    if not ciphers:
        return ()
    target_limbs = int(target_limbs)
    current = {int(cipher.state.cur_limbs) for cipher in ciphers}
    if len(current) > 1:
        raise ValueError(
            "modulus drop expects one common limb count, got "
            f"{sorted(current)}."
        )
    have = next(iter(current))
    working = ciphers
    refreshed: list[object] = []
    try:
        if have < target_limbs:
            if bootstrap_operator is None:
                raise ValueError(
                    f"too few limbs for the drop: {have} < {target_limbs}."
                )
            for cipher in ciphers:
                refreshed.append(
                    bootstrap_operator.refresh(
                        cipher, iterations=int(bootstrap_iterations)
                    )
                )
            working = tuple(refreshed)
            refreshed = []
        dropped = drop_to_limbs(
            working,
            target_limbs=target_limbs,
            crypto_context=crypto_context,
        )
        for cipher in ciphers:
            release_if_supported(cipher)
        if working is not ciphers:
            for cipher in working:
                release_if_supported(cipher)
        return dropped
    except Exception:
        for cipher in refreshed:
            release_if_supported(cipher)
        if working is not ciphers:
            for cipher in working:
                release_if_supported(cipher)
        raise


__all__ = [
    "add_c_pages",
    "copy_c_pages",
    "drop_c_ciphers_to_limbs",
    "extend_c_pages",
    "extend_kv_c_pages",
    "flatten_c_pages",
    "split_c_pages",
]
