from __future__ import annotations

"""Shared BSGS transform for public slot-permutation masks."""

import os
import time
from math import ceil

import numpy as np
from easyfhe import fhe

from llama3fhe.backend import profile_sync_device, release_all, release_if_supported

#: ``LLAMA_FHE_TRANSFORM_HOIST``: BSGS hoisting of the mask transforms
#: (:func:`apply_transform_by_level`).  ``ext_normal`` (default, the production
#: path) keeps the mask plaintexts in the extended basis (``is_ext=True``:
#: ``(cur_limbs + K)`` limbs, +73% bytes/encode at 11 limbs on L3 ``K=8``) so
#: the hoisted MAC skips one moddown per baby step; ``normal`` encodes plain
#: plaintexts and lets ``hoisted_mac_sum(strategy="normal")`` do the moddown.
#: On the 4090 the ext ``blocked`` pairfm mask (512 rows x 19 limbs ~= 3.25 GiB)
#: is what triggers the caching-allocator retry (plan §12.22); ``normal``
#: trades that for more rotation work.  Read once per process.
_TRANSFORM_HOIST_ENV = "LLAMA_FHE_TRANSFORM_HOIST"
_TRANSFORM_HOIST_STRATEGIES = ("ext_normal", "normal")


def transform_hoist_strategy() -> str:
    value = os.environ.get(_TRANSFORM_HOIST_ENV, "").strip() or "ext_normal"
    if value not in _TRANSFORM_HOIST_STRATEGIES:
        raise ValueError(
            f"{_TRANSFORM_HOIST_ENV}={value!r}: expected one of {_TRANSFORM_HOIST_STRATEGIES}"
        )
    return value


def transform_rotation_set(
    *, seq_len: int, stride: int, baby_steps: int
) -> set[int]:
    rows = 2 * int(seq_len) - 1
    giant_count = int(ceil(rows / int(baby_steps)))
    rotations = {-(int(seq_len) - 1) * int(stride)}
    rotations.update(baby * int(stride) for baby in range(1, int(baby_steps)))
    if giant_count > 1:
        rotations.add(int(baby_steps) * int(stride))
    return {int(rotation) for rotation in rotations if int(rotation)}


#: Memo of the padded + giant-rolled BSGS rows, keyed by the *identity* of a
#: read-only mask array (the memoised PairFM masks of
#: :mod:`llama3fhe.operators.attention.prefill.output`) plus ``(stride, baby_steps)``.
#: Only read-only inputs are cached: a writable array may be mutated by its
#: owner, and ``id()`` reuse after free would then alias a stale row set.  The
#: cached array itself is kept alive by the entry, so its ``id`` is stable.
_ROWS_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, tuple[int, ...], int, int]] = {}


def _bsgs_plaintext_rows(
    masks: np.ndarray, *, stride: int, baby_steps: int
) -> tuple[np.ndarray, tuple[int, ...], int, int]:
    masks = np.asarray(masks)
    cacheable = not masks.flags.writeable
    key = (id(masks), int(stride), int(baby_steps))
    if cacheable:
        entry = _ROWS_CACHE.get(key)
        if entry is not None and entry[0] is masks:
            return entry[1:]
    result = _build_bsgs_plaintext_rows(masks, stride=int(stride), baby_steps=int(baby_steps))
    if cacheable:
        result[0].setflags(write=False)
        _ROWS_CACHE[key] = (masks, *result)
    return result


def _build_bsgs_plaintext_rows(
    masks: np.ndarray, *, stride: int, baby_steps: int
) -> tuple[np.ndarray, tuple[int, ...], int, int]:
    giant_count = int(ceil(masks.shape[0] / int(baby_steps)))
    padded = np.zeros(
        (giant_count * int(baby_steps), masks.shape[1]), dtype=masks.dtype
    )
    for relative in range(masks.shape[0]):
        giant = relative // int(baby_steps)
        padded[relative] = np.roll(
            masks[relative], giant * int(baby_steps) * int(stride)
        )
    baby_offsets = tuple(
        baby * int(stride) for baby in range(int(baby_steps))
    )
    base_rotation = -(masks.shape[0] // 2) * int(stride)
    return padded, baby_offsets, giant_count, base_rotation


_TRACE_ENCODE = os.environ.get("LLAMA_FHE_PLAIN_CACHE_TRACE", "0") not in ("", "0", "false", "no")

#: ``LLAMA_FHE_TRANSFORM_MEMO=0`` disables :func:`transform_plaintext_scope`
#: (every call re-encodes, the pre-E6 behaviour).
_MEMO_ENABLED = os.environ.get("LLAMA_FHE_TRANSFORM_MEMO", "1") not in ("", "0", "false", "no")


class TransformPlaintextMemo:
    """Encoded mask plaintexts shared by the calls inside one scope."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int, bool, int], tuple[np.ndarray, object]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(rows: np.ndarray, cur_limbs: int, is_ext: bool, slots: int):
        if rows.flags.writeable:
            return None
        return (id(rows), int(cur_limbs), bool(is_ext), int(slots))

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        self.hits += 1
        return entry[1]

    def put(self, key, rows: np.ndarray, plaintext) -> None:
        self.misses += 1
        self._entries[key] = (rows, plaintext)

    def release(self) -> None:
        entries, self._entries = tuple(self._entries.values()), {}
        for _, plaintext in entries:
            release_if_supported(plaintext)


_MEMO_STACK: list[TransformPlaintextMemo] = []


class transform_plaintext_scope:
    """``with transform_plaintext_scope() as memo:`` -- memoise mask encodes until exit."""

    def __enter__(self) -> TransformPlaintextMemo:
        self.memo = TransformPlaintextMemo()
        if _MEMO_ENABLED:
            _MEMO_STACK.append(self.memo)
        return self.memo

    def __exit__(self, exc_type, exc, traceback) -> None:
        if _MEMO_ENABLED and _MEMO_STACK and _MEMO_STACK[-1] is self.memo:
            _MEMO_STACK.pop()
        self.memo.release()


def _active_memo() -> TransformPlaintextMemo | None:
    return _MEMO_STACK[-1] if _MEMO_STACK else None


def _alloc_retries(device) -> int:
    try:
        import easyfhe as torch

        return int(torch.cuda.memory_stats(device).get("num_alloc_retries", 0))
    except Exception:  # pragma: no cover - diagnostics only
        return 0


def encode_transform_rows(
    rows: np.ndarray, *, name: str, level: int, crypto_context, is_ext: bool = True
):
    rows = np.asarray(rows)
    cur_limbs = int(crypto_context.context.L) - int(level)
    memo = _active_memo()
    memo_key = None
    if memo is not None:
        memo_key = memo.key(rows, cur_limbs, bool(is_ext), int(crypto_context.max_slots))
        if memo_key is not None:
            cached = memo.get(memo_key)
            if cached is not None:
                if _TRACE_ENCODE:
                    print(f"[encode-trace] memo hit rows={tuple(rows.shape)} limbs={cur_limbs} name={name}", flush=True)
                return cached
    if _TRACE_ENCODE:
        profile_sync_device(crypto_context.device)
        retries = _alloc_retries(crypto_context.device)
        started = time.perf_counter()
    bundle = crypto_context.constant_bundle(
        {"rows": rows},
        cache_key=("transform_rows", str(name), bool(is_ext)),
    )
    if _TRACE_ENCODE:
        profile_sync_device(crypto_context.device)
        bundle_s = time.perf_counter() - started
        started = time.perf_counter()
    plaintext = bundle.plaintext(
        "rows",
        state=fhe.CipherState(
            cur_limbs=cur_limbs,
            scale_degree=1,
            scaling_factor=crypto_context.context.scale_at(cur_limbs),
        ),
        slots=int(crypto_context.max_slots),
        context=crypto_context.context,
        is_ext=bool(is_ext),
        cache=True,
    )
    if _TRACE_ENCODE:
        profile_sync_device(crypto_context.device)
        print(
            f"[encode-trace] rows={tuple(rows.shape)} {rows.dtype} limbs={cur_limbs} ext={bool(is_ext)} "
            f"bundle={bundle_s * 1e3:.1f}ms plaintext={(time.perf_counter() - started) * 1e3:.1f}ms "
            f"retries+={_alloc_retries(crypto_context.device) - retries} name={name}",
            flush=True,
        )
    if memo_key is not None:
        memo.put(memo_key, rows, plaintext)
    return plaintext


def _apply_level_group(
    ciphers: tuple[object, ...],
    *,
    masks: np.ndarray,
    stride: int,
    baby_steps: int,
    crypto_context,
    name: str,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    if not ciphers:
        return ()
    levels = {int(crypto_context.level_for_cipher(cipher)) for cipher in ciphers}
    if len(levels) != 1:
        raise ValueError("one transform group must contain one ciphertext level.")
    rows, baby_offsets, giant_count, base_rotation = _bsgs_plaintext_rows(
        masks, stride=int(stride), baby_steps=int(baby_steps)
    )
    strategy = transform_hoist_strategy()
    started = time.perf_counter()
    plaintexts = None
    outputs: list[object] = []
    try:
        plaintexts = encode_transform_rows(
            rows,
            name=name,
            level=levels.pop(),
            crypto_context=crypto_context,
            is_ext=(strategy != "normal"),
        )
        profile_sync_device(crypto_context.device)
        if profile is not None:
            profile["pack_encode"] = (
                profile.get("pack_encode", 0.0) + time.perf_counter() - started
            )
        started = time.perf_counter()
        for cipher in ciphers:
            base = crypto_context.fhe.homo_rotate(
                cipher, base_rotation, crypto_context.context
            )
            try:
                accumulated = crypto_context.fhe.hoisted_mac_sum(
                    base,
                    baby_offsets,
                    plaintexts,
                    int(baby_steps) * int(stride),
                    giant_count,
                    crypto_context.context,
                    strategy=strategy,
                )
            finally:
                release_if_supported(base)
            try:
                outputs.append(
                    crypto_context.fhe.rescale(
                        accumulated, crypto_context.context
                    )
                )
            finally:
                release_if_supported(accumulated)
        profile_sync_device(crypto_context.device)
        if profile is not None:
            profile["online_rescale"] = (
                profile.get("online_rescale", 0.0)
                + time.perf_counter()
                - started
            )
        return tuple(outputs)
    except Exception:
        release_all(outputs)
        raise
    finally:
        release_if_supported(plaintexts)


def apply_transform_by_level(
    ciphers: tuple[object, ...],
    *,
    masks: np.ndarray,
    stride: int,
    baby_steps: int,
    crypto_context,
    name: str,
    profile: dict[str, float] | None = None,
) -> tuple[object, ...]:
    """Apply one mask transform, encoding the masks once per input level.

    Ciphertexts at the same level share one plaintext encode, so the number of
    distinct levels in ``ciphers`` is what this costs; it is reported as
    ``profile["level_groups"]`` rather than returned, because every caller
    wants only the transformed stream.
    """

    by_level: dict[int, list[tuple[int, object]]] = {}
    for index, cipher in enumerate(ciphers):
        level = int(crypto_context.level_for_cipher(cipher))
        by_level.setdefault(level, []).append((index, cipher))
    result: list[object | None] = [None] * len(ciphers)
    try:
        for level, entries in by_level.items():
            outputs = _apply_level_group(
                tuple(cipher for _, cipher in entries),
                masks=masks,
                stride=int(stride),
                baby_steps=int(baby_steps),
                crypto_context=crypto_context,
                name=f"{name}.level{level}",
                profile=profile,
            )
            for (index, _), output in zip(entries, outputs, strict=True):
                result[index] = output
        if any(output is None for output in result):
            raise RuntimeError("transform produced an incomplete output set.")
        if profile is not None:
            profile["level_groups"] = float(len(by_level))
        return tuple(result)  # type: ignore[return-value]
    except Exception:
        release_all(output for output in result if output is not None)
        raise


__all__ = [
    "TransformPlaintextMemo",
    "apply_transform_by_level",
    "encode_transform_rows",
    "transform_plaintext_scope",
    "transform_hoist_strategy",
    "transform_rotation_set",
]
