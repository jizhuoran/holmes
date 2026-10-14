"""transform_plaintext_scope: one encode per (read-only rows, limbs, ext) inside a scope."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from llama3fhe.operators import transform
from llama3fhe.operators.transform import encode_transform_rows, transform_plaintext_scope


class _Bundle:
    def __init__(self, log):
        self.log = log

    def plaintext(self, name, **kwargs):
        self.log.append((name, kwargs["state"].cur_limbs, kwargs["is_ext"]))
        return object()


def _fake_context(log):
    return SimpleNamespace(
        context=SimpleNamespace(L=20, scale_at=lambda limbs: 2.0**40),
        max_slots=16,
        device="cpu",
        constant_bundle=lambda vectors, cache_key: _Bundle(log),
    )


def test_memo_shares_encodes_inside_scope_only(monkeypatch) -> None:
    monkeypatch.setattr(transform, "_MEMO_ENABLED", True)
    log: list = []
    ctx = _fake_context(log)
    rows = np.zeros((4, 16))
    rows.setflags(write=False)
    writable = np.zeros((4, 16))

    # outside a scope: every call encodes
    a = encode_transform_rows(rows, name="x", level=10, crypto_context=ctx)
    b = encode_transform_rows(rows, name="x", level=10, crypto_context=ctx)
    assert a is not b and len(log) == 2

    with transform_plaintext_scope() as memo:
        c = encode_transform_rows(rows, name="s0", level=10, crypto_context=ctx)
        d = encode_transform_rows(rows, name="s1", level=10, crypto_context=ctx)
        assert c is d and len(log) == 3                    # same rows, limbs, ext: one encode
        e = encode_transform_rows(rows, name="s2", level=11, crypto_context=ctx)
        assert e is not c and len(log) == 4                # other level: new encode
        f = encode_transform_rows(rows, name="s3", level=10, crypto_context=ctx, is_ext=False)
        assert f is not c and len(log) == 5                # other basis: new encode
        g = encode_transform_rows(writable, name="w", level=10, crypto_context=ctx)
        h = encode_transform_rows(writable, name="w", level=10, crypto_context=ctx)
        assert g is not h and len(log) == 7                # writable rows are never memoised
        assert memo.hits == 1 and memo.misses == 3
    assert transform._active_memo() is None

    # after the scope: encodes again
    encode_transform_rows(rows, name="s0", level=10, crypto_context=ctx)
    assert len(log) == 8


def test_memo_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setattr(transform, "_MEMO_ENABLED", False)
    log: list = []
    ctx = _fake_context(log)
    rows = np.zeros((4, 16))
    rows.setflags(write=False)
    with transform_plaintext_scope() as memo:
        encode_transform_rows(rows, name="s0", level=10, crypto_context=ctx)
        encode_transform_rows(rows, name="s1", level=10, crypto_context=ctx)
    assert len(log) == 2 and memo.hits == 0
