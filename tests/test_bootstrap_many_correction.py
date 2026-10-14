"""The batched Meta-BTS routine matches the single-cipher one step for step (fake arithmetic)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from llama3fhe.operators.nonlinear import bootstrap as bmod


@dataclass(frozen=True)
class State:
    cur_limbs: int
    scale_degree: int = 1
    scaling_factor: float | None = None

    def replace(self, **kw):
        return replace(self, **kw)


class Cipher:
    def __init__(self, value: float, limbs: int, trace: list[str]):
        self.value, self.state, self.trace = value, State(limbs), trace

    def deep_copy(self):
        self.trace.append("copy")
        return Cipher(self.value, self.state.cur_limbs, self.trace)


class FakeFhe:
    """Arithmetic on the plaintext value; the bootstrap adds a deterministic error and resets the limbs."""

    def align_to(self, c, state, ctx):
        c.trace.append(f"align{state.cur_limbs}")
        return Cipher(c.value, state.cur_limbs, c.trace)

    def homo_sub(self, a, b, ctx):
        a.trace.append("sub")
        return Cipher(a.value - b.value, min(a.state.cur_limbs, b.state.cur_limbs), a.trace)

    def encode_scalar(self, value, **kw):
        return value

    def homo_mul_scalar(self, c, v, ctx):
        c.trace.append("mulint")
        return Cipher(c.value * v, c.state.cur_limbs, c.trace)

    def homo_mul_scalar_rescale(self, c, v, ctx):
        c.trace.append("muldbl")
        return Cipher(c.value * v, c.state.cur_limbs - 1, c.trace)


def fake_bootstrap(c, ctx, program):
    c.trace.append("bs")
    return Cipher(c.value + 1e-3 * (1 + c.value), 17, c.trace)


@pytest.mark.parametrize("precision_bits", [0, 8])
def test_batched_two_pass_matches_single(monkeypatch, precision_bits) -> None:
    fake = FakeFhe()
    monkeypatch.setattr(bmod, "fhe", fake)
    monkeypatch.setattr(bmod.bs, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(bmod, "release_if_supported", lambda c: None)
    values = [0.25, -0.5, 0.75]

    single_trace: list[str] = []
    sp = [0]
    singles = [
        bmod._bootstrap_with_optional_correction(
            Cipher(v, 2, single_trace), crypto_context=None, program=None, iterations=2,
            precision_bits=precision_bits, native_passes=sp)
        for v in values
    ]
    batched_trace: list[str] = []
    calls: list[int] = []

    def many(items):
        calls.append(len(items))
        return [fake_bootstrap(c, None, None) for c in items]

    bp = [0]
    batched = bmod._bootstrap_many_with_optional_correction(
        [Cipher(v, 2, batched_trace) for v in values], crypto_context=None, program=None, iterations=2,
        precision_bits=precision_bits, native_passes=bp, bootstrap_many=many)

    assert [c.value for c in batched] == pytest.approx([c.value for c in singles], rel=0, abs=0)
    assert [c.state.cur_limbs for c in batched] == [c.state.cur_limbs for c in singles]
    assert calls == [3, 3]                      # two batched passes, not two per cipher
    assert bp[0] == sp[0] == 6
    assert sorted(batched_trace) == sorted(single_trace)   # same per-cipher operations


def test_one_pass_is_a_single_batched_call() -> None:
    calls = []
    out = bmod._bootstrap_many_with_optional_correction(
        [1, 2, 3], crypto_context=None, program=None, iterations=1, precision_bits=8, native_passes=(n := [0]),
        bootstrap_many=lambda items: (calls.append(len(items)), [i * 10 for i in items])[1])
    assert out == [10, 20, 30] and calls == [3] and n[0] == 3
