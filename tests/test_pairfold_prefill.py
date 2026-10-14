from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.codec import decrypt_complex_rows
from llama3fhe.operators.attention.decode.decode import (
    correct_prefill_emitted_key_c_numpy,
)
from llama3fhe.operators.attention.pairfold import pairfold_prefill
from llama3fhe.operators.attention.pairfold.pairfold_prefill import (
    encrypt_pairfold_plaintext_page,
    prefill_c_to_pairfold_fhe,
    prefill_c_to_pairfold_numpy,
    prefill_c_to_pairfold_rotations,
    prefill_emitted_key_c_direct_group_to_pairfold_fhe,
    prefill_emitted_key_c_direct_to_pairfold_fhe,
    prefill_emitted_key_c_to_pairfold_fhe,
    prefill_real_value_c_group_to_pairfold_fhe,
    prefill_real_value_c_to_pairfold_fhe,
)


def _logical_page(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (128, 8, 128)
    return (
        rng.normal(size=shape).astype(np.float32),
        rng.normal(size=shape).astype(np.float32),
    )


def _graph_emitted_key_c(
    key: np.ndarray, *, attention_layout: AttentionPairLayout
) -> np.ndarray:
    """Reproduce the graph's K_C feature-half expansion before KV emit."""

    folded = attention_layout.pack_key_c_feature_folded(key, scale=1.0)
    half = int(attention_layout.payload_slots) // 2
    lower = np.conjugate(folded[:, :half])
    emitted = np.empty_like(folded)
    emitted[:, :half] = lower
    emitted[:, half:] = 1j * np.conjugate(lower)
    return emitted


def test_prefill_c_numpy_is_slot_exact_pairfold_pack() -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    key, value = _logical_page(26092201)
    raw_key = attention.pack_key_c_feature_folded(key)
    raw_value = attention.pack_value_c(value)

    actual_key, actual_value = prefill_c_to_pairfold_numpy(
        raw_key,
        raw_value,
        attention_layout=attention,
        pairfold_layout=pairfold,
    )

    expected_key = pairfold.pack_key(key)
    expected_value = pairfold.pack_value(value)
    np.testing.assert_array_equal(actual_key, expected_key)
    np.testing.assert_array_equal(actual_value, expected_value)

    # Pin the full-scale V contract at a nontrivial boundary slot.  Unlike Q
    # and raw Prefill K, PairFold V does not carry a 0.5 factor.
    group, pair, lane, feature, token = 1, 1, 1, 63, 127
    head = pairfold.key_value_head(group, pair, lane)
    slot = pairfold.page_slot(pair, feature, token, lane)
    assert actual_value[group, slot] == complex(
        value[token, head, feature],
        value[token, head, feature + pairfold.fold_dim],
    )


def test_prefill_c_numpy_rejects_non_raw_payloads() -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    rows = np.zeros(
        (attention.key_value_pair_count, attention.slots), dtype=np.float64
    )
    with pytest.raises(TypeError, match="folded Prefill K_C must be complex"):
        prefill_c_to_pairfold_numpy(
            rows,
            rows,
            attention_layout=attention,
            pairfold_layout=pairfold,
        )
    with pytest.raises(TypeError, match="Prefill V_C must not be complex"):
        prefill_c_to_pairfold_numpy(
            rows.astype(np.complex128),
            rows.astype(np.complex128),
            attention_layout=attention,
            pairfold_layout=pairfold,
        )


def test_prefill_c_to_pairfold_rotation_contract() -> None:
    layout = PairFold128Layout()
    assert prefill_c_to_pairfold_rotations(layout) == (16384, 131071)


class _FakeCipher:
    def __init__(self, values: np.ndarray, owner: "_FakeContext") -> None:
        self.values = np.asarray(values)
        self.owner = owner
        self.state = SimpleNamespace(cur_limbs=4)
        self.release_calls = 0
        owner.ciphers.append(self)

    def release(self) -> None:
        self.release_calls += 1


class _FakePlain:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class _FakeBundle:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def plaintext(self, name: str, **_kwargs) -> _FakePlain:
        return _FakePlain(self.vectors[name])


class _FakeFhe:
    def __init__(self, owner: "_FakeContext", *, fail_mul_at: int | None) -> None:
        self.owner = owner
        self.fail_mul_at = fail_mul_at
        self.mul_calls = 0

    def homo_rotate(self, cipher, offset: int, _context):  # noqa: ANN001
        if int(offset) == int(self.owner.context.M) - 1:
            values = np.conjugate(cipher.values)
        else:
            values = np.roll(cipher.values, -int(offset))
        return _FakeCipher(values, self.owner)

    def homo_add(self, left, right, _context):  # noqa: ANN001
        return _FakeCipher(left.values + right.values, self.owner)

    def homo_mul_pt(self, cipher, plain, _context):  # noqa: ANN001
        self.mul_calls += 1
        if self.fail_mul_at == self.mul_calls:
            raise RuntimeError("injected plaintext multiply failure")
        return _FakeCipher(cipher.values * plain.values, self.owner)

    def rescale(self, cipher, _context):  # noqa: ANN001
        return _FakeCipher(np.array(cipher.values, copy=True), self.owner)


class _FakeContext:
    def __init__(
        self,
        *,
        slots: int,
        fail_mul_at: int | None = None,
        fail_encrypt_at: int | None = None,
    ) -> None:
        self.max_slots = int(slots)
        self.context = SimpleNamespace(
            M=4 * int(slots),
            scale_at=lambda _limbs: 1.0,
        )
        self.ciphers: list[_FakeCipher] = []
        self.fhe = _FakeFhe(self, fail_mul_at=fail_mul_at)
        self.fail_encrypt_at = fail_encrypt_at
        self.encrypt_calls: list[tuple[int, int]] = []

    def constant_bundle(
        self, vectors: dict[str, np.ndarray], *, cache_key
    ) -> _FakeBundle:
        del cache_key
        return _FakeBundle(vectors)

    def cipher(self, values: np.ndarray) -> _FakeCipher:
        return _FakeCipher(values, self)

    def encrypt(self, values: np.ndarray, *, level: int, slots: int) -> _FakeCipher:
        call_index = len(self.encrypt_calls)
        self.encrypt_calls.append((int(level), int(slots)))
        if call_index == self.fail_encrypt_at:
            raise RuntimeError(f"injected encryption failure at {call_index}")
        return self.cipher(np.asarray(values))


def test_plaintext_page_encrypts_directly_into_owned_pairfold_page() -> None:
    layout = PairFold128Layout()
    key, value = _logical_page(26092205)
    key = key[:3]
    value = value[:3]
    context = _FakeContext(slots=layout.slots)

    page = encrypt_pairfold_plaintext_page(
        key,
        value,
        pairfold_layout=layout,
        crypto_context=context,
        level=2,
    )

    assert page.occupied == 3
    assert context.encrypt_calls == [(2, layout.slots)] * 4
    np.testing.assert_array_equal(
        np.stack([cipher.values for cipher in page.key_groups]),
        layout.pack_key(key),
    )
    np.testing.assert_array_equal(
        np.stack([cipher.values for cipher in page.value_groups]),
        layout.pack_value(value),
    )
    handles = page.handles
    page.release()
    assert all(cipher.release_calls == 1 for cipher in handles)


def test_plaintext_page_encrypt_failure_releases_partial_outputs() -> None:
    layout = PairFold128Layout()
    key, value = _logical_page(26092206)
    context = _FakeContext(slots=layout.slots, fail_encrypt_at=2)

    with pytest.raises(RuntimeError, match="encryption failure at 2"):
        encrypt_pairfold_plaintext_page(
            key[:1],
            value[:1],
            pairfold_layout=layout,
            crypto_context=context,
        )

    assert context.encrypt_calls == [(0, layout.slots)] * 3
    assert len(context.ciphers) == 2
    assert all(cipher.release_calls == 1 for cipher in context.ciphers)


def _patch_fake_imul(monkeypatch: pytest.MonkeyPatch) -> None:
    def multiply_i(cipher, *, crypto_context):  # noqa: ANN001
        return _FakeCipher(1j * cipher.values, crypto_context)

    monkeypatch.setattr(pairfold_prefill, "_multiply_i", multiply_i)


def test_prefill_c_fhe_route_matches_numpy_and_consumes_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    key, value = _logical_page(26092202)
    raw_key = attention.pack_key_c_feature_folded(key)
    raw_value = attention.pack_value_c(value)
    expected_key, expected_value = prefill_c_to_pairfold_numpy(
        raw_key,
        raw_value,
        attention_layout=attention,
        pairfold_layout=pairfold,
    )
    context = _FakeContext(slots=pairfold.slots)
    key_inputs = tuple(context.cipher(row) for row in raw_key)
    value_inputs = tuple(context.cipher(row) for row in raw_value)
    _patch_fake_imul(monkeypatch)

    key_outputs, value_outputs = prefill_c_to_pairfold_fhe(
        key_inputs,
        value_inputs,
        attention_layout=attention,
        pairfold_layout=pairfold,
        crypto_context=context,
    )

    np.testing.assert_array_equal(
        np.stack([cipher.values for cipher in key_outputs]), expected_key
    )
    np.testing.assert_array_equal(
        np.stack([cipher.values for cipher in value_outputs]), expected_value
    )
    assert all(cipher.release_calls == 1 for cipher in key_inputs + value_inputs)
    for cipher in key_outputs + value_outputs:
        release_if_supported(cipher)
    assert all(cipher.release_calls >= 1 for cipher in context.ciphers)


def test_direct_emitted_key_and_value_routes_match_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    key, value = _logical_page(26092207)
    raw_key = _graph_emitted_key_c(key, attention_layout=attention)
    raw_value = attention.pack_value_c(value)
    expected_key = pairfold.pack_key(key)
    expected_value = pairfold.pack_value(value)
    repaired = correct_prefill_emitted_key_c_numpy(
        2.0 * raw_key.real, layout=attention
    )
    np.testing.assert_allclose(repaired, attention.pack_key_c(key), atol=1e-6)
    context = _FakeContext(slots=pairfold.slots)
    key_inputs = tuple(context.cipher(row) for row in raw_key)
    value_inputs = tuple(context.cipher(row) for row in raw_value)
    _patch_fake_imul(monkeypatch)

    actual_key = prefill_emitted_key_c_direct_to_pairfold_fhe(
        key_inputs,
        attention_layout=attention,
        pairfold_layout=pairfold,
        crypto_context=context,
    )
    actual_value = prefill_real_value_c_to_pairfold_fhe(
        value_inputs,
        attention_layout=attention,
        pairfold_layout=pairfold,
        crypto_context=context,
    )
    np.testing.assert_array_equal(np.stack([c.values for c in actual_key]), expected_key)
    np.testing.assert_array_equal(np.stack([c.values for c in actual_value]), expected_value)
    for cipher in actual_key + actual_value:
        release_if_supported(cipher)


def test_grouped_direct_routes_match_their_full_converter_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner routing's one-group converters preserve the full-page contract."""

    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    key, value = _logical_page(26092208)
    raw_key = _graph_emitted_key_c(key, attention_layout=attention)
    raw_value = attention.pack_value_c(value)
    context = _FakeContext(slots=pairfold.slots)
    _patch_fake_imul(monkeypatch)

    full_key = prefill_emitted_key_c_direct_to_pairfold_fhe(
        tuple(context.cipher(row) for row in raw_key),
        attention_layout=attention,
        pairfold_layout=pairfold,
        crypto_context=context,
    )
    full_value = prefill_real_value_c_to_pairfold_fhe(
        tuple(context.cipher(row) for row in raw_value),
        attention_layout=attention,
        pairfold_layout=pairfold,
        crypto_context=context,
    )
    for group in range(pairfold.key_value_groups):
        grouped_key = prefill_emitted_key_c_direct_group_to_pairfold_fhe(
            tuple(context.cipher(row) for row in raw_key[group * pairfold.pair_blocks : (group + 1) * pairfold.pair_blocks]),
            pairfold_layout=pairfold,
            crypto_context=context,
        )
        grouped_value = prefill_real_value_c_group_to_pairfold_fhe(
            tuple(context.cipher(row) for row in raw_value[group * pairfold.pair_blocks : (group + 1) * pairfold.pair_blocks]),
            pairfold_layout=pairfold,
            crypto_context=context,
        )
        np.testing.assert_array_equal(grouped_key.values, full_key[group].values)
        np.testing.assert_array_equal(grouped_value.values, full_value[group].values)
        release_if_supported(grouped_key)
        release_if_supported(grouped_value)
    for cipher in full_key + full_value:
        release_if_supported(cipher)


@pytest.mark.parametrize("occupied", [1, 2, 127])
def test_prefill_c_fhe_partial_page_masks_suffix_without_extra_level(
    monkeypatch: pytest.MonkeyPatch, occupied: int
) -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    key, value = _logical_page(26092202)
    raw_key = attention.pack_key_c_feature_folded(key)
    raw_value = attention.pack_value_c(value)
    expected_key, expected_value = prefill_c_to_pairfold_numpy(
        raw_key,
        raw_value,
        attention_layout=attention,
        pairfold_layout=pairfold,
    )
    for expected in (expected_key, expected_value):
        expected.reshape(2, 2, 64, 128, 2)[:, :, :, occupied:, :] = 0.0
    context = _FakeContext(slots=pairfold.slots)
    key_inputs = tuple(context.cipher(row) for row in raw_key)
    value_inputs = tuple(context.cipher(row) for row in raw_value)
    _patch_fake_imul(monkeypatch)

    key_outputs, value_outputs = prefill_c_to_pairfold_fhe(
        key_inputs,
        value_inputs,
        attention_layout=attention,
        pairfold_layout=pairfold,
        crypto_context=context,
        occupied=occupied,
    )

    np.testing.assert_array_equal(
        np.stack([cipher.values for cipher in key_outputs]), expected_key
    )
    np.testing.assert_array_equal(
        np.stack([cipher.values for cipher in value_outputs]), expected_value
    )
    assert all(cipher.release_calls == 1 for cipher in key_inputs + value_inputs)
    for cipher in key_outputs + value_outputs:
        release_if_supported(cipher)


def test_prefill_c_fhe_failure_releases_inputs_and_partial_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    key, value = _logical_page(26092203)
    raw_key = attention.pack_key_c_feature_folded(key)
    raw_value = attention.pack_value_c(value)
    context = _FakeContext(slots=pairfold.slots, fail_mul_at=5)
    key_inputs = tuple(context.cipher(row) for row in raw_key)
    value_inputs = tuple(context.cipher(row) for row in raw_value)
    _patch_fake_imul(monkeypatch)

    with pytest.raises(RuntimeError, match="injected plaintext multiply failure"):
        prefill_c_to_pairfold_fhe(
            key_inputs,
            value_inputs,
            attention_layout=attention,
            pairfold_layout=pairfold,
            crypto_context=context,
        )

    assert all(cipher.release_calls >= 1 for cipher in context.ciphers)
    assert all(cipher.release_calls == 1 for cipher in key_inputs + value_inputs)


@pytest.mark.gpu
def test_prefill_c_to_pairfold_fhe_matches_oracle_on_gpu() -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    key, value = _logical_page(26092204)
    key *= 0.02
    value *= 0.02
    raw_key = attention.pack_key_c_feature_folded(key)
    raw_value = attention.pack_value_c(value)
    expected_key, expected_value = prefill_c_to_pairfold_numpy(
        raw_key,
        raw_value,
        attention_layout=attention,
        pairfold_layout=pairfold,
    )

    config = Llama3CKKSConfig()
    config.simulator.logN = 16
    config.simulator.maxLevelsRemaining = 3
    config.simulator.dnum = 1
    config.simulator.device = "cuda"
    context = create_sim_context(
        config.simulator,
        input_dim=1,
        block_out=1,
        rotations=prefill_c_to_pairfold_rotations(pairfold),
        include_default_rotations=False,
    )
    key_inputs = tuple(
        context.encrypt(row, level=0, slots=pairfold.slots) for row in raw_key
    )
    value_inputs = tuple(
        context.encrypt(row, level=0, slots=pairfold.slots) for row in raw_value
    )
    key_outputs = value_outputs = ()
    transferred = False
    try:
        transferred = True
        key_outputs, value_outputs = prefill_c_to_pairfold_fhe(
            key_inputs,
            value_inputs,
            attention_layout=attention,
            pairfold_layout=pairfold,
            crypto_context=context,
        )
        actual_key = decrypt_complex_rows(
            key_outputs, crypto_context=context
        )
        actual_value = decrypt_complex_rows(
            value_outputs, crypto_context=context
        )
        assert {
            int(context.level_for_cipher(cipher))
            for cipher in key_outputs + value_outputs
        } == {1}
        assert {
            int(cipher.state.cur_limbs)
            for cipher in key_outputs + value_outputs
        } == {int(context.context.L) - 1}
        assert {
            int(cipher.state.scale_degree)
            for cipher in key_outputs + value_outputs
        } == {1}
        np.testing.assert_allclose(
            actual_key, expected_key, atol=3e-3, rtol=3e-3
        )
        np.testing.assert_allclose(
            actual_value, expected_value, atol=3e-3, rtol=3e-3
        )
    finally:
        if not transferred:
            for cipher in key_inputs + value_inputs:
                release_if_supported(cipher)
        for cipher in key_outputs + value_outputs:
            release_if_supported(cipher)


@pytest.mark.gpu
def test_graph_emitted_direct_key_matches_legacy_on_gpu() -> None:
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    key, _ = _logical_page(26092209)
    key *= 0.02
    emitted = _graph_emitted_key_c(key, attention_layout=attention)
    expected = pairfold.pack_key(key)

    config = Llama3CKKSConfig()
    config.simulator.logN = 16
    config.simulator.maxLevelsRemaining = 4
    config.simulator.dnum = 1
    config.simulator.device = "cuda"
    context = create_sim_context(
        config.simulator,
        input_dim=1,
        block_out=1,
        rotations=prefill_c_to_pairfold_rotations(pairfold),
        include_default_rotations=False,
    )
    direct_inputs = tuple(
        context.encrypt(row, level=0, slots=pairfold.slots) for row in emitted
    )
    legacy_inputs = tuple(
        context.encrypt(row, level=0, slots=pairfold.slots) for row in emitted
    )
    direct_outputs = legacy_outputs = ()
    direct_transferred = legacy_transferred = False
    try:
        direct_transferred = True
        direct_outputs = prefill_emitted_key_c_direct_to_pairfold_fhe(
            direct_inputs,
            attention_layout=attention,
            pairfold_layout=pairfold,
            crypto_context=context,
        )
        legacy_transferred = True
        legacy_outputs = prefill_emitted_key_c_to_pairfold_fhe(
            legacy_inputs,
            attention_layout=attention,
            pairfold_layout=pairfold,
            crypto_context=context,
        )
        actual_direct = decrypt_complex_rows(direct_outputs, crypto_context=context)
        actual_legacy = decrypt_complex_rows(legacy_outputs, crypto_context=context)
        assert {context.level_for_cipher(c) for c in direct_outputs} == {1}
        assert {context.level_for_cipher(c) for c in legacy_outputs} == {2}
        np.testing.assert_allclose(actual_legacy, expected, atol=3e-3, rtol=3e-3)
        np.testing.assert_allclose(actual_direct, expected, atol=3e-3, rtol=3e-3)
        np.testing.assert_allclose(
            actual_direct, actual_legacy, atol=3e-3, rtol=3e-3
        )
    finally:
        if not direct_transferred:
            for cipher in direct_inputs:
                release_if_supported(cipher)
        if not legacy_transferred:
            for cipher in legacy_inputs:
                release_if_supported(cipher)
        for cipher in direct_outputs + legacy_outputs:
            release_if_supported(cipher)
