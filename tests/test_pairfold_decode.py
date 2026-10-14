from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.pairfold.pairfold128 import (
    pv_pairfold128_numpy,
    qk_pairfold128_numpy,
)
from llama3fhe.operators.attention.pairfold.pairfold_decode import (
    add_pairfold_output_contributions_fhe,
    pairfold_decode_complexity,
    pairfold_decode_rotations,
    pv_pairfold_pages_fhe,
    pv_pairfold_prefix_fhe,
    qk_pairfold_pages_fhe,
    qk_pairfold_pages_from_queries_fhe,
    qk_pairfold_prefix_from_queries_fhe,
    validate_pairfold_score_compatibility,
)


class _FakeState:
    def __init__(self, cur_limbs: int) -> None:
        self.cur_limbs = int(cur_limbs)
        self.scale_degree = 1
        self.scaling_factor = 1.0


class _FakeCipher:
    def __init__(
        self,
        values: np.ndarray,
        owner: _FakeContext,
        *,
        cur_limbs: int,
        label: str,
    ) -> None:
        self.values: np.ndarray | None = np.asarray(values, dtype=np.complex128)
        self.owner = owner
        self.state = _FakeState(cur_limbs)
        self.slots = int(self.values.size)
        self.label = str(label)
        self.release_count = 0
        owner.ciphers.append(self)

    def deep_copy(self) -> _FakeCipher:
        assert self.values is not None
        self.owner.counts["deep_copy"] += 1
        return _FakeCipher(
            np.array(self.values, copy=True),
            self.owner,
            cur_limbs=self.state.cur_limbs,
            label=f"copy({self.label})",
        )

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"cipher released twice: {self.label}")
        self.release_count = 1
        self.values = None


class _FakePlain:
    def __init__(self, values: np.ndarray, owner: _FakeContext, *, label: str) -> None:
        self.values = np.asarray(values)
        self.owner = owner
        self.label = str(label)
        self.release_count = 0
        owner.plains.append(self)

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"plaintext released twice: {self.label}")
        self.release_count = 1


class _FakeFhe:
    def __init__(
        self,
        owner: _FakeContext,
        *,
        fail_mul_relin_at: int | None,
    ) -> None:
        self.owner = owner
        self.fail_mul_relin_at = fail_mul_relin_at

    @staticmethod
    def _values(cipher: _FakeCipher) -> np.ndarray:
        assert cipher.values is not None, f"use after release: {cipher.label}"
        return cipher.values

    def homo_rotate(
        self, cipher: _FakeCipher, offset: int, native_context: object
    ) -> _FakeCipher:
        assert native_context is self.owner.context
        offset = int(offset)
        self.owner.counts["rotate"] += 1
        self.owner.rotations.append(offset)
        values = self._values(cipher)
        if offset == int(self.owner.context.M) - 1:
            result = np.conjugate(values)
        else:
            result = np.roll(values, -offset)
        return _FakeCipher(
            result,
            self.owner,
            cur_limbs=cipher.state.cur_limbs,
            label=f"rot({offset},{cipher.label})",
        )

    def homo_add(
        self,
        left: _FakeCipher,
        right: _FakeCipher,
        native_context: object,
    ) -> _FakeCipher:
        assert native_context is self.owner.context
        self.owner.counts["add"] += 1
        assert left.state.cur_limbs == right.state.cur_limbs
        return _FakeCipher(
            self._values(left) + self._values(right),
            self.owner,
            cur_limbs=left.state.cur_limbs,
            label=f"add({left.label},{right.label})",
        )

    def homo_mul_relin(
        self,
        left: _FakeCipher,
        right: _FakeCipher,
        native_context: object,
    ) -> _FakeCipher:
        assert native_context is self.owner.context
        call = int(self.owner.counts["mul_relin"])
        self.owner.counts["mul_relin"] += 1
        if self.fail_mul_relin_at == call:
            raise RuntimeError(f"injected mul_relin failure at {call}")
        assert left.state.cur_limbs == right.state.cur_limbs
        return _FakeCipher(
            self._values(left) * self._values(right),
            self.owner,
            cur_limbs=left.state.cur_limbs,
            label=f"mul({left.label},{right.label})",
        )

    def homo_mul_pt(
        self,
        cipher: _FakeCipher,
        plain: _FakePlain,
        native_context: object,
    ) -> _FakeCipher:
        assert native_context is self.owner.context
        self.owner.counts["mul_plain"] += 1
        return _FakeCipher(
            self._values(cipher) * plain.values,
            self.owner,
            cur_limbs=cipher.state.cur_limbs,
            label=f"mul_pt({cipher.label},{plain.label})",
        )

    def rescale(
        self, cipher: _FakeCipher, native_context: object
    ) -> _FakeCipher:
        assert native_context is self.owner.context
        self.owner.counts["rescale"] += 1
        return _FakeCipher(
            np.array(self._values(cipher), copy=True),
            self.owner,
            cur_limbs=cipher.state.cur_limbs - 1,
            label=f"rescale({cipher.label})",
        )

    def align_to(
        self,
        cipher: _FakeCipher,
        state: _FakeState,
        native_context: object,
    ) -> _FakeCipher:
        assert native_context is self.owner.context
        if int(cipher.state.cur_limbs) == int(state.cur_limbs):
            return cipher
        if int(cipher.state.cur_limbs) < int(state.cur_limbs):
            raise AssertionError("fake backend refuses a modulus-level increase")
        self.owner.counts["align"] += 1
        return _FakeCipher(
            np.array(self._values(cipher), copy=True),
            self.owner,
            cur_limbs=state.cur_limbs,
            label=f"align({cipher.label})",
        )


class _FakeContext:
    def __init__(
        self,
        *,
        slots: int,
        fail_mul_relin_at: int | None = None,
    ) -> None:
        self.max_slots = int(slots)
        # The production logN=16 CKKS context has M=4*slots, not 2*slots.
        self.context = SimpleNamespace(M=4 * int(slots), L=12)
        self.ciphers: list[_FakeCipher] = []
        self.plains: list[_FakePlain] = []
        self.counts: Counter[str] = Counter()
        self.rotations: list[int] = []
        self.fhe = _FakeFhe(
            self, fail_mul_relin_at=fail_mul_relin_at
        )

    def cipher(
        self,
        values: np.ndarray,
        *,
        cur_limbs: int = 10,
        label: str = "input",
    ) -> _FakeCipher:
        return _FakeCipher(
            values,
            self,
            cur_limbs=cur_limbs,
            label=label,
        )

    def level_for_cipher(self, cipher: _FakeCipher) -> int:
        return int(self.context.L) - int(cipher.state.cur_limbs)

    def plaintext(
        self,
        values: np.ndarray,
        *,
        name: str,
        level: int,
        slots: int,
        dtype,
        is_ext: bool,
    ) -> _FakePlain:
        del level, is_ext
        assert int(slots) == self.max_slots
        return _FakePlain(
            np.asarray(values, dtype=dtype), self, label=name
        )


class _StreamingQueryExtractor:
    """Owned-query provider that asserts the previous Q was released first."""

    def __init__(
        self,
        context: _FakeContext,
        packed_query: np.ndarray,
        carrier: _FakeCipher,
    ) -> None:
        self.context = context
        self.packed_query = np.asarray(packed_query)
        self.carrier = carrier
        self.outputs: list[_FakeCipher] = []
        self.calls = 0
        self.max_live = 0

    def __call__(
        self,
        carrier: _FakeCipher,
        *,
        group: int,
        query_offset: int,
    ) -> _FakeCipher:
        assert carrier is self.carrier
        live = sum(cipher.release_count == 0 for cipher in self.outputs)
        # The production kernel must retire one route before requesting next.
        assert live == 0
        index = int(group) * 4 + int(query_offset)
        output = self.context.cipher(
            self.packed_query[index],
            label=f"stream_q.{group}.{query_offset}",
        )
        self.outputs.append(output)
        self.calls += 1
        self.max_live = max(self.max_live, live + 1)
        return output


def _borrowed_ids(ciphers) -> set[int]:
    return {id(cipher) for cipher in ciphers}


def _assert_borrowed_and_temporaries_clean(
    context: _FakeContext, borrowed: set[int]
) -> None:
    for cipher in context.ciphers:
        if id(cipher) in borrowed:
            assert cipher.release_count == 0, cipher.label
        else:
            assert cipher.release_count == 1, cipher.label
    assert all(plain.release_count == 1 for plain in context.plains)


def _release_inputs(ciphers) -> None:
    for cipher in ciphers:
        release_if_supported(cipher)


def test_pairfold_decode_contract_rotation_union_and_complexity() -> None:
    layout = PairFold128Layout()
    validate_pairfold_score_compatibility(layout, AttentionPairLayout())
    rotations = set(pairfold_decode_rotations(layout))
    assert 4 * layout.slots - 1 in rotations
    assert 2 * layout.slots - 1 not in rotations
    assert set(layout.token_rotations()) <= rotations
    assert {-step for step in layout.token_rotations()} <= rotations
    assert set(layout.feature_rotations()) <= rotations
    assert {-step for step in layout.feature_rotations()} <= rotations

    complexity = pairfold_decode_complexity(2, prefix_plaintext=True)
    assert complexity.qk_cipher_multiplications == 16
    assert complexity.pv_cipher_multiplications == 16
    assert complexity.prefix_plain_multiplications == 16
    assert complexity.query_broadcasts == 8
    assert complexity.query_peak_live_ciphers == 1
    assert complexity.qk_feature_reductions == 24
    assert complexity.pv_token_reductions == 8


@pytest.mark.parametrize(
    "occupied_pages",
    ((1,), (128, 1)),
    ids=("one_page_boundary", "two_pages_full_plus_one"),
)
def test_qk_pairfold_pages_matches_oracle_and_masks_full_slots(
    occupied_pages: tuple[int, ...],
) -> None:
    rng = np.random.default_rng(26092211 + len(occupied_pages))
    layout = PairFold128Layout()
    query = rng.normal(
        scale=0.2, size=(layout.query_heads, layout.head_dim)
    )
    packed_query = layout.pack_query(query)
    logical_keys = [
        rng.normal(
            scale=0.2,
            size=(layout.seq_len, layout.key_value_heads, layout.head_dim),
        )
        for _ in occupied_pages
    ]
    # Invalid cache columns are deliberately enormous.  They must not leak
    # into either valid scores or any inactive ciphertext slot.
    for key, occupied in zip(logical_keys, occupied_pages, strict=True):
        key[occupied:] = 1.0e4
    packed_keys = [layout.pack_key(key) for key in logical_keys]

    context = _FakeContext(slots=layout.slots)
    query_ciphers = tuple(
        context.cipher(row, label=f"q{index}")
        for index, row in enumerate(packed_query)
    )
    key_pages = tuple(
        tuple(
            context.cipher(row, label=f"k{page}.{group}")
            for group, row in enumerate(packed)
        )
        for page, packed in enumerate(packed_keys)
    )
    inputs = query_ciphers + tuple(
        cipher for page in key_pages for cipher in page
    )
    borrowed = _borrowed_ids(inputs)
    profile: dict[str, float] = {}

    outputs = qk_pairfold_pages_from_queries_fhe(
        query_ciphers,
        key_pages,
        layout=layout,
        crypto_context=context,
        occupied_pages=occupied_pages,
        profile=profile,
    )

    assert len(outputs) == len(occupied_pages)
    for output, packed_key, occupied in zip(
        outputs, packed_keys, occupied_pages, strict=True
    ):
        expected = qk_pairfold128_numpy(
            packed_query,
            packed_key,
            layout=layout,
            occupied=occupied,
        )[0]
        assert output[0].values is not None
        np.testing.assert_allclose(
            output[0].values, expected, atol=2e-12, rtol=2e-12
        )
        inactive = np.ones((layout.slots,), dtype=bool)
        inactive[np.flatnonzero(expected)] = False
        np.testing.assert_array_equal(output[0].values[inactive], 0.0)

    assert context.counts["mul_relin"] == 8 * len(occupied_pages)
    assert context.rotations.count(4 * layout.slots - 1) == (
        8 * len(occupied_pages)
    )
    # Query broadcast is shared across pages.
    for step in layout.token_rotations():
        assert context.rotations.count(-step) == 8
    assert profile["ct_ct"] == 8.0 * len(occupied_pages)
    score_masks = [
        plain
        for plain in context.plains
        if plain.label.startswith("pairfold.qk.score")
    ]
    # The source mask depends only on page occupancy and pair, not on the
    # GQA stream.  It should be encoded once and reused by all eight streams.
    assert len(score_masks) == 2 * len(set(occupied_pages))

    for page in outputs:
        _release_inputs(page)
    _assert_borrowed_and_temporaries_clean(context, borrowed)
    _release_inputs(inputs)


def test_pv_pairfold_two_pages_accumulates_before_one_token_reduction() -> None:
    rng = np.random.default_rng(26092214)
    layout = PairFold128Layout()
    occupied_pages = (128, 3)
    logical_probabilities = []
    logical_values = []
    for occupied in occupied_pages:
        probability = rng.uniform(
            0.0, 0.02, size=(layout.query_heads, layout.seq_len)
        )
        probability[:, occupied:] = 777.0
        value = rng.normal(
            scale=0.2,
            size=(layout.seq_len, layout.key_value_heads, layout.head_dim),
        )
        value[occupied:] = 1.0e4
        logical_probabilities.append(probability)
        logical_values.append(value)
    packed_probabilities = [
        layout.pack_scores(probability)
        for probability in logical_probabilities
    ]
    packed_values = [layout.pack_value(value) for value in logical_values]
    expected = sum(
        (
            pv_pairfold128_numpy(
                probability,
                value,
                layout=layout,
                occupied=occupied,
            )
            for probability, value, occupied in zip(
                packed_probabilities,
                packed_values,
                occupied_pages,
                strict=True,
            )
        ),
        start=np.zeros((1, layout.slots), dtype=np.complex128),
    )[0]

    context = _FakeContext(slots=layout.slots)
    probability_pages = tuple(
        (context.cipher(row[0], cur_limbs=10, label=f"p{index}"),)
        for index, row in enumerate(packed_probabilities)
    )
    # Different page levels force both operand and cross-page alignment.
    value_pages = tuple(
        tuple(
            context.cipher(
                row,
                cur_limbs=9 - page,
                label=f"v{page}.{group}",
            )
            for group, row in enumerate(packed)
        )
        for page, packed in enumerate(packed_values)
    )
    inputs = tuple(
        cipher for page in probability_pages + value_pages for cipher in page
    )
    borrowed = _borrowed_ids(inputs)
    profile: dict[str, float] = {}

    (output,) = pv_pairfold_pages_fhe(
        probability_pages,
        value_pages,
        layout=layout,
        crypto_context=context,
        occupied_pages=occupied_pages,
        profile=profile,
    )

    assert output.values is not None
    np.testing.assert_allclose(output.values, expected, atol=3e-12, rtol=3e-12)
    inactive = np.ones((layout.slots,), dtype=bool)
    inactive[np.flatnonzero(expected)] = False
    np.testing.assert_array_equal(output.values[inactive], 0.0)
    assert context.counts["mul_relin"] == 16
    assert context.counts["align"] > 0
    # Eight (g,r) units, not eight per content page.
    for step in layout.token_rotations():
        assert context.rotations.count(step) == 8
    assert profile["ct_ct"] == 16.0
    assert profile["token_reductions"] == 8.0

    release_if_supported(output)
    _assert_borrowed_and_temporaries_clean(context, borrowed)
    _release_inputs(inputs)


def test_pv_pairfold_single_page_transfers_product_without_copy() -> None:
    """A one-page serial Decode PV must not deep-copy its sole product."""

    rng = np.random.default_rng(26092213)
    layout = PairFold128Layout()
    occupied = 7
    probabilities = rng.uniform(
        0.0, 0.02, size=(layout.query_heads, layout.seq_len)
    )
    probabilities[:, occupied:] = 999.0
    values = rng.normal(
        scale=0.2,
        size=(layout.seq_len, layout.key_value_heads, layout.head_dim),
    )
    values[occupied:] = 1.0e4
    packed_probability = layout.pack_scores(probabilities)
    packed_value = layout.pack_value(values)
    expected = pv_pairfold128_numpy(
        packed_probability,
        packed_value,
        layout=layout,
        occupied=occupied,
    )[0]

    context = _FakeContext(slots=layout.slots)
    probability = context.cipher(packed_probability[0], label="probability")
    value_ciphers = tuple(
        context.cipher(row, label=f"v{group}")
        for group, row in enumerate(packed_value)
    )
    inputs = (probability, *value_ciphers)
    borrowed = _borrowed_ids(inputs)

    (output,) = pv_pairfold_pages_fhe(
        ((probability,),),
        (value_ciphers,),
        layout=layout,
        crypto_context=context,
        occupied_pages=(occupied,),
    )

    assert output.values is not None
    np.testing.assert_allclose(output.values, expected, atol=3e-12, rtol=3e-12)
    assert context.counts["deep_copy"] == 0
    release_if_supported(output)
    _assert_borrowed_and_temporaries_clean(context, borrowed)
    _release_inputs(inputs)


def test_pairfold_plaintext_prefix_qk_and_pv_match_oracles() -> None:
    rng = np.random.default_rng(26092215)
    layout = PairFold128Layout()
    prefix_k = 3
    query = rng.normal(
        scale=0.15, size=(layout.query_heads, layout.head_dim)
    )
    key = rng.normal(
        scale=0.15,
        size=(prefix_k, layout.key_value_heads, layout.head_dim),
    )
    value = rng.normal(scale=0.15, size=key.shape)
    packed_query = layout.pack_query(query)
    packed_key = layout.pack_key(key)
    packed_value = layout.pack_value(value)

    context = _FakeContext(slots=layout.slots)
    query_ciphers = tuple(
        context.cipher(row, label=f"q{index}")
        for index, row in enumerate(packed_query)
    )
    qk_profile: dict[str, float] = {}
    (score,) = qk_pairfold_prefix_from_queries_fhe(
        query_ciphers,
        key,
        prefix_k=prefix_k,
        layout=layout,
        crypto_context=context,
        profile=qk_profile,
    )
    expected_score = qk_pairfold128_numpy(
        packed_query,
        packed_key,
        layout=layout,
        occupied=prefix_k,
    )[0]
    assert score.values is not None
    np.testing.assert_allclose(
        score.values, expected_score, atol=2e-12, rtol=2e-12
    )
    assert qk_profile["ct_pt"] == 8.0

    probabilities = rng.uniform(
        0.0, 0.1, size=(layout.query_heads, layout.seq_len)
    )
    probabilities[:, prefix_k:] = 999.0
    packed_probability = layout.pack_scores(probabilities)
    probability = context.cipher(packed_probability[0], label="probability")
    pv_profile: dict[str, float] = {}
    (prefix_output,) = pv_pairfold_prefix_fhe(
        (probability,),
        value,
        prefix_k=prefix_k,
        layout=layout,
        crypto_context=context,
        profile=pv_profile,
    )
    expected_output = pv_pairfold128_numpy(
        packed_probability,
        packed_value,
        layout=layout,
        occupied=prefix_k,
    )[0]
    assert prefix_output.values is not None
    np.testing.assert_allclose(
        prefix_output.values, expected_output, atol=3e-12, rtol=3e-12
    )
    assert pv_profile["ct_pt"] == 8.0
    assert context.counts["mul_relin"] == 0

    # Prefix/content contribution API is a borrowed, level-aligned add.
    (doubled,) = add_pairfold_output_contributions_fhe(
        (prefix_output, prefix_output), crypto_context=context
    )
    assert doubled.values is not None
    np.testing.assert_allclose(doubled.values, 2.0 * expected_output)

    inputs = query_ciphers + (probability,)
    borrowed = _borrowed_ids(inputs)
    for cipher in (score, prefix_output, doubled):
        release_if_supported(cipher)
    _assert_borrowed_and_temporaries_clean(context, borrowed)
    _release_inputs(inputs)


def test_prefix_and_content_share_q_broadcast_and_pv_token_reduction() -> None:
    rng = np.random.default_rng(26092216)
    layout = PairFold128Layout()
    prefix_k = 2
    occupied = 1
    query = rng.normal(
        scale=0.1, size=(layout.query_heads, layout.head_dim)
    )
    prefix_key = rng.normal(
        scale=0.1,
        size=(prefix_k, layout.key_value_heads, layout.head_dim),
    )
    content_key = rng.normal(
        scale=0.1,
        size=(layout.seq_len, layout.key_value_heads, layout.head_dim),
    )
    packed_query = layout.pack_query(query)
    packed_prefix_key = layout.pack_key(prefix_key)
    packed_content_key = layout.pack_key(content_key)

    qk_context = _FakeContext(slots=layout.slots)
    carrier = qk_context.cipher(
        np.zeros((layout.slots,), dtype=np.complex128), label="carrier"
    )
    extractor = _StreamingQueryExtractor(qk_context, packed_query, carrier)
    key_ciphers = tuple(
        qk_context.cipher(row, label=f"k{index}")
        for index, row in enumerate(packed_content_key)
    )
    qk_profile: dict[str, float] = {}
    score_pages = qk_pairfold_pages_fhe(
        carrier,
        (key_ciphers,),
        query_extractor=extractor,
        layout=layout,
        crypto_context=qk_context,
        occupied_pages=(occupied,),
        key_prefix=prefix_key,
        prefix_k=prefix_k,
        profile=qk_profile,
    )
    assert len(score_pages) == 2
    expected_scores = (
        qk_pairfold128_numpy(
            packed_query,
            packed_prefix_key,
            layout=layout,
            occupied=prefix_k,
        )[0],
        qk_pairfold128_numpy(
            packed_query,
            packed_content_key,
            layout=layout,
            occupied=occupied,
        )[0],
    )
    for page, expected in zip(score_pages, expected_scores, strict=True):
        assert page[0].values is not None
        np.testing.assert_allclose(page[0].values, expected, atol=2e-12)
    for step in layout.token_rotations():
        assert qk_context.rotations.count(-step) == 8
    assert qk_context.counts["mul_relin"] == 8
    assert qk_profile["data_ct_pt"] == 8.0
    assert qk_profile["query_peak_live"] == 1.0
    assert qk_profile["query_peak_live_ciphers"] == 1.0
    assert qk_profile["query_extract_count"] == 8.0
    assert extractor.calls == 8
    assert extractor.max_live == 1
    assert all(cipher.release_count == 1 for cipher in extractor.outputs)
    qk_inputs = (carrier, *key_ciphers)
    qk_borrowed = _borrowed_ids(qk_inputs)
    for page in score_pages:
        _release_inputs(page)
    _assert_borrowed_and_temporaries_clean(qk_context, qk_borrowed)
    _release_inputs(qk_inputs)

    prefix_probability = rng.uniform(
        0.0, 0.1, size=(layout.query_heads, layout.seq_len)
    )
    prefix_probability[:, prefix_k:] = 123.0
    content_probability = rng.uniform(
        0.0, 0.1, size=(layout.query_heads, layout.seq_len)
    )
    content_probability[:, occupied:] = 456.0
    prefix_value = rng.normal(
        scale=0.1,
        size=(prefix_k, layout.key_value_heads, layout.head_dim),
    )
    content_value = rng.normal(
        scale=0.1,
        size=(layout.seq_len, layout.key_value_heads, layout.head_dim),
    )
    packed_prefix_probability = layout.pack_scores(prefix_probability)
    packed_content_probability = layout.pack_scores(content_probability)
    packed_prefix_value = layout.pack_value(prefix_value)
    packed_content_value = layout.pack_value(content_value)
    expected_output = (
        pv_pairfold128_numpy(
            packed_prefix_probability,
            packed_prefix_value,
            layout=layout,
            occupied=prefix_k,
        )
        + pv_pairfold128_numpy(
            packed_content_probability,
            packed_content_value,
            layout=layout,
            occupied=occupied,
        )
    )[0]

    pv_context = _FakeContext(slots=layout.slots)
    prefix_probability_cipher = pv_context.cipher(
        packed_prefix_probability[0], label="prefix_probability"
    )
    content_probability_cipher = pv_context.cipher(
        packed_content_probability[0], label="content_probability"
    )
    value_ciphers = tuple(
        pv_context.cipher(row, cur_limbs=9, label=f"v{index}")
        for index, row in enumerate(packed_content_value)
    )
    pv_profile: dict[str, float] = {}
    (output,) = pv_pairfold_pages_fhe(
        ((content_probability_cipher,),),
        (value_ciphers,),
        layout=layout,
        crypto_context=pv_context,
        occupied_pages=(occupied,),
        prefix_probability_ciphers=(prefix_probability_cipher,),
        value_prefix=prefix_value,
        prefix_k=prefix_k,
        profile=pv_profile,
    )
    assert output.values is not None
    np.testing.assert_allclose(output.values, expected_output, atol=3e-12)
    for step in layout.token_rotations():
        assert pv_context.rotations.count(step) == 8
    assert pv_context.counts["mul_relin"] == 8
    assert pv_profile["data_ct_pt"] == 8.0
    assert pv_profile["token_reductions"] == 8.0
    pv_inputs = (
        prefix_probability_cipher,
        content_probability_cipher,
        *value_ciphers,
    )
    pv_borrowed = _borrowed_ids(pv_inputs)
    release_if_supported(output)
    _assert_borrowed_and_temporaries_clean(pv_context, pv_borrowed)
    _release_inputs(pv_inputs)


def test_qk_group_extractor_batches_routes_but_consumes_each_query() -> None:
    rng = np.random.default_rng(26092217)
    layout = PairFold128Layout()
    query = rng.normal(scale=0.1, size=(layout.query_heads, layout.head_dim))
    key = rng.normal(
        scale=0.1,
        size=(layout.seq_len, layout.key_value_heads, layout.head_dim),
    )
    packed_query = layout.pack_query(query)
    packed_key = layout.pack_key(key)
    context = _FakeContext(slots=layout.slots)
    carrier = context.cipher(
        np.zeros((layout.slots,), dtype=np.complex128), label="carrier"
    )
    key_ciphers = tuple(
        context.cipher(row, label=f"k{index}")
        for index, row in enumerate(packed_key)
    )
    extracted: list[_FakeCipher] = []
    calls: list[tuple[int, tuple[int, ...]]] = []

    def extract_group(_carrier, *, group: int, query_offsets):
        offsets = tuple(int(offset) for offset in query_offsets)
        calls.append((int(group), offsets))
        outputs = tuple(
            context.cipher(
                packed_query[int(group) * int(layout.gqa_ratio) + offset],
                label=f"group_q.{group}.{offset}",
            )
            for offset in offsets
        )
        extracted.extend(outputs)
        return outputs

    profile: dict[str, float] = {}
    score_pages = qk_pairfold_pages_fhe(
        carrier,
        (key_ciphers,),
        query_extractor=lambda *_args, **_kwargs: pytest.fail(
            "the grouped extractor should handle every stream"
        ),
        query_group_extractor=extract_group,
        layout=layout,
        crypto_context=context,
        occupied_pages=(3,),
        profile=profile,
    )
    expected = qk_pairfold128_numpy(
        packed_query,
        packed_key,
        layout=layout,
        occupied=3,
    )[0]
    assert len(score_pages) == 1
    assert score_pages[0][0].values is not None
    np.testing.assert_allclose(score_pages[0][0].values, expected, atol=2e-12)
    assert calls == [
        (0, (0, 1, 2, 3)),
        (1, (0, 1, 2, 3)),
    ]
    assert profile["query_group_batched"] == 1.0
    assert profile["query_peak_live_ciphers"] == float(layout.gqa_ratio)
    for page in score_pages:
        _release_inputs(page)
    _assert_borrowed_and_temporaries_clean(context, {id(carrier), *(id(c) for c in key_ciphers)})
    assert all(cipher.release_count == 1 for cipher in extracted)
    _release_inputs((carrier, *key_ciphers))


def test_qk_pairfold_failure_releases_all_temporaries_and_borrows_inputs() -> None:
    layout = PairFold128Layout()
    packed_query = np.zeros(
        (layout.query_residue_count, layout.slots), dtype=np.complex128
    )
    packed_key = np.zeros(
        (layout.key_value_groups, layout.slots), dtype=np.complex128
    )
    context = _FakeContext(
        slots=layout.slots,
        fail_mul_relin_at=2,
    )
    carrier = context.cipher(
        np.zeros((layout.slots,), dtype=np.complex128), label="carrier"
    )
    extractor = _StreamingQueryExtractor(context, packed_query, carrier)
    key = tuple(
        context.cipher(row, label=f"k{index}")
        for index, row in enumerate(packed_key)
    )
    inputs = (carrier, *key)
    borrowed = _borrowed_ids(inputs)

    with pytest.raises(RuntimeError, match="injected mul_relin failure"):
        qk_pairfold_pages_fhe(
            carrier,
            (key,),
            query_extractor=extractor,
            layout=layout,
            crypto_context=context,
            occupied_pages=(1,),
        )

    assert extractor.max_live == 1
    assert all(cipher.release_count == 1 for cipher in extractor.outputs)
    _assert_borrowed_and_temporaries_clean(context, borrowed)
    _release_inputs(inputs)
