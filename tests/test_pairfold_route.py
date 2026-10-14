from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.approx.attention import rope_cos_sin
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.attention.pairfold import pairfold_route
from llama3fhe.operators.attention.pairfold.pairfold_route import (
    pairfold_kv_from_carrier_fhe,
    pairfold_kv_from_carrier_numpy,
    pairfold_query_from_carrier_fhe,
    pairfold_query_group_from_carrier_fhe,
    pairfold_query_from_carrier_numpy,
    pairfold_route_complexity,
    pairfold_route_rotations,
)
from llama3fhe.operators.attention.sparse_bsgs_scheduler import sparse_route_arrays

_BOUNDARY_COLUMNS = (0, 1, 63, 64, 126, 127)


def _logical_values(seed: int = 26092211):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(32, 128)),
        rng.normal(size=(8, 128)),
        rng.normal(size=(8, 128)),
    )


def _phase(layout: PairFold128Layout, *, position: int) -> np.ndarray:
    cos, sin = rope_cos_sin(
        seq_len=1,
        head_dim=layout.head_dim,
        theta=500000.0,
        start_pos=int(position),
    )
    return np.asarray(
        cos[0, : layout.fold_dim] + 1j * sin[0, : layout.fold_dim],
        dtype=np.complex128,
    )


def _expected_kv(
    key: np.ndarray,
    value: np.ndarray,
    *,
    layout: PairFold128Layout,
    position: int,
    column: int,
) -> tuple[np.ndarray, np.ndarray]:
    key_rows = np.zeros(
        (layout.key_value_groups, layout.slots), dtype=np.complex128
    )
    value_rows = np.zeros_like(key_rows)
    phase = _phase(layout, position=position)
    for group in range(layout.key_value_groups):
        for pair in range(layout.pair_blocks):
            for lane in range(layout.head_lanes):
                head = layout.key_value_head(group, pair, lane)
                for feature in range(layout.fold_dim):
                    slot = layout.page_slot(pair, feature, column, lane)
                    key_rows[group, slot] = complex(
                        key[head, feature], -key[head, feature + layout.fold_dim]
                    ) * np.conjugate(phase[feature])
                    value_rows[group, slot] = complex(
                        value[head, feature],
                        value[head, feature + layout.fold_dim],
                    )
    return key_rows, value_rows


def _expected_query(
    query: np.ndarray,
    *,
    layout: PairFold128Layout,
    group: int,
    query_offset: int,
    position: int,
) -> np.ndarray:
    output = np.zeros((layout.slots,), dtype=np.complex128)
    phase = _phase(layout, position=position)
    for pair in range(layout.pair_blocks):
        for lane in range(layout.head_lanes):
            head = layout.query_head(group, pair, lane, query_offset)
            for feature in range(layout.fold_dim):
                slot = layout.page_slot(pair, feature, 0, lane)
                output[slot] = 0.5 * complex(
                    query[head, feature],
                    query[head, feature + layout.fold_dim],
                ) * phase[feature]
    return output


@pytest.mark.parametrize("column", _BOUNDARY_COLUMNS)
def test_numpy_kv_routes_are_full_slot_exact_at_boundaries(column: int) -> None:
    layout = PairFold128Layout()
    query, key, value = _logical_values()
    carrier = layout.pack_qkv_carrier(query, key, value)

    actual_key, actual_value = pairfold_kv_from_carrier_numpy(
        carrier,
        layout=layout,
        position=37,
        token_column=column,
    )
    expected_key, expected_value = _expected_kv(
        key,
        value,
        layout=layout,
        position=37,
        column=column,
    )

    np.testing.assert_allclose(actual_key, expected_key, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual_value, expected_value, rtol=0.0, atol=1e-14)
    assert np.count_nonzero(actual_key) == 2 * 2 * 64 * 2
    assert np.count_nonzero(actual_value) == 2 * 2 * 64 * 2


@pytest.mark.parametrize("group", (0, 1))
@pytest.mark.parametrize("query_offset", range(4))
def test_numpy_query_route_streams_one_group_offset(
    group: int, query_offset: int
) -> None:
    layout = PairFold128Layout()
    query, key, value = _logical_values()
    carrier = layout.pack_qkv_carrier(query, key, value)

    actual = pairfold_query_from_carrier_numpy(
        carrier,
        layout=layout,
        group=group,
        query_offset=query_offset,
        position=29,
    )
    expected = _expected_query(
        query,
        layout=layout,
        group=group,
        query_offset=query_offset,
        position=29,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert np.count_nonzero(actual) == 2 * 64 * 2


def test_inactive_physical_carrier_slots_cannot_pollute_outputs() -> None:
    layout = PairFold128Layout()
    query, key, value = _logical_values()
    clean = layout.pack_qkv_carrier(query, key, value)[0]
    active_slots = {
        layout.carrier_slot(pair, feature, residue, lane)
        for group in range(layout.key_value_groups)
        for residue in (
            layout.key_residue(group),
            layout.value_residue(group),
            *(layout.query_residue(group, r) for r in range(layout.gqa_ratio)),
        )
        for pair in range(layout.pair_blocks)
        for feature in range(layout.fold_dim)
        for lane in range(layout.head_lanes)
    }
    assert len(active_slots) == layout.active_channel_count
    rng = np.random.default_rng(26092212)
    polluted = 1e6 * (
        rng.normal(size=layout.slots) + 1j * rng.normal(size=layout.slots)
    )
    indices = np.fromiter(active_slots, dtype=np.int64)
    polluted[indices] = clean[indices]

    clean_key, clean_value = pairfold_kv_from_carrier_numpy(
        clean,
        layout=layout,
        position=11,
        token_column=127,
    )
    dirty_key, dirty_value = pairfold_kv_from_carrier_numpy(
        polluted,
        layout=layout,
        position=11,
        token_column=127,
    )
    np.testing.assert_array_equal(dirty_key, clean_key)
    np.testing.assert_array_equal(dirty_value, clean_value)
    for group in range(layout.key_value_groups):
        for query_offset in range(layout.gqa_ratio):
            clean_q = pairfold_query_from_carrier_numpy(
                clean,
                layout=layout,
                group=group,
                query_offset=query_offset,
                position=11,
            )
            dirty_q = pairfold_query_from_carrier_numpy(
                polluted,
                layout=layout,
                group=group,
                query_offset=query_offset,
                position=11,
            )
            np.testing.assert_array_equal(dirty_q, clean_q)


def test_numpy_route_rejects_active_prefix_instead_of_full_carrier() -> None:
    layout = PairFold128Layout()
    prefix = np.zeros(
        (layout.active_channel_count * layout.native_token_lanes,),
        dtype=np.complex128,
    )
    with pytest.raises(ValueError, match="complete physical carrier"):
        pairfold_kv_from_carrier_numpy(
            prefix,
            layout=layout,
            position=0,
            token_column=0,
        )


@pytest.mark.parametrize("column", _BOUNDARY_COLUMNS)
def test_complexity_pins_24_routes_and_shared_kv_basis(column: int) -> None:
    layout = PairFold128Layout()
    complexity = pairfold_route_complexity(layout, token_column=column)

    assert complexity.kv_transform_count == 4
    assert complexity.kv_output_cipher_count == 4
    assert complexity.kv_pt_ct_multiplications == 8
    assert complexity.kv_shared_baby_basis_size == 2
    assert complexity.query_stream_count == 8
    assert complexity.query_live_cipher_count == 1
    assert complexity.query_pt_ct_multiplications == 16
    assert complexity.pt_ct_multiplications == 24
    assert complexity.output_rescales == 12
    assert complexity.to_dict()["rotations"] == complexity.rotations


def test_rotation_union_is_canonical_unique_and_column_sensitive() -> None:
    layout = PairFold128Layout()
    column_zero = pairfold_route_rotations(layout, token_columns=0)
    column_last = pairfold_route_rotations(layout, token_columns=127)
    all_columns = pairfold_route_rotations(layout)

    assert column_zero == tuple(sorted(set(column_zero)))
    assert column_last == tuple(sorted(set(column_last)))
    assert 0 not in all_columns
    assert set(column_zero) <= set(all_columns)
    assert set(column_last) <= set(all_columns)
    assert column_zero != column_last
    assert 7 in column_zero  # lane-1 baby route
    assert -256 in column_last  # final-page K giant route


class _FakeCipher:
    def __init__(self, values: np.ndarray, *, label: str) -> None:
        self.values = np.asarray(values)
        self.label = str(label)
        self.release_count = 0

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"double release: {self.label}")
        self.release_count = 1


class _FakeContext:
    def __init__(self) -> None:
        self.context = object()
        self.fhe = SimpleNamespace()


class _FakePlain:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.release_count = 0

    def release(self) -> None:
        if self.release_count:
            raise AssertionError("plaintext released twice")
        self.release_count = 1


class _ScalarFhe:
    def __init__(self) -> None:
        self.created: list[_FakeCipher] = []

    def _cipher(self, values: np.ndarray, *, label: str) -> _FakeCipher:
        output = _FakeCipher(values, label=label)
        self.created.append(output)
        return output

    def homo_rotate(self, cipher, offset, _context):  # noqa: ANN001
        return self._cipher(
            np.roll(cipher.values, -int(offset)), label=f"rot.{offset}"
        )

    def homo_mul_pt(self, cipher, plain, _context):  # noqa: ANN001
        return self._cipher(cipher.values * plain.values, label="mul_pt")

    def homo_add(self, left, right, _context):  # noqa: ANN001
        return self._cipher(left.values + right.values, label="add")

    def rescale(self, cipher, _context):  # noqa: ANN001
        return self._cipher(np.array(cipher.values, copy=True), label="rescale")


class _ScalarContext:
    def __init__(self, *, slots: int) -> None:
        self.max_slots = int(slots)
        self.context = object()
        self.fhe = _ScalarFhe()

    @staticmethod
    def level_for_cipher(_cipher) -> int:  # noqa: ANN001
        return 0

    def plaintext(self, values, **_kwargs):  # noqa: ANN001
        return _FakePlain(np.asarray(values))


class _FakeRouteExecutor:
    def __init__(self, *, fail_selected_at: int | None = None) -> None:
        self.fail_selected_at = fail_selected_at
        self.selected_calls = 0
        self.shared_calls = 0
        self.outputs: list[_FakeCipher] = []
        self.shared_transforms = ()

    @staticmethod
    def _apply(values: np.ndarray, routes) -> np.ndarray:  # noqa: ANN001
        output = np.zeros_like(values, dtype=np.complex128)
        for offset, route in routes.items():
            destinations, coefficients = sparse_route_arrays(route)
            rotated = np.roll(values, -int(offset))
            output[destinations] += rotated[destinations] * coefficients
        return output

    def selected(self, ciphers, *, routes, profile=None, **_kwargs):  # noqa: ANN001
        call_index = self.selected_calls
        self.selected_calls += 1
        if call_index == self.fail_selected_at:
            raise RuntimeError(f"selected failed at {call_index}")
        output = _FakeCipher(
            self._apply(ciphers[0].values, routes),
            label=f"selected.{call_index}",
        )
        self.outputs.append(output)
        if profile is not None:
            profile["wall_seconds"] = 0.125
            profile["batched"] = 0.0
        return (output,)

    def shared(
        self,
        cipher,
        *,
        transforms,
        profiles=None,
        **_kwargs,
    ):
        self.shared_calls += 1
        self.shared_transforms = tuple(transforms)
        outputs = tuple(
            _FakeCipher(
                self._apply(cipher.values, routes),
                label=f"shared.{index}",
            )
            for index, (routes, _plan, _name) in enumerate(transforms)
        )
        self.outputs.extend(outputs)
        if profiles is not None:
            for profile in profiles:
                profile["shared_basis_size"] = 2.0
        return outputs


def _install_executor(
    monkeypatch: pytest.MonkeyPatch,
    executor: _FakeRouteExecutor,
) -> None:
    monkeypatch.setattr(
        pairfold_route, "_apply_sparse_routes_selected", executor.selected
    )
    monkeypatch.setattr(
        pairfold_route,
        "_apply_sparse_routes_batched_shared",
        executor.shared,
    )


def _carrier_cipher(layout: PairFold128Layout) -> tuple[_FakeCipher, tuple]:
    logical = _logical_values()
    return (
        _FakeCipher(layout.pack_qkv_carrier(*logical)[0], label="carrier"),
        logical,
    )


def test_fhe_kv_uses_one_shared_four_transform_group_and_borrows_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = PairFold128Layout()
    carrier_layout = LinearCarrierLayout(seq_len=1, dimension=4096)
    carrier, (_query, key, value) = _carrier_cipher(layout)
    executor = _FakeRouteExecutor()
    _install_executor(monkeypatch, executor)
    profile: dict[str, float] = {}

    token = pairfold_kv_from_carrier_fhe(
        carrier,
        carrier_layout=carrier_layout,
        layout=layout,
        crypto_context=_FakeContext(),
        position=19,
        token_column=64,
        profile=profile,
    )

    assert executor.shared_calls == 1
    assert executor.selected_calls == 0
    assert len(executor.shared_transforms) == 4
    assert [name.rsplit(".", 2)[-2:] for _, _, name in executor.shared_transforms]
    shared_basis = {
        offset
        for _, plan, _ in executor.shared_transforms
        for offset in plan.baby_offsets
    }
    assert len(shared_basis) == 2
    actual_key = np.stack([cipher.values for cipher in token.key_groups])
    actual_value = np.stack([cipher.values for cipher in token.value_groups])
    expected_key, expected_value = _expected_kv(
        key,
        value,
        layout=layout,
        position=19,
        column=64,
    )
    np.testing.assert_allclose(actual_key, expected_key, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual_value, expected_value, rtol=0.0, atol=1e-14)
    assert carrier.release_count == 0
    assert profile["logical_pt_ct"] == 8.0
    assert profile["shared_group_count"] == 1.0
    token.release()
    assert all(output.release_count == 1 for output in executor.outputs)
    assert carrier.release_count == 0


def test_fhe_query_extracts_one_stream_and_reports_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = PairFold128Layout()
    carrier_layout = LinearCarrierLayout(seq_len=1, dimension=4096)
    carrier, (query, _key, _value) = _carrier_cipher(layout)
    executor = _FakeRouteExecutor()
    _install_executor(monkeypatch, executor)
    profile: dict[str, float] = {}

    output = pairfold_query_from_carrier_fhe(
        carrier,
        carrier_layout=carrier_layout,
        layout=layout,
        crypto_context=_FakeContext(),
        group=1,
        query_offset=3,
        position=23,
        profile=profile,
    )

    expected = _expected_query(
        query,
        layout=layout,
        group=1,
        query_offset=3,
        position=23,
    )
    np.testing.assert_allclose(output.values, expected, rtol=0.0, atol=1e-14)
    assert executor.selected_calls == 1
    assert executor.shared_calls == 0
    assert carrier.release_count == 0
    assert profile["logical_pt_ct"] == 2.0
    assert profile["apply"] == 0.125
    output.release()
    assert output.release_count == 1


def test_fhe_query_group_shares_one_baby_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = PairFold128Layout()
    carrier_layout = LinearCarrierLayout(seq_len=1, dimension=4096)
    carrier, (query, _key, _value) = _carrier_cipher(layout)
    executor = _FakeRouteExecutor()
    _install_executor(monkeypatch, executor)
    profile: dict[str, float] = {}

    outputs = pairfold_query_group_from_carrier_fhe(
        carrier,
        carrier_layout=carrier_layout,
        layout=layout,
        crypto_context=_FakeContext(),
        group=1,
        position=23,
        profile=profile,
    )

    assert len(outputs) == layout.gqa_ratio
    assert executor.shared_calls == 1
    assert executor.selected_calls == 0
    assert profile["stream_count"] == float(layout.gqa_ratio)
    assert profile["shared_basis_size"] == 2.0
    for offset, output in enumerate(outputs):
        expected = _expected_query(
            query,
            layout=layout,
            group=1,
            query_offset=offset,
            position=23,
        )
        np.testing.assert_allclose(output.values, expected, rtol=0.0, atol=1e-14)
        output.release()
    assert all(output.release_count == 1 for output in executor.outputs)
    assert carrier.release_count == 0


def test_scalar_scheduler_executes_custom_radix_plan_slot_exactly() -> None:
    """Exercise the real correctness-first sparse scheduler, not a route stub."""

    layout = PairFold128Layout()
    carrier_layout = LinearCarrierLayout(seq_len=1, dimension=4096)
    carrier, (query, key, value) = _carrier_cipher(layout)
    context = _ScalarContext(slots=layout.slots)

    token = pairfold_kv_from_carrier_fhe(
        carrier,
        carrier_layout=carrier_layout,
        layout=layout,
        crypto_context=context,
        position=31,
        token_column=127,
        batched_sparse_bsgs=False,
        shared_baby_rotations=False,
    )
    query_cipher = pairfold_query_from_carrier_fhe(
        carrier,
        carrier_layout=carrier_layout,
        layout=layout,
        crypto_context=context,
        group=1,
        query_offset=2,
        position=31,
        batched_sparse_bsgs=False,
    )

    expected_key, expected_value = _expected_kv(
        key,
        value,
        layout=layout,
        position=31,
        column=127,
    )
    expected_query = _expected_query(
        query,
        layout=layout,
        group=1,
        query_offset=2,
        position=31,
    )
    np.testing.assert_allclose(
        np.stack([cipher.values for cipher in token.key_groups]),
        expected_key,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.stack([cipher.values for cipher in token.value_groups]),
        expected_value,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        query_cipher.values, expected_query, rtol=0.0, atol=1e-14
    )
    assert carrier.release_count == 0
    token.release()
    query_cipher.release()


def test_fhe_kv_failure_releases_completed_outputs_and_preserves_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = PairFold128Layout()
    carrier_layout = LinearCarrierLayout(seq_len=1, dimension=4096)
    carrier, _ = _carrier_cipher(layout)
    executor = _FakeRouteExecutor(fail_selected_at=2)
    _install_executor(monkeypatch, executor)

    with pytest.raises(RuntimeError, match="selected failed at 2"):
        pairfold_kv_from_carrier_fhe(
            carrier,
            carrier_layout=carrier_layout,
            layout=layout,
            crypto_context=_FakeContext(),
            position=0,
            token_column=0,
            batched_sparse_bsgs=False,
            shared_baby_rotations=False,
        )

    assert executor.selected_calls == 3
    assert len(executor.outputs) == 2
    assert all(output.release_count == 1 for output in executor.outputs)
    assert carrier.release_count == 0


def test_shared_failure_falls_back_without_consuming_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = PairFold128Layout()
    carrier_layout = LinearCarrierLayout(seq_len=1, dimension=4096)
    carrier, _ = _carrier_cipher(layout)
    executor = _FakeRouteExecutor()

    def fail_shared(*_args, **_kwargs):
        executor.shared_calls += 1
        raise NotImplementedError("no batch API")

    monkeypatch.setattr(
        pairfold_route, "_apply_sparse_routes_batched_shared", fail_shared
    )
    monkeypatch.setattr(
        pairfold_route, "_apply_sparse_routes_selected", executor.selected
    )
    profile: dict[str, float] = {}

    with pytest.warns(RuntimeWarning, match="per-transform fallback"):
        token = pairfold_kv_from_carrier_fhe(
            carrier,
            carrier_layout=carrier_layout,
            layout=layout,
            crypto_context=_FakeContext(),
            position=0,
            token_column=0,
            profile=profile,
        )

    assert executor.shared_calls == 1
    assert executor.selected_calls == 4
    assert profile["shared_fallbacks"] == 1.0
    assert carrier.release_count == 0
    token.release()
    assert all(output.release_count == 1 for output in executor.outputs)


def test_fhe_preflight_rejects_wrong_layout_and_column_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = PairFold128Layout()
    carrier, _ = _carrier_cipher(layout)
    executor = _FakeRouteExecutor()
    _install_executor(monkeypatch, executor)

    with pytest.raises(ValueError, match="physical carrier width"):
        pairfold_kv_from_carrier_fhe(
            carrier,
            carrier_layout=LinearCarrierLayout(seq_len=1, dimension=2048),
            layout=layout,
            crypto_context=_FakeContext(),
            position=0,
            token_column=0,
        )
    with pytest.raises(IndexError):
        pairfold_kv_from_carrier_fhe(
            carrier,
            carrier_layout=LinearCarrierLayout(seq_len=1, dimension=4096),
            layout=layout,
            crypto_context=_FakeContext(),
            position=0,
            token_column=128,
        )

    assert executor.shared_calls == 0
    assert executor.selected_calls == 0
    assert carrier.release_count == 0
