from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.config import SimulatorConfig
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.linear.pairfold import pairfold_output
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.pairfold.pairfold_output import (
    pairfold_w_o_fhe,
    pairfold_w_o_needs_refresh,
    pairfold_w_o_rotations,
    pairfold_w_o_weight,
)


def _column_weight(dimension: int) -> np.ndarray:
    columns = np.linspace(-0.75, 1.25, dimension, dtype=np.float32)
    return np.broadcast_to(columns.reshape(1, -1), (dimension, dimension))


def test_pairfold_w_o_descriptor_maps_sparse_low_and_high_columns() -> None:
    layout = PairFold128Layout()
    dimension = layout.carrier_dimension
    weight = _column_weight(dimension)
    packed = pairfold_w_o_weight(weight, layout=layout)
    descriptor = packed.descriptor

    assert packed.physical_width == 4096
    assert packed.active_input_count == 2048
    assert len(set(packed.active_input_channels)) == 2048
    assert max(packed.active_input_channels) == 4087
    np.testing.assert_array_equal(descriptor.real_row, np.arange(dimension))
    np.testing.assert_array_equal(descriptor.imag_row, np.arange(dimension))
    np.testing.assert_array_equal(descriptor.real_scale, 0.5)
    np.testing.assert_array_equal(descriptor.imag_scale, -0.5)

    active = np.zeros((dimension,), dtype=bool)
    active[np.asarray(packed.active_input_channels)] = True
    np.testing.assert_array_equal(descriptor.column[~active], -1)
    np.testing.assert_array_equal(descriptor.real_column[~active], -1)
    np.testing.assert_array_equal(descriptor.imag_column[~active], -1)

    for group in range(layout.key_value_groups):
        for pair in range(layout.pair_blocks):
            for lane in range(layout.head_lanes):
                for query_offset in range(layout.gqa_ratio):
                    head = layout.query_head(group, pair, lane, query_offset)
                    residue = layout.query_residue(group, query_offset)
                    for feature in (0, 1, 31, 63):
                        channel = layout.carrier_channel(
                            pair, feature, residue, lane
                        )
                        low = head * layout.head_dim + feature
                        assert descriptor.real_column[channel] == low
                        assert descriptor.imag_column[channel] == low + layout.fold_dim
                        assert descriptor.column[channel] == low


def test_pairfold_w_o_descriptor_matches_dense_numpy_identity() -> None:
    layout = PairFold128Layout()
    dimension = layout.carrier_dimension
    logical = np.zeros((layout.query_heads, layout.head_dim), dtype=np.float64)
    logical[0, 3] = 0.75
    logical[0, 67] = -0.5
    logical[17, 31] = 1.25
    logical[17, 95] = 0.375
    logical[31, 63] = -0.625
    logical[31, 127] = 1.5
    carrier = layout.pack_output(logical, dtype=np.complex128)
    features = carrier.reshape(dimension, layout.native_token_lanes)[:, 0]
    weight = _column_weight(dimension)
    descriptor = pairfold_w_o_weight(weight, layout=layout).descriptor

    active = np.flatnonzero(features)
    projected = np.zeros((dimension,), dtype=np.complex128)
    for physical_input in active:
        coefficient = 0.5 * (
            weight[:, descriptor.real_column[physical_input]]
            - 1j * weight[:, descriptor.imag_column[physical_input]]
        )
        projected += coefficient * features[physical_input]
    actual = (projected + np.conjugate(projected)).real
    expected = logical.reshape(1, -1) @ weight.T
    np.testing.assert_allclose(actual, expected[0], atol=1e-11, rtol=1e-11)


def test_pairfold_w_o_rotation_union_contains_actual_conjugation() -> None:
    layout = PairFold128Layout()
    rotations = pairfold_w_o_rotations(layout, extra_rotations=(-7, 0))
    assert 4 * layout.slots - 1 in rotations
    assert -7 in rotations
    assert 0 not in rotations


class _Cipher:
    def __init__(
        self, values: np.ndarray, owner: "_Context", *, limbs: int = 4
    ) -> None:
        self.values = np.asarray(values)
        self.state = SimpleNamespace(cur_limbs=int(limbs))
        self.release_count = 0
        owner.ciphers.append(self)

    def release(self) -> None:
        self.release_count += 1


class _Fhe:
    def __init__(self, owner: "_Context", *, fail_add: bool = False) -> None:
        self.owner = owner
        self.fail_add = bool(fail_add)
        self.rotations: list[int] = []

    def homo_rotate(self, cipher, rotation: int, context):  # noqa: ANN001
        assert rotation == int(context.M) - 1
        self.rotations.append(int(rotation))
        return _Cipher(np.conjugate(cipher.values), self.owner)

    def homo_add(self, left, right, _context):  # noqa: ANN001
        if self.fail_add:
            raise RuntimeError("injected output-conjugation failure")
        return _Cipher(left.values + right.values, self.owner)


class _Context:
    def __init__(self, *, slots: int, fail_add: bool = False) -> None:
        self.max_slots = int(slots)
        self.context = SimpleNamespace(M=4 * int(slots))
        self.device = "cpu"
        self.ciphers: list[_Cipher] = []
        self.fhe = _Fhe(self, fail_add=fail_add)

    @staticmethod
    def level_for_cipher(_cipher) -> int:  # noqa: ANN001
        return 4


class _LinearOperator:
    instances: list["_LinearOperator"] = []

    def __init__(self, *, crypto_context, token_lanes: int, config) -> None:  # noqa: ANN001
        self.crypto_context = crypto_context
        self.token_lanes = int(token_lanes)
        self.config = config
        self.closed = False
        self.weight = None
        type(self).instances.append(self)

    def project(self, inputs, weight, *, dtype):  # noqa: ANN001
        self.weight = weight
        assert np.issubdtype(np.dtype(dtype), np.complexfloating)
        assert len(inputs) == 1
        values = np.arange(weight.dimension, dtype=np.float64)
        values = values + 1j * values[::-1]
        lanes = np.zeros(
            (weight.dimension, self.token_lanes), dtype=np.complex128
        )
        lanes[:, 0] = values
        return (
            (_Cipher(lanes.reshape(-1), self.crypto_context),),
            {"fake_kernel": 0.125},
        )

    def close(self) -> None:
        self.closed = True


def _patch_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    _LinearOperator.instances.clear()
    monkeypatch.setattr(pairfold_output, "LinearOperator", _LinearOperator)
    monkeypatch.setattr(pairfold_output, "sync_device", lambda _device: None)


def test_pairfold_w_o_fhe_returns_owned_real_ct_and_borrows_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_kernel(monkeypatch)
    layout = PairFold128Layout()
    context = _Context(slots=layout.slots)
    input_cipher = _Cipher(np.zeros((layout.slots,)), context)
    profile: dict[str, float] = {}

    (output,) = pairfold_w_o_fhe(
        (input_cipher,),
        _column_weight(layout.carrier_dimension),
        layout=layout,
        crypto_context=context,
        profile=profile,
    )

    operator = _LinearOperator.instances[-1]
    assert operator.closed
    assert operator.token_lanes == layout.native_token_lanes
    assert operator.weight.real_column is not operator.weight.imag_column
    assert context.fhe.rotations == [int(context.context.M) - 1]
    assert input_cipher.release_count == 0
    assert output.release_count == 0
    assert np.max(np.abs(output.values.imag)) == 0.0
    assert profile["projection_fake_kernel"] == 0.125
    temporaries = [
        cipher
        for cipher in context.ciphers
        if cipher is not input_cipher and cipher is not output
    ]
    assert temporaries
    assert all(cipher.release_count == 1 for cipher in temporaries)

    output.release()
    assert output.release_count == 1
    assert input_cipher.release_count == 0


def test_pairfold_w_o_fhe_failure_releases_temporaries_not_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_kernel(monkeypatch)
    layout = PairFold128Layout()
    context = _Context(slots=layout.slots, fail_add=True)
    input_cipher = _Cipher(np.zeros((layout.slots,)), context)

    with pytest.raises(RuntimeError, match="injected output-conjugation failure"):
        pairfold_w_o_fhe(
            (input_cipher,),
            _column_weight(layout.carrier_dimension),
            layout=layout,
            crypto_context=context,
        )

    assert _LinearOperator.instances[-1].closed
    assert input_cipher.release_count == 0
    assert all(
        cipher.release_count == 1
        for cipher in context.ciphers
        if cipher is not input_cipher
    )


def test_pairfold_w_o_exposes_graph_level_refresh_precondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_kernel(monkeypatch)
    layout = PairFold128Layout()
    context = _Context(slots=layout.slots)
    enough = _Cipher(np.zeros((layout.slots,)), context, limbs=2)
    short = _Cipher(np.zeros((layout.slots,)), context, limbs=1)

    assert not pairfold_w_o_needs_refresh(
        (enough,), crypto_context=context
    )
    assert pairfold_w_o_needs_refresh((short,), crypto_context=context)
    with pytest.raises(ValueError, match="graph must apply"):
        pairfold_w_o_fhe(
            (short,),
            _column_weight(layout.carrier_dimension),
            layout=layout,
            crypto_context=context,
        )
    assert short.release_count == 0
    assert not _LinearOperator.instances


@pytest.mark.gpu
def test_pairfold_w_o_gpu_matches_dense_numpy() -> None:
    layout = PairFold128Layout()
    dimension = layout.carrier_dimension
    operator_config = LinearOperatorConfig(
        baby_steps=64,
        baby_anchor_step=64,
        max_plaintext_rows=512,
        constant_cache_mode="none",
        raw_weight_policy="device",
    )
    simulator = SimulatorConfig(
        logN=16,
        maxLevelsRemaining=3,
        dnum=1,
        dcrtBits=59,
        firstMod=60,
        device="cuda",
    )
    context = create_sim_context(
        simulator,
        input_dim=dimension,
        block_out=layout.native_token_lanes,
        rotations=pairfold_w_o_rotations(
            layout, operator_config=operator_config
        ),
        include_default_rotations=False,
    )
    rng = np.random.default_rng(26092201)
    logical = rng.normal(
        scale=0.01, size=(layout.query_heads, layout.head_dim)
    )
    carrier = layout.pack_output(logical, dtype=np.complex128)[0]
    weight = np.eye(dimension, dtype=np.float32)
    input_cipher = context.encrypt(carrier, level=0, slots=layout.slots)
    outputs: tuple[object, ...] = ()
    try:
        outputs = pairfold_w_o_fhe(
            (input_cipher,),
            weight,
            layout=layout,
            crypto_context=context,
            operator_config=operator_config,
        )
        decrypted = np.asarray(context.decrypt(outputs[0])).reshape(
            dimension, layout.native_token_lanes
        )
        expected = logical.reshape(-1)
        np.testing.assert_allclose(
            decrypted[:, 0].real, expected, atol=2e-7, rtol=2e-7
        )
        np.testing.assert_allclose(
            decrypted[:, 0].imag, 0.0, atol=2e-7, rtol=0.0
        )
        np.testing.assert_allclose(
            decrypted[:, 1:], 0.0, atol=2e-7, rtol=0.0
        )
    finally:
        release_if_supported(input_cipher)
        for output in outputs:
            release_if_supported(output)
