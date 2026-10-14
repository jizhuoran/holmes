from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.linear import LinearCarrierLayout
from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.linear.prefill import qkv_kernel
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.pairfold.pairfold_qkv import (
    _two_sided_real_input_reference,
    pairfold_qkv_fhe,
    pairfold_qkv_rotations,
    pairfold_qkv_weight,
)
from llama3fhe.operators.linear.prefill.qkv_kernel import (
    ComplexTokenDiagonalLayout,
    _qkv_fhe_complex_token_diagonal_prepared,
    complex_token_diagonal_qkv_rotations,
    unpack_complex_carrier_shards,
)


def _weights(layout: PairFold128Layout):
    dimension = layout.carrier_dimension
    q = np.broadcast_to(
        np.zeros((), dtype=np.float32),
        (layout.query_heads * layout.head_dim, dimension),
    )
    k = np.broadcast_to(
        np.zeros((), dtype=np.float32),
        (layout.key_value_heads * layout.head_dim, dimension),
    )
    v = np.broadcast_to(np.zeros((), dtype=np.float32), k.shape)
    return q, k, v


def test_pairfold_qkv_two_sided_descriptor_matches_every_active_channel() -> None:
    layout = PairFold128Layout()
    packed = pairfold_qkv_weight(*_weights(layout), layout=layout)
    descriptor = packed.descriptor

    assert packed.physical_width == 4096
    assert packed.active_channel_count == 3072
    assert len(set(packed.active_channels)) == 3072
    assert max(packed.active_channels) == 4087
    assert packed.two_sided_recovery

    half = layout.fold_dim
    for group in range(layout.key_value_groups):
        for pair in range(layout.pair_blocks):
            for lane in range(layout.head_lanes):
                kv_head = layout.key_value_head(group, pair, lane)
                for feature in range(half):
                    low = kv_head * layout.head_dim + feature
                    high = low + half
                    key_channel = layout.carrier_channel(
                        pair, feature, layout.key_residue(group), lane
                    )
                    assert descriptor.real_source[key_channel] == 1
                    assert descriptor.imag_source[key_channel] == 1
                    assert descriptor.real_row[key_channel] == low
                    assert descriptor.imag_row[key_channel] == high
                    assert descriptor.real_scale[key_channel] == 0.5
                    assert descriptor.imag_scale[key_channel] == -0.5

                    value_channel = layout.carrier_channel(
                        pair, feature, layout.value_residue(group), lane
                    )
                    assert descriptor.real_source[value_channel] == 2
                    assert descriptor.imag_source[value_channel] == 2
                    assert descriptor.real_row[value_channel] == low
                    assert descriptor.imag_row[value_channel] == high
                    assert descriptor.real_scale[value_channel] == 0.5
                    assert descriptor.imag_scale[value_channel] == 0.5

                    for query_offset in range(layout.gqa_ratio):
                        query_head = layout.query_head(
                            group, pair, lane, query_offset
                        )
                        query_low = query_head * layout.head_dim + feature
                        query_channel = layout.carrier_channel(
                            pair,
                            feature,
                            layout.query_residue(group, query_offset),
                            lane,
                        )
                        assert descriptor.real_source[query_channel] == 0
                        assert descriptor.imag_source[query_channel] == 0
                        assert descriptor.real_row[query_channel] == query_low
                        assert descriptor.imag_row[query_channel] == query_low + half
                        assert descriptor.real_scale[query_channel] == 0.25
                        assert descriptor.imag_scale[query_channel] == 0.25

    inactive = np.ones((layout.carrier_dimension,), dtype=bool)
    inactive[np.asarray(packed.active_channels, dtype=np.int64)] = False
    np.testing.assert_array_equal(descriptor.real_source[inactive], -1)
    np.testing.assert_array_equal(descriptor.imag_source[inactive], -1)
    np.testing.assert_array_equal(descriptor.real_scale[inactive], 0.0)
    np.testing.assert_array_equal(descriptor.imag_scale[inactive], 0.0)


def test_pairfold_qkv_one_sided_scales_and_input_column_scale() -> None:
    layout = PairFold128Layout()
    column_scale = np.linspace(0.5, 1.5, layout.carrier_dimension)
    packed = pairfold_qkv_weight(
        *_weights(layout),
        layout=layout,
        input_column_scale=column_scale,
        two_sided_recovery=False,
    )
    descriptor = packed.descriptor
    np.testing.assert_array_equal(descriptor.column, np.arange(4096))
    np.testing.assert_allclose(descriptor.column_scale, column_scale)
    assert not packed.two_sided_recovery

    key_channel = layout.carrier_channel(0, 0, layout.key_residue(0), 0)
    value_channel = layout.carrier_channel(0, 0, layout.value_residue(0), 0)
    query_channel = layout.carrier_channel(0, 0, layout.query_residue(0, 0), 0)
    assert descriptor.real_scale[key_channel] == 1.0
    assert descriptor.imag_scale[key_channel] == -1.0
    assert descriptor.real_scale[value_channel] == 1.0
    assert descriptor.imag_scale[value_channel] == 1.0
    assert descriptor.real_scale[query_channel] == 0.5
    assert descriptor.imag_scale[query_channel] == 0.5


def test_pairfold_qkv_weight_rejects_wrong_shapes() -> None:
    layout = PairFold128Layout()
    q, k, v = _weights(layout)
    with pytest.raises(ValueError, match="Q weight"):
        pairfold_qkv_weight(q[:-1], k, v, layout=layout)
    with pytest.raises(ValueError, match="input_column_scale"):
        pairfold_qkv_weight(
            q, k, v, layout=layout, input_column_scale=np.ones((4095,))
        )


def test_pairfold_qkv_descriptor_projects_to_logical_qkv() -> None:
    layout = PairFold128Layout()
    hidden = np.linspace(-0.5, 0.75, layout.carrier_dimension).reshape(1, -1)
    column_scale = np.linspace(0.75, 1.25, layout.carrier_dimension)
    q_rows = np.linspace(-0.2, 0.3, layout.query_heads * layout.head_dim)
    kv_rows = np.linspace(-0.4, 0.1, layout.key_value_heads * layout.head_dim)
    q = np.broadcast_to(q_rows[:, None], (q_rows.size, layout.carrier_dimension))
    k = np.broadcast_to(kv_rows[:, None], (kv_rows.size, layout.carrier_dimension))
    v = np.broadcast_to(
        (kv_rows[::-1] + 0.125)[:, None],
        (kv_rows.size, layout.carrier_dimension),
    )
    packed_weight = pairfold_qkv_weight(
        q.astype(np.float32),
        k.astype(np.float32),
        v.astype(np.float32),
        layout=layout,
        input_column_scale=column_scale,
    )

    carrier_rows = _two_sided_real_input_reference(
        hidden, packed_weight.descriptor
    )
    carrier_layout = LinearCarrierLayout(
        seq_len=1,
        dimension=layout.carrier_dimension,
        slots=layout.slots,
    )
    logical_q, logical_k, logical_v = layout.unpack_qkv_carrier(
        carrier_layout.pack(carrier_rows, dtype=np.complex128)
    )
    scaled_hidden_sum = float(np.sum(hidden * column_scale[None, :]))
    np.testing.assert_allclose(
        logical_q.reshape(-1), q_rows * scaled_hidden_sum, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        logical_k.reshape(-1), kv_rows * scaled_hidden_sum, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        logical_v.reshape(-1),
        (kv_rows[::-1] + 0.125) * scaled_hidden_sum,
        rtol=2e-6,
        atol=2e-6,
    )
    inactive = np.ones((layout.carrier_dimension,), dtype=bool)
    inactive[np.asarray(packed_weight.active_channels)] = False
    np.testing.assert_array_equal(carrier_rows[:, inactive], 0.0)


class _OwnedCipher:
    def __init__(self, values=None) -> None:  # noqa: ANN001
        self.values = values
        self.release_count = 0

    def release(self) -> None:
        self.release_count += 1


class _FakeProjection:
    def __init__(self, cipher: _OwnedCipher, *, carrier_width: int = 4096) -> None:
        self.carrier_ciphers = (cipher,)
        self.output_layout = LinearCarrierLayout(
            seq_len=1, dimension=4096, slots=32768
        )
        self.carrier_width = int(carrier_width)
        self.complexity = SimpleNamespace(output_cipher_count=1)
        self.wall_seconds = 0.25
        self.stage_seconds = {"paired_projection": 0.2}
        self.output_levels = (1,)
        self.max_abs_diff = None

    def release(self) -> None:
        for cipher in self.carrier_ciphers:
            cipher.release()


def test_pairfold_qkv_public_wrapper_preserves_sparse_physical_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = PairFold128Layout()
    q, k, v = _weights(layout)
    hidden = np.zeros((1, layout.carrier_dimension), dtype=np.float32)
    input_cipher = _OwnedCipher()
    output_cipher = _OwnedCipher()
    captured = {}
    projection = _FakeProjection(output_cipher)

    def fake_projection(hidden_states, projection_weight, **kwargs):  # noqa: ANN001
        captured["hidden_states"] = hidden_states
        captured["projection_weight"] = projection_weight
        captured.update(kwargs)
        return projection

    monkeypatch.setattr(
        qkv_kernel, "_qkv_fhe_complex_token_diagonal_prepared", fake_projection
    )
    result = pairfold_qkv_fhe(
        hidden,
        q,
        k,
        v,
        layout=layout,
        crypto_context=SimpleNamespace(max_slots=layout.slots),
        input_ciphers=(input_cipher,),
        verify=False,
    )

    assert result.carrier_cipher is output_cipher
    assert result.carrier_ciphers == (output_cipher,)
    assert result.physical_width == 4096
    assert result.active_channel_count == 3072
    assert len(result.active_channels) == 3072
    assert result.max_active_channel == 4087
    assert 24 not in result.active_channels
    assert 32 in result.active_channels
    assert captured["carrier_width"] == 4096
    assert captured["carrier_reference"] is None
    assert captured["projection_weight"].dimension == 4096
    assert captured["input_ciphers"] == (input_cipher,)
    assert input_cipher.release_count == 0
    assert projection.carrier_ciphers == ()

    result.release()
    projection.release()
    result.release()
    assert output_cipher.release_count == 1
    assert input_cipher.release_count == 0


def test_pairfold_qkv_rejects_active_prefix_result_and_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = PairFold128Layout()
    q, k, v = _weights(layout)
    input_cipher = _OwnedCipher()
    output_cipher = _OwnedCipher()

    monkeypatch.setattr(
        qkv_kernel,
        "_qkv_fhe_complex_token_diagonal_prepared",
        lambda *_args, **_kwargs: _FakeProjection(
            output_cipher, carrier_width=3072
        ),
    )
    with pytest.raises(RuntimeError, match="truncated the physical carrier"):
        pairfold_qkv_fhe(
            np.zeros((1, layout.carrier_dimension), dtype=np.float32),
            q,
            k,
            v,
            layout=layout,
            crypto_context=SimpleNamespace(max_slots=layout.slots),
            input_ciphers=(input_cipher,),
            verify=False,
        )

    assert output_cipher.release_count == 1
    assert input_cipher.release_count == 0


def test_pairfold_qkv_rotation_api_is_exact_kernel_union() -> None:
    layout = PairFold128Layout()
    operator_config = LinearOperatorConfig(
        baby_steps=64,
        baby_anchor_step=32,
        max_plaintext_rows=512,
    )
    expected = set(
        complex_token_diagonal_qkv_rotations(
            dimension=layout.carrier_dimension,
            slots=layout.slots,
            real_token_lanes=layout.native_token_lanes,
            operator_config=operator_config,
        )
    )
    expected.add(-7)
    assert pairfold_qkv_rotations(
        layout,
        operator_config=operator_config,
        extra_rotations=(-7, 0),
    ) == tuple(sorted(expected))
    assert 4 * layout.slots - 1 in expected


class _KernelCipher(_OwnedCipher):
    def __init__(self, values: np.ndarray, owner: "_KernelContext") -> None:
        super().__init__(np.asarray(values))
        owner.ciphers.append(self)


class _KernelFhe:
    def __init__(self, owner: "_KernelContext", *, fail_add: bool = False) -> None:
        self.owner = owner
        self.fail_add = bool(fail_add)

    def homo_rotate(self, cipher, rotation: int, context):  # noqa: ANN001
        assert rotation == int(context.M) - 1
        return _KernelCipher(np.conjugate(cipher.values), self.owner)

    def homo_add(self, left, right, _context):  # noqa: ANN001
        if self.fail_add:
            raise RuntimeError("injected recovery failure")
        return _KernelCipher(left.values + right.values, self.owner)

    def homo_sub(self, left, right, _context):  # noqa: ANN001
        return _KernelCipher(left.values - right.values, self.owner)


class _KernelContext:
    def __init__(self, *, slots: int, fail_add: bool = False) -> None:
        self.context = SimpleNamespace(M=4 * int(slots))
        self.device = "cpu"
        self.ciphers: list[_KernelCipher] = []
        self.fhe = _KernelFhe(self, fail_add=fail_add)

    @staticmethod
    def level_for_cipher(_cipher) -> int:  # noqa: ANN001
        return 2

    @staticmethod
    def decrypt(cipher) -> np.ndarray:  # noqa: ANN001
        return np.asarray(cipher.values.real)


class _KernelLinearOperator:
    def __init__(self, *, crypto_context, token_lanes: int, config) -> None:  # noqa: ANN001
        del config
        self.crypto_context = crypto_context
        self.token_lanes = int(token_lanes)
        self.closed = False

    def project(self, input_ciphers, weight, *, dtype):  # noqa: ANN001
        matrix = np.asarray(weight, dtype=dtype)
        dimension = int(matrix.shape[0])
        outputs = []
        for cipher in input_ciphers:
            feature_lanes = cipher.values.reshape(dimension, self.token_lanes)
            outputs.append(
                _KernelCipher(
                    (matrix @ feature_lanes).reshape(-1), self.crypto_context
                )
            )
        return tuple(outputs), {"fake": 0.0}

    def close(self) -> None:
        self.closed = True


def _patch_prepared_kernel_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qkv_kernel, "LinearOperator", _KernelLinearOperator)
    monkeypatch.setattr(qkv_kernel, "sync_device", lambda _device: None)

    def fake_mul_i(cipher, _context, *, negative: bool = False):  # noqa: ANN001
        factor = -1j if negative else 1j
        return _KernelCipher(factor * cipher.values, cipher.owner)

    # Newly created ciphers need their context for module-level multiply-i.
    original_init = _KernelCipher.__init__

    def init_with_owner(self, values, owner):  # noqa: ANN001
        self.owner = owner
        original_init(self, values, owner)

    monkeypatch.setattr(_KernelCipher, "__init__", init_with_owner)
    monkeypatch.setattr(qkv_kernel.fhe, "homo_mul_i", fake_mul_i)


def test_prepared_two_sided_kernel_matches_complex_oracle_and_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prepared_kernel_fakes(monkeypatch)
    dimension, slots = 4, 16
    hidden = np.asarray([[0.25, -0.5, 0.75, 1.0]], dtype=np.float32)
    desired_weight = np.asarray(
        [
            [1 + 2j, 0, -0.5j, 0.25],
            [0, -2 + 0.5j, 0.75, 1j],
            [0.5, 0.25j, -1, 0],
            [-0.25j, 0.5, 0, 1.5 - 0.5j],
        ],
        dtype=np.complex128,
    )
    input_layout = ComplexTokenDiagonalLayout(
        seq_len=1, dimension=dimension, slots=slots
    )
    context = _KernelContext(slots=slots)
    input_cipher = _KernelCipher(input_layout.pack(hidden)[0], context)
    config = Llama3CKKSConfig()
    config.simulator.logN = 5
    operator_config = LinearOperatorConfig(
        baby_steps=2, baby_anchor_step=2, max_plaintext_rows=4
    )

    result = _qkv_fhe_complex_token_diagonal_prepared(
        hidden,
        0.5 * desired_weight,
        carrier_width=dimension,
        carrier_weight=desired_weight,
        config=config,
        operator_config=operator_config,
        extra_rotations=(),
        crypto_context=context,
        input_ciphers=(input_cipher,),
        verify=True,
        wall_start=time.perf_counter(),
        weight_transform_seconds=0.0,
    )
    output = result.carrier_ciphers[0]
    actual = unpack_complex_carrier_shards(
        output.values.reshape(1, slots),
        layout=result.output_layout,
        output_width=dimension,
    )
    np.testing.assert_allclose(
        actual, hidden.astype(np.float64) @ desired_weight.T, atol=1e-7
    )
    assert result.max_abs_diff is not None
    assert result.max_abs_diff < 1e-7
    assert input_cipher.release_count == 0
    assert all(
        cipher.release_count == 1
        for cipher in context.ciphers
        if cipher is not input_cipher and cipher is not output
    )
    assert output.release_count == 0
    result.release()
    assert output.release_count == 1


def test_prepared_two_sided_kernel_failure_releases_temporaries_not_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prepared_kernel_fakes(monkeypatch)
    dimension, slots = 4, 16
    hidden = np.ones((1, dimension), dtype=np.float32)
    input_layout = ComplexTokenDiagonalLayout(
        seq_len=1, dimension=dimension, slots=slots
    )
    context = _KernelContext(slots=slots, fail_add=True)
    input_cipher = _KernelCipher(input_layout.pack(hidden)[0], context)
    config = Llama3CKKSConfig()
    config.simulator.logN = 5

    with pytest.raises(RuntimeError, match="injected recovery failure"):
        _qkv_fhe_complex_token_diagonal_prepared(
            hidden,
            0.5 * np.eye(dimension),
            carrier_width=dimension,
            carrier_weight=np.eye(dimension),
            config=config,
            operator_config=LinearOperatorConfig(
                baby_steps=2, baby_anchor_step=2, max_plaintext_rows=4
            ),
            extra_rotations=(),
            crypto_context=context,
            input_ciphers=(input_cipher,),
            verify=False,
            wall_start=time.perf_counter(),
            weight_transform_seconds=0.0,
        )

    assert input_cipher.release_count == 0
    assert all(
        cipher.release_count == 1
        for cipher in context.ciphers
        if cipher is not input_cipher
    )
