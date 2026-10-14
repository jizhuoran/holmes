from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.backend import create_sim_context, release_if_supported
from llama3fhe.config import SimulatorConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.linear import (
    LinearOperator,
    LinearOperatorConfig,
    LinearWeight,
    gpu_packer,
    linear_rotations,
)
from llama3fhe.operators.linear.prefill.qkv_kernel import (
    _projection_weights,
    _structured_projection_weight,
)
from llama3fhe.operators.linear.structured_weight import (
    structured_linear_weight,
)


def test_structured_linear_weight_materializes_permutation_scale_and_complex_rows() -> None:
    source0 = np.arange(24, dtype=np.float32).reshape(4, 6)
    source1 = 100.0 + np.arange(24, dtype=np.float32).reshape(4, 6)
    descriptor = structured_linear_weight(
        (source0, source1),
        dimension=4,
        real_source=np.asarray([0, 1, -1, 0]),
        real_row=np.asarray([2, 1, -1, 3]),
        real_scale=np.asarray([0.5, -1.0, 0.0, 2.0]),
        imag_source=np.asarray([1, -1, -1, 1]),
        imag_row=np.asarray([0, -1, -1, 3]),
        imag_scale=np.asarray([-0.25, 0.0, 0.0, 0.75]),
        column=np.asarray([5, 2, -1, 0]),
        column_scale=np.asarray([1.0, 2.0, 3.0, -0.5]),
    )

    actual = descriptor.materialize(dtype=np.complex128)
    expected = np.zeros((4, 4), dtype=np.complex128)
    for output in range(4):
        for physical_input in range(4):
            column = descriptor.column[physical_input]
            if column < 0:
                continue
            real_source = descriptor.real_source[output]
            if real_source >= 0:
                expected[output, physical_input] += (
                    descriptor.column_scale[physical_input]
                    * descriptor.real_scale[output]
                    * descriptor.sources[real_source][
                        descriptor.real_row[output], column
                    ]
                )
            imag_source = descriptor.imag_source[output]
            if imag_source >= 0:
                expected[output, physical_input] += 1j * (
                    descriptor.column_scale[physical_input]
                    * descriptor.imag_scale[output]
                    * descriptor.sources[imag_source][
                        descriptor.imag_row[output], column
                    ]
                )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(descriptor.real_column, descriptor.column)
    np.testing.assert_array_equal(descriptor.imag_column, descriptor.column)


def test_structured_linear_weight_supports_independent_complex_columns() -> None:
    source = np.arange(30, dtype=np.float64).reshape(5, 6)
    shared_column = np.asarray([0, 0, 0, 0], dtype=np.int64)
    real_column = np.asarray([5, 1, -1, 2], dtype=np.int64)
    imag_column = np.asarray([0, 4, 3, -1], dtype=np.int64)
    descriptor = structured_linear_weight(
        (source,),
        dimension=4,
        real_source=np.zeros((4,), dtype=np.int64),
        real_row=np.asarray([4, 3, 2, 1]),
        real_scale=np.asarray([0.5, -1.0, 2.0, 0.25]),
        imag_source=np.zeros((4,), dtype=np.int64),
        imag_row=np.asarray([0, 1, 2, 3]),
        imag_scale=np.asarray([-0.25, 0.75, 1.5, -2.0]),
        column=shared_column,
        real_column=real_column,
        imag_column=imag_column,
        column_scale=np.asarray([1.0, 2.0, -0.5, 3.0]),
    )

    expected = np.zeros((4, 4), dtype=np.complex128)
    for output in range(4):
        for physical_input in range(4):
            if real_column[physical_input] >= 0:
                expected[output, physical_input] += (
                    descriptor.real_scale[output]
                    * source[descriptor.real_row[output], real_column[physical_input]]
                    * descriptor.column_scale[physical_input]
                )
            if imag_column[physical_input] >= 0:
                expected[output, physical_input] += 1j * (
                    descriptor.imag_scale[output]
                    * source[descriptor.imag_row[output], imag_column[physical_input]]
                    * descriptor.column_scale[physical_input]
                )

    np.testing.assert_array_equal(descriptor.materialize(), expected)
    np.testing.assert_array_equal(descriptor.column, shared_column)
    assert not np.array_equal(descriptor.real_column, descriptor.imag_column)


def test_structured_qkv_descriptor_matches_legacy_dense_physical_weight() -> None:
    rng = np.random.default_rng(26081601)
    seq_len = head_dim = 8
    hidden_dim = 64
    query_heads = 4
    key_value_heads = 2
    hidden = rng.normal(size=(seq_len, hidden_dim)).astype(np.float32)
    q_weight = rng.normal(
        size=(query_heads * head_dim, hidden_dim)
    ).astype(np.float32)
    k_weight = rng.normal(
        size=(key_value_heads * head_dim, hidden_dim)
    ).astype(np.float32)
    v_weight = rng.normal(
        size=(key_value_heads * head_dim, hidden_dim)
    ).astype(np.float32)
    column_scale = rng.uniform(0.5, 1.5, size=(hidden_dim,)).astype(
        np.float32
    )
    layout = AttentionPairLayout(
        seq_len=seq_len,
        head_dim=head_dim,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        slots=2 * seq_len * head_dim,
    )

    _, _, legacy = _projection_weights(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        attention_layout=layout,
        input_column_scale=column_scale,
    )
    _, structured, carrier_width = _structured_projection_weight(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        attention_layout=layout,
        input_column_scale=column_scale,
    )

    assert carrier_width == 32
    np.testing.assert_allclose(
        structured.materialize(dtype=np.complex128),
        legacy,
        atol=1e-7,
        rtol=1e-7,
    )


def test_structured_gate_up_and_down_page_maps_match_dense_oracles() -> None:
    rng = np.random.default_rng(26081602)
    dimension = 8
    active = 5
    gate = rng.normal(size=(active, dimension)).astype(np.float32)
    up = rng.normal(size=(active, dimension)).astype(np.float32)
    column_scale = rng.uniform(0.5, 1.5, size=(dimension,))
    rows = np.arange(dimension, dtype=np.int64)
    active_rows = rows < active
    gate_up = structured_linear_weight(
        (gate, up),
        dimension=dimension,
        real_source=np.where(active_rows, 0, -1),
        real_row=np.where(active_rows, rows, -1),
        real_scale=0.375,
        imag_source=np.where(active_rows, 1, -1),
        imag_row=np.where(active_rows, rows, -1),
        imag_scale=0.25,
        column_scale=column_scale,
    )
    gate_up_expected = np.zeros((dimension, dimension), dtype=np.complex128)
    gate_up_expected[:active] = (
        0.375 * gate + 0.25j * up
    ) * column_scale[None, :]
    np.testing.assert_allclose(gate_up.materialize(), gate_up_expected)

    down = rng.normal(size=(dimension, active)).astype(np.float32)
    columns = np.full((dimension,), -1, dtype=np.int64)
    columns[:active] = np.arange(active)
    down_page = structured_linear_weight(
        (down,),
        dimension=dimension,
        real_source=np.zeros((dimension,), dtype=np.int64),
        real_row=rows,
        real_scale=0.5,
        column=columns,
    )
    down_expected = np.zeros((dimension, dimension), dtype=np.float64)
    down_expected[:, :active] = 0.5 * down
    np.testing.assert_array_equal(down_page.materialize(), down_expected)


@pytest.mark.gpu
def test_structured_linear_weight_runs_through_real_fhe_gpu_packer() -> None:
    dimension = 64
    token_lanes = 128
    slots = dimension * token_lanes
    config = LinearOperatorConfig(
        baby_steps=8,
        baby_anchor_step=8,
        max_plaintext_rows=64,
        constant_cache_mode="none",
        raw_weight_policy="device",
    )
    simulator = SimulatorConfig(
        logN=14,
        maxLevelsRemaining=3,
        dnum=1,
        dcrtBits=59,
        firstMod=60,
        device="cuda",
    )
    context = create_sim_context(
        simulator,
        input_dim=dimension,
        block_out=token_lanes,
        rotations=linear_rotations(
            dimension=dimension,
            token_lanes=token_lanes,
            operator_config=config,
        ),
        include_default_rotations=False,
    )
    rng = np.random.default_rng(26081604)
    values = rng.normal(scale=0.02, size=(token_lanes, dimension))
    source = rng.normal(scale=0.02, size=(dimension, dimension)).astype(
        np.float32
    )
    permutation = rng.permutation(dimension)
    row_scale = rng.uniform(0.5, 1.5, size=(dimension,))
    column_scale = rng.uniform(0.5, 1.5, size=(dimension,))
    descriptor = structured_linear_weight(
        (source,),
        dimension=dimension,
        real_source=np.zeros((dimension,), dtype=np.int64),
        real_row=np.arange(dimension),
        real_scale=row_scale,
        column=permutation,
        column_scale=column_scale,
    )
    cipher = context.encrypt(values.T.reshape(-1), level=0, slots=slots)
    outputs = ()
    try:
        with LinearOperator(
            crypto_context=context, token_lanes=token_lanes, config=config
        ) as operator, LinearWeight(
            descriptor, dtype=np.float64, crypto_context=context
        ) as prepared:
            outputs, _ = operator.apply((cipher,), prepared)
        actual = np.asarray(context.decrypt(outputs[0])).real.reshape(
            dimension, token_lanes
        ).T
        expected = values @ descriptor.materialize(dtype=np.float64).T
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
    finally:
        release_if_supported(cipher)
        for output in outputs:
            release_if_supported(output)


@pytest.mark.gpu
def test_structured_triton_diagonals_match_dense_cpu_oracle() -> None:
    rng = np.random.default_rng(26081603)
    dimension = 16
    token_lanes = 8
    slots = dimension * token_lanes
    baby_steps = 4
    giant_base = 1
    giant_count = 2
    rows = np.arange(dimension, dtype=np.int64)
    descriptor = structured_linear_weight(
        (
            rng.normal(size=(dimension, dimension)).astype(np.float32),
            rng.normal(size=(dimension, dimension)).astype(np.float32),
        ),
        dimension=dimension,
        real_source=rows % 2,
        real_row=(3 * rows) % dimension,
        real_scale=rng.normal(size=dimension),
        imag_source=(rows + 1) % 2,
        imag_row=(5 * rows) % dimension,
        imag_scale=rng.normal(size=dimension),
        column=rows[::-1],
        real_column=(3 * rows + 1) % dimension,
        imag_column=(5 * rows + 2) % dimension,
        column_scale=rng.uniform(0.5, 1.5, size=dimension),
    )
    sources = tuple(
        gpu_packer.torch.as_tensor(source, device="cuda")
        for source in descriptor.sources
    )
    metadata = {
        name: gpu_packer.torch.as_tensor(
            getattr(descriptor, name), device="cuda"
        )
        for name in (
            "real_source",
            "real_row",
            "real_scale",
            "imag_source",
            "imag_row",
            "imag_scale",
            "column",
            "column_scale",
            "real_column",
            "imag_column",
        )
    }
    actual = (
        gpu_packer.pack_structured_token_lane_bsgs_weight_chunk_triton_gpu(
            sources,
            metadata,
            slots=slots,
            dimension=dimension,
            token_lanes=token_lanes,
            baby_steps=baby_steps,
            giant_base=giant_base,
            giant_count=giant_count,
            device="cuda",
        )
    )
    gpu_packer.synchronize()
    actual = actual.cpu().numpy()

    feature = np.arange(slots, dtype=np.int64) // token_lanes
    def expected_diagonals(dense: np.ndarray) -> np.ndarray:
        expected = []
        for giant_relative in range(giant_count):
            giant = giant_base + giant_relative
            output_feature = (feature - giant * baby_steps) % dimension
            for baby in range(baby_steps):
                input_feature = (feature + baby) % dimension
                expected.append(dense[output_feature, input_feature])
        return np.stack(expected)

    np.testing.assert_allclose(
        actual,
        expected_diagonals(descriptor.materialize(dtype=np.complex128)),
        atol=1e-12,
        rtol=1e-12,
    )

    # Direct packer callers predating side-specific columns still upload only
    # ``column``.  The GPU entry point must preserve that shared-map contract.
    legacy_metadata = {
        name: value
        for name, value in metadata.items()
        if name not in {"real_column", "imag_column"}
    }
    legacy_actual = (
        gpu_packer.pack_structured_token_lane_bsgs_weight_chunk_triton_gpu(
            sources,
            legacy_metadata,
            slots=slots,
            dimension=dimension,
            token_lanes=token_lanes,
            baby_steps=baby_steps,
            giant_base=giant_base,
            giant_count=giant_count,
            device="cuda",
        )
    )
    gpu_packer.synchronize()
    legacy_descriptor = structured_linear_weight(
        descriptor.sources,
        dimension=dimension,
        real_source=descriptor.real_source,
        real_row=descriptor.real_row,
        real_scale=descriptor.real_scale,
        imag_source=descriptor.imag_source,
        imag_row=descriptor.imag_row,
        imag_scale=descriptor.imag_scale,
        column=descriptor.column,
        column_scale=descriptor.column_scale,
    )
    np.testing.assert_allclose(
        legacy_actual.cpu().numpy(),
        expected_diagonals(legacy_descriptor.materialize(dtype=np.complex128)),
        atol=1e-12,
        rtol=1e-12,
    )
