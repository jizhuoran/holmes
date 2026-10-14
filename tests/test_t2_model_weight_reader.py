"""The T2 model runner's direct npz reader (``preadv`` into pinned buffers, no zipfile CRC pass)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from bench.t2_model import WEIGHT_KEYS, npz_stored_members, read_into

SHAPES = {
    "input_layernorm": (64,), "q_proj": (64, 64), "k_proj": (16, 64), "v_proj": (16, 64), "o_proj": (64, 64),
    "post_attention_layernorm": (64,), "gate_proj": (224, 64), "up_proj": (224, 64), "down_proj": (64, 224),
}


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, shape in SHAPES.items()}


def test_stored_members_read_back_identically(tmp_path, arrays):
    path = tmp_path / "layer3_weights.npz"
    np.savez(path, **arrays)
    members = npz_stored_members(path)
    assert members is not None and set(members) == set(WEIGHT_KEYS)
    fd = os.open(path, os.O_RDONLY)
    try:
        for name, (offset, shape) in members.items():
            assert shape == SHAPES[name]
            buffer = np.empty(shape, dtype=np.float32)
            read_into(fd, memoryview(buffer).cast("B"), offset)
            np.testing.assert_array_equal(buffer, arrays[name])
    finally:
        os.close(fd)


def test_compressed_or_wrong_dtype_falls_back(tmp_path, arrays):
    compressed = tmp_path / "layer4_weights.npz"
    np.savez_compressed(compressed, **arrays)
    assert npz_stored_members(compressed) is None
    half = tmp_path / "layer5_weights.npz"
    np.savez(half, **{name: (value.astype(np.float16) if name == "q_proj" else value) for name, value in arrays.items()})
    assert npz_stored_members(half) is None
    missing = tmp_path / "layer6_weights.npz"
    np.savez(missing, **{name: value for name, value in arrays.items() if name != "down_proj"})
    assert npz_stored_members(missing) is None


def test_read_into_loops_over_short_reads(tmp_path):
    path = tmp_path / "blob.bin"
    payload = np.arange(1 << 16, dtype=np.uint8)
    path.write_bytes(b"xx" + payload.tobytes())
    fd = os.open(path, os.O_RDONLY)
    try:
        out = np.empty(payload.shape, dtype=np.uint8)
        read_into(fd, memoryview(out), 2)
        np.testing.assert_array_equal(out, payload)
        with pytest.raises(OSError):
            read_into(fd, memoryview(np.empty(8, dtype=np.uint8)), 2 + payload.nbytes)  # past EOF -> short read
    finally:
        os.close(fd)
