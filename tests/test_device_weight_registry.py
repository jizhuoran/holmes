"""CPU tests of :class:`llama3fhe.operators.linear.weight.DeviceWeightRegistry` (views resolve to strided device views)."""

from __future__ import annotations

import numpy as np
import pytest

import easyfhe as torch

from llama3fhe.operators.linear.weight import DeviceWeightRegistry


def _registry_with_root(shape=(8, 6)):
    root = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    device = torch.from_numpy(root).clone()  # stands in for the device copy (same shape, contiguous)
    registry = DeviceWeightRegistry()
    registry.register(root, device)
    return registry, root, device


def _expect(registry, array):
    view = registry.view(array, wait=False)
    assert view is not None
    np.testing.assert_array_equal(view.numpy(), array)
    return view


def test_root_and_row_and_column_slices_resolve() -> None:
    registry, root, _ = _registry_with_root()
    _expect(registry, root)
    _expect(registry, root[2:5])          # gate[a:b] page
    _expect(registry, root[:, 1:4])       # down[:, a:b] page (non-contiguous)
    _expect(registry, root[1:7][2:4, 1:3])  # view of a view
    _expect(registry, np.asarray(root[3:4], dtype=np.float32))  # asarray keeps the view
    assert registry.hits == 5 and registry.misses == 0


def test_view_is_strided_not_copied() -> None:
    registry, root, device = _registry_with_root()
    view = registry.view(root[:, 1:4], wait=False)
    assert view.data_ptr() == device.data_ptr() + 1 * 4
    assert tuple(view.stride()) == (6, 1)
    view = registry.view(root[2:5], wait=False)
    assert view.data_ptr() == device.data_ptr() + 2 * 6 * 4


def test_misses() -> None:
    registry, root, _ = _registry_with_root()
    assert registry.view(np.ones((3, 3), dtype=np.float32), wait=False) is None  # foreign array
    assert registry.view(root.astype(np.float64), wait=False) is None            # copy -> no base link
    assert registry.view(root[:0], wait=False) is None                            # empty
    assert registry.view("not an array", wait=False) is None
    assert registry.view(root[::-1], wait=False) is None                          # negative stride
    assert registry.misses == 4
    registry.unregister(root)
    assert len(registry) == 0
    assert registry.view(root[1:2], wait=False) is None


def test_register_rejects_shape_mismatch_and_non_contiguous() -> None:
    registry = DeviceWeightRegistry()
    root = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        registry.register(root, torch.zeros((4, 5)))
    with pytest.raises(ValueError):
        registry.register(root[:, ::2], torch.zeros((4, 2)))


def test_torch_backed_root_keeps_base_link() -> None:
    """Pinned-buffer style root: a NumPy array whose base is a tensor still anchors its views."""

    tensor = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    root = tensor.numpy()
    registry = DeviceWeightRegistry()
    registry.register(root, tensor.clone())
    _expect(registry, root[2:5])
    _expect(registry, root[:, 1:4])
    _expect(registry, np.asarray(root, dtype=np.float32))
