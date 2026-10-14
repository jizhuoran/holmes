"""Shared pytest configuration: the ``gpu`` marker.

Tests marked ``@pytest.mark.gpu`` create real CUDA contexts (several GiB of
device memory each). They are skipped automatically when no CUDA device is
available, and can be deselected explicitly — e.g. while a GPU measurement
is running — with ``pytest -m "not gpu"``.
"""

from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import easyfhe

        return bool(easyfhe.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items) -> None:
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="needs an NVIDIA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
