"""CUDA device selection for a T2 worker.

The launcher pins each worker to one physical card with
``CUDA_VISIBLE_DEVICES``; inside the process that card is ``cuda`` / ``cuda:0``
and the EasyFHE context is created with the exact device string ``"cuda"``
(some EasyFHE key/bootstrap loaders still compare against that literal).
:func:`device_scope` is therefore only needed by single-process tools
(``bench/probes/p9_l3_bs_rmsnorm_probe.py --device N``) that pick a card without
the launcher.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


def _easyfhe():
    import easyfhe

    return easyfhe


def visible_device_count() -> int:
    try:
        return int(_easyfhe().cuda.device_count())
    except Exception:
        return 0


def pinned_device_index() -> int | None:
    """Physical index from ``CUDA_VISIBLE_DEVICES`` when exactly one is exposed."""

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 1 or not parts[0].isdigit():
        return None
    return int(parts[0])


@contextmanager
def device_scope(rank: int) -> Iterator[None]:
    """Select physical CUDA device ``rank`` for the duration of the block."""

    if isinstance(rank, bool) or int(rank) < 0:
        raise ValueError(f"CUDA device index must be non-negative, got {rank!r}")
    cuda = _easyfhe().cuda
    previous = int(cuda.current_device())
    selected = int(rank)
    if previous != selected:
        cuda.set_device(selected)
    try:
        yield
    finally:
        if previous != selected:
            cuda.set_device(previous)


def synchronize_current_device() -> None:
    _easyfhe().cuda.synchronize()


__all__ = [
    "device_scope",
    "pinned_device_index",
    "synchronize_current_device",
    "visible_device_count",
]
