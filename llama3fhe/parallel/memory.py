"""Device-memory accounting for one T2 worker."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

GIB = float(2**30)


def cuda_memory_gib(device: str = "cuda") -> dict[str, float | bool]:
    """Allocator + device memory of ``device`` in GiB (``available`` flag first)."""

    from llama3fhe.backend import cuda_memory_snapshot

    raw = cuda_memory_snapshot(str(device))
    if not raw.get("available", False):
        return {"available": False}
    return {
        "available": True,
        "alloc_gib": raw.get("allocator_allocated_bytes", 0) / GIB,
        "reserved_gib": raw.get("allocator_reserved_bytes", 0) / GIB,
        "peak_alloc_gib": raw.get("allocator_peak_allocated_bytes", 0) / GIB,
        "peak_reserved_gib": raw.get("allocator_peak_reserved_bytes", 0) / GIB,
        "device_used_gib": raw.get("device_used_bytes", 0) / GIB,
        "device_free_gib": raw.get("device_free_bytes", 0) / GIB,
        "device_total_gib": raw.get("device_total_bytes", 0) / GIB,
        "alloc_retries": _alloc_retries(str(device)),
    }


def _alloc_retries(device: str) -> int:
    """``torch`` caching-allocator ``num_alloc_retries`` (cumulative per process).

    Each retry is one failed ``cudaMalloc`` followed by a device-wide
    synchronize + release of every cached free block; a rising count between
    two snapshots is the allocator, not the kernels, in the stage time.
    """

    try:
        import torch

        return int(torch.cuda.memory_stats(device).get("num_alloc_retries", 0))
    except Exception:
        return 0


def reset_peak_memory_stats(device: str = "cuda") -> None:
    try:
        import easyfhe

        try:
            easyfhe.cuda.reset_peak_memory_stats(str(device))
        except TypeError:
            easyfhe.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _tensor_bytes(value: Any, *, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            pointer = int(value.data_ptr()) if hasattr(value, "data_ptr") else id(value)
            if pointer in seen:
                return 0
            seen.add(pointer)
            return int(value.numel()) * int(value.element_size())
        except Exception:
            return 0
    if isinstance(value, dict):
        return sum(
            _tensor_bytes(key, seen=seen) + _tensor_bytes(item, seen=seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set)):
        return sum(_tensor_bytes(item, seen=seen) for item in value)
    return 0


_NATIVE_CACHES = (
    "_rotation_key_cuda_cache",
    "_precompute_auto_cuda_cache",
    "_inverse_precompute_auto_cuda_cache",
    "precompute_auto_maps_cache",
)


def native_cache_inventory(native: Any, *, details: bool = False) -> dict[str, Any]:
    """Size of the resident CUDA key/precompute caches of a native EasyFHE context.

    ``native`` is ``EasyFHEContext.context``.  Returns one entry per cache
    with ``entries`` and ``bytes`` (plus ``entry_details`` when requested) and
    ``total_bytes`` / ``total_gib`` across all of them.
    """

    result: dict[str, Any] = {
        "context": {
            "L": int(getattr(native, "L", 0)),
            "N": int(getattr(native, "N", 0)),
            "alpha": int(getattr(native, "alpha", 0)),
            "auto_load_keys_resolved": bool(getattr(native, "auto_load_keys_resolved", False)),
        }
    }
    total = 0
    for name in _NATIVE_CACHES:
        cache = getattr(native, name, None)
        if not isinstance(cache, dict):
            continue
        size = int(_tensor_bytes(cache))
        total += size
        entry: dict[str, Any] = {"entries": len(cache), "bytes": size}
        if details:
            entry["entry_details"] = [
                {"key": repr(key), "bytes": int(_tensor_bytes(value))}
                for key, value in sorted(cache.items(), key=lambda item: repr(item[0]))
            ]
        result[name] = entry
    result["total_bytes"] = int(total)
    result["total_gib"] = total / GIB
    return result


def bootstrap_constant_cache_gib(program: Any) -> dict[str, Any]:
    """``ConstantBundle.cache_info()`` with byte fields converted to GiB."""

    constants = getattr(program, "constants", None)
    info = getattr(constants, "cache_info", None)
    if not callable(info):
        return {}
    try:
        raw = info()
    except Exception as exc:
        return {"error": repr(exc)}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key.endswith("_bytes") and isinstance(value, (int, float)):
            out[key[: -len("_bytes")] + "_gib"] = float(value) / GIB
        else:
            out[key] = value
    return out


@dataclass
class MemoryProbe:
    """Timeline of labelled memory snapshots for one rank."""

    rank: int
    device: str = "cuda"
    echo: bool = False
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def snap(self, label: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "rank": int(self.rank),
            "label": str(label),
            "t": time.perf_counter(),
            **cuda_memory_gib(self.device),
        }
        if extra:
            row.update(extra)
        self.timeline.append(row)
        if self.echo:
            print(json.dumps(row, default=str), flush=True)
        return row

    def reset_peak(self) -> None:
        reset_peak_memory_stats(self.device)

    def peak_since(self, label_prefix: str) -> float:
        rows = [row for row in self.timeline if str(row["label"]).startswith(label_prefix)]
        return max((float(row.get("peak_alloc_gib", 0.0)) for row in rows), default=0.0)

    def last(self) -> dict[str, Any]:
        if not self.timeline:
            raise ValueError("no snapshots recorded")
        return self.timeline[-1]


__all__ = [
    "GIB",
    "MemoryProbe",
    "bootstrap_constant_cache_gib",
    "cuda_memory_gib",
    "native_cache_inventory",
    "reset_peak_memory_stats",
]
