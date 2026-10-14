"""Spawn one worker process per rank and collect their results.

The launcher runs on the host side, imports nothing CUDA-related itself and
pins each child to one physical card with ``CUDA_VISIBLE_DEVICES=<device>``
*before* the child imports ``easyfhe``.  Inside the child the card is
therefore ``cuda`` and every EasyFHE context can use the exact device string
``"cuda"`` (some loaders still compare against that literal).

The rank -> device map is :func:`expand_devices`: one rank per card by
default, or ``ranks_per_device`` ranks sharing each card when the host has
fewer cards than ``preset.world_size`` (the stations are written for one T2
slot per rank, so the 16-ciphertext layout always needs 8 ranks; a 4 x 96 GB
host runs them as two ~40 GiB processes per card).  The transport backend is
chosen per child through ``T2_TRANSPORT`` (:func:`~.transport.build_transport`).

``entry(rank, preset, endpoints, **kwargs)`` runs in the child and returns a
picklable object; :func:`spawn_workers` returns them indexed by rank, or
re-raises the first child failure (all children are terminated on any
failure so ``nvidia-smi`` shows no orphans).
"""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .presets import ParallelPreset
from .prefill.transport import TransportEndpoints, build_transport_endpoints


@dataclass
class WorkerOutcome:
    rank: int
    ok: bool
    result: Any = None
    error: str | None = None
    wall_seconds: float = 0.0


def _child_main(
    rank: int,
    preset: ParallelPreset,
    endpoints: TransportEndpoints,
    entry: Callable[..., Any],
    kwargs: dict[str, Any],
    result_queue,
) -> None:
    started = time.perf_counter()
    try:
        result = entry(rank, preset, endpoints, **kwargs)
        result_queue.put(
            WorkerOutcome(rank=rank, ok=True, result=result, wall_seconds=time.perf_counter() - started)
        )
    except BaseException as exc:  # noqa: BLE001 - report every failure to the parent
        result_queue.put(
            WorkerOutcome(
                rank=rank,
                ok=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                wall_seconds=time.perf_counter() - started,
            )
        )
        raise


def _mp_context():
    return multiprocessing.get_context("spawn")


def expand_devices(
    world_size: int,
    devices: Sequence[int] | None = None,
    *,
    ranks_per_device: int = 1,
) -> tuple[int, ...]:
    """Rank -> physical CUDA index for ``world_size`` ranks.

    * ``devices`` given: one entry per rank, repeats allowed (ranks sharing a
      card); ``ranks_per_device`` must then match the repetition.
    * ``devices`` omitted: cards ``0 .. world_size/ranks_per_device - 1``, each
      repeated ``ranks_per_device`` times (adjacent ranks share a card, so a
      ``mlp_feature_shard_k=2`` group is card-local on a 4-card host).
    """

    world_size = int(world_size)
    ranks_per_device = int(ranks_per_device)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if ranks_per_device <= 0 or world_size % ranks_per_device:
        raise ValueError(
            f"ranks_per_device {ranks_per_device} must be a positive divisor of world_size {world_size}"
        )
    if devices is None:
        cards = world_size // ranks_per_device
        return tuple(card for card in range(cards) for _ in range(ranks_per_device))
    devices = tuple(int(device) for device in devices)
    if len(devices) != world_size:
        raise ValueError(f"need one device per rank ({world_size}), got {devices}")
    if any(device < 0 for device in devices):
        raise ValueError(f"device indices must be non-negative, got {devices}")
    counts = {device: devices.count(device) for device in set(devices)}
    if set(counts.values()) != {ranks_per_device}:
        raise ValueError(
            f"devices {devices} put {sorted(counts.values())} ranks per card; "
            f"expected exactly {ranks_per_device} (--ranks-per-device)"
        )
    return devices


def spawn_workers(
    preset: ParallelPreset,
    entry: Callable[..., Any],
    *,
    kwargs: dict[str, Any] | None = None,
    devices: Sequence[int] | None = None,
    ranks_per_device: int = 1,
    transport: str | None = None,
    env: dict[str, str] | None = None,
    join_timeout: float | None = 3600.0,
) -> list[WorkerOutcome]:
    """Run ``entry`` in ``preset.world_size`` processes and return one outcome per rank."""

    world_size = int(preset.world_size)
    devices = expand_devices(world_size, devices, ranks_per_device=ranks_per_device)
    kwargs = {} if kwargs is None else dict(kwargs)

    context = _mp_context()
    endpoints = build_transport_endpoints(world_size, context=context)
    result_queue = context.Queue()
    base_env = dict(os.environ)
    base_env.setdefault("OMP_NUM_THREADS", "1")
    base_env.setdefault("LLAMA_FHE_PLAIN_CACHE_GB", "0")
    base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # A child killed by the kernel (host OOM -> SIGKILL) cannot report through
    # the queue; unbuffered stdio keeps whatever it printed before that.
    base_env.setdefault("PYTHONUNBUFFERED", "1")
    base_env.setdefault("LLAMA_FHE_TRANSFORM_HOIST", str(preset.transform_hoist_strategy))
    base_env.setdefault("T2_TRANSPORT", str(preset.transport) if transport is None else str(transport))
    if env:
        base_env.update({str(k): str(v) for k, v in env.items()})

    processes = []
    try:
        for rank in range(world_size):
            child_env = dict(base_env)
            child_env["CUDA_VISIBLE_DEVICES"] = str(devices[rank])
            child_env["T2_RANK"] = str(rank)
            child_env["T2_WORLD_SIZE"] = str(world_size)
            process = context.Process(
                target=_with_env,
                args=(child_env, _child_main, rank, preset, endpoints, entry, kwargs, result_queue),
                name=f"t2-rank{rank}",
                daemon=False,
            )
            process.start()
            processes.append(process)

        outcomes: dict[int, WorkerOutcome] = {}
        deadline = None if join_timeout is None else time.perf_counter() + float(join_timeout)
        while len(outcomes) < world_size:
            # Poll in short slices: a child the kernel killed (host OOM ->
            # SIGKILL, exitcode -9) never reports, and without this check the
            # survivors would sit at their next barrier -- each holding its
            # card and tens of GiB of host keys -- until ``join_timeout``.
            remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
            slice_timeout = 5.0 if remaining is None else min(5.0, remaining)
            try:
                outcome = result_queue.get(timeout=slice_timeout)
            except Exception as exc:
                _drain_queue(result_queue, outcomes)
                for outcome in outcomes.values():
                    if not outcome.ok:
                        raise RuntimeError(f"rank {outcome.rank} failed:\n{outcome.error}") from None
                for rank, process in enumerate(processes):
                    if rank not in outcomes and not process.is_alive():
                        raise RuntimeError(
                            f"rank {rank} exited with code {process.exitcode} without reporting "
                            f"(host OOM kill shows as -9); reported={sorted(outcomes)}"
                        ) from None
                if remaining is not None and remaining <= 0.0:
                    dead = [p.name for p in processes if not p.is_alive()]
                    raise TimeoutError(
                        f"workers did not all report within {join_timeout}s; "
                        f"reported={sorted(outcomes)} exited={dead}"
                    ) from exc
                continue
            outcomes[int(outcome.rank)] = outcome
            if not outcome.ok:
                raise RuntimeError(f"rank {outcome.rank} failed:\n{outcome.error}")
        for process in processes:
            process.join(timeout=60.0)
        return [outcomes[rank] for rank in range(world_size)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10.0)


def _drain_queue(result_queue, outcomes: dict[int, WorkerOutcome]) -> None:
    while True:
        try:
            outcome = result_queue.get_nowait()
        except Exception:
            return
        outcomes[int(outcome.rank)] = outcome


def _with_env(env: dict[str, str], target: Callable[..., Any], *args: Any) -> None:
    os.environ.clear()
    os.environ.update(env)
    target(*args)


__all__ = ["WorkerOutcome", "expand_devices", "spawn_workers"]
