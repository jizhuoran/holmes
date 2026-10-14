"""CPU tests of the host-staging transport with real spawned processes."""

from __future__ import annotations

import multiprocessing
import time

import numpy as np
import pytest

from llama3fhe.parallel.prefill.transport import (
    HostStagingTransport,
    PinnedHandle,
    PinnedStagingTransport,
    SharedArrayHandle,
    _payload_has_arrays,
    _stage_shared,
    _unstage_shared,
    build_transport_endpoints,
)
from llama3fhe.parallel.wire import CipherPacket, cipher_metadata


def _worker(rank, endpoints, results, shared_memory=True):
    transport = HostStagingTransport(rank, endpoints, default_timeout=60.0, shared_memory=shared_memory)
    world = transport.world_size
    trace = []

    # round 1: all_gather stages one block per rank (same handle to every peer)
    gathered = transport.all_gather("lanes", np.full(4, rank, dtype=np.int64))
    assert [int(item[0]) for item in gathered] == list(range(world))
    trace.append(transport.staged_blocks)

    # round 2: all_to_all stages one block per destination
    to_send = [np.array([rank * 100 + dst]) for dst in range(world)]
    received = transport.all_to_all("partials", to_send)
    assert [int(item[0]) for item in received] == [src * 100 + rank for src in range(world)]
    trace.append(transport.staged_blocks)

    # point-to-point never stages, even with ndarray payloads; out-of-order
    # delivery: rank 0 sends "late" before "early" to rank 1
    if rank == 0:
        transport.send(1, "late", np.array([7, 7]))
        transport.send(1, "early", "E")
    if rank == 1:
        assert transport.recv(0, "early") == "E"
        assert transport.recv(0, "late").tolist() == [7, 7]
    trace.append(transport.staged_blocks)

    transport.barrier()  # round 3: releases round 1
    trace.append(transport.staged_blocks)

    # round 4: a bigger array (1 MiB) through all_gather, releases round 2
    big = transport.all_gather("big", np.full(131072, rank, dtype=np.uint64))
    assert all(int(item[0]) == src and item.dtype == np.uint64 for src, item in enumerate(big))
    trace.append(transport.staged_blocks)

    # gather_to is asymmetric -> pickled, never staged, not a symmetric round
    # (own sequence): it releases nothing, arrays still fine
    root = transport.gather_to(0, "report", np.array([rank * rank]))
    if rank == 0:
        assert [int(item[0]) for item in root] == [r * r for r in range(world)]
    else:
        assert root is None
    trace.append(transport.staged_blocks)

    transport.barrier()  # round 5: releases round 3 (the empty barrier), round 4 stays mapped
    trace.append(transport.staged_blocks)
    transport.close()
    assert transport.staged_blocks == 0
    row = transport.stats.to_dict()
    row["trace"] = trace
    results.put((rank, row))


@pytest.mark.parametrize("world_size,shared_memory", [(2, True), (4, True), (2, False)])
def test_collectives_across_spawned_processes(world_size, shared_memory) -> None:
    context = multiprocessing.get_context("spawn")
    endpoints = build_transport_endpoints(world_size, context=context)
    results = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, endpoints, results, shared_memory))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    stats = {}
    for _ in range(world_size):
        rank, row = results.get(timeout=120)
        stats[rank] = row
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert sorted(stats) == list(range(world_size))
    peers = world_size - 1
    for row in stats.values():
        assert row["collectives"] == 6
        assert row["messages_sent"] >= 2 * peers
        assert row["messages_received"] >= 2 * peers
        if shared_memory:
            # staged blocks after rounds 1..4, the gather and round 5 (p2p sends never stage)
            assert row["trace"] == [1, 1 + peers, 1 + peers, peers, 1, 1, 1]
        else:
            assert row["trace"] == [0] * 7


def _late_receiver(rank, endpoints, results):
    """Regression: ``all_to_all -> gather_to -> barrier`` with one slow receiver.

    Before ``gather_to`` got its own sequence it counted as a symmetric round,
    so a fast non-root rank reaching the barrier freed the blocks of the
    ``all_to_all`` two "rounds" earlier while the slow rank was still copying
    them out (``FileNotFoundError: /psm_...`` on the 8x4090 layer run).
    """

    transport = HostStagingTransport(rank, endpoints, default_timeout=60.0)
    world = transport.world_size
    if rank == 1:
        # the window is between this rank's sends and its receives: pause there
        # so every other rank has received, reported and entered the barrier
        original_recv = transport.recv

        def slow_recv(src, tag, *, timeout=None):
            time.sleep(1.0)
            return original_recv(src, tag, timeout=timeout)

        transport.recv = slow_recv
    received = transport.all_to_all(
        "partials", [np.full(65536, rank * 100 + dst, dtype=np.int64) for dst in range(world)]
    )
    assert [int(item[0]) for item in received] == [src * 100 + rank for src in range(world)]
    transport.gather_to(0, "report", {"rank": rank})
    transport.barrier()
    transport.close()
    results.put((rank, transport.stats.to_dict()))


def test_gather_to_does_not_release_a_peers_pending_blocks() -> None:
    context = multiprocessing.get_context("spawn")
    world_size = 3
    endpoints = build_transport_endpoints(world_size, context=context)
    results = context.Queue()
    processes = [context.Process(target=_late_receiver, args=(rank, endpoints, results)) for rank in range(world_size)]
    for process in processes:
        process.start()
    stats = {}
    for _ in range(world_size):
        rank, row = results.get(timeout=120)
        stats[rank] = row
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert sorted(stats) == list(range(world_size))
    assert all(row["collectives"] == 3 for row in stats.values())


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.device = "cuda:0"
        self.shape = self.array.shape
        self.dtype = "torch.uint64"

    def cpu(self):
        clone = _Tensor(self.array)
        clone.device = "cpu"
        return clone

    def numpy(self):
        return self.array


def test_stage_and_unstage_round_trip_cipher_packet() -> None:
    from types import SimpleNamespace

    cipher = SimpleNamespace(
        cv=[_Tensor(np.full((1, 3, 8), 5, dtype=np.uint64)), _Tensor(np.full((1, 3, 8), 6, dtype=np.uint64))],
        state=SimpleNamespace(cur_limbs=3, scale_degree=1, scaling_factor=2.0**40),
        slots=32768,
        is_ext=False,
        batch_size=1,
    )
    from llama3fhe.parallel.wire import pack_cipher

    packet = pack_cipher(cipher, tag="attn_in/lane/0")
    blocks = []
    staged = _stage_shared([packet, "meta", 3], blocks)
    assert len(blocks) == 2
    assert isinstance(staged[0], CipherPacket)
    assert all(isinstance(c, SharedArrayHandle) for c in staged[0].components)
    # the original packet is untouched (dataclasses.replace made a copy)
    assert all(isinstance(c, np.ndarray) for c in packet.components)
    assert staged[0].metadata == packet.metadata and staged[0].state is packet.state

    restored = _unstage_shared(staged)
    assert restored[1:] == ["meta", 3]
    assert all(isinstance(c, np.ndarray) for c in restored[0].components)
    assert [int(c[0, 0, 0]) for c in restored[0].components] == [5, 6]
    assert restored[0].metadata == cipher_metadata(cipher)
    for block in blocks:
        block.close()
        block.unlink()


def test_single_rank_protocol_errors() -> None:
    endpoints = build_transport_endpoints(2)
    transport = HostStagingTransport(0, endpoints, default_timeout=0.05)
    with pytest.raises(ValueError):
        transport.send(0, "self", None)
    with pytest.raises(IndexError):
        transport.send(5, "oob", None)
    with pytest.raises(TimeoutError):
        transport.recv(1, "never")
    with pytest.raises(ValueError):
        transport.all_to_all("bad", [None])
    with pytest.raises(IndexError):
        HostStagingTransport(2, endpoints)


def test_transport_contract_and_factory(monkeypatch) -> None:
    """Every backend satisfies the :class:`Transport` protocol; the factory picks by name / env."""

    from llama3fhe.parallel.emulation import EmulationWorld, InProcessTransport
    from llama3fhe.parallel.prefill.transport import (
        DEFAULT_TRANSPORT,
        TRANSPORT_BACKENDS,
        Transport,
        build_transport,
    )

    assert DEFAULT_TRANSPORT == "host" and set(TRANSPORT_BACKENDS) == {"host", "pinned", "inprocess"}
    endpoints = build_transport_endpoints(2)

    monkeypatch.delenv("T2_TRANSPORT", raising=False)
    host = build_transport(0, endpoints, default_timeout=0.05)
    assert isinstance(host, HostStagingTransport) and isinstance(host, Transport)
    assert host.rank == 0 and host.world_size == 2 and host.peers == (1,)
    assert host.device_payloads is False

    monkeypatch.setenv("T2_TRANSPORT", "host")
    assert isinstance(build_transport(1, endpoints), HostStagingTransport)

    monkeypatch.setenv("T2_TRANSPORT", "pinned")
    pinned = build_transport(1, endpoints, default_timeout=0.05, arena_mb=1)
    assert isinstance(pinned, PinnedStagingTransport) and isinstance(pinned, Transport)
    assert pinned.device_payloads is True and pinned.arena.slab_bytes == 1 << 20
    assert pinned.rank == 1 and pinned.peers == (0,)
    pinned.close()  # never touched CUDA: no slab was created
    monkeypatch.setenv("T2_PINNED_ARENA_MB", "3")
    pinned = build_transport(0, endpoints, kind="pinned", default_timeout=0.05)
    assert pinned.arena.slab_bytes == 3 << 20
    pinned.close()
    monkeypatch.delenv("T2_PINNED_ARENA_MB", raising=False)
    with pytest.raises(ValueError):
        build_transport(0, None, kind="pinned")  # needs endpoints

    world = EmulationWorld(2)
    inproc = build_transport(1, None, kind="inprocess", world=world)
    assert isinstance(inproc, InProcessTransport) and isinstance(inproc, Transport)
    assert inproc.rank == 1 and inproc.world_size == 2 and inproc.peers == (0,)
    monkeypatch.setenv("T2_TRANSPORT", "inprocess")
    assert isinstance(build_transport(0, None, world=world), InProcessTransport)

    with pytest.raises(ValueError):
        build_transport(0, None, kind="inprocess")  # needs world=
    with pytest.raises(ValueError):
        build_transport(0, None, kind="host")  # needs endpoints
    with pytest.raises(ValueError):
        build_transport(0, endpoints, kind="nccl")
    monkeypatch.setenv("T2_TRANSPORT", "bogus")
    with pytest.raises(ValueError):
        build_transport(0, endpoints)


class _DeviceTensor(_Tensor):
    """A fake *device* tensor: what ``pack_cipher(host=False)`` leaves in a packet."""

    is_cuda = True

    def data_ptr(self):
        return self.array.ctypes.data

    def numel(self):
        return int(self.array.size)

    def element_size(self):
        return int(self.array.itemsize)

    def cpu(self):
        clone = _Tensor(self.array)
        clone.device = "cpu"
        return clone


def test_device_tensors_are_staged_to_host_without_a_hook() -> None:
    """The ``"host"`` backend copies device components itself (``host=False`` is safe anywhere)."""

    tensor = _DeviceTensor(np.arange(6, dtype=np.uint64).reshape(1, 2, 3))
    assert _payload_has_arrays([tensor]) and _payload_has_arrays((("k", tensor),))
    blocks = []
    staged = _stage_shared(("k", tensor), blocks)
    assert len(blocks) == 1 and isinstance(staged[1], SharedArrayHandle)
    assert staged[1].shape == (1, 2, 3) and staged[1].dtype == "uint64"
    restored = _unstage_shared(staged)
    assert isinstance(restored[1], np.ndarray) and restored[1].tolist() == tensor.array.tolist()
    for block in blocks:
        block.close()
        block.unlink()


def test_device_hook_and_pinned_handle_fallback() -> None:
    """A hook may claim device tensors; ``None`` falls back to the host copy.

    Without a handle hook a :class:`PinnedHandle` is copied out as NumPy (the
    receiver-side fallback), honouring its offset in the slab.
    """

    from multiprocessing import shared_memory

    tensor = _DeviceTensor(np.full((2, 4), 9, dtype=np.uint64))
    claimed = []

    def hook(value):
        claimed.append(value)
        return "HANDLE" if len(claimed) == 1 else None

    blocks = []
    staged = _stage_shared([tensor, tensor], blocks, device_hook=hook)
    assert staged[0] == "HANDLE" and isinstance(staged[1], SharedArrayHandle) and len(blocks) == 1
    for block in blocks:
        block.close()
        block.unlink()

    block = shared_memory.SharedMemory(create=True, size=4096)
    try:
        view = np.ndarray((2, 4), dtype=np.uint64, buffer=block.buf, offset=512)
        view[...] = 11
        del view
        handle = PinnedHandle(name=block.name, offset=512, shape=(2, 4), dtype="torch.uint64", nbytes=64)
        assert handle.numpy_dtype == "uint64"
        out = _unstage_shared([handle])
        assert isinstance(out[0], np.ndarray) and out[0].shape == (2, 4) and int(out[0][1, 3]) == 11
        seen = []
        assert _unstage_shared((handle,), handle_hook=lambda h: seen.append(h) or "T")[0] == "T"
        assert seen == [handle]
    finally:
        block.close()
        block.unlink()


def _pinned_worker(rank, endpoints, results):
    """The pinned transport with NumPy payloads behaves exactly like ``"host"`` (no CUDA)."""

    transport = PinnedStagingTransport(rank, endpoints, default_timeout=60.0, arena_mb=1)
    world = transport.world_size
    gathered = transport.all_gather("lanes", np.full(4, rank, dtype=np.int64))
    assert [int(item[0]) for item in gathered] == list(range(world))
    received = transport.all_to_all("partials", [np.array([rank * 100 + dst]) for dst in range(world)])
    assert [int(item[0]) for item in received] == [src * 100 + rank for src in range(world)]
    transport.barrier()
    root = transport.gather_to(0, "report", {"rank": rank})
    assert (root is None) == (rank != 0)
    transport.barrier()
    transport.close()
    assert transport.staged_blocks == 0
    assert transport.arena.stats["staged"] == 0 and transport.arena._slabs == [None, None]  # noqa: SLF001
    results.put((rank, transport.stats.to_dict()))


def test_pinned_transport_numpy_payloads_across_processes() -> None:
    context = multiprocessing.get_context("spawn")
    world_size = 3
    endpoints = build_transport_endpoints(world_size, context=context)
    results = context.Queue()
    processes = [context.Process(target=_pinned_worker, args=(rank, endpoints, results)) for rank in range(world_size)]
    for process in processes:
        process.start()
    stats = {}
    for _ in range(world_size):
        rank, row = results.get(timeout=120)
        stats[rank] = row
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert sorted(stats) == list(range(world_size))
    assert all(row["collectives"] == 5 for row in stats.values())
    assert all(row["messages_sent"] >= 2 * (world_size - 1) for row in stats.values())
