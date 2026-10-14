"""CPU tests for the decode ciphertext all-reduce helpers (fake ciphers = NumPy arrays)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.parallel.decode import decode_collectives
from llama3fhe.parallel.decode.decode_collectives import (
    all_gather_ciphers,
    all_reduce_ciphers,
    broadcast_ciphers,
    group_all_reduce_ciphers,
    packet_bytes,
)
from llama3fhe.parallel.emulation import EmulationWorld, InProcessTransport


class _Packet:
    """Stand-in for CipherPacket: carries the array, reports its bytes."""

    def __init__(self, array, tag):
        self.array = np.array(array, copy=True)
        self.tag = tag

    @property
    def nbytes(self):
        return int(self.array.nbytes)


@pytest.fixture
def fake_wire(monkeypatch):
    monkeypatch.setattr(decode_collectives, "pack_cipher", lambda cipher, tag="", host=True: _Packet(cipher, tag))
    monkeypatch.setattr(decode_collectives, "unpack_cipher", lambda packet, device="cuda": np.array(packet.array))


def _context():
    return SimpleNamespace(
        device="cpu",
        context=None,
        fhe=SimpleNamespace(homo_add=lambda a, b, ctx: a + b),
    )


def _run_ranks(world_size, body):
    world = EmulationWorld(world_size, synchronize=None)
    results = [None] * world_size
    errors = []

    def target(rank):
        transport = InProcessTransport(rank, world)
        try:
            world.acquire(rank)
            results[rank] = body(rank, transport)
            world.finish(rank)
        except BaseException as exc:  # noqa: BLE001
            errors.append((rank, exc))
            world.abort(exc)
            try:
                world.finish(rank)
            except Exception:
                pass

    threads = [threading.Thread(target=target, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    return results


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_all_reduce_sums_every_rank_in_order(world_size, fake_wire) -> None:
    def body(rank, transport):
        comm = {}
        partials = [np.full(4, 10 ** rank, dtype=np.int64), np.full(4, rank + 1, dtype=np.int64)]
        outputs = all_reduce_ciphers(transport, "qkv", partials, crypto_context=_context(), comm=comm)
        return outputs, comm, transport.stats.to_dict()

    results = _run_ranks(world_size, body)
    expected0 = sum(10 ** r for r in range(world_size))
    expected1 = sum(r + 1 for r in range(world_size))
    for outputs, comm, stats in results:
        assert outputs[0].tolist() == [expected0] * 4
        assert outputs[1].tolist() == [expected1] * 4
        if world_size == 1:
            assert comm == {} and stats["collectives"] == 0
        else:
            assert comm["qkv_rounds"] == 1
            assert comm["qkv_sent"] == 2 * 32 * (world_size - 1)
            assert comm["qkv_received"] == 2 * 32 * (world_size - 1)
            assert stats["collectives"] == 1


def test_group_all_reduce_only_touches_the_group(fake_wire) -> None:
    world_size = 4

    def body(rank, transport):
        comm = {}
        group = (0, 1) if rank < 2 else (2, 3)
        partial = [np.full(2, float(rank), dtype=np.float64)]
        outputs = group_all_reduce_ciphers(transport, "gu", partial, group=group, crypto_context=_context(), comm=comm)
        return outputs, comm

    results = _run_ranks(world_size, body)
    assert results[0][0][0].tolist() == [1.0, 1.0]
    assert results[1][0][0].tolist() == [1.0, 1.0]
    assert results[2][0][0].tolist() == [5.0, 5.0]
    assert results[3][0][0].tolist() == [5.0, 5.0]
    assert all(comm["gu_sent"] == 16 and comm["gu_received"] == 16 for _, comm in results)


def test_ranks_outside_the_group_keep_their_input(fake_wire) -> None:
    def body(rank, transport):
        partial = [np.full(2, float(rank), dtype=np.float64)]
        outputs = group_all_reduce_ciphers(transport, "gu", partial, group=(0, 2), crypto_context=_context())
        return outputs

    results = _run_ranks(3, body)
    assert results[0][0].tolist() == [2.0, 2.0]
    assert results[2][0].tolist() == [2.0, 2.0]
    assert results[1][0].tolist() == [1.0, 1.0]  # untouched


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_broadcast_hands_root_ciphers_to_every_rank(world_size, fake_wire) -> None:
    def body(rank, transport):
        comm = {}
        own = [np.arange(3) + 100, np.arange(3) + 200] if rank == 0 else None
        outputs = broadcast_ciphers(transport, "input", own, root=0, crypto_context=_context(), comm=comm)
        return outputs, comm

    results = _run_ranks(world_size, body)
    for rank, (outputs, comm) in enumerate(results):
        assert [o.tolist() for o in outputs] == [[100, 101, 102], [200, 201, 202]]
        if world_size == 1:
            assert comm == {}
        elif rank == 0:
            assert comm["input_sent"] == 2 * 24 * (world_size - 1) and comm["input_received"] == 0
        else:
            assert comm["input_sent"] == 0 and comm["input_received"] == 2 * 24


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_all_gather_assembles_disjoint_subsets(world_size, fake_wire) -> None:
    from llama3fhe.parallel.decode.decode_ownership import DecodeRankOwnership

    def body(rank, transport):
        comm = {}
        mine = DecodeRankOwnership(rank, world_size).kv_refresh_indices(4)
        owned = {i: np.full(2, 10 * i + 1, dtype=np.int64) for i in mine}
        outputs = all_gather_ciphers(transport, "kv_refresh", owned, count=4, crypto_context=_context(), comm=comm)
        return outputs, comm, mine

    results = _run_ranks(world_size, body)
    covered = sorted(i for _, _, mine in results for i in mine)
    assert covered == [0, 1, 2, 3]
    for outputs, comm, mine in results:
        assert [o.tolist() for o in outputs] == [[1, 1], [11, 11], [21, 21], [31, 31]]
        if world_size == 1:
            assert comm == {}
        else:
            assert comm["kv_refresh_rounds"] == 1
            assert comm["kv_refresh_received"] == 16 * (4 - len(mine))


def test_all_gather_rejects_missing_or_duplicate_indices(fake_wire) -> None:
    def body(rank, transport):
        owned = {0: np.zeros(2)} if rank == 0 else {0: np.ones(2)}
        return all_gather_ciphers(transport, "x", owned, count=2, crypto_context=_context())

    with pytest.raises(AssertionError):
        _run_ranks(2, body)


def test_all_reduce_rejects_mismatched_counts(fake_wire) -> None:
    def body(rank, transport):
        partials = [np.zeros(2)] * (1 if rank == 0 else 2)
        return all_reduce_ciphers(transport, "bad", partials, crypto_context=_context())

    world = EmulationWorld(2, synchronize=None)
    errors = []

    def target(rank):
        transport = InProcessTransport(rank, world)
        try:
            world.acquire(rank)
            body(rank, transport)
            world.finish(rank)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            world.abort(exc)
            try:
                world.finish(rank)
            except Exception:
                pass

    threads = [threading.Thread(target=target, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert any(isinstance(e, RuntimeError) for e in errors)


def test_packet_bytes_handles_nesting() -> None:
    assert packet_bytes(None) == 0
    assert packet_bytes([_Packet(np.zeros(4), ""), (None, _Packet(np.zeros(2), ""))]) == 48



def test_emulation_rejects_rank_dependent_collective_tags():
    """The host/pinned transports match rounds by tag, so the emulation must too."""

    world = EmulationWorld(2, synchronize=None)
    errors = []

    def target(rank):
        transport = InProcessTransport(rank, world)
        try:
            world.acquire(rank)
            transport.all_gather(f"mlp/gu/{rank}", rank)
            world.finish(rank)
        except BaseException as exc:  # noqa: BLE001
            errors.append((rank, exc))
            world.abort(exc)

    threads = [threading.Thread(target=target, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors, "rank-dependent tags must be rejected by the emulation"
    assert any("collective tags must be identical" in str(exc) for _, exc in errors)
