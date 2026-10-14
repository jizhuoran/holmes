"""Split-phase collectives (``*_isend`` / ``wait``) and the overlapped attention-input order."""

from __future__ import annotations

import multiprocessing
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.parallel.prefill import stations
from llama3fhe.parallel.prefill.transport import (
    CollectiveHandle,
    HostStagingTransport,
    ReadyCollective,
    build_transport_endpoints,
)


# ------------------------------------------------------------------ transport
def _split_worker(rank, endpoints, results):
    transport = HostStagingTransport(rank, endpoints, default_timeout=60.0)
    world = transport.world_size
    # round 1: lanes all-gather issued, "GPU work" with rank skew, then waited for
    handle = transport.all_gather_isend("lanes", np.full(4, rank, dtype=np.int64))
    assert isinstance(handle, CollectiveHandle) and not handle.done
    time.sleep(0.02 * rank)
    gathered = handle.wait()
    assert handle.done
    assert [int(item[0]) for item in gathered] == list(range(world))
    assert handle.wait() is gathered  # idempotent
    # round 2: partials all-to-all issued, more skewed work, waited for
    to_send = [np.array([rank * 100 + dst]) for dst in range(world)]
    handle = transport.all_to_all_isend("partials", to_send)
    time.sleep(0.02 * (world - 1 - rank))
    received = handle.wait()
    assert [int(item[0]) for item in received] == [src * 100 + rank for src in range(world)]
    # round 3/4: the blocking calls still work and interleave with split rounds
    big = transport.all_gather("big", np.full(4096, rank, dtype=np.uint64))
    assert all(int(item[0]) == src for src, item in enumerate(big))
    transport.barrier()
    transport.close()
    assert transport.staged_blocks == 0
    results.put((rank, transport.stats.to_dict()))


@pytest.mark.parametrize("world_size", [2, 4])
def test_split_phase_collectives_across_spawned_processes(world_size) -> None:
    context = multiprocessing.get_context("spawn")
    endpoints = build_transport_endpoints(world_size, context=context)
    results = context.Queue()
    processes = [context.Process(target=_split_worker, args=(rank, endpoints, results)) for rank in range(world_size)]
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
    for row in stats.values():
        assert row["collectives"] == 4
        assert row["messages_sent"] == 4 * (world_size - 1)  # 4 symmetric rounds incl. the barrier
        assert row["messages_received"] == 4 * (world_size - 1)


def test_wait_rejects_a_second_round_in_flight() -> None:
    transport = HostStagingTransport(0, build_transport_endpoints(1), default_timeout=0.05)
    first = transport.all_gather_isend("a", np.array([1]))
    second = transport.all_gather_isend("b", np.array([2]))  # starts round 2 while round 1 is in flight
    with pytest.raises(RuntimeError, match="in flight"):
        first.wait()
    assert second.wait()[0].tolist() == [2]
    # the blocking collectives are the split pair back to back
    assert transport.all_to_all("c", [np.array([3])])[0].tolist() == [3]
    transport.close()


def test_inprocess_transport_split_phase_is_ready() -> None:
    from llama3fhe.parallel.emulation import EmulationWorld, InProcessTransport

    world_size = 2
    world = EmulationWorld(world_size)
    results = [None] * world_size
    errors = []

    def body(rank):
        transport = InProcessTransport(rank, world)
        try:
            world.acquire(rank)
            handle = transport.all_gather_isend("x", rank + 7)
            assert isinstance(handle, ReadyCollective) and handle.done
            gathered = handle.wait()
            handle = transport.all_to_all_isend("y", [(rank, dst) for dst in range(world_size)])
            results[rank] = (gathered, handle.wait())
            world.finish(rank)
        except BaseException as exc:  # noqa: BLE001
            errors.append((rank, exc))
            world.abort(exc)
            try:
                world.finish(rank)
            except Exception:
                pass

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    for rank in range(world_size):
        assert results[rank] == ([7, 8], [(src, rank) for src in range(world_size)])


# ------------------------------------------------------ attention_input_exchange order
class _Cipher:
    def __init__(self, name: str) -> None:
        self.name = name
        self.state = SimpleNamespace(cur_limbs=13, scale_degree=1)
        self.released = False

    def release(self) -> None:
        self.released = True


class _Packet:
    def __init__(self, cipher, tag: str) -> None:
        self.cipher = cipher
        self.tag = tag
        self.nbytes = 8


class _ThreadWorld:
    """Rank threads + a barrier per collective round; records every rank's call order."""

    def __init__(self, world_size: int, *, split: bool) -> None:
        self.world_size = int(world_size)
        self.split = bool(split)
        self.barrier = threading.Barrier(self.world_size)
        self.rounds: dict[int, dict[int, object]] = {}
        self.lock = threading.Lock()
        self.log: dict[int, list[str]] = {r: [] for r in range(self.world_size)}
        self.transports = [self._Transport(self, r) for r in range(self.world_size)]

    def deposit(self, rank: int, sequence: int, payload) -> None:
        with self.lock:
            self.rounds.setdefault(sequence, {})[rank] = payload

    class _Transport:
        device_payloads = False

        def __init__(self, world, rank: int) -> None:
            self.world = world
            self.rank = int(rank)
            self.world_size = world.world_size
            self.sequence = 0

        def _begin(self, payload):
            self.sequence += 1
            self.world.deposit(self.rank, self.sequence, payload)
            return self.sequence

        def _collect(self, sequence: int, kind: str):
            self.world.barrier.wait(timeout=30)
            table = self.world.rounds[sequence]
            if kind == "all_gather":
                return [table[src] for src in range(self.world_size)]
            return [table[src][self.rank] for src in range(self.world_size)]

        def all_gather(self, tag, payload):
            self.world.log[self.rank].append("ag")
            return self._collect(self._begin(payload), "all_gather")

        def all_to_all(self, tag, payloads_by_dst):
            self.world.log[self.rank].append("a2a")
            return self._collect(self._begin(list(payloads_by_dst)), "all_to_all")

        def _handle(self, kind, payload):
            sequence = self._begin(payload)
            world, rank = self.world, self.rank

            class _Handle:
                done = False

                def wait(_self):
                    world.log[rank].append(f"{kind}_wait")
                    out = self._collect(sequence, kind)
                    _self.done = True
                    return out

            return _Handle()

        def all_gather_isend(self, tag, payload):
            self.world.log[self.rank].append("all_gather_isend")
            return self._handle("all_gather", payload)

        def all_to_all_isend(self, tag, payloads_by_dst):
            self.world.log[self.rank].append("all_to_all_isend")
            return self._handle("all_to_all", list(payloads_by_dst))

    def transport(self, rank: int):
        transport = self.transports[rank]
        if not self.split:
            # a duck-typed transport without the split API
            return SimpleNamespace(
                device_payloads=False, rank=rank, world_size=self.world_size,
                all_gather=transport.all_gather, all_to_all=transport.all_to_all,
            )
        return transport


def _fake_attention_input(monkeypatch: pytest.MonkeyPatch, log: dict[int, list[str]]):
    import llama3fhe.operators.attention.distributed.distributed_input as dist
    from llama3fhe.parallel import wire

    current_rank = threading.local()

    def rank_of(prefix: str) -> int:
        return int(prefix.split("rank")[1].split(".")[0])

    def result(**fields):
        return SimpleNamespace(release=lambda: None, **fields)

    def fake_lane(own, *, source_indices, function_prefix, **_):
        log[rank_of(function_prefix)].append("lane")
        return result(lane_ciphers=tuple(_Cipher(f"lane{s}") for s in source_indices))

    def fake_target(lanes, *, query_pair_indices, key_pair_indices, value_carrier_indices, function_prefix,
                    value_source_indices=(), value_return_partial=False, emit_cache_c=False, **_):
        rank = rank_of(function_prefix)
        if value_return_partial:
            # own lanes only (the foreign ones have not been gathered yet)
            assert [i for i, lane in enumerate(lanes) if lane is not None] == list(value_source_indices)
            log[rank].append("vpart" + ("+c" if emit_cache_c else ""))
            partials = tuple((c, _Cipher(f"vpart{c}@{rank}")) for c in value_carrier_indices)
            c_partials = tuple((c, _Cipher(f"vcpart{c}@{rank}")) for c in value_carrier_indices) if emit_cache_c else ()
            return result(value_partials=partials, query_items=(), key_items=(), scheduler_stats={},
                          key_c_items=(), value_c_partials=c_partials)
        assert all(lane is not None for lane in lanes), "Q/K need every lane"
        if query_pair_indices:
            log[rank].append("query")
            return result(value_partials=(), key_items=(), scheduler_stats={}, key_c_items=(), value_c_partials=(),
                          query_items=tuple((p, _Cipher(f"q{p}")) for p in query_pair_indices))
        log[rank].append("key" + ("+c" if emit_cache_c else ""))
        return result(value_partials=(), query_items=(), scheduler_stats={}, value_c_partials=(),
                      key_items=tuple((p, _Cipher(f"k{p}")) for p in key_pair_indices),
                      key_c_items=tuple((p, _Cipher(f"kc{p}")) for p in key_pair_indices) if emit_cache_c else ())

    def fake_finish(summands, *, attention_layout, crypto_context):
        rank = int(crypto_context.rank)
        carriers = sorted({int(c) for c, _ in summands})
        assert len(carriers) == 1
        sources = sorted(cipher.name for _, cipher in summands)
        cache = all(name.startswith("vcpart") for name in sources)
        log[rank].append("finish_c" if cache else "finish")
        out = _Cipher(f"{'vc' if cache else 'v'}{carriers[0]}")
        out.sources = tuple(sources)
        return ((2 * carriers[0], out), (2 * carriers[0] + 1, out))

    monkeypatch.setattr(dist, "attention_input_lane_shard_fhe", fake_lane)
    monkeypatch.setattr(dist, "attention_input_target_shard_fhe", fake_target)
    monkeypatch.setattr(dist, "attention_input_value_finish_fhe", fake_finish)
    monkeypatch.setattr(wire, "pack_cipher", lambda cipher, *, tag="", host=True: _Packet(cipher, tag))
    monkeypatch.setattr(wire, "unpack_cipher", lambda packet, *, device=None: packet.cipher)
    monkeypatch.setattr(stations, "_sync", lambda _context: None)
    return current_rank


def _run_attention_input(monkeypatch: pytest.MonkeyPatch, *, world_size: int, split: bool, overlap: bool,
                         emit_kv_cache: bool = False):
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts

    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    world = ownership_from_layouts(feature, attention, world_size=world_size)
    tw = _ThreadWorld(world_size, split=split)
    _fake_attention_input(monkeypatch, tw.log)
    carrier_layout = SimpleNamespace(cipher_count=int(feature.cipher_count))
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def run(rank_ownership):
        rank = int(rank_ownership.rank)
        try:
            results[rank] = stations.attention_input_exchange(
                tuple(_Cipher(f"fm{c}") for c in rank_ownership.cipher_indices),
                ownership=rank_ownership, world=world, transport=tw.transport(rank),
                carrier_layout=carrier_layout, attention_layout=attention,
                context=SimpleNamespace(device="cpu", rank=rank), cos=None, sin=None,
                function_prefix=f"t2.rank{rank}.attn_in", overlap=overlap, emit_kv_cache=emit_kv_cache,
            )
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            errors[rank] = exc
            tw.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in world]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    return world, tw, results


BASELINE_ORDER = ["lane", "vpart", "ag", "a2a", "finish", "query", "key"]
OVERLAP_ORDER = ["lane", "all_gather_isend", "vpart", "all_gather_wait", "all_to_all_isend",
                 "query", "key", "all_to_all_wait", "finish"]


def _check_results(world, results, world_size: int, *, packets_per_partial: int = 1) -> None:
    from llama3fhe.parallel.prefill.ownership import key_pairs_for_rank, value_carriers_for_rank

    for rank in world:
        out = results[rank.rank]
        assert [p for p, _ in out.query_items] == list(rank.query_pair_indices)
        assert [p for p, _ in out.key_items] == list(key_pairs_for_rank(rank, gqa_ratio=4))
        carriers = value_carriers_for_rank(rank, gqa_ratio=4)
        assert len(carriers) == 1
        assert [p for p, _ in out.value_items] == [2 * carriers[0], 2 * carriers[0] + 1]
        # the finished V carrier summed exactly one partial from every rank
        assert out.value_items[0][1].sources == tuple(sorted(f"vpart{carriers[0]}@{r}" for r in range(world_size)))
        assert out.comm_bytes["lane_received"] == 8 * len(rank.cipher_indices) * (world_size - 1)
        assert out.comm_bytes["vpart_received"] == packets_per_partial * 8 * (world_size - 1)


@pytest.mark.parametrize("world_size", [8, 4])
def test_attention_input_baseline_order_unchanged(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    world, tw, results = _run_attention_input(monkeypatch, world_size=world_size, split=True, overlap=False)
    for rank in range(world_size):
        assert tw.log[rank] == BASELINE_ORDER
    _check_results(world, results, world_size)
    assert set(results[0].stage_seconds) == {"lane", "value_partial", "exchange", "unpack", "value_finish", "query", "key"}


@pytest.mark.parametrize("world_size", [8, 4])
def test_attention_input_overlap_order(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    world, tw, results = _run_attention_input(monkeypatch, world_size=world_size, split=True, overlap=True)
    for rank in range(world_size):
        assert tw.log[rank] == OVERLAP_ORDER
    _check_results(world, results, world_size)
    assert set(results[0].stage_seconds) == {
        "lane", "lane_isend", "value_partial", "exchange", "unpack", "query", "key",
        "exchange_vpart", "unpack_vpart", "value_finish",
    }


@pytest.mark.parametrize("world_size", [8, 4])
@pytest.mark.parametrize("overlap", [False, True])
def test_attention_input_emit_kv_cache_rides_the_same_rounds(monkeypatch: pytest.MonkeyPatch, world_size: int,
                                                              overlap: bool) -> None:
    """``emit_kv_cache=True``: the V_C partials travel in the Delta all-to-all (no extra round), the K_C
    of the own key pair comes out of the K shard, and the Delta results are untouched."""

    from llama3fhe.parallel.prefill.ownership import key_pairs_for_rank, value_carriers_for_rank

    world, tw, results = _run_attention_input(monkeypatch, world_size=world_size, split=True, overlap=overlap,
                                              emit_kv_cache=True)
    expected = [s.replace("vpart", "vpart+c").replace("key", "key+c") for s in (OVERLAP_ORDER if overlap else BASELINE_ORDER)]
    # the cache V finish follows the Delta finish inside the same step
    expected.insert(expected.index("finish") + 1, "finish_c")
    for rank in range(world_size):
        assert tw.log[rank] == expected
    _check_results(world, results, world_size, packets_per_partial=2)
    for rank in world:
        out = results[rank.rank]
        assert [p for p, _ in out.key_c_items] == list(key_pairs_for_rank(rank, gqa_ratio=4))
        assert all(c.name == f"kc{p}" for p, c in out.key_c_items)
        carrier = value_carriers_for_rank(rank, gqa_ratio=4)[0]
        assert [p for p, _ in out.value_c_items] == [2 * carrier, 2 * carrier + 1]
        assert out.value_c_items[0][1].sources == tuple(sorted(f"vcpart{carrier}@{r}" for r in range(world_size)))
        # twice the Delta partial bytes on the wire (one packet per partial, 8 bytes each in the fake)
        assert out.comm_bytes["vpart_received"] == 2 * 8 * (world_size - 1)
        assert out.comm_bytes["vpart_sent"] == 2 * 8 * (world_size - 1)
        key_c, value_c = out.take_kv_cache_items()
        assert out.key_c_items == () and out.value_c_items == ()
        assert len(key_c) == 1 and len(value_c) == 2
        out.release()
        assert not key_c[0][1].released  # detached items are the caller's


def test_attention_input_overlap_falls_back_on_blocking_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    world, tw, results = _run_attention_input(monkeypatch, world_size=8, split=False, overlap=True)
    # no split API: the collectives complete inside the "isend"; order of the rest is the overlapped one
    for rank in range(8):
        assert tw.log[rank] == ["lane", "ag", "vpart", "a2a", "query", "key", "finish"]
    _check_results(world, results, 8)


def test_overlap_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("T2_ATTN_IN_OVERLAP", raising=False)
    assert stations.attention_input_overlap_enabled() is False
    assert stations.attention_input_overlap_enabled(default=True) is True
    for value, expected in (("1", True), ("on", True), ("0", False), ("off", False), ("", False)):
        monkeypatch.setenv("T2_ATTN_IN_OVERLAP", value)
        assert stations.attention_input_overlap_enabled() is expected
