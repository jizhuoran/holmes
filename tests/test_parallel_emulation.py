"""CPU tests for the single-card emulation of the T2 world (llama3fhe.parallel.emulation)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from llama3fhe.parallel.emulation import (
    EmulationWorld,
    InProcessTransport,
    all_rank_key_union,
    run_emulated_world,
)
from llama3fhe.parallel.prefill.keys import KeyTracer, build_key_plan
from llama3fhe.parallel.presets import H100_EMU, PRESETS, ParallelPreset, preset_by_name


def _run_ranks(world_size, body, *, synchronize=None):
    """Run ``body(rank, transport, world)`` on one thread per rank, lock-stepped; return the results."""

    world = EmulationWorld(world_size, synchronize=synchronize)
    results = [None] * world_size
    errors = []

    def target(rank):
        transport = InProcessTransport(rank, world)
        try:
            world.acquire(rank)
            results[rank] = body(rank, transport, world)
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
    assert not any(t.is_alive() for t in threads), "emulated ranks hung"
    return results, errors, world


@pytest.mark.parametrize("world_size", [1, 3, 8])
def test_collectives_match_the_host_transport_contract(world_size) -> None:
    def body(rank, transport, world):
        gathered = transport.all_gather("lanes", np.full((4,), rank, dtype=np.float64))
        exchanged = transport.all_to_all("partials", [(rank, dst) for dst in range(world_size)])
        transport.barrier("mid")
        gathered2 = transport.all_gather("second", rank * 10)
        root = transport.gather_to(0, "summary", {"rank": rank})
        transport.barrier("final")
        return gathered, exchanged, gathered2, root, transport.stats.to_dict()

    results, errors, world = _run_ranks(world_size, body)
    assert not errors, errors
    for rank, (gathered, exchanged, gathered2, root, stats) in enumerate(results):
        assert [float(g[0]) for g in gathered] == [float(r) for r in range(world_size)]
        # all_to_all: what rank ``src`` addressed to us
        assert exchanged == [(src, rank) for src in range(world_size)]
        assert gathered2 == [r * 10 for r in range(world_size)]
        if rank == 0:
            assert root == [{"rank": r} for r in range(world_size)]
        else:
            assert root is None
        assert set(stats) == {"messages_sent", "messages_received", "bytes_sent", "bytes_received",
                              "send_seconds", "recv_wait_seconds", "collectives"}
        # all_gather of 4 float64 = 32 bytes to each peer
        assert stats["bytes_sent"] >= 32 * (world_size - 1)
        assert stats["collectives"] == 6
    assert all(state == "done" for state in world.states())


def test_rounds_pair_by_sequence_but_reject_mismatched_tags() -> None:
    """The host/pinned transports key their inbox by ``tag#seq``, so two ranks
    using different tags for the same round deadlock there.  The emulation
    pairs rounds by sequence and raises on a tag mismatch so the bug surfaces
    in the single-device emulation instead of as a 900 s timeout on 8 GPUs."""

    def body(rank, transport, world):
        return transport.all_gather(f"tag-{rank}", rank)

    results, errors, _ = _run_ranks(2, body)
    assert errors
    assert any("collective tags must be identical" in str(exc) for _, exc in errors)


def test_identical_tags_pair_by_sequence() -> None:
    def body(rank, transport, world):
        return transport.all_gather("same-tag", rank)

    results, errors, _ = _run_ranks(2, body)
    assert not errors
    assert results == [[0, 1], [0, 1]]


def test_exactly_one_rank_runs_at_a_time_and_waits_are_booked() -> None:
    running = []
    lock = threading.Lock()
    violations = []

    def body(rank, transport, world):
        for step in range(3):
            with lock:
                running.append(rank)
                if len(running) != 1:
                    violations.append(list(running))
            time.sleep(0.01)
            with lock:
                running.remove(rank)
            transport.barrier(f"step-{step}")
        return transport.stats.recv_wait_seconds

    results, errors, world = _run_ranks(4, body)
    assert not errors
    assert not violations, violations
    assert results[0] >= 3 * 3 * 0.009
    assert world.handovers >= 4 * 3
    # deterministic round robin: after rank 0 blocks, ranks 1, 2, 3 run in order
    assert world.schedule[:4] == [0, 1, 2, 3]


def test_synchronize_hook_runs_at_every_hand_over() -> None:
    calls = []

    def body(rank, transport, world):
        transport.barrier("a")
        transport.barrier("b")

    _, errors, world = _run_ranks(2, body, synchronize=lambda: calls.append(1))
    assert not errors
    # each rank: 2 barriers + finish = 3 releases
    assert len(calls) == 2 * 3


def test_mismatched_collectives_are_reported_as_a_deadlock() -> None:
    def body(rank, transport, world):
        if rank == 0:
            transport.barrier("only-rank-0")
        # rank 1 finishes without joining -> rank 0 can never complete its round
        return rank

    results, errors, world = _run_ranks(2, body)
    assert errors and all("deadlock" in str(exc) or "aborted" in str(exc) for _, exc in errors), errors
    assert world.failure is not None


def test_failure_in_one_rank_aborts_the_others() -> None:
    def body(rank, transport, world):
        if rank == 1:
            raise ValueError("boom")
        transport.barrier("after-boom")

    results, errors, world = _run_ranks(3, body)
    kinds = {rank: type(exc).__name__ for rank, exc in errors}
    assert kinds[1] == "ValueError"
    assert all(k in kinds for k in (0, 2))


def test_point_to_point_send_recv_blocks_until_delivery() -> None:
    def body(rank, transport, world):
        if rank == 0:
            got = transport.recv(1, "hello")
            return got
        transport.send(0, "hello", np.arange(3))
        return None

    results, errors, _ = _run_ranks(2, body)
    assert not errors
    assert list(results[0]) == [0, 1, 2]


def test_run_emulated_world_passes_the_emulation_handle_and_shared_state() -> None:
    preset = ParallelPreset(name="tiny", world_size=3, key_policy="resident")
    shared = object()

    def entry(rank, preset_, endpoints, *, emulation, extra):
        assert endpoints is None
        assert emulation.shared_state is shared
        assert emulation.rank == rank
        gathered = emulation.transport.all_gather("x", rank * extra)
        emulation.transport.gather_to(0, "s", rank)
        emulation.transport.barrier("final")
        return gathered if rank == 0 else None

    outcomes = run_emulated_world(preset, entry, shared_state=shared, kwargs={"extra": 2}, synchronize=None)
    assert [o.rank for o in outcomes] == [0, 1, 2]
    assert outcomes[0].result == [0, 2, 4]
    assert all(o.ok for o in outcomes)


def test_run_emulated_world_reraises_the_first_failure() -> None:
    preset = ParallelPreset(name="tiny", world_size=2, key_policy="resident")

    def entry(rank, preset_, endpoints, *, emulation):
        if rank == 1:
            raise RuntimeError("rank 1 exploded")
        emulation.transport.barrier("never")

    with pytest.raises(RuntimeError, match="failed"):
        run_emulated_world(preset, entry, shared_state=None, synchronize=None)


def test_all_rank_key_union_covers_every_ranks_plan() -> None:
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts
    from llama3fhe.parallel.worker import production_layouts

    preset = H100_EMU
    feature_layout, attention_layout = production_layouts()
    tiers = ("bootstrap", "rmsnorm", "linear", "attn_input_lane", "attn_input_q", "attn_input_k", "attn_input_v",
             "attn_qk", "attn_pv", "softmax", "pairfm_output")
    union = all_rank_key_union(preset, feature_layout, attention_layout, tiers=tiers)
    assert set(union) == set(tiers)
    world = ownership_from_layouts(feature_layout, attention_layout, world_size=preset.world_size)
    per_rank = [
        build_key_plan(preset, feature_layout, attention_layout, tiers=tiers, schedule=preset.layer_schedule(),
                       limb_caps=preset.limb_caps, resident_extra=preset.key_resident_extra,
                       policy=preset.key_policy, ownership=world[rank])
        for rank in range(preset.world_size)
    ]
    for plan in per_rank:
        for name in tiers:
            assert set(plan.tier(name).rotations) <= set(union[name])
    # the rank-dependent tiers really differ between ranks (otherwise the union would be moot)
    assert len({tuple(plan.tier("attn_input_q").rotations) for plan in per_rank}) > 1
    # a shared-context plan built with the union as extras holds every rotation of every rank
    shared = build_key_plan(preset, feature_layout, attention_layout, tiers=tiers, schedule=preset.layer_schedule(),
                            limb_caps=preset.limb_caps, resident_extra=preset.key_resident_extra,
                            policy=preset.key_policy, ownership=world[0], extra_rotations=union)
    for plan in per_rank:
        assert set(plan.union) <= set(shared.union)


def test_h100_emu_preset_is_efficiency_first_on_the_l3_bootstrap() -> None:
    preset = preset_by_name("h100-emu")
    assert preset is H100_EMU and "h100-emu" in PRESETS
    assert preset.key_policy == "resident"
    assert preset.bootstrap.level_budget == (5, 5) and preset.bootstrap.cache_mode == "plain"
    assert preset.bootstrap_post_levels == 17
    assert preset.linear_max_plaintext_rows == 512  # the 2048-row chunk is the 96 GB card's alone
    assert preset.attention_input_overlap is True
    assert preset.mlp_feature_shard_k == 4
    assert preset.transform_hoist_strategy == "ext_normal"
    assert preset.device_budget_gib >= 70.0


def test_key_tracer_stage_is_per_thread() -> None:
    tracer = KeyTracer(native=object())
    seen = {}
    gate = threading.Barrier(2)

    def worker(name):
        with tracer.stage(name):
            gate.wait(timeout=5)
            seen[name] = tracer.current
            gate.wait(timeout=5)

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert seen == {"a": "a", "b": "b"}
    assert tracer.current == KeyTracer.OUTSIDE
