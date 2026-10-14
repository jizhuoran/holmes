"""Emulate the T2 world on one card: eight ranks as lock-stepped threads.

The 8x4090 runs compare a parallel program on one machine with a serial
program on another, which makes the speed-up hard to read.  This module lets
one card (H100 today, A100 later) *play* the eight ranks of the T2 schedule so
that numerator and denominator come from the same silicon:

* one process, one CKKS context (one keygen; the rotation-key union of every
  rank stays resident), one bootstrap program;
* one thread per rank running the unchanged station code
  (:mod:`llama3fhe.parallel.prefill.stations`) against an :class:`InProcessTransport`
  that has the same interface as :class:`~.transport.HostStagingTransport`;
* a *baton*: at any time exactly one rank thread is on the GPU.  A rank keeps
  the baton until it reaches a collective (or finishes), synchronises the
  device and hands over to the next runnable rank.  Collectives complete once
  every rank has arrived; a rank blocked in a collective books the time
  between its arrival and its resumption as ``recv_wait_seconds``, exactly
  where the real transport books its queue wait.

Per rank and station, ``wall - recv_wait`` is therefore the GPU time that
rank's share costs *alone on the card*; the emulated parallel layer time is
the sum over stations of the slowest rank's share (communication modelled
separately, from the recorded bytes).  Summing the shares over all ranks gives
the "T2 code path run serially" number for the same card.

What the emulation cannot reproduce: real inter-card transfer time (bytes are
handed over by reference), allocator behaviour of eight independent processes
(one shared caching allocator here; ``peak_alloc_gib`` of a station includes
the ciphertexts the other seven ranks keep between stations), and the CPU
contention of eight encoders on one host (every rank thread sees the whole
host; pin ``OMP_NUM_THREADS`` for a like-for-like comparison).
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .prefill.transport import ReadyCollective, TransportStats, _payload_bytes

READY = "ready"
RUNNING = "running"
BLOCKED = "blocked"
DONE = "done"


def _cuda_synchronize() -> None:
    """Drain the device before the baton changes hands (no-op without CUDA)."""

    try:
        import easyfhe

        easyfhe.cuda.synchronize()
        return
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class _Round:
    """One collective round: payload per rank, complete once every rank arrived."""

    kind: str
    payloads: dict[int, Any] = field(default_factory=dict)
    complete: bool = False
    readers: int = 0
    tag: str | None = None


class EmulationWorld:
    """Baton + mailboxes shared by the rank threads of one emulated world.

    Everything is guarded by one condition variable.  ``synchronize`` runs
    (outside the lock) every time the baton is released, so the GPU work a
    rank enqueued is finished before the next rank starts its clock.

    Scheduling is deterministic: a rank keeps the baton until it blocks
    (collective / receive) or finishes; the baton then goes to the next
    *runnable* rank in cyclic order (the releasing rank itself last).  A
    blocked rank becomes runnable when its wake predicate holds, which is
    re-evaluated at every hand-over.  ``rank 0`` starts.
    """

    def __init__(self, world_size: int, *, synchronize: Callable[[], None] | None = _cuda_synchronize):
        self.world_size = int(world_size)
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        self._cv = threading.Condition()
        self._state = [READY] * self.world_size
        self._holder: int | None = None
        self._blocked_on: dict[int, Callable[[], bool]] = {}
        self._rounds: dict[str, dict[int, _Round]] = collections.defaultdict(dict)
        self._sequence = [collections.Counter() for _ in range(self.world_size)]
        self._inbox: list[dict[tuple[int, str], collections.deque]] = [
            collections.defaultdict(collections.deque) for _ in range(self.world_size)
        ]
        self._synchronize = synchronize
        self.failure: BaseException | None = None
        #: rank -> seconds spent parked (baton not held), all causes.
        self.wait_seconds = [0.0] * self.world_size
        self.handovers = 0
        #: hand-over order, for tests / debugging.
        self.schedule: list[int] = []

    # ----------------------------------------------------------------- baton
    def _check_failure(self) -> None:
        if self.failure is not None:
            raise RuntimeError(f"emulated world aborted: {self.failure!r}") from self.failure

    @property
    def holder(self) -> int | None:
        return self._holder

    def states(self) -> tuple[str, ...]:
        with self._cv:
            return tuple(self._state)

    def acquire(self, rank: int) -> None:
        """Block until ``rank`` holds the baton (rank 0 takes it first)."""

        rank = int(rank)
        started = time.perf_counter()
        with self._cv:
            if self._holder is None and rank == 0 and self._state[0] == READY:
                self._holder = 0
                self.schedule.append(0)
            while self._holder != rank:
                self._check_failure()
                self._cv.wait(timeout=1.0)
            self._state[rank] = RUNNING
        self.wait_seconds[rank] += time.perf_counter() - started

    def _refresh_blocked(self) -> None:
        for rank, ready in list(self._blocked_on.items()):
            if ready():
                self._state[rank] = READY
                del self._blocked_on[rank]

    def _next_after(self, rank: int) -> int | None:
        """Next READY rank after ``rank`` in cyclic order (``rank`` itself last)."""

        for step in range(1, self.world_size + 1):
            candidate = (rank + step) % self.world_size
            if self._state[candidate] == READY:
                return candidate
        return None

    def _hand_over(self, rank: int, new_state: str) -> None:
        """Caller holds the lock and the baton; pass it to the next runnable rank."""

        if self._holder != rank:
            raise RuntimeError(f"rank {rank} released a baton held by {self._holder}")
        self._state[rank] = new_state
        self._refresh_blocked()
        nxt = self._next_after(rank)
        self.handovers += 1
        if nxt is None:
            self._holder = None
            if any(state == BLOCKED for state in self._state):
                blocked = [r for r in range(self.world_size) if self._state[r] == BLOCKED]
                if self.failure is None:
                    self.failure = RuntimeError(
                        f"deadlock: no runnable rank; blocked={blocked}, states={self._state}"
                    )
            self._cv.notify_all()
            self._check_failure()
            return
        self._holder = nxt
        self.schedule.append(nxt)
        self._cv.notify_all()

    def _release_and_wait(self, rank: int, ready: Callable[[], bool]) -> None:
        """Give up the baton until ``ready()`` holds and the baton comes back."""

        if self._synchronize is not None:
            self._synchronize()
        started = time.perf_counter()
        with self._cv:
            self._blocked_on[rank] = ready
            self._hand_over(rank, BLOCKED)
            while self._holder != rank:
                self._check_failure()
                self._cv.wait(timeout=1.0)
            self._state[rank] = RUNNING
        self.wait_seconds[rank] += time.perf_counter() - started

    def yield_baton(self, rank: int) -> None:
        """Voluntarily pass the baton (rank stays runnable)."""

        self._release_and_wait(int(rank), lambda: True)

    def finish(self, rank: int) -> None:
        """The rank's thread is done; release the baton for good."""

        rank = int(rank)
        if self._synchronize is not None:
            self._synchronize()
        with self._cv:
            if self._holder == rank:
                self._hand_over(rank, DONE)
            else:
                self._state[rank] = DONE
                self._blocked_on.pop(rank, None)
                self._cv.notify_all()

    def abort(self, exc: BaseException) -> None:
        with self._cv:
            if self.failure is None:
                self.failure = exc
            self._cv.notify_all()

    # ------------------------------------------------------------- collectives
    def collective(
        self, rank: int, kind: str, payload: Any, *, symmetric: bool = True, tag: str | None = None
    ) -> dict[int, Any] | None:
        """Join round ``kind``#seq with ``payload``; return every rank's payload.

        ``symmetric=False`` (``gather_to`` non-root side): deposit and return
        ``None`` immediately, keeping the baton.  ``tag`` (when given) must be
        identical on every rank of the round: the host/pinned transports match
        rounds by tag, so a rank-dependent tag deadlocks there -- the emulation
        turns it into a ``RuntimeError`` instead of silently succeeding.
        """

        rank = int(rank)
        seq = self._sequence[rank][kind]
        self._sequence[rank][kind] += 1
        with self._cv:
            self._check_failure()
            rounds = self._rounds[kind]
            rnd = rounds.get(seq)
            if rnd is None:
                rnd = rounds[seq] = _Round(kind=kind, tag=tag)
            if rank in rnd.payloads:
                raise RuntimeError(f"rank {rank} joined {kind}#{seq} twice")
            if tag is not None and rnd.tag is not None and tag != rnd.tag:
                raise RuntimeError(
                    f"rank {rank} joined {kind}#{seq} with tag {tag!r} but another rank used {rnd.tag!r}; "
                    "collective tags must be identical on every rank"
                )
            rnd.payloads[rank] = payload
            if len(rnd.payloads) == self.world_size:
                rnd.complete = True
            if not symmetric:
                return None
        self._release_and_wait(rank, lambda: rnd.complete)
        with self._cv:
            result = dict(rnd.payloads)
            rnd.readers += 1
            if rnd.readers == self.world_size:
                rounds.pop(seq, None)
        return result

    def gather_root_wait(self, rank: int, kind: str, seq: int) -> dict[int, Any]:
        """Root side of ``gather_to``: wait until round ``seq`` is complete."""

        rank = int(rank)
        with self._cv:
            rnd = self._rounds[kind].setdefault(seq, _Round(kind=kind))
        self._release_and_wait(rank, lambda: rnd.complete)
        with self._cv:
            result = dict(rnd.payloads)
            self._rounds[kind].pop(seq, None)
        return result

    # --------------------------------------------------------------------- p2p
    def send(self, src: int, dst: int, tag: str, payload: Any) -> None:
        with self._cv:
            self._check_failure()
            self._inbox[int(dst)][(int(src), str(tag))].append(payload)

    def recv(self, rank: int, src: int, tag: str) -> Any:
        rank = int(rank)
        key = (int(src), str(tag))
        inbox = self._inbox[rank]
        with self._cv:
            if inbox[key]:
                return inbox[key].popleft()
        self._release_and_wait(rank, lambda: bool(inbox[key]))
        with self._cv:
            return inbox[key].popleft()


@dataclass
class InProcessTransport:
    """Same surface as :class:`~.transport.HostStagingTransport`, backed by :class:`EmulationWorld`.

    Payloads travel by reference (no pickle, no shared memory); the byte
    counters still record what a real transport would have moved so the
    communication can be modelled afterwards.  ``recv_wait_seconds`` is the
    time this rank spent parked at collectives / receives (baton not held).
    """

    rank: int
    world: EmulationWorld
    stats: TransportStats = field(default_factory=TransportStats)
    #: Payloads travel by reference, so stations keep copying components to
    #: the host (``pack_cipher(host=True)``): that copy is what makes the
    #: receiver's cipher independent of the sender's storage here.
    device_payloads: bool = False

    def __post_init__(self) -> None:
        self.rank = int(self.rank)
        if self.rank < 0 or self.rank >= self.world.world_size:
            raise IndexError(f"rank {self.rank} outside world of size {self.world.world_size}")
        self._gather_sequence = 0

    def __enter__(self) -> "InProcessTransport":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        return None

    @property
    def staged_blocks(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return int(self.world.world_size)

    @property
    def peers(self) -> tuple[int, ...]:
        return tuple(other for other in range(self.world_size) if other != self.rank)

    # -- p2p
    def send(self, dst: int, tag: str, payload: Any) -> None:
        dst = int(dst)
        if dst < 0 or dst >= self.world_size:
            raise IndexError(f"destination {dst} outside world of size {self.world_size}")
        if dst == self.rank:
            raise ValueError("use local hand-off instead of sending to self")
        started = time.perf_counter()
        self.world.send(self.rank, dst, tag, payload)
        self.stats.send_seconds += time.perf_counter() - started
        self.stats.messages_sent += 1
        self.stats.bytes_sent += _payload_bytes(payload)

    def recv(self, src: int, tag: str, *, timeout: float | None = None) -> Any:
        started = time.perf_counter()
        payload = self.world.recv(self.rank, src, tag)
        self.stats.recv_wait_seconds += time.perf_counter() - started
        self.stats.messages_received += 1
        self.stats.bytes_received += _payload_bytes(payload)
        return payload

    # -- collectives (same round semantics as the host transport: every rank,
    #    same order; the tag is informational, the sequence number pairs rounds)
    def _joined(self, kind: str, payload: Any, tag: str | None = None) -> dict[int, Any]:
        self.stats.collectives += 1
        started = time.perf_counter()
        result = self.world.collective(self.rank, kind, payload, tag=tag)
        self.stats.recv_wait_seconds += time.perf_counter() - started
        assert result is not None
        return result

    def all_gather(self, tag: str, payload: Any, *, timeout: float | None = None) -> list[Any]:
        result = self._joined("all_gather", payload, str(tag))
        nbytes = _payload_bytes(payload)
        self.stats.bytes_sent += nbytes * (self.world_size - 1)
        self.stats.messages_sent += self.world_size - 1
        gathered = [result[r] for r in range(self.world_size)]
        for src in self.peers:
            self.stats.bytes_received += _payload_bytes(gathered[src])
            self.stats.messages_received += 1
        return gathered

    def all_to_all(self, tag: str, payloads_by_dst: Sequence[Any], *, timeout: float | None = None) -> list[Any]:
        if len(payloads_by_dst) != self.world_size:
            raise ValueError("all_to_all needs exactly one payload per rank")
        payloads_by_dst = list(payloads_by_dst)
        for dst in self.peers:
            self.stats.bytes_sent += _payload_bytes(payloads_by_dst[dst])
            self.stats.messages_sent += 1
        result = self._joined("all_to_all", payloads_by_dst, str(tag))
        received = [result[src][self.rank] for src in range(self.world_size)]
        for src in self.peers:
            self.stats.bytes_received += _payload_bytes(received[src])
            self.stats.messages_received += 1
        return received

    # -- split-phase variants: the baton collective is synchronous, so the
    #    send half already completes the round; ``wait()`` just hands it over.
    def all_gather_isend(self, tag: str, payload: Any, *, timeout: float | None = None):
        return ReadyCollective(self.all_gather(tag, payload, timeout=timeout))

    def all_to_all_isend(self, tag: str, payloads_by_dst: Sequence[Any], *, timeout: float | None = None):
        return ReadyCollective(self.all_to_all(tag, payloads_by_dst, timeout=timeout))

    def barrier(self, tag: str = "barrier", *, timeout: float | None = None) -> None:
        self._joined("all_gather", None, str(tag))

    def gather_to(self, root: int, tag: str, payload: Any, *, timeout: float | None = None):
        root = int(root)
        self.stats.collectives += 1
        seq = self._gather_sequence
        self._gather_sequence += 1
        kind = f"gather_to:{root}"
        if self.rank != root:
            started = time.perf_counter()
            self.world.collective(self.rank, kind, payload, symmetric=False)
            self.stats.send_seconds += time.perf_counter() - started
            self.stats.messages_sent += 1
            self.stats.bytes_sent += _payload_bytes(payload)
            return None
        self.world.collective(self.rank, kind, payload, symmetric=False)
        started = time.perf_counter()
        result = self.world.gather_root_wait(self.rank, kind, seq)
        self.stats.recv_wait_seconds += time.perf_counter() - started
        gathered = [result[r] for r in range(self.world_size)]
        for src in self.peers:
            self.stats.bytes_received += _payload_bytes(gathered[src])
            self.stats.messages_received += 1
        return gathered


# ------------------------------------------------------------------ key union
def all_rank_key_union(preset, feature_layout, attention_layout, *, tiers: Iterable[str], schedule=None,
                       limb_caps: Mapping[str, int] | None = None) -> dict[str, tuple[int, ...]]:
    """Rotations every rank's key plan asks for, per tier (``extra_rotations`` for one shared context).

    Only the ``attn_input_*`` phase tiers depend on the rank (Q pairs / K pair /
    V sources follow the ownership); the union is computed for every requested
    tier anyway so a shared-context build can never miss a key.
    """

    from .prefill.keys import build_key_plan
    from .prefill.ownership import ownership_from_layouts

    world = ownership_from_layouts(feature_layout, attention_layout, world_size=int(preset.world_size))
    schedule = preset.layer_schedule() if schedule is None else schedule
    caps = dict(preset.limb_caps) if limb_caps is None else dict(limb_caps)
    union: dict[str, set[int]] = collections.defaultdict(set)
    for rank in range(int(preset.world_size)):
        plan = build_key_plan(
            preset, feature_layout, attention_layout, tiers=tuple(tiers), schedule=schedule, limb_caps=caps,
            resident_extra=tuple(preset.key_resident_extra), policy=str(preset.key_policy), ownership=world[rank],
        )
        for name in plan.names:
            union[name].update(int(v) for v in plan.tier(name).rotations)
    return {name: tuple(sorted(values)) for name, values in union.items()}


# --------------------------------------------------------------------- runner
@dataclass
class EmulatedRank:
    """What a worker receives instead of building its own context: shared state + its transport."""

    transport: InProcessTransport
    shared_state: Any
    world: EmulationWorld

    @property
    def rank(self) -> int:
        return int(self.transport.rank)


@dataclass
class EmulatedOutcome:
    rank: int
    ok: bool
    result: Any = None
    error: str | None = None
    wall_seconds: float = 0.0
    baton_wait_seconds: float = 0.0


def run_emulated_world(preset, entry: Callable[..., Any], *, shared_state: Any, kwargs: Mapping[str, Any] | None = None,
                       synchronize: Callable[[], None] | None = _cuda_synchronize,
                       join_timeout: float | None = 3600.0) -> list[EmulatedOutcome]:
    """Run ``entry(rank, preset, None, emulation=EmulatedRank, **kwargs)`` for every rank, lock-stepped.

    Threads start in rank order; rank 0 takes the baton first.  The first
    exception aborts every other rank (they raise at their next baton wait).
    """

    import traceback

    world_size = int(preset.world_size)
    world = EmulationWorld(world_size, synchronize=synchronize)
    kwargs = {} if kwargs is None else dict(kwargs)
    outcomes: list[EmulatedOutcome | None] = [None] * world_size

    def target(rank: int) -> None:
        started = time.perf_counter()
        transport = InProcessTransport(rank, world)
        handle = EmulatedRank(transport=transport, shared_state=shared_state, world=world)
        try:
            world.acquire(rank)
            result = entry(rank, preset, None, emulation=handle, **kwargs)
            world.finish(rank)
            outcomes[rank] = EmulatedOutcome(rank=rank, ok=True, result=result,
                                             wall_seconds=time.perf_counter() - started,
                                             baton_wait_seconds=world.wait_seconds[rank])
        except BaseException as exc:  # noqa: BLE001 - report every failure
            world.abort(exc)
            outcomes[rank] = EmulatedOutcome(
                rank=rank, ok=False, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                wall_seconds=time.perf_counter() - started, baton_wait_seconds=world.wait_seconds[rank])
            try:
                world.finish(rank)
            except Exception:
                pass

    threads = [threading.Thread(target=target, args=(rank,), name=f"emu-rank{rank}", daemon=True)
               for rank in range(world_size)]
    for thread in threads:
        thread.start()
    deadline = None if join_timeout is None else time.perf_counter() + float(join_timeout)
    for thread in threads:
        remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
        thread.join(timeout=remaining)
        if thread.is_alive():
            world.abort(TimeoutError(f"{thread.name} did not finish within {join_timeout}s"))
            raise TimeoutError(f"emulated ranks did not all finish within {join_timeout}s")
    for outcome in outcomes:
        if outcome is not None and not outcome.ok:
            raise RuntimeError(f"rank {outcome.rank} failed:\n{outcome.error}")
    return [outcome for outcome in outcomes if outcome is not None]


__all__ = [
    "EmulatedOutcome",
    "EmulatedRank",
    "EmulationWorld",
    "InProcessTransport",
    "all_rank_key_union",
    "run_emulated_world",
]
