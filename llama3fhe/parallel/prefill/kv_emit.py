"""Decode KV-cache emission of the multi-card Prefill (one layer).

The monolithic graph builds the PairFold128 Decode cache of a layer from the
raw ``K_C`` / ``V_C`` snapshots the attention-input adapter emits
(:func:`llama3fhe.graph.common._pairfold_prefill_cache_from_raw_c`).  On the
T2 station chain those snapshots are scattered over the ranks
(:func:`~.stations.attention_input_exchange` with ``emit_kv_cache=True``):
rank ``r`` holds the folded ``K_C`` of its key pair (``r // 2`` on W=8, so two
ranks hold the same pair) and the real ``V_C`` of the two pairs of its value
carrier (``r // 4``; four ranks hold the same two pairs).  Decode wants the
*whole* cache of the layer on every rank (the decode split is over heads /
diagonals of the same four PairFold ciphertexts), hence this station:

1. ``all_gather`` the eight raw C ciphertexts (4 ``K_C`` + 4 ``V_C``), each
   contributed by the lowest rank that holds it (duplicates are dropped);
2. every rank converts them to the four PairFold ciphertexts (cheap: masks,
   one half-payload rotation, one conjugate per group; the same converter the
   graph uses, ``emitted_key_c=True``);
3. the four outputs sit two levels under the attention-input level, i.e. below
   ``kv_limbs`` (12) -> each needs one dense bootstrap before the modulus drop.
   Rank ``i < 4`` refreshes + drops output ``i`` (one BS per rank instead of
   four per rank), the result is gathered back (4 x 12-limb ciphers);
4. ``PairFoldTokenWrite(column=0)`` -> ``PairFoldCipherPage`` ->
   :class:`~llama3fhe.operators.pairfold_kv_cache.PairFoldLayerCache`
   (``content_tokens=128``), identical on every rank.

Rotation keys: ``build_key_plan(..., emit_kv_cache=True)`` adds the converter's
``(slots // 2, M - 1)`` to ``attn_input_k`` / ``attn_input_v``; the bootstrap
is the worker's resident program.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .ownership import RankOwnership, WorldOwnership, key_pairs_for_rank, value_group_ranks

__all__ = [
    "PrefillKVCacheResult",
    "PrefillKVCachePending",
    "PrefillKVCacheStreamPending",
    "kv_cache_contributions",
    "kv_cache_raw_output_index",
    "kv_cache_raw_owner",
    "kv_cache_refresh_indices",
    "prefill_kv_cache_begin",
    "prefill_kv_cache_stream_begin",
    "prefill_kv_cache_station",
]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def _owner_directed_raw_route(
    owned: dict[int, Any],
    *,
    pair_count: int,
    pair_blocks: int,
    group_count: int,
    rank: int,
    world_size: int,
    transport,
    context,
    tag: str,
    comm: dict[str, Any],
) -> dict[int, Any]:
    """Send each raw K/V pair directly to its PairFold output owner.

    The legacy path all-gathers every raw pair to every rank.  A group
    converter only needs the two raw pairs belonging to one output group, so
    this route sends raw index ``p`` to the rank owning that output.  Every
    rank receives only the inputs for the groups it will convert.
    """

    from ..wire import pack_cipher, unpack_cipher
    from ..decode.decode_collectives import packet_bytes

    payloads_by_dst: list[Any] = [None] * int(world_size)
    packets_by_dst: list[list[tuple[int, Any]]] = [[] for _ in range(int(world_size))]
    host = not getattr(transport, "device_payloads", False)
    try:
        for raw_index, cipher in sorted(owned.items()):
            raw_index = int(raw_index)
            if raw_index < int(pair_count):
                output_index = raw_index // int(pair_blocks)
            else:
                output_index = int(group_count) + (raw_index - int(pair_count)) // int(pair_blocks)
            destination = int(output_index) % int(world_size)
            packets_by_dst[destination].append(
                (raw_index, pack_cipher(cipher, tag=f"{tag}/{rank}/{raw_index}", host=host))
            )
    except BaseException:
        from llama3fhe.backend import release_if_supported

        for cipher in owned.values():
            release_if_supported(cipher)
        raise
    payloads_by_dst = [tuple(items) if items else None for items in packets_by_dst]
    started = time.perf_counter()
    try:
        received = transport.all_to_all(f"{tag}/owner", payloads_by_dst)
    except BaseException:
        from llama3fhe.backend import release_if_supported

        for cipher in owned.values():
            release_if_supported(cipher)
        raise
    elapsed = time.perf_counter() - started
    sent = sum(packet_bytes(payload) for destination, payload in enumerate(payloads_by_dst) if destination != int(rank))
    received_bytes = sum(packet_bytes(payload) for source, payload in enumerate(received) if source != int(rank))
    comm[f"{tag}/owner_sent"] = int(comm.get(f"{tag}/owner_sent", 0)) + int(sent)
    comm[f"{tag}/owner_received"] = int(comm.get(f"{tag}/owner_received", 0)) + int(received_bytes)
    comm[f"{tag}/owner_seconds"] = float(comm.get(f"{tag}/owner_seconds", 0.0)) + float(elapsed)
    comm[f"{tag}/owner_rounds"] = int(comm.get(f"{tag}/owner_rounds", 0)) + 1

    result: dict[int, Any] = {}
    success = False
    try:
        for source, source_payload in enumerate(received):
            for raw_index, packet in tuple(source_payload or ()):
                raw_index = int(raw_index)
                expected_owner = kv_cache_raw_owner(
                    raw_index,
                    world_size=world_size,
                    pair_count=pair_count,
                    pair_blocks=pair_blocks,
                    group_count=group_count,
                )
                if expected_owner != int(rank):
                    raise RuntimeError(
                        f"rank {rank} received raw pair {raw_index} routed to {expected_owner}"
                    )
                if raw_index in result:
                    raise RuntimeError(f"raw ciphertext {raw_index} received more than once")
                # Keep a self-routed cipher in device memory.  Besides
                # avoiding a needless pack/unpack round trip this preserves
                # ownership for lightweight mock packets used by tests where
                # ``unpack_cipher`` returns the original object.
                if source == int(rank):
                    try:
                        result[raw_index] = owned.pop(raw_index)
                    except KeyError as exc:
                        raise RuntimeError(
                            f"rank {rank} self-routed unknown raw pair {raw_index}"
                        ) from exc
                else:
                    result[raw_index] = unpack_cipher(packet, device=context.device)
        expected = {
            index
            for index in range(2 * int(pair_count))
            if kv_cache_raw_owner(
                index,
                world_size=world_size,
                pair_count=pair_count,
                pair_blocks=pair_blocks,
                group_count=group_count,
            ) == int(rank)
        }
        if set(result) != expected:
            raise RuntimeError(
                f"rank {rank} raw route received {sorted(result)}, expected {sorted(expected)}"
            )
        success = True
    finally:
        # ``pack_cipher`` copies the residues; the sending rank no longer owns
        # the original raw ciphertexts after the collective.
        for cipher in owned.values():
            from llama3fhe.backend import release_if_supported

            release_if_supported(cipher)
        if not success:
            from llama3fhe.backend import release_if_supported

            for cipher in result.values():
                release_if_supported(cipher)
    return result


def kv_cache_contributions(
    world: WorldOwnership, ownership: RankOwnership, *, gqa_ratio: int = 4
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(key_pairs, value_pairs)`` this rank contributes to the raw all-gather.

    A key pair is contributed by the lowest rank that reads it; the two value
    pairs ``2c, 2c+1`` of carrier ``c`` by the lowest rank of the carrier's
    group.  Every pair ends up with exactly one contributor.
    """

    rank = int(ownership.rank)
    key_pairs: list[int] = []
    for pair in key_pairs_for_rank(ownership, gqa_ratio=gqa_ratio):
        readers = [item.rank for item in world if pair in key_pairs_for_rank(item, gqa_ratio=gqa_ratio)]
        if min(readers) == rank:
            key_pairs.append(int(pair))
    value_pairs: list[int] = []
    carriers = sorted({int(pair) // 2 for pair in key_pairs_for_rank(ownership, gqa_ratio=gqa_ratio)})
    for carrier in carriers:
        if min(value_group_ranks(world, carrier, gqa_ratio=gqa_ratio)) == rank:
            value_pairs.extend((2 * carrier, 2 * carrier + 1))
    return tuple(key_pairs), tuple(value_pairs)


def kv_cache_refresh_indices(rank: int, world_size: int, *, count: int = 4) -> tuple[int, ...]:
    """PairFold outputs (``K0 K1 V0 V1`` -> 0..3) rank ``rank`` refreshes and drops."""

    return tuple(index for index in range(int(count)) if index % int(world_size) == int(rank))


def kv_cache_raw_output_index(
    raw_index: int,
    *,
    pair_count: int = 4,
    pair_blocks: int = 2,
    group_count: int = 2,
) -> int:
    """Map one raw K/V pair index to its PairFold output index.

    Raw indices ``0..pair_count-1`` are K pairs and
    ``pair_count..2*pair_count-1`` are V pairs.  Each PairFold output group
    consumes ``pair_blocks`` adjacent raw pairs; output order is ``K groups``
    followed by ``V groups`` (``K0,K1,V0,V1`` in production).
    """

    raw_index = int(raw_index)
    pair_count = int(pair_count)
    pair_blocks = int(pair_blocks)
    group_count = int(group_count)
    if pair_count <= 0 or pair_blocks <= 0 or group_count <= 0:
        raise ValueError("pair_count, pair_blocks and group_count must be positive")
    if pair_count != pair_blocks * group_count:
        raise ValueError(
            "pair_count must equal pair_blocks * group_count: "
            f"{pair_count} != {pair_blocks} * {group_count}"
        )
    if not 0 <= raw_index < 2 * pair_count:
        raise IndexError(f"raw_index={raw_index} outside [0, {2 * pair_count})")
    if raw_index < pair_count:
        return raw_index // pair_blocks
    return group_count + (raw_index - pair_count) // pair_blocks


def kv_cache_raw_owner(
    raw_index: int,
    *,
    world_size: int,
    pair_count: int = 4,
    pair_blocks: int = 2,
    group_count: int = 2,
) -> int:
    """Return the rank that owns conversion/refresh for one raw pair."""

    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    output = kv_cache_raw_output_index(
        raw_index,
        pair_count=pair_count,
        pair_blocks=pair_blocks,
        group_count=group_count,
    )
    return output % world_size


@dataclass
class PrefillKVCacheResult:
    """One owned :class:`PairFoldLayerCache` plus the station's bookkeeping."""

    cache: Any
    stage_seconds: dict[str, float]
    comm_bytes: dict[str, int]
    comm_seconds: float
    contributed: dict[str, tuple[int, ...]]
    refreshed: tuple[int, ...]
    bootstrap_calls: int
    limbs: dict[str, list[int]] = field(default_factory=dict)

    @property
    def wall_seconds(self) -> float:
        return float(sum(self.stage_seconds.values()))

    def take_cache(self):
        """Detach the cache; the caller owns it from here on."""

        cache, self.cache = self.cache, None
        return cache

    def release(self) -> None:
        cache, self.cache = self.cache, None
        if cache is not None:
            cache.release()


@dataclass
class PrefillKVCachePending:
    """Split-phase raw KV route for safe overlap with Attention Core."""

    handle: Any
    mode: str
    owned: dict[int, Any]
    pair_count: int
    pair_blocks: int
    group_count: int
    key_pairs: tuple[int, ...]
    value_pairs: tuple[int, ...]
    rank: int
    world_size: int
    context: Any
    comm: dict[str, Any]
    tag: str
    _waited: bool = False

    def wait_raw(self) -> tuple[list[Any], dict[int, Any], float]:
        if self._waited:
            raise RuntimeError("PrefillKVCachePending.wait_raw() called twice")
        self._waited = True
        from llama3fhe.backend import release_if_supported
        from ..wire import unpack_cipher
        from ..decode.decode_collectives import packet_bytes

        started = time.perf_counter()
        try:
            received = self.handle.wait()
        except BaseException:
            for cipher in self.owned.values():
                release_if_supported(cipher)
            self.owned = {}
            raise
        self.comm[f"{self.tag}/route_received"] = int(
            sum(packet_bytes(payload) for source, payload in enumerate(received) if source != int(self.rank))
        )
        raw: list[Any] = []
        raw_by_index: dict[int, Any] = {}
        try:
            if self.mode == "owner":
                for source, source_payload in enumerate(received):
                    for raw_index, packet in tuple(source_payload or ()):
                        raw_index = int(raw_index)
                        expected_owner = kv_cache_raw_owner(
                            raw_index,
                            world_size=self.world_size,
                            pair_count=self.pair_count,
                            pair_blocks=self.pair_blocks,
                            group_count=self.group_count,
                        )
                        if expected_owner != int(self.rank):
                            raise RuntimeError(
                                f"rank {self.rank} received raw pair {raw_index} routed to {expected_owner}"
                            )
                        if raw_index in raw_by_index:
                            raise RuntimeError(f"raw ciphertext {raw_index} received more than once")
                        if source == int(self.rank):
                            try:
                                raw_by_index[raw_index] = self.owned.pop(raw_index)
                            except KeyError as exc:
                                raise RuntimeError(
                                    f"rank {self.rank} self-routed unknown raw pair {raw_index}"
                                ) from exc
                        else:
                            raw_by_index[raw_index] = unpack_cipher(packet, device=self.context.device)
                expected = {
                    index
                    for index in range(2 * int(self.pair_count))
                    if kv_cache_raw_owner(
                        index,
                        world_size=self.world_size,
                        pair_count=self.pair_count,
                        pair_blocks=self.pair_blocks,
                        group_count=self.group_count,
                    ) == int(self.rank)
                }
                if set(raw_by_index) != expected:
                    raise RuntimeError(
                        f"rank {self.rank} raw route received {sorted(raw_by_index)}, expected {sorted(expected)}"
                    )
            else:
                slots: list[Any] = [None] * (2 * int(self.pair_count))
                for source, source_payload in enumerate(received):
                    for raw_index, packet in tuple(source_payload or ()):
                        raw_index = int(raw_index)
                        if raw_index < 0 or raw_index >= len(slots) or slots[raw_index] is not None:
                            raise RuntimeError(f"invalid or duplicate raw ciphertext index {raw_index}")
                        if source == int(self.rank):
                            try:
                                slots[raw_index] = self.owned.pop(raw_index)
                            except KeyError as exc:
                                raise RuntimeError(
                                    f"rank {self.rank} self-routed unknown raw pair {raw_index}"
                                ) from exc
                        else:
                            slots[raw_index] = unpack_cipher(packet, device=self.context.device)
                missing = [index for index, value in enumerate(slots) if value is None]
                if missing:
                    raise RuntimeError(f"raw KV route missing ciphertexts {missing}")
                raw = list(slots)
        except BaseException:
            for cipher in list(raw_by_index.values()) + [value for value in raw if value is not None]:
                release_if_supported(cipher)
            raise
        finally:
            for cipher in self.owned.values():
                release_if_supported(cipher)
            self.owned = {}
        elapsed = time.perf_counter() - started
        self.comm[f"{self.tag}/route_wait_seconds"] = float(
            self.comm.get(f"{self.tag}/route_wait_seconds", 0.0)
        ) + float(elapsed)
        return raw, raw_by_index, elapsed


@dataclass
class PrefillKVCacheStreamPending:
    """Background KV conversion/refresh launched after a raw route.

    The raw transport collective is waited in one worker thread, then
    conversion/refresh runs under ``stream``.  The worker publishes a
    conversion event and waits before the final PairFold gather so PairFM can
    use the transport round first; :meth:`finish` joins the final event to the
    caller's stream.  Callers must not consume the returned cache before
    finishing.  This object is only created for a resident CUDA key plan.  The
    normal synchronous station remains the fallback for CPU, leased keys, and
    stream construction failures.
    """

    raw_pending: PrefillKVCachePending
    station_kwargs: dict[str, Any]
    stream: Any
    ready: Any
    thread: threading.Thread
    raw_ready: threading.Event = field(default_factory=threading.Event)
    conversion_ready: Any = None
    gather_gate: threading.Event = field(default_factory=threading.Event)
    box: dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"
    _finished: bool = False
    _cancelled: bool = False

    def wait_raw_route(self) -> None:
        """Wait until the raw transport round has been consumed by the worker.

        PairFM uses the same transport and cannot start its barrier/all-to-all
        while the raw KV route is still in flight.  This host event does not
        wait for conversion or refresh, so those GPU stages remain overlapped
        with the subsequent PairFM work.
        """

        # Older/fake station callables may not know about the optional signal;
        # if their worker has already exited there is no collective left to
        # protect.  A live worker still must publish the signal before PairFM.
        if not self.raw_ready.is_set() and self.thread.is_alive():
            self.raw_ready.wait()
        error = self.box.get("error")
        if error is not None:
            # The worker sets this after waking raw_ready on a failed route.
            # Join before propagating so the side thread cannot outlive the
            # layer and touch buffers that the caller is about to reclaim.
            self.thread.join()
            raise error

    def allow_final_gather(self) -> None:
        """Release the final KV gather after the competing PairFM round.

        The transport permits only one symmetric collective in flight.  The
        side worker therefore stops after conversion/refresh and waits here;
        the PairFM all-to-all can run first, after which its local target
        reduction overlaps this KV all-gather.
        """

        if self._cancelled:
            return
        self.gather_gate.set()

    def _wait_event_on_current_stream(self, event=None) -> None:
        """Make default-stream consumers wait for side-stream CUDA work."""

        event = self.ready if event is None else event
        if event is None:
            return

        try:
            import easyfhe

            current = easyfhe.cuda.current_stream(device=self.device)
            wait_event = getattr(current, "wait_event", None)
            if callable(wait_event):
                wait_event(event)
            else:  # pragma: no cover - defensive fallback for old EasyFHE
                synchronize = getattr(event, "synchronize", None)
                if callable(synchronize):
                    synchronize()
        except Exception:
            # ``finish`` is called only on CUDA in production.  CPU fakes and
            # old wrappers may not expose stream wait; host synchronization is
            # still correct and keeps the split-phase API testable.
            synchronize = getattr(event, "synchronize", None)
            if callable(synchronize):
                synchronize()

    def wait_conversion(self) -> None:
        """Join conversion/refresh to the caller stream before PairFM.

        The worker deliberately blocks before its final transport collective.
        This dependency lets PairFM use the transport first while the worker
        waits on :meth:`allow_final_gather`.
        """

        if self._cancelled:
            raise RuntimeError("PrefillKVCacheStreamPending was cancelled")
        # The transport allows only one symmetric collective in flight.  Wait
        # for the side worker to consume the raw route before the caller can
        # enter PairFM's all-to-all; conversion itself remains asynchronous
        # and is ordered with the caller's CUDA stream by the event below.
        self.wait_raw_route()
        self._wait_event_on_current_stream(self.conversion_ready)
        # An exception path records the conversion event before exiting.  It
        # is safe to inspect the box now; a successful worker remains blocked
        # on ``gather_gate`` and must not be joined here.
        error = self.box.get("error")
        if error is not None:
            self.thread.join()
            raise error

    def finish(self) -> "PrefillKVCacheResult":
        """Wait for the worker, enqueue the event dependency, and return cache."""

        if self._cancelled:
            raise RuntimeError("PrefillKVCacheStreamPending was cancelled")
        if self._finished and self.box.get("error") is not None:
            # Preserve the original worker exception on repeated calls.  The
            # first call releases any partially constructed result below, so
            # returning a generic "missing result" would hide the cause.
            raise self.box["error"]
        if not self._finished:
            # Direct callers that do not have a PairFM phase still get the
            # complete station.  The normal pipeline calls this after
            # ``allow_final_gather`` from the PairFM exchange callback.
            self.allow_final_gather()
            # Joining first ensures the result object and the final PairFold
            # collective are fully constructed before the event is published.
            self.thread.join()
            error = self.box.get("error")
            if error is not None:
                # ``prefill_kv_cache_station`` normally owns its own cleanup,
                # but an event-record or stream-wrapper failure can happen
                # after it has returned a result.  The result is never handed
                # to a consumer on this path, so release it before surfacing
                # the worker error.  This also covers a partial result placed
                # in ``box`` by a test/fake worker.
                result = self.box.get("result")
                if result is not None:
                    release = getattr(result, "release", None)
                    if callable(release):
                        release()
                    self.box["result"] = None
                self._finished = True
                raise error
            self._wait_event_on_current_stream()
            self._finished = True
        result = self.box.get("result")
        if result is None:
            raise RuntimeError("KV stream worker finished without a result")
        return result

    def cancel(self) -> None:
        """Drain the worker and release a result when the owning layer aborts."""

        if self._cancelled:
            return
        self.raw_ready.set()
        # Do not leave a worker blocked on the gate while draining an error
        # path.  It is safe to complete the final collective during cleanup.
        self.gather_gate.set()
        self.thread.join()
        result = self.box.get("result")
        if result is not None and not self._finished:
            # No default-stream consumer will follow on an error path.  Host
            # synchronization prevents freeing buffers while side kernels are
            # still in flight; a fake event may simply have no synchronize().
            synchronize = getattr(self.ready, "synchronize", None)
            if callable(synchronize):
                try:
                    synchronize()
                except Exception:
                    pass
            release = getattr(result, "release", None)
            if callable(release):
                release()
            self.box["result"] = None
        self._cancelled = True


def _stream_key_plan_is_resident(context, keys) -> bool:
    """Return whether side-stream KV work can safely share rotation keys."""

    if not str(getattr(context, "device", "")).startswith("cuda"):
        return False
    plan = getattr(keys, "plan", None)
    return str(getattr(plan, "policy", "")) == "resident"


def prefill_kv_cache_stream_begin(
    raw_pending: PrefillKVCachePending,
    *,
    keys=None,
    stream=None,
    **station_kwargs,
) -> PrefillKVCacheStreamPending | None:
    """Launch conversion/refresh/gather on a resident-only CUDA side stream.

    ``raw_pending`` must come from :func:`prefill_kv_cache_begin`.  ``None``
    is returned when the context/key plan is not eligible, allowing the
    caller to pass the same pending object to the synchronous station.  The
    worker invokes :func:`prefill_kv_cache_station` with
    ``synchronize=False`` so the bootstrap cannot issue a device-wide sync.
    """

    context = station_kwargs.get("context")
    if not _stream_key_plan_is_resident(context, keys):
        return None
    try:
        import easyfhe

        side = stream
        if side is None:
            side = easyfhe.cuda.Stream(device=str(context.device))
        ready = easyfhe.cuda.Event()
        conversion_ready = easyfhe.cuda.Event()
    except Exception:
        return None

    kwargs = dict(station_kwargs)
    # The raw pending object already owns the detached K/V snapshots.  Pass
    # empty positional inputs explicitly because the station keeps those
    # arguments required for its synchronous call sites.
    kwargs.setdefault("key_c_items", ())
    kwargs.setdefault("value_c_items", ())
    kwargs["raw_pending"] = raw_pending
    kwargs["synchronize"] = False
    gather_gate = threading.Event()
    raw_ready = threading.Event()
    kwargs["raw_ready_event"] = raw_ready
    kwargs["gather_ready_event"] = conversion_ready
    kwargs["gather_gate"] = gather_gate
    box: dict[str, Any] = {}

    def run() -> None:
        started = time.perf_counter()
        try:
            with easyfhe.cuda.stream(side):
                result = prefill_kv_cache_station(**kwargs)
                box["result"] = result
                ready.record(side)
                box["result_wall_s"] = float(time.perf_counter() - started)
        except BaseException as exc:  # surfaced by finish(); never lost in a daemon
            box["error"] = exc
            raw_ready.set()
            box["result_wall_s"] = float(time.perf_counter() - started)
            # If the station failed after issuing kernels, let the side stream
            # drain before the owner releases any remaining intermediates.
            try:
                side.synchronize()
            except Exception:
                pass
            # Make a failed conversion event safe for a caller that already
            # queued a stream wait before observing ``box['error']``.
            try:
                conversion_ready.record(side)
            except Exception:
                pass
            # Wake a caller that is waiting on conversion_ready so it can
            # surface the worker exception instead of waiting on an unrecorded
            # CUDA event forever.
            try:
                with easyfhe.cuda.stream(side):
                    conversion_ready.record(side)
                    ready.record(side)
            except Exception:
                pass

    thread = threading.Thread(target=run, name="prefill-kv-stream", daemon=True)
    thread.start()
    return PrefillKVCacheStreamPending(
        raw_pending=raw_pending,
        station_kwargs=kwargs,
        stream=side,
        ready=ready,
        raw_ready=raw_ready,
        conversion_ready=conversion_ready,
        gather_gate=gather_gate,
        thread=thread,
        box=box,
        device=str(context.device),
    )


def _collective_isend(transport, kind: str, tag: str, payload):
    from .transport import ReadyCollective

    isend = getattr(transport, f"{kind}_isend", None)
    if isend is not None:
        return isend(tag, payload)
    return ReadyCollective(getattr(transport, kind)(tag, payload))


def _bootstrap_refresh(bootstrap_operator, cipher, *, iterations: int, synchronize: bool):
    """Call refresh with the split-phase flag while retaining fake operators."""

    if bool(synchronize):
        return bootstrap_operator.refresh(cipher, iterations=int(iterations))
    try:
        return bootstrap_operator.refresh(cipher, iterations=int(iterations), synchronize=False)
    except TypeError as exc:
        # CPU station fakes in the package tests predate the optional keyword.
        # Do not hide unrelated TypeErrors raised by a real refresh method.
        if "synchronize" not in str(exc):
            raise
        return bootstrap_operator.refresh(cipher, iterations=int(iterations))


def prefill_kv_cache_begin(
    key_c_items,
    value_c_items,
    *,
    ownership: RankOwnership,
    world: WorldOwnership,
    transport,
    attention_layout,
    context,
    pairfold_layout=None,
    tag: str = "kv_emit",
) -> PrefillKVCachePending:
    """Start raw KV routing without waiting for its collective."""

    from ..decode.decode_collectives import packet_bytes
    from ..wire import pack_cipher

    pair_count = int(attention_layout.key_value_pair_count)
    pair_blocks = int(getattr(pairfold_layout, "pair_blocks", 2))
    group_count = int(getattr(pairfold_layout, "key_value_groups", 2))
    rank = int(ownership.rank)
    world_size = int(world.world_size)
    key_c = {int(pair): cipher for pair, cipher in key_c_items}
    value_c = {int(pair): cipher for pair, cipher in value_c_items}
    key_pairs, value_pairs = kv_cache_contributions(world, ownership, gqa_ratio=int(attention_layout.gqa_ratio))
    owned: dict[int, Any] = {}
    for pair in key_pairs:
        owned[int(pair)] = key_c.pop(int(pair))
    for pair in value_pairs:
        owned[pair_count + int(pair)] = value_c.pop(int(pair))
    from llama3fhe.backend import release_if_supported

    for cipher in list(key_c.values()) + list(value_c.values()):
        release_if_supported(cipher)
    host = not getattr(transport, "device_payloads", False)
    mode = "owner" if _env_flag("LLAMA_FHE_PREFILL_OWNER_ROUTING") else "gather"
    route_started = time.perf_counter()
    if mode == "owner":
        packets_by_dst: list[list[tuple[int, Any]]] = [[] for _ in range(world_size)]
        try:
            for raw_index, cipher in sorted(owned.items()):
                # PairFold128 has two groups of two raw pairs.  Spell the layout
                # out here so this route stays aligned with the grouped converter
                # and does not silently depend on helper defaults.
                destination = kv_cache_raw_owner(
                    raw_index,
                    world_size=world_size,
                    pair_count=pair_count,
                    pair_blocks=pair_blocks,
                    group_count=group_count,
                )
                packets_by_dst[destination].append(
                    (raw_index, pack_cipher(cipher, tag=f"{tag}/begin/{rank}/{raw_index}", host=host))
                )
        except BaseException:
            from llama3fhe.backend import release_if_supported

            for cipher in owned.values():
                release_if_supported(cipher)
            raise
        payloads = [tuple(items) if items else None for items in packets_by_dst]
        handle = _collective_isend(transport, "all_to_all", f"{tag}/owner", payloads)
        sent = sum(packet_bytes(payload) for dst, payload in enumerate(payloads) if dst != rank)
    else:
        try:
            packets = tuple(
                (raw_index, pack_cipher(cipher, tag=f"{tag}/begin/{rank}/{raw_index}", host=host))
                for raw_index, cipher in sorted(owned.items())
            )
        except BaseException:
            from llama3fhe.backend import release_if_supported

            for cipher in owned.values():
                release_if_supported(cipher)
            raise
        handle = _collective_isend(transport, "all_gather", f"{tag}/raw", packets)
        sent = packet_bytes(packets) * (world_size - 1)
    comm = {
        f"{tag}/route_sent": int(sent),
        f"{tag}/route_rounds": 1,
        f"{tag}/route_begin_seconds": float(time.perf_counter() - route_started),
    }
    return PrefillKVCachePending(
        handle=handle, mode=mode, owned=owned, pair_count=pair_count,
        pair_blocks=pair_blocks, group_count=group_count,
        key_pairs=tuple(key_pairs), value_pairs=tuple(value_pairs),
        rank=rank, world_size=world_size, context=context, comm=comm, tag=tag,
    )


def prefill_kv_cache_station(
    key_c_items,
    value_c_items,
    *,
    ownership: RankOwnership,
    world: WorldOwnership,
    transport,
    attention_layout,
    context,
    bootstrap_operator,
    kv_limbs: int | None,
    bootstrap_iterations: int = 1,
    occupied: int | None = None,
    pairfold_layout=None,
    tag: str = "kv_emit",
    raw_pending: PrefillKVCachePending | None = None,
    synchronize: bool = True,
    raw_ready_event: threading.Event | None = None,
    gather_ready_event: Any | None = None,
    gather_gate: threading.Event | None = None,
) -> PrefillKVCacheResult:
    """Raw per-rank ``K_C`` / ``V_C`` -> the layer's PairFold cache on every rank.

    ``key_c_items`` / ``value_c_items`` are the ``(pair, cipher)`` tuples of
    :class:`~.stations.AttentionInputExchangeResult` (``take_kv_cache_items``);
    they are consumed.  ``kv_limbs=None`` skips the refresh / drop (the cache
    keeps the converter's level).  Every rank of the world must call this for
    the same layer.  With ``LLAMA_FHE_PREFILL_OWNER_ROUTING=1`` the raw route
    is an ``all_to_all`` and only the final PairFold cache remains an
    all-gather; the default uses the historical raw all-gather.
    ``synchronize=False`` is reserved for
    :func:`prefill_kv_cache_stream_begin`; bootstrap launches then remain on
    the side stream and the caller owns the final event dependency.
    """

    from llama3fhe.backend import release_if_supported
    from llama3fhe.layouts.pairfold128 import PairFold128Layout
    from llama3fhe.operators.attention.pairfold.pairfold_prefill import (
        prefill_c_to_pairfold_fhe,
        prefill_emitted_key_c_group_to_pairfold_fhe,
        prefill_emitted_key_c_direct_group_to_pairfold_fhe,
        prefill_emitted_key_c_direct_to_pairfold_fhe,
        prefill_real_value_c_group_to_pairfold_fhe,
        prefill_real_value_c_to_pairfold_fhe,
    )
    from llama3fhe.operators.pairfold_kv_cache import (
        PAIRFOLD_GROUP_COUNT,
        PairFoldCipherPage,
        PairFoldLayerCache,
        PairFoldTokenWrite,
    )
    from llama3fhe.operators.primitives import drop_to_limbs
    from ..decode.decode_collectives import all_gather_ciphers

    pairfold_layout = PairFold128Layout() if pairfold_layout is None else pairfold_layout
    pair_count = int(attention_layout.key_value_pair_count)
    ratio = int(attention_layout.gqa_ratio)
    rank = int(ownership.rank)
    world_size = int(world.world_size)
    occupied = int(pairfold_layout.seq_len) if occupied is None else int(occupied)
    if occupied != int(pairfold_layout.seq_len):
        raise ValueError(
            "the PairFold Prefill converter routes the full page; "
            f"occupied={occupied}, expected {pairfold_layout.seq_len}."
        )
    key_c = {int(pair): cipher for pair, cipher in key_c_items}
    value_c = {int(pair): cipher for pair, cipher in value_c_items}
    stage: dict[str, float] = {}
    comm: dict[str, Any] = {}
    limbs: dict[str, list[int]] = {}
    owned: dict[int, Any] = {}
    raw: list[Any] = []
    outputs: list[Any] = []
    refreshed_out: dict[int, Any] = {}
    final: list[Any] = []
    output_map: dict[int, Any] = {}
    raw_by_index: dict[int, Any] = {}
    # The grouped/direct converters consume their input ciphers.  Keep
    # intermediate outputs here until ownership is transferred to ``outputs``
    # so an exception in the second converter cannot leak the first result.
    pairfold_keys: tuple[Any, ...] = ()
    pairfold_values: tuple[Any, ...] = ()
    owner_routing = _env_flag("LLAMA_FHE_PREFILL_OWNER_ROUTING")
    bootstrap_calls = 0
    token = None
    page = None
    try:
        if raw_pending is not None:
            if raw_pending.context is not context:
                for cipher in raw_pending.owned.values():
                    release_if_supported(cipher)
                raw_pending.owned = {}
                raise ValueError("raw_pending was created with a different crypto context")
            try:
                raw, raw_by_index, route_wait = raw_pending.wait_raw()
            except BaseException:
                if raw_ready_event is not None:
                    raw_ready_event.set()
                raise
            if raw_ready_event is not None:
                raw_ready_event.set()
            comm.update(raw_pending.comm)
            key_pairs = tuple(raw_pending.key_pairs) if hasattr(raw_pending, "key_pairs") else ()
            value_pairs = tuple(raw_pending.value_pairs) if hasattr(raw_pending, "value_pairs") else ()
            owner_routing = raw_pending.mode == "owner"
            stage["gather_raw"] = float(route_wait)
            limbs["raw_in"] = [int(cipher.state.cur_limbs) for cipher in (list(raw_by_index.values()) or raw)]
        else:
            # 1. one contributor per raw cipher; the duplicates this rank holds are dropped
            key_pairs, value_pairs = kv_cache_contributions(world, ownership, gqa_ratio=ratio)
            for pair in key_pairs:
                owned[pair] = key_c.pop(pair)
            for pair in value_pairs:
                owned[pair_count + pair] = value_c.pop(pair)
            for cipher in list(key_c.values()) + list(value_c.values()):
                release_if_supported(cipher)
            key_c.clear()
            value_c.clear()
            limbs["raw_in"] = [int(c.state.cur_limbs) for c in owned.values()]
            if owner_routing:
                raw_by_index = _owner_directed_raw_route(
                    owned,
                    pair_count=pair_count,
                    pair_blocks=int(pairfold_layout.pair_blocks),
                    group_count=int(pairfold_layout.key_value_groups),
                    rank=rank,
                    world_size=world_size,
                    transport=transport,
                    context=context,
                    tag=tag,
                    comm=comm,
                )
                owned = {}
                stage["gather_raw"] = float(comm.get(f"{tag}/owner_seconds", 0.0))
            else:
                started = time.perf_counter()
                raw = list(all_gather_ciphers(transport, f"{tag}/raw", owned, count=2 * pair_count, crypto_context=context, comm=comm))
                owned = {}
                stage["gather_raw"] = time.perf_counter() - started

        # 2. the graph's converter (consumes the eight inputs)
        started = time.perf_counter()
        direct_key = _env_flag("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K")
        if owner_routing:
            started = time.perf_counter()
            pairs = int(pairfold_layout.pair_blocks)
            group_count = int(pairfold_layout.key_value_groups)
            for output_index in range(2 * group_count):
                if int(output_index) % world_size != rank:
                    continue
                group = output_index if output_index < group_count else output_index - group_count
                first = group * pairs
                if output_index < group_count:
                    group_indices = tuple(first + pair for pair in range(pairs))
                    # Transfer these references before entering the converter:
                    # the converter owns and releases the inputs on both
                    # success and failure.  Leaving them in ``raw_by_index``
                    # would double-release an input if a later group fails.
                    group_inputs = tuple(raw_by_index[index] for index in group_indices)
                    for index in group_indices:
                        raw_by_index.pop(index)
                    if direct_key:
                        output_map[output_index] = prefill_emitted_key_c_direct_group_to_pairfold_fhe(
                            group_inputs,
                            pairfold_layout=pairfold_layout,
                            crypto_context=context,
                        )
                    else:
                        output_map[output_index] = prefill_emitted_key_c_group_to_pairfold_fhe(
                            group_inputs,
                            attention_layout=attention_layout,
                            pairfold_layout=pairfold_layout,
                            group=group,
                            crypto_context=context,
                        )
                else:
                    first_value = pair_count + first
                    group_indices = tuple(first_value + pair for pair in range(pairs))
                    group_inputs = tuple(raw_by_index[index] for index in group_indices)
                    for index in group_indices:
                        raw_by_index.pop(index)
                    output_map[output_index] = prefill_real_value_c_group_to_pairfold_fhe(
                        group_inputs,
                        pairfold_layout=pairfold_layout,
                        crypto_context=context,
                    )
            raw_by_index.clear()
            outputs = list(output_map.values())
            limbs["converted"] = [int(c.state.cur_limbs) for c in outputs]
            stage["convert"] = time.perf_counter() - started
        elif direct_key:
            keys, values = raw[:pair_count], raw[pair_count:]
            # ``prefill_emitted_key_c_direct_to_pairfold_fhe`` consumes K
            # inputs.  Retain only the not-yet-transferred V inputs while it
            # runs so a K-side exception still releases V exactly once.
            raw = list(values)
            pairfold_keys = prefill_emitted_key_c_direct_to_pairfold_fhe(
                keys,
                attention_layout=attention_layout,
                pairfold_layout=pairfold_layout,
                crypto_context=context,
            )
            # The V converter owns its inputs from the call boundary onward;
            # its own finally block handles an exception after this point.
            raw = []
            pairfold_values = prefill_real_value_c_to_pairfold_fhe(
                values,
                attention_layout=attention_layout,
                pairfold_layout=pairfold_layout,
                crypto_context=context,
            )
            outputs = list(pairfold_keys) + list(pairfold_values)
            pairfold_keys = ()
            pairfold_values = ()
        else:
            keys, values = raw[:pair_count], raw[pair_count:]
            raw = []
            pairfold_keys, pairfold_values = prefill_c_to_pairfold_fhe(
                keys,
                values,
                attention_layout=attention_layout,
                pairfold_layout=pairfold_layout,
                crypto_context=context,
                emitted_key_c=True,
            )
            outputs = list(pairfold_keys) + list(pairfold_values)
        limbs["converted"] = [int(c.state.cur_limbs) for c in outputs]
        if not owner_routing:
            stage["convert"] = time.perf_counter() - started

        # 3. refresh + drop my share, gather the four cache ciphers
        # In the owner route ``outputs`` is a sparse local subset keyed by
        # global PairFold output index.  The legacy route keeps the dense
        # four-output list and its original refresh partition.
        mine = (
            tuple(sorted(output_map))
            if owner_routing
            else kv_cache_refresh_indices(rank, world_size, count=len(outputs))
        )
        if kv_limbs is None:
            if owner_routing:
                if gather_ready_event is not None:
                    record = getattr(gather_ready_event, "record", None)
                    if callable(record):
                        record()
                if gather_gate is not None:
                    gather_gate.wait()
                final = list(
                    all_gather_ciphers(
                        transport, f"{tag}/pairfold", output_map,
                        count=2 * int(pairfold_layout.key_value_groups),
                        crypto_context=context, comm=comm,
                    )
                )
                output_map = {}
            else:
                final = outputs
                # The dense no-refresh path has no final collective/gate, but
                # side-stream callers still need a conversion completion event
                # before issuing the next FHE station on the caller stream.
                if gather_ready_event is not None:
                    record = getattr(gather_ready_event, "record", None)
                    if callable(record):
                        record()
            outputs = []
            stage["refresh"] = 0.0
            stage["gather_pairfold"] = 0.0 if not owner_routing else float(comm.get(f"{tag}/pairfold_seconds", 0.0))
        else:
            target = int(kv_limbs)
            started = time.perf_counter()
            refresh_items = output_map.items() if owner_routing else ((index, outputs[index]) for index in mine)
            for index, cipher in refresh_items:
                working = cipher
                fresh = None
                try:
                    if int(cipher.state.cur_limbs) < target:
                        fresh = _bootstrap_refresh(
                            bootstrap_operator,
                            cipher,
                            iterations=int(bootstrap_iterations),
                            synchronize=bool(synchronize),
                        )
                        bootstrap_calls += 1
                        working = fresh
                    refreshed_out[int(index)] = drop_to_limbs((working,), target_limbs=target, crypto_context=context)[0]
                finally:
                    if fresh is not None:
                        release_if_supported(fresh)
            for cipher in outputs:
                release_if_supported(cipher)
            outputs = []
            stage["refresh"] = time.perf_counter() - started
            # A resident side-stream caller can publish the device dependency
            # before the final collective, then hold that collective until the
            # competing PairFM all-to-all has completed.  This preserves the
            # transport's one-round-in-flight rule while allowing PairFM's
            # local reduction to overlap the KV gather.
            if gather_ready_event is not None:
                record = getattr(gather_ready_event, "record", None)
                if callable(record):
                    record()
            if gather_gate is not None:
                gather_gate.wait()
            started = time.perf_counter()
            final = list(
                all_gather_ciphers(
                    transport, f"{tag}/pairfold", refreshed_out,
                    count=2 * int(pairfold_layout.key_value_groups),
                    crypto_context=context, comm=comm,
                )
            )
            refreshed_out = {}
            output_map = {}
            stage["gather_pairfold"] = time.perf_counter() - started
        limbs["cache"] = [int(c.state.cur_limbs) for c in final]

        # 4. the page
        started = time.perf_counter()
        token = PairFoldTokenWrite(
            key_groups=tuple(final[:PAIRFOLD_GROUP_COUNT]),
            value_groups=tuple(final[PAIRFOLD_GROUP_COUNT:]),
            column=0,
        )
        final = []
        page = PairFoldCipherPage.from_token_write(token, occupied=occupied)
        token = None
        cache = PairFoldLayerCache(pages=(page,), content_tokens=occupied)
        page = None
        stage["page"] = time.perf_counter() - started
        return PrefillKVCacheResult(
            cache=cache,
            stage_seconds={k: float(v) for k, v in stage.items()},
            comm_bytes={k: int(v) for k, v in comm.items() if k.endswith(("_sent", "_received"))},
            comm_seconds=float(sum(v for k, v in comm.items() if k.endswith("_seconds"))),
            contributed={"key_pairs": tuple(key_pairs), "value_pairs": tuple(value_pairs)},
            refreshed=tuple(mine),
            bootstrap_calls=int(bootstrap_calls),
            limbs=limbs,
        )
    except BaseException:
        seen: set[int] = set()
        for cipher in (
            list(key_c.values()) + list(value_c.values()) + list(owned.values())
            + raw + list(raw_by_index.values()) + outputs
            + list(pairfold_keys) + list(pairfold_values) + list(output_map.values())
            + list(refreshed_out.values()) + final
        ):
            identity = id(cipher)
            if identity not in seen:
                release_if_supported(cipher)
                seen.add(identity)
        if token is not None:
            token.release()
        if page is not None:
            page.release()
        raise
