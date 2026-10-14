"""Host-staging transport between T2 worker processes.

Each rank owns one inbox (a ``multiprocessing`` queue).  ``send`` puts a
:class:`Message` into the destination's inbox; ``recv`` pops from the local
inbox and buffers anything that arrives out of order.  Collectives are built
from those two primitives and are *synchronous* by construction: every rank
must issue the same collectives in the same order (a per-transport sequence
number is folded into the tag so two rounds with the same user tag cannot be
confused).

Payloads are usually :class:`~llama3fhe.parallel.wire.CipherPacket` objects whose
components are host NumPy arrays, but any picklable object works (the
transport is exercised on the CPU with numpy arrays in the unit tests).



Staged blocks are owned by the sender and released two *symmetric* collective
rounds later (``all_gather``, ``all_to_all``, ``barrier``).  That is only safe
because in those rounds every rank both sends and receives, so a rank cannot
finish round ``n+1`` before all its peers have consumed its round-``n``
blocks.  ``send`` and ``gather_to`` are not symmetric (the non-root side never
waits), so they always pickle the payload through the queue and ``gather_to``
runs on its own sequence: it must not advance the symmetric round counter,
otherwise a fast non-root rank running ``all_to_all -> gather_to -> barrier``
would unlink its ``all_to_all`` blocks while a slower peer is still copying
them out.  Whatever is still staged is unlinked in
:meth:`HostStagingTransport.close`.  Set ``shared_memory=False`` to disable
staging altogether.

**Split-phase rounds.**  ``all_gather_isend`` / ``all_to_all_isend`` perform
the send half of a round and return a :class:`CollectiveHandle` whose
``wait()`` is the receive half, so a station can overlap the peers' skew and
transfer with its own GPU work (attention input: the lane all-gather hides
behind the V partials, the partial all-to-all behind Q/K).  The release
protocol above is unchanged *because* at most one round may be in flight:
``wait()`` raises if another round was started in between.

The attention stages use two collectives:
``all_gather`` for the attention-input lane ciphers and ``all_to_all`` for the
PairFM partials.  There is intentionally no reduce: CKKS partials may only be
added by the owning rank with the FHE kernel.

**Backends.**  :class:`Transport` is the contract every station is written
against; :func:`build_transport` picks the implementation by name
(``T2_TRANSPORT`` in the worker environment).  Three exist:



A device-native backend (NCCL) would slot in as a fourth class that
satisfies :class:`Transport`; ``tests/test_parallel_transport.py::
test_transport_contract_and_factory`` is the checklist it must pass.
"""

from __future__ import annotations

import collections
import dataclasses
import multiprocessing
import os
import queue as _queue
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory as _shm
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Message:
    src: int
    dst: int
    tag: str
    payload: Any = None
    sent_at: float = 0.0


def _default_mp_context():
    for module_name in ("easyfhe.multiprocessing", "torch.multiprocessing"):
        try:
            module = __import__(module_name, fromlist=["get_context"])
            return module.get_context("spawn")
        except Exception:
            continue
    return multiprocessing.get_context("spawn")


@dataclass(frozen=True)
class TransportEndpoints:
    """One inbox per rank.  Picklable, so it can be handed to spawned workers."""

    world_size: int
    inboxes: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.inboxes) != int(self.world_size):
            raise ValueError("inboxes must contain exactly one queue per rank")


def build_transport_endpoints(world_size: int, *, context=None) -> TransportEndpoints:
    """Create the inboxes with a spawn-capable multiprocessing context."""

    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    context = _default_mp_context() if context is None else context
    return TransportEndpoints(
        world_size=world_size,
        inboxes=tuple(context.Queue() for _ in range(world_size)),
    )


# ------------------------------------------------------- shared memory staging
@dataclass(frozen=True)
class SharedArrayHandle:
    """Reference to one NumPy array staged in a shared-memory block."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def _attach_block(name: str) -> _shm.SharedMemory:
    """Attach to an existing block without registering it with the resource tracker.

    The *sender* owns (and unlinks) the block; a receiver that merely maps it
    must not let ``resource_tracker`` think it leaked when it closes.
    """

    try:
        return _shm.SharedMemory(name=name, track=False)  # Python >= 3.13
    except TypeError:
        block = _shm.SharedMemory(name=name)
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(block._name, "shared_memory")  # noqa: SLF001
        except Exception:
            pass
        return block


def _stage_array(array: np.ndarray, blocks: list[_shm.SharedMemory]) -> SharedArrayHandle:
    array = np.ascontiguousarray(array)
    block = _shm.SharedMemory(create=True, size=max(1, int(array.nbytes)))
    blocks.append(block)
    view = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
    view[...] = array
    return SharedArrayHandle(
        name=block.name, shape=tuple(int(v) for v in array.shape), dtype=str(array.dtype), nbytes=int(array.nbytes)
    )


def _is_device_tensor(value: Any) -> bool:
    """A (shim) torch tensor living on a CUDA device -- never a NumPy array."""

    if isinstance(value, np.ndarray):
        return False
    return bool(getattr(value, "is_cuda", False)) and callable(getattr(value, "data_ptr", None))


def _device_tensor_to_host(value: Any) -> np.ndarray:
    """``.cpu().numpy()`` copy of a device tensor (the ``"host"`` backend path)."""

    host = value.cpu()
    detach = getattr(host, "detach", None)
    if callable(detach):
        host = detach()
    return np.array(host.numpy(), copy=True)


def _stage_shared(payload: Any, blocks: list[_shm.SharedMemory], *, device_hook=None) -> Any:
    """Replace every ``ndarray`` in ``payload`` with a :class:`SharedArrayHandle`.

    Device tensors (``pack_cipher(host=False)``) are handed to ``device_hook``
    when given (the pinned arena) and otherwise copied to the host first, so a
    station may keep components on the device with any backend.
    """

    if isinstance(payload, np.ndarray):
        return _stage_array(payload, blocks)
    if _is_device_tensor(payload):
        if device_hook is not None:
            staged = device_hook(payload)
            if staged is not None:
                return staged
        return _stage_array(_device_tensor_to_host(payload), blocks)
    if isinstance(payload, (list, tuple)):
        staged = [_stage_shared(item, blocks, device_hook=device_hook) for item in payload]
        return type(payload)(staged) if isinstance(payload, tuple) else staged
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        changes = {}
        for field_ in dataclasses.fields(payload):
            value = getattr(payload, field_.name)
            staged = _stage_shared(value, blocks, device_hook=device_hook)
            if staged is not value:
                changes[field_.name] = staged
        return dataclasses.replace(payload, **changes) if changes else payload
    return payload


def _unstage_shared(payload: Any, *, handle_hook=None) -> Any:
    """Copy every :class:`SharedArrayHandle` in ``payload`` back into a private array.

    ``handle_hook`` (the pinned arena) turns :class:`PinnedHandle` references
    into device tensors; without it they are copied out as NumPy arrays.
    """

    if isinstance(payload, PinnedHandle):
        if handle_hook is not None:
            return handle_hook(payload)
        block = _attach_block(payload.name)
        try:
            view = np.ndarray(
                payload.shape, dtype=np.dtype(payload.numpy_dtype), buffer=block.buf, offset=int(payload.offset)
            )
            return np.array(view, copy=True)
        finally:
            del view
            block.close()
    if isinstance(payload, SharedArrayHandle):
        block = _attach_block(payload.name)
        try:
            view = np.ndarray(payload.shape, dtype=np.dtype(payload.dtype), buffer=block.buf)
            return np.array(view, copy=True)
        finally:
            del view
            block.close()
    if isinstance(payload, (list, tuple)):
        items = [_unstage_shared(item, handle_hook=handle_hook) for item in payload]
        return type(payload)(items) if isinstance(payload, tuple) else items
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        changes = {}
        for field_ in dataclasses.fields(payload):
            value = getattr(payload, field_.name)
            restored = _unstage_shared(value, handle_hook=handle_hook)
            if restored is not value:
                changes[field_.name] = restored
        return dataclasses.replace(payload, **changes) if changes else payload
    return payload


def _payload_has_arrays(payload: Any) -> bool:
    """True when ``payload`` carries a NumPy array or a device tensor (needs staging)."""

    if isinstance(payload, np.ndarray) or _is_device_tensor(payload):
        return True
    if isinstance(payload, (list, tuple)):
        return any(_payload_has_arrays(item) for item in payload)
    if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
        return any(_payload_has_arrays(getattr(payload, f.name)) for f in dataclasses.fields(payload))
    return False


# ------------------------------------------------------- pinned host arena
@dataclass(frozen=True)
class PinnedHandle:
    """Reference to one device tensor staged in a rank's pinned arena slab."""

    name: str
    offset: int
    shape: tuple[int, ...]
    #: ``str(tensor.dtype)`` of the shim tensor (``"torch.uint64"``).
    dtype: str
    nbytes: int

    @property
    def numpy_dtype(self) -> str:
        return self.dtype.removeprefix("torch.")


_PINNED_ALIGN = 256


class PinnedArena:
    """Two pinned shared-memory slabs (one per round parity) owned by one rank.

    Sending: :meth:`stage` copies a device tensor into the slab of the current
    round (``copy_(non_blocking=True)`` on the current stream) and returns a
    :class:`PinnedHandle`; the caller synchronises the device once per round
    before queueing the handles.  Receiving: :meth:`restore` maps the peer's
    slab (kept mapped and registered, one ``cudaHostRegister`` per peer slab)
    and copies the bytes into a fresh device tensor, again stream-ordered and
    without a host sync.

    Reuse: a slab is rewritten two symmetric rounds later, the same distance
    at which :class:`HostStagingTransport` unlinks its blocks -- entering
    round ``n`` means every peer finished round ``n-1``, and a peer only
    enters a round after its previous H2D copies completed
    (:meth:`PinnedStagingTransport._put` drains them).  ``cudaHostRegister``
    happens lazily on the first device tensor, so CPU-only tests never touch
    CUDA.  Slabs that would overflow return ``None`` from :meth:`stage` and
    the caller falls back to the plain shared-memory path.
    """

    def __init__(self, rank: int, slab_bytes: int) -> None:
        self.rank = int(rank)
        self.slab_bytes = max(int(slab_bytes), _PINNED_ALIGN)
        self._slabs: list[_shm.SharedMemory | None] = [None, None]
        self._registered: set[str] = set()
        self._offsets = [0, 0]
        self._parity = 0
        self._peer_blocks: dict[str, _shm.SharedMemory] = {}
        self._torch = None
        #: ``(data_ptr, nbytes) -> handle`` of the current round: the same
        #: tensor sent to several peers (``all_to_all`` with one packet per
        #: group member) is copied into the slab once.
        self._round_cache: dict[tuple[int, int], PinnedHandle] = {}
        self.stats = {"staged": 0, "staged_bytes": 0, "restored": 0, "restored_bytes": 0, "overflow": 0, "deduped": 0}

    # -- lazy CUDA
    def _module(self):
        if self._torch is None:
            import easyfhe  # the torch shim; only imported once a device tensor shows up

            self._torch = easyfhe
        return self._torch

    def _register(self, block: _shm.SharedMemory) -> None:
        if block.name in self._registered:
            return
        torch = self._module()
        pointer = int(np.frombuffer(block.buf, dtype=np.uint8).ctypes.data)
        err = torch.cuda.cudart().cudaHostRegister(pointer, int(block.size), 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({block.name}, {block.size} B) failed: {err}")
        self._registered.add(block.name)

    def _unregister(self, block: _shm.SharedMemory) -> None:
        if block.name not in self._registered:
            return
        try:
            pointer = int(np.frombuffer(block.buf, dtype=np.uint8).ctypes.data)
            self._module().cuda.cudart().cudaHostUnregister(pointer)
        except Exception:
            pass
        self._registered.discard(block.name)

    def _slab(self, parity: int) -> _shm.SharedMemory:
        block = self._slabs[parity]
        if block is None:
            name = f"t2arena_r{self.rank}_p{parity}_{os.getpid()}"
            try:
                block = _shm.SharedMemory(name=name, create=True, size=self.slab_bytes)
            except FileExistsError:
                stale = _attach_block(name)
                stale.close()
                stale.unlink()
                block = _shm.SharedMemory(name=name, create=True, size=self.slab_bytes)
            self._slabs[parity] = block
        self._register(block)
        return block

    # -- rounds
    def begin_round(self, sequence: int) -> None:
        """Select (and reset) the slab of the round ``sequence`` (parity ``sequence % 2``)."""

        self._parity = int(sequence) % 2
        self._offsets[self._parity] = 0
        self._round_cache = {}

    # -- sender
    def stage(self, tensor: Any) -> PinnedHandle | None:
        """Copy ``tensor`` (device) into the current slab; ``None`` when it does not fit."""

        nbytes = int(tensor.numel()) * int(tensor.element_size())
        key = (int(tensor.data_ptr()), nbytes)
        cached = self._round_cache.get(key)
        if cached is not None and cached.shape == tuple(int(v) for v in tensor.shape):
            self.stats["deduped"] += 1
            return cached
        offset = self._offsets[self._parity]
        end = offset + nbytes
        if end > self.slab_bytes:
            self.stats["overflow"] += 1
            return None
        block = self._slab(self._parity)
        torch = self._module()
        shape = tuple(int(v) for v in tensor.shape)
        dtype_name = str(tensor.dtype)
        dtype = getattr(torch, dtype_name.removeprefix("torch."))
        count = int(tensor.numel())
        view = torch.frombuffer(block.buf, dtype=dtype, count=count, offset=int(offset))
        try:
            view.view(shape).copy_(tensor, non_blocking=True)
        finally:
            del view  # an exported buffer would block ``block.close()``
        self._offsets[self._parity] = (end + _PINNED_ALIGN - 1) // _PINNED_ALIGN * _PINNED_ALIGN
        self.stats["staged"] += 1
        self.stats["staged_bytes"] += nbytes
        handle = PinnedHandle(name=block.name, offset=int(offset), shape=shape, dtype=dtype_name, nbytes=nbytes)
        self._round_cache[key] = handle
        return handle

    # -- receiver
    def _peer_block(self, name: str) -> _shm.SharedMemory:
        block = self._peer_blocks.get(name)
        if block is None:
            block = _attach_block(name)
            self._peer_blocks[name] = block
        self._register(block)
        return block

    def restore(self, handle: PinnedHandle, device: str = "cuda") -> Any:
        """A fresh device tensor filled from the peer's slab (stream-ordered, no host sync)."""

        block = self._peer_block(handle.name)
        torch = self._module()
        dtype = getattr(torch, handle.numpy_dtype)
        count = 1
        for extent in handle.shape:
            count *= int(extent)
        view = torch.frombuffer(block.buf, dtype=dtype, count=count, offset=int(handle.offset))
        try:
            out = torch.empty(handle.shape, dtype=dtype, device=str(device))
            out.reshape(-1).copy_(view, non_blocking=True)
        finally:
            del view
        self.stats["restored"] += 1
        self.stats["restored_bytes"] += int(handle.nbytes)
        return out

    # -- teardown
    def close(self) -> None:
        for name, block in list(self._peer_blocks.items()):
            self._unregister(block)
            try:
                block.close()
            except BufferError:
                pass
            self._peer_blocks.pop(name, None)
        for parity, block in enumerate(self._slabs):
            if block is None:
                continue
            self._unregister(block)
            try:
                block.close()
            except BufferError:
                pass
            try:
                block.unlink()
            except FileNotFoundError:
                pass
            self._slabs[parity] = None


@dataclass
class TransportStats:
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    send_seconds: float = 0.0
    recv_wait_seconds: float = 0.0
    collectives: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "messages_sent": int(self.messages_sent),
            "messages_received": int(self.messages_received),
            "bytes_sent": int(self.bytes_sent),
            "bytes_received": int(self.bytes_received),
            "send_seconds": float(self.send_seconds),
            "recv_wait_seconds": float(self.recv_wait_seconds),
            "collectives": int(self.collectives),
        }


def _payload_bytes(payload: Any) -> int:
    size = getattr(payload, "nbytes", None)
    if isinstance(size, (int, float)):
        return int(size)
    if isinstance(payload, (list, tuple)):
        return sum(_payload_bytes(item) for item in payload)
    return 0


@runtime_checkable
class Transport(Protocol):
    """What a station may ask of the communication layer (one instance per rank).

    Every collective is synchronous and must be issued by all ranks in the
    same order; ``tag`` is a human label only (implementations fold their own
    sequence numbers in).  Payloads are picklable objects -- in practice
    :class:`~llama3fhe.parallel.wire.CipherPacket` (host NumPy components) or
    ``None`` -- and are returned indexed by rank.  ``stats`` accumulates
    ``bytes_sent`` / ``bytes_received`` / ``recv_wait_seconds`` /
    ``send_seconds`` / ``collectives`` so every backend reports comparable
    numbers in ``summary.json``.  ``device_payloads`` tells a station whether
    it may hand over components that still live on the device
    (``pack_cipher(host=False)``); the host backends convert them anyway, so
    the flag is an optimisation hint, not a contract.  Implementations:
    :class:`HostStagingTransport`, :class:`PinnedStagingTransport`,
    :class:`~llama3fhe.parallel.emulation.InProcessTransport`.
    """

    rank: int
    stats: TransportStats
    device_payloads: bool

    @property
    def world_size(self) -> int: ...

    @property
    def peers(self) -> tuple[int, ...]: ...

    def send(self, dst: int, tag: str, payload: Any) -> None: ...

    def recv(self, src: int, tag: str, *, timeout: float | None = None) -> Any: ...

    def all_gather(self, tag: str, payload: Any, *, timeout: float | None = None) -> list[Any]: ...

    def all_to_all(
        self, tag: str, payloads_by_dst: Sequence[Any], *, timeout: float | None = None
    ) -> list[Any]: ...

    def all_gather_isend(self, tag: str, payload: Any, *, timeout: float | None = None) -> "CollectiveHandle": ...

    def all_to_all_isend(
        self, tag: str, payloads_by_dst: Sequence[Any], *, timeout: float | None = None
    ) -> "CollectiveHandle": ...

    def barrier(self, tag: str = "barrier", *, timeout: float | None = None) -> None: ...

    def gather_to(self, root: int, tag: str, payload: Any, *, timeout: float | None = None): ...

    def close(self) -> None: ...


#: Backend names accepted by :func:`build_transport` / ``T2_TRANSPORT``.
TRANSPORT_BACKENDS: tuple[str, ...] = ("host", "pinned", "inprocess")
DEFAULT_TRANSPORT = "host"
#: ``T2_PINNED_ARENA_MB``: size of each of the two pinned slabs per rank.
DEFAULT_PINNED_ARENA_MB = 512


def build_transport(rank: int, endpoints: "TransportEndpoints | None", *, kind: str | None = None, **options: Any):
    """Instantiate the transport backend ``kind`` for ``rank``.

    ``kind`` defaults to ``T2_TRANSPORT`` in the environment, then ``"host"``.
    ``"host"`` needs the launcher's :class:`TransportEndpoints`; ``"inprocess"``
    needs ``world=`` (an :class:`~llama3fhe.parallel.emulation.EmulationWorld`)
    and is normally constructed by the emulation itself.  Unknown names raise
    ``ValueError`` listing :data:`TRANSPORT_BACKENDS`, so a typo in the
    environment fails at worker start instead of at the first collective.
    """

    name = str(kind if kind is not None else os.environ.get("T2_TRANSPORT", DEFAULT_TRANSPORT)).lower()
    if name == "host":
        if endpoints is None:
            raise ValueError("the host-staging transport needs the launcher's TransportEndpoints")
        return HostStagingTransport(int(rank), endpoints, **options)
    if name == "pinned":
        if endpoints is None:
            raise ValueError("the pinned-staging transport needs the launcher's TransportEndpoints")
        return PinnedStagingTransport(int(rank), endpoints, **options)
    if name == "inprocess":
        from ..emulation import InProcessTransport

        world = options.pop("world", None)
        if world is None:
            raise ValueError("the in-process transport needs world=EmulationWorld")
        return InProcessTransport(int(rank), world, **options)
    raise ValueError(f"unknown transport backend {name!r}; choose from {TRANSPORT_BACKENDS}")


@dataclass
class CollectiveHandle:
    """The receive half of a split-phase collective (``*_isend`` -> ``wait``).

    ``wait()`` returns exactly what the blocking collective would have
    returned (payloads indexed by rank).  It may be called once; calling it
    after another round started on the same transport raises, because the
    release protocol allows only one in-flight symmetric round.
    """

    transport: Any
    round_tag: str
    sequence: int
    own: Any
    srcs: tuple[int, ...]
    timeout: float | None = None
    _result: Any = None
    _done: bool = False

    def wait(self) -> list[Any]:
        if not self._done:
            self._result = self.transport._collective_wait(self)
            self._done = True
        return self._result

    @property
    def done(self) -> bool:
        return bool(self._done)


class ReadyCollective:
    """A :class:`CollectiveHandle` look-alike wrapping an already completed result."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def wait(self) -> Any:
        return self._result

    @property
    def done(self) -> bool:
        return True


@dataclass
class HostStagingTransport:
    """Point-to-point plus the two collectives, for one rank."""

    rank: int
    endpoints: TransportEndpoints
    #: Seconds a ``recv`` may wait before raising ``TimeoutError`` (``None`` =
    #: forever).  A finite default turns a protocol bug into an error instead
    #: of eight silently hung processes.
    default_timeout: float | None = 900.0
    #: Stage NumPy payload arrays in shared memory instead of pickling them.
    shared_memory: bool = True
    stats: TransportStats = field(default_factory=TransportStats)
    _pending: dict[tuple[int, str], collections.deque] = field(
        default_factory=lambda: collections.defaultdict(collections.deque)
    )
    _sequence: int = 0
    #: Separate counter for the asymmetric ``gather_to`` (never releases blocks).
    _gather_sequence: int = 0
    #: Shared-memory blocks still owned by this rank, keyed by the symmetric
    #: collective round (``_sequence`` value) they were staged in.
    _staged: dict[int, list[_shm.SharedMemory]] = field(default_factory=dict)
    #: Stations may keep components on the device: this backend copies them
    #: to the host itself, so keeping them there only costs an extra sync.
    device_payloads: bool = False

    def __post_init__(self) -> None:
        self.rank = int(self.rank)
        if self.rank < 0 or self.rank >= int(self.endpoints.world_size):
            raise IndexError(f"rank {self.rank} outside world of size {self.endpoints.world_size}")

    def __enter__(self) -> "HostStagingTransport":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -------------------------------------------------------- shared memory
    def _stage(self, payload: Any) -> Any:
        """Stage ``payload`` arrays for the current collective round.

        No-op without arrays or with ``shared_memory=False``.  Only the
        symmetric collectives call this (see the module docstring).
        """

        if not self.shared_memory or not _payload_has_arrays(payload):
            return payload
        blocks = self._staged.setdefault(self._sequence, [])
        return _stage_shared(payload, blocks, device_hook=self._device_hook())

    def _device_hook(self):
        """Hook :func:`_stage_shared` uses for device tensors (``None``: copy to host)."""

        return None

    def _handle_hook(self):
        """Hook :func:`_unstage_shared` uses for :class:`PinnedHandle` (``None``: NumPy copy)."""

        return None

    def _end_stage(self) -> None:
        """Called once per round after every payload of the round was staged."""

    def _release_rounds(self, *, before_round: int) -> None:
        """Unlink blocks staged in collective rounds ``< before_round``."""

        for round_index in [r for r in self._staged if r < int(before_round)]:
            for block in self._staged.pop(round_index):
                try:
                    block.close()
                finally:
                    try:
                        block.unlink()
                    except FileNotFoundError:
                        pass

    def close(self) -> None:
        """Unlink every shared-memory block this rank still owns.

        Call once all peers have finished their last receive (after a final
        :meth:`barrier` or at process exit).
        """

        self._release_rounds(before_round=self._sequence + 1)

    @property
    def staged_blocks(self) -> int:
        return sum(len(blocks) for blocks in self._staged.values())

    @property
    def world_size(self) -> int:
        return int(self.endpoints.world_size)

    @property
    def peers(self) -> tuple[int, ...]:
        return tuple(other for other in range(self.world_size) if other != self.rank)

    # ------------------------------------------------------------------ p2p
    def send(self, dst: int, tag: str, payload: Any) -> None:
        """Point-to-point send; always pickles (no shared-memory staging)."""

        self._send(dst, tag, payload, stage=False)

    def _send(self, dst: int, tag: str, payload: Any, *, stage: bool) -> None:
        dst = int(dst)
        if dst < 0 or dst >= self.world_size:
            raise IndexError(f"destination {dst} outside world of size {self.world_size}")
        if dst == self.rank:
            raise ValueError("use local hand-off instead of sending to self")
        started = time.perf_counter()
        self._put(dst, str(tag), self._stage(payload) if stage else payload, started)
        self.stats.bytes_sent += _payload_bytes(payload)

    def _put(self, dst: int, tag: str, staged_payload: Any, started: float | None) -> None:
        """Queue one message; ``started`` (when given) is charged to ``send_seconds``."""

        self.endpoints.inboxes[dst].put(
            Message(
                src=self.rank, dst=dst, tag=tag, payload=staged_payload,
                sent_at=time.perf_counter() if started is None else started,
            )
        )
        if started is not None:
            self.stats.send_seconds += time.perf_counter() - started
        self.stats.messages_sent += 1

    def recv(self, src: int, tag: str, *, timeout: float | None = None) -> Any:
        src = int(src)
        key = (src, str(tag))
        waiting = self._pending.get(key)
        if waiting:
            message = waiting.popleft()
            return self._accept(message)
        deadline = None
        limit = self.default_timeout if timeout is None else timeout
        if limit is not None:
            deadline = time.perf_counter() + float(limit)
        inbox = self.endpoints.inboxes[self.rank]
        started = time.perf_counter()
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
            try:
                message = inbox.get(timeout=remaining)
            except _queue.Empty as exc:
                raise TimeoutError(
                    f"rank {self.rank}: no message from {src} tag {tag!r} within {limit}s "
                    f"(pending keys: {sorted(k for k, v in self._pending.items() if v)})"
                ) from exc
            if int(message.dst) != self.rank:
                raise RuntimeError(
                    f"rank {self.rank} received a message addressed to {message.dst}"
                )
            if (int(message.src), str(message.tag)) == key:
                self.stats.recv_wait_seconds += time.perf_counter() - started
                return self._accept(message)
            self._pending[(int(message.src), str(message.tag))].append(message)

    def _accept(self, message: Message) -> Any:
        payload = _unstage_shared(message.payload, handle_hook=self._handle_hook())
        self.stats.messages_received += 1
        self.stats.bytes_received += _payload_bytes(payload)
        return payload

    # ----------------------------------------------------------- collectives
    def _round_tag(self, tag: str) -> str:
        """Tag of the next *symmetric* round; frees the blocks of round ``n-2``.

        Entering round n: every peer has finished round n-2 (it could not have
        answered our round n-1 otherwise), so those blocks are free.  Round n-1
        blocks may still be mapped by a slower peer -> keep them.
        """

        self._sequence += 1
        self.stats.collectives += 1
        self._release_rounds(before_round=self._sequence - 1)
        return f"{tag}#{self._sequence}"

    def _gather_tag(self, tag: str) -> str:
        """Tag of the next ``gather_to``; asymmetric, so it releases nothing."""

        self._gather_sequence += 1
        self.stats.collectives += 1
        return f"{tag}#g{self._gather_sequence}"

    def all_gather(self, tag: str, payload: Any, *, timeout: float | None = None) -> list[Any]:
        """Every rank contributes one payload; returns them indexed by rank."""

        return self.all_gather_isend(tag, payload, timeout=timeout).wait()

    def all_to_all(
        self, tag: str, payloads_by_dst: Sequence[Any], *, timeout: float | None = None
    ) -> list[Any]:
        """``payloads_by_dst[d]`` goes to rank ``d``; returns what each rank sent us."""

        return self.all_to_all_isend(tag, payloads_by_dst, timeout=timeout).wait()

    # -- split-phase collectives: the send half now, the receive half later
    def all_gather_isend(self, tag: str, payload: Any, *, timeout: float | None = None) -> "CollectiveHandle":
        """Send half of :meth:`all_gather`; ``handle.wait()`` is the receive half.

        Between ``isend`` and ``wait`` the caller may run any amount of local
        (GPU) work but **no other collective round** (``wait`` checks this):
        the slab / block release protocol relies on at most one symmetric
        round being in flight (module docstring).
        """

        round_tag = self._round_tag(tag)
        started = time.perf_counter()
        staged = self._stage(payload)
        self._end_stage()
        for dst in self.peers:
            self._put(dst, round_tag, staged, None)
            self.stats.bytes_sent += _payload_bytes(payload)
        self.stats.send_seconds += time.perf_counter() - started  # staging + every put, once
        return CollectiveHandle(
            transport=self, round_tag=round_tag, sequence=self._sequence,
            own=payload, srcs=self.peers, timeout=timeout,
        )

    def all_to_all_isend(
        self, tag: str, payloads_by_dst: Sequence[Any], *, timeout: float | None = None
    ) -> "CollectiveHandle":
        """Send half of :meth:`all_to_all`; same in-flight rule as :meth:`all_gather_isend`."""

        if len(payloads_by_dst) != self.world_size:
            raise ValueError("all_to_all needs exactly one payload per rank")
        round_tag = self._round_tag(tag)
        started = time.perf_counter()
        staged = {dst: self._stage(payloads_by_dst[dst]) for dst in self.peers}
        self._end_stage()
        for dst in self.peers:
            self._put(dst, round_tag, staged[dst], None)
            self.stats.bytes_sent += _payload_bytes(payloads_by_dst[dst])
        self.stats.send_seconds += time.perf_counter() - started
        return CollectiveHandle(
            transport=self, round_tag=round_tag, sequence=self._sequence,
            own=payloads_by_dst[self.rank], srcs=self.peers, timeout=timeout,
        )

    def _collective_wait(self, handle: "CollectiveHandle") -> list[Any]:
        if int(handle.sequence) != int(self._sequence):
            raise RuntimeError(
                f"rank {self.rank}: collective round {handle.round_tag!r} was still in flight when "
                f"round #{self._sequence} started; wait() for it before issuing the next collective"
            )
        received: list[Any] = [None] * self.world_size
        received[self.rank] = handle.own
        for src in handle.srcs:
            received[src] = self.recv(src, handle.round_tag, timeout=handle.timeout)
        return received

    def barrier(self, tag: str = "barrier", *, timeout: float | None = None) -> None:
        self.all_gather(tag, None, timeout=timeout)

    def gather_to(self, root: int, tag: str, payload: Any, *, timeout: float | None = None):
        """Collect one payload per rank on ``root`` (others return ``None``).

        Asymmetric (non-root ranks do not wait), hence never staged and not a
        symmetric round: it neither advances ``_sequence`` nor frees blocks.
        """

        root = int(root)
        round_tag = self._gather_tag(tag)
        if self.rank != root:
            self._send(root, round_tag, payload, stage=False)
            return None
        gathered: list[Any] = [None] * self.world_size
        gathered[self.rank] = payload
        for src in self.peers:
            gathered[src] = self.recv(src, round_tag, timeout=timeout)
        return gathered


class PinnedStagingTransport(HostStagingTransport):
    """:class:`HostStagingTransport` whose device tensors ride a pinned arena.

    Same queues, tags and release protocol; only the payload path differs:
    device tensors in a symmetric collective are copied into this rank's
    :class:`PinnedArena` (one ``synchronize`` per round after all copies are
    queued) and restored by the receiver as fresh device tensors on its own
    card without a host sync.  NumPy arrays still use the plain shared-memory
    path, so the CPU tests and the ``"host"`` semantics are unchanged.
    """

    def __init__(
        self,
        rank: int,
        endpoints: TransportEndpoints,
        *,
        arena_mb: int | None = None,
        device: str = "cuda",
        **options: Any,
    ) -> None:
        super().__init__(int(rank), endpoints, **options)
        self.device_payloads = True
        self.device = str(device)
        size_mb = int(os.environ.get("T2_PINNED_ARENA_MB", DEFAULT_PINNED_ARENA_MB) if arena_mb is None else arena_mb)
        self.arena = PinnedArena(self.rank, size_mb << 20)
        self._device_dirty = False   # staged copies queued but not yet synchronised
        self._pending_h2d = False    # restored copies queued but not yet synchronised

    def _synchronize(self) -> None:
        from llama3fhe.backend import synchronize_device

        synchronize_device(self.device)

    def _device_hook(self):
        def stage(tensor):
            handle = self.arena.stage(tensor)
            if handle is not None:
                self._device_dirty = True
            return handle

        return stage

    def _handle_hook(self):
        def restore(handle: PinnedHandle):
            self._pending_h2d = True
            return self.arena.restore(handle, self.device)

        return restore

    def _round_tag(self, tag: str) -> str:
        round_tag = super()._round_tag(tag)
        self.arena.begin_round(self._sequence)
        return round_tag

    def _end_stage(self) -> None:
        """One device sync per round, before the first put of the round.

        Two reasons share the sync: the pinned bytes we staged must be
        complete before a peer reads them, and our previous-round H2D copies
        out of a peer's slab must have landed before that peer may learn
        (through our message) that it is allowed to overwrite the slab.
        """

        if self._device_dirty or self._pending_h2d:
            self._synchronize()
            self._device_dirty = self._pending_h2d = False

    def _put(self, dst: int, tag: str, staged_payload: Any, started: float | None) -> None:
        if self._pending_h2d:  # asymmetric sends (``gather_to``) skip ``_end_stage``
            self._synchronize()
            self._pending_h2d = False
        super()._put(dst, tag, staged_payload, started)

    def close(self) -> None:
        try:
            if self._device_dirty or self._pending_h2d:
                try:
                    self._synchronize()
                except Exception:
                    pass
            self._device_dirty = self._pending_h2d = False
            self.arena.close()
        finally:
            super().close()


__all__ = [
    "CollectiveHandle",
    "ReadyCollective",
    "DEFAULT_PINNED_ARENA_MB",
    "DEFAULT_TRANSPORT",
    "HostStagingTransport",
    "Message",
    "PinnedArena",
    "PinnedHandle",
    "PinnedStagingTransport",
    "SharedArrayHandle",
    "TRANSPORT_BACKENDS",
    "Transport",
    "TransportEndpoints",
    "TransportStats",
    "build_transport",
    "build_transport_endpoints",
]
