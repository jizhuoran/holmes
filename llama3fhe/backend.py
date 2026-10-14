from __future__ import annotations

"""Small execution-session boundary around the native EasyFHE context.

Production context construction lives in :mod:`llama3fhe.application`.  The
``create_sim_context`` function remains only for isolated operator tests and
benchmarks; model and operator code must not call it implicitly.
"""

import hashlib
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from math import log2

import easyfhe
import numpy as np
from easyfhe import fhe

from .config import SimulatorConfig

_LAYER_CACHE_NAME = re.compile(r"model\.layers\.\d+\.")
# ``bench/t2_layer.py`` prefixes every constant with ``t2l.r{rank}.i{iteration}``;
# public layout masks are identical across ranks and iterations, so the prefix
# must not split the session cache (it made every lookup a miss).
_RANK_ITER_CACHE_NAME = re.compile(r"\.r\d+\.i\d+\.")

_SIGNATURE_MODE = os.environ.get("LLAMA_FHE_PLAIN_CACHE_SIGNATURE", "fast").strip().lower()
_VERIFY_SIGNATURE = os.environ.get("LLAMA_FHE_PLAIN_CACHE_VERIFY", "0") not in ("", "0", "false", "no")
_FULL_HASH_MAX_BYTES = 1 << 20
# ``LLAMA_FHE_PLAIN_CACHE_TRACE=1``: print one line per ``constant_bundle``
# lookup (outcome, key, source MiB, cache bytes) to inventory the mask encodes.
_TRACE_CACHE = os.environ.get("LLAMA_FHE_PLAIN_CACHE_TRACE", "0") not in ("", "0", "false", "no")


def _trace_cache(outcome: str, cache_key: object, source_bytes: int, cache) -> None:
    info = cache.info()
    print(
        f"[plain-cache] {outcome:9s} src={source_bytes / 2**20:8.1f}MiB "
        f"cached={info['bytes'] / 2**30:6.2f}GiB entries={info['entries']} "
        f"key={_normalized_constant_name(cache_key)!r}",
        flush=True,
    )
_SAMPLE_STRIDE_BYTES = 509  # prime; strided so periodic layouts do not alias


def _normalized_constant_name(value: object) -> object:
    if isinstance(value, str):
        value = _LAYER_CACHE_NAME.sub("model.layers.*.", value)
        return _RANK_ITER_CACHE_NAME.sub(".r*.i*.", value)
    if isinstance(value, tuple):
        return tuple(_normalized_constant_name(item) for item in value)
    if isinstance(value, list):
        return tuple(_normalized_constant_name(item) for item in value)
    return value


def _full_array_digest(array: np.ndarray) -> str:
    return hashlib.blake2b(
        memoryview(array).cast("B"), digest_size=16
    ).hexdigest()


_FAST_BLOCK_WORDS = 256
# Odd 64-bit multiplier ramp: position-weighted so that moving one value inside
# a block changes the block sum (a plain sum would not see a 0/1 mask shift).
_FAST_RAMP = np.arange(1, _FAST_BLOCK_WORDS + 1, dtype=np.uint64) * np.uint64(
    0x9E3779B97F4A7C15
)


def _fast_array_digest(array: np.ndarray) -> str:
    """Cheap, position-sensitive fingerprint (see ``_SIGNATURE_MODE``)."""

    raw = memoryview(array).cast("B")
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(bytes(raw[::_SAMPLE_STRIDE_BYTES]))
    whole = array.nbytes - array.nbytes % 8
    if whole:
        words = np.frombuffer(raw[:whole], dtype=np.uint64)
        usable = words.size - words.size % _FAST_BLOCK_WORDS
        with np.errstate(over="ignore"):
            if usable:
                blocks = words[:usable].reshape(-1, _FAST_BLOCK_WORDS) * _FAST_RAMP
                hasher.update(blocks.sum(axis=1, dtype=np.uint64).tobytes())
            if usable < words.size:
                tail = words[usable:] * _FAST_RAMP[: words.size - usable]
                hasher.update(tail.sum(dtype=np.uint64).reshape(1).tobytes())
    return hasher.hexdigest()


def _array_cache_signature(values: np.ndarray) -> tuple[object, ...]:
    array = np.ascontiguousarray(values)
    if _SIGNATURE_MODE == "full" or array.nbytes <= _FULL_HASH_MAX_BYTES:
        digest = _full_array_digest(array)
    else:
        digest = _fast_array_digest(array)
    return (tuple(int(v) for v in array.shape), array.dtype.str, digest)


class _ManagedConstantBundle:
    """ConstantBundle proxy whose final plaintext bytes obey a session cap."""

    def __init__(self, owner, key, bundle, source_bytes: int, cached: bool):
        self._owner = owner
        self._key = key
        self._bundle = bundle
        self._source_bytes = int(source_bytes)
        self._cached = bool(cached)
        self._accounted_bytes = int(source_bytes) if cached else 0

    def plaintext(self, *args, **kwargs):
        plaintext = self._bundle.plaintext(*args, **kwargs)
        self._owner._after_plaintext(self)
        return plaintext

    def cache_info(self):
        return self._bundle.cache_info()

    def clear_cache(self) -> None:
        self._bundle.clear_cache()

    def __getattr__(self, name):
        return getattr(self._bundle, name)


class SessionPlaintextCache:
    """First-fit cache for immutable public masks shared by all layers.

    The cache intentionally never evicts an admitted entry during a forward
    scan. Replacing an early mask with a later one would make the next layer's
    identical scan thrash and produce no hits. An entry that would cross the
    byte budget remains a transient bundle for its current caller.
    """

    def __init__(self, *, limit_bytes: int, max_source_bytes: int) -> None:
        self.limit_bytes = max(0, int(limit_bytes))
        self.max_source_bytes = max(0, int(max_source_bytes))
        self._entries: OrderedDict[object, _ManagedConstantBundle] = OrderedDict()
        self._verify_digests: dict[object, object] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.skips = 0
        self.closed = False

    def accepts_source(self, source_bytes: int) -> bool:
        return bool(
            self.limit_bytes > 0
            and int(source_bytes) <= int(self.max_source_bytes)
        )

    def transient(self, factory, *, source_bytes: int):
        if self.closed:
            raise RuntimeError("session plaintext cache is closed")
        self.misses += 1
        self.skips += 1
        return _ManagedConstantBundle(
            self,
            None,
            factory(False),
            source_bytes=int(source_bytes),
            cached=False,
        )

    def verify(self, key, full_digests) -> None:
        """Debug check: a hit must carry the same full digest as its admission."""

        previous = self._verify_digests.get(key)
        if previous is None:
            self._verify_digests[key] = full_digests
        elif previous != full_digests:
            raise RuntimeError(
                "session plaintext cache signature collision: two constants "
                f"with different contents share the key {key[0]!r}"
            )

    def get(self, key, factory, *, source_bytes: int):
        if self.closed:
            raise RuntimeError("session plaintext cache is closed")
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        source_bytes = int(source_bytes)
        admit = self.accepts_source(source_bytes) and (
            self._bytes + source_bytes <= self.limit_bytes
        )
        wrapped = _ManagedConstantBundle(
            self,
            key,
            factory(admit),
            source_bytes=source_bytes,
            cached=admit,
        )
        if admit:
            self._entries[key] = wrapped
            self._bytes += source_bytes
        else:
            self.skips += 1
        return wrapped

    def _after_plaintext(self, entry: _ManagedConstantBundle) -> None:
        if not entry._cached:
            return
        cache_bytes = int(entry._bundle.cache_info().get("total_bytes", 0))
        wanted = int(entry._source_bytes + cache_bytes)
        delta = wanted - int(entry._accounted_bytes)
        if delta <= 0:
            return
        if self._bytes + delta <= self.limit_bytes:
            self._bytes += delta
            entry._accounted_bytes = wanted
            return
        # Stable first-fit: reject this newest/growing entry instead of
        # evicting earlier masks that the next layer will encounter first.
        self._entries.pop(entry._key, None)
        self._bytes -= int(entry._accounted_bytes)
        entry._accounted_bytes = 0
        entry._cached = False
        entry._bundle.clear_cache()
        self.skips += 1

    def info(self) -> dict[str, int]:
        return {
            "entries": int(len(self._entries)),
            "bytes": int(self._bytes),
            "limit_bytes": int(self.limit_bytes),
            "max_source_bytes": int(self.max_source_bytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "skips": int(self.skips),
        }

    def clear(self) -> None:
        entries, self._entries = tuple(self._entries.values()), OrderedDict()
        self._verify_digests.clear()
        self._bytes = 0
        self.closed = True
        for entry in entries:
            entry._cached = False
            entry._accounted_bytes = 0
            entry._bundle.clear_cache()


@dataclass
class EasyFHEContext:
    """Data-only session joining the client and native CKKS context."""

    tensor: object
    fhe: object
    client: object
    context: object
    max_slots: int
    keygen_wall_time_s: float = 0.0
    plaintext_cache: SessionPlaintextCache = field(init=False, repr=False)

    def __post_init__(self) -> None:
        limit_gb = float(os.environ.get("LLAMA_FHE_PLAIN_CACHE_GB", "0"))
        max_source_mb = float(
            os.environ.get("LLAMA_FHE_PLAIN_CACHE_MAX_SOURCE_MB", "1")
        )
        self.plaintext_cache = SessionPlaintextCache(
            limit_bytes=int(limit_gb * (1024**3)),
            max_source_bytes=int(max_source_mb * (1024**2)),
        )

    def configure_plaintext_cache(
        self, *, limit_bytes: int, max_source_bytes: int
    ) -> None:
        """Configure the session cache before the first public mask is used."""

        current = self.plaintext_cache.info()
        if int(current["entries"]) or int(current["misses"]):
            raise RuntimeError(
                "plaintext cache policy must be fixed before first use"
            )
        self.plaintext_cache = SessionPlaintextCache(
            limit_bytes=int(limit_bytes),
            max_source_bytes=int(max_source_bytes),
        )

    @property
    def device(self) -> str:
        return str(self.context.device)

    def level_for_cipher(self, cipher) -> int:
        return int(self.context.L) - int(cipher.state.cur_limbs)

    def encrypt(self, values: np.ndarray, *, level: int, slots: int | None = None):
        cur_limbs = int(self.context.L) - int(level)
        return self.client.encrypt(
            np.asarray(values).reshape(-1),
            device=self.device,
            slots=self.max_slots if slots is None else int(slots),
            cur_limbs=cur_limbs,
            scaling_factor=self.context.scale_at(cur_limbs),
        )

    def plaintext(
        self,
        values: np.ndarray,
        *,
        name: str,
        level: int,
        slots: int | None = None,
        dtype: np.dtype = np.float64,
        is_ext: bool = False,
    ):
        array = np.asarray(values, dtype=dtype).reshape(-1)
        bundle = self.constant_bundle(
            {"value": array},
            cache_key=("vector", _normalized_constant_name(str(name))),
        )
        return bundle.plaintext(
            "value",
            state=fhe.CipherState(
                cur_limbs=int(self.context.L) - int(level),
                scale_degree=1,
                scaling_factor=self.context.scale_at(
                    int(self.context.L) - int(level)
                ),
            ),
            slots=self.max_slots if slots is None else int(slots),
            context=self.context,
            is_ext=bool(is_ext),
        )

    def constant_bundle(
        self,
        vectors: dict[str, np.ndarray],
        *,
        cache_key: object,
    ):
        """Return a session-owned plain-caching bundle for immutable arrays."""

        arrays = {
            str(name): np.ascontiguousarray(values)
            for name, values in vectors.items()
        }
        source_bytes = sum(int(values.nbytes) for values in arrays.values())

        def factory(cache: bool):
            wrapped = {}
            for name, values in arrays.items():
                tensor = self.tensor.from_numpy(values)
                if hasattr(tensor, "to"):
                    tensor = tensor.to(self.device)
                wrapped[name] = self.fhe.PackedRaw(tensor)
            return self.fhe.ConstantBundle(
                vectors=wrapped,
                cache_mode="plain" if bool(cache) else "none",
            )

        if not self.plaintext_cache.accepts_source(source_bytes):
            if _TRACE_CACHE:
                _trace_cache("transient", cache_key, source_bytes, self.plaintext_cache)
            return self.plaintext_cache.transient(
                factory, source_bytes=source_bytes
            )
        content = tuple(
            (name, _array_cache_signature(values))
            for name, values in sorted(arrays.items())
        )
        key = (_normalized_constant_name(cache_key), content)
        if _VERIFY_SIGNATURE:
            full = tuple(
                (name, _full_array_digest(values))
                for name, values in sorted(arrays.items())
            )
            self.plaintext_cache.verify(key, full)
        if _TRACE_CACHE:
            hit = key in self.plaintext_cache._entries
            _trace_cache("hit" if hit else "miss", cache_key, source_bytes, self.plaintext_cache)
        return self.plaintext_cache.get(
            key, factory, source_bytes=source_bytes
        )

    def plaintext_cache_info(self) -> dict[str, int]:
        return self.plaintext_cache.info()

    def clear_plaintext_cache(self) -> None:
        self.plaintext_cache.clear()

    def decrypt(self, cipher) -> np.ndarray:
        value = self.client.decrypt(cipher)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value).reshape(-1)


def plaintext_cache_snapshot(context) -> dict[str, int]:
    """Session plaintext-cache counters, zeros when the context has no cache."""

    inspect = getattr(context, "plaintext_cache_info", None)
    if callable(inspect):
        return {name: int(value) for name, value in inspect().items()}
    return {
        name: 0
        for name in (
            "entries",
            "bytes",
            "limit_bytes",
            "max_source_bytes",
            "hits",
            "misses",
            "skips",
        )
    }


def release_if_supported(value) -> None:
    """Release explicit handles on backends that expose them.

    Current source EasyFHE tensors use Python ownership, while older generated
    objects may expose ``release``.  This is resource cleanup, not an FHE-op
    compatibility layer.
    """

    release = getattr(value, "release", None)
    if callable(release):
        release()
        return
    constants = getattr(value, "constants", None)
    clear_cache = getattr(constants, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def release_all(values) -> None:
    """Release every owned backend object in ``values``."""

    for value in values:
        release_if_supported(value)


def synchronize_device(device: str) -> None:
    """Synchronize the public EasyFHE device module selected by ``device``."""

    easyfhe.get_device_module(str(device)).synchronize()


# ``T2_STAGE_SYNC`` (default 1): whether the operators' *stage-timer* device
# synchronisations run.  Nearly every operator brackets its sub-stages with a
# full device sync so that ``stage_seconds[...] = perf_counter() - start`` is a
# device time; the bootstrap operator and the T2 bench's ``_BootstrapTimer``
# add two more around every refresh.  Those syncs exist only for the profile:
# the stations run on one stream, so stream order already serialises the
# work.  Each sync drains the device and then leaves it idle while the host
# re-enqueues the next kernels (Python-heavy launch code), roughly 60-100
# syncs per layer on 8 x Pro 6000.  With ``T2_STAGE_SYNC=0`` the
# :func:`profile_sync_device` sites turn into no-ops, the sub-stage timings
# become host enqueue times (station walls stay exact: the station wrapper
# still synchronises its stream), and the sync points that carry semantics
# -- the pinned transport, the weight upload in ``LinearWeight.prepare``, the
# key-eviction barriers -- keep calling :func:`synchronize_device`.
_STAGE_SYNC: bool | None = None


def stage_sync_enabled() -> bool:
    """True unless ``T2_STAGE_SYNC=0`` (cached after the first call)."""

    global _STAGE_SYNC
    if _STAGE_SYNC is None:
        _STAGE_SYNC = os.environ.get("T2_STAGE_SYNC", "1") != "0"
    return bool(_STAGE_SYNC)


def set_stage_sync(enabled: bool | None) -> None:
    """Override (or with ``None`` re-read ``T2_STAGE_SYNC``) the stage-sync switch."""

    global _STAGE_SYNC
    _STAGE_SYNC = None if enabled is None else bool(enabled)


def profile_sync_device(device: str) -> None:
    """A device synchronisation that only delimits a stage timer; no-op under ``T2_STAGE_SYNC=0``."""

    if stage_sync_enabled():
        synchronize_device(device)


def trim_device_allocator_cache(device: str) -> None:
    """Return dead allocator blocks without evicting live EasyFHE caches.



    Set ``LLAMA_FHE_ALLOCATOR_TRIM=1`` to restore the trims (for
    non-expandable allocators or tighter-memory devices where carrying one
    operator's reserve into the next matters more than the stalls).
    
    """

    if os.environ.get("LLAMA_FHE_ALLOCATOR_TRIM", "") != "1":
        return
    if not str(device).startswith("cuda"):
        return
    module = easyfhe.get_device_module(str(device))
    module.synchronize()
    module.empty_cache()


def cuda_memory_snapshot(device: str) -> dict[str, int | bool]:
    """Return allocator and whole-device memory for an optional CUDA profile."""

    if not str(device).startswith("cuda"):
        return {"available": False}
    try:
        import torch

        selected = torch.device(str(device))
        free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
        return {
            "available": True,
            "allocator_allocated_bytes": int(
                torch.cuda.memory_allocated(selected)
            ),
            "allocator_reserved_bytes": int(
                torch.cuda.memory_reserved(selected)
            ),
            "allocator_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(selected)
            ),
            "allocator_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(selected)
            ),
            "device_used_bytes": int(total_bytes - free_bytes),
            "device_free_bytes": int(free_bytes),
            "device_total_bytes": int(total_bytes),
        }
    except (ImportError, RuntimeError):
        return {"available": False}


def _default_rotations(input_dim: int, block_out: int) -> set[int]:
    input_dim = int(input_dim)
    block_out = int(block_out)
    if input_dim <= 0 or input_dim & (input_dim - 1):
        raise ValueError("input_dim must be a positive power of two.")
    if block_out <= 0 or block_out & (block_out - 1):
        raise ValueError("block_out must be a positive power of two.")
    rotations = {
        block_out * (1 << index) for index in range(int(log2(input_dim)))
    }
    rotations.update(1 << index for index in range(int(log2(block_out))))
    return rotations


def create_sim_context(
    sim_cfg: SimulatorConfig,
    *,
    input_dim: int,
    block_out: int,
    rotations: tuple[int, ...] | list[int] | None = None,
    include_default_rotations: bool = True,
):
    """Build an uncached context for isolated tests and benchmarks."""

    requested = (
        _default_rotations(input_dim, block_out)
        if bool(include_default_rotations)
        else set()
    )
    requested.update(int(value) for value in rotations or () if int(value))
    normalized = tuple(sorted(requested))
    start = time.perf_counter()
    client, context = fhe.generate_client_context(
        fhe.CKKSContextSpec(
            depth=int(sim_cfg.maxLevelsRemaining),
            log_n=int(sim_cfg.logN),
            dnum=int(sim_cfg.dnum),
            dcrt_bits=int(sim_cfg.dcrtBits),
            first_mod=int(sim_cfg.firstMod),
            secret_key_dist=str(sim_cfg.secretKeyDist),
            scale_mode="fixed",
            rescale_policy=sim_cfg.rescale_policy(),
            rotations=normalized,
            auto_load_keys=True,
        ),
        device=str(sim_cfg.device),
    )
    return EasyFHEContext(
        tensor=easyfhe,
        fhe=fhe,
        client=client,
        context=context,
        max_slots=int(context.max_slots),
        keygen_wall_time_s=float(time.perf_counter() - start),
    )


__all__ = [
    "EasyFHEContext",
    "plaintext_cache_snapshot",
    "create_sim_context",
    "cuda_memory_snapshot",
    "release_all",
    "release_if_supported",
    "synchronize_device",
    "profile_sync_device",
    "set_stage_sync",
    "stage_sync_enabled",
    "trim_device_allocator_cache",
]
