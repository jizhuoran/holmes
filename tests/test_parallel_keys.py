from __future__ import annotations

import pytest

from llama3fhe.parallel.prefill.keys import (
    LEASE_POLICY,
    RESIDENT_POLICY,
    TIER_ORDER,
    KeyLifecycle,
    KeyPlan,
    KeyTier,
    KeyTracer,
    normalize_rotation,
    rotation_key_bytes,
)


class _Component:
    """Stand-in for a CUDA uint64 tensor: only ``numel``/``element_size`` are read."""

    def __init__(self, beta: int, limbs: int, ring_dim: int):
        self._numel = beta * limbs * ring_dim

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return 8


class FakeNative:
    """Mimics the EasyFHE ``Context`` key cache closely enough for the lifecycle."""

    def __init__(self, rotations, *, ring_dim=64, L=10, K=2, alpha=3):
        self.N = ring_dim
        self.L = L
        self.K = K
        self.alpha = alpha
        self.device = "cuda"
        self.auto_load_keys_resolved = True
        self.left_rot_key_map = {int(rot): ("bx", "ax") for rot in rotations}
        self._rotation_key_cuda_cache = {}
        self._precompute_auto_cuda_cache = {}
        self._inverse_precompute_auto_cuda_cache = {}
        self.loads = []
        self.clears = []

    def get_rotation_key_for_limbs(self, rot_index, cur_limbs):
        rot_index = int(rot_index)
        cur_limbs = int(cur_limbs)
        if rot_index not in self.left_rot_key_map:
            raise KeyError(rot_index)
        beta = (cur_limbs + self.alpha - 1) // self.alpha
        for (rot, start, cached_beta), value in self._rotation_key_cuda_cache.items():
            if rot == rot_index and start >= cur_limbs and cached_beta >= beta:
                return value, start
        key = (rot_index, cur_limbs, beta)
        value = self._load_rotation_key_to_cuda(rot_index, cur_limbs, beta, self.L)
        self._rotation_key_cuda_cache[key] = value
        return value, cur_limbs

    def _load_rotation_key_to_cuda(self, rot_index, special_mod_start, beta, available):
        self.loads.append((int(rot_index), int(special_mod_start), int(beta)))
        limbs = int(special_mod_start) + self.K
        return [_Component(beta, limbs, self.N), _Component(beta, limbs, self.N)]

    def _rotation_key_special_mod_start(self, rot_index):
        return self.L

    def _evict_dominated_rotation_key_versions(self, retained_key):
        rot, limbs, beta = retained_key
        for key in tuple(self._rotation_key_cuda_cache):
            if key != retained_key and key[0] == rot and key[1] <= limbs and key[2] <= beta:
                del self._rotation_key_cuda_cache[key]

    def clear_cuda_rotation_cache(self, *, keep_rotations=(), empty_allocator_cache=True):
        keep = {normalize_rotation(v, ring_dim=self.N) for v in keep_rotations}
        before = len(self._rotation_key_cuda_cache)
        for key in tuple(self._rotation_key_cuda_cache):
            if key[0] not in keep:
                del self._rotation_key_cuda_cache[key]
        self.clears.append(tuple(sorted(keep)))
        return {
            "_rotation_key_cuda_cache_entries": before,
            "_rotation_key_cuda_cache_retained_entries": len(self._rotation_key_cuda_cache),
        }

    def cached_rotations(self):
        return sorted({key[0] for key in self._rotation_key_cuda_cache})


def _plan(policy=LEASE_POLICY, resident_extra=(), caps=None):
    caps = caps or {}
    tiers = (
        KeyTier("bootstrap", 0, (1, 2, 4), caps.get("bootstrap")),
        KeyTier("rmsnorm", 0, (2, 8, 31), caps.get("rmsnorm")),  # 31 == normalize(-1)
        KeyTier("linear", 1, (4, 16, 17), caps.get("linear")),
        KeyTier("attn_input", 2, (16, 20, 21, 22), caps.get("attn_input")),
        KeyTier("attn_core", 2, (21, 30), caps.get("attn_core")),
    )
    return KeyPlan(tiers=tiers, policy=policy, ring_dim=64, full_limbs=10, resident_extra=tuple(resident_extra))


# ------------------------------------------------------------------ plan
def test_normalize_rotation_matches_easyfhe_convention() -> None:
    assert normalize_rotation(5, ring_dim=64) == 5
    assert normalize_rotation(-1, ring_dim=64) == 31
    assert normalize_rotation(-1, ring_dim=1 << 16) == (1 << 15) - 1


def test_rotation_key_bytes_scales_with_limbs_and_digits() -> None:
    # dnum5 on L=38 -> alpha 8, K 8, N 2^16: full key ~0.21 GiB, 18 limbs ~0.10, 6 limbs ~0.013.
    full = rotation_key_bytes(ring_dim=1 << 16, limbs=38, alpha=8, special_limbs=8)
    mid = rotation_key_bytes(ring_dim=1 << 16, limbs=18, alpha=8, special_limbs=8)
    low = rotation_key_bytes(ring_dim=1 << 16, limbs=6, alpha=8, special_limbs=8)
    assert full == 2 * 8 * 5 * 46 * (1 << 16)
    assert mid == 2 * 8 * 3 * 26 * (1 << 16)
    assert low == 2 * 8 * 1 * 14 * (1 << 16)
    assert full > mid > low
    assert rotation_key_bytes(ring_dim=1 << 16, limbs=0, alpha=8, special_limbs=8) == 0


def test_tier_normalises_and_validates() -> None:
    tier = KeyTier("linear", 1, (5, 3, 3, 0, 9))
    assert tier.rotations == (3, 5, 9)
    assert len(tier) == 3
    with pytest.raises(ValueError):
        KeyTier("unknown", 1, (1,))
    with pytest.raises(ValueError):
        KeyTier("linear", 3, (1,))
    with pytest.raises(ValueError):
        KeyTier("linear", 1, (1,), limb_cap=0)


def test_plan_orders_tiers_and_reports_exclusive_rotations() -> None:
    plan = _plan()
    assert plan.names == tuple(name for name in TIER_ORDER if plan.has_tier(name))
    assert plan.union == (1, 2, 4, 8, 16, 17, 20, 21, 22, 30, 31)
    assert plan.exclusive_rotations("rmsnorm") == (8, 31)
    assert plan.exclusive_rotations("linear") == (16, 17)
    assert plan.exclusive_rotations("attn_core") == (30,)
    with pytest.raises(KeyError):
        plan.tier("softmax")
    with pytest.raises(ValueError):
        KeyPlan(tiers=(KeyTier("linear", 1, (1,)), KeyTier("linear", 1, (2,))))
    with pytest.raises(ValueError):
        KeyPlan(tiers=(), policy="sometimes")


def test_residency_follows_policy_and_resident_extra() -> None:
    lease = _plan()
    assert lease.resident_names == ("bootstrap", "rmsnorm")
    assert lease.resident_rotations == frozenset({1, 2, 4, 8, 31})
    promoted = _plan(resident_extra=("linear",))
    assert promoted.resident_names == ("bootstrap", "rmsnorm", "linear")
    resident = _plan(policy=RESIDENT_POLICY)
    assert resident.resident_names == resident.names
    assert resident.resident_rotations == frozenset(resident.union)


def test_limb_limits_take_the_largest_cap_and_skip_uncapped_rotations() -> None:
    plan = _plan(caps={"linear": 18, "attn_input": 6, "attn_core": 9})
    limits = plan.limb_limits()
    # 4 is shared with the uncapped bootstrap tier -> no limit; 16 shared linear/attn_input -> 18.
    assert 4 not in limits
    assert limits[16] == 18 and limits[17] == 18
    assert limits[20] == 6 and limits[22] == 6
    assert limits[21] == 9  # attn_input 6 vs attn_core 9 -> larger
    assert limits[30] == 9
    summary = plan.summary(alpha=3, special_limbs=2)
    by_name = {row["tier"]: row for row in summary["tiers"]}
    assert by_name["linear"]["limb_cap"] == 18 and by_name["linear"]["resident"] is False
    assert by_name["bootstrap"]["resident"] is True
    assert summary["capped_keys"] == len(limits)
    assert by_name["attn_input"]["est_gib"] < by_name["bootstrap"]["est_gib"]


# ------------------------------------------------------------- lifecycle
def test_lease_prefetches_then_evicts_everything_but_resident_and_active() -> None:
    plan = _plan(caps={"attn_input": 6})
    native = FakeNative(plan.union)
    life = KeyLifecycle(native, plan, synchronize=lambda: None)
    # T0 gets loaded by "the bootstrap" at full limbs.
    for rot in (1, 2, 4, 8, 31):
        native.get_rotation_key_for_limbs(rot, 10)
    with life.lease("attn_input") as record:
        assert record.prefetched_keys == 4
        assert set(life.active_tiers) == {"attn_input"}
        assert (16, 6, 2) in native.loads  # cap 6 -> beta ceil(6/3) = 2
        assert set(native.cached_rotations()) >= {16, 20, 21, 22}
        with life.lease("attn_core") as inner:
            assert inner.prefetched_keys == 2
            assert life.active_tiers == ("attn_input", "attn_core")
        # inner exit evicts, but keeps the outer lease alive
        assert native.clears[-1] == tuple(sorted({1, 2, 4, 8, 31, 16, 20, 21, 22}))
        assert set(native.cached_rotations()) == {1, 2, 4, 8, 31, 16, 20, 21, 22}
    assert native.cached_rotations() == [1, 2, 4, 8, 31]
    assert life.active_tiers == ()
    assert [r.tier for r in life.records] == ["attn_core", "attn_input"]
    # cache entries are (rot, limbs, beta): 16/20/21/22 from the cap-6 prefetch plus
    # the full-limb (21, 10, 4) the inner lease added (the fake keeps dominated
    # versions; the real context would have replaced (21, 6, 2)) -> 5 evicted, 5 kept.
    assert life.records[-1].evicted_entries == 5
    assert life.records[-1].retained_entries == 5


def test_resident_policy_never_evicts_and_resident_tier_lease_is_cheap() -> None:
    plan = _plan(policy=RESIDENT_POLICY)
    native = FakeNative(plan.union)
    life = KeyLifecycle(native, plan, synchronize=lambda: None)
    with life.lease("attn_input"):
        pass
    assert native.clears == []
    assert set(native.cached_rotations()) == {16, 20, 21, 22}
    # lease policy but a resident tier: prefetch happens, no eviction.
    lease_plan = _plan()
    native2 = FakeNative(lease_plan.union)
    life2 = KeyLifecycle(native2, lease_plan, synchronize=lambda: None)
    with life2.lease("rmsnorm"):
        pass
    assert native2.clears == []
    assert native2.cached_rotations() == [2, 8, 31]


def test_evict_keep_extra_and_explicit_prefetch_limbs() -> None:
    plan = _plan()
    native = FakeNative(plan.union)
    life = KeyLifecycle(native, plan, synchronize=lambda: None)
    count, _ = life.prefetch("linear", limbs=5)
    assert count == 3
    assert (17, 5, 2) in native.loads
    life.prefetch("attn_core")
    stats = life.evict(keep_extra=(30,))
    # 4 is resident (bootstrap), 30 is kept explicitly; 16, 17, 21 go.
    assert stats["_rotation_key_cuda_cache_retained_entries"] == 2
    assert native.cached_rotations() == [4, 30]


# ---------------------------------------------------------------- tracer
def test_tracer_counts_requests_misses_bytes_per_stage_and_restores_native() -> None:
    plan = _plan()
    native = FakeNative(plan.union)
    original_get = native.get_rotation_key_for_limbs
    tracer = KeyTracer(native, synchronize=lambda: None)
    life = KeyLifecycle(native, plan, tracer=tracer, synchronize=lambda: None)
    with tracer:
        with tracer.stage("rmsnorm_in"):
            native.get_rotation_key_for_limbs(2, 4)
            native.get_rotation_key_for_limbs(2, 4)  # hit
            native.get_rotation_key_for_limbs(8, 7)
        with tracer.stage("attn"), life.lease("attn_input"):
            native.get_rotation_key_for_limbs(20, 3)  # prefetched at full limbs -> hit
        native.get_rotation_key_for_limbs(30, 2)
    rows = {row["stage"]: row for row in tracer.rows()}
    rms = rows["rmsnorm_in"]
    assert rms["requests"] == 3 and rms["misses"] == 2 and rms["hits"] == 1 and rms["loads"] == 2
    assert rms["distinct_rotations"] == 2 and rms["max_limbs"] == 7
    assert rms["bytes_loaded"] == 2 * 8 * (2 * (4 + 2) * 64 + 3 * (7 + 2) * 64)
    attn = rows["attn"]
    assert attn["prefetch_keys"] == 4 and attn["misses"] == 4 and attn["requests"] == 5 and attn["hits"] == 1
    assert attn["evictions"] == 1
    assert rows[KeyTracer.OUTSIDE]["requests"] == 1
    cov = tracer.coverage("rmsnorm_in", plan, "rmsnorm")
    assert cov["planned"] == 3 and cov["used"] == 2 and cov["used_in_plan"] == 2 and cov["unused_planned"] == 1
    # the wrappers are gone after the context exits
    assert "get_rotation_key_for_limbs" not in vars(native)
    assert native.get_rotation_key_for_limbs.__func__ is original_get.__func__
    tracer.uninstall()  # idempotent


# ------------------------------------------------------- async prefetch
class _FakeStream:
    synchronized = []

    def __init__(self, name):
        self.name = name
        self.waited_on = []

    def wait_stream(self, other):
        self.waited_on.append(other.name)

    def synchronize(self):
        _FakeStream.synchronized.append(self.name)


class _FakeEvent:
    recorded_on = None
    waited_by = None
    synchronized = 0

    def record(self, stream):
        self.recorded_on = stream.name

    def wait(self, stream):
        self.waited_by = stream.name

    def synchronize(self):
        self.synchronized += 1


class _StreamCtx:
    current = "default"

    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        self.previous = _StreamCtx.current
        _StreamCtx.current = self.stream.name

    def __exit__(self, *exc):
        _StreamCtx.current = self.previous


def _fake_torch(free_bytes: int):
    import types

    cuda = types.SimpleNamespace(
        Stream=lambda device=None: _FakeStream("side"),
        default_stream=lambda device=None: _FakeStream("default"),
        current_stream=lambda device=None: _FakeStream(_StreamCtx.current),
        stream=_StreamCtx,
        Event=_FakeEvent,
        mem_get_info=lambda device=None: (free_bytes, free_bytes),
        memory_reserved=lambda device=None: 0,
        memory_allocated=lambda device=None: 0,
        current_device=lambda: 0,
    )
    return types.SimpleNamespace(cuda=cuda, device=lambda *a, **k: "cuda:0")


def _chain_plan():
    tiers = (
        KeyTier("bootstrap", 0, (1, 2), None),
        KeyTier("linear", 1, (4, 16, 17), 5),
        KeyTier("attn_input_lane", 2, (17, 20), 4),
        KeyTier("attn_input_v", 2, (20, 21, 22), 4),
    )
    return KeyPlan(tiers=tiers, policy=LEASE_POLICY, ring_dim=64, full_limbs=10, resident_extra=("linear",))


def test_async_prefetch_learns_limbs_then_uploads_next_tier_on_the_side_stream(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free_bytes=1 << 40))
    plan = _chain_plan()
    native = FakeNative(plan.union)
    tracer = KeyTracer(native, synchronize=lambda: None).install()
    life = KeyLifecycle(native, plan, tracer=tracer, synchronize=lambda: None, prefetch_mode="async", device="cuda:0")
    life.install_profile_hook()

    # pass 1: nothing is known -> lazy loads, but the profile learns per-tier limbs.
    with life.lease("linear") as lin:
        native.get_rotation_key_for_limbs(16, 5)
        native.get_rotation_key_for_limbs(17, 5)
    assert lin.async_issued_tier == "attn_input_lane" and lin.async_skipped == "no_profile"
    with life.lease("attn_input_lane"):
        native.get_rotation_key_for_limbs(17, 3)
        native.get_rotation_key_for_limbs(20, 4)
    with life.lease("attn_input_v"):
        native.get_rotation_key_for_limbs(20, 4)
        native.get_rotation_key_for_limbs(21, 2)
    assert life.profile == {"attn_input_lane": {17: 3, 20: 4}, "attn_input_v": {20: 4, 21: 2}}
    assert native.cached_rotations() == [16, 17]  # linear is resident, the rest was evicted

    # pass 2: entering linear issues the lane tier; 17 is busy (active linear) -> only 20 goes.
    loads_before = len(native.loads)
    with life.lease("linear") as lin:
        assert lin.async_issued_tier == "attn_input_lane" and lin.async_issued_keys == 1
        assert "attn_input_lane" in life.pending and _StreamCtx.current == "default"
        pending = life.pending["attn_input_lane"]
        assert pending.limbs == {20: 4} and pending.event.recorded_on == "side"
        assert (20, 4, 2) in native.loads[loads_before:]
        assert set(native.cached_rotations()) == {16, 17, 20}
        assert life.keep_set() >= {1, 2, 4, 16, 17, 20}
    # the lane keys survived linear's exit (pending tiers are kept)
    assert 20 in native.cached_rotations()
    with life.lease("attn_input_lane") as lane:
        assert lane.async_keys == 1 and lane.async_wait_seconds >= 0.0
        # The dependency is inserted on the compute stream, but entering the
        # lease must not host-block on the side-stream event.  A teardown
        # drain remains responsible for waiting on copies that are still in
        # flight.
        assert pending.event.waited_by == "default" and pending.event.synchronized == 0
        assert "attn_input_lane" not in life.pending
        # v was issued: 20 is busy (lane active) -> only 21 at its learnt 2 limbs
        assert lane.async_issued_tier == "attn_input_v" and lane.async_issued_keys == 1
        assert life.pending["attn_input_v"].limbs == {21: 2}
        native.get_rotation_key_for_limbs(20, 4)  # cache hit on the prefetched key
    rows = {row["stage"]: row for row in tracer.rows()}
    assert rows[KeyTracer.OUTSIDE]["prefetch_keys"] == 2 and rows[KeyTracer.OUTSIDE]["prefetch_gib"] > 0
    life.drain()
    assert life.pending == {}
    issued = [e for e in life.prefetch_log if e.get("issued") is True]
    assert [e["tier"] for e in issued] == ["attn_input_lane", "attn_input_v"]
    tracer.uninstall()


def test_async_prefetch_respects_the_memory_budget(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free_bytes=1 << 20))
    plan = _chain_plan()
    native = FakeNative(plan.union)
    life = KeyLifecycle(native, plan, synchronize=lambda: None, prefetch_mode="async", prefetch_margin_gib=1.0)
    life.install_profile_hook()
    with life.lease("attn_input_lane"):
        native.get_rotation_key_for_limbs(20, 4)
    with life.lease("linear") as lin:
        assert lin.async_issued_tier == "attn_input_lane"
        assert lin.async_skipped is not None and lin.async_skipped.startswith("budget")
        assert life.pending == {}
    assert life.prefetch_log[-1]["issued"] is False


def test_prefetch_mode_is_validated() -> None:
    with pytest.raises(ValueError):
        KeyLifecycle(FakeNative(()), _plan(), prefetch_mode="eager")


def test_stream_ordered_evict_matches_easyfhe_semantics_without_a_device_sync(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free_bytes=1 << 40))
    _FakeStream.synchronized.clear()
    plan = _plan(caps={"attn_input": 6})
    native = FakeNative(plan.union)
    native.precompute_auto_maps_cache = {((16, 20), "x"): 1, ((2,), "y"): 2}
    life = KeyLifecycle(native, plan, synchronize=lambda: None, evict_sync="stream", device="cuda:0")
    assert life.evict_empty_cache is False
    for rot in (1, 2, 4, 8, 31):
        native.get_rotation_key_for_limbs(rot, 10)
    with life.lease("attn_input") as record:
        assert set(native.cached_rotations()) >= {16, 20, 21, 22}
    # EasyFHE's method was not used (FakeNative.clears untouched) ...
    assert native.clears == []
    # ... but the outcome is the same: resident kept, leased tier dropped, counters filled.
    assert native.cached_rotations() == [1, 2, 4, 8, 31]
    assert record.evicted_entries == 4 and record.retained_entries == 5
    assert native.precompute_auto_maps_cache == {((2,), "y"): 2}
    assert _FakeStream.synchronized == ["default"]
    with pytest.raises(ValueError):
        KeyLifecycle(native, plan, evict_sync="never")
