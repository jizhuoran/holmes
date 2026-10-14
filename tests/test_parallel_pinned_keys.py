"""CPU checks for :class:`llama3fhe.parallel.prefill.keys.PinnedCompactKeys`.

The pinned compact of a rotation must equal EasyFHE's per-load
``cat(q_part[:cap], p_part)`` for every ``(limbs, beta)`` request at or under
the tier cap, the hook must fall through for rotations it does not hold, and
the tracer installed on top must still see the loads.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from llama3fhe.parallel.prefill.keys import (  # noqa: E402
    LEASE_POLICY,
    KeyLifecycle,
    KeyPlan,
    KeyTier,
    KeyTracer,
    PinnedCompactKeys,
)

L, K, ALPHA, DNUM, N = 10, 2, 3, 4, 8


class TorchNative:
    """EasyFHE ``Context`` key path on CPU tensors (the real slicing arithmetic)."""

    def __init__(self, rotations, limb_limits=None):
        self.L, self.K, self.alpha, self.N = L, K, ALPHA, N
        self.device = "cuda"
        self.auto_load_keys_resolved = True
        self.rotation_key_limb_limits = dict(limb_limits or {})
        rng = np.random.default_rng(7)

        def material():
            return torch.from_numpy(rng.integers(0, 2**40, size=(DNUM, L + K, N), dtype=np.int64))

        self.left_rot_key_map = {int(rot): (material(), material()) for rot in rotations}
        self._rotation_key_cuda_cache = {}
        self.loads = []

    def reference(self, rot, limbs):
        beta = (limbs + self.alpha - 1) // self.alpha
        available = self._rotation_key_special_mod_start(rot)
        return [self._compact(c, limbs, beta, available) for c in self.left_rot_key_map[rot]]

    def _rotation_key_special_mod_start(self, rot):
        bx, _ = self.left_rot_key_map[rot]
        return min(int(self.rotation_key_limb_limits.get(rot, self.L)), int(bx.shape[1]) - self.K)

    def _compact(self, component, special_mod_start, beta, available):
        component = component[:beta]
        q_part = component[:, :special_mod_start, :]
        p_part = component[:, available:available + self.K, :]
        return torch.cat((q_part, p_part), dim=1).contiguous()

    def get_rotation_key_for_limbs(self, rot, limbs):
        rot, limbs = int(rot), int(limbs)
        available = self._rotation_key_special_mod_start(rot)
        beta = (limbs + self.alpha - 1) // self.alpha
        for (cached_rot, start, cached_beta), value in self._rotation_key_cuda_cache.items():
            if cached_rot == rot and start >= limbs and cached_beta >= beta:
                return value, start
        value = self._load_rotation_key_to_cuda(rot, limbs, beta, available)
        self._rotation_key_cuda_cache[(rot, limbs, beta)] = value
        return value, limbs

    def _load_rotation_key_to_cuda(self, rot, special_mod_start, beta, available):
        self.loads.append((int(rot), int(special_mod_start), int(beta)))
        bx, ax = self.left_rot_key_map[rot]
        return [self._compact(bx, special_mod_start, beta, available), self._compact(ax, special_mod_start, beta, available)]


def _plan():
    tiers = (
        KeyTier("bootstrap", 0, (1, 2), None),
        KeyTier("linear", 1, (3, 4), 6),
        KeyTier("attn_input_q", 2, (4, 5), 5),
    )
    return KeyPlan(tiers=tiers, policy=LEASE_POLICY, ring_dim=2 * N, full_limbs=L)


def _cache(native, plan):
    return PinnedCompactKeys(native, plan, device="cpu", pin_memory=False)


def test_default_tiers_are_the_leased_capped_ones_and_caps_take_the_max() -> None:
    plan = _plan()
    native = TorchNative((1, 2, 3, 4, 5), plan.limb_limits())
    cache = _cache(native, plan)
    assert cache.default_tiers() == ("linear", "attn_input_q")
    report = cache.prepare()
    assert report["keys"] == 3 and set(cache.entries) == {3, 4, 5}
    # rotation 4 is in linear (cap 6) and attn_input_q (cap 5) -> pinned at 6, beta 2
    assert cache.entries[4][2] == 6 and cache.entries[4][3] == 2
    assert cache.entries[5][2] == 5 and cache.entries[5][3] == 2
    assert cache.entries[3][0].shape == (2, 6 + K, N)
    assert cache.gib > 0 and cache.nbytes == sum(
        bx.numel() * 8 + ax.numel() * 8 for bx, ax, _, _ in cache.entries.values()
    )


@pytest.mark.parametrize("limbs", [1, 2, 3, 4, 5, 6])
def test_pinned_load_matches_easyfhe_compact_for_every_request_under_the_cap(limbs) -> None:
    plan = _plan()
    native = TorchNative((1, 2, 3, 4, 5), plan.limb_limits())
    cache = _cache(native, plan)
    cache.prepare()
    cache.install()
    value, start = native.get_rotation_key_for_limbs(4, limbs)
    assert start == limbs
    expected = native.reference(4, limbs)
    for got, ref in zip(value, expected, strict=True):
        assert tuple(got.shape) == tuple(ref.shape)
        assert np.array_equal(got.numpy(), ref.numpy())
    assert cache.hits == 1 and cache.misses == 0
    assert native.loads == []  # served without the original loader


def test_hook_falls_through_for_unpinned_rotations_and_over_cap_requests() -> None:
    plan = _plan()
    native = TorchNative((1, 2, 3, 4, 5), plan.limb_limits())
    cache = _cache(native, plan)
    cache.prepare()
    cache.install()
    # resident rotation 1 is not pinned -> original loader
    native.get_rotation_key_for_limbs(1, 9)
    assert native.loads == [(1, 9, 3)] and cache.misses == 1
    # pinned at cap 5 but asked for 6 (would be clipped by limb_limits in EasyFHE; here we force it)
    native.rotation_key_limb_limits[5] = 6
    native.get_rotation_key_for_limbs(5, 6)
    assert native.loads[-1] == (5, 6, 2) and cache.misses == 2
    cache.uninstall()
    native.get_rotation_key_for_limbs(3, 3)
    assert native.loads[-1] == (3, 3, 1) and cache.hits == 0


def test_lifecycle_installs_before_tracer_and_tracer_still_books_the_load() -> None:
    plan = _plan()
    native = TorchNative((1, 2, 3, 4, 5), plan.limb_limits())
    life = KeyLifecycle(native, plan, synchronize=lambda: None)
    life.pinned = _cache(native, plan)
    report = life.pin_leased_keys()
    assert report["keys"] == 3
    tracer = KeyTracer(native, synchronize=lambda: None).install()
    life.tracer = tracer
    with tracer.stage("q"):
        native.get_rotation_key_for_limbs(5, 5)
        native.get_rotation_key_for_limbs(5, 4)  # dominated -> cache hit
    row = tracer.stages["q"]
    assert row.requests == 2 and row.misses == 1 and row.loads == 1
    assert row.bytes_loaded == 2 * 2 * (5 + K) * N * 8
    assert life.pinned.hits == 1 and native.loads == []
    life.release()
    assert life.pinned.entries == {}
    tracer.uninstall()
    native.get_rotation_key_for_limbs(3, 3)
    assert native.loads == [(3, 3, 1)]
