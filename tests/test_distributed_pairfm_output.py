from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.operators.attention.distributed.distributed_output import (
    PairFMSourcePartialResult,
    build_pairfm_reduce_scatter_plan,
    distributed_pairfm_carriers_numpy,
    distributed_pairfm_complexity,
    pairfm_rank_partials_fhe,
    pairfm_source_partial_carriers_numpy,
    pairfm_target_reduce_fhe,
)
from llama3fhe.operators.attention.prefill.output import (
    output_c_to_feature_major_pairfm_numpy,
)


def _layouts():
    return (
        FeatureMajorPrefillLayout(
            seq_len=128, hidden_dim=4096, slots=32768
        ),
        AttentionPairLayout(
            seq_len=128,
            head_dim=128,
            query_heads=32,
            key_value_heads=8,
            slots=32768,
        ),
    )


def test_eight_device_plan_is_all_to_all_without_owner_gather():
    feature, attention = _layouts()
    plan = build_pairfm_reduce_scatter_plan(feature, attention, range(8))

    assert plan.query_pair_count == 16
    assert plan.source_shards == plan.target_carriers == 8
    assert plan.remote_transfer_count == 56
    assert plan.local_contribution_count == 8
    assert [row.query_pair_indices for row in plan.placements] == [
        (2 * rank, 2 * rank + 1) for rank in range(8)
    ]
    assert [row.target_token_shards for row in plan.placements] == [
        (2 * rank, 2 * rank + 1) for rank in range(8)
    ]
    for target in range(8):
        incoming = [
            edge
            for edge in plan.transport_edges
            if edge["target_carrier_index"] == target
        ]
        assert len(incoming) == 8
        assert sum(bool(edge["remote"]) for edge in incoming) == 7


def test_single_device_plan_is_matched_sharded_denominator():
    feature, attention = _layouts()
    plan = build_pairfm_reduce_scatter_plan(feature, attention, [3] * 8)

    assert plan.unique_devices == (3,)
    assert plan.remote_transfer_count == 0
    assert plan.local_contribution_count == 64
    # Logical source and target work is unchanged even when transport vanishes.
    assert len(plan.transport_edges) == 64


def test_source_partials_reduce_to_official_single_pairfm_oracle():
    feature, attention = _layouts()
    rng = np.random.default_rng(26092721)
    output_c = rng.standard_normal((16, 32768), dtype=np.float64)

    partials = [
        pairfm_source_partial_carriers_numpy(
            output_c,
            source_logical_rank=rank,
            feature_layout=feature,
            attention_layout=attention,
        )
        for rank in range(8)
    ]
    actual = np.sum(partials, axis=0)
    official_real = output_c_to_feature_major_pairfm_numpy(
        output_c, feature_layout=feature, attention_layout=attention
    )
    expected = official_real[0::2] + 1j * official_real[1::2]

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        distributed_pairfm_carriers_numpy(
            output_c, feature_layout=feature, attention_layout=attention
        ),
        expected,
    )


@pytest.mark.parametrize("world_size", [8, 4, 2])
def test_rank_level_pre_add_then_reduce_matches_oracle(world_size: int):
    """A rank owning several slots may add its same-target slot partials
    before the all-to-all; the target sum over ranks is unchanged."""

    feature, attention = _layouts()
    rng = np.random.default_rng(26092922 + world_size)
    output_c = rng.standard_normal((16, 32768), dtype=np.float64)
    slots_per_rank = 8 // world_size

    def rank_partial(rank: int) -> np.ndarray:
        return sum(
            pairfm_source_partial_carriers_numpy(
                output_c,
                source_logical_rank=slot,
                feature_layout=feature,
                attention_layout=attention,
            )
            for slot in range(rank * slots_per_rank, (rank + 1) * slots_per_rank)
        )

    rank_partials = [rank_partial(rank) for rank in range(world_size)]
    expected = distributed_pairfm_carriers_numpy(
        output_c, feature_layout=feature, attention_layout=attention
    )
    for target in range(8):
        carrier = sum(partial[target] for partial in rank_partials)
        np.testing.assert_allclose(carrier, expected[target], rtol=0, atol=1e-12)


def test_source_partial_occupies_only_its_two_global_feature_blocks():
    feature, attention = _layouts()
    output_c = np.arange(16 * 32768, dtype=np.float64).reshape(16, 32768)
    source_rank = 5
    partial = pairfm_source_partial_carriers_numpy(
        output_c,
        source_logical_rank=source_rank,
        feature_layout=feature,
        attention_layout=attention,
    )

    block = 32768 // 16
    occupied = np.any(partial != 0, axis=0)
    expected = np.zeros(32768, dtype=bool)
    expected[10 * block : 12 * block] = True
    np.testing.assert_array_equal(occupied, expected)


def test_distributed_complexity_records_changed_epilogue_work():
    feature, attention = _layouts()
    complexity = distributed_pairfm_complexity(
        feature, attention, devices=range(8)
    )

    assert complexity.conversion_pt_ct == 6400
    assert complexity.conversion_rotations == 864
    assert complexity.partial_carriers_total == 64
    assert complexity.remote_partial_transfers == 56
    assert complexity.target_reduce_additions == 56
    assert complexity.final_rescales == 8
    assert complexity.target_pair_imults == 64
    # Each physical source context encodes both transforms independently.
    assert complexity.distributed_encoded_rows_total == 8 * (256 + 512 + 2)

class _FakePartial:
    """Just enough of a Cipher for ``pairfm_target_reduce_fhe`` on the CPU."""

    def __init__(self, value: float, log):
        self.value = float(value)
        self.released = False
        self._log = log

    def release(self) -> None:
        self.released = True
        self._log.append(("release", self.value))


class _FakeFHE:
    def __init__(self, log):
        self._log = log

    def homo_add(self, left, right, _context):
        self._log.append(("add", left.value, right.value))
        return _FakePartial(left.value + right.value, self._log)

    def rescale(self, cipher, _context):
        self._log.append(("rescale", cipher.value))
        return _FakePartial(cipher.value / 2.0, self._log)


def _fake_context(log):
    from types import SimpleNamespace

    return SimpleNamespace(fhe=_FakeFHE(log), context=object(), device="cpu")


def test_target_reduce_adds_every_partial_then_rescales_once(monkeypatch):
    import llama3fhe.operators.attention.distributed.distributed_output as module

    monkeypatch.setattr(module, "sync_device", lambda _device: None)
    log: list = []
    partials = [_FakePartial(v, log) for v in (1.0, 2.0, 4.0, 8.0)]
    out = pairfm_target_reduce_fhe(partials, crypto_context=_fake_context(log))
    assert out.value == (1.0 + 2.0 + 4.0 + 8.0) / 2.0
    assert [entry for entry in log if entry[0] == "rescale"] == [("rescale", 15.0)]
    assert sum(1 for entry in log if entry[0] == "add") == 3
    # every input partial and every intermediate accumulator is released;
    # only the returned carrier is still owned by the caller
    assert all(p.released for p in partials)
    assert not out.released


def test_target_reduce_releases_inputs_when_the_add_fails(monkeypatch):
    import llama3fhe.operators.attention.distributed.distributed_output as module

    monkeypatch.setattr(module, "sync_device", lambda _device: None)
    log: list = []
    context = _fake_context(log)

    def failing_add(_left, _right, _context):
        raise RuntimeError("boom")

    context.fhe.homo_add = failing_add
    partials = [_FakePartial(v, log) for v in (1.0, 2.0, 4.0)]
    with pytest.raises(RuntimeError, match="boom"):
        pairfm_target_reduce_fhe(partials, crypto_context=context)
    assert all(p.released for p in partials)


def test_target_reduce_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one partial"):
        pairfm_target_reduce_fhe([], crypto_context=_fake_context([]))


def _fake_source_partials(monkeypatch, log, *, level_of=lambda slot: 7):
    """Replace the device source half by a fake returning 8 tagged partials per slot."""

    import llama3fhe.operators.attention.distributed.distributed_output as module

    calls: list = []

    def fake(output_ciphers, *, source_logical_rank, **kwargs):
        calls.append((tuple(output_ciphers), int(source_logical_rank)))
        slot = int(source_logical_rank)
        partials = tuple(
            _FakePartial(1000 * slot + target, log) for target in range(8)
        )
        for partial in partials:
            partial.level = level_of(slot)
        return PairFMSourcePartialResult(
            partial_ciphers=partials,
            source_logical_rank=slot,
            global_query_pairs=(2 * slot, 2 * slot + 1),
            wall_seconds=0.5,
            stage_seconds={"pairfm_pack_encode": 0.1, "partial_online": 0.2},
        )

    monkeypatch.setattr(module, "pairfm_source_partials_fhe", fake)
    monkeypatch.setattr(module, "sync_device", lambda _device: None)
    monkeypatch.setattr(
        module, "_geometry", lambda _f, _a: (16, None, None, None)
    )
    return calls


def _fake_rank_context(log):
    context = _fake_context(log)
    context.level_for_cipher = lambda cipher: getattr(cipher, "level", 7)
    context.max_slots = 32768
    return context


def test_rank_partials_single_slot_is_the_source_half_unchanged(monkeypatch):
    log: list = []
    calls = _fake_source_partials(monkeypatch, log)
    oc = ("oc0", "oc1")
    result = pairfm_rank_partials_fhe(
        [oc], slot_indices=[3], feature_layout=None, attention_layout=None,
        crypto_context=_fake_rank_context(log),
    )
    assert calls == [(oc, 3)]
    assert [p.value for p in result.partial_ciphers] == [3000 + t for t in range(8)]
    assert result.slot_indices == (3,)
    assert result.global_query_pairs == (6, 7)
    assert result.slot_wall_seconds == (0.5,)
    assert not any(entry[0] in {"add", "release"} for entry in log)
    assert result.stage_seconds["pairfm_pack_encode"] == pytest.approx(0.1)


def test_rank_partials_pre_add_same_target_across_slots(monkeypatch):
    log: list = []
    calls = _fake_source_partials(monkeypatch, log)
    groups = [("a0", "a1"), ("b0", "b1")]
    result = pairfm_rank_partials_fhe(
        groups, slot_indices=(4, 5), feature_layout=None, attention_layout=None,
        crypto_context=_fake_rank_context(log),
    )
    assert calls == [(("a0", "a1"), 4), (("b0", "b1"), 5)]
    # partial[target] = slot4[target] + slot5[target]
    assert [p.value for p in result.partial_ciphers] == [
        (4000 + t) + (5000 + t) for t in range(8)
    ]
    assert sum(1 for entry in log if entry[0] == "add") == 8
    # both slots' raw partials are released, the 8 sums are owned by the caller
    assert sum(1 for entry in log if entry[0] == "release") == 16
    assert not any(p.released for p in result.partial_ciphers)
    assert result.slot_indices == (4, 5)
    assert result.global_query_pairs == (8, 9, 10, 11)
    assert result.slot_wall_seconds == (0.5, 0.5)
    assert result.stage_seconds["partial_online"] == pytest.approx(0.4)
    assert "slot_pre_add" in result.stage_seconds


def test_rank_partials_reject_level_mismatch_and_bad_groups(monkeypatch):
    log: list = []
    _fake_source_partials(monkeypatch, log, level_of=lambda slot: 7 if slot == 0 else 6)
    context = _fake_rank_context(log)
    with pytest.raises(ValueError, match="level differs"):
        pairfm_rank_partials_fhe(
            [("a0", "a1"), ("b0", "b1")], slot_indices=(0, 1),
            feature_layout=None, attention_layout=None, crypto_context=context,
        )
    # nothing leaks: slot 0's eight partials and slot 1's eight are all released
    assert sum(1 for entry in log if entry[0] == "release") == 16
    with pytest.raises(ValueError, match="at least one slot"):
        pairfm_rank_partials_fhe([], slot_indices=(), feature_layout=None,
                                 attention_layout=None, crypto_context=context)
    with pytest.raises(ValueError, match="groups"):
        pairfm_rank_partials_fhe([("a0", "a1")], slot_indices=(0, 1), feature_layout=None,
                                 attention_layout=None, crypto_context=context)
    with pytest.raises(ValueError, match="duplicate"):
        pairfm_rank_partials_fhe([("a0", "a1"), ("b0", "b1")], slot_indices=(2, 2),
                                 feature_layout=None, attention_layout=None, crypto_context=context)


@pytest.mark.parametrize("devices", [range(7), range(9), [0, -1] * 4])
def test_plan_rejects_invalid_device_placement(devices):
    feature, attention = _layouts()
    with pytest.raises(ValueError):
        build_pairfm_reduce_scatter_plan(feature, attention, devices)
