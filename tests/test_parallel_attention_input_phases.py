"""CPU checks for the T2 phase-1 attention-input split (docs §12.12).

Own-source V partials, duplicated K, and the per-phase rotation planner
that feeds the ``attn_input_*`` key sub-tiers.  No GPU: only the route
generators and the pure-Python ownership mapping.
"""

from __future__ import annotations

import pytest

from llama3fhe.parallel.prefill.keys import ATTN_INPUT_PHASE_TIERS, TIER_ORDER, KeyTier
from llama3fhe.parallel.prefill.ownership import (
    build_world_ownership,
    key_pairs_for_rank,
    value_carriers_for_rank,
    value_group_ranks,
)


def test_gqa_roles_follow_the_layout_mapping() -> None:
    world = build_world_ownership()
    for rank in world:
        assert key_pairs_for_rank(rank) == (rank.rank // 2,)
        assert value_carriers_for_rank(rank) == (rank.rank // 4,)
    assert value_group_ranks(world, 0) == (0, 1, 2, 3)
    assert value_group_ranks(world, 1) == (4, 5, 6, 7)
    # every key pair has exactly two readers, every carrier four
    readers = {}
    for rank in world:
        for pair in key_pairs_for_rank(rank):
            readers.setdefault(pair, []).append(rank.rank)
    assert readers == {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
    with pytest.raises(ValueError):
        key_pairs_for_rank(world[0], gqa_ratio=0)


def test_phase_tiers_are_known_and_ordered_after_the_monolithic_tier() -> None:
    assert ATTN_INPUT_PHASE_TIERS == ("attn_input_lane", "attn_input_q", "attn_input_k", "attn_input_v")
    index = TIER_ORDER.index
    assert index("attn_input") < index("attn_input_lane") < index("attn_input_v") < index("attn_core")
    tier = KeyTier("attn_input_v", 2, (3, -1, 0), 13)
    assert tier.rotations == (-1, 3)


def _production():
    pytest.importorskip("numpy")
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.prefill import input as core

    carrier = LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768)
    attention = AttentionPairLayout(seq_len=128, head_dim=128, query_heads=32, key_value_heads=8, slots=32768)
    return carrier, attention, core


def test_own_source_value_routes_partition_the_full_route_set() -> None:
    carrier, attention, core = _production()
    params = core._layout_parameters(carrier, attention)
    full = core._value_delta_routes(params)
    world = build_world_ownership()
    union: dict[int, list[int]] = {}
    for rank in world:
        part = core._value_delta_routes(params, sources=tuple(rank.cipher_indices))
        for offset, destinations in part.items():
            union.setdefault(int(offset), []).extend(int(v) for v in destinations)
    assert set(union) == {int(v) for v in full}
    for offset, destinations in full.items():
        assert sorted(union[int(offset)]) == sorted(int(v) for v in destinations)
    # each rank's own-source plan is far smaller than the 32-baby full plan (238 -> 1190 modulus)
    own = core.choose_sparse_diagonal_plan(
        tuple(core._value_delta_routes(params, sources=(0, 1))), slots=32768, max_baby_offsets=32
    )
    assert len(own.baby_offsets) <= 32
    assert len(own.diagonal_offsets) < len(full)
    with pytest.raises(IndexError):
        core._value_delta_routes(params, sources=(16,))


def test_shard_rotations_are_a_subset_of_the_monolithic_set_and_cover_it() -> None:
    carrier, attention, core = _production()
    from llama3fhe.operators.attention.distributed.distributed_input import attention_input_shard_rotations

    monolithic = set(core.complex_token_attention_rotations(carrier, attention))
    world = build_world_ownership()
    union: set[int] = set()
    sizes = {}
    for rank in world:
        phases = attention_input_shard_rotations(
            carrier,
            attention,
            query_pair_indices=rank.query_pair_indices,
            key_pair_indices=key_pairs_for_rank(rank),
            value_carrier_indices=(0, 1),
            value_source_indices=rank.cipher_indices,
        )
        assert set(phases) == {"lane", "query", "key", "value"}
        for name, rotations in phases.items():
            assert rotations == tuple(sorted(set(rotations)))
            assert 0 not in rotations
            union.update(rotations)
        sizes[rank.rank] = {name: len(v) for name, v in phases.items()}
    # own-source V plans use a larger modulus than the monolithic 238 plan, so
    # their baby/giant keys are new; everything else is drawn from the monolithic set
    conj = (1 << 17) - 1
    for rank in world:
        phases = attention_input_shard_rotations(
            carrier, attention, query_pair_indices=rank.query_pair_indices,
            key_pair_indices=key_pairs_for_rank(rank), value_carrier_indices=(0, 1),
            value_source_indices=rank.cipher_indices,
        )
        assert set(phases["lane"]) <= monolithic
        assert set(phases["query"]) <= monolithic
        assert set(phases["key"]) <= monolithic
        assert conj in phases["query"] and conj in phases["key"] and conj in phases["value"]
    assert sizes[0]["lane"] == 14
    assert sizes[0]["query"] < sizes[0]["key"] < 200
    assert sizes[0]["value"] < 100
    # the lane phase is source independent; Q/K phases repeat the axis2 + conjugate fixed sets
    assert all(sizes[r]["lane"] == 14 for r in sizes)


def test_empty_phase_returns_no_rotations() -> None:
    carrier, attention, _ = _production()
    from llama3fhe.operators.attention.distributed.distributed_input import attention_input_shard_rotations

    phases = attention_input_shard_rotations(
        carrier, attention, query_pair_indices=(), key_pair_indices=(), value_carrier_indices=(),
        value_source_indices=(0, 1),
    )
    assert phases["query"] == () and phases["key"] == () and phases["value"] == ()
    assert len(phases["lane"]) == 14


def test_value_sub_tier_covers_both_carriers_without_extra_rotations() -> None:
    """The V phase emits the own-source partial of every carrier, so the
    ``attn_input_v`` tier of a rank must hold the block-gather offsets of both
    carriers even though the rank only consumes one of them (run12 bug)."""

    carrier, attention, core = _production()
    from llama3fhe.operators.attention.distributed.distributed_input import attention_input_shard_rotations
    from llama3fhe.parallel.prefill.keys import build_key_plan, normalize_rotation
    from llama3fhe.parallel.presets import RTX4090_LOW
    from llama3fhe.parallel.worker import production_layouts

    feature, attention_layout = production_layouts()
    world = build_world_ownership()
    rank0 = world[0]
    plan = build_key_plan(
        RTX4090_LOW,
        feature,
        attention_layout,
        tiers=("bootstrap", "attn_input_v"),
        ownership=rank0,
    )
    tier = {item.name: item for item in plan.tiers}["attn_input_v"]
    used = attention_input_shard_rotations(
        carrier,
        attention,
        query_pair_indices=(),
        key_pair_indices=(),
        value_carrier_indices=(0, 1),
        value_source_indices=rank0.cipher_indices,
    )["value"]
    needed = {normalize_rotation(offset, ring_dim=plan.ring_dim) for offset in used}
    assert needed <= set(tier.rotations)
    only_own = attention_input_shard_rotations(
        carrier,
        attention,
        query_pair_indices=(),
        key_pair_indices=(),
        value_carrier_indices=value_carriers_for_rank(rank0),
        value_source_indices=rank0.cipher_indices,
    )["value"]
    assert len(only_own) < len(used)
