from __future__ import annotations

import pytest

from llama3fhe.parallel.prefill.ownership import (
    attention_query_pairs_for_rank,
    build_world_ownership,
    cipher_indices_for_rank,
    owner_of_cipher,
    owner_of_query_pair,
    ownership_from_layouts,
)


def test_production_assignment_is_two_adjacent_ciphers_per_rank() -> None:
    world = build_world_ownership()
    assert world.world_size == 8
    assert world.cipher_count == 16
    assert world.query_pair_count == 16
    assert world.seq_len == 128
    assert world[0].cipher_indices == (0, 1)
    assert world[7].cipher_indices == (14, 15)
    assert (world[3].token_start, world[3].token_stop) == (48, 64)
    assert world[5].query_pair_indices == (10, 11)
    assert world[5].target_carrier_index == 5
    covered = sorted(index for rank in world for index in rank.cipher_indices)
    assert covered == list(range(16))
    pairs = sorted(index for rank in world for index in rank.query_pair_indices)
    assert pairs == list(range(16))


def test_owner_lookups_agree_with_assignment() -> None:
    world = build_world_ownership()
    for cipher in range(16):
        assert world.owner_of_cipher(cipher) == cipher // 2
        assert owner_of_cipher(cipher) == cipher // 2
        assert cipher in world[cipher // 2].cipher_indices
    for pair in range(16):
        assert world.owner_of_query_pair(pair) == pair // 2
        assert owner_of_query_pair(pair) == pair // 2
    assert cipher_indices_for_rank(6) == (12, 13)
    assert attention_query_pairs_for_rank(6) == (12, 13)
    assert world.peers(2) == (0, 1, 3, 4, 5, 6, 7)
    with pytest.raises(IndexError):
        world.owner_of_cipher(16)
    with pytest.raises(IndexError):
        world[8]


def test_four_rank_world_owns_two_slots_per_rank() -> None:
    """``world_size 4``: four adjacent FM ciphers = two slots = two W_O carriers per rank."""

    world = build_world_ownership(world_size=4, ciphers_per_rank=4)
    assert (world.cipher_count, world.query_pair_count, world.seq_len) == (16, 16, 128)
    assert (world.slots_per_rank, world.slot_count) == (2, 8)
    for rank in world:
        r = rank.rank
        assert rank.cipher_indices == tuple(range(4 * r, 4 * r + 4))
        assert rank.query_pair_indices == tuple(range(4 * r, 4 * r + 4))
        assert (rank.token_start, rank.token_stop) == (32 * r, 32 * r + 32)
        assert rank.target_carrier_indices == (2 * r, 2 * r + 1)
        assert rank.slot_indices == (2 * r, 2 * r + 1)
        assert rank.cipher_pairs() == ((4 * r, 4 * r + 1), (4 * r + 2, 4 * r + 3))
        assert rank.slot_query_pairs() == ((4 * r, 4 * r + 1), (4 * r + 2, 4 * r + 3))
        assert (rank.ciphers_per_rank, rank.slots_per_rank) == (4, 2)
        with pytest.raises(ValueError):
            rank.target_carrier_index  # ambiguous on a two-slot rank
        with pytest.raises(ValueError):
            rank.cipher_pair
        assert rank.to_dict()["target_carrier_indices"] == [2 * r, 2 * r + 1]
        assert rank.to_dict()["slot_indices"] == [2 * r, 2 * r + 1]
    for carrier in range(8):
        assert world.owner_of_target_carrier(carrier) == carrier // 2
        assert world.owner_of_slot(carrier) == carrier // 2
    for cipher in range(16):
        assert world.owner_of_cipher(cipher) == cipher // 4
    for pair in range(16):
        assert world.owner_of_query_pair(pair) == pair // 4
    with pytest.raises(IndexError):
        world.owner_of_target_carrier(8)
    # the eight-rank world still exposes the single-carrier view
    eight = build_world_ownership()
    assert eight[5].target_carrier_indices == (5,)
    assert eight[5].target_carrier_index == 5
    assert eight[5].slot_query_pairs() == ((10, 11),)
    assert eight.owner_of_target_carrier(5) == 5


def test_world_of_two_and_rejected_worlds() -> None:
    two = build_world_ownership(world_size=2, ciphers_per_rank=8)
    assert two[1].cipher_indices == tuple(range(8, 16))
    assert two[1].target_carrier_indices == (4, 5, 6, 7)
    assert two[1].slot_query_pairs() == ((8, 9), (10, 11), (12, 13), (14, 15))
    with pytest.raises(ValueError):
        build_world_ownership(world_size=16, ciphers_per_rank=1)
    # a smaller world is legal in the abstract (8 ciphers), but not for the
    # 16-ciphertext production layout
    small = build_world_ownership(world_size=4, ciphers_per_rank=2)
    assert small.cipher_count == 8

    class Layout:
        cipher_count = 16
        tokens_per_cipher = 8

    with pytest.raises(ValueError):
        ownership_from_layouts(Layout(), world_size=16)  # one cipher per rank
    with pytest.raises(ValueError):
        ownership_from_layouts(Layout(), world_size=3)  # does not divide
    assert ownership_from_layouts(Layout(), world_size=4).ciphers_per_rank == 4


def test_rank_ownership_validates_carriers_and_slots() -> None:
    from llama3fhe.parallel.prefill.ownership import RankOwnership

    # explicit carriers must be the ciphers' pairs
    with pytest.raises(ValueError):
        RankOwnership(rank=0, cipher_indices=(0, 1, 2, 3), token_start=0, token_count=32,
                      query_pair_indices=(0, 1, 2, 3), target_carrier_indices=(0,))
    # non-adjacent / odd-start / odd-count cipher runs are rejected
    for bad in ((0, 2), (1, 2), (0, 1, 2)):
        with pytest.raises(ValueError):
            RankOwnership(rank=0, cipher_indices=bad, token_start=0, token_count=8, query_pair_indices=bad)
    # query pairs that do not line up with the owned slots cannot be grouped
    skew = RankOwnership(rank=0, cipher_indices=(0, 1, 2, 3), token_start=0, token_count=32,
                         query_pair_indices=(2, 3, 4, 5))
    with pytest.raises(ValueError):
        skew.slot_query_pairs()


@pytest.mark.parametrize("world_size", [8, 4, 2])
def test_assignment_matches_shard_plans_and_pairfm_plan(world_size: int) -> None:
    pytest.importorskip("numpy")
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.operators.attention.distributed.distributed_output import (
        build_pairfm_reduce_scatter_plan,
    )
    from llama3fhe.operators.linear.shard.mlp_token_shard import mlp_token_shard_plan
    from llama3fhe.operators.linear.shard.qkv_source_shard import qkv_token_shard_plan

    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    world = ownership_from_layouts(feature, attention, world_size=world_size)
    qkv = qkv_token_shard_plan(feature, tuple(range(world_size)))
    mlp = mlp_token_shard_plan(feature, tuple(range(world_size)))
    # the PairFM algebra is fixed at 8 source slots; slot s lives on its owner
    pairfm = build_pairfm_reduce_scatter_plan(
        feature, attention, [world.owner_of_slot(slot) for slot in range(8)]
    )
    for rank in world:
        spec = qkv.for_rank(rank.rank)
        assert spec.input_indices == rank.cipher_indices
        assert spec.output_indices == rank.cipher_indices
        assert spec.cipher_count == rank.ciphers_per_rank
        assert (spec.token_start, spec.token_count) == (rank.token_start, rank.token_count)
        assert mlp.for_rank(rank.rank).input_indices == rank.cipher_indices
        for slot, query_pairs in zip(rank.slot_indices, rank.slot_query_pairs(), strict=True):
            placement = pairfm.placements[slot]
            assert placement.device_rank == rank.rank
            assert placement.query_pair_indices == query_pairs
            assert placement.target_carrier_index in rank.target_carrier_indices
            assert world.owner_of_target_carrier(placement.target_carrier_index) == rank.rank


def test_rejects_uneven_or_odd_topologies() -> None:
    with pytest.raises(ValueError):
        build_world_ownership(ciphers_per_rank=1)
    with pytest.raises(ValueError):
        build_world_ownership(query_pair_count=12)

    class Layout:
        cipher_count = 12
        tokens_per_cipher = 8

    with pytest.raises(ValueError):
        ownership_from_layouts(Layout(), world_size=8)
    small = ownership_from_layouts(Layout(), world_size=6)
    assert small.ciphers_per_rank == 2
    assert small.query_pair_count == 12
