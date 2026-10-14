"""CPU tests for the decode ownership map (llama3fhe.parallel.decode.decode_ownership)."""

from __future__ import annotations

import itertools

import pytest

from llama3fhe.parallel.decode.decode_ownership import DecodeRankOwnership, contiguous_shard, decode_world


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_query_streams_partition_all_eight(world_size) -> None:
    world = decode_world(world_size)
    streams = [own.query_streams for own in world]
    assert all(len(s) == 8 // world_size for s in streams)
    flat = list(itertools.chain.from_iterable(streams))
    assert len(flat) == len(set(flat)) == 8
    assert sorted(flat) == [(g, r) for g in range(2) for r in range(4)]
    # contiguous in stream order: rank 0 starts at (0, 0)
    assert flat == [divmod(q, 4) for q in range(8)]


def test_single_rank_owns_everything() -> None:
    own = DecodeRankOwnership(0, 1)
    assert own.is_full_attention
    assert own.query_streams == tuple((g, r) for g in range(2) for r in range(4))
    assert own.chunk_indices(8) == tuple(range(8))
    assert own.mlp_page_indices == (0, 1, 2, 3)
    assert own.mlp_chunk_split == 1 and own.mlp_partners == (0,)
    assert own.mlp_chunk_indices(16) == tuple(range(16))


@pytest.mark.parametrize("world_size,chunk_count", [(2, 8), (4, 8), (8, 8), (8, 16), (4, 16)])
def test_chunk_indices_partition(world_size, chunk_count) -> None:
    shards = [own.chunk_indices(chunk_count) for own in decode_world(world_size)]
    flat = list(itertools.chain.from_iterable(shards))
    assert sorted(flat) == list(range(chunk_count))
    assert len(set(map(len, shards))) == 1


def test_chunk_indices_reject_uneven_split() -> None:
    with pytest.raises(ValueError):
        DecodeRankOwnership(0, 8).chunk_indices(12)
    with pytest.raises(ValueError):
        contiguous_shard(8, 3, 3)


def test_mlp_pages_whole_pages_for_small_worlds() -> None:
    w2 = decode_world(2)
    assert [own.mlp_page_indices for own in w2] == [(0, 1), (2, 3)]
    assert all(own.mlp_chunk_split == 1 and own.mlp_partners == (own.rank,) for own in w2)
    w4 = decode_world(4)
    assert [own.mlp_page_indices for own in w4] == [(0,), (1,), (2,), (3,)]


def test_mlp_pages_split_chunks_for_eight_ranks() -> None:
    world = decode_world(8)
    assert [own.mlp_page_indices for own in world] == [(0,), (0,), (1,), (1,), (2,), (2,), (3,), (3,)]
    assert all(own.mlp_chunk_split == 2 for own in world)
    assert [own.mlp_chunk_index for own in world] == [0, 1] * 4
    assert world[5].mlp_partners == (4, 5)
    assert world[0].mlp_chunk_indices(16) == tuple(range(8))
    assert world[1].mlp_chunk_indices(16) == tuple(range(8, 16))
    # partners together cover each page's chunks exactly once
    for page in range(4):
        members = [own for own in world if own.mlp_page_indices == (page,)]
        flat = list(itertools.chain.from_iterable(own.mlp_chunk_indices(8) for own in members))
        assert sorted(flat) == list(range(8))


def test_invalid_worlds_are_rejected() -> None:
    with pytest.raises(ValueError):
        DecodeRankOwnership(0, 3)
    with pytest.raises(ValueError):
        DecodeRankOwnership(0, 16)  # 8 streams cannot feed 16 ranks
    with pytest.raises(IndexError):
        DecodeRankOwnership(8, 8)
    with pytest.raises(TypeError):
        DecodeRankOwnership(True, 8)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_kv_refresh_indices_cover_the_four_ciphers_once(world_size) -> None:
    covered = []
    for rank in range(world_size):
        mine = DecodeRankOwnership(rank, world_size).kv_refresh_indices(4)
        assert len(mine) == (4 // world_size if world_size <= 4 else (1 if rank < 4 else 0))
        covered.extend(mine)
    assert sorted(covered) == [0, 1, 2, 3]
