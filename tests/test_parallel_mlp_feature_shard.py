from __future__ import annotations

from dataclasses import replace

import pytest

from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.shard.mlp_feature_shard import (
    feature_shard_group,
    feature_shard_pages,
)
from llama3fhe.parallel.presets import A100_HIGH, RTX4090_LOW, ParallelPreset


def test_groups_partition_the_world_into_adjacent_ranks() -> None:
    for k in (1, 2, 4, 8):
        seen: list[int] = []
        for rank in range(8):
            group = feature_shard_group(rank, 8, k)
            assert len(group) == k and rank in group
            assert group == tuple(range(group[0], group[0] + k))
            assert all(feature_shard_group(peer, 8, k) == group for peer in group)
            if rank % k == 0:
                seen.extend(group)
        assert sorted(seen) == list(range(8))
    assert feature_shard_group(3, 8, 2) == (2, 3)
    assert feature_shard_group(5, 8, 4) == (4, 5, 6, 7)


def test_group_rejects_bad_k_or_rank() -> None:
    with pytest.raises(ValueError):
        feature_shard_group(0, 8, 3)
    with pytest.raises(ValueError):
        feature_shard_group(0, 8, 0)
    with pytest.raises(IndexError):
        feature_shard_group(8, 8, 2)


def test_pages_tile_the_intermediate_without_overlap() -> None:
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.operators.linear.prefill.mlp import FeatureMajorMLPLayout

    layout = FeatureMajorPrefillLayout(seq_len=16, hidden_dim=4096, slots=32768)
    pages = FeatureMajorMLPLayout(layout, 14336).feature_page_count
    assert pages == 4
    for k in (1, 2, 4):
        union: list[int] = []
        for position in range(k):
            mine = feature_shard_pages(pages, position, k)
            assert len(mine) == pages // k
            assert mine == tuple(range(mine[0], mine[0] + len(mine)))
            union.extend(mine)
        assert sorted(union) == list(range(pages))
    assert feature_shard_pages(4, 1, 2) == (2, 3)
    with pytest.raises(ValueError):
        feature_shard_pages(4, 0, 3)
    with pytest.raises(ValueError):
        feature_shard_pages(4, 0, 8)
    with pytest.raises(ValueError):
        feature_shard_pages(4, 2, 2)


def test_operator_apply_accepts_chunk_subsets() -> None:
    import inspect

    from llama3fhe.operators.linear.operator import LinearOperator

    parameters = inspect.signature(LinearOperator.apply).parameters
    assert parameters["chunk_indices"].default is None
    assert parameters["rescale"].default is True
    parameters = inspect.signature(LinearOperator.project).parameters
    assert "chunk_indices" in parameters and "rescale" in parameters


def test_presets_pick_the_measured_k_per_card() -> None:
    assert RTX4090_LOW.mlp_feature_shard_k == 2
    assert A100_HIGH.mlp_feature_shard_k == 4
    assert ParallelPreset(name="x").mlp_feature_shard_k == 1
    assert replace(RTX4090_LOW, mlp_feature_shard_k=1).mlp_feature_shard_k == 1
    with pytest.raises(ValueError):
        replace(RTX4090_LOW, mlp_feature_shard_k=3)
    with pytest.raises(ValueError):
        replace(RTX4090_LOW, mlp_feature_shard_k=0)


def test_exchange_tags_are_rank_independent() -> None:
    """A rank-specific tag deadlocks the world (run17): the two group peers
    would each wait for a round tag the other never sends."""

    from llama3fhe.operators.linear.shard import mlp_feature_shard as module

    class _Transport:
        world_size = 8

        def __init__(self) -> None:
            self.tags: list[str] = []

        def all_to_all(self, tag, payloads_by_dst):
            self.tags.append(str(tag))
            assert len(payloads_by_dst) == self.world_size
            return [None] * self.world_size

    for rank in (2, 3):
        transport = _Transport()
        comm: dict[str, int] = {}
        module._exchange(transport, "mlp/gu_in", rank=rank, group=(2, 3), payload_for={}, comm=comm)
        assert transport.tags == ["mlp/gu_in"]
        assert comm == {"mlp/gu_in_sent": 0, "mlp/gu_in_received": 0}
