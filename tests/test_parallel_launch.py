"""CPU tests of the launcher's rank -> device map (llama3fhe.parallel.launch)."""

from __future__ import annotations

import pytest

from llama3fhe.parallel.launch import expand_devices


def test_one_rank_per_card_by_default() -> None:
    assert expand_devices(8) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert expand_devices(8, [7, 6, 5, 4, 3, 2, 1, 0]) == (7, 6, 5, 4, 3, 2, 1, 0)


def test_four_card_host_runs_two_adjacent_ranks_per_card() -> None:
    # 4 x RTX PRO 6000: ranks (2c, 2c+1) share card c, so a mlp_feature_shard_k=2
    # group never crosses a card.
    assert expand_devices(8, ranks_per_device=2) == (0, 0, 1, 1, 2, 2, 3, 3)
    assert expand_devices(8, [0, 0, 1, 1, 2, 2, 3, 3], ranks_per_device=2) == (0, 0, 1, 1, 2, 2, 3, 3)
    assert expand_devices(8, [3, 3, 2, 2, 1, 1, 0, 0], ranks_per_device=2) == (3, 3, 2, 2, 1, 1, 0, 0)
    assert expand_devices(8, ranks_per_device=8) == (0,) * 8


def test_bad_maps_fail_before_any_process_starts() -> None:
    with pytest.raises(ValueError):
        expand_devices(8, [0, 1, 2, 3])  # fewer devices than ranks
    with pytest.raises(ValueError):
        expand_devices(8, [0, 0, 1, 1, 2, 2, 3, 3])  # repeats without --ranks-per-device
    with pytest.raises(ValueError):
        expand_devices(8, [0, 0, 0, 1, 1, 2, 3, 3], ranks_per_device=2)  # uneven sharing
    with pytest.raises(ValueError):
        expand_devices(8, ranks_per_device=3)  # must divide world_size
    with pytest.raises(ValueError):
        expand_devices(8, [-1, 1, 2, 3, 4, 5, 6, 7])
    with pytest.raises(ValueError):
        expand_devices(0)
