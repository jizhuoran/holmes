"""CPU preflight for the T2 plaintext Prefix benchmark path."""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest

from bench.t2_prefill_decode import (
    _prefix_cache_token_count,
    combined_preset,
    decode_preset_and_schedule,
    parse_args,
    rank_phase_key_plans,
)
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.attention.prefix.prefix_softmax import prefix_softmax_rotations
from llama3fhe.parallel.prefill.keys import normalize_rotation
from llama3fhe.parallel.presets import preset_by_name


def test_prefix_cache_token_count_for_key_planning(tmp_path) -> None:
    path = tmp_path / "prefix.npz"
    np.savez(path, prefix_k=np.int64(2))
    assert _prefix_cache_token_count(path) == 2
    np.savez(path, prefix_k=np.int64(0))
    with pytest.raises(ValueError, match="unsupported prefix_k"):
        _prefix_cache_token_count(path)
    np.savez(path, prefix_k=np.int64(129))
    with pytest.raises(ValueError, match="unsupported prefix_k"):
        _prefix_cache_token_count(path)


def test_prefix_flags_are_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "t2_prefill_decode.py",
            "--prefix-kv-cache",
            "prefix.npz",
            "--poly-kv-matches-prefix",
        ],
    )
    args = parse_args()
    assert args.prefix_kv_cache == "prefix.npz"
    assert args.poly_kv_matches_prefix is True


def test_prefix_plan_keeps_phase_warmup_and_adds_local_rotations() -> None:
    preset = combined_preset(preset_by_name("rtx6000pro-high"))
    decode_preset, schedule = decode_preset_and_schedule(
        preset, kv_limbs=12, shifted_lanes=True, sl_baby=16
    )
    kwargs = dict(
        decode_preset=decode_preset,
        decode_schedule=schedule,
        layers=tuple(range(32)),
    )
    plain, _, plain_phase, plain_decode = rank_phase_key_plans(preset, 2, **kwargs)
    prefix, _, prefix_phase, prefix_decode = rank_phase_key_plans(
        preset, 2, prefix_k=2, **kwargs
    )
    local_layout = replace(AttentionPairLayout(), query_heads=4, key_value_heads=2)
    ring_dim = 1 << int(preset.log_n)
    rotations = {
        normalize_rotation(rotation, ring_dim=ring_dim)
        for rotation in prefix_softmax_rotations(local_layout, prefix_k=2)
    }
    assert rotations <= set(prefix.tier("softmax").rotations)
    assert rotations <= set(prefix_phase.tier("softmax").rotations)
    assert set(plain.union) <= set(prefix.union)
    assert set(plain_phase.union) <= set(prefix_phase.union)
    assert plain_decode.union == prefix_decode.union
