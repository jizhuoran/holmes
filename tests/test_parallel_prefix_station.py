"""Plaintext Prefix routing in the rank-local Prefill attention core."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from llama3fhe.approx.prefix import PrefixKVBundle
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.parallel.prefill.keys import build_key_plan, normalize_rotation
from llama3fhe.parallel.prefill.ownership import ownership_from_layouts
from llama3fhe.parallel.prefill.stations import attention_core_station
from llama3fhe.parallel.presets import RTX4090_LOW


def test_attention_core_passes_only_owned_plaintext_prefix_heads(monkeypatch) -> None:
    from llama3fhe.operators.attention.prefill import operator
    from llama3fhe.parallel.prefill import stations

    layout = AttentionPairLayout()
    keys = np.arange(
        2 * layout.key_value_heads * layout.head_dim, dtype=np.float32
    ).reshape(2, layout.key_value_heads, layout.head_dim)
    bundle = PrefixKVBundle.from_logical(keys, -keys, layout=layout)
    captured = {}

    class Cipher:
        state = SimpleNamespace(cur_limbs=8)

    def fake_attention(query, key, value, **kwargs):
        captured.update(kwargs)
        kwargs["on_qk_last_use"]()
        kwargs["on_pv_last_use"]()
        return SimpleNamespace(
            output_ciphers=(Cipher(), Cipher()), stage_seconds={}, level_trace={}
        )

    monkeypatch.setattr(operator, "attention_fhe_delta_alg2", fake_attention)
    monkeypatch.setattr(stations, "_sync", lambda context: None)
    result = attention_core_station(
        ((4, Cipher()), (5, Cipher())),
        ((1, Cipher()),),
        ((1, Cipher()),),
        ownership=None,
        attention_layout=layout,
        context=SimpleNamespace(device="cpu"),
        softmax_config=object(),
        operator_config=object(),
        bootstrap_operator=SimpleNamespace(calls=0),
        layer_idx=0,
        prefix_bundle=bundle,
    )
    local = captured["prefix_bundle"]
    assert isinstance(local, PrefixKVBundle)
    assert local.prefix_k == 2
    assert local.key.shape == (2, 2 * layout.head_dim)
    np.testing.assert_array_equal(
        local.key.reshape(2, 2, layout.head_dim), keys[:, 2:4, :]
    )
    np.testing.assert_array_equal(
        local.value.reshape(2, 2, layout.head_dim), -keys[:, 2:4, :]
    )
    assert result.sub_layout.query_heads == 4
    assert result.sub_layout.key_value_heads == 2
    assert tuple(i for i, _ in result.output_items) == (4, 5)


def test_prefix_key_plan_covers_rank_local_core_rotations() -> None:
    from llama3fhe.operators.attention.prefix.prefix_pc import (
        prefix_pc_pv_rotations,
        prefix_pc_qk_rotations,
    )
    from llama3fhe.operators.attention.prefix.prefix_softmax import (
        prefix_softmax_rotations,
    )

    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    rank = ownership_from_layouts(feature, attention, world_size=8)[2]
    tiers = ("attn_qk", "attn_pv", "softmax")
    plain = build_key_plan(RTX4090_LOW, feature, attention, tiers=tiers, ownership=rank)
    prefix = build_key_plan(
        RTX4090_LOW, feature, attention, tiers=tiers, ownership=rank, prefix_k=2
    )
    local_layout = replace(attention, query_heads=4, key_value_heads=2)
    ring_dim = 1 << int(RTX4090_LOW.log_n)
    expected = {
        "attn_qk": prefix_pc_qk_rotations(local_layout, prefix_k=2),
        "attn_pv": prefix_pc_pv_rotations(local_layout, prefix_k=2),
        "softmax": prefix_softmax_rotations(local_layout, prefix_k=2),
    }
    for tier in tiers:
        assert set(plain.tier(tier).rotations) <= set(prefix.tier(tier).rotations)
        assert {
            normalize_rotation(rot, ring_dim=ring_dim) for rot in expected[tier]
        } <= set(prefix.tier(tier).rotations)


def test_prefix_key_plan_supports_partitioned_fallback() -> None:
    from llama3fhe.operators.attention.prefix.prefix_softmax import (
        prefix_softmax_rotations,
    )

    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    rank = ownership_from_layouts(feature, attention, world_size=8)[0]
    plan = build_key_plan(
        RTX4090_LOW,
        feature,
        attention,
        tiers=("attn_qk", "attn_pv", "softmax"),
        ownership=rank,
        prefix_k=65,
    )
    local_layout = replace(attention, query_heads=4, key_value_heads=2)
    ring_dim = 1 << int(RTX4090_LOW.log_n)
    partitioned = {
        normalize_rotation(rotation, ring_dim=ring_dim)
        for rotation in prefix_softmax_rotations(local_layout, prefix_k=0)
    }
    assert partitioned <= set(plan.tier("softmax").rotations)
