"""P6 packing selection and rotation-planning contracts."""

from __future__ import annotations

from inspect import signature
from typing import get_args

import pytest

from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.decode_packing import DecodePacking, validate_decode_packing
from llama3fhe.graph import LayerResult, decode_layer_rotations, required_rotations
from llama3fhe.layouts import (
    AttentionPairLayout,
    FeatureMajorPrefillLayout,
    LinearCarrierLayout,
    PairFold128Layout,
)
from llama3fhe.model import Llama3Model
from llama3fhe.operators.attention.decode.decode import decode_output_c_rotations
from llama3fhe.operators.attention.decode.decode_adapter import decode_adapter_rotations
from llama3fhe.operators.attention.pairfold.pairfold_decode import (
    pairfold_decode_rotations,
)
from llama3fhe.operators.attention.pairfold.pairfold_route import pairfold_route_rotations
from llama3fhe.operators.linear.config import LinearOperatorConfig
from llama3fhe.operators.linear.pairfold.pairfold_output import pairfold_w_o_rotations
from llama3fhe.operators.linear.pairfold.pairfold_qkv import pairfold_qkv_rotations
from llama3fhe.operators.pairfold_kv_cache import (
    PairFoldCipherPage,
    PairFoldLayerCache,
)
from llama3fhe.schedule import LayerSchedule


def _linear_config(
    schedule: LayerSchedule, *, max_plaintext_rows: int, hoist_strategy: str
) -> LinearOperatorConfig:
    return LinearOperatorConfig(
        baby_steps=int(schedule.linear_baby_steps),
        baby_anchor_step=int(schedule.linear_anchor_step),
        max_plaintext_rows=int(max_plaintext_rows),
        hoist_strategy=str(hoist_strategy),
    )


def test_decode_packing_default_selects_pairfold_rotation_contract() -> None:
    assert set(get_args(DecodePacking)) == {"legacy_c", "pairfold128"}
    assert validate_decode_packing("legacy_c") == "legacy_c"
    assert validate_decode_packing("pairfold128") == "pairfold128"
    assert signature(required_rotations).parameters[
        "decode_packing"
    ].default == "pairfold128"
    assert signature(decode_layer_rotations).parameters[
        "decode_packing"
    ].default == "pairfold128"
    assert signature(Llama3Model.required_rotations).parameters[
        "decode_packing"
    ].default == "pairfold128"
    assert signature(Llama3Model.required_decode_rotations).parameters[
        "decode_packing"
    ].default == "pairfold128"

    prefill = FeatureMajorPrefillLayout()
    decode = FeatureMajorPrefillLayout(seq_len=1)
    attention = AttentionPairLayout()
    assert required_rotations(prefill, attention, emit_kv_cache=True) == (
        required_rotations(
            prefill,
            attention,
            emit_kv_cache=True,
            decode_packing="pairfold128",
        )
    )
    assert decode_layer_rotations(
        decode, attention, skip_attention=False
    ) == decode_layer_rotations(
        decode,
        attention,
        skip_attention=False,
        decode_packing="pairfold128",
    )

    model = Llama3Model(config=Llama3CKKSConfig())
    assert model.required_rotations(emit_kv_cache=True) == model.required_rotations(
        emit_kv_cache=True, decode_packing="pairfold128"
    )


@pytest.mark.parametrize("invalid", ("pairfold", "legacy", "", None, 1))
def test_decode_packing_rejects_invalid_values(invalid: object) -> None:
    prefill = FeatureMajorPrefillLayout()
    decode = FeatureMajorPrefillLayout(seq_len=1)
    attention = AttentionPairLayout()
    with pytest.raises(ValueError, match="decode_packing must be one of"):
        validate_decode_packing(invalid)
    with pytest.raises(ValueError, match="decode_packing must be one of"):
        required_rotations(prefill, attention, decode_packing=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decode_packing must be one of"):
        decode_layer_rotations(
            decode, attention, decode_packing=invalid  # type: ignore[arg-type]
        )


def test_pairfold_prefill_adds_converter_to_raw_c_emit_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama3fhe.operators.attention.pairfold import pairfold_prefill

    sentinel = 987_654_321
    monkeypatch.setattr(
        pairfold_prefill,
        "prefill_c_to_pairfold_rotations",
        lambda _layout: (sentinel,),
    )
    feature = FeatureMajorPrefillLayout()
    attention = AttentionPairLayout()
    raw_c_emit = set(
        required_rotations(
            feature,
            attention,
            emit_kv_cache=True,
            decode_packing="legacy_c",
        )
    )
    pairfold_emit = set(
        required_rotations(
            feature,
            attention,
            emit_kv_cache=True,
            decode_packing="pairfold128",
        )
    )

    assert sentinel not in raw_c_emit
    assert pairfold_emit == raw_c_emit | {sentinel}
    assert sentinel not in required_rotations(
        feature,
        attention,
        emit_kv_cache=False,
        decode_packing="pairfold128",
    )


def test_pairfold_decode_uses_native_union_without_legacy_adapter_or_output_c() -> None:
    feature = FeatureMajorPrefillLayout(seq_len=1)
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    schedule = LayerSchedule()
    qkv_config = _linear_config(
        schedule,
        max_plaintext_rows=schedule.qkv_max_plaintext_rows,
        hoist_strategy=schedule.qkv_hoist_strategy,
    )
    output_config = _linear_config(
        schedule,
        max_plaintext_rows=schedule.output_max_plaintext_rows,
        hoist_strategy=schedule.output_hoist_strategy,
    )
    common = set(
        decode_layer_rotations(
            feature,
            attention,
            schedule=schedule,
            skip_attention=True,
            decode_packing="pairfold128",
        )
    )
    route = set(
        pairfold_route_rotations(
            pairfold,
            token_columns=None,
            max_baby_offsets=int(schedule.sparse_max_baby_offsets),
        )
    )
    expected = common.copy()
    expected.update(
        pairfold_qkv_rotations(pairfold, operator_config=qkv_config)
    )
    expected.update(route)
    expected.update(pairfold_decode_rotations(pairfold))
    expected.update(
        pairfold_w_o_rotations(pairfold, operator_config=output_config)
    )
    actual = set(
        decode_layer_rotations(
            feature,
            attention,
            schedule=schedule,
            skip_attention=False,
            decode_packing="pairfold128",
        )
    )
    assert actual == expected

    legacy = set(
        decode_layer_rotations(
            feature,
            attention,
            schedule=schedule,
            skip_attention=False,
            decode_packing="legacy_c",
        )
    )
    carrier = LinearCarrierLayout(
        seq_len=1,
        dimension=feature.hidden_dim,
        slots=feature.slots,
    )
    legacy_adapter_output = set(
        decode_adapter_rotations(carrier, attention)
    ) | set(decode_output_c_rotations(attention, feature))
    legacy_only = legacy_adapter_output - expected
    pairfold_only = route - legacy
    assert legacy_only
    assert legacy_only <= legacy
    assert legacy_only.isdisjoint(actual)
    assert pairfold_only
    assert pairfold_only <= actual


class _Owned:
    def __init__(self) -> None:
        self.release_count = 0

    def release(self) -> None:
        if self.release_count:
            raise AssertionError("owned handle released more than once")
        self.release_count += 1


def test_layer_result_releases_pairfold_cache_exactly_once() -> None:
    output = _Owned()
    legacy_key = _Owned()
    legacy_value = _Owned()
    page_handles = tuple(_Owned() for _ in range(4))
    page = PairFoldCipherPage(
        key_groups=page_handles[:2],
        value_groups=page_handles[2:],
        occupied=1,
    )
    cache = PairFoldLayerCache(pages=(page,), content_tokens=1)
    result = LayerResult(
        ciphers=(output,),
        wall_seconds=0.0,
        stage_seconds={},
        nested_stage_seconds={},
        softmax_bootstrap_calls=0,
        residual_bootstrap_calls=0,
        softmax_native_bootstrap_calls=0,
        residual_native_bootstrap_calls=0,
        output_levels=(),
        key_c_ciphers=(legacy_key,),
        value_c_ciphers=(legacy_value,),
        pairfold_kv_cache=cache,
    )

    result.release()
    result.release()

    assert result.ciphers == ()
    assert result.key_c_ciphers == ()
    assert result.value_c_ciphers == ()
    assert result.pairfold_kv_cache is None
    assert cache.is_released
    assert output.release_count == 1
    assert legacy_key.release_count == 1
    assert legacy_value.release_count == 1
    assert all(handle.release_count == 1 for handle in page_handles)
