from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.operators.attention.prefill import pv, qk
from llama3fhe.operators.attention.config import AttentionOperatorConfig


class _State:
    def __init__(self, cur_limbs: int = 9) -> None:
        self.cur_limbs = int(cur_limbs)

    def replace(self, **changes):
        return _State(changes.get("cur_limbs", self.cur_limbs))


class _Cipher:
    def __init__(self, name: str, *, cur_limbs: int = 9) -> None:
        self.name = str(name)
        self.state = _State(cur_limbs)

    def release(self) -> None:
        pass


def _layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4,
        head_dim=4,
        query_heads=4,
        key_value_heads=2,
        slots=64,
    )


def _install_kernel_fakes(monkeypatch, module):
    rotation_refs: list[weakref.ReferenceType[_Cipher]] = []
    folded_query_source_refs: list[weakref.ReferenceType[_Cipher]] = []
    folded_key_refs: list[weakref.ReferenceType[_Cipher]] = []
    check_before_first_rotation = None

    def align_owned(cipher, target_state, *, crypto_context):
        del crypto_context
        aligned = _Cipher(f"aligned.{cipher.name}", cur_limbs=target_state.cur_limbs)
        if cipher.name.startswith("query") and target_state.cur_limbs == 9:
            folded_query_source_refs.append(weakref.ref(aligned))
        return aligned

    def fast_rotations(cipher, offsets, *, crypto_context):
        nonlocal check_before_first_rotation
        del cipher, crypto_context
        if check_before_first_rotation is not None:
            callback, check_before_first_rotation = check_before_first_rotation, None
            callback()
        values = tuple(_Cipher(f"rotation.{offset}") for offset in offsets)
        rotation_refs.extend(weakref.ref(value) for value in values)
        return values

    def homo_add(left, right, context):
        del left, context
        output = _Cipher("add")
        if right.name == "negative_imaginary":
            folded_key_refs.append(weakref.ref(output))
        return output

    context = SimpleNamespace(
        context=SimpleNamespace(L=12, M=128),
        fhe=SimpleNamespace(
            homo_add=homo_add,
            homo_rotate=lambda cipher, offset, context: _Cipher("rotated"),
        ),
        max_slots=64,
    )
    fake_fhe = SimpleNamespace(
        homo_mul_no_relin=lambda left, right, context: _Cipher("product"),
        homo_mul_i=lambda cipher, context, negative=False: _Cipher(
            "negative_imaginary" if negative else "imaginary"
        ),
    )

    def finalize(accumulators, *, crypto_context, extract_real):
        del crypto_context, extract_real
        gc.collect()
        assert rotation_refs
        assert all(reference() is None for reference in rotation_refs)
        outputs = tuple(_Cipher("output") for _ in accumulators)
        accumulators[:] = [None] * len(accumulators)
        return outputs

    monkeypatch.setattr(module, "fhe", fake_fhe)
    monkeypatch.setattr(module, "_raw_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "_align_owned", align_owned)
    monkeypatch.setattr(module, "_bundle_plain", lambda *args, **kwargs: _Cipher("plain"))
    monkeypatch.setattr(module, "_fast_rotations", fast_rotations)
    monkeypatch.setattr(module, "_two_masked_rotations", lambda *args, **kwargs: _Cipher("shifted"))
    monkeypatch.setattr(module, "_add_owned", lambda *args, **kwargs: _Cipher("sum"))
    monkeypatch.setattr(module, "_finalize_triplet_accumulators", finalize)
    if module is qk:
        monkeypatch.setattr(module, "_mask_rescale", lambda *args, **kwargs: _Cipher("scaled"))
        monkeypatch.setattr(
            module,
            "_two_rotation_masked_shift",
            lambda *args, **kwargs: _Cipher("second_half"),
        )

    def set_first_rotation_check(callback) -> None:
        nonlocal check_before_first_rotation
        check_before_first_rotation = callback

    return (
        context,
        folded_query_source_refs,
        folded_key_refs,
        set_first_rotation_check,
    )


def test_qk_folded_temporaries_and_rotation_chunks_drop_last_references(
    monkeypatch,
) -> None:
    context, query_source_refs, folded_key_refs, set_rotation_check = (
        _install_kernel_fakes(monkeypatch, qk)
    )

    def check_folded_temporaries() -> None:
        gc.collect()
        assert query_source_refs and folded_key_refs
        assert all(reference() is None for reference in query_source_refs)
        assert all(reference() is None for reference in folded_key_refs)

    set_rotation_check(check_folded_temporaries)
    outputs = qk.qk_delta_fhe(
        tuple(_Cipher(f"query.{index}") for index in range(2)),
        (_Cipher("key.0"),),
        layout=_layout(),
        crypto_context=context,
        operator_config=AttentionOperatorConfig(
            rotation_chunk_size=2,
            qk_output_level=None,
            qk_complex_feature_folding=True,
            qk_key_features_pre_folded=False,
            qk_query_features_pre_scaled=False,
            qk_query_features_pre_folded=False,
        ),
    )

    assert len(outputs) == 2


@pytest.mark.parametrize("complex_fold", [False, True])
def test_pv_rotation_chunks_drop_last_references(monkeypatch, complex_fold) -> None:
    context, _, _, _ = _install_kernel_fakes(monkeypatch, pv)
    outputs = pv.pv_delta_fhe(
        tuple(_Cipher(f"probability.{index}") for index in range(2)),
        (_Cipher("value.0"),),
        layout=_layout(),
        crypto_context=context,
        operator_config=AttentionOperatorConfig(
            rotation_chunk_size=2,
            pv_complex_delta_folding=complex_fold,
        ),
        inputs_pre_folded=complex_fold,
    )

    assert len(outputs) == 2
