"""CPU-only graph ownership tests for PairFold128 Decode integration."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe import graph
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.graph import evaluate_decode
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.model import TransformerLayer
from llama3fhe.operators.pairfold_kv_cache import (
    PairFoldCipherPage,
    PairFoldLayerCache,
    PairFoldTokenWrite,
)
from llama3fhe.schedule import LayerSchedule


class _Cipher:
    instances: list[_Cipher] = []

    def __init__(self, label: str, *, limbs: int = 12) -> None:
        self.label = str(label)
        self.slots = 32768
        self.state = SimpleNamespace(
            cur_limbs=int(limbs), scale_degree=1, scaling_factor=1.0
        )
        self.is_ext = False
        self.release_count = 0
        type(self).instances.append(self)

    def deep_copy(self):
        return _Cipher(f"{self.label}.clone", limbs=self.state.cur_limbs)

    def release(self) -> None:
        if self.release_count:
            raise AssertionError(f"{self.label} released more than once")
        self.release_count += 1


class _StageResult:
    def __init__(self, label: str) -> None:
        self.ciphers = (_Cipher(label),)
        self.stage_seconds = {"fake": 0.0}
        self.output_levels = (0,)
        self.bootstrap_output_levels = ()
        self.level_schedule = {}
        self.deferred_output_scale = None

    def release(self) -> None:
        owned = self.ciphers
        self.ciphers = ()
        for cipher in owned:
            cipher.release()


class _QKVResult:
    def __init__(self) -> None:
        self.carrier_cipher = _Cipher("carrier")
        self.output_layout = SimpleNamespace(
            seq_len=1, dimension=4096, slots=32768
        )
        self.stage_seconds = {"projection": 0.01}
        self.output_levels = (1,)

    def release(self) -> None:
        cipher = self.carrier_cipher
        self.carrier_cipher = None
        if cipher is not None:
            cipher.release()


class _AttentionResult:
    def __init__(self) -> None:
        self.folded_o_cipher = _Cipher("folded_o")
        self.wall_seconds = 0.1
        self.stage_seconds = {
            "q_extract_and_broadcast": 0.01,
            "qk_prefix": 0.02,
            "qk_content": 0.03,
            "qk": 0.05,
            "softmax": 0.04,
            "probability_refresh": 0.01,
            "pv_prefix": 0.02,
            "pv_content": 0.03,
            "pv": 0.06,
            "folded_o_assemble": 0.01,
            "wall": 0.1,
        }
        self.profile = {"fake": 1.0}

    def release(self) -> None:
        cipher = self.folded_o_cipher
        self.folded_o_cipher = None
        if cipher is not None:
            cipher.release()


class _FakeFHE:
    def homo_add(self, left, right, _context):
        return _Cipher(f"add({left.label},{right.label})")


class _Context:
    def __init__(self) -> None:
        self.max_slots = 32768
        self.device = "cpu"
        self.context = SimpleNamespace(L=20, M=131072)
        self.fhe = _FakeFHE()

    @staticmethod
    def level_for_cipher(_cipher) -> int:
        return 0


class _Bootstrap:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.native_calls = 0
        self.fail = bool(fail)

    def refresh(self, _cipher, **_kwargs):
        self.calls += 1
        self.native_calls += 1
        if self.fail:
            raise RuntimeError("injected refresh failure")
        return _Cipher("refreshed_o", limbs=12)


class _ReturningBootstrap:
    def __init__(self, value) -> None:
        self.calls = 0
        self.native_calls = 0
        self.value = value

    def refresh(self, _cipher, **_kwargs):
        self.calls += 1
        self.native_calls += 1
        return self.value


def _weights():
    return SimpleNamespace(
        input_layernorm=None,
        q_proj=None,
        k_proj=None,
        v_proj=None,
        o_proj=None,
        post_attention_layernorm=None,
        gate_proj=None,
        up_proj=None,
        down_proj=None,
    )


def _approximations():
    return SimpleNamespace(
        input_norm={},
        post_attention_norm={},
        silu={"coefficients": [1.0]},
        softmax=object(),
    )


def _cache(occupied: int) -> PairFoldLayerCache:
    pages = []
    remaining = int(occupied)
    page_index = 0
    while remaining:
        count = min(128, remaining)
        handles = tuple(
            _Cipher(f"cache{page_index}.{index}") for index in range(4)
        )
        pages.append(
            PairFoldCipherPage(
                key_groups=handles[:2],
                value_groups=handles[2:],
                occupied=count,
            )
        )
        remaining -= count
        page_index += 1
    return PairFoldLayerCache(pages=tuple(pages), content_tokens=occupied)


def _patch_pairfold_graph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_stage: str | None = None,
    capture: dict[str, object] | None = None,
) -> None:
    from llama3fhe.operators import pairfold_kv_cache
    from llama3fhe.operators.attention.pairfold import pairfold_operator, pairfold_route
    from llama3fhe.operators.linear.pairfold import pairfold_output, pairfold_qkv

    capture = {} if capture is None else capture
    rmsnorm_calls = 0
    residual_calls = 0

    def fake_rmsnorm(*_args, **_kwargs):
        nonlocal rmsnorm_calls
        rmsnorm_calls += 1
        if fail_stage == "post_norm" and rmsnorm_calls == 2:
            raise RuntimeError("injected post_norm failure")
        return _StageResult(f"rmsnorm{rmsnorm_calls}")

    def fake_residual(*_args, **_kwargs):
        nonlocal residual_calls
        residual_calls += 1
        if fail_stage == "residual" and residual_calls == 1:
            raise RuntimeError("injected residual failure")
        return _StageResult(f"residual{residual_calls}")

    def fake_mlp(*_args, **_kwargs):
        if fail_stage == "mlp":
            raise RuntimeError("injected mlp failure")
        return _StageResult("mlp")

    def fake_qkv(*_args, **_kwargs):
        if fail_stage == "qkv":
            raise RuntimeError("injected qkv failure")
        return _QKVResult()

    def fake_kv_route(_carrier, *, token_column, profile, **_kwargs):
        if fail_stage == "kv_extract":
            raise RuntimeError("injected kv_extract failure")
        profile.update({"wall": 0.01, "token_column": float(token_column)})
        handles = tuple(_Cipher(f"token.{index}") for index in range(4))
        return PairFoldTokenWrite(
            key_groups=handles[:2],
            value_groups=handles[2:],
            column=int(token_column),
        )

    def fake_query_route(
        _carrier, *, group, query_offset, profile, **_kwargs
    ):
        if fail_stage == "query_extract":
            raise RuntimeError("injected query_extract failure")
        profile.update(
            {"group": float(group), "query_offset": float(query_offset)}
        )
        return _Cipher(f"query.{group}.{query_offset}")

    def fake_attention(
        carrier,
        key_pages,
        value_pages,
        *,
        query_extractor,
        occupied_pages,
        prefix_bundle,
        **_kwargs,
    ):
        capture["carrier"] = carrier
        capture["key_pages"] = tuple(tuple(page) for page in key_pages)
        capture["value_pages"] = tuple(tuple(page) for page in value_pages)
        capture["occupied_pages"] = tuple(occupied_pages)
        capture["prefix_bundle"] = prefix_bundle
        for group in range(2):
            for query_offset in range(4):
                query = query_extractor(
                    carrier, group=group, query_offset=query_offset
                )
                query.release()
        if fail_stage == "attention":
            raise RuntimeError("injected attention failure")
        return _AttentionResult()

    def fake_w_o_needs_refresh(*_args, **_kwargs):
        return fail_stage == "refresh"

    def fake_w_o(*_args, profile, **_kwargs):
        if fail_stage == "folded_w_o":
            raise RuntimeError("injected folded_w_o failure")
        profile["projection"] = 0.01
        return (_Cipher("w_o"),)

    def fake_drop(token, **_kwargs):
        if fail_stage == "kv_limb_drop":
            raise RuntimeError("injected kv_limb_drop failure")
        return token.move()

    monkeypatch.setattr(graph.decode, "rmsnorm_fhe", fake_rmsnorm)
    monkeypatch.setattr(graph.decode, "residual_add", fake_residual)
    monkeypatch.setattr(graph.decode, "mlp_fhe", fake_mlp)
    monkeypatch.setattr(graph.decode, "trim_device_allocator_cache", lambda _device: None)
    monkeypatch.setattr(pairfold_qkv, "pairfold_qkv_fhe", fake_qkv)
    monkeypatch.setattr(
        pairfold_route, "pairfold_kv_from_carrier_fhe", fake_kv_route
    )
    monkeypatch.setattr(
        pairfold_route, "pairfold_query_from_carrier_fhe", fake_query_route
    )
    monkeypatch.setattr(pairfold_operator, "pairfold_attention_fhe", fake_attention)
    monkeypatch.setattr(
        pairfold_output,
        "pairfold_w_o_needs_refresh",
        fake_w_o_needs_refresh,
    )
    monkeypatch.setattr(pairfold_output, "pairfold_w_o_fhe", fake_w_o)
    monkeypatch.setattr(
        pairfold_kv_cache, "drop_pairfold_token_to_limbs", fake_drop
    )
    if fail_stage == "kv_append":
        monkeypatch.setattr(
            PairFoldLayerCache,
            "append",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected kv_append failure")
            ),
        )


def _run_pairfold(
    *,
    cache: PairFoldLayerCache | None = None,
    occupied: int = 0,
    consume: bool = False,
    schedule: LayerSchedule | None = None,
    prefix_bundle=None,
    bootstrap_operator=None,
    position: int | None = None,
):
    feature = FeatureMajorPrefillLayout(seq_len=1)
    attention = AttentionPairLayout()
    input_cipher = _Cipher("input")
    prefix_tokens = (
        0 if prefix_bundle is None else int(prefix_bundle.prefix_k)
    )
    result = evaluate_decode(
        np.zeros((1, 4096), dtype=np.float32),
        layer_idx=0,
        weights=_weights(),
        approximations=_approximations(),
        feature_layout=feature,
        attention_layout=attention,
        config=Llama3CKKSConfig(),
        crypto_context=_Context(),
        input_ciphers=(input_cipher,),
        bootstrap_operator=bootstrap_operator,
        schedule=LayerSchedule() if schedule is None else schedule,
        skip_attention=False,
        pairfold_kv_cache=cache,
        occupied=int(occupied),
        position=(
            int(occupied) + prefix_tokens
            if position is None
            else int(position)
        ),
        max_kv_pages=3,
        prefix_bundle=prefix_bundle,
        consume_kv_cache=bool(consume),
        decode_packing="pairfold128",
    )
    return result, input_cipher


@pytest.mark.parametrize(
    ("occupied", "expected_pages", "expected_tail"),
    ((0, 1, 1), (1, 1, 2), (128, 2, 1)),
)
def test_pairfold_decode_appends_first_partial_and_new_page(
    monkeypatch: pytest.MonkeyPatch,
    occupied: int,
    expected_pages: int,
    expected_tail: int,
) -> None:
    _Cipher.instances = []
    capture: dict[str, object] = {}
    _patch_pairfold_graph(monkeypatch, capture=capture)
    source = None if occupied == 0 else _cache(occupied)
    source_handles = () if source is None else tuple(
        handle for page in source.pages for handle in page.handles
    )
    prefix = SimpleNamespace(
        prefix_k=3,
        key=np.zeros((3, 8, 128), dtype=np.float32),
        value=np.zeros((3, 8, 128), dtype=np.float32),
    )

    result, input_cipher = _run_pairfold(
        cache=source,
        occupied=occupied,
        consume=False,
        prefix_bundle=prefix,
    )
    returned = result.pairfold_kv_cache
    assert returned is not None
    assert returned is not source
    assert returned.content_tokens == occupied + 1
    assert len(returned.pages) == expected_pages
    assert returned.pages[-1].occupied == expected_tail
    assert capture["occupied_pages"] == tuple(
        page.occupied for page in returned.pages
    )
    assert capture["prefix_bundle"] is prefix
    assert result.key_c_ciphers == ()
    assert result.value_c_ciphers == ()
    for key in (
        "qkv_projection",
        "kv_extract",
        "kv_append",
        "q_extract_and_broadcast",
        "qk_prefix",
        "qk_content",
        "softmax",
        "pv_prefix",
        "pv_content",
        "folded_o_assemble",
        "folded_w_o",
        "attention_path_total",
        "attention_residual",
        "attention_residual_add",
    ):
        assert key in result.stage_seconds
    assert (
        result.stage_seconds["attention_residual"]
        == result.stage_seconds["attention_residual_add"]
    )
    assert result.stage_seconds["attention_path_total"] >= (
        result.stage_seconds["attention_residual"]
    )
    assert input_cipher.release_count == 0
    assert all(handle.release_count == 0 for handle in source_handles)

    result.release()
    assert input_cipher.release_count == 0
    assert all(handle.release_count == 0 for handle in source_handles)
    if source is not None:
        source.release()
    assert all(cipher.release_count == 1 for cipher in _Cipher.instances if cipher is not input_cipher)


def test_pairfold_decode_bootstrap_categories_are_disjoint_and_sum_to_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama3fhe.operators import pairfold_kv_cache
    from llama3fhe.operators.attention.pairfold import pairfold_operator
    from llama3fhe.operators.linear.pairfold import pairfold_output

    class CategorizedBootstrap:
        def __init__(self) -> None:
            # Nonzero entry values prove that every reported value is a delta.
            self.calls = 10
            self.native_calls = 20

        def bump(self, calls: int, native_calls: int) -> None:
            self.calls += int(calls)
            self.native_calls += int(native_calls)

        def refresh(self, _cipher, **_kwargs):
            self.bump(1, 2)
            return _Cipher("refreshed_o", limbs=12)

    _Cipher.instances = []
    _patch_pairfold_graph(monkeypatch)
    bootstrap = CategorizedBootstrap()

    captured_drop: dict[str, object] = {}

    def fake_drop(token, *, bootstrap_operator, **kwargs):
        captured_drop.update(kwargs)
        bootstrap_operator.bump(2, 3)
        return token.move()

    def fake_attention(*_args, bootstrap_operator, **_kwargs):
        bootstrap_operator.bump(5, 7)
        return _AttentionResult()

    def fake_bootstrap_rmsnorm(*_args, bootstrap_operator, **_kwargs):
        bootstrap_operator.bump(4, 6)
        return _StageResult("post_norm")

    monkeypatch.setattr(
        pairfold_kv_cache, "drop_pairfold_token_to_limbs", fake_drop
    )
    monkeypatch.setattr(
        pairfold_operator, "pairfold_attention_fhe", fake_attention
    )
    monkeypatch.setattr(
        pairfold_output,
        "pairfold_w_o_needs_refresh",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        graph.decode, "bootstrap_rmsnorm_fhe", fake_bootstrap_rmsnorm
    )
    monkeypatch.setattr(
        graph.decode,
        "drop_levels",
        lambda ciphers, _levels, **_kwargs: tuple(
            cipher.deep_copy() for cipher in ciphers
        ),
    )

    result, input_cipher = _run_pairfold(
        schedule=replace(LayerSchedule(), kv_limbs=12, kv_value_limbs=8),
        bootstrap_operator=bootstrap,
    )

    assert captured_drop["target_limbs"] == 12
    assert captured_drop["value_target_limbs"] == 8
    categories = result.nested_stage_seconds["bootstrap_calls"]
    assert categories == {
        "kv_limb_drop": {"calls": 2, "native_calls": 3},
        "attention_softmax": {"calls": 5, "native_calls": 7},
        "folded_o_refresh": {"calls": 1, "native_calls": 2},
        "other_residual": {"calls": 4, "native_calls": 6},
        "total": {"calls": 12, "native_calls": 18},
    }
    detail = tuple(
        value for name, value in categories.items() if name != "total"
    )
    assert sum(value["calls"] for value in detail) == categories["total"][
        "calls"
    ]
    assert sum(value["native_calls"] for value in detail) == categories[
        "total"
    ]["native_calls"]
    assert result.softmax_bootstrap_calls == 5
    assert result.softmax_native_bootstrap_calls == 7
    assert result.residual_bootstrap_calls == 7
    assert result.residual_native_bootstrap_calls == 11

    result.release()
    assert input_cipher.release_count == 0


def test_legacy_decode_residual_stage_matches_pairfold_residual_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama3fhe.operators import kv_cache
    from llama3fhe.operators.attention.decode import decode, decode_adapter
    from llama3fhe.operators.linear.prefill import output

    class LegacyQKVResult:
        def __init__(self) -> None:
            self.carrier_ciphers = (_Cipher("legacy.carrier"),)
            self.output_layout = object()
            self.stage_seconds = {"projection": 0.01}
            self.output_levels = (1,)

        def release(self) -> None:
            owned = self.carrier_ciphers
            self.carrier_ciphers = ()
            for cipher in owned:
                cipher.release()

    class LegacyAdapterResult:
        def __init__(self) -> None:
            self.query_ciphers = (_Cipher("legacy.query"),)
            self.key_ciphers = (_Cipher("legacy.key"),)
            self.value_ciphers = (_Cipher("legacy.value"),)
            self.stage_seconds = {"route": 0.01}
            self.route_stats = {}

        def release(self) -> None:
            owned = (
                *self.query_ciphers,
                *self.key_ciphers,
                *self.value_ciphers,
            )
            self.query_ciphers = ()
            self.key_ciphers = ()
            self.value_ciphers = ()
            for cipher in owned:
                cipher.release()

    class LegacyAttentionResult:
        def __init__(self) -> None:
            self.output_ciphers = (_Cipher("legacy.output_c"),)
            self.stage_seconds = {"softmax": 0.01}

        def release(self) -> None:
            owned = self.output_ciphers
            self.output_ciphers = ()
            for cipher in owned:
                cipher.release()

    _Cipher.instances = []
    _patch_pairfold_graph(monkeypatch)
    monkeypatch.setattr(graph.decode, "qkv_fhe", lambda *_a, **_k: LegacyQKVResult())
    monkeypatch.setattr(
        decode_adapter,
        "decode_prepare_qkv_fhe",
        lambda *_a, **_k: LegacyAdapterResult(),
    )
    monkeypatch.setattr(
        kv_cache,
        "extend_kv_c_pages",
        lambda _old_k, _old_v, new_k, new_v, **_kwargs: (
            (tuple(new_k),),
            (tuple(new_v),),
        ),
    )
    monkeypatch.setattr(
        kv_cache,
        "flatten_c_pages",
        lambda pages: tuple(cipher for page in pages for cipher in page),
    )
    monkeypatch.setattr(
        decode,
        "decode_attention_pages_fhe",
        lambda *_a, **_k: LegacyAttentionResult(),
    )
    monkeypatch.setattr(
        decode,
        "output_c_to_decode_fm_fhe",
        lambda *_a, **_k: (_Cipher("legacy.fm"),),
    )
    monkeypatch.setattr(
        output,
        "decode_w_o_fhe",
        lambda *_a, **_k: (_Cipher("legacy.w_o"),),
    )

    input_cipher = _Cipher("input")
    result = evaluate_decode(
        np.zeros((1, 4096), dtype=np.float32),
        layer_idx=0,
        weights=_weights(),
        approximations=_approximations(),
        feature_layout=FeatureMajorPrefillLayout(seq_len=1),
        attention_layout=AttentionPairLayout(),
        config=Llama3CKKSConfig(),
        crypto_context=_Context(),
        input_ciphers=(input_cipher,),
        skip_attention=False,
        occupied=0,
        decode_packing="legacy_c",
    )

    assert (
        result.stage_seconds["attention_residual"]
        == result.stage_seconds["attention_residual_add"]
    )
    assert result.stage_seconds["attention_output_total"] >= (
        result.stage_seconds["attention_residual"]
    )
    assert "attention_path_total" not in result.stage_seconds

    result.release()
    assert input_cipher.release_count == 0


@pytest.mark.parametrize("consume", (False, True))
def test_pairfold_decode_cache_clone_vs_move(
    monkeypatch: pytest.MonkeyPatch, consume: bool
) -> None:
    _Cipher.instances = []
    _patch_pairfold_graph(monkeypatch)
    source = _cache(1)
    source_handles = tuple(source.pages[0].handles)

    result, input_cipher = _run_pairfold(
        cache=source, occupied=1, consume=consume
    )
    if consume:
        assert result.pairfold_kv_cache is source
        assert all(handle.release_count == 1 for handle in source_handles)
    else:
        assert result.pairfold_kv_cache is not source
        assert source.content_tokens == 1
        assert source.pages[0].occupied == 1
        assert all(handle.release_count == 0 for handle in source_handles)

    result.release()
    if not consume:
        source.release()
    assert input_cipher.release_count == 0
    assert all(cipher.release_count == 1 for cipher in _Cipher.instances if cipher is not input_cipher)


@pytest.mark.parametrize(
    "fail_stage",
    (
        "qkv",
        "kv_extract",
        "kv_limb_drop",
        "kv_append",
        "query_extract",
        "attention",
        "refresh",
        "folded_w_o",
        "residual",
        "post_norm",
        "mlp",
    ),
)
@pytest.mark.parametrize("consume", (False, True))
def test_pairfold_decode_stage_failure_cleans_every_owned_cipher_once(
    monkeypatch: pytest.MonkeyPatch,
    fail_stage: str,
    consume: bool,
) -> None:
    _Cipher.instances = []
    _patch_pairfold_graph(monkeypatch, fail_stage=fail_stage)
    source = _cache(1)
    source_handles = tuple(source.pages[0].handles)
    schedule = LayerSchedule()
    bootstrap = None
    if fail_stage == "kv_limb_drop":
        schedule = replace(schedule, kv_limbs=6)
    if fail_stage == "refresh":
        bootstrap = _Bootstrap(fail=True)
    input_cipher = None

    with pytest.raises(RuntimeError, match="injected"):
        try:
            _result, input_cipher = _run_pairfold(
                cache=source,
                occupied=1,
                consume=consume,
                schedule=schedule,
                bootstrap_operator=bootstrap,
            )
        finally:
            if input_cipher is None:
                input_cipher = next(
                    cipher
                    for cipher in _Cipher.instances
                    if cipher.label == "input"
                )

    assert input_cipher.release_count == 0
    borrowed = {id(input_cipher)}
    if not consume:
        borrowed.update(id(handle) for handle in source_handles)
        assert not source.is_released
        source.release()
    else:
        assert source.is_released
    for cipher in _Cipher.instances:
        if id(cipher) in borrowed:
            continue
        assert cipher.release_count == 1, cipher.label


def test_pairfold_decode_preflight_rejects_mixed_or_mismatched_caches() -> None:
    feature = FeatureMajorPrefillLayout(seq_len=1)
    attention = AttentionPairLayout()
    common = dict(
        hidden_states=np.zeros((1, 4096), dtype=np.float32),
        layer_idx=0,
        weights=_weights(),
        approximations=_approximations(),
        feature_layout=feature,
        attention_layout=attention,
        config=Llama3CKKSConfig(),
        crypto_context=_Context(),
        input_ciphers=(),
        skip_attention=False,
    )
    cache = _cache(1)
    full_cache = _cache(256)
    try:
        with pytest.raises(ValueError, match="legacy K/V"):
            evaluate_decode(
                **common,
                decode_packing="pairfold128",
                kv_cache_k_ciphers=(),
            )
        with pytest.raises(ValueError, match="occupancy mismatch"):
            evaluate_decode(
                **common,
                decode_packing="pairfold128",
                pairfold_kv_cache=cache,
                occupied=0,
            )
        with pytest.raises(ValueError, match="does not accept pairfold"):
            evaluate_decode(
                **common,
                decode_packing="legacy_c",
                pairfold_kv_cache=cache,
                occupied=1,
            )
        with pytest.raises(ValueError, match="pages are full"):
            evaluate_decode(
                **common,
                decode_packing="pairfold128",
                pairfold_kv_cache=full_cache,
                occupied=256,
                max_kv_pages=2,
            )
        with pytest.raises(ValueError, match="requires a K limb target"):
            evaluate_decode(
                **common,
                decode_packing="pairfold128",
                schedule=LayerSchedule(kv_value_limbs=8),
            )
        # Skip-attention preserves legacy's external occupied semantics and
        # therefore reaches the ordinary input validation without a cache.
        skip_common = dict(common)
        skip_common["skip_attention"] = True
        with pytest.raises(ValueError, match="expects 1 DecodeFM"):
            evaluate_decode(
                **skip_common,
                decode_packing="pairfold128",
                occupied=7,
            )
        assert not cache.is_released
        assert not full_cache.is_released
    finally:
        cache.release()
        full_cache.release()


def test_pairfold_decode_preflight_rejects_noncontiguous_position() -> None:
    source = _cache(2)
    source_handles = tuple(source.handles)
    try:
        with pytest.raises(ValueError, match="position must be contiguous"):
            _run_pairfold(cache=source, occupied=2, position=3)
        assert not source.is_released
        assert all(handle.release_count == 0 for handle in source_handles)
    finally:
        source.release()


def test_pairfold_decode_preflight_uses_logical_prefix_occupancy() -> None:
    prefix = SimpleNamespace(
        prefix_k=2,
        key=np.zeros((3, 8, 128), dtype=np.float32),
        value=np.zeros((3, 8, 128), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="logical payload occupancy"):
        _run_pairfold(prefix_bundle=prefix, position=2)


@pytest.mark.parametrize("refresh_kind", ("none", "borrowed_cache"))
def test_pairfold_decode_rejects_invalid_refresh_output_without_releasing_borrowed(
    monkeypatch: pytest.MonkeyPatch, refresh_kind: str
) -> None:
    _Cipher.instances = []
    _patch_pairfold_graph(monkeypatch, fail_stage="refresh")
    source = _cache(1)
    source_handles = tuple(source.pages[0].handles)
    refresh_value = None if refresh_kind == "none" else source_handles[0]
    bootstrap = _ReturningBootstrap(refresh_value)

    expected = "returned no ciphertext" if refresh_kind == "none" else "may not alias"
    with pytest.raises(RuntimeError, match=expected):
        _run_pairfold(
            cache=source,
            occupied=1,
            consume=False,
            bootstrap_operator=bootstrap,
        )

    assert not source.is_released
    assert all(handle.release_count == 0 for handle in source_handles)
    source.release()
    input_cipher = next(
        cipher for cipher in _Cipher.instances if cipher.label == "input"
    )
    assert input_cipher.release_count == 0
    assert all(
        cipher.release_count == 1
        for cipher in _Cipher.instances
        if cipher is not input_cipher
    )


def test_evaluate_decode_explicit_legacy_reaches_legacy_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama3fhe.operators.linear.pairfold import pairfold_qkv

    _Cipher.instances = []
    monkeypatch.setattr(graph.decode, "rmsnorm_fhe", lambda *_a, **_k: _StageResult("rmsn"))
    monkeypatch.setattr(
        pairfold_qkv,
        "pairfold_qkv_fhe",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("PairFold QKV must not run on the default path")
        ),
    )
    monkeypatch.setattr(
        graph.decode,
        "qkv_fhe",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("legacy qkv reached")
        ),
    )
    input_cipher = _Cipher("input")
    with pytest.raises(RuntimeError, match="legacy qkv reached"):
        evaluate_decode(
            np.zeros((1, 4096), dtype=np.float32),
            layer_idx=0,
            weights=_weights(),
            approximations=_approximations(),
            feature_layout=FeatureMajorPrefillLayout(seq_len=1),
            attention_layout=AttentionPairLayout(),
            config=Llama3CKKSConfig(),
            crypto_context=_Context(),
            input_ciphers=(input_cipher,),
            skip_attention=False,
            decode_packing="legacy_c",
        )
    assert input_cipher.release_count == 0
    assert all(
        cipher.release_count == 1
        for cipher in _Cipher.instances
        if cipher is not input_cipher
    )


def test_transformer_layer_forwards_pairfold_decode_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_evaluate_decode(*_args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(graph, "evaluate_decode", fake_evaluate_decode)
    layer = TransformerLayer(
        index=3,
        weights=_weights(),
        approximations=_approximations(),
        config=Llama3CKKSConfig(),
        feature_layout=FeatureMajorPrefillLayout(seq_len=1),
        attention_layout=AttentionPairLayout(),
    )
    cache = PairFoldLayerCache()
    result = layer.forward_decode(
        np.zeros((1, 4096), dtype=np.float32),
        crypto_context=object(),
        input_ciphers=(),
        pairfold_kv_cache=cache,
        decode_packing="pairfold128",
        consume_kv_cache=True,
    )

    assert result is sentinel
    assert captured["decode_packing"] == "pairfold128"
    assert captured["pairfold_kv_cache"] is cache
    assert captured["consume_kv_cache"] is True
