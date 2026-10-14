from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llama3fhe.application import layer_stream, session
from llama3fhe.application.approx_config import ApproximationConfig
from llama3fhe.application.checkpoint import (
    CheckpointMetadata,
    SafeTensorReader,
    WeightLoader,
    load_layer_weights,
    load_norm_weight,
    validate_runtime_checkpoint,
)
from llama3fhe.application.client import (
    ClientHead,
    decrypt_hidden,
    final_logits,
    load_input_ids,
    top_k_logits,
)
from llama3fhe.backend import EasyFHEContext
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.model import LayerWeights


def _write_safetensors(path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, (dtype_tag, values) in tensors.items():
        raw = np.asarray(values).tobytes(order="C")
        header[name] = {
            "dtype": dtype_tag,
            "shape": list(values.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def test_client_ids_feature_decode_and_top_k(tmp_path) -> None:
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps({"input_ids": [3, 4]}), encoding="utf-8")
    assert load_input_ids(ids_path, seq_len=2).tolist() == [3, 4]

    layout = FeatureMajorPrefillLayout(seq_len=2, hidden_dim=4, slots=8)
    logical = np.arange(8, dtype=np.float32).reshape(2, 4)
    packed = layout.pack(logical)

    class Context:
        @staticmethod
        def decrypt(cipher):
            return cipher

    decoded = decrypt_hidden(
        tuple(packed), crypto_context=Context(), layout=layout
    )
    np.testing.assert_array_equal(decoded, logical)
    assert [row["token_id"] for row in top_k_logits([1.0, 3.0, 2.0], 2)] == [1, 2]
    with pytest.raises(ValueError, match="positive"):
        top_k_logits([1.0], 0)


def test_client_final_logits_uses_exact_final_norm() -> None:
    metadata = CheckpointMetadata(
        hidden_size=2,
        intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        eos_token_id=None,
        pad_token_id=None,
    )

    class Loader:
        values = {
            "model.norm.weight": np.array([2.0, 3.0], dtype=np.float32),
            "model.embed_tokens.weight": np.array(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
        }

        def has_tensor(self, name):
            return name in self.values

        def get(self, name, **_kwargs):
            return self.values[name].copy()

    hidden = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    logits = final_logits(hidden, loader=Loader(), metadata=metadata)
    rms = np.sqrt((9.0 + 16.0) / 2.0 + 1e-5)
    np.testing.assert_allclose(logits, [6.0 / rms, 12.0 / rms], rtol=1e-6)
    # the resident head gives the same logits without re-reading the checkpoint per call
    head = ClientHead(Loader(), metadata)
    np.testing.assert_allclose(head.logits(hidden, token_index=-1), logits, rtol=1e-6)
    np.testing.assert_allclose(head.logits(hidden[1:], token_index=0), logits, rtol=1e-6)


def test_checkpoint_metadata_validation_is_explicit(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000,
                "eos_token_id": [128001, 128009],
            }
        ),
        encoding="utf-8",
    )
    metadata = CheckpointMetadata.from_json(path)
    metadata.validate_llama3_8b(num_layers=32)
    assert metadata.eos_token_id == (128001, 128009)
    with pytest.raises(ValueError, match="requested"):
        metadata.validate_llama3_8b(num_layers=33)

    with pytest.raises(ValueError, match="prepare_quarot_weights"):
        validate_runtime_checkpoint(tmp_path)
    (tmp_path / "export_metadata.json").write_text(
        json.dumps(
            {
                "format": "llama3_8b_feature_major_quarot_v1",
                "full_fuse": True,
                "runtime_weight_layout": "ordinary_out_in",
                "he_packing": "prepared_online_by_linear_operator",
            }
        ),
        encoding="utf-8",
    )
    assert validate_runtime_checkpoint(tmp_path)["full_fuse"] is True


def test_safetensors_reader_and_loader_are_lazy_and_close_idempotently(tmp_path) -> None:
    shard = tmp_path / "weights.safetensors"
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    _write_safetensors(shard, {"x": ("F32", values)})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard.name}}), encoding="utf-8"
    )

    loader = WeightLoader(tmp_path)
    assert loader.readers == {}
    np.testing.assert_array_equal(loader.get("x"), values)
    assert tuple(loader.readers) == (shard.name,)
    loader.close()
    loader.close()
    with pytest.raises(RuntimeError, match="closed"):
        loader.get("x")

    reader = SafeTensorReader(shard)
    np.testing.assert_array_equal(reader.get_tensor("x"), values)
    reader.close()
    reader.close()


def test_layer_weight_loader_rejects_bad_index_and_shape() -> None:
    class Loader:
        @staticmethod
        def get(_name):
            return np.zeros((1,), dtype=np.float32)

    with pytest.raises(ValueError, match="non-negative"):
        load_layer_weights(Loader(), -1)
    with pytest.raises(ValueError, match="must have shape"):
        load_layer_weights(Loader(), 0)


def test_quarot_scalar_norm_placeholder_means_unit_gamma() -> None:
    class Loader:
        @staticmethod
        def get(_name):
            return np.zeros((1,), dtype=np.float32)

    np.testing.assert_array_equal(
        load_norm_weight(Loader(), "norm.weight", 4),
        np.ones((4,), dtype=np.float32),
    )


def test_approximation_config_caches_layer_selection() -> None:
    root = Path(__file__).resolve().parents[1] / "assets/polynomials"
    approximations = ApproximationConfig(
        rmsnorm_path=root / "rmsnorm_poly_coeffs.json",
        silu_path=root / "silu_poly_coeffs.json",
        softmax_path=root / "softmax_poly_coeffs.json",
    )
    assert approximations.for_layer(0) is approximations.for_layer(0)
    assert approximations.degree_summary(0) == {
        "input_rmsnorm": 16,
        "post_attention_rmsnorm": 32,
        "silu": 48,
    }
    with pytest.raises(ValueError, match="non-negative"):
        approximations.for_layer(-1)


def test_joint_tol_v1_pack_loads_reduced_degrees() -> None:
    root = Path(__file__).resolve().parents[1] / "assets/polynomials"
    approximations = ApproximationConfig(
        rmsnorm_path=root / "rmsnorm_poly_coeffs_joint_tol_v1.json",
        silu_path=root / "silu_poly_coeffs_joint_tol_v1.json",
        softmax_path=root
        / "softmax_poly_coeffs_joint_tol_v1_j1hifloor8.json",
    )
    approximations.validate_layers(32)
    assert approximations.degree_summary(0) == {
        "input_rmsnorm": 11,
        "post_attention_rmsnorm": 10,
        "silu": 18,
    }
    last = approximations.degree_summary(31)
    assert last["silu"] == 236
    j1 = approximations.for_layer(4).softmax.poly_for_layer(
        approximations.for_layer(4).softmax.invsqrt_alg2["j1_rough"],
        4,
    )
    assert float(j1["fit_interval"][1]) >= 8.0
    l31 = approximations.for_layer(31).softmax.poly_for_layer(
        approximations.for_layer(31).softmax.invsqrt_alg2["j1_rough"],
        31,
    )
    assert "coefficients" in l31
    assert l31["fit_interval"] is not None


def test_session_plain_cache_reuses_layer_invariant_masks(monkeypatch) -> None:
    class Tensor:
        def to(self, _device):
            return self

    class TensorModule:
        @staticmethod
        def from_numpy(_values):
            return Tensor()

    class Bundle:
        def __init__(self, *, vectors, cache_mode):
            assert cache_mode in {"none", "plain"}
            self.vectors = vectors
            self.cache_mode = cache_mode
            self.calls = 0
            self.cleared = False

        def plaintext(self, *_args, **_kwargs):
            self.calls += 1
            return object()

        def cache_info(self):
            return {"total_bytes": 64 if self.calls else 0}

        def clear_cache(self):
            self.cleared = True

    fake_fhe = SimpleNamespace(PackedRaw=lambda value: value, ConstantBundle=Bundle)
    monkeypatch.setenv("LLAMA_FHE_PLAIN_CACHE_GB", "0.001")
    context = EasyFHEContext(
        tensor=TensorModule(),
        fhe=fake_fhe,
        client=None,
        context=SimpleNamespace(device="cpu"),
        max_slots=8,
    )
    values = np.arange(8, dtype=np.float64)
    first = context.constant_bundle(
        {"mask": values},
        cache_key=("mask", "model.layers.0.transpose"),
    )
    first.plaintext("mask")
    second = context.constant_bundle(
        {"mask": values.copy()},
        cache_key=("mask", "model.layers.1.transpose"),
    )
    assert second is first
    second.plaintext("mask")
    changed = context.constant_bundle(
        {"mask": values + 1},
        cache_key=("mask", "model.layers.2.transpose"),
    )
    assert changed is not first
    info = context.plaintext_cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 2
    assert info["entries"] == 2
    context.clear_plaintext_cache()
    assert first._bundle.cleared
    assert changed._bundle.cleared

    monkeypatch.setenv("LLAMA_FHE_PLAIN_CACHE_GB", "0")
    uncached_context = EasyFHEContext(
        tensor=TensorModule(),
        fhe=fake_fhe,
        client=None,
        context=SimpleNamespace(device="cpu"),
        max_slots=8,
    )
    uncached = uncached_context.constant_bundle(
        {"mask": values}, cache_key=("mask", "uncached")
    )
    assert uncached._bundle.cache_mode == "none"
    assert uncached_context.plaintext_cache_info()["entries"] == 0


def test_session_context_forwards_host_backed_key_mode_without_gpu(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_spec(**kwargs):
        captured["spec"] = SimpleNamespace(**kwargs)
        return captured["spec"]

    def fake_generate_client_context(spec, *, device):
        captured["device"] = device
        return object(), SimpleNamespace(max_slots=32768, device=device)

    monkeypatch.setattr(session.fhe, "CKKSContextSpec", fake_spec)
    monkeypatch.setattr(
        session.fhe, "generate_client_context", fake_generate_client_context
    )
    config = Llama3CKKSConfig()
    config.simulator.device = "cpu"

    context = session._create_context(
        config, (1, -1), auto_load_keys=False
    )

    assert captured["device"] == "cpu"
    assert captured["spec"].rotations == (1, -1)
    assert captured["spec"].auto_load_keys is False
    assert context.context.device == "cpu"


def test_raw_weight_double_buffer_overlaps_layer0_compute_with_layer1_load(
    monkeypatch,
) -> None:
    loaded: list[int] = []

    def slow_load(_loader, layer_idx):
        time.sleep(0.05)
        loaded.append(int(layer_idx))
        value = np.full((1,), float(layer_idx), dtype=np.float32)
        return LayerWeights(**{
            name: value for name in LayerWeights.__dataclass_fields__
        })

    monkeypatch.setattr(layer_stream, "load_layer_weights", slow_load)
    approximations = SimpleNamespace(
        for_layer=lambda layer_idx: SimpleNamespace(layer=int(layer_idx))
    )
    stream = layer_stream.DoubleBufferedLayerStream(
        loader=SimpleNamespace(),
        synthetic_weights=None,
        approx_config=approximations,
        config=Llama3CKKSConfig(),
        feature_layout=FeatureMajorPrefillLayout(),
        attention_layout=AttentionPairLayout(),
        num_layers=2,
        rope_theta=500000.0,
    )
    try:
        layer0 = next(stream)
        compute_started = time.perf_counter()
        time.sleep(0.08)
        compute_finished = time.perf_counter()
        stream.record_compute_window(0, compute_started, compute_finished)
        wait_started = time.perf_counter()
        layer1 = next(stream)
        consumer_wait = time.perf_counter() - wait_started
        assert float(layer0.weights.q_proj[0]) == 0.0
        assert float(layer1.weights.q_proj[0]) == 1.0
        assert loaded == [0, 1]
        assert consumer_wait < 0.02
        stats = stream.stats()
        layer1_stats = stats["layers"][1]
        assert layer1_stats["asynchronous"] is True
        assert layer1_stats["worker_thread"] != layer1_stats["consumer_thread"]
        assert layer1_stats["overlap_with_previous_layer_seconds"] > 0.03
    finally:
        stream.close()
