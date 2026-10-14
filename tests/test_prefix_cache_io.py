from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from llama3fhe.approx.prefix_cache_io import (
    load_prefix_kv_cache,
    model_cache_metadata,
    save_prefix_kv_cache,
)
from llama3fhe.layouts.attention import AttentionPairLayout
from tools import build_prefix_kv_cache as exporter


def _layout() -> AttentionPairLayout:
    return AttentionPairLayout(
        seq_len=4, head_dim=4, query_heads=4, key_value_heads=2, slots=32
    )


def _model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "rope_theta": 10000.0,
        }),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"checkpoint weights")
    return model_dir


def _cache(tmp_path):
    model_dir = _model_dir(tmp_path)
    rng = np.random.default_rng(7)
    keys = rng.normal(size=(2, 2, 8)).astype(np.float32)
    values = rng.normal(size=(2, 2, 8)).astype(np.float32)
    path = tmp_path / "nested" / "prefix.npz"
    save_prefix_kv_cache(
        path,
        keys,
        values,
        layers=[0, 1],
        prefix_k=2,
        prefix_token_id=198,
        rope_theta=10000.0,
        model_metadata=model_cache_metadata(model_dir),
    )
    return path, model_dir, keys, values


def test_prefix_cache_roundtrip_reconstructs_plaintext_c_pages(tmp_path) -> None:
    path, model_dir, keys, values = _cache(tmp_path)
    layout = _layout()
    bundles = load_prefix_kv_cache(
        path,
        layout=layout,
        expected_layers=[0, 1],
        expected_prefix_k=2,
        expected_prefix_token_id=198,
        expected_rope_theta=10000.0,
        expected_model_dir=model_dir,
    )
    assert len(bundles) == 2
    for idx, bundle in enumerate(bundles):
        np.testing.assert_array_equal(bundle.key, keys[idx])
        np.testing.assert_array_equal(bundle.value, values[idx])
        np.testing.assert_allclose(layout.unpack_key_c(bundle.packed_key_c)[:2], keys[idx])
        np.testing.assert_allclose(layout.unpack_value_c(bundle.packed_value_c)[:2], values[idx])


def test_prefix_cache_can_select_layers_from_full_snapshot(tmp_path) -> None:
    path, _, keys, values = _cache(tmp_path)
    bundles = load_prefix_kv_cache(path, layout=_layout(), expected_layers=[1])
    assert len(bundles) == 1
    np.testing.assert_array_equal(bundles[0].key, keys[1])
    np.testing.assert_array_equal(bundles[0].value, values[1])


@pytest.mark.parametrize(
    "override,match",
    [
        ({"expected_layers": [1, 2]}, "layer IDs"),
        ({"expected_prefix_k": 1}, "prefix_k"),
        ({"expected_prefix_token_id": 13}, "token_id"),
        ({"expected_rope_theta": 500000.0}, "rope_theta"),
    ],
)
def test_prefix_cache_rejects_incompatible_runtime(tmp_path, override, match) -> None:
    path, _, _, _ = _cache(tmp_path)
    with pytest.raises(ValueError, match=match):
        load_prefix_kv_cache(path, layout=_layout(), **override)


def test_prefix_cache_rejects_changed_checkpoint_config(tmp_path) -> None:
    path, model_dir, _, _ = _cache(tmp_path)
    config_path = model_dir / "config.json"
    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["rope_theta"] = 500000.0
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="another checkpoint"):
        load_prefix_kv_cache(path, layout=_layout(), expected_model_dir=model_dir)


def test_prefix_cache_rejects_same_size_weight_change(tmp_path) -> None:
    path, model_dir, _, _ = _cache(tmp_path)
    weights_path = model_dir / "model.safetensors"
    original = weights_path.read_bytes()
    weights_path.write_bytes(b"X" + original[1:])
    assert weights_path.stat().st_size == len(original)
    with pytest.raises(ValueError, match="another checkpoint"):
        load_prefix_kv_cache(path, layout=_layout(), expected_model_dir=model_dir)


def test_prefix_cache_hashes_every_weight_shard(tmp_path) -> None:
    model_dir = _model_dir(tmp_path)
    (model_dir / "model.safetensors").unlink()
    first = model_dir / "model-00001-of-00002.safetensors"
    second = model_dir / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first shard")
    second.write_bytes(b"second shard")
    original = model_cache_metadata(model_dir)["weights_sha256"]
    second.write_bytes(b"Second shard")
    assert model_cache_metadata(model_dir)["weights_sha256"] != original


def test_prefix_cache_hashes_shards_referenced_by_index(tmp_path) -> None:
    model_dir = _model_dir(tmp_path)
    (model_dir / "model.safetensors").unlink()
    shard_dir = model_dir / "shards"
    shard_dir.mkdir()
    shard = shard_dir / "part.safetensors"
    shard.write_bytes(b"first version")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "shards/part.safetensors"}}),
        encoding="utf-8",
    )
    original = model_cache_metadata(model_dir)["weights_sha256"]
    shard.write_bytes(b"other version")
    assert model_cache_metadata(model_dir)["weights_sha256"] != original


def test_prefix_cache_accepts_precomputed_model_metadata(tmp_path, monkeypatch) -> None:
    path, model_dir, _, _ = _cache(tmp_path)
    expected = model_cache_metadata(model_dir)
    monkeypatch.setattr(
        "llama3fhe.approx.prefix_cache_io._weights_sha256",
        lambda _model_dir: pytest.fail("worker must not hash checkpoint weights"),
    )
    bundles = load_prefix_kv_cache(
        path,
        layout=_layout(),
        expected_model_dir=model_dir,
        expected_model_metadata=expected,
    )
    assert len(bundles) == 2


def test_prefix_cache_rejects_mismatched_precomputed_fingerprint(tmp_path) -> None:
    path, model_dir, _, _ = _cache(tmp_path)
    expected = model_cache_metadata(model_dir)
    expected["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another checkpoint"):
        load_prefix_kv_cache(
            path, layout=_layout(), expected_model_metadata=expected
        )


def test_prefix_cache_rejects_legacy_format(tmp_path) -> None:
    path, _, _, _ = _cache(tmp_path)
    with np.load(path, allow_pickle=False) as cache:
        data = {name: cache[name] for name in cache.files}
    data["format"] = np.asarray("plaintext_prefix_kv_v1")
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="Unsupported Prefix KV cache format"):
        load_prefix_kv_cache(path, layout=_layout())


def test_prefix_cache_accepts_same_checkpoint_at_another_root(tmp_path) -> None:
    path, model_dir, _, _ = _cache(tmp_path)
    copied_dir = tmp_path / "remote" / model_dir.name
    copied_dir.mkdir(parents=True)
    shutil.copy2(model_dir / "config.json", copied_dir / "config.json")
    shutil.copy2(model_dir / "model.safetensors", copied_dir / "model.safetensors")
    bundles = load_prefix_kv_cache(
        path, layout=_layout(), expected_model_dir=copied_dir
    )
    assert len(bundles) == 2


def test_prefix_cache_rejects_invalid_shape_before_write(tmp_path) -> None:
    with pytest.raises(ValueError, match="shape"):
        save_prefix_kv_cache(
            tmp_path / "bad.npz",
            np.zeros((2, 2, 8), dtype=np.float32),
            np.zeros((2, 1, 8), dtype=np.float32),
            layers=[0, 1],
            prefix_k=2,
            rope_theta=10000.0,
            model_metadata={},
        )


def test_prefix_cache_requires_weight_fingerprint_before_write(tmp_path) -> None:
    with pytest.raises(ValueError, match="SHA256 of checkpoint weights"):
        save_prefix_kv_cache(
            tmp_path / "bad.npz",
            np.zeros((1, 2, 8), dtype=np.float32),
            np.zeros((1, 2, 8), dtype=np.float32),
            layers=[0],
            prefix_k=2,
            rope_theta=10000.0,
            model_metadata={"checkpoint_name": "test", "config_sha256": "0" * 64},
        )


def test_exporter_streams_plaintext_prefix_hidden_across_layers(tmp_path, monkeypatch) -> None:
    calls = []

    class Metadata:
        rope_theta = 10000.0
        rms_norm_eps = 1e-5

        @classmethod
        def from_json(cls, _path):
            return cls()

        def validate_llama3_8b(self, *, num_layers):
            assert num_layers == 2

    class Loader:
        def __init__(self, _model_dir, *, tensor_dtype):
            assert tensor_dtype == np.float32

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def get(self, _name, *, out_dtype):
            embedding = np.zeros((199, 4096), dtype=out_dtype)
            embedding[198] = 1.5
            return embedding

    def run_layer(hidden, *, weights, layout, rope_theta, eps):
        calls.append((weights, float(hidden[0, 0]), rope_theta, eps))
        key = np.full((2, 1024), float(hidden[0, 0]), dtype=np.float32)
        value = np.full((2, 1024), float(weights), dtype=np.float32)
        return hidden + float(weights + 1), key, value

    monkeypatch.setattr(exporter, "validate_runtime_checkpoint", lambda _path: None)
    monkeypatch.setattr(exporter, "CheckpointMetadata", Metadata)
    monkeypatch.setattr(exporter, "WeightLoader", Loader)
    monkeypatch.setattr(exporter, "load_layer_weights", lambda _loader, layer: layer)
    monkeypatch.setattr(exporter, "run_exact_prefix_layer", run_layer)
    monkeypatch.setattr(
        exporter,
        "model_cache_metadata",
        lambda _path: {"checkpoint_name": "test", "config_sha256": "0" * 64, "weights_sha256": "1" * 64},
    )
    path = tmp_path / "prefix.npz"
    exporter.export_prefix_kv_cache(tmp_path, path, num_layers=2)
    assert calls == [(0, 1.5, 10000.0, 1e-5), (1, 2.5, 10000.0, 1e-5)]
    with np.load(path, allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache["layers"], [0, 1])
        np.testing.assert_array_equal(cache["keys"][:, 0, 0], [1.5, 2.5])
        np.testing.assert_array_equal(cache["values"][:, 0, 0], [0.0, 1.0])
