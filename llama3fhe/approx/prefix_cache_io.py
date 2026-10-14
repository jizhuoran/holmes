"""Portable logical Prefix KV snapshots for the plaintext T2 prefix path."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from llama3fhe.application.checkpoint import CheckpointMetadata
from llama3fhe.layouts.attention import AttentionPairLayout

from .prefix import PrefixKVBundle

PREFIX_KV_CACHE_FORMAT = "plaintext_prefix_kv_v2"


def _weights_sha256(model_dir: Path) -> str:
    """Hash checkpoint files by name and content, independent of their root path."""

    paths = {path.name: path for path in model_dir.glob("*.safetensors")}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors index: {index_path}.")
        for filename in weight_map.values():
            if not isinstance(filename, str):
                raise ValueError(f"Invalid safetensors shard name in {index_path}.")
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid safetensors shard path {filename!r}.")
            path = model_dir / relative
            if not path.is_file() or not path.resolve().is_relative_to(model_dir):
                raise FileNotFoundError(f"Safetensors shard not found in {model_dir}: {filename}.")
            paths[relative.as_posix()] = path
        paths[index_path.name] = index_path
    if not paths:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}.")
    digest = hashlib.sha256()
    for relative_name, path in sorted(paths.items()):
        name = relative_name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        digit in "0123456789abcdef" for digit in value
    )


def model_cache_metadata(model_dir: str | Path) -> dict[str, object]:
    """Identify the exact local checkpoint used to generate the Prefix page."""

    model_dir = Path(model_dir).resolve()
    config_path = model_dir / "config.json"
    config_bytes = config_path.read_bytes()
    metadata = CheckpointMetadata.from_json(config_path)
    return {
        "model_dir": str(model_dir),
        "checkpoint_name": model_dir.name,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weights_sha256": _weights_sha256(model_dir),
        "config": asdict(metadata),
    }


def _validated_arrays(
    keys: np.ndarray,
    values: np.ndarray,
    layers: Sequence[int],
    prefix_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.asarray(keys, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    layers = np.asarray(layers, dtype=np.int64)
    prefix_k = int(prefix_k)
    if keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("Prefix keys and values must share shape [layers, tokens, kv_dim].")
    if prefix_k < 1 or keys.shape[1] != prefix_k or keys.shape[2] < 1:
        raise ValueError(f"Prefix KV shape {keys.shape} does not match prefix_k={prefix_k}.")
    if layers.ndim != 1 or layers.shape[0] != keys.shape[0] or not layers.size:
        raise ValueError("Layer IDs must contain one entry per Prefix KV layer.")
    if np.any(layers < 0) or np.any(np.diff(layers) <= 0):
        raise ValueError("Layer IDs must be non-negative and strictly increasing.")
    if not np.isfinite(keys).all() or not np.isfinite(values).all():
        raise ValueError("Prefix KV arrays must contain only finite values.")
    return keys, values, layers


def save_prefix_kv_cache(
    path: str | Path,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    layers: Sequence[int],
    prefix_k: int,
    rope_theta: float,
    model_metadata: Mapping[str, object],
    prefix_token_id: int = 198,
) -> None:
    """Atomically save RoPE'd K and unrotated V as compressed float32 arrays."""

    keys, values, layers = _validated_arrays(keys, values, layers, prefix_k)
    if not np.isfinite(rope_theta) or float(rope_theta) <= 0:
        raise ValueError("rope_theta must be finite and positive.")
    if int(prefix_token_id) < 0:
        raise ValueError("prefix_token_id must be non-negative.")
    if not isinstance(model_metadata, Mapping):
        raise ValueError("model_metadata must be a mapping.")
    if not _valid_sha256(model_metadata.get("weights_sha256")):
        raise ValueError("model_metadata must contain a SHA256 of checkpoint weights.")
    metadata_json = json.dumps(dict(model_metadata), sort_keys=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as stream:
            tmp_path = Path(stream.name)
            np.savez_compressed(
                stream,
                format=np.asarray(PREFIX_KV_CACHE_FORMAT),
                keys=keys,
                values=values,
                layers=layers,
                prefix_k=np.int64(prefix_k),
                prefix_token_id=np.int64(prefix_token_id),
                rope_theta=np.float64(rope_theta),
                model_metadata=np.asarray(metadata_json),
            )
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def load_prefix_kv_cache(
    path: str | Path,
    *,
    layout: AttentionPairLayout,
    expected_layers: Sequence[int] | None = None,
    expected_prefix_k: int | None = None,
    expected_rope_theta: float | None = None,
    expected_model_dir: str | Path | None = None,
    expected_model_metadata: Mapping[str, object] | None = None,
    expected_prefix_token_id: int | None = None,
) -> list[PrefixKVBundle]:
    """Validate a snapshot and reconstruct plaintext C-form Prefix pages."""

    with np.load(path, allow_pickle=False) as cache:
        required = {
            "format", "keys", "values", "layers", "prefix_k", "prefix_token_id",
            "rope_theta", "model_metadata",
        }
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f"Prefix KV cache is missing {sorted(missing)}.")
        if str(cache["format"].item()) != PREFIX_KV_CACHE_FORMAT:
            raise ValueError("Unsupported Prefix KV cache format.")
        if cache["keys"].dtype != np.float32 or cache["values"].dtype != np.float32:
            raise ValueError("Prefix KV cache keys and values must be float32.")
        prefix_k = int(cache["prefix_k"].item())
        token_id = int(cache["prefix_token_id"].item())
        rope_theta = float(cache["rope_theta"].item())
        keys, values, layers = _validated_arrays(
            cache["keys"], cache["values"], cache["layers"], prefix_k
        )
        raw_metadata = json.loads(str(cache["model_metadata"].item()))

    if not isinstance(raw_metadata, dict):
        raise ValueError("Prefix KV model metadata must be a JSON object.")
    if not _valid_sha256(raw_metadata.get("weights_sha256")):
        raise ValueError("Prefix KV cache lacks a checkpoint weights SHA256; rebuild it.")
    if prefix_k > int(layout.seq_len):
        raise ValueError(f"prefix_k={prefix_k} exceeds page capacity {layout.seq_len}.")
    expected_kv_dim = int(layout.key_value_heads) * int(layout.head_dim)
    if keys.shape[2] != expected_kv_dim:
        raise ValueError(f"Prefix KV dimension must be {expected_kv_dim}, got {keys.shape[2]}.")
    if expected_layers is not None:
        requested = [int(layer) for layer in expected_layers]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Requested Prefix layer IDs must be non-empty and unique.")
        offsets = {int(layer): index for index, layer in enumerate(layers)}
        missing_layers = [layer for layer in requested if layer not in offsets]
        if missing_layers:
            raise ValueError(
                f"Prefix KV layer IDs {layers.tolist()} lack requested layers {missing_layers}."
            )
        selected = [offsets[layer] for layer in requested]
        keys = keys[selected]
        values = values[selected]
    if expected_prefix_k is not None and prefix_k != int(expected_prefix_k):
        raise ValueError(f"Prefix KV cache has prefix_k={prefix_k}, expected {expected_prefix_k}.")
    if expected_rope_theta is not None and not np.isclose(
        rope_theta, float(expected_rope_theta), rtol=0, atol=1e-8
    ):
        raise ValueError(f"Prefix KV cache has rope_theta={rope_theta}, expected {expected_rope_theta}.")
    if expected_prefix_token_id is not None and token_id != int(expected_prefix_token_id):
        raise ValueError(f"Prefix KV cache has token_id={token_id}, expected {expected_prefix_token_id}.")
    if expected_model_metadata is not None or expected_model_dir is not None:
        actual = (
            expected_model_metadata
            if expected_model_metadata is not None
            else model_cache_metadata(expected_model_dir)
        )
        if not isinstance(actual, Mapping) or not _valid_sha256(actual.get("weights_sha256")):
            raise ValueError("Expected model metadata must contain a checkpoint weights SHA256.")
        recorded_name = raw_metadata.get("checkpoint_name")
        if recorded_name is None and raw_metadata.get("model_dir"):
            recorded_name = Path(str(raw_metadata["model_dir"])).name
        if (
            recorded_name != actual["checkpoint_name"]
            or raw_metadata.get("config_sha256") != actual["config_sha256"]
            or raw_metadata.get("weights_sha256") != actual["weights_sha256"]
        ):
            raise ValueError("Prefix KV cache was built for another checkpoint, weights, or config.")
    return [
        PrefixKVBundle.from_logical(key, value, layout=layout)
        for key, value in zip(keys, values)
    ]


__all__ = [
    "PREFIX_KV_CACHE_FORMAT",
    "model_cache_metadata",
    "save_prefix_kv_cache",
    "load_prefix_kv_cache",
]
