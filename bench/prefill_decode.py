#!/usr/bin/env python3
"""Variable-length encrypted Prefill/Decode experiment (one card).

End-to-end single-process chain: encrypted Prefill (``emit_kv_cache=True``)
hands its PairFold128 KV caches to ``Llama3Model.decode``; every later token
is embedded, encrypted as DecodeFM, walked through the 32 decode layers and
decrypted for the client LM head.  ``--decode-shifted-lanes`` (a runner
flag, see ``llama3fhe/application/args.py``) runs the decode QKV / MLP in the
shifted-lane packing; the lane-replicated default is the pre-SL baseline.
Gold greedy continuation of Paris128: ``12366, 13, 578, 6864``.

    python bench/prefill_decode.py --num-tokens 4 [--decode-shifted-lanes] \
        --output-json artifacts/prefill_decode/sl.json

(Formerly ``archive/experiments/paris128_decode_n.py``.)

The input may contain 1..N tokens.  By default, the first 128 or fewer use one
fixed-width Prefill page; later input tokens are forced Decode steps before
generation.  ``--chunked-prefill`` evaluates up to 256 input tokens as two
encrypted 128-slot Prefill pages sharing one Softmax row sum and lambda chain
per query block; the calibrated two-page coefficient table is selected
automatically.  Prefill produces token 1 from its last valid FM row when
there is no input tail.  Each later Decode token embeds under DecodeFM,
attends the resident cache, and appends one column.  ``--decode-packing`` selects either
the Decode-native PairFold128 cache or the archived legacy C-form cache;
PairFold128 is the default and legacy C is retained only as an A/B oracle.

``--skip-prefill`` encrypts a plaintext-poly Prefill K/V cache instead of
running the encrypted Prefill graph (keygen still runs).  PairFold128 is
packed and encrypted directly from the logical K/V arrays, without creating
an intermediate legacy C cache.  The npz is reused on later Decode-only
iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from llama3fhe import Llama3CKKSConfig
from llama3fhe.application.approx_config import ApproximationConfig
from llama3fhe.application.args import (
    DEFAULT_INPUT_IDS,
    DEFAULT_SOFTMAX,
    PREFILL_256_SOFTMAX,
    PROJECT_ROOT,
    build_parser,
    resolve_poly_paths,
)
from llama3fhe.application.checkpoint import (
    CheckpointMetadata,
    WeightLoader,
    load_layer_weights,
    validate_runtime_checkpoint,
)
from llama3fhe.application.client import (
    begin_cuda_allocator_sample,
    decrypt_hidden,
    end_cuda_allocator_sample,
    final_logits,
    load_input_ids,
    top_k_logits,
)
from llama3fhe.application.inputs import (
    split_prefill_pages,
    split_variable_prefill_ids,
)
from llama3fhe.application.layer_stream import DoubleBufferedLayerStream
from llama3fhe.application.session import (
    _create_context,
    _encrypt_hidden,
    _release_transient_device_memory,
)
from llama3fhe.application.settings import layer_schedule, resolve_settings
from llama3fhe.approx.attention import apply_rope_heads, rope_cos_sin
from llama3fhe.approx.layer import poly_decode_layer_numpy, poly_prefill_stack_numpy
from llama3fhe.approx.prefix_stack import build_prefix_kv_bundles, prefix_token_ids
from llama3fhe.backend import (
    cuda_memory_snapshot,
    release_all,
    synchronize_device,
    trim_device_allocator_cache,
)
from llama3fhe.decode_packing import (
    DEFAULT_DECODE_PACKING,
    DecodePacking,
    validate_decode_packing,
)
from llama3fhe.layouts import (
    AttentionPairLayout,
    FeatureMajorPrefillLayout,
    PairFold128Layout,
)
from llama3fhe.model import (
    BootstrapOperator,
    Llama3Model,
    prepare_bootstrap_operator,
    with_decode_sparse_bootstrap,
)
from llama3fhe.operators.attention.codec import decrypt_complex_rows
from llama3fhe.operators.attention.pairfold.pairfold_prefill import (
    encrypt_pairfold_plaintext_page,
)
from llama3fhe.operators.pairfold_kv_cache import PairFoldLayerCache

DEFAULT_MODEL = PROJECT_ROOT / "weights" / "Meta-Llama-3-8b-quarot-safetensors-v2"
KV_LIMBS = 12


def _load_experiment_input_ids(extra) -> tuple[np.ndarray, object | None]:
    if extra.prompt is None:
        return load_input_ids(extra.input_ids_file), None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("--prompt requires transformers.") from exc
    tokenizer = AutoTokenizer.from_pretrained(extra.model_dir)
    ids = np.asarray(
        tokenizer.encode(str(extra.prompt), add_special_tokens=True),
        dtype=np.int64,
    )
    if ids.size == 0:
        raise ValueError("prompt tokenization produced no tokens.")
    return ids, tokenizer


def _gpu_used_mib() -> int | None:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return int(raw.splitlines()[0].strip())


def _wait_gpu_idle(*, max_used_mib: int, poll_seconds: float) -> None:
    while True:
        used = _gpu_used_mib()
        if used is None or used <= int(max_used_mib):
            print(f"gpu idle used={used} MiB", flush=True)
            return
        print(f"gpu busy used={used} MiB; wait {poll_seconds:.0f}s", flush=True)
        time.sleep(float(poll_seconds))


def _token_text(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return ""
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _exact_rmsnorm(hidden: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    scale = 1.0 / np.sqrt(
        np.mean(np.square(hidden.astype(np.float64)), axis=-1, keepdims=True)
        + float(eps)
    )
    return np.asarray(hidden * scale * gamma, dtype=np.float32)


def _exact_gqa(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    seq_len = int(query.shape[0])
    query_heads, kv_heads, head_dim = 32, 8, 128
    query = query.reshape(seq_len, query_heads, head_dim)
    key = key.reshape(seq_len, kv_heads, head_dim)
    value = value.reshape(seq_len, kv_heads, head_dim)
    ratio = query_heads // kv_heads
    key = np.repeat(key, ratio, axis=1)
    value = np.repeat(value, ratio, axis=1)
    scores = np.einsum("qhd,khd->hqk", query, key) / np.float32(head_dim**0.5)
    queries = np.arange(seq_len)[:, None]
    keys = np.arange(seq_len)[None, :]
    valid = keys <= queries
    masked = np.where(valid[None, :, :], scores, -np.inf)
    row_max = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.where(valid[None, :, :], np.exp(masked - row_max), 0.0)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    output = np.einsum("hqk,khd->qhd", probabilities, value)
    return output.reshape(seq_len, -1).astype(np.float32)


def _exact_layer(hidden: np.ndarray, weights, *, rope_theta: float, eps: float) -> np.ndarray:
    residual = np.asarray(hidden, dtype=np.float32)
    seq_len = int(residual.shape[0])
    attn_in = _exact_rmsnorm(residual, weights.input_layernorm, eps)
    query = attn_in @ np.asarray(weights.q_proj, dtype=np.float32).T
    key = attn_in @ np.asarray(weights.k_proj, dtype=np.float32).T
    value = attn_in @ np.asarray(weights.v_proj, dtype=np.float32).T
    cos, sin = rope_cos_sin(seq_len=seq_len, head_dim=128, theta=float(rope_theta))
    query = apply_rope_heads(query, cos, sin, num_heads=32, head_dim=128)
    key = apply_rope_heads(key, cos, sin, num_heads=8, head_dim=128)
    attn_out = _exact_gqa(query, key, value)
    residual = residual + attn_out @ np.asarray(weights.o_proj, dtype=np.float32).T
    mlp_in = _exact_rmsnorm(residual, weights.post_attention_layernorm, eps)
    gate = mlp_in @ np.asarray(weights.gate_proj, dtype=np.float32).T
    up = mlp_in @ np.asarray(weights.up_proj, dtype=np.float32).T
    activated = (gate / (1.0 + np.exp(-gate.astype(np.float64)))).astype(np.float32) * up
    return residual + activated @ np.asarray(weights.down_proj, dtype=np.float32).T


def run_exact_oracle(
    *,
    model_dir: Path,
    input_ids: np.ndarray,
    num_tokens: int,
    tokenizer=None,
) -> dict[str, object]:
    """Greedy exact-numpy continuation; client LM head matches production."""

    metadata = CheckpointMetadata.from_json(model_dir / "config.json")
    loader = WeightLoader(model_dir, tensor_dtype=np.float32)
    try:
        embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
        hidden = np.asarray(embedding[input_ids], dtype=np.float32)
        layers = [
            load_layer_weights(loader, index)
            for index in range(int(metadata.num_hidden_layers))
        ]
        generated: list[dict[str, object]] = []
        started = time.perf_counter()
        for step in range(int(num_tokens)):
            current = hidden
            for weights in layers:
                current = _exact_layer(
                    current,
                    weights,
                    rope_theta=float(metadata.rope_theta),
                    eps=float(metadata.rms_norm_eps),
                )
            logits = final_logits(
                current, loader=loader, metadata=metadata, token_index=-1
            )
            top = top_k_logits(logits, 5)
            token_id = int(top[0]["token_id"])
            generated.append(
                {
                    "step": step,
                    "token_id": token_id,
                    "text": _token_text(tokenizer, token_id),
                    "logit": float(top[0]["logit"]),
                    "top_k": top,
                    "seq_len": int(hidden.shape[0]),
                }
            )
            print(
                f"oracle token[{step}] id={token_id} "
                f"text={generated[-1]['text']!r} logit={top[0]['logit']:.3f}",
                flush=True,
            )
            hidden = np.concatenate(
                [hidden, embedding[token_id][None, :].astype(np.float32)],
                axis=0,
            )
        token_ids = [int(row["token_id"]) for row in generated]
        return {
            "token_ids": token_ids,
            "texts": [str(row["text"]) for row in generated],
            "joined": "".join(str(row["text"]) for row in generated),
            "generated": generated,
            "wall_seconds": float(time.perf_counter() - started),
        }
    finally:
        loader.close()


DEFAULT_CACHE_NPZ = PROJECT_ROOT / "fixtures" / "paris128_prefill_kv_poly.npz"


def _release_legacy_caches(caches) -> None:
    for key_ciphers, value_ciphers in caches:
        release_all(key_ciphers)
        release_all(value_ciphers)


def _release_pairfold_caches(caches) -> None:
    """Release live PairFold owners, tolerating slots detached by Decode."""

    seen: set[int] = set()
    for cache in caches:
        if cache is None or id(cache) in seen:
            continue
        seen.add(id(cache))
        cache.release()


def _cache_encryption_level(*, context, kv_limbs: int) -> int:
    level = int(context.context.L) - int(kv_limbs)
    if level < 0:
        raise ValueError(
            f"kv_limbs={kv_limbs} exceeds context L={int(context.context.L)}."
        )
    return level


def _save_poly_kv(
    path: Path,
    *,
    keys: np.ndarray,
    values: np.ndarray,
    token_id: int,
    logit: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keys=np.asarray(keys, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        token_id=np.int64(token_id),
        logit=np.float32(logit),
    )


def _load_poly_kv(path: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    with np.load(path) as payload:
        return (
            np.asarray(payload["keys"], dtype=np.float32),
            np.asarray(payload["values"], dtype=np.float32),
            int(payload["token_id"]),
            float(payload["logit"]),
        )


def _poly_cache_is_reusable(
    path: Path,
    *,
    prefix_k: int,
    layer_start: int,
    num_layers: int,
    checkpoint_layers: int,
) -> bool:
    """Only reuse the legacy NPZ for the exact state it represents.

    The current file format predates a manifest and records neither Prefix nor
    the checkpoint-layer range.  It is only written by the complete no-Prefix
    stack, so treating it as a subset-layer cache would silently pair layer-0
    K/V (and a full-stack token) with another checkpoint layer.
    """

    return bool(
        path.is_file()
        and int(prefix_k) == 0
        and int(layer_start) == 0
        and int(num_layers) == int(checkpoint_layers)
    )


def _require_finite_poly(
    name: str,
    values,
    *,
    layer_start: int,
) -> None:
    """Reject a broken plaintext oracle before GPU context/key generation."""

    array = np.asarray(values)
    finite = np.isfinite(array)
    if bool(np.all(finite)):
        return
    nonfinite = int(array.size - np.count_nonzero(finite))
    subset_hint = ""
    if int(layer_start):
        subset_hint = (
            f" layer_start={int(layer_start)} feeds prompt/token embeddings "
            "directly to a later checkpoint layer; this subset path is an "
            "operator/timing harness, not a semantic model oracle."
        )
    raise RuntimeError(
        f"poly {name} contains {nonfinite}/{int(array.size)} non-finite values "
        f"before GPU setup.{subset_hint}"
    )


def _compare_full_logits(
    fhe_logits: np.ndarray,
    poly_logits: np.ndarray,
) -> dict[str, float | int | bool]:
    """Compare the complete client-side vocab vectors for one Decode token."""

    fhe = np.asarray(fhe_logits, dtype=np.float64)
    poly = np.asarray(poly_logits, dtype=np.float64)
    if fhe.shape != poly.shape:
        raise ValueError(
            "FHE/poly full-logit shapes must match, got "
            f"{fhe.shape} and {poly.shape}."
        )
    if fhe.size == 0:
        raise ValueError("FHE/poly full-logit vectors must not be empty.")
    for name, values in (("FHE", fhe), ("poly", poly)):
        finite = np.isfinite(values)
        if not bool(np.all(finite)):
            nonfinite = int(values.size - np.count_nonzero(finite))
            raise RuntimeError(
                f"{name} full logits contain {nonfinite}/{int(values.size)} "
                "non-finite values."
            )

    delta = np.abs(fhe - poly)
    fhe_argmax = int(np.argmax(fhe))
    poly_argmax = int(np.argmax(poly))
    fhe_top_logit = float(fhe[fhe_argmax])
    poly_top_logit = float(poly[poly_argmax])
    return {
        "max_abs": float(np.max(delta)),
        "mean_abs": float(np.mean(delta)),
        "fhe_argmax_token_id": fhe_argmax,
        "poly_argmax_token_id": poly_argmax,
        "argmax_match": bool(fhe_argmax == poly_argmax),
        "fhe_top_logit": fhe_top_logit,
        "poly_top_logit": poly_top_logit,
        "top_logit_abs_diff": float(abs(fhe_top_logit - poly_top_logit)),
    }


def _encrypt_packed_rows(packed: np.ndarray, *, context, kv_limbs: int, slots: int):
    packed = np.asarray(packed)
    level = _cache_encryption_level(context=context, kv_limbs=kv_limbs)
    encrypted: list[object] = []
    try:
        for row in packed:
            encrypted.append(
                context.encrypt(row, level=level, slots=int(slots))
            )
        result = tuple(encrypted)
        encrypted = []
        return result
    finally:
        release_all(encrypted)


def _encrypt_poly_kv_caches(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    context,
    attention: AttentionPairLayout,
    kv_limbs: int,
):
    caches = []
    try:
        for key, value in zip(keys, values, strict=True):
            packed_k = attention.pack_key_c(key)
            packed_v = attention.pack_value_c(value)
            cache_k = _encrypt_packed_rows(
                packed_k,
                context=context,
                kv_limbs=kv_limbs,
                slots=int(attention.slots),
            )
            cache_v = _encrypt_packed_rows(
                packed_v,
                context=context,
                kv_limbs=kv_limbs,
                slots=int(attention.slots),
            )
            caches.append((cache_k, cache_v))
        result = caches
        caches = []
        return result
    finally:
        _release_legacy_caches(caches)


def _encrypt_poly_pairfold_caches(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    context,
    pairfold: PairFold128Layout,
    kv_limbs: int,
) -> list[PairFoldLayerCache]:
    """Encrypt logical poly K/V directly into independently owned PF caches."""

    keys = np.asarray(keys)
    values = np.asarray(values)
    if len(keys) != len(values):
        raise ValueError(
            "poly PairFold K/V layer counts must match, got "
            f"{len(keys)} and {len(values)}."
        )
    level = _cache_encryption_level(context=context, kv_limbs=kv_limbs)
    caches: list[PairFoldLayerCache] = []
    pending_page = None
    try:
        for key, value in zip(keys, values, strict=True):
            pending_page = encrypt_pairfold_plaintext_page(
                key,
                value,
                pairfold_layout=pairfold,
                crypto_context=context,
                level=level,
            )
            cache = PairFoldLayerCache(
                pages=(pending_page,),
                content_tokens=int(pending_page.occupied),
            )
            pending_page = None
            caches.append(cache)
        result = caches
        caches = []
        return result
    finally:
        if pending_page is not None:
            pending_page.release()
        _release_pairfold_caches(caches)


def _legacy_cache_inventory(
    caches,
    *,
    pair_count: int,
) -> dict[str, object]:
    """Return JSON-safe page/CT counts for a live legacy C cache list."""

    pair_count = int(pair_count)
    if pair_count <= 0:
        raise ValueError("pair_count must be positive.")
    layers: list[dict[str, int]] = []
    for layer_index, (key_ciphers, value_ciphers) in enumerate(caches):
        key_count = len(key_ciphers)
        value_count = len(value_ciphers)
        if key_count % pair_count or value_count % pair_count:
            raise ValueError(
                f"layer {layer_index} legacy cache width is not divisible by "
                f"pair_count={pair_count}: {key_count}/{value_count}."
            )
        key_pages = key_count // pair_count
        value_pages = value_count // pair_count
        if key_pages != value_pages:
            raise ValueError(
                f"layer {layer_index} legacy K/V page counts differ: "
                f"{key_pages}/{value_pages}."
            )
        layers.append(
            {
                "layer": int(layer_index),
                "pages": int(key_pages),
                "key_ciphertexts": int(key_count),
                "value_ciphertexts": int(value_count),
                "ciphertexts": int(key_count + value_count),
            }
        )
    return {
        "packing": "legacy_c",
        "layer_count": len(layers),
        "pages": sum(row["pages"] for row in layers),
        "ciphertexts": sum(row["ciphertexts"] for row in layers),
        "layers": layers,
    }


def _pairfold_cache_inventory(caches) -> dict[str, object]:
    """Return JSON-safe page/CT counts for live PairFold cache owners."""

    layers: list[dict[str, int]] = []
    for layer_index, cache in enumerate(caches):
        if not isinstance(cache, PairFoldLayerCache):
            raise TypeError(
                "PairFold cache inventory requires live PairFoldLayerCache "
                f"entries; layer {layer_index} has {type(cache).__name__}."
            )
        if cache.is_released:
            raise ValueError(f"PairFold cache for layer {layer_index} is released.")
        cipher_count = sum(len(page.handles) for page in cache.pages)
        layers.append(
            {
                "layer": int(layer_index),
                "pages": len(cache.pages),
                "ciphertexts": int(cipher_count),
                "content_tokens": int(cache.content_tokens),
            }
        )
    return {
        "packing": "pairfold128",
        "layer_count": len(layers),
        "pages": sum(row["pages"] for row in layers),
        "ciphertexts": sum(row["ciphertexts"] for row in layers),
        "layers": layers,
    }


def _cache_inventory(
    decode_packing: DecodePacking,
    *,
    legacy_caches,
    pairfold_caches,
    pair_count: int,
) -> dict[str, object]:
    decode_packing = validate_decode_packing(decode_packing)
    if decode_packing == "legacy_c":
        if pairfold_caches:
            raise ValueError("legacy_c inventory received PairFold caches.")
        return _legacy_cache_inventory(legacy_caches, pair_count=pair_count)
    if legacy_caches:
        raise ValueError("pairfold128 inventory received legacy C caches.")
    return _pairfold_cache_inventory(pairfold_caches)


def _pairfold_cache_limb_groups(cache: PairFoldLayerCache) -> tuple[set[int], set[int]]:
    key_limbs = {
        int(cipher.state.cur_limbs)
        for page in cache.pages
        for cipher in page.key_groups
    }
    value_limbs = {
        int(cipher.state.cur_limbs)
        for page in cache.pages
        for cipher in page.value_groups
    }
    return key_limbs, value_limbs


def _assert_pairfold_cache_limbs(
    cache: PairFoldLayerCache,
    *,
    layer_index: int,
    key_limbs: int,
    value_limbs: int,
) -> tuple[set[int], set[int]]:
    actual_key, actual_value = _pairfold_cache_limb_groups(cache)
    if actual_key != {key_limbs} or actual_value != {value_limbs}:
        raise RuntimeError(
            f"layer {layer_index} PairFold cache limbs "
            f"K={sorted(actual_key)} V={sorted(actual_value)} "
            f"!= K={key_limbs} V={value_limbs}."
        )
    return actual_key, actual_value


def _stage_totals(rows: list[dict[str, object]]) -> dict[str, float]:
    """Sum the raw per-layer fine-grained stage timings without hiding rows."""

    totals: dict[str, float] = {}
    for row in rows:
        stages = row.get("stage_seconds", {})
        if not isinstance(stages, dict):
            continue
        for name, seconds in stages.items():
            if isinstance(seconds, (int, float)):
                totals[str(name)] = totals.get(str(name), 0.0) + float(seconds)
    return totals


def _clone_cipher_tuple(ciphers) -> tuple[object, ...]:
    """Deep-copy a ciphertext sequence with transactional cleanup."""

    copied: list[object] = []
    try:
        for cipher in ciphers:
            copied.append(cipher.deep_copy())
        result = tuple(copied)
        copied = []
        return result
    finally:
        release_all(copied)


def _clone_legacy_caches(caches) -> list[tuple[tuple[object, ...], tuple[object, ...]]]:
    """Clone legacy K/V owners without leaving a half-cloned layer."""

    copied: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
    try:
        for key_ciphers, value_ciphers in caches:
            copied_key = _clone_cipher_tuple(key_ciphers)
            try:
                copied_value = _clone_cipher_tuple(value_ciphers)
            except BaseException:
                release_all(copied_key)
                raise
            copied.append((copied_key, copied_value))
        result = copied
        copied = []
        return result
    finally:
        _release_legacy_caches(copied)


def _clone_pairfold_caches(caches) -> list[PairFoldLayerCache]:
    """Clone PairFold cache owners transactionally across all layers."""

    copied: list[PairFoldLayerCache] = []
    try:
        for cache in caches:
            if not isinstance(cache, PairFoldLayerCache):
                raise TypeError(
                    "PairFold benchmark source entries must be "
                    f"PairFoldLayerCache, got {type(cache).__name__}."
                )
            copied.append(cache.clone())
        result = copied
        copied = []
        return result
    finally:
        _release_pairfold_caches(copied)


def _clone_decode_cache_state(
    decode_packing: DecodePacking,
    *,
    legacy_caches,
    pairfold_caches,
) -> tuple[list, list[PairFoldLayerCache | None]]:
    """Return independent mutable cache lists for one consuming Decode."""

    decode_packing = validate_decode_packing(decode_packing)
    if decode_packing == "legacy_c":
        if pairfold_caches:
            raise ValueError("legacy_c clone received PairFold caches.")
        return _clone_legacy_caches(legacy_caches), []
    if legacy_caches:
        raise ValueError("pairfold128 clone received legacy C caches.")
    return [], list(_clone_pairfold_caches(pairfold_caches))


def _distribution_summary(values) -> dict[str, float | int]:
    """Return a JSON-safe distribution summary used by formal benchmarks."""

    samples = np.asarray(tuple(values), dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("benchmark summary requires at least one sample.")
    if not np.isfinite(samples).all():
        raise ValueError("benchmark samples must be finite.")
    return {
        "count": int(samples.size),
        "min": float(np.min(samples)),
        "median": float(np.median(samples)),
        "p90": float(np.percentile(samples, 90.0)),
        "max": float(np.max(samples)),
    }


def _summarize_fixed_decode_trials(
    trials: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate measured trials without folding warmups into statistics."""

    if not trials:
        raise ValueError("at least one measured Decode trial is required.")
    stage_names = sorted(
        {
            str(name)
            for trial in trials
            for name in dict(trial.get("stage_seconds", {}))
        }
    )
    def optional_summary(name: str) -> dict[str, float | int] | None:
        values = [trial.get(name) for trial in trials]
        numeric = [value for value in values if isinstance(value, (int, float))]
        return None if not numeric else _distribution_summary(numeric)

    return {
        "wall_seconds": _distribution_summary(
            float(trial["wall_seconds"]) for trial in trials
        ),
        "model_wall_seconds": _distribution_summary(
            float(trial["model_wall_seconds"]) for trial in trials
        ),
        "stage_seconds": {
            name: _distribution_summary(
                float(dict(trial["stage_seconds"])[name]) for trial in trials
            )
            for name in stage_names
            if all(name in dict(trial["stage_seconds"]) for trial in trials)
        },
        "cuda_allocator_peak_bytes": optional_summary(
            "cuda_allocator_peak_bytes"
        ),
        "cuda_allocator_incremental_peak_bytes": optional_summary(
            "cuda_allocator_incremental_peak_bytes"
        ),
    }


def _materialize_fixed_decode_layers(
    decode_layers,
    *,
    expected_layers: int,
) -> tuple[object, ...]:
    """Load one immutable layer set for every fixed-state benchmark trial.

    ``DoubleBufferedLayerStream`` reloads checkpoint arrays on each
    construction.  A fixed-state benchmark must instead keep the exact same
    layer, weight, and approximation object identities across warmup and
    measured trials.  Materialization is deliberately outside every timed
    region; the ordinary generation path remains streaming.
    """

    stream = decode_layers()
    try:
        layers = tuple(stream)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    expected_layers = int(expected_layers)
    if len(layers) != expected_layers:
        raise ValueError(
            "fixed Decode layer factory returned "
            f"{len(layers)} layers; expected {expected_layers}."
        )
    return layers


def _run_fixed_decode_benchmark(
    *,
    decode_model: Llama3Model,
    hidden: np.ndarray,
    seed_input_ciphers,
    decode_layers,
    context,
    schedule,
    bootstrap,
    refresh_first_layer: bool = False,
    decode_packing: DecodePacking,
    legacy_caches,
    pairfold_caches,
    prefix_bundles,
    occupied: int,
    position: int,
    max_kv_pages: int,
    pair_count: int,
    warmup_runs: int,
    measured_runs: int,
    order_label: str | None,
) -> dict[str, object]:
    """Benchmark one fixed Decode state with isolated consuming cache clones."""

    decode_packing = validate_decode_packing(decode_packing)
    warmup_runs = int(warmup_runs)
    measured_runs = int(measured_runs)
    if warmup_runs < 0:
        raise ValueError("benchmark-warmups must be non-negative.")
    if measured_runs <= 0:
        raise ValueError("benchmark-repetitions must be positive.")
    benchmark_layers = _materialize_fixed_decode_layers(
        decode_layers,
        expected_layers=int(decode_model.num_layers),
    )

    def run_trial(*, phase: str, index: int) -> dict[str, object]:
        trial_inputs: tuple[object, ...] = ()
        trial_legacy: list = []
        trial_pairfold: list[PairFoldLayerCache | None] = []
        result = None
        try:
            trial_inputs = _clone_cipher_tuple(seed_input_ciphers)
            trial_legacy, trial_pairfold = _clone_decode_cache_state(
                decode_packing,
                legacy_caches=legacy_caches,
                pairfold_caches=pairfold_caches,
            )
            cache_before = _cache_inventory(
                decode_packing,
                legacy_caches=trial_legacy,
                pairfold_caches=trial_pairfold,
                pair_count=pair_count,
            )
            synchronize_device(context.device)
            allocator_baseline = begin_cuda_allocator_sample(context.device)
            memory_before = cuda_memory_snapshot(context.device)
            started = time.perf_counter()
            result = decode_model.decode(
                hidden,
                crypto_context=context,
                input_ciphers=trial_inputs,
                layers=benchmark_layers,
                schedule=schedule,
                bootstrap_operator=bootstrap,
                refresh_first_layer=bool(refresh_first_layer),
                refresh_post_attention=True,
                skip_attention=False,
                kv_caches=(
                    trial_legacy if decode_packing == "legacy_c" else None
                ),
                pairfold_kv_caches=(
                    trial_pairfold
                    if decode_packing == "pairfold128"
                    else None
                ),
                prefix_bundles=prefix_bundles,
                occupied=int(occupied),
                position=int(position),
                max_kv_pages=int(max_kv_pages),
                consume_kv_cache=True,
                decode_packing=decode_packing,
            )
            synchronize_device(context.device)
            wall_seconds = float(time.perf_counter() - started)
            allocator_peak = end_cuda_allocator_sample(context.device)
            memory_after = cuda_memory_snapshot(context.device)
            cache_after = _cache_inventory(
                decode_packing,
                legacy_caches=(
                    result.kv_caches if decode_packing == "legacy_c" else []
                ),
                pairfold_caches=(
                    result.pairfold_kv_caches
                    if decode_packing == "pairfold128"
                    else []
                ),
                pair_count=pair_count,
            )
            row = {
                "phase": str(phase),
                "index": int(index),
                "wall_seconds": wall_seconds,
                "model_wall_seconds": float(result.wall_seconds),
                "stage_seconds": _stage_totals(result.per_layer),
                "per_layer": list(result.per_layer),
                "cache_before": cache_before,
                "cache_after": cache_after,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "cuda_allocator_peak_bytes": allocator_peak,
                "cuda_allocator_incremental_peak_bytes": (
                    None
                    if allocator_peak is None or allocator_baseline is None
                    else max(0, int(allocator_peak) - int(allocator_baseline))
                ),
            }
            print(
                f"benchmark {phase}[{index + 1}] packing={decode_packing} "
                f"wall={wall_seconds:.3f}s stages={row['stage_seconds']}",
                flush=True,
            )
            return row
        finally:
            try:
                if result is not None:
                    result.release()
            finally:
                try:
                    _release_legacy_caches(trial_legacy)
                finally:
                    try:
                        _release_pairfold_caches(trial_pairfold)
                    finally:
                        release_all(trial_inputs)
                        synchronize_device(context.device)
                        # Keep session plaintext/rotation caches warm, but
                        # return dead encoded-weight and ciphertext blocks to
                        # the CUDA allocator before the next full trial.  This
                        # mirrors the production runner's sample boundary.
                        _release_transient_device_memory(context.device)

    warmups = [
        run_trial(phase="warmup", index=index)
        for index in range(warmup_runs)
    ]
    measured = [
        run_trial(phase="measured", index=index)
        for index in range(measured_runs)
    ]
    return {
        "protocol": {
            "name": "fixed_decode_state_v2",
            "fixed_hidden_position_cache": True,
            "cache_clone_outside_timed_region": True,
            "layers_materialized_once_outside_timed_region": True,
            "stable_layer_weight_approximation_identity": True,
            "transient_device_cleanup_after_each_trial": True,
            "session_plaintext_cache_preserved_between_trials": True,
            "gpu_synchronized_wall_time": True,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "packing": decode_packing,
            "refresh_first_layer": bool(refresh_first_layer),
            "comparison_scope": "single_packing_invocation",
            "order_label": None if order_label is None else str(order_label),
            "abba_orchestration": "external",
            "claims_abba": False,
        },
        "state": {
            "occupied": int(occupied),
            "position": int(position),
            "cache": _cache_inventory(
                decode_packing,
                legacy_caches=legacy_caches,
                pairfold_caches=pairfold_caches,
                pair_count=pair_count,
            ),
        },
        "warmups": warmups,
        "measured": measured,
        "summary": _summarize_fixed_decode_trials(measured),
    }


def _emit_vs_poly(
    caches,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    context,
    attention: AttentionPairLayout,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for layer_index, ((cache_k, cache_v), key, value) in enumerate(
        zip(caches, keys, values, strict=True)
    ):
        fhe_k = decrypt_complex_rows(cache_k, crypto_context=context)
        fhe_v = decrypt_complex_rows(cache_v, crypto_context=context)
        packed_k = attention.pack_key_c(key)
        packed_v = attention.pack_value_c(value)
        delta_k = np.abs(
            np.asarray(fhe_k, dtype=np.complex128).real
            - np.asarray(packed_k, dtype=np.float64)
        )
        delta_v = np.abs(
            np.asarray(fhe_v, dtype=np.complex128).real
            - np.asarray(packed_v, dtype=np.float64)
        )
        row = {
            "layer": int(layer_index),
            "k_max_abs": float(np.max(delta_k)),
            "k_mean_abs": float(np.mean(delta_k)),
            "v_max_abs": float(np.max(delta_v)),
            "v_mean_abs": float(np.mean(delta_v)),
        }
        rows.append(row)
        print(
            f"emit-vs-poly layer={layer_index} "
            f"k_max={row['k_max_abs']:.3e} k_mean={row['k_mean_abs']:.3e} "
            f"v_max={row['v_max_abs']:.3e} v_mean={row['v_mean_abs']:.3e}",
            flush=True,
        )
    return rows


def _emit_pairfold_vs_poly(
    caches,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    context,
    pairfold: PairFold128Layout,
) -> list[dict[str, float | int]]:
    """Decrypt PairFold pages and compare the full complex slot payload."""

    rows: list[dict[str, float | int]] = []
    for layer_index, (cache, key, value) in enumerate(
        zip(caches, keys, values, strict=True)
    ):
        if len(cache.pages) != 1:
            raise ValueError(
                "Prefill PairFold verification expects one page per layer, got "
                f"{len(cache.pages)} at layer {layer_index}."
            )
        page = cache.pages[0]
        fhe_k = decrypt_complex_rows(page.key_groups, crypto_context=context)
        fhe_v = decrypt_complex_rows(page.value_groups, crypto_context=context)
        packed_k = pairfold.pack_key(key)
        packed_v = pairfold.pack_value(value)
        delta_k = np.abs(
            np.asarray(fhe_k, dtype=np.complex128)
            - np.asarray(packed_k, dtype=np.complex128)
        )
        delta_v = np.abs(
            np.asarray(fhe_v, dtype=np.complex128)
            - np.asarray(packed_v, dtype=np.complex128)
        )
        row = {
            "layer": int(layer_index),
            "k_max_abs": float(np.max(delta_k)),
            "k_mean_abs": float(np.mean(delta_k)),
            "v_max_abs": float(np.max(delta_v)),
            "v_mean_abs": float(np.mean(delta_v)),
        }
        rows.append(row)
        print(
            f"emit-vs-poly layer={layer_index} packing=pairfold128 "
            f"k_max={row['k_max_abs']:.3e} k_mean={row['k_mean_abs']:.3e} "
            f"v_max={row['v_max_abs']:.3e} v_mean={row['v_mean_abs']:.3e}",
            flush=True,
        )
    return rows


def _poly_decode_token(
    hidden: np.ndarray,
    keys: list[np.ndarray],
    values: list[np.ndarray],
    *,
    layer_weights,
    approx_config,
    attention: AttentionPairLayout,
    position: int,
    rope_theta: float,
    layer_start: int = 0,
    prefix_bundles=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    current = np.asarray(hidden, dtype=np.float32)
    layer_start = int(layer_start)
    if layer_start < 0:
        raise ValueError("layer_start must be non-negative.")
    next_keys: list[np.ndarray] = []
    next_values: list[np.ndarray] = []
    for local_layer_idx, weights in enumerate(layer_weights):
        layer_idx = layer_start + int(local_layer_idx)
        result = poly_decode_layer_numpy(
            current,
            cache_key=keys[local_layer_idx],
            cache_value=values[local_layer_idx],
            weights=weights,
            approximations=approx_config.for_layer(int(layer_idx)),
            attention_layout=attention,
            layer_idx=int(layer_idx),
            position=int(position),
            rope_theta=float(rope_theta),
            use_alg2_softmax=True,
            prefix_bundle=(
                None
                if prefix_bundles is None
                else prefix_bundles[int(local_layer_idx)]
            ),
        )
        current = result.hidden
        next_keys.append(
            np.concatenate([keys[local_layer_idx], result.key], axis=0)
        )
        next_values.append(
            np.concatenate([values[local_layer_idx], result.value], axis=0)
        )
    return current, next_keys, next_values


def _build_experiment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-ids-file", type=Path, default=DEFAULT_INPUT_IDS)
    parser.add_argument("--prompt", help="Tokenize an arbitrary-length prompt instead of reading JSON IDs.")
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument(
        "--layer-start",
        type=int,
        default=0,
        help="Checkpoint layer index for a single-/subset-layer profile.",
    )
    parser.add_argument("--kv-limbs", type=int, default=KV_LIMBS)
    parser.add_argument(
        "--kv-value-limbs",
        type=int,
        default=None,
        help="PairFold V cache limb target; defaults to --kv-limbs.",
    )
    parser.add_argument("--wait-gpu-mib", type=int, default=512)
    parser.add_argument("--no-wait-gpu", action="store_true")
    parser.add_argument(
        "--decode-packing",
        choices=("legacy_c", "pairfold128"),
        default=DEFAULT_DECODE_PACKING,
        help=(
            "Persistent Decode KV layout.  pairfold128 is the active default; "
            "legacy_c is retained only for archived A/B replay."
        ),
    )
    parser.add_argument(
        "--rotation-key-limb-cap",
        type=int,
        default=None,
        help=(
            "Diagnostic cap for generated rotation-key limbs. Keeps the "
            "full rotation index set but trims host keys; use only when "
            "keygen memory is the limiting factor."
        ),
    )
    parser.add_argument(
        "--tiered-rotation-key-limits",
        action="store_true",
        help=(
            "Use the measured Prefill key tiers: bootstrap/RMSNorm stay at "
            "full limbs while linear and attention tiers use their validated "
            "caps. Decode caps are merged when needed and require "
            "--steady-state-layer; unplanned rotations remain full length."
        ),
    )
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Exact-numpy greedy continuation; no FHE / GPU.",
    )
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help=(
            "Skip encrypted Prefill.  Encrypt a poly-plaintext K/V cache "
            "(from --cache-npz or a ~15s CPU Prefill) and run Decode only."
        ),
    )
    parser.add_argument(
        "--no-prefill-kv-cache",
        action="store_true",
        help=(
            "Run a single-page, one-token encrypted Prefill baseline without "
            "producing a Decode KV cache."
        ),
    )
    parser.add_argument(
        "--match-prefill-decode-key-plan",
        action="store_true",
        help=(
            "With --no-prefill-kv-cache, generate the PairFold KV-emission and "
            "Decode rotation keys and limb caps used by the corresponding "
            "Prefill+Decode run, while keeping KV production disabled."
        ),
    )
    parser.add_argument(
        "--cache-npz",
        type=Path,
        default=DEFAULT_CACHE_NPZ,
        help="Poly Prefill K/V dump used by --skip-prefill.",
    )
    parser.add_argument(
        "--verify-poly",
        action="store_true",
        help="After each Decode step, decrypt and diff vs the poly Decode stack.",
    )
    parser.add_argument(
        "--chunked-prefill",
        action="store_true",
        help="Experimentally run every 128-token input block as encrypted Prefill.",
    )
    parser.add_argument(
        "--fixed-decode-benchmark",
        action="store_true",
        help=(
            "Run a fixed-state Decode benchmark in addition to normal "
            "generation. Each trial consumes an independent hidden/cache "
            "clone; the seed state remains unchanged."
        ),
    )
    parser.add_argument(
        "--benchmark-warmups",
        type=int,
        default=2,
        help="Full fixed-state Decode warmups (formal protocol default: 2).",
    )
    parser.add_argument(
        "--benchmark-repetitions",
        type=int,
        default=5,
        help="Measured fixed-state Decode trials (formal protocol default: 5).",
    )
    parser.add_argument(
        "--benchmark-order-label",
        default=None,
        help=(
            "External comparison-order label such as A1/B1/B2/A2. This "
            "single-packing process does not claim to orchestrate ABBA."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "prefill_decode" / "prefill_decode.json",
    )
    return parser


def _validate_no_prefill_kv_cache_mode(
    extra,
    *,
    continuation_count: int = 0,
    prefill_page_count: int = 1,
) -> None:
    if bool(extra.match_prefill_decode_key_plan) and (
        not bool(extra.no_prefill_kv_cache)
        or extra.decode_packing != "pairfold128"
    ):
        raise ValueError(
            "--match-prefill-decode-key-plan requires "
            "--no-prefill-kv-cache and --decode-packing pairfold128."
        )
    if not bool(extra.no_prefill_kv_cache):
        return
    if (
        int(extra.num_tokens) != 1
        or bool(extra.skip_prefill)
        or bool(extra.oracle_only)
        or bool(extra.fixed_decode_benchmark)
        or int(continuation_count) != 0
        or int(prefill_page_count) != 1
    ):
        raise ValueError(
            "--no-prefill-kv-cache requires encrypted Prefill with "
            "--num-tokens 1, one Prefill page, no input continuation, and "
            "no --fixed-decode-benchmark."
        )


def _rotation_planning_modes(
    *, emit_prefill_kv_cache: bool, decode_steps_requested: bool, match_key_plan: bool
) -> tuple[bool, bool]:
    return (
        bool(emit_prefill_kv_cache or match_key_plan),
        bool(decode_steps_requested or match_key_plan),
    )


def _planned_rotation_keys(
    *,
    prefill_model: Llama3Model | None,
    decode_model: Llama3Model,
    schedule,
    plan_prefill_kv_cache: bool,
    plan_decode_keys: bool,
    prefix_k: int,
    decode_packing: DecodePacking,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    decode_rotations = decode_model.required_decode_rotations(
        schedule,
        skip_attention=False,
        include_bootstrap=True,
        decode_packing=decode_packing,
    )
    if not plan_decode_keys:
        decode_rotations = ()
    if prefill_model is None:
        rotations = tuple(sorted(set(decode_rotations)))
    else:
        rotations = tuple(
            sorted(
                set(
                    prefill_model.required_rotations(
                        schedule,
                        emit_kv_cache=plan_prefill_kv_cache,
                        prefix_k=prefix_k,
                        decode_packing=decode_packing,
                    )
                )
                | set(decode_rotations)
            )
        )
    return rotations, tuple(decode_rotations)


def _rotation_key_plan_summary(
    rotations,
    *,
    ring_dim: int,
    full_limbs: int,
    rotation_key_limb_cap: int | None,
    rotation_key_limb_limits: dict[int, int] | None,
) -> dict[str, object]:
    from llama3fhe.parallel.prefill.keys import normalize_rotation

    normalized = sorted(
        {normalize_rotation(int(rotation), ring_dim=ring_dim) for rotation in rotations}
    )
    if rotation_key_limb_cap is not None:
        cap = int(rotation_key_limb_cap)
        if cap <= 0:
            raise ValueError("rotation_key_limb_cap must be positive.")
        effective_limits = {rotation: min(cap, full_limbs) for rotation in normalized}
    else:
        limits = rotation_key_limb_limits or {}
        effective_limits = {
            rotation: min(int(limits.get(rotation, full_limbs)), full_limbs)
            for rotation in normalized
        }
    rotation_bytes = json.dumps(normalized, separators=(",", ":")).encode("ascii")
    limb_bytes = json.dumps(
        [[rotation, effective_limits[rotation]] for rotation in normalized],
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "rotation_count": len(normalized),
        "capped_rotation_count": sum(
            limit < full_limbs for limit in effective_limits.values()
        ),
        "full_limbs": int(full_limbs),
        "rotations_sha256": hashlib.sha256(rotation_bytes).hexdigest(),
        "effective_limb_caps_sha256": hashlib.sha256(limb_bytes).hexdigest(),
    }


def _joint_rotation_key_limb_limits(
    plans,
    *,
    rotations,
    ring_dim: int,
    uncapped_rotations=(),
) -> dict[int, int]:
    """Merge phase caps; any phase needing a full key wins for that rotation."""

    from llama3fhe.parallel.prefill.keys import normalize_rotation

    required = {
        normalize_rotation(int(rotation), ring_dim=ring_dim)
        for rotation in rotations
    }
    uncapped = {
        normalize_rotation(int(rotation), ring_dim=ring_dim)
        for rotation in uncapped_rotations
    }
    limits: dict[int, int] = {}
    for plan in plans:
        phase_limits = plan.limb_limits()
        uncapped.update(set(plan.union) - set(phase_limits))
        for rotation, limit in phase_limits.items():
            limits[int(rotation)] = max(int(limit), limits.get(int(rotation), 0))
    return {
        rotation: limit
        for rotation, limit in limits.items()
        if rotation in required and rotation not in uncapped
    }


def _validate_tiered_decode_mode(
    *, tiered: bool, decode_requested: bool, steady_state_layer: bool
) -> None:
    if tiered and decode_requested and not steady_state_layer:
        raise ValueError(
            "--tiered-rotation-key-limits with Decode requires "
            "--steady-state-layer: the measured Decode caps require "
            "an RMSNorm refresh before first-layer QKV routing."
        )


def main(argv: list[str] | None = None) -> None:
    parser = _build_experiment_parser()
    extra, passthrough = parser.parse_known_args(argv)
    decode_packing = validate_decode_packing(extra.decode_packing)
    if int(extra.num_tokens) <= 0:
        raise ValueError("num-tokens must be positive.")
    if int(extra.benchmark_warmups) < 0:
        raise ValueError("benchmark-warmups must be non-negative.")
    if int(extra.benchmark_repetitions) <= 0:
        raise ValueError("benchmark-repetitions must be positive.")
    _validate_no_prefill_kv_cache_mode(extra)
    if bool(extra.oracle_only):
        input_ids, tokenizer = _load_experiment_input_ids(extra)
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(extra.model_dir)
            except Exception as exc:
                print(f"tokenizer unavailable: {exc}", flush=True)
        report = run_exact_oracle(
            model_dir=extra.model_dir,
            input_ids=input_ids,
            num_tokens=int(extra.num_tokens),
            tokenizer=tokenizer,
        )
        report["decode_packing"] = decode_packing
        extra.output_json.parent.mkdir(parents=True, exist_ok=True)
        extra.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"oracle tokens={report['token_ids']} joined={report['joined']!r} "
            f"wall={report['wall_seconds']:.1f}s json={extra.output_json}",
            flush=True,
        )
        return

    runner_argv = [
        "--model-dir",
        str(extra.model_dir),
        *(
            ["--prompt", str(extra.prompt)]
            if extra.prompt is not None
            else ["--input-ids-file", str(extra.input_ids_file)]
        ),
        "--num-layers",
        str(int(extra.num_layers)),
        "--warmup",
        "0",
        "--weights-source",
        "real",
        *passthrough,
    ]
    args = resolve_poly_paths(build_parser().parse_args(runner_argv))
    application_start = time.perf_counter()
    num_layers = int(extra.num_layers)
    layer_start = int(extra.layer_start)
    if layer_start < 0:
        raise ValueError("layer-start must be non-negative.")
    kv_limbs = int(extra.kv_limbs)
    kv_value_limbs = (
        kv_limbs if extra.kv_value_limbs is None else int(extra.kv_value_limbs)
    )
    if kv_value_limbs <= 1:
        raise ValueError("--kv-value-limbs must be greater than one.")
    if kv_value_limbs != kv_limbs and (
        decode_packing != "pairfold128" or bool(extra.skip_prefill)
    ):
        raise ValueError(
            "different K/V cache limbs require encrypted PairFold Prefill."
        )
    num_tokens = int(extra.num_tokens)
    prefill_feature = FeatureMajorPrefillLayout()
    decode_feature = FeatureMajorPrefillLayout(
        seq_len=1,
        hidden_dim=int(prefill_feature.hidden_dim),
        slots=int(prefill_feature.slots),
    )
    attention = AttentionPairLayout()
    pairfold = PairFold128Layout()
    config = Llama3CKKSConfig.from_json(args.config)
    settings = resolve_settings(args, config)
    schedule = replace(
        layer_schedule(args, settings),
        kv_limbs=kv_limbs,
        kv_value_limbs=(
            None if extra.kv_value_limbs is None else kv_value_limbs
        ),
    )
    if bool(extra.chunked_prefill):
        schedule = replace(
            schedule,
            attention_rotation_chunk_size=min(
                int(schedule.attention_rotation_chunk_size), 2
            ),
            profile_device_memory=True,
        )
    if bool(getattr(args, "decode_sparse_bs", True)):
        # Period-1 slim RMSNorm row (log_slots 3) + merged-page Softmax rows
        # (log_slots 12): the operator gets both sparse programs, every
        # rotation planner below keys them.  Same as the multi-card stations.
        schedule = with_decode_sparse_bootstrap(schedule, decode_feature, attention)
    print(
        f"decode_sparse_bs={list(schedule.bootstrap_sparse_log_slots)} "
        f"slim_period_rows={schedule.decode_rmsnorm_slim_period_rows} "
        f"shifted_lanes={schedule.decode_shifted_lanes}",
        flush=True,
    )
    skip_prefill = bool(extra.skip_prefill)

    validate_runtime_checkpoint(args.model_dir)
    metadata = CheckpointMetadata.from_json(args.model_dir / "config.json")
    metadata.validate_llama3_8b(num_layers=num_layers)
    if layer_start + num_layers > int(metadata.num_hidden_layers):
        raise ValueError(
            f"layer range [{layer_start}, {layer_start + num_layers}) exceeds "
            f"checkpoint depth {metadata.num_hidden_layers}."
        )
    input_ids, tokenizer = _load_experiment_input_ids(extra)
    prefill_ids, prefill_valid_mask, prefill_occupied, continuation_ids = (
        split_variable_prefill_ids(
            input_ids, page_size=int(prefill_feature.seq_len)
        )
    )
    prefill_pages = (
        split_prefill_pages(input_ids, page_size=int(prefill_feature.seq_len))
        if bool(extra.chunked_prefill)
        else ((prefill_ids, prefill_valid_mask, prefill_occupied),)
    )
    if bool(extra.chunked_prefill):
        continuation_ids = np.empty(0, dtype=np.int64)
    _validate_no_prefill_kv_cache_mode(
        extra,
        continuation_count=len(continuation_ids),
        prefill_page_count=len(prefill_pages),
    )
    emit_prefill_kv_cache = not bool(extra.no_prefill_kv_cache)
    prefix_k = int(getattr(args, "prefix_k", 0) or 0)
    if len(prefill_pages) > 1 and prefix_k:
        raise ValueError(
            "--chunked-prefill with Prefix needs the multi-page Prefix "
            "Softmax path; use the encrypted Decode tail for now."
        )
    explicit_softmax = any(
        option == "--softmax-coeffs" or option.startswith("--softmax-coeffs=")
        for option in passthrough
    )
    if (
        bool(extra.chunked_prefill)
        and len(prefill_pages) > 1
        and not explicit_softmax
        and args.softmax_coeffs == DEFAULT_SOFTMAX
    ):
        args.softmax_coeffs = PREFILL_256_SOFTMAX
    approx_config = ApproximationConfig(
        rmsnorm_path=args.rms_coeffs,
        silu_path=args.silu_coeffs,
        softmax_path=args.softmax_coeffs,
    )
    for layer_idx in range(layer_start, layer_start + num_layers):
        approx_config.for_layer(layer_idx)
    supported_pages = int(
        approx_config.for_layer(layer_start).softmax.max_prefill_pages
    )
    if bool(extra.chunked_prefill) and len(prefill_pages) > supported_pages:
        raise ValueError(
            f"Softmax coefficients support at most {supported_pages} encrypted "
            f"Prefill page(s), got {len(prefill_pages)}; use the encrypted "
            "Decode tail or fit a longer-length artifact."
        )
    if input_ids.size != int(prefill_feature.seq_len) and (
        skip_prefill or bool(extra.fixed_decode_benchmark)
    ):
        raise ValueError(
            "variable-length encrypted Prefill currently requires the normal "
            "FHE path without --skip-prefill or --fixed-decode-benchmark."
        )
    if (
        emit_prefill_kv_cache
        and decode_packing != "pairfold128"
        and prefill_occupied < 128
    ):
        raise ValueError("partial Prefill pages require --decode-packing pairfold128.")
    if bool(extra.chunked_prefill) and decode_packing != "pairfold128":
        raise ValueError("--chunked-prefill requires --decode-packing pairfold128.")
    generated_decode_steps = num_tokens - 1
    forced_decode_steps = len(continuation_ids)
    total_decode_steps = forced_decode_steps + generated_decode_steps
    decode_steps_requested = bool(total_decode_steps) or bool(
        extra.fixed_decode_benchmark
    )
    plan_prefill_kv_cache, plan_decode_keys = _rotation_planning_modes(
        emit_prefill_kv_cache=emit_prefill_kv_cache,
        decode_steps_requested=decode_steps_requested,
        match_key_plan=bool(extra.match_prefill_decode_key_plan),
    )
    refresh_first_decode_layer = bool(getattr(args, "steady_state_layer", False))
    _validate_tiered_decode_mode(
        tiered=bool(extra.tiered_rotation_key_limits),
        decode_requested=plan_decode_keys,
        steady_state_layer=refresh_first_decode_layer,
    )
    estimate_s = (
        180
        + (20 if skip_prefill else 60 + 400 * len(prefill_pages))
        + total_decode_steps * 540
    )
    print(
        f"plan input_tokens={len(input_ids)} output_tokens={num_tokens} "
        f"({'poly cache' if skip_prefill else f'{len(prefill_pages)} prefill page(s)'}, "
        f"{forced_decode_steps} input-tail decode, "
        f"{generated_decode_steps} generation decode) "
        f"packing={decode_packing} emit_kv_cache={emit_prefill_kv_cache} "
        f"kv_limbs={kv_limbs} kv_value_limbs={kv_value_limbs} "
        f"estimate≈{estimate_s/60:.0f} min",
        flush=True,
    )
    loader = WeightLoader(args.model_dir, tensor_dtype=np.float32)
    embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
    hidden = np.asarray(embedding[prefill_ids], dtype=np.float32)
    rope_theta = float(metadata.rope_theta)
    prefix_token_id = int(getattr(args, "prefix_token_id", 198))
    prefix_bundles = None
    if prefix_k:
        prefix_ids = prefix_token_ids(
            prefix_k, token_id=prefix_token_id
        )
        prefix_hidden = np.asarray(embedding[prefix_ids], dtype=np.float32)
        prefix_weights = [
            load_layer_weights(loader, index)
            for index in range(layer_start, layer_start + num_layers)
        ]
        prefix_bundles, _ = build_prefix_kv_bundles(
            prefix_hidden,
            prefix_weights,
            layout=attention,
            rope_theta=rope_theta,
            eps=float(metadata.rms_norm_eps),
        )
        for local_layer_idx, bundle in enumerate(prefix_bundles):
            _require_finite_poly(
                f"prefix K at checkpoint layer {layer_start + local_layer_idx}",
                bundle.key,
                layer_start=layer_start,
            )
            _require_finite_poly(
                f"prefix V at checkpoint layer {layer_start + local_layer_idx}",
                bundle.value,
                layer_start=layer_start,
            )
        del prefix_weights, prefix_hidden
        print(f"prepared plaintext Prefix KV pages k={prefix_k}", flush=True)
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        except Exception as exc:
            print(f"tokenizer unavailable: {exc}", flush=True)

    poly_keys = poly_values = None
    poly_token_id = poly_logit = None
    poly_cache_source = "not_requested"
    if skip_prefill or bool(extra.verify_poly):
        cache_path = extra.cache_npz
        # The legacy NPZ format has no Prefix metadata.  Reusing a cache made
        # without the current Prefix would silently verify/decode a different
        # KV history, so force a rebuild whenever Prefix is enabled.
        cache_reusable = input_ids.size == int(prefill_feature.seq_len) and _poly_cache_is_reusable(
            cache_path,
            prefix_k=prefix_k,
            layer_start=layer_start,
            num_layers=num_layers,
            checkpoint_layers=int(metadata.num_hidden_layers),
        )
        if cache_reusable:
            poly_cache_source = "reused_npz"
            poly_keys, poly_values, poly_token_id, poly_logit = _load_poly_kv(
                cache_path
            )
            print(
                f"loaded poly cache {cache_path} layers={len(poly_keys)} "
                f"token[0]={poly_token_id} logit={poly_logit:.3f}",
                flush=True,
            )
        else:
            poly_cache_source = "rebuilt"
            if cache_path.is_file():
                print(
                    "ignoring manifest-free full-stack/no-Prefix poly cache "
                    f"{cache_path} for prefix_k={prefix_k} "
                    f"layer_range=[{layer_start}, {layer_start + num_layers}); "
                    "rebuilding",
                    flush=True,
                )
            started_poly = time.perf_counter()
            layer_weights = [
                load_layer_weights(loader, index)
                for index in range(layer_start, layer_start + num_layers)
            ]
            poly_input_ids = (
                np.concatenate([page for page, _, _ in prefill_pages])
                if len(prefill_pages) > 1
                else prefill_ids
            )
            poly_valid_mask = (
                np.concatenate([valid for _, valid, _ in prefill_pages])
                if len(prefill_pages) > 1
                else prefill_valid_mask
            )
            poly_hidden, poly_keys, poly_values = poly_prefill_stack_numpy(
                np.asarray(embedding[poly_input_ids], dtype=np.float32),
                layer_weights,
                approx_config=approx_config,
                attention_layout=attention,
                start_layer=layer_start,
                rope_theta=rope_theta,
                use_alg2_softmax=True,
                prefix_bundles=prefix_bundles,
                token_valid_mask=poly_valid_mask,
            )
            poly_content_tokens = sum(occupied for _, _, occupied in prefill_pages)
            poly_keys = np.asarray(poly_keys[:, :poly_content_tokens])
            poly_values = np.asarray(poly_values[:, :poly_content_tokens])
            del layer_weights
            _require_finite_poly(
                "Prefill hidden", poly_hidden, layer_start=layer_start
            )
            _require_finite_poly(
                "Prefill K cache", poly_keys, layer_start=layer_start
            )
            _require_finite_poly(
                "Prefill V cache", poly_values, layer_start=layer_start
            )
            logits = final_logits(
                poly_hidden,
                loader=loader,
                metadata=metadata,
                token_index=poly_content_tokens - 1,
            )
            _require_finite_poly("Prefill logits", logits, layer_start=layer_start)
            top = top_k_logits(logits, 5)
            poly_token_id = int(top[0]["token_id"])
            poly_logit = float(top[0]["logit"])
            # Keep the legacy cache format reserved for the no-Prefix path;
            # it carries no prefix length/token metadata and must not be
            # overwritten by a Prefix-specific KV history.
            if num_layers == 32 and prefix_k == 0 and prefill_occupied == 128:
                _save_poly_kv(
                    cache_path,
                    keys=poly_keys,
                    values=poly_values,
                    token_id=poly_token_id,
                    logit=poly_logit,
                )
            print(
                f"poly prefill wall={time.perf_counter() - started_poly:.1f}s "
                f"token[0]={poly_token_id} logit={poly_logit:.3f} "
                f"dumped={cache_path if num_layers == 32 and prefix_k == 0 and prefill_occupied == 128 else 'skipped'}",
                flush=True,
            )
        _require_finite_poly("Prefill K cache", poly_keys, layer_start=layer_start)
        _require_finite_poly("Prefill V cache", poly_values, layer_start=layer_start)
        _require_finite_poly("Prefill top logit", poly_logit, layer_start=layer_start)
        if int(poly_keys.shape[0]) < num_layers:
            raise RuntimeError(
                f"poly cache has {poly_keys.shape[0]} layers, expected >={num_layers}."
            )
        poly_keys = np.asarray(poly_keys[:num_layers], dtype=np.float32)
        poly_values = np.asarray(poly_values[:num_layers], dtype=np.float32)

    if not bool(extra.no_wait_gpu):
        _wait_gpu_idle(max_used_mib=int(extra.wait_gpu_mib), poll_seconds=30.0)

    decode_model = Llama3Model(
        config=config,
        num_layers=num_layers,
        feature_layout=decode_feature,
        attention_layout=attention,
    )
    prefill_model = None if skip_prefill else Llama3Model(
        config=config,
        num_layers=num_layers,
        feature_layout=prefill_feature,
        attention_layout=attention,
    )
    rotations, decode_rotations = _planned_rotation_keys(
        prefill_model=prefill_model,
        decode_model=decode_model,
        schedule=schedule,
        plan_prefill_kv_cache=plan_prefill_kv_cache,
        plan_decode_keys=plan_decode_keys,
        prefix_k=prefix_k,
        decode_packing=decode_packing,
    )
    print(
        f"rotations={len(rotations)} packing={decode_packing}", flush=True
    )
    context = bootstrap = None
    prefill_ciphers: tuple[object, ...] = ()
    prefill_result = decode_result = None
    legacy_caches: list = []
    pairfold_caches: list[PairFoldLayerCache | None] = []
    generated: list[dict[str, object]] = []
    memory_samples: list[dict[str, object]] = []
    cache_samples: list[dict[str, object]] = []
    # Keep the per-layer timing records emitted by Llama3Model.  The
    # production runner already computes a stage breakdown (QKV, attention,
    # MLP, KV packing/copy, etc.); retaining it here makes this end-to-end
    # benchmark useful for bottleneck analysis without a second run.
    prefill_layer_profile: list[dict[str, object]] = []
    decode_layer_profiles: list[dict[str, object]] = []
    fixed_decode_benchmark_report: dict[str, object] | None = None

    def record_memory(phase: str, layer: int | None = None) -> None:
        snapshot = cuda_memory_snapshot(context.device)
        snapshot["phase"] = str(phase)
        if layer is not None:
            snapshot["layer"] = int(layer)
        memory_samples.append(snapshot)

    def record_cache(phase: str, *, step: int) -> dict[str, object]:
        snapshot = _cache_inventory(
            decode_packing,
            legacy_caches=legacy_caches,
            pairfold_caches=pairfold_caches,
            pair_count=int(attention.key_value_pair_count),
        )
        snapshot["phase"] = str(phase)
        snapshot["step"] = int(step)
        cache_samples.append(snapshot)
        print(
            f"cache phase={phase} step={step} packing={decode_packing} "
            f"pages={snapshot['pages']} ciphertexts={snapshot['ciphertexts']}",
            flush=True,
        )
        return snapshot

    emit_vs_poly = None
    prefill_vs_poly = None
    prefill_block_vs_poly: list[dict[str, object]] = []
    delta_histories: list[list[object]] = []
    try:
        # Keep the selected key lifecycle explicit.  ``--key-mode host`` is
        # useful for large Prefill rotation plans: the native context still
        # contains the same rotations, while EasyFHE loads them on demand
        # instead of eagerly materializing every device copy during keygen.
        tiered_key_limits = None
        if bool(extra.tiered_rotation_key_limits):
            from llama3fhe.parallel.decode.decode_keys import build_decode_key_plan
            from llama3fhe.parallel.decode.decode_stations import (
                decode_linear_limb_floor,
            )
            from llama3fhe.parallel.prefill.keys import (
                build_key_plan,
                normalize_rotation,
            )
            from llama3fhe.parallel.prefill.ownership import ownership_from_layouts
            from llama3fhe.parallel.presets import RTX6000PRO_HIGH

            preset = RTX6000PRO_HIGH.with_world_size(1)
            tier_limb_caps = dict(preset.limb_caps)
            # This runner starts from fresh L0 ciphertexts.  The production
            # multi-card plan's linear=14 cap assumes a post-bootstrap
            # steady-state layer; QKV here is applied at about L28.
            if not bool(getattr(args, "steady_state_layer", False)):
                tier_limb_caps["linear"] = 28
                tier_limb_caps["attn_input"] = 27
            # The first single-card Prefill Softmax keeps one extra Q limb
            # through the lambda broadcast (the production cap=16 was
            # measured after the layer-10 steady-state checkpoint).  Keep
            # this diagnostic path at 17 so the key cannot fail at the first
            # multi-page Softmax refresh.
            tier_limb_caps["softmax"] = max(
                int(tier_limb_caps.get("softmax", 0)), 17
            )
            ownership = ownership_from_layouts(
                prefill_feature, attention, world_size=1
            )[0]
            tier_plan = build_key_plan(
                preset,
                prefill_feature,
                attention,
                tiers=(
                    "bootstrap",
                    "rmsnorm",
                    "linear",
                    "attn_input",
                    "attn_core",
                    "softmax",
                    "pairfm_output",
                ),
                schedule=schedule,
                limb_caps=tier_limb_caps,
                resident_extra=preset.key_resident_extra,
                policy=preset.key_policy,
                ownership=ownership,
                emit_kv_cache=plan_prefill_kv_cache,
            )
            plans = [tier_plan] if not skip_prefill else []
            if plan_decode_keys and decode_packing == "pairfold128":
                plans.append(
                    build_decode_key_plan(
                        preset,
                        schedule=schedule,
                        feature_layout=decode_feature,
                        attention_layout=attention,
                        pairfold_layout=pairfold,
                        limb_caps=tier_limb_caps,
                        attention_limb_cap=int(tier_limb_caps["attn_core"]),
                        probability_route_limb_cap=int(
                            schedule.bootstrap_post_levels
                        ),
                        linear_limb_floor=decode_linear_limb_floor(
                            schedule,
                            approx_config,
                            range(layer_start, layer_start + num_layers),
                        ),
                    )
                )
            # ``build_key_plan`` uses the preset's bootstrap shape when it
            # asks the production rotation planner for the bootstrap tier.
            # The single-card bench, however, can override the bootstrap
            # budget and sparse programs through ``LayerSchedule`` (the
            # default schedule is (4,4), while the high-card preset records
            # (5,5)).  A rotation shared with a capped attention tier must
            # remain full-limb whenever the actual bootstrap program can use
            # it.  Otherwise the first RMSNorm refresh fails only at runtime
            # with ``rotation key ... needs 37``.
            actual_bootstrap_rotations = {
                normalize_rotation(int(rotation), ring_dim=1 << int(config.simulator.logN))
                for rotation in BootstrapOperator.required_rotations(
                    log_n=int(config.simulator.logN),
                    log_slots=int(schedule.bootstrap_log_slots),
                    level_budget=tuple(int(v) for v in schedule.bootstrap_level_budget),
                    deep_level_budget=(
                        (3, 3)
                        if bool(schedule.softmax_main_deep_bootstrap)
                        else None
                    ),
                    strategy="double_hoist",
                    sparse_log_slots=(
                        tuple(int(v) for v in schedule.bootstrap_sparse_log_slots)
                        or None
                    ),
                )
            }
            tiered_key_limits = _joint_rotation_key_limb_limits(
                plans,
                rotations=rotations,
                ring_dim=1 << int(config.simulator.logN),
                uncapped_rotations=(
                    actual_bootstrap_rotations
                    | (
                        {
                            normalize_rotation(
                                int(rotation), ring_dim=1 << int(config.simulator.logN)
                            )
                            for rotation in decode_rotations
                        }
                        if decode_packing == "legacy_c"
                        else set()
                    )
                ),
            )
            required_normalized = {
                normalize_rotation(int(rotation), ring_dim=1 << int(config.simulator.logN))
                for rotation in rotations
            }
            missing = required_normalized - set(tiered_key_limits)
            print(
                f"tiered rotation keys: planned={len(tiered_key_limits)} "
                f"full_fallback={len(missing)}",
                flush=True,
            )

        rotation_key_plan = _rotation_key_plan_summary(
            rotations,
            ring_dim=1 << int(config.simulator.logN),
            full_limbs=int(config.simulator.maxLevelsRemaining) + 1,
            rotation_key_limb_cap=extra.rotation_key_limb_cap,
            rotation_key_limb_limits=tiered_key_limits,
        )
        rotation_key_plan["matched_prefill_decode_key_plan"] = bool(
            extra.match_prefill_decode_key_plan
        )
        context = _create_context(
            config,
            rotations,
            auto_load_keys=bool(settings.auto_load_keys),
            rotation_key_limb_cap=extra.rotation_key_limb_cap,
            rotation_key_limb_limits=tiered_key_limits,
        )
        print(f"keygen {context.keygen_wall_time_s:.1f}s", flush=True)
        context.configure_plaintext_cache(
            limit_bytes=int(float(args.plaintext_cache_gb) * (1024**3)),
            max_source_bytes=int(
                float(args.plaintext_cache_max_source_mb) * (1024**2)
            ),
        )
        bootstrap = prepare_bootstrap_operator(context, schedule)
        print(f"bootstrap_setup {bootstrap.setup_seconds:.1f}s", flush=True)

        def decode_layers():
            return DoubleBufferedLayerStream(
                loader=loader,
                synthetic_weights=None,
                approx_config=approx_config,
                config=config,
                feature_layout=decode_feature,
                attention_layout=attention,
                num_layers=num_layers,
                rope_theta=rope_theta,
                start_layer=layer_start,
            )

        if skip_prefill:
            started_enc = time.perf_counter()
            if decode_packing == "legacy_c":
                legacy_caches = _encrypt_poly_kv_caches(
                    poly_keys,
                    poly_values,
                    context=context,
                    attention=attention,
                    kv_limbs=kv_limbs,
                )
            else:
                pairfold_caches = _encrypt_poly_pairfold_caches(
                    poly_keys,
                    poly_values,
                    context=context,
                    pairfold=pairfold,
                    kv_limbs=kv_limbs,
                )
            print(
                f"encrypt cache wall={time.perf_counter() - started_enc:.1f}s "
                f"packing={decode_packing} layers={num_layers}",
                flush=True,
            )
            if decode_packing == "legacy_c":
                for layer_index, (cache_k, cache_v) in enumerate(legacy_caches):
                    limbs = {
                        int(cipher.state.cur_limbs)
                        for cipher in (*cache_k, *cache_v)
                    }
                    if limbs != {kv_limbs}:
                        raise RuntimeError(
                            f"layer {layer_index} cache limbs {sorted(limbs)} "
                            f"!= kv_limbs={kv_limbs}."
                        )
            else:
                for layer_index, cache in enumerate(pairfold_caches):
                    assert isinstance(cache, PairFoldLayerCache)
                    _assert_pairfold_cache_limbs(
                        cache,
                        layer_index=layer_index,
                        key_limbs=kv_limbs,
                        value_limbs=kv_value_limbs,
                    )
            token_id = int(poly_token_id)
            generated.append(
                {
                    "step": 0,
                    "source": "poly_cache",
                    "token_id": token_id,
                    "text": _token_text(tokenizer, token_id),
                    "logit": float(poly_logit),
                    "top_k": [
                        {"token_id": token_id, "logit": float(poly_logit)}
                    ],
                    "wall_seconds": 0.0,
                }
            )
            print(
                f"token[0] poly-cache id={token_id!r} "
                f"text={generated[-1]['text']!r} logit={poly_logit:.3f}",
                flush=True,
            )
            initial_cache = record_cache("poly_cache", step=0)
            generated[-1]["cache_pages"] = int(initial_cache["pages"])
            generated[-1]["cache_ciphertexts"] = int(
                initial_cache["ciphertexts"]
            )
        else:
            def prefill_layers(layer_count: int = num_layers):
                return DoubleBufferedLayerStream(
                    loader=loader,
                    synthetic_weights=None,
                    approx_config=approx_config,
                    config=config,
                    feature_layout=prefill_feature,
                    attention_layout=attention,
                    num_layers=int(layer_count),
                    rope_theta=rope_theta,
                    start_layer=layer_start,
                )

            # The production runner uses a representative steady-state
            # warmup to populate EasyFHE's rotation/plaintext caches.  This
            # decode experiment used to force warmup=0 and therefore measured
            # a cold first layer even when callers passed --warmup.  Mirror
            # the runner's default steady-layer warmup here, while keeping
            # the output JSON's prefill wall limited to the real run.
            warmup_repetitions = int(args.warmup)
            if warmup_repetitions:
                warmup_layers = (
                    num_layers
                    if str(settings.warmup_scope) == "model"
                    else 1
                )
                warmup_model = (
                    prefill_model
                    if warmup_layers == num_layers
                    else Llama3Model(
                        config=config,
                        num_layers=1,
                        feature_layout=prefill_feature,
                        attention_layout=attention,
                    )
                )
                warmup_prefix = (
                    prefix_bundles
                    if warmup_layers == num_layers or prefix_bundles is None
                    else prefix_bundles[:1]
                )
                for warmup_index in range(warmup_repetitions):
                    warmup_ciphers = _encrypt_hidden(
                        hidden,
                        context=context,
                        layout=prefill_feature,
                        input_level=int(settings.input_level),
                    )
                    warmup_result = None
                    started_warmup = time.perf_counter()
                    try:
                        warmup_result = warmup_model.forward(
                            hidden,
                            crypto_context=context,
                            input_ciphers=warmup_ciphers,
                            layers=prefill_layers(warmup_layers),
                            bootstrap_operator=bootstrap,
                            schedule=schedule,
                            refresh_first_layer=(
                                str(settings.warmup_scope) == "steady-layer"
                            ),
                            emit_kv_cache=False,
                            prefix_bundles=warmup_prefix,
                            decode_packing=decode_packing,
                        )
                    finally:
                        if warmup_result is not None:
                            warmup_result.release()
                        release_all(warmup_ciphers)
                    print(
                        f"warmup[{warmup_index + 1}] layers={warmup_layers} "
                        f"wall={time.perf_counter() - started_warmup:.1f}s",
                        flush=True,
                    )

            prefill_ciphers = _encrypt_hidden(
                hidden,
                context=context,
                layout=prefill_feature,
                input_level=int(settings.input_level),
            )
            def on_prefill_layer(row):
                profile_row = dict(row)
                profile_row["block_index"] = 0
                profile_row["checkpoint_layer"] = int(
                    layer_start + int(row["layer"])
                )
                prefill_layer_profile.append(profile_row)
                message = (
                    f"prefill layer={row['layer']} wall={row['wall_seconds']:.1f}s "
                    f"residual_bs={row['residual_bootstrap_calls']} "
                    f"stages={row.get('stage_seconds', {})}"
                )
                if not bool(extra.chunked_prefill):
                    message += f" nested={row.get('nested_stage_seconds', {})}"
                print(message, flush=True)

            prefill_result = prefill_model.forward(
                hidden,
                crypto_context=context,
                input_ciphers=prefill_ciphers,
                layers=prefill_layers(),
                bootstrap_operator=bootstrap,
                schedule=schedule,
                refresh_first_layer=bool(getattr(args, "steady_state_layer", False)),
                emit_kv_cache=emit_prefill_kv_cache,
                attention_token_valid_mask=prefill_valid_mask,
                prefix_bundles=prefix_bundles,
                progress_callback=on_prefill_layer,
                decode_packing=decode_packing,
                prefill_occupied=prefill_occupied,
                emit_delta_kv_page=len(prefill_pages) > 1,
            )
            record_memory("prefill_complete")
            emitted_cache_count = (
                len(prefill_result.kv_caches)
                if decode_packing == "legacy_c"
                else len(prefill_result.pairfold_kv_caches)
            )
            print(
                f"prefill wall={prefill_result.wall_seconds:.1f}s "
                f"packing={decode_packing} caches={emitted_cache_count}",
                flush=True,
            )
            if emitted_cache_count != (num_layers if emit_prefill_kv_cache else 0):
                raise RuntimeError(
                    f"Prefill emitted {emitted_cache_count} {decode_packing} "
                    f"caches; expected {num_layers if emit_prefill_kv_cache else 0}."
                )
            if not emit_prefill_kv_cache:
                if prefill_result.kv_caches or prefill_result.pairfold_kv_caches:
                    raise RuntimeError("Prefill-only baseline emitted a KV cache.")
            elif decode_packing == "legacy_c":
                if prefill_result.pairfold_kv_caches:
                    raise RuntimeError("legacy Prefill emitted PairFold caches.")
                for layer_index, (cache_k, cache_v) in enumerate(
                    prefill_result.kv_caches
                ):
                    limbs = {
                        int(cipher.state.cur_limbs)
                        for cipher in (*cache_k, *cache_v)
                    }
                    print(
                        f"prefill cache layer={layer_index} packing=legacy_c "
                        f"k_c={len(cache_k)} v_c={len(cache_v)} "
                        f"limbs={sorted(limbs)}",
                        flush=True,
                    )
                    if limbs != {kv_limbs}:
                        raise RuntimeError(
                            f"layer {layer_index} cache limbs {sorted(limbs)} "
                            f"!= kv_limbs={kv_limbs}."
                        )
            else:
                if prefill_result.kv_caches:
                    raise RuntimeError("PairFold Prefill emitted legacy C caches.")
                for layer_index, cache in enumerate(
                    prefill_result.pairfold_kv_caches
                ):
                    key_limb_set, value_limb_set = _assert_pairfold_cache_limbs(
                        cache,
                        layer_index=layer_index,
                        key_limbs=kv_limbs,
                        value_limbs=kv_value_limbs,
                    )
                    print(
                        f"prefill cache layer={layer_index} packing=pairfold128 "
                        f"pages={len(cache.pages)} "
                        f"ciphertexts={sum(len(page.handles) for page in cache.pages)} "
                        f"k_limbs={sorted(key_limb_set)} "
                        f"v_limbs={sorted(value_limb_set)}",
                        flush=True,
                    )

            decoded_hidden = decrypt_hidden(
                prefill_result.ciphers,
                crypto_context=context,
                layout=prefill_feature,
            )
            if not np.isfinite(decoded_hidden).all():
                raise RuntimeError("Prefill hidden is not finite.")
            logits = final_logits(
                decoded_hidden,
                loader=loader,
                metadata=metadata,
                token_index=prefill_occupied - 1,
            )
            top = top_k_logits(logits, 5)
            token_id = int(top[0]["token_id"])
            if extra.verify_poly and input_ids.size != int(prefill_feature.seq_len):
                poly_logits = final_logits(
                    poly_hidden,
                    loader=loader,
                    metadata=metadata,
                    token_index=prefill_occupied - 1,
                )
                prefill_hidden_delta = np.abs(
                    np.asarray(decoded_hidden[:prefill_occupied], dtype=np.float64)
                    - np.asarray(poly_hidden[:prefill_occupied], dtype=np.float64)
                )
                prefill_vs_poly = {
                    "hidden_max_abs": float(np.max(prefill_hidden_delta)),
                    "hidden_mean_abs": float(np.mean(prefill_hidden_delta)),
                    "full_logits": _compare_full_logits(logits, poly_logits),
                }
                prefill_block_vs_poly.append(
                    {"block_index": 0, **prefill_vs_poly}
                )
            if len(prefill_pages) == 1 and not len(continuation_ids):
                generated.append(
                    {
                        "step": 0,
                        "source": "prefill",
                        "token_id": token_id,
                        "text": _token_text(tokenizer, token_id),
                        "logit": float(top[0]["logit"]),
                        "top_k": top,
                        "wall_seconds": float(prefill_result.wall_seconds),
                    }
                )
                print(
                    f"token[0] prefill id={token_id!r} "
                    f"text={generated[-1]['text']!r} logit={top[0]['logit']:.3f}",
                    flush=True,
                )
            if emit_prefill_kv_cache:
                if decode_packing == "legacy_c":
                    legacy_caches = prefill_result.kv_caches
                    prefill_result.kv_caches = []
                else:
                    pairfold_caches = prefill_result.pairfold_kv_caches
                    prefill_result.pairfold_kv_caches = []
            initial_cache = record_cache("prefill", step=0)
            if len(prefill_pages) == 1 and not len(continuation_ids):
                generated[-1]["cache_pages"] = int(initial_cache["pages"])
                generated[-1]["cache_ciphertexts"] = int(
                    initial_cache["ciphertexts"]
                )
            if extra.verify_poly and emit_prefill_kv_cache:
                if decode_packing == "legacy_c":
                    emit_vs_poly = _emit_vs_poly(
                        legacy_caches[:1],
                        np.asarray(poly_keys[:1]),
                        np.asarray(poly_values[:1]),
                        context=context,
                        attention=attention,
                    )
                else:
                    emit_vs_poly = _emit_pairfold_vs_poly(
                        pairfold_caches[:1],
                        np.asarray(poly_keys[:1, :prefill_occupied]),
                        np.asarray(poly_values[:1, :prefill_occupied]),
                        context=context,
                        pairfold=pairfold,
                    )
                for row in emit_vs_poly:
                    row["block_index"] = 0
            if len(prefill_pages) > 1:
                if len(prefill_result.delta_kv_pages) != num_layers:
                    raise RuntimeError(
                        "Prefill did not emit one Delta K/V page per layer."
                    )
                delta_histories = [
                    [page] for page in prefill_result.delta_kv_pages
                ]
                prefill_result.delta_kv_pages = []
            prefill_result.release()
            prefill_result = None
            release_all(prefill_ciphers)
            prefill_ciphers = ()
            trim_device_allocator_cache(context.device)
            _release_transient_device_memory(context.device)

            for block_index, (block_ids, block_valid, block_occupied) in enumerate(
                prefill_pages[1:], start=1
            ):
                synchronize_device(context.device)
                context.context.clear_cuda_rotation_cache(
                    empty_allocator_cache=True
                )
                block_start = block_index * int(prefill_feature.seq_len)
                block_hidden = np.asarray(embedding[block_ids], dtype=np.float32)
                prefill_ciphers = _encrypt_hidden(
                    block_hidden,
                    context=context,
                    layout=prefill_feature,
                    input_level=int(settings.input_level),
                )
                is_last_block = block_index == len(prefill_pages) - 1

                def on_block_layer(row, _block_index=block_index):
                    profile_row = {
                        "block_index": int(_block_index),
                        **dict(row),
                    }
                    profile_row["checkpoint_layer"] = int(
                        layer_start + int(row["layer"])
                    )
                    prefill_layer_profile.append(profile_row)
                    print(
                        f"prefill block={_block_index} layer={row['layer']} "
                        f"wall={row['wall_seconds']:.1f}s "
                        f"stages={row.get('stage_seconds', {})}",
                        flush=True,
                    )

                prefill_result = prefill_model.forward(
                    block_hidden,
                    crypto_context=context,
                    input_ciphers=prefill_ciphers,
                    layers=prefill_layers(),
                    bootstrap_operator=bootstrap,
                    schedule=schedule,
                    refresh_first_layer=bool(getattr(args, "steady_state_layer", False)),
                    emit_kv_cache=True,
                    attention_token_valid_mask=block_valid,
                    historical_delta_pages=delta_histories,
                    emit_delta_kv_page=not is_last_block,
                    content_offset=block_start,
                    progress_callback=on_block_layer,
                    decode_packing=decode_packing,
                    prefill_occupied=block_occupied,
                )
                if len(prefill_result.pairfold_kv_caches) != num_layers:
                    raise RuntimeError(
                        "Prefill block did not emit one PairFold cache per layer."
                    )
                if extra.verify_poly:
                    block_poly_keys = np.asarray(
                        poly_keys[:, block_start : block_start + block_occupied]
                    )
                    block_poly_values = np.asarray(
                        poly_values[:, block_start : block_start + block_occupied]
                    )
                    block_emit = _emit_pairfold_vs_poly(
                        prefill_result.pairfold_kv_caches,
                        block_poly_keys,
                        block_poly_values,
                        context=context,
                        pairfold=pairfold,
                    )
                    for row in block_emit:
                        row["block_index"] = block_index
                    emit_vs_poly.extend(block_emit)
                for resident, incoming in zip(
                    pairfold_caches,
                    prefill_result.pairfold_kv_caches,
                    strict=True,
                ):
                    resident.extend_pages(incoming)
                prefill_result.pairfold_kv_caches = []
                if not is_last_block:
                    if len(prefill_result.delta_kv_pages) != num_layers:
                        raise RuntimeError(
                            "Prefill block did not emit one Delta K/V page per layer."
                        )
                    for history, page in zip(
                        delta_histories, prefill_result.delta_kv_pages, strict=True
                    ):
                        history.append(page)
                    prefill_result.delta_kv_pages = []
                decoded_hidden = decrypt_hidden(
                    prefill_result.ciphers,
                    crypto_context=context,
                    layout=prefill_feature,
                )
                if not np.isfinite(decoded_hidden).all():
                    raise RuntimeError(
                        f"Prefill block {block_index} hidden is not finite."
                    )
                logits = final_logits(
                    decoded_hidden,
                    loader=loader,
                    metadata=metadata,
                    token_index=block_occupied - 1,
                )
                top = top_k_logits(logits, 5)
                token_id = int(top[0]["token_id"])
                if extra.verify_poly:
                    poly_logits = final_logits(
                        poly_hidden,
                        loader=loader,
                        metadata=metadata,
                        token_index=block_start + block_occupied - 1,
                    )
                    hidden_delta = np.abs(
                        np.asarray(decoded_hidden[:block_occupied], dtype=np.float64)
                        - np.asarray(
                            poly_hidden[
                                block_start : block_start + block_occupied
                            ],
                            dtype=np.float64,
                        )
                    )
                    prefill_vs_poly = {
                        "hidden_max_abs": float(np.max(hidden_delta)),
                        "hidden_mean_abs": float(np.mean(hidden_delta)),
                        "full_logits": _compare_full_logits(logits, poly_logits),
                    }
                    prefill_block_vs_poly.append(
                        {"block_index": block_index, **prefill_vs_poly}
                    )
                record_cache("prefill_block", step=block_index)
                if is_last_block:
                    generated.append(
                        {
                            "step": 0,
                            "source": "prefill",
                            "token_id": token_id,
                            "text": _token_text(tokenizer, token_id),
                            "logit": float(top[0]["logit"]),
                            "top_k": top,
                            "wall_seconds": float(prefill_result.wall_seconds),
                            "cache_pages": int(len(pairfold_caches[0].pages)),
                            "cache_ciphertexts": int(
                                sum(len(page.handles) for page in pairfold_caches[0].pages)
                                * num_layers
                            ),
                        }
                    )
                prefill_result.release()
                prefill_result = None
                release_all(prefill_ciphers)
                prefill_ciphers = ()
                trim_device_allocator_cache(context.device)
                _release_transient_device_memory(context.device)

            for history in delta_histories:
                for page in history:
                    page.release()
            delta_histories = []

        occupied = int(input_ids.size) if len(prefill_pages) > 1 else prefill_occupied
        position = occupied + prefix_k
        max_kv_pages = max(
            2,
            (int(input_ids.size) + num_tokens + int(attention.seq_len) - 2)
            // int(attention.seq_len),
        )
        verify_poly = bool(extra.verify_poly)
        verify_weights = None
        poly_key_pages = poly_value_pages = None
        if verify_poly:
            verify_weights = [
                load_layer_weights(loader, index)
                for index in range(layer_start, layer_start + num_layers)
            ]
            poly_key_pages = [np.asarray(row) for row in poly_keys]
            poly_value_pages = [np.asarray(row) for row in poly_values]
        if bool(extra.fixed_decode_benchmark):
            benchmark_hidden = np.asarray(
                embedding[token_id], dtype=np.float32
            ).reshape(1, int(decode_feature.hidden_dim))
            benchmark_seed_ciphers = _encrypt_hidden(
                benchmark_hidden,
                context=context,
                layout=decode_feature,
                input_level=0,
            )
            try:
                fixed_decode_benchmark_report = _run_fixed_decode_benchmark(
                    decode_model=decode_model,
                    hidden=benchmark_hidden,
                    seed_input_ciphers=benchmark_seed_ciphers,
                    decode_layers=decode_layers,
                    context=context,
                    schedule=schedule,
                    bootstrap=bootstrap,
                    refresh_first_layer=refresh_first_decode_layer,
                    decode_packing=decode_packing,
                    legacy_caches=legacy_caches,
                    pairfold_caches=pairfold_caches,
                    prefix_bundles=prefix_bundles,
                    occupied=occupied,
                    position=position,
                    max_kv_pages=max_kv_pages,
                    pair_count=int(attention.key_value_pair_count),
                    warmup_runs=int(extra.benchmark_warmups),
                    measured_runs=int(extra.benchmark_repetitions),
                    order_label=extra.benchmark_order_label,
                )
            finally:
                release_all(benchmark_seed_ciphers)
                synchronize_device(context.device)
            wall_summary = fixed_decode_benchmark_report["summary"][
                "wall_seconds"
            ]
            print(
                "fixed Decode benchmark "
                f"packing={decode_packing} "
                f"median={wall_summary['median']:.3f}s "
                f"p90={wall_summary['p90']:.3f}s",
                flush=True,
            )
        continuation_rows: list[dict[str, object]] = []
        for step in range(1, num_tokens + len(continuation_ids)):
            forced_input = step <= len(continuation_ids)
            input_token_id = (
                int(continuation_ids[step - 1]) if forced_input else token_id
            )
            token_hidden = np.asarray(
                embedding[input_token_id], dtype=np.float32
            ).reshape(1, int(decode_feature.hidden_dim))
            decode_ciphers = _encrypt_hidden(
                token_hidden,
                context=context,
                layout=decode_feature,
                input_level=0,
            )
            decode_result = None
            try:
                def on_decode_layer(row, _step=step):
                    profile_row = {"step": int(_step), **dict(row)}
                    profile_row["checkpoint_layer"] = int(
                        layer_start + int(row["layer"])
                    )
                    decode_layer_profiles.append(profile_row)
                    print(
                        f"decode[{_step}] layer={row['layer']} "
                        f"wall={row['wall_seconds']:.1f}s "
                        f"softmax_bs={row['softmax_bootstrap_calls']} "
                        f"residual_bs={row['residual_bootstrap_calls']} "
                        f"out_levels={row['output_levels']} "
                        f"stages={row.get('stage_seconds', {})} "
                        f"nested={row.get('nested_stage_seconds', {})}",
                        flush=True,
                    )

                decode_result = decode_model.decode(
                    token_hidden,
                    crypto_context=context,
                    input_ciphers=decode_ciphers,
                    layers=decode_layers(),
                    schedule=schedule,
                    bootstrap_operator=bootstrap,
                    refresh_first_layer=refresh_first_decode_layer,
                    refresh_post_attention=True,
                    skip_attention=False,
                    kv_caches=(
                        legacy_caches if decode_packing == "legacy_c" else None
                    ),
                    pairfold_kv_caches=(
                        pairfold_caches
                        if decode_packing == "pairfold128"
                        else None
                    ),
                    prefix_bundles=prefix_bundles,
                    occupied=occupied,
                    position=position,
                    max_kv_pages=max_kv_pages,
                    consume_kv_cache=True,
                    progress_callback=on_decode_layer,
                    decode_packing=decode_packing,
                )
                record_memory("decode_complete", layer=None)
                print(
                    f"decode[{step}] wall={decode_result.wall_seconds:.1f}s "
                    f"occupied={decode_result.occupied}",
                    flush=True,
                )
                step_hidden = decrypt_hidden(
                    decode_result.ciphers,
                    crypto_context=context,
                    layout=decode_feature,
                )
                if not np.isfinite(step_hidden).all():
                    raise RuntimeError(f"Decode step {step} hidden is not finite.")
                logits = final_logits(
                    step_hidden,
                    loader=loader,
                    metadata=metadata,
                    token_index=0,
                )
                top = top_k_logits(logits, 5)
                token_id = int(top[0]["token_id"])
                output_row = {
                        "step": max(0, step - len(continuation_ids)),
                        "source": (
                            "prefill_tail" if forced_input else "decode"
                        ),
                        "input_token_id": input_token_id,
                        "token_id": token_id,
                        "text": _token_text(tokenizer, token_id),
                        "logit": float(top[0]["logit"]),
                        "top_k": top,
                        "wall_seconds": float(decode_result.wall_seconds),
                        "occupied": int(decode_result.occupied),
                    }
                if forced_input and step < len(continuation_ids):
                    continuation_rows.append(output_row)
                else:
                    generated.append(output_row)
                print(
                    f"token[{step}] decode id={token_id!r} "
                    f"text={output_row['text']!r} logit={top[0]['logit']:.3f}",
                    flush=True,
                )
                if verify_poly:
                    poly_hidden, poly_key_pages, poly_value_pages = _poly_decode_token(
                        token_hidden,
                        poly_key_pages,
                        poly_value_pages,
                        layer_weights=verify_weights,
                        approx_config=approx_config,
                        attention=attention,
                        position=position,
                        rope_theta=rope_theta,
                        layer_start=layer_start,
                        prefix_bundles=prefix_bundles,
                    )
                    _require_finite_poly(
                        f"Decode step {step} hidden",
                        poly_hidden,
                        layer_start=layer_start,
                    )
                    _require_finite_poly(
                        f"Decode step {step} K cache",
                        poly_key_pages,
                        layer_start=layer_start,
                    )
                    _require_finite_poly(
                        f"Decode step {step} V cache",
                        poly_value_pages,
                        layer_start=layer_start,
                    )
                    delta = np.abs(
                        np.asarray(step_hidden, dtype=np.float64)
                        - np.asarray(poly_hidden, dtype=np.float64)
                    )
                    poly_logits = final_logits(
                        poly_hidden,
                        loader=loader,
                        metadata=metadata,
                        token_index=0,
                    )
                    full_logits_vs_poly = _compare_full_logits(
                        logits,
                        poly_logits,
                    )
                    vs_poly = {
                        "max_abs": float(np.max(delta)),
                        "mean_abs": float(np.mean(delta)),
                        "full_logits": full_logits_vs_poly,
                    }
                    output_row["vs_poly"] = vs_poly
                    print(
                        f"verify-poly max_abs={vs_poly['max_abs']:.3e} "
                        f"mean_abs={vs_poly['mean_abs']:.3e} "
                        "full_logits_max_abs="
                        f"{full_logits_vs_poly['max_abs']:.3e} "
                        "full_logits_mean_abs="
                        f"{full_logits_vs_poly['mean_abs']:.3e} "
                        "argmax_match="
                        f"{full_logits_vs_poly['argmax_match']} "
                        "top_logit_abs_diff="
                        f"{full_logits_vs_poly['top_logit_abs_diff']:.3e}",
                        flush=True,
                    )
                if decode_packing == "legacy_c":
                    next_legacy_caches = decode_result.kv_caches
                    decode_result.kv_caches = []
                    _release_legacy_caches(legacy_caches)
                    legacy_caches = next_legacy_caches
                else:
                    next_pairfold_caches = decode_result.pairfold_kv_caches
                    decode_result.pairfold_kv_caches = []
                    _release_pairfold_caches(pairfold_caches)
                    pairfold_caches = next_pairfold_caches
                    for layer_index, cache in enumerate(pairfold_caches):
                        _assert_pairfold_cache_limbs(
                            cache,
                            layer_index=layer_index,
                            key_limbs=kv_limbs,
                            value_limbs=kv_value_limbs,
                        )
                occupied = int(decode_result.occupied)
                position = int(decode_result.position)
                current_cache = record_cache("decode", step=step)
                output_row["cache_pages"] = int(current_cache["pages"])
                output_row["cache_ciphertexts"] = int(
                    current_cache["ciphertexts"]
                )
            finally:
                if decode_result is not None:
                    decode_result.release()
                    decode_result = None
                release_all(decode_ciphers)
                trim_device_allocator_cache(context.device)

        token_ids = [int(row["token_id"]) for row in generated]
        texts = [str(row["text"]) for row in generated]
        report = {
            "prompt_file": (
                None if extra.prompt is not None else str(extra.input_ids_file)
            ),
            "input_source": "prompt" if extra.prompt is not None else "json",
            "prompt_token_count": int(input_ids.size),
            "prefill_occupied": prefill_occupied,
            "prefill_page_occupancies": [occupied for _, _, occupied in prefill_pages],
            "prefill_mode": "chunked" if len(prefill_pages) > 1 else "single_page",
            "continuation_token_count": int(len(continuation_ids)),
            "continuation_rows": continuation_rows,
            "model_dir": str(args.model_dir),
            "num_layers": num_layers,
            "layer_start": layer_start,
            "num_tokens": num_tokens,
            "kv_limbs": kv_limbs,
            "kv_value_limbs": kv_value_limbs,
            "decode_packing": decode_packing,
            "emit_prefill_kv_cache": emit_prefill_kv_cache,
            "rotation_key_plan": rotation_key_plan,
            "decode_refresh_first_layer": refresh_first_decode_layer,
            "decode_shifted_lanes": bool(schedule.decode_shifted_lanes),
            "decode_sl_baby_steps": int(schedule.decode_sl_baby_steps),
            "decode_rmsnorm_slim_period_rows": schedule.decode_rmsnorm_slim_period_rows,
            "bootstrap_sparse_log_slots": list(schedule.bootstrap_sparse_log_slots),
            "skip_prefill": skip_prefill,
            "prefix_k": prefix_k,
            "prefix_token_id": prefix_token_id if prefix_k else None,
            "cache_npz": str(extra.cache_npz) if skip_prefill else None,
            "poly_cache_source": poly_cache_source,
            "approximation_paths": {
                "rmsnorm": str(args.rms_coeffs),
                "silu": str(args.silu_coeffs),
                "softmax": str(args.softmax_coeffs),
            },
            "token_ids": token_ids,
            "texts": texts,
            "joined": "".join(texts),
            "generated": generated,
            "prefill_layer_profile": prefill_layer_profile,
            "decode_layer_profiles": decode_layer_profiles,
            "prefill_stage_totals": _stage_totals(prefill_layer_profile),
            "decode_stage_totals": _stage_totals(decode_layer_profiles),
            "cache_samples": cache_samples,
            "emit_vs_poly": emit_vs_poly,
            "prefill_vs_poly": prefill_vs_poly,
            "prefill_block_vs_poly": prefill_block_vs_poly,
            "wall_seconds": float(time.perf_counter() - application_start),
            "keygen_seconds": float(context.keygen_wall_time_s),
            "device_memory_samples": memory_samples,
        }
        if fixed_decode_benchmark_report is not None:
            report["fixed_decode_benchmark"] = fixed_decode_benchmark_report
        extra.output_json.parent.mkdir(parents=True, exist_ok=True)
        extra.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"done tokens={token_ids} joined={report['joined']!r} "
            f"wall={report['wall_seconds']:.1f}s json={extra.output_json}",
            flush=True,
        )
    finally:
        _release_legacy_caches(legacy_caches)
        _release_pairfold_caches(pairfold_caches)
        for history in delta_histories:
            for page in history:
                page.release()
        if prefill_result is not None:
            prefill_result.release()
        if decode_result is not None:
            decode_result.release()
        release_all(prefill_ciphers)
        if bootstrap is not None:
            bootstrap.release()
        if context is not None:
            context.clear_plaintext_cache()
        loader.close()


if __name__ == "__main__":
    main()
