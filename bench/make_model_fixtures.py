#!/usr/bin/env python3
"""Build the 32-layer real-weight fixtures for bench/t2_model.py in one pass.

``bench/make_layer_fixture.py`` re-runs layers ``0..L-1`` for every target ``L``
(O(L^2) over the model); this script walks the plaintext poly model once
(poly RMSNorm, Alg2 softmax, poly SiLU -- the mathematics the encrypted
stations approximate) over all layers of the canonical prefill128 fixture and
writes, per layer, the nine weight tensors and, once, the whole hidden-state
trajectory:

    <OUT>/layer{i}_weights.npz     input_layernorm, q_proj, k_proj, v_proj, o_proj,
                                   post_attention_layernorm, gate_proj, up_proj, down_proj
                                   (float32, ~878 MB each)
    <OUT>/hidden_trajectory.npz    input_ids, hidden_0 (embedding) .. hidden_{L} (layer L-1 output),
                                   poly_logits (client-side exact final RMSNorm + LM head on the
                                   last token of hidden_{L}), poly_top_tokens
    <OUT>/manifest.json            per-layer OOB statistics, timings, checkpoint name
    --poly-kv-out PATH             (optional) keys / values [layers, 128, 1024] (RoPE'd C-form
                                   sources of every layer) + token_id / logit of the poly
                                   prefill's next token: the ``--poly-kv`` input of
                                   ``bench/t2_decode.py`` (``paris128_prefill_kv_poly.npz``)

``bench/t2_model.py`` streams ``layer{i}_weights.npz`` through the encrypted
station chain and compares its decrypted layer outputs with ``hidden_{i+1}``.
``layer{i}_weights.npz`` + ``hidden_{i}`` / ``hidden_{i+1}`` together are the same
data as ``make_layer_fixture.py``'s ``layer{i}_fixture.npz`` (minus key_rope / value).

Usage::

    python bench/make_model_fixtures.py <ROOT> <OUT> [--layers 32] [--model-dir DIR] [--skip-existing]
                                        [--poly-kv-out fixtures/paris128_prefill_kv_poly.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("out")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="do not rewrite layer{i}_weights.npz files that already exist")
    parser.add_argument("--poly-kv-out", default=None,
                        help="also write the decode --poly-kv npz (keys/values of every layer + next token)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    sys.path.insert(0, str(root))

    from llama3fhe.application.args import DEFAULT_INPUT_IDS, DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
    from llama3fhe.application.approx_config import ApproximationConfig
    from llama3fhe.application.checkpoint import CheckpointMetadata, WeightLoader, load_layer_weights
    from llama3fhe.application.client import final_logits, top_k_logits
    from llama3fhe.approx.layer import poly_prefill_layer_numpy
    from llama3fhe.layouts import AttentionPairLayout, FeatureMajorPrefillLayout

    model_dir = (Path(args.model_dir).resolve() if args.model_dir
                 else root / "weights" / "Meta-Llama-3-8b-quarot-safetensors-v2")
    metadata = CheckpointMetadata.from_json(model_dir / "config.json")
    attention_layout = AttentionPairLayout()
    feature_layout = FeatureMajorPrefillLayout()
    fixture = json.loads(Path(DEFAULT_INPUT_IDS).read_text())
    input_ids = np.asarray(fixture["input_ids"], dtype=np.int64)
    assert input_ids.shape == (int(feature_layout.seq_len),)
    approx = ApproximationConfig(rmsnorm_path=DEFAULT_RMS, silu_path=DEFAULT_SILU, softmax_path=DEFAULT_SOFTMAX)

    out.mkdir(parents=True, exist_ok=True)
    loader = WeightLoader(model_dir, tensor_dtype=np.float32)
    embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
    hidden = np.asarray(embedding[input_ids], dtype=np.float32)
    del embedding
    print(f"embedding hidden: shape {hidden.shape} max|x| {np.max(np.abs(hidden)):.4g} "
          f"rms {np.sqrt(np.mean(hidden**2)):.4g}", flush=True)

    trajectory = {"input_ids": input_ids, "hidden_0": hidden.copy()}
    per_layer = []
    poly_keys, poly_values = [], []
    total_started = time.perf_counter()
    for idx in range(int(args.layers)):
        started = time.perf_counter()
        weights = load_layer_weights(loader, idx)
        load_seconds = time.perf_counter() - started
        weight_path = out / f"layer{idx}_weights.npz"
        if not (args.skip_existing and weight_path.exists()):
            np.savez(
                weight_path,
                input_layernorm=np.asarray(weights.input_layernorm, dtype=np.float32),
                q_proj=np.asarray(weights.q_proj, dtype=np.float32),
                k_proj=np.asarray(weights.k_proj, dtype=np.float32),
                v_proj=np.asarray(weights.v_proj, dtype=np.float32),
                o_proj=np.asarray(weights.o_proj, dtype=np.float32),
                post_attention_layernorm=np.asarray(weights.post_attention_layernorm, dtype=np.float32),
                gate_proj=np.asarray(weights.gate_proj, dtype=np.float32),
                up_proj=np.asarray(weights.up_proj, dtype=np.float32),
                down_proj=np.asarray(weights.down_proj, dtype=np.float32),
            )
        save_seconds = time.perf_counter() - started - load_seconds
        result = poly_prefill_layer_numpy(
            hidden, weights=weights, approximations=approx.for_layer(idx), attention_layout=attention_layout,
            layer_idx=idx, rope_theta=float(metadata.rope_theta), use_alg2_softmax=True,
        )
        hidden = np.asarray(result.hidden, dtype=np.float32)
        trajectory[f"hidden_{idx + 1}"] = hidden.copy()
        if args.poly_kv_out:
            poly_keys.append(np.asarray(result.key, dtype=np.float32))
            poly_values.append(np.asarray(result.value, dtype=np.float32))
        row = {
            "layer": idx, "max_abs": float(np.max(np.abs(hidden))), "rms": float(np.sqrt(np.mean(hidden ** 2))),
            "rmsn1_oob_frac": result.rmsn1_oob_frac, "rmsn2_oob_frac": result.rmsn2_oob_frac,
            "silu_oob_frac": result.silu_oob_frac, "softmax_exp_in_range": result.softmax_exp_in_range,
            "softmax_invsqrt_in_range": result.softmax_invsqrt_in_range,
            "gamma_is_unit": {"input": bool(np.all(weights.input_layernorm == 1.0)),
                              "post": bool(np.all(weights.post_attention_layernorm == 1.0))},
            "seconds": round(time.perf_counter() - started, 1),
            "load_s": round(load_seconds, 2), "save_s": round(save_seconds, 2),
        }
        per_layer.append(row)
        print(json.dumps(row), flush=True)
        del weights

    logits = final_logits(hidden, loader=loader, metadata=metadata)
    top = top_k_logits(logits, 10)
    trajectory["poly_logits"] = np.asarray(logits, dtype=np.float32)
    trajectory["poly_top_tokens"] = np.asarray([int(t["token_id"]) for t in top], dtype=np.int64)
    np.savez(out / "hidden_trajectory.npz", **trajectory)
    if args.poly_kv_out:
        # Same format as archive/experiments/paris128_decode_n.py::_save_poly_kv.
        kv_path = Path(args.poly_kv_out).resolve()
        kv_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            kv_path,
            keys=np.stack(poly_keys, axis=0),
            values=np.stack(poly_values, axis=0),
            token_id=np.int64(top[0]["token_id"]),
            logit=np.float32(top[0]["logit"]),
        )
        print(json.dumps({"poly_kv_out": str(kv_path), "keys_shape": list(np.stack(poly_keys, axis=0).shape)}), flush=True)
    manifest = {
        "source": str(model_dir.name), "input_fixture": str(Path(DEFAULT_INPUT_IDS).name),
        "layers": int(args.layers), "rope_theta": float(metadata.rope_theta),
        "rms_eps": float(getattr(metadata, "rms_norm_eps", 1e-5)),
        "weights_dtype": "float32",
        "files": {"weights": "layer{i}_weights.npz", "trajectory": "hidden_trajectory.npz"},
        "trajectory_keys": sorted(trajectory.keys()),
        "poly_top_k": top,
        "per_layer": per_layer,
        "total_seconds": round(time.perf_counter() - total_started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    loader.close()
    print(json.dumps({"poly_top_k": top[:5], "total_seconds": manifest["total_seconds"], "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
