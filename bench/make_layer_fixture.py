#!/usr/bin/env python3
"""Build the L10 real-weight fixture for bench/t2_layer.py --real-weights.

Runs the plaintext poly model (the same numpy mathematics the encrypted path
approximates: poly RMSNorm, Alg2 softmax, poly SiLU) over layers 0..9 of the
canonical prefill128 fixture to obtain the layer-10 input hidden state, then the
layer-10 oracle output.  Stores layer 10's nine weight tensors plus the hidden
state, the oracle output and every intermediate the stations expose.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 10
sys.path.insert(0, str(ROOT))

from llama3fhe.application.args import DEFAULT_INPUT_IDS, DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
from llama3fhe.application.approx_config import ApproximationConfig
from llama3fhe.application.checkpoint import CheckpointMetadata, WeightLoader, load_layer_weights
from llama3fhe.approx.layer import poly_prefill_layer_numpy
from llama3fhe.layouts import AttentionPairLayout, FeatureMajorPrefillLayout

model_dir = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else ROOT / "weights" / "Meta-Llama-3-8b-quarot-safetensors-v2"
metadata = CheckpointMetadata.from_json(model_dir / "config.json")
attention_layout = AttentionPairLayout()
feature_layout = FeatureMajorPrefillLayout()
fixture = json.loads(Path(DEFAULT_INPUT_IDS).read_text())
input_ids = np.asarray(fixture["input_ids"], dtype=np.int64)
assert input_ids.shape == (int(feature_layout.seq_len),)
approx = ApproximationConfig(rmsnorm_path=DEFAULT_RMS, silu_path=DEFAULT_SILU, softmax_path=DEFAULT_SOFTMAX)

loader = WeightLoader(model_dir, tensor_dtype=np.float32)
embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
hidden = np.asarray(embedding[input_ids], dtype=np.float32)
del embedding
print(f"embedding hidden: shape {hidden.shape} max|x| {np.max(np.abs(hidden)):.4g} rms {np.sqrt(np.mean(hidden**2)):.4g}")

per_layer = []
for idx in range(LAYER):
    started = time.perf_counter()
    weights = load_layer_weights(loader, idx)
    result = poly_prefill_layer_numpy(
        hidden, weights=weights, approximations=approx.for_layer(idx), attention_layout=attention_layout,
        layer_idx=idx, rope_theta=float(metadata.rope_theta), use_alg2_softmax=True,
    )
    hidden = np.asarray(result.hidden, dtype=np.float32)
    row = {"layer": idx, "max_abs": float(np.max(np.abs(hidden))), "rms": float(np.sqrt(np.mean(hidden ** 2))),
           "rmsn1_oob_frac": result.rmsn1_oob_frac, "rmsn2_oob_frac": result.rmsn2_oob_frac,
           "silu_oob_frac": result.silu_oob_frac, "softmax_exp_in_range": result.softmax_exp_in_range,
           "softmax_invsqrt_in_range": result.softmax_invsqrt_in_range, "seconds": round(time.perf_counter() - started, 1)}
    per_layer.append(row)
    print(json.dumps(row))

layer_input = hidden.copy()
weights = load_layer_weights(loader, LAYER)
approximations = approx.for_layer(LAYER)
result = poly_prefill_layer_numpy(
    layer_input, weights=weights, approximations=approximations, attention_layout=attention_layout,
    layer_idx=LAYER, rope_theta=float(metadata.rope_theta), use_alg2_softmax=True,
)
row = {"layer": LAYER, "max_abs": float(np.max(np.abs(result.hidden))), "rms": float(np.sqrt(np.mean(result.hidden ** 2))),
       "rmsn1_oob_frac": result.rmsn1_oob_frac, "rmsn2_oob_frac": result.rmsn2_oob_frac,
       "silu_oob_frac": result.silu_oob_frac, "softmax_exp_in_range": result.softmax_exp_in_range,
       "softmax_invsqrt_in_range": result.softmax_invsqrt_in_range}
print("oracle", json.dumps(row))
per_layer.append(row)

OUT.mkdir(parents=True, exist_ok=True)
np.savez(
    OUT / f"layer{LAYER}_fixture.npz",
    input_ids=input_ids,
    hidden_in=layer_input,
    hidden_out=np.asarray(result.hidden, dtype=np.float32),
    key_rope=np.asarray(result.key, dtype=np.float32),
    value=np.asarray(result.value, dtype=np.float32),
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
(OUT / f"layer{LAYER}_fixture.json").write_text(json.dumps({
    "source": str(model_dir.name), "input_fixture": str(Path(DEFAULT_INPUT_IDS).name), "layer": LAYER,
    "rope_theta": float(metadata.rope_theta), "rms_eps": float(getattr(metadata, "rms_norm_eps", 1e-5)),
    "per_layer": per_layer,
    "gamma_is_unit": {"input": bool(np.all(weights.input_layernorm == 1.0)), "post": bool(np.all(weights.post_attention_layernorm == 1.0))},
}, indent=2))
loader.close()
print("wrote", OUT / f"layer{LAYER}_fixture.npz")
