#!/usr/bin/env python3
"""Export the exact plaintext Prefix KV pages consumed by T2 Prefill/Decode.

Example::

    python tools/build_prefix_kv_cache.py \\
        --model-dir weights/Meta-Llama-3-8b-quarot-safetensors-v2 \\
        --output fixtures/model/prefix2_kv.npz --layers 32

The k sink tokens run through every layer in plaintext. Only their RoPE'd K
and unrotated V are saved; content tokens never enter this computation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llama3fhe.application.checkpoint import (  # noqa: E402
    CheckpointMetadata,
    WeightLoader,
    load_layer_weights,
    validate_runtime_checkpoint,
)
from llama3fhe.approx.prefix_cache_io import (  # noqa: E402
    model_cache_metadata,
    save_prefix_kv_cache,
)
from llama3fhe.approx.prefix_stack import (  # noqa: E402
    prefix_token_ids,
    run_exact_prefix_layer,
)
from llama3fhe.layouts.attention import AttentionPairLayout  # noqa: E402


def export_prefix_kv_cache(
    model_dir: str | Path,
    output: str | Path,
    *,
    num_layers: int,
    prefix_k: int = 2,
    prefix_token_id: int = 198,
) -> None:
    model_dir = Path(model_dir).resolve()
    validate_runtime_checkpoint(model_dir)
    metadata = CheckpointMetadata.from_json(model_dir / "config.json")
    metadata.validate_llama3_8b(num_layers=num_layers)
    layout = AttentionPairLayout()
    if not 0 < prefix_k <= layout.seq_len:
        raise ValueError(f"prefix_k must be in 1..{layout.seq_len}.")

    with WeightLoader(model_dir, tensor_dtype=np.float32) as loader:
        embedding = loader.get("model.embed_tokens.weight", out_dtype=np.float32)
        token_ids = prefix_token_ids(prefix_k, token_id=prefix_token_id)
        if np.any(token_ids < 0) or np.any(token_ids >= embedding.shape[0]):
            raise ValueError(f"prefix_token_id={prefix_token_id} is outside the vocabulary.")
        hidden = np.asarray(embedding[token_ids], dtype=np.float32).copy()
        del embedding
        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for layer_idx in range(int(num_layers)):
            weights = load_layer_weights(loader, layer_idx)
            hidden, key, value = run_exact_prefix_layer(
                hidden,
                weights=weights,
                layout=layout,
                rope_theta=metadata.rope_theta,
                eps=metadata.rms_norm_eps,
            )
            keys.append(np.asarray(key, dtype=np.float32))
            values.append(np.asarray(value, dtype=np.float32))
            del weights
            print(json.dumps({"layer": layer_idx, "prefix_k": prefix_k}), flush=True)

    key_array = np.stack(keys, axis=0)
    value_array = np.stack(values, axis=0)
    save_prefix_kv_cache(
        output,
        key_array,
        value_array,
        layers=range(int(num_layers)),
        prefix_k=prefix_k,
        prefix_token_id=prefix_token_id,
        rope_theta=metadata.rope_theta,
        model_metadata=model_cache_metadata(model_dir),
    )
    print(json.dumps({"output": str(Path(output).resolve()), "keys_shape": list(key_array.shape)}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--prefix-k", type=int, default=2)
    parser.add_argument("--prefix-token-id", type=int, default=198)
    args = parser.parse_args(argv)
    export_prefix_kv_cache(
        args.model_dir,
        args.output,
        num_layers=args.layers,
        prefix_k=args.prefix_k,
        prefix_token_id=args.prefix_token_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
