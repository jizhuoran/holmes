#!/usr/bin/env bash
# Prepare FHE weights:
#   Meta-Llama-3-8B → QuaRot rotate → Meta-Llama-3-8B-QuaRot (safetensors)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCAL_DIR="${1:-$ROOT/assets/weights/Meta-Llama-3-8B}"
FP16_DIR="${QUAROT_FP16_DIR:-$ROOT/assets/weights/Meta-Llama-3-8B-QuaRot-fp16}"
OUT_DIR="${QUAROT_OUT_DIR:-$ROOT/assets/weights/Meta-Llama-3-8B-QuaRot}"
# Stage A needs CUDA PyTorch + fast-hadamard-transform (not EasyFHE's torch).
ROTATE_PYTHON="${ROTATE_PYTHON:-python3}"
# Stage B must also use a standard PyTorch: EasyFHE disables torch.load.
EXPORT_PYTHON="${EXPORT_PYTHON:-$ROTATE_PYTHON}"

if [[ ! -d "$ROOT/tools/third_party/QuaRot/fake_quant" ]]; then
  echo "error: clone QuaRot first (see README.md)" >&2
  exit 1
fi
if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "error: Llama-3-8B weights not found: $LOCAL_DIR" >&2
  exit 1
fi

echo "[1/2] QuaRot rotate → $FP16_DIR"
PYTHONPATH="$ROOT/tools/convert/quarot_rotate:${PYTHONPATH:-}" \
  "$ROTATE_PYTHON" -m experiments.rotate_checkpoint \
  --local-dir "$LOCAL_DIR" \
  --out-dir "$FP16_DIR"

echo "[2/2] Export fused safetensors → $OUT_DIR (python=$EXPORT_PYTHON)"
if [[ ! -x "$EXPORT_PYTHON" ]]; then
  echo "error: EXPORT_PYTHON not executable: $EXPORT_PYTHON" >&2
  exit 1
fi
PYTHONPATH="$ROOT:$ROOT/tools:${PYTHONPATH:-}" \
  "$EXPORT_PYTHON" tools/export_quarot_weights.py \
  --quarot-dir "$FP16_DIR" \
  --out-dir "$OUT_DIR" \
  --overwrite \
  --full-fuse

echo "Done. MODEL_DIR=$OUT_DIR"
