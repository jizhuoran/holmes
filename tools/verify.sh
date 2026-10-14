#!/usr/bin/env bash
# Validate the source artifact. GPU stages add encrypted execution.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE="${1:-unit}"
case "$STAGE" in
  unit|steady|full) ;;
  *) echo "Usage: $0 [unit|steady|full]" >&2; exit 2 ;;
esac

PY="${LLAMA3_FHE_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${EASYFHE_SOURCE_DIR:-}" ]]; then
  export PYTHONPATH="$ROOT:$EASYFHE_SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

"$PY" -m pytest tests -q -m 'not gpu' -p no:cacheprovider
"$PY" run_layer0.py --audit-only >/dev/null
"$PY" run_model.py --audit-only >/dev/null
echo "CPU regression and static audits passed."
[[ "$STAGE" == unit ]] && exit 0

"$PY" -c 'import easyfhe; assert easyfhe.cuda.is_available(), "CUDA GPU required"'
OUT="${HOLMES_VERIFY_OUTPUT:-$ROOT/artifacts/verification}"
mkdir -p "$OUT"
"$PY" run_layer0.py --config assets/feature_major.json --weights-source synthetic \
  --steady-state-layer --warmup 1 --repetitions 3 \
  --output-json "$OUT/steady.json"
[[ "$STAGE" == steady ]] && exit 0

: "${MODEL_DIR:?Set MODEL_DIR to the rotated Llama-3-8B checkpoint}"
"$PY" run_model.py --model-dir "$MODEL_DIR" --num-layers 32 \
  --output-json "$OUT/full.json"
