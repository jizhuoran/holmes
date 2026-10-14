#!/usr/bin/env bash
# Activate the FHE venv created by tools/setup_env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Accept an explicit override, then the setup_env.sh default, then a
# pre-existing repo/workspace .venv.
ENV_DIR="${LLAMA3_FHE_ENV_DIR:-$ROOT/env_fhe}"
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
  for candidate in "$ROOT/.venv" "$ROOT/../.venv"; do
    if [[ -f "$candidate/bin/activate" ]]; then
      ENV_DIR="$candidate"
      break
    fi
  done
fi
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
  echo "Missing env: $ENV_DIR — run ./tools/setup_env.sh first" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export LLAMA3_FHE_ROOT="$ROOT"
export MODEL_DIR="${MODEL_DIR:-$ROOT/assets/weights/Meta-Llama-3-8B-QuaRot}"
# Long encrypted runs fragment the allocator without expandable segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${EASYFHE_SOURCE_DIR:-}" && -f "$HOME/EasyFHE/easyfhe/__init__.py" ]]; then
  export EASYFHE_SOURCE_DIR="$HOME/EasyFHE"
fi
if [[ -n "${EASYFHE_SOURCE_DIR:-}" ]]; then
  if [[ ! -f "$EASYFHE_SOURCE_DIR/easyfhe/__init__.py" ]]; then
    echo "Invalid EASYFHE_SOURCE_DIR=$EASYFHE_SOURCE_DIR" >&2
    return 1 2>/dev/null || exit 1
  fi
  # Source checkout must precede any wheel in site-packages. The model package
  # no longer mutates sys.path or monkey-patches wheel APIs at import time.
  export PYTHONPATH="${EASYFHE_SOURCE_DIR}:${ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
fi
echo "Holmes environment activated"
echo "  MODEL_DIR=$MODEL_DIR"
echo "  EASYFHE_SOURCE_DIR=${EASYFHE_SOURCE_DIR:-<active-environment-wheel>}"
