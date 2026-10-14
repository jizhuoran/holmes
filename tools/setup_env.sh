#!/usr/bin/env bash
# Create a Python 3.12 dependency env. EasyFHE may come from a compiled source
# checkout (preferred for the hoisted-BSGS extension) or a compatible wheel.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${LLAMA3_FHE_ENV_DIR:-$ROOT/env_fhe}"
WHEELS_DIR="$ROOT/tools/third_party/wheels"
WHEEL="${EASYFHE_WHEEL:-}"

SOURCE_DIR="${EASYFHE_SOURCE_DIR:-}"
if [[ -z "$SOURCE_DIR" && -f "$HOME/EasyFHE/easyfhe/__init__.py" ]]; then
  SOURCE_DIR="$HOME/EasyFHE"
fi
if [[ -z "$SOURCE_DIR" && -z "$WHEEL" && -d "$WHEELS_DIR" ]]; then
  mapfile -t compatible_wheels < <(
    find "$WHEELS_DIR" -maxdepth 1 -type f -name 'easyfhe-*.whl' | sort
  )
  if [[ "${#compatible_wheels[@]}" -eq 1 ]]; then
    WHEEL="${compatible_wheels[0]}"
  fi
fi
if [[ -z "$SOURCE_DIR" && ( -z "$WHEEL" || ! -f "$WHEEL" ) ]]; then
  echo "Missing a compatible EasyFHE build." >&2
  echo "Set EASYFHE_SOURCE_DIR to a compiled source checkout, or" >&2
  echo "set EASYFHE_WHEEL to the release wheel path." >&2
  echo "See README.md." >&2
  exit 1
fi

if [[ ! -d "$ENV_DIR" ]]; then
  python3.12 -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
# Editable install: dependencies come from pyproject.toml, and the
# llama3fhe package resolves without PYTHONPATH in this env.
pip install -e "$ROOT"
if [[ -n "$SOURCE_DIR" ]]; then
  if [[ ! -f "$SOURCE_DIR/easyfhe/__init__.py" ]]; then
    echo "error: invalid EASYFHE_SOURCE_DIR=$SOURCE_DIR" >&2
    exit 2
  fi
  echo "Using compiled EasyFHE source checkout: $SOURCE_DIR"
else
  pip install --no-deps --no-index "$WHEEL"
fi

echo "=== FHE environment ready ==="
echo "  source $ROOT/tools/activate_env.sh"
echo "  Then run: python $ROOT/run_layer0.py --audit-only"
