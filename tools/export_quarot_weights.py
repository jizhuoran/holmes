#!/usr/bin/env python3
"""Export a rotated Llama-3-8B checkpoint as full-fused safetensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from convert.quarot_export import export_quarot_safetensors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export QuaRot quarot_model.pt to full-fused safetensors."
    )
    parser.add_argument(
        "--quarot-dir",
        type=Path,
        default=PROJECT_ROOT / "assets/weights/Meta-Llama-3-8B-QuaRot-fp16",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "assets/weights/Meta-Llama-3-8B-QuaRot",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--full-fuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse the public o_proj/down_proj Hadamard transforms offline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_quarot_safetensors(
        args.quarot_dir,
        args.out_dir,
        overwrite=bool(args.overwrite),
        full_fuse=bool(args.full_fuse),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(output.resolve()),
                "full_fuse": bool(args.full_fuse),
                "runtime_weight_layout": "ordinary_out_in",
                "he_packing": "prepared_online_by_linear_operator",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
