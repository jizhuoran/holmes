"""Command-line surface for the encrypted Llama-3 runners.

One parser serves both entry points: ``run_model.py`` gets the full 32-layer
defaults, ``run_layer0.py`` passes ``layer0_preset=True`` for the single-layer
defaults. Keeping it here means the flag surface can be read without paging
through the orchestration, and the orchestration never has to mention argparse.

Flag help describes the available runtime controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = PROJECT_ROOT / "assets/feature_major.json"

DEFAULT_RMS = PROJECT_ROOT / "assets/polynomials/rmsnorm_poly_coeffs.json"

DEFAULT_SILU = PROJECT_ROOT / "assets/polynomials/silu_poly_coeffs.json"

DEFAULT_SOFTMAX = PROJECT_ROOT / "assets/polynomials/softmax_poly_coeffs.json"

PREFILL_256_SOFTMAX = (
    PROJECT_ROOT / "assets/polynomials/softmax_poly_coeffs_prefill256.json"
)

# Prefix k=2 changes the score-row distribution.  The production no-Prefix
# pack has a narrow layer-4 j1 interval (hi=2.83), while the Prefix trace
# reaches about 3.23.  Keep the calibrated Prefix pack opt-in at the file
# level, but select it automatically for the ordinary ``--prefix-k 2`` path.
# An explicit ``--softmax-coeffs`` still wins below.
PREFIX_K2_SOFTMAX = (
    PROJECT_ROOT / "assets/polynomials/softmax_poly_coeffs_prefix_k2.json"
)

JOINT_TOL_V1_RMS = (
    PROJECT_ROOT / "assets/polynomials/rmsnorm_poly_coeffs_joint_tol_v1.json"
)
JOINT_TOL_V1_SILU = (
    PROJECT_ROOT / "assets/polynomials/silu_poly_coeffs_joint_tol_v1.json"
)
JOINT_TOL_V1_SOFTMAX = (
    PROJECT_ROOT
    / "assets/polynomials/softmax_poly_coeffs_joint_tol_v1_j1hifloor8.json"
)

DEFAULT_INPUT_IDS = PROJECT_ROOT / "assets/prefill128_input_ids.json"


def _fixture_expected_token_id(path: str | Path) -> int | None:
    """Return an optional next-token assertion embedded in an input fixture."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    value = payload.get("expected_next_token_id")
    return None if value is None else int(value)


def build_parser(*, layer0_preset: bool = False) -> argparse.ArgumentParser:
    """Build the shared CLI; the Layer0 executable changes defaults only."""

    scope = "one layer" if layer0_preset else "a streamed model"
    parser = argparse.ArgumentParser(
        description=f"Run {scope} of persistent feature-major Llama-3-8B."
    )
    parser.add_argument("--model-dir", type=Path)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--input-ids-file",
        type=Path,
        help=(
            "JSON token fixture. run_model.py defaults to "
            "assets/prefill128_input_ids.json when neither input option is "
            "provided."
        ),
    )
    sources.add_argument(
        "--prompt",
        help="Tokenize this prompt instead of using the canonical fixture.",
    )
    parser.set_defaults(
        default_input_ids_file=None if layer0_preset else DEFAULT_INPUT_IDS
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    if layer0_preset:
        parser.set_defaults(num_layers=1)
    else:
        parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument(
        "--weights-source",
        choices=("real", "synthetic"),
        default="synthetic" if layer0_preset else "real",
    )
    parser.add_argument("--seed", type=int, default=26081461)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--key-mode",
        choices=("host", "eager"),
        default="eager",
        help=(
            "Keep rotation keys host-backed and load them on demand, or "
            "eagerly preload them into the device context (default: eager)."
        ),
    )
    parser.add_argument(
        "--input-level",
        type=int,
        default=None,
        help=(
            "Initial ciphertext level. Auto selects L0 for an ordinary "
            "single-layer run, L(depth-1) for --steady-state-layer, and "
            "the bootstrap output level for a streamed multi-layer run."
        ),
    )
    parser.add_argument(
        "--steady-state-layer",
        action="store_true",
        help=(
            "Run Layer0 weights with the same input refresh and attention "
            "checkpoint schedule as a later transformer layer."
        ),
    )
    parser.add_argument(
        "--softmax-bootstrap-position",
        choices=("auto", "pre", "post", "off"),
        default="auto",
        help=(
            "Place the steady-state Softmax main-stream refresh before or "
            "after exp, disable it, or use auto (pre for steady state)."
        ),
    )
    parser.add_argument(
        "--softmax-deep-main-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Refresh the steady-layer Softmax main checkpoint through a "
            "second (3,3)-budget bootstrap program whose smaller depth "
            "frees two extra output levels. The PV/PairFM chain then "
            "arrives at the 6-limb threshold and skips the paired-input "
            "re-bootstrap entirely."
        ),
    )
    parser.add_argument(
        "--pairfm-bootstrap-mode",
        choices=("auto", "always", "off"),
        default="auto",
        help=(
            "Refresh PV output before PairFM always, never, or only when "
            "PairFM + W_O would leave fewer than two limbs."
        ),
    )
    parser.add_argument(
        "--pairfm-post-refresh-input-limbs",
        type=int,
        default=None,
        help=(
            "After a PairFM checkpoint, modulus-drop the refreshed carriers "
            "to this limb count without changing their CKKS scale. Defaults "
            "to the 6 limbs the PairFM + W_O suffix needs. Set 0 to disable "
            "the drop."
        ),
    )
    parser.add_argument(
        "--softmax-complex-pre-exp-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pair adjacent real score ciphertexts into complex values for "
            "the pre-exp checkpoint and fuse the exp domain map into score "
            "preprocessing."
        ),
    )
    parser.add_argument(
        "--softmax-reuse-round-square",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse each Alg2 round's row-sum square in the main update, "
            "computing y^2*lambda^2 instead of recomputing (y*lambda)^2."
        ),
    )
    parser.add_argument(
        "--softmax-direct-slim-inactive-fill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fill inactive slim polynomial rows directly after the disjoint "
            "slim pack, avoiding a redundant active-row mask/rescale."
        ),
    )
    parser.add_argument(
        "--softmax-fuse-slim-chebyshev-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fuse the inverse-square-root Chebyshev domain map into the "
            "existing slim-pack plaintext and fill inactive rows by addition."
        ),
    )
    parser.add_argument(
        "--adaptive-rmsnorm-slim-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refresh the single RMSNorm rsqrt slim ciphertext when needed, "
            "instead of refreshing the 16 materialized normalized outputs "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--batched-sparse-attention-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Batch sparse attention-input masks and reuse each input's "
            "hoisted baby rotations (GPU batch cap 32; default: enabled)."
        ),
    )
    parser.add_argument(
        "--fuse-rmsnorm-output-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Absorb RMSNorm gamma/paired-bootstrap scale into the following "
            "QKV or gate/up plaintext weights (default: enabled)."
        ),
    )
    parser.add_argument(
        "--fuse-silu-domain-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Absorb the SiLU Chebyshev alpha into gate plaintext weights "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--bootstrap-post-levels",
        type=int,
        help=(
            "Bootstrap output levels (default: depth-18 for the (4,4) program)."
        ),
    )
    parser.add_argument(
        "--softmax-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        default=2,
        help="Native bootstrap passes per Softmax refresh (default: 2).",
    )
    parser.add_argument(
        "--softmax-main-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Native passes for the 16/8-cipher main Softmax checkpoint "
            "only (default: 1)."
        ),
    )
    parser.add_argument(
        "--softmax-lambda-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Native passes for compact Alg2 lambda refreshes only "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--softmax-single-pass-match-two-pass-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a one-pass Softmax bootstrap, cheaply modulus-drop one "
            "limb so downstream kernels see the same level as Meta-BTS "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--residual-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Native bootstrap passes per residual refresh (default: 1; use "
            "2 for the higher-precision reference schedule)."
        ),
    )
    parser.add_argument(
        "--input-residual-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        help="Override native passes for the layer-input residual checkpoint.",
    )
    parser.add_argument(
        "--post-attention-residual-bootstrap-iterations",
        type=int,
        choices=(1, 2),
        help="Override native passes for the post-attention residual checkpoint.",
    )
    parser.add_argument(
        "--softmax-bootstrap-precision",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--residual-bootstrap-precision",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--residual-bootstrap-level-drop",
        type=int,
        default=0,
        help=(
            "Maximum modulus-only levels dropped after input/post-attention "
            "residual bootstraps. The layer clamps this per RMSNorm/consumer "
            "so attention and MLP retain their required limb budgets "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--preserve-prebootstrap-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep each pre-norm identity branch on its original ciphertext "
            "instead of materializing a residual copy from the RMSNorm "
            "bootstrap output (default: enabled)."
        ),
    )
    parser.add_argument(
        "--profile-device-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record allocated/reserved allocator memory and total device "
            "usage at major layer boundaries (default: disabled)."
        ),
    )
    parser.add_argument(
        "--profile-operator-breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep each measured layer's nested operator timing tree in the "
            "JSON report (default: disabled)."
        ),
    )
    parser.add_argument(
        "--plaintext-cache-gb",
        type=float,
        default=0.0,
        help=(
            "Session budget for cross-layer encoded public masks. Disabled "
            "by default because the current A100 two-layer profile is faster "
            "without it."
        ),
    )
    parser.add_argument(
        "--plaintext-cache-max-source-mb",
        type=float,
        default=1.0,
        help=(
            "Largest raw public-mask bundle eligible for the session cache; "
            "larger BSGS tables stay transient (default: 1 MiB)."
        ),
    )
    parser.add_argument(
        "--attention-rotation-chunk-size",
        type=int,
        choices=(1, 2, 4, 8, 16),
        default=8,
        help=(
            "Number of QK/PV shifts scheduled per fast-rotation chunk. "
            "PV emits two value rotations per shift, so 16 reaches the "
            "backend's 32-output fast-rotation limit."
        ),
    )
    parser.add_argument(
        "--linear-hoist-strategy",
        choices=("normal", "ext_normal", "ext_double_hoist"),
        default="normal",
        help=(
            "Hoisted-MAC schedule used by QKV, W_O, gate/up and down. "
            "The default is the B32 normal-hoist production "
            "schedule; this switch remains available for matched ablation."
        ),
    )
    parser.add_argument(
        "--decode-shifted-lanes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Decode only: run QKV and the MLP in the shifted-lane (SL) "
            "packing (LayerSchedule.decode_shifted_lanes). Prefill is "
            "untouched; W_O stays lane-replicated."
        ),
    )
    parser.add_argument(
        "--decode-sl-baby-steps",
        type=int,
        default=32,
        help="Baby steps of the SL decode linears (default 32).",
    )
    parser.add_argument(
        "--decode-sparse-bs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Decode only: prepare the log_slots=3 / 12 sparse bootstrap "
            "programs and route the period-1 slim RMSNorm rsqrt refresh and "
            "the merged-page Softmax total/filled/lambda refreshes through "
            "them (model.with_decode_sparse_bootstrap). --no-decode-sparse-bs "
            "keeps every Decode BS on the dense program."
        ),
    )
    parser.add_argument(
        "--poly-pack",
        choices=("production", "joint_tol_v1"),
        default="production",
        help=(
            "Coefficient pack. production keeps the Pass-B defaults. "
            "joint_tol_v1 selects the reduced RMSNorm/SiLU/Softmax freeze "
            "(silu_degree=236) with Softmax j1 hi-floor 8 for FHE. Explicit "
            "--rms-coeffs/--silu-coeffs/--softmax-coeffs override the pack."
        ),
    )
    parser.add_argument(
        "--rms-coeffs",
        type=Path,
        default=None,
        help=(
            "RMSNorm Chebyshev artifact. Default follows --poly-pack "
            "(production Pass-B, or rmsnorm_poly_coeffs_joint_tol_v1.json)."
        ),
    )
    parser.add_argument(
        "--silu-coeffs",
        type=Path,
        default=None,
        help=(
            "SiLU Chebyshev artifact. Default follows --poly-pack. "
            "Pass assets/polynomials/silu_poly_coeffs_silu_priority_v1.json "
            "for the SiLU-only reduced-degree experiment."
        ),
    )
    parser.add_argument(
        "--softmax-coeffs",
        type=Path,
        default=None,
        help=(
            "Softmax Alg2 artifact. Default follows --poly-pack "
            "(production, or joint_tol_v1 with j1 hi-floor 8)."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0 if layer0_preset else 1,
        help=(
            "Warmup repetitions. run_model.py defaults to one lightweight "
            "steady-layer warmup; run_layer0.py defaults to zero."
        ),
    )
    parser.add_argument(
        "--warmup-scope",
        choices=("auto", "steady-layer", "model"),
        default="auto",
        help=(
            "Warm one representative steady-state layer, the complete model, "
            "or choose steady-layer automatically for multi-layer runs."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--prefix-k",
        type=int,
        default=0,
        help=(
            "Plaintext attention-sink tokens packed as a C-form page "
            "(token columns 0..k-1).  Content stays length 128; RoPE starts "
            "at k.  Default 0 keeps the existing no-prefix Holmes graph."
        ),
    )
    parser.add_argument(
        "--prefix-token-id",
        type=int,
        default=198,
        help="Token id repeated k times for the offline prefix PP stack (Llama \\\\n).",
    )
    parser.add_argument(
        "--prefix-kv-cache",
        type=Path,
        help="Reuse an offline plaintext Prefix KV cache for this checkpoint.",
    )
    parser.add_argument(
        "--evict-rotation-cuda-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Drop the CUDA rotation-key cache before Softmax bootstrap, "
            "W_O, and MLP.  Needed on 80 GiB GPUs when rotation keys fill "
            "the device.  Leave off for Prefix / 80 GiB H100 Prefill "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--expect-token-id",
        type=int,
        help=(
            "Fail after writing the report if the predicted token ID differs. "
            "A fixture's expected_next_token_id is used when this is omitted."
        ),
    )
    parser.add_argument("--pad-side", choices=("left", "right"), default="left")
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--diagnostic-decrypt-layers",
        action="store_true",
        help=(
            "Debug only: decrypt each layer boundary and print FHE vs poly "
            "diffs (not hidden-state amplitude)."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--output-hidden-npy",
        type=Path,
        help="Diagnostic: save the final decrypted hidden state as a NumPy array.",
    )
    return parser


def resolve_poly_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill RMSNorm/SiLU/Softmax paths from ``--poly-pack`` unless overridden."""

    pack = str(getattr(args, "poly_pack", "production") or "production")
    if pack == "joint_tol_v1":
        rms, silu, softmax = JOINT_TOL_V1_RMS, JOINT_TOL_V1_SILU, JOINT_TOL_V1_SOFTMAX
    else:
        rms, silu, softmax = DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX
    if args.rms_coeffs is None:
        args.rms_coeffs = rms
    if args.silu_coeffs is None:
        args.silu_coeffs = silu
    if args.softmax_coeffs is None:
        prefix_k = int(getattr(args, "prefix_k", 0) or 0)
        if pack == "production" and prefix_k == 2:
            args.softmax_coeffs = PREFIX_K2_SOFTMAX
        else:
            args.softmax_coeffs = softmax
    return args


__all__ = [
    "build_parser",
    "resolve_poly_paths",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG",
    "DEFAULT_RMS",
    "DEFAULT_SILU",
    "DEFAULT_SOFTMAX",
    "PREFIX_K2_SOFTMAX",
    "JOINT_TOL_V1_RMS",
    "JOINT_TOL_V1_SILU",
    "JOINT_TOL_V1_SOFTMAX",
    "DEFAULT_INPUT_IDS",
]
