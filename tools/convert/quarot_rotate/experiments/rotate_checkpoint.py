"""
Apply official QuaRot rotation to Meta-Llama-3-8B and save checkpoint for reuse.

Usage (from tools/convert/quarot_rotate with PYTHONPATH set):
  python -m experiments.rotate_checkpoint --local-dir ../../../assets/weights/Meta-Llama-3-8B
  python -m experiments.rotate_checkpoint --out-dir ../../../assets/weights/Meta-Llama-3-8B-QuaRot-fp16
  python -m experiments.rotate_checkpoint --verify-only  # load saved ckpt and smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import torch
from config.paths import (
    ARTIFACTS_DIR,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_LOCAL_WEIGHTS_DIR,
    DEFAULT_QUAROT_WEIGHTS_DIR,
    default_run_id,
)
from model.llama_baseline import resolve_model_path
from model.quarot_official import (
    QuaRotRotateConfig,
    apply_official_quarot_pipeline,
    load_quarot_checkpoint,
    save_quarot_checkpoint,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "The capital of France is"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_baseline(path: str, device: torch.device) -> AutoModelForCausalLM:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map=None,
    )
    model.eval()
    model.to(device)
    return model


def smoke_forward(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
) -> Dict[str, Any]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    last = logits[0, -1]
    top_id = int(last.argmax().item())
    return {
        "prompt": prompt,
        "forward_ms": round(elapsed_ms, 2),
        "logits_shape": list(logits.shape),
        "last_token_id": top_id,
        "last_token": tokenizer.decode([top_id]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate Llama-2-7B with official QuaRot and save")
    parser.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--local-dir", type=str, default=None, help="Source HF weights dir")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_QUAROT_WEIGHTS_DIR),
        help="Output directory for quarot checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate-mode", choices=["hadamard", "random"], default="hadamard")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp32-had", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip rotation; load existing checkpoint and run smoke test",
    )
    parser.add_argument("--skip-save-if-exists", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    device = torch.device(args.device)
    source_path = resolve_model_path(args.model_id, args.local_dir, DEFAULT_LOCAL_WEIGHTS_DIR)
    out_dir = Path(args.out_dir).expanduser().resolve()

    tokenizer = AutoTokenizer.from_pretrained(source_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    manifest_path = out_dir / "quarot_manifest.json"
    if args.verify_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No checkpoint at {out_dir}; run without --verify-only first")
        logger.info("Loading saved QuaRot checkpoint from %s", out_dir)
        model = load_quarot_checkpoint(out_dir, device=args.device)
        metrics = smoke_forward(model, tokenizer, device, args.prompt)
        logger.info("Smoke test: %s", json.dumps(metrics, indent=2))
        return

    if args.skip_save_if_exists and manifest_path.is_file():
        logger.info("Checkpoint already exists at %s; use --verify-only to test", out_dir)
        return

    logger.info("Source model: %s", source_path)
    logger.info("Output dir:   %s", out_dir)
    logger.info("rotate_mode=%s seed=%s fp32_had=%s", args.rotate_mode, args.seed, args.fp32_had)

    t_load = time.perf_counter()
    model = load_baseline(source_path, device)
    logger.info("Loaded baseline in %.1fs", time.perf_counter() - t_load)

    cfg = QuaRotRotateConfig(
        rotate_mode=args.rotate_mode,
        seed=args.seed,
        device=args.device,
    )
    t_rot = time.perf_counter()
    apply_official_quarot_pipeline(model, cfg, fp32_had=args.fp32_had)
    logger.info("QuaRot pipeline done in %.1fs", time.perf_counter() - t_rot)

    rot_metrics = smoke_forward(model, tokenizer, device, args.prompt)
    logger.info("Rotated smoke: %s", json.dumps(rot_metrics, indent=2))

    t_save = time.perf_counter()
    save_quarot_checkpoint(
        model,
        out_dir,
        source_model_path=source_path,
        cfg=cfg,
        fp32_had=args.fp32_had,
    )
    logger.info("Saved checkpoint to %s in %.1fs", out_dir, time.perf_counter() - t_save)

    # Quick reload verification
    del model
    torch.cuda.empty_cache()
    reloaded = load_quarot_checkpoint(out_dir, device=args.device)
    reload_metrics = smoke_forward(reloaded, tokenizer, device, args.prompt)
    logger.info("Reload smoke: %s", json.dumps(reload_metrics, indent=2))
    match = reload_metrics["last_token_id"] == rot_metrics["last_token_id"]
    if not match:
        raise RuntimeError("Reloaded checkpoint argmax mismatch")

    summary = {
        "source_model_path": source_path,
        "out_dir": str(out_dir),
        "rotate_mode": args.rotate_mode,
        "seed": args.seed,
        "fp32_had": args.fp32_had,
        "rotated_smoke": rot_metrics,
        "reload_smoke": reload_metrics,
        "reload_argmax_match": match,
    }
    summary_path = ARTIFACTS_DIR / "quarot_rotate" / f"rotate_{default_run_id()}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote summary %s", summary_path)
    logger.info("Done. Next: python -m experiments.rotate_checkpoint --verify-only --out-dir %s", out_dir)


if __name__ == "__main__":
    main()
