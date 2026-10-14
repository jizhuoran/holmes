"""Thin wrapper around official QuaRot fake_quant rotation utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from config.paths import RELEASE_ROOT
from transformers import PreTrainedModel

QUAROT_FAKE_QUANT_DIR = RELEASE_ROOT / "tools" / "third_party" / "QuaRot" / "fake_quant"
RotateMode = Literal["hadamard", "random"]


def quarot_fake_quant_dir() -> Path:
    if not QUAROT_FAKE_QUANT_DIR.is_dir():
        raise FileNotFoundError(
            f"QuaRot fake_quant not found at {QUAROT_FAKE_QUANT_DIR}. "
            "Clone https://github.com/spcl/QuaRot into tools/third_party/QuaRot."
        )
    return QUAROT_FAKE_QUANT_DIR


def _ensure_quarot_import_path() -> None:
    path = str(quarot_fake_quant_dir())
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass(frozen=True)
class QuaRotRotateConfig:
    rotate_mode: RotateMode = "hadamard"
    seed: int = 0
    fuse_layer_norms: bool = True
    # Step 1a debug: skip head-internal Hadamard fusion
    skip_ov_proj: bool = False
    skip_down_had: bool = False
    device: Optional[str] = None


def _patch_quarot_runtime(device: torch.device) -> None:
    _ensure_quarot_import_path()
    import utils  # noqa: WPS433 — official QuaRot module

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    utils.DEV = device


def _move_model_to_device(model: PreTrainedModel, device: torch.device) -> None:
    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")
    model.to(device)


def apply_official_quarot_rotation(
    model: PreTrainedModel,
    cfg: QuaRotRotateConfig = QuaRotRotateConfig(),
) -> PreTrainedModel:
    """
    Apply official fuse_layer_norms + rotate_model on an in-memory LlamaForCausalLM.

    Official rotation temporarily moves weight shards to CPU; this wrapper moves the
    full model back to `cfg.device` after rotation.
    """
    _ensure_quarot_import_path()
    import rotation_utils  # noqa: WPS433
    import transformers

    device = torch.device(cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    _patch_quarot_runtime(device)
    transformers.set_seed(cfg.seed)

    if cfg.fuse_layer_norms:
        rotation_utils.fuse_layer_norms(model)

    args = type("RotateArgs", (), {"rotate_mode": cfg.rotate_mode})()

    if cfg.skip_ov_proj and cfg.skip_down_had:
        _rotate_global_q_only(model, args, rotation_utils)
    elif cfg.skip_down_had:
        _rotate_without_down_had(model, args, rotation_utils)
    elif cfg.skip_ov_proj:
        _rotate_without_ov_proj(model, args, rotation_utils)
    else:
        rotation_utils.rotate_model(model, args)

    _move_model_to_device(model, device)
    return model


def apply_official_online_had_hooks(
    model: PreTrainedModel,
    *,
    fp32_had: bool = False,
) -> PreTrainedModel:
    """
    Mirror fake_quant/main.py after rotate_model: wrap Linears with ActQuantWrapper
    and enable online Hadamard on down_proj / o_proj (required for logits equivalence).
    """
    _ensure_quarot_import_path()
    import hadamard_utils  # noqa: WPS433
    import quant_utils  # noqa: WPS433

    _add_actquant_safe(model, quant_utils=quant_utils)
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" in name:
            had_k, k = hadamard_utils.get_hadK(model.config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_k
            qlayers[name].K = k
            qlayers[name].fp32_had = fp32_had
        if "o_proj" in name:
            had_k, k = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_k
            qlayers[name].K = k
            qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
            qlayers[name].fp32_had = fp32_had
    return model


def _add_actquant_safe(module, *, quant_utils) -> None:
    """Wrap Linear children without probing every dynamic module attribute.

    QuaRot's original helper iterates over ``dir(module)`` and calls
    ``getattr`` on every entry.  Newer Transformers exposes lazy/deprecated
    RoPE cache attributes in ``dir(LlamaRotaryEmbedding)``; probing those
    attributes can remove one cache before the second probe and makes a
    checkpoint-only conversion fail.  Traversing registered children is both
    sufficient for Linear replacement and independent of such properties.
    """

    linear_types = (
        torch.nn.Linear,
        quant_utils.transformers.models.falcon.modeling_falcon.FalconLinear,
    )
    for name, child in tuple(module.named_children()):
        if isinstance(child, quant_utils.ActQuantWrapper):
            continue
        if type(child) in linear_types:
            setattr(module, name, quant_utils.ActQuantWrapper(child))
        else:
            _add_actquant_safe(child, quant_utils=quant_utils)


def apply_official_quarot_pipeline(
    model: PreTrainedModel,
    cfg: QuaRotRotateConfig = QuaRotRotateConfig(),
    *,
    online_had: bool = True,
    fp32_had: bool = False,
) -> PreTrainedModel:
    """fuse_layer_norms + rotate_model + optional online Hadamard hooks (official main.py path)."""
    apply_official_quarot_rotation(model, cfg)
    if online_had and not (cfg.skip_ov_proj or cfg.skip_down_had):
        apply_official_online_had_hooks(model, fp32_had=fp32_had)
    return model


def apply_quarot_full_fuse_pipeline(
    model: PreTrainedModel,
    cfg: QuaRotRotateConfig = QuaRotRotateConfig(),
    *,
    fp32_had: bool = False,
) -> PreTrainedModel:
    """QuaRot rotation with online Hadamard fully fused into weights (FHE-friendly path).

  Equivalent to ``apply_official_quarot_pipeline(..., online_had=True)`` but without
  runtime FWHT on ``o_proj`` / ``down_proj``.
    """
    from model.quarot_full_fuse import apply_full_fuse_to_model

    apply_official_quarot_rotation(model, cfg)
    if not (cfg.skip_ov_proj or cfg.skip_down_had):
        apply_full_fuse_to_model(model)
        # ActQuantWrapper structure without online Hadamard flags.
        apply_official_online_had_hooks(model, fp32_had=fp32_had)
        _disable_online_had_hooks(model)
    return model


def disable_online_had_hooks(model: PreTrainedModel) -> None:
    """Turn off online Hadamard on ActQuantWrapper layers after weight fusion."""
    _ensure_quarot_import_path()
    import quant_utils  # noqa: WPS433

    for qlayer in quant_utils.find_qlayers(model).values():
        qlayer.online_full_had = False
        qlayer.online_partial_had = False


def _disable_online_had_hooks(model: PreTrainedModel) -> None:
    disable_online_had_hooks(model)


def apply_quarot_runtime_structure(
    model: PreTrainedModel,
    *,
    fuse_layer_norms: bool = True,
    fp32_had: bool = False,
) -> PreTrainedModel:
    """
    Build the QuaRot forward structure without recomputing weight rotation.

    Used when loading a saved quarot checkpoint: fuse_layer_norms + ActQuantWrapper hooks.
    """
    _ensure_quarot_import_path()
    import rotation_utils  # noqa: WPS433

    if fuse_layer_norms:
        rotation_utils.fuse_layer_norms(model)
    apply_official_online_had_hooks(model, fp32_had=fp32_had)
    return model


def save_quarot_checkpoint(
    model: PreTrainedModel,
    out_dir: Path,
    *,
    source_model_path: str,
    cfg: QuaRotRotateConfig,
    fp32_had: bool = False,
    online_had: bool = True,
) -> Path:
    """Save rotated weights + manifest for fast reload."""
    import json
    import shutil

    from config.paths import QUAROT_MANIFEST_FILENAME, QUAROT_STATE_FILENAME
    from transformers import AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / QUAROT_STATE_FILENAME
    torch.save(model.state_dict(), state_path)

    manifest = {
        "format": "llama3_8b_quarot_v1",
        "source_model_path": str(source_model_path),
        "rotate_mode": cfg.rotate_mode,
        "seed": cfg.seed,
        "fuse_layer_norms": cfg.fuse_layer_norms,
        "skip_ov_proj": cfg.skip_ov_proj,
        "skip_down_had": cfg.skip_down_had,
        "online_had": online_had,
        "fp32_had": fp32_had,
        "torch_dtype": "float16",
    }
    manifest_path = out_dir / QUAROT_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(source_model_path)
    tokenizer.save_pretrained(out_dir)

    for name in ("config.json", "generation_config.json"):
        src = Path(source_model_path) / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)

    return out_dir


def load_quarot_checkpoint(
    quarot_dir: Path,
    *,
    device: Optional[str] = None,
) -> PreTrainedModel:
    """Load a saved QuaRot model (structure + state_dict, no re-rotation)."""
    import json

    from config.paths import QUAROT_MANIFEST_FILENAME, QUAROT_STATE_FILENAME
    from transformers import AutoModelForCausalLM

    quarot_dir = Path(quarot_dir)
    manifest_path = quarot_dir / QUAROT_MANIFEST_FILENAME
    state_path = quarot_dir / QUAROT_STATE_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing state dict: {state_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source_model_path"]
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.float16,
        device_map=None,
    )
    model.eval()
    apply_quarot_runtime_structure(
        model,
        fuse_layer_norms=manifest.get("fuse_layer_norms", True),
        fp32_had=manifest.get("fp32_had", False),
    )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(dev)
    return model


def _rotate_global_q_only(model, args, rotation_utils) -> None:
    """Step 1a: global Q only (no rotate_ov_proj / no down exact had)."""
    import model_utils  # noqa: WPS433
    import tqdm
    import utils  # noqa: WPS433

    q = rotation_utils.get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    model_type = model_utils.model_type_extractor(model)
    rotation_utils.rotate_embeddings(model, q)
    rotation_utils.rotate_head(model, q)
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx in tqdm.tqdm(range(len(layers)), unit="layer", desc="Rotating (global Q only)"):
        rotation_utils.rotate_attention_inputs(layers[idx], q, model_type)
        rotation_utils.rotate_attention_output(layers[idx], q, model_type)
        rotation_utils.rotate_mlp_input(layers[idx], q, model_type)
        if model_type == model_utils.LLAMA_MODEL:
            w = layers[idx].mlp.down_proj
            dtype = w.weight.data.dtype
            w_ = w.weight.data.to(device=utils.DEV, dtype=torch.float64)
            w.weight.data = torch.matmul(q.T, w_).to(device="cpu", dtype=dtype)
            if w.bias is not None:
                b = w.bias.data.to(device=utils.DEV, dtype=torch.float64)
                w.bias.data = torch.matmul(q.T, b).to(device="cpu", dtype=dtype)


def _rotate_without_ov_proj(model, args, rotation_utils) -> None:
    import model_utils  # noqa: WPS433
    import tqdm

    q = rotation_utils.get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    model_type = model_utils.model_type_extractor(model)
    rotation_utils.rotate_embeddings(model, q)
    rotation_utils.rotate_head(model, q)
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx in tqdm.tqdm(range(len(layers)), unit="layer", desc="Rotating (no ov_proj)"):
        rotation_utils.rotate_attention_inputs(layers[idx], q, model_type)
        rotation_utils.rotate_attention_output(layers[idx], q, model_type)
        rotation_utils.rotate_mlp_input(layers[idx], q, model_type)
        rotation_utils.rotate_mlp_output(layers[idx], q, model_type)


def _rotate_without_down_had(model, args, rotation_utils) -> None:
    import model_utils  # noqa: WPS433
    import tqdm

    q = rotation_utils.get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    model_type = model_utils.model_type_extractor(model)
    rotation_utils.rotate_embeddings(model, q)
    rotation_utils.rotate_head(model, q)
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx in tqdm.tqdm(range(len(layers)), unit="layer", desc="Rotating (no down had)"):
        rotation_utils.rotate_attention_inputs(layers[idx], q, model_type)
        rotation_utils.rotate_attention_output(layers[idx], q, model_type)
        rotation_utils.rotate_mlp_input(layers[idx], q, model_type)
        rotation_utils.rotate_mlp_output(layers[idx], q, model_type)
        rotation_utils.rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)
