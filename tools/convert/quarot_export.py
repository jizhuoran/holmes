from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .hadamard import apply_full_fuse_to_state_dict

QUAROT_MANIFEST_FILENAME = "quarot_manifest.json"
QUAROT_STATE_FILENAME = "quarot_model.pt"

EXPORT_METADATA_FILENAME = "export_metadata.json"
FULL_FUSE_MANIFEST_KEY = "full_fuse"

_V_PROJ_TRANSPOSE_BUG_L0_PREFIX = np.asarray(
    [6.7234039e-04, -8.8214874e-05, 4.1913986e-04, 4.7087669e-06, -5.7935715e-04],
    dtype=np.float32,
)


def undo_v_proj_transpose_reshape(weight: np.ndarray) -> np.ndarray:
    """Undo a transposed ``v_proj`` layout if present."""

    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"v_proj weight must be 2-D, got {weight.shape}.")
    return np.ascontiguousarray(weight.reshape(weight.shape[1], weight.shape[0]).T)


def v_proj_has_transpose_reshape_bug(weight: np.ndarray) -> bool:
    """Return True when ``weight`` matches the known L0 corrupted prefix."""

    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape != (1024, 4096):
        return False
    return bool(
        np.allclose(
            weight[0, :5],
            _V_PROJ_TRANSPOSE_BUG_L0_PREFIX,
            rtol=0.0,
            atol=1e-8,
        )
    )


def _is_exportable_hf_tensor_name(name: str) -> bool:
    """Drop ActQuant ``*.module.*`` / quantizer sidecars; keep HF Linear / Norm weights."""

    if "quantizer" in name or ".module." in name:
        return False
    return name.endswith(".weight") or name.endswith(".bias")


def load_quarot_manifest(quarot_dir: str | Path) -> dict[str, Any]:
    """Read ``quarot_manifest.json`` from a Llama-3-8B QuaRot directory."""

    manifest_path = Path(quarot_dir) / QUAROT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing QuaRot manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _state_dict_from_checkpoint(quarot_dir: Path) -> dict[str, np.ndarray]:
    """Load ``quarot_model.pt`` tensors as float32 NumPy arrays."""

    state_path = quarot_dir / QUAROT_STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(
            f"Missing QuaRot state dict at {state_path}. "
            "Run the bundled Llama-3-8B rotate_checkpoint first."
        )

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "export_quarot_safetensors requires PyTorch to read quarot_model.pt."
        ) from exc

    try:
        raw = torch.load(state_path, map_location="cpu", weights_only=True)
    except RuntimeError as exc:
        if "torch serialization is disabled" in str(exc):
            raise RuntimeError(
                "torch.load is unavailable in the EasyFHE env (serialization disabled). "
                "Run export_quarot_weights.py with a standard PyTorch Python, e.g. "
                "the standard PyTorch conversion environment."
            ) from exc
        raise
    if not isinstance(raw, dict):
        raise TypeError(f"Expected state dict mapping, got {type(raw)}.")

    arrays: dict[str, np.ndarray] = {}
    for name, tensor in raw.items():
        # Snap to the fp16 numeric grid when exporting from fp16 QuaRot checkpoints so
        # Keep exported NumPy matmuls bit-aligned with the fp16 source model.
        arr = tensor.detach().cpu().float().numpy()
        if tensor.dtype == torch.float16:
            arr = arr.astype(np.float16).astype(np.float32)
        arrays[str(name)] = np.asarray(arr, dtype=np.float32)
    return arrays


def is_quarot_weight_dir(model_dir: str | Path) -> bool:
    """Return True when ``model_dir`` carries a QuaRot manifest sidecar."""

    return (Path(model_dir) / QUAROT_MANIFEST_FILENAME).is_file()


def load_export_metadata(model_dir: str | Path) -> dict[str, Any] | None:
    path = Path(model_dir) / EXPORT_METADATA_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_full_fused_quarot_dir(model_dir: str | Path) -> bool:
    """True when safetensors were exported with offline online-Hadamard fusion."""

    meta = load_export_metadata(model_dir)
    if meta is None:
        return False
    return bool(meta.get(FULL_FUSE_MANIFEST_KEY))


def recommended_online_had(model_dir: str | Path) -> bool:
    """FHE inference should use ``online_had=False`` when weights are full-fused."""

    return not is_full_fused_quarot_dir(model_dir)


def export_quarot_safetensors(
    quarot_dir: str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    full_fuse: bool = True,
    num_layers: int = 32,
    num_heads: int = 32,
    had_dim: int = 128,
    intermediate_size: int = 14336,
) -> Path:
    """Write QuaRot rotated weights as HF-compatible ``model.safetensors`` for ``WeightLoader``.

    Tensor names and shapes are unchanged from HuggingFace Llama; only values differ.
    Also copies ``config.json``, tokenizer files, and ``quarot_manifest.json`` sidecar.
    """

    quarot_dir = Path(quarot_dir)
    out_dir = Path(out_dir)
    manifest = load_quarot_manifest(quarot_dir)
    source_model = Path(str(manifest.get("source_model_path", quarot_dir)))

    if out_dir.exists():
        if not overwrite and any(out_dir.iterdir()):
            raise FileExistsError(f"{out_dir} exists and is not empty. Pass overwrite=True.")
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _state_dict_from_checkpoint(quarot_dir)
    state = {name: arr for name, arr in state.items() if _is_exportable_hf_tensor_name(name)}
    if full_fuse:
        state = apply_full_fuse_to_state_dict(
            state,
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            had_dim=int(had_dim),
            intermediate_size=int(intermediate_size),
        )

    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise ImportError("export_quarot_safetensors requires the safetensors package.") from exc

    # Store fp16 weights natively to avoid float32 header drift on reload.
    # Keep arrays C-contiguous at the declared HF shape.
    save_tensors: dict[str, np.ndarray] = {}
    for name, arr in state.items():
        arr = np.ascontiguousarray(arr)
        save_tensors[name] = arr.astype(np.float16) if arr.dtype == np.float32 else arr
    save_file(save_tensors, out_dir / "model.safetensors")

    # Copy HF config / tokenizer from source or quarot checkpoint directory.
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        for base in (quarot_dir, source_model):
            src = base / name
            if src.is_file():
                shutil.copy2(src, out_dir / name)
                break

    shutil.copy2(quarot_dir / QUAROT_MANIFEST_FILENAME, out_dir / QUAROT_MANIFEST_FILENAME)

    export_manifest = dict(manifest)
    export_manifest["online_had"] = False if full_fuse else bool(manifest.get("online_had", True))
    export_manifest[FULL_FUSE_MANIFEST_KEY] = bool(full_fuse)

    metadata = {
        "format": "llama3_8b_feature_major_quarot_v1",
        "source_quarot_dir": str(quarot_dir.resolve()),
        FULL_FUSE_MANIFEST_KEY: bool(full_fuse),
        "manifest": export_manifest,
        "num_tensors": len(state),
        "runtime_weight_layout": "ordinary_out_in",
        "he_packing": "prepared_online_by_linear_operator",
    }
    (out_dir / EXPORT_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_dir


def fuse_quarot_safetensors_inplace(
    model_dir: str | Path,
    *,
    num_layers: int = 32,
    num_heads: int = 32,
    had_dim: int = 128,
    intermediate_size: int = 14336,
) -> Path:
    """Apply full-fuse to an existing safetensors directory (rewrite model.safetensors)."""

    model_dir = Path(model_dir)
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Missing {safetensors_path}")

    try:
        from safetensors.numpy import load_file, save_file
    except ImportError as exc:
        raise ImportError("fuse_quarot_safetensors_inplace requires safetensors.") from exc

    raw = load_file(str(safetensors_path))
    state = {name: np.asarray(arr, dtype=np.float32) for name, arr in raw.items()}
    fused = apply_full_fuse_to_state_dict(
        state,
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        had_dim=int(had_dim),
        intermediate_size=int(intermediate_size),
    )
    save_tensors = {
        name: arr.astype(np.float16) if arr.dtype == np.float32 else arr
        for name, arr in fused.items()
    }
    save_file(save_tensors, safetensors_path)

    meta = load_export_metadata(model_dir) or {
        "format": "llama3_8b_feature_major_quarot_v1",
        "source_quarot_dir": str(model_dir),
        "manifest": {},
        "num_tensors": len(fused),
        "runtime_weight_layout": "ordinary_out_in",
        "he_packing": "prepared_online_by_linear_operator",
    }
    meta[FULL_FUSE_MANIFEST_KEY] = True
    manifest = dict(meta.get("manifest") or {})
    manifest["online_had"] = False
    manifest[FULL_FUSE_MANIFEST_KEY] = True
    meta["manifest"] = manifest
    meta["num_tensors"] = len(fused)
    (model_dir / EXPORT_METADATA_FILENAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return model_dir
