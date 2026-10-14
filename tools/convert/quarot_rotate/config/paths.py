"""Path conventions for the QuaRot rotation stage (open-source package)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# tools/convert/quarot_rotate/config/paths.py -> project root is parents[4]
CONVERT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = CONVERT_ROOT

ARTIFACTS_DIR = REPO_ROOT / "weights" / "_rotate_artifacts"
LOGS_ROOT = REPO_ROOT / "weights" / "_rotate_logs"

DEFAULT_HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DEFAULT_LOCAL_WEIGHTS_DIR = REPO_ROOT / "weights" / "Meta-Llama-3-8B"
DEFAULT_QUAROT_WEIGHTS_DIR = REPO_ROOT / "weights" / "Meta-Llama-3-8B-QuaRot-fp16"

# Used by quarot_official to locate tools/third_party/QuaRot.
RELEASE_ROOT = REPO_ROOT

QUAROT_STATE_FILENAME = "quarot_model.pt"
QUAROT_MANIFEST_FILENAME = "quarot_manifest.json"


@dataclass(frozen=True)
class RunPaths:
    experiment: str
    run_id: str

    @property
    def logs_dir(self) -> Path:
        return LOGS_ROOT / self.experiment / self.run_id

    @property
    def artifacts_dir(self) -> Path:
        return ARTIFACTS_DIR / self.experiment / self.run_id

    def ensure_dirs(self) -> RunPaths:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self


def default_run_id(prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def make_run_paths(experiment: str, run_id: Optional[str] = None) -> RunPaths:
    rid = run_id or default_run_id(prefix=experiment)
    return RunPaths(experiment=experiment, run_id=rid)
