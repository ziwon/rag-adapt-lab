from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_HF_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_relative(config_path: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = Path(config_path).resolve().parent / path
    if candidate.exists():
        return candidate
    return path


def require_pinned_hf_revision(revision: object, *, model_id: str) -> str:
    """Return a Hugging Face commit SHA or reject a mutable model revision."""
    if not isinstance(revision, str) or _HF_COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError(
            f"Model {model_id!r} must use an immutable 40-character Hugging Face commit SHA; "
            f"got revision {revision!r}"
        )
    return revision


def validate_hf_model_config(config: dict[str, Any]) -> tuple[str, str]:
    """Validate the reproducibility and remote-code policy for a model config."""
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Model config must contain a non-empty model_id")
    revision = require_pinned_hf_revision(config.get("revision"), model_id=model_id)
    trust_remote_code = config.get("trust_remote_code", False)
    if not isinstance(trust_remote_code, bool):
        raise ValueError("trust_remote_code must be a boolean")
    if trust_remote_code:
        raise ValueError(
            f"Model {model_id!r} requests trust_remote_code=true; remote model code is disabled"
        )
    return model_id, revision
