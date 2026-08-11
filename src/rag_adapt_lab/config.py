from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


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
