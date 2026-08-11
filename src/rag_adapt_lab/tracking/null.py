from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tracker


class NullTracker(Tracker):
    def start_run(self, *, name: str | None = None, config: dict[str, Any] | None = None) -> None:
        return None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_artifact(self, path: str | Path, *, name: str, artifact_type: str) -> None:
        return None

    def finish(self) -> None:
        return None
