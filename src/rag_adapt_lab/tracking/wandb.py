from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Tracker


class WandbTracker(Tracker):
    def __init__(self, project: str | None = None, entity: str | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install W&B extras: pip install -e '.[wandb]'") from exc
        self.wandb = wandb
        self.project = project or os.getenv("WANDB_PROJECT", "rag-adapt-lab")
        self.entity = entity or os.getenv("WANDB_ENTITY") or None
        self.run = None

    def start_run(self, *, name: str | None = None, config: dict[str, Any] | None = None) -> None:
        self.run = self.wandb.init(project=self.project, entity=self.entity, name=name, config=config or {})

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.wandb.log(metrics, step=step)

    def log_artifact(self, path: str | Path, *, name: str, artifact_type: str) -> None:
        artifact = self.wandb.Artifact(name=name, type=artifact_type)
        path = Path(path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.wandb.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None:
            self.wandb.finish()
            self.run = None
