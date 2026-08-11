from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tracker(ABC):
    @abstractmethod
    def start_run(self, *, name: str | None = None, config: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str | Path, *, name: str, artifact_type: str) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...
