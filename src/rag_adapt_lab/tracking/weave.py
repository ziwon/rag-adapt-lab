from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class WeaveTracer:
    def __init__(self, project: str | None = None) -> None:
        try:
            import weave
        except ImportError as exc:
            raise RuntimeError("Install W&B extras: pip install -e '.[wandb]'") from exc
        self.weave = weave
        self.project = project or os.getenv("WEAVE_PROJECT", "rag-adapt-lab")
        self._initialized = False

    def init(self) -> None:
        if not self._initialized:
            self.weave.init(self.project)
            self._initialized = True

    def op(self, fn: F) -> F:
        self.init()
        return self.weave.op(fn)  # type: ignore[return-value]
