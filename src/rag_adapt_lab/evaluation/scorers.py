from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Scorer(ABC):
    name: str

    @abstractmethod
    def score(
        self, *, question: str, answer: str, reference: str | None, contexts: list[str]
    ) -> dict[str, Any]: ...


class NoOpScorer(Scorer):
    name = "noop"

    def score(
        self, *, question: str, answer: str, reference: str | None, contexts: list[str]
    ) -> dict[str, Any]:
        return {}
