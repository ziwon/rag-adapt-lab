from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None


class Generator(ABC):
    @abstractmethod
    def generate(self, *, question: str, contexts: list[str] | None = None) -> GenerationResult: ...
