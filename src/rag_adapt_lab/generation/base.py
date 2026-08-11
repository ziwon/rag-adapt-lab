from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class GenerationResult:
    """One decoded answer plus an auditable inference-stage breakdown.

    ``latency_s`` remains as a deprecated alias for
    ``model_generate_latency_s`` through schema version 2.
    """

    text: str
    raw_text: str | None = None
    reasoning: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    answer_tokens: int | None = None
    prompt_build_latency_s: float | None = None
    chat_template_latency_s: float | None = None
    tokenization_latency_s: float | None = None
    device_transfer_latency_s: float | None = None
    model_generate_latency_s: float | None = None
    decode_latency_s: float | None = None
    inference_e2e_latency_s: float | None = None
    batch_size: int = 1
    generation_mode: str = "sequential"
    latency_s: float | None = None

    def __post_init__(self) -> None:
        if self.model_generate_latency_s is None:
            self.model_generate_latency_s = self.latency_s
        if self.latency_s is None:
            self.latency_s = self.model_generate_latency_s
        if self.raw_text is None:
            self.raw_text = self.text
        if self.answer_tokens is None:
            self.answer_tokens = self.output_tokens


class Generator(ABC):
    @abstractmethod
    def generate(self, *, question: str, contexts: list[str] | None = None) -> GenerationResult: ...

    def reset_runtime_metrics(self) -> None:
        """Reset optional backend-specific runtime counters."""
        return None

    def peak_memory_gb(self) -> float | None:
        """Deprecated alias for peak allocated accelerator memory."""
        return None

    def peak_allocated_vram_gb(self) -> float | None:
        return self.peak_memory_gb()

    def peak_reserved_vram_gb(self) -> float | None:
        return None

    def close(self) -> None:
        """Release optional backend resources."""
        return None
