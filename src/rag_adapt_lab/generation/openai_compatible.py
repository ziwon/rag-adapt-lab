from __future__ import annotations

import time

from .base import GenerationResult, Generator
from .prompts import format_rag_user_prompt


class OpenAICompatibleGenerator(Generator):
    """Minimal runner for vLLM/SGLang/other OpenAI-compatible local endpoints."""

    def __init__(self, *, base_url: str, model: str, api_key: str = "local") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install `openai` to use the OpenAI-compatible runner.") from exc
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, *, question: str, contexts: list[str] | None = None) -> GenerationResult:
        prompt = format_rag_user_prompt(question=question, contexts=contexts or [])
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        latency = time.perf_counter() - start
        usage = response.usage
        return GenerationResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_s=latency,
        )
