from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .generation import exact_match, normalize_text, token_f1

ScoreValue = float | int | bool | str | None
ScoreResult = dict[str, ScoreValue]


class Scorer(ABC):
    name: str
    version = "1"

    @abstractmethod
    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str | None,
        contexts: list[str],
        references: Sequence[str] | None = None,
        context_ids: Sequence[str] | None = None,
        relevant_doc_ids: Sequence[str] | None = None,
    ) -> ScoreResult: ...

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}


class NoOpScorer(Scorer):
    name = "noop"

    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str | None,
        contexts: list[str],
        references: Sequence[str] | None = None,
        context_ids: Sequence[str] | None = None,
        relevant_doc_ids: Sequence[str] | None = None,
    ) -> ScoreResult:
        return {}


class DeterministicCorrectnessScorer(Scorer):
    """Reference-based correctness that always accompanies optional judges."""

    name = "deterministic-correctness"

    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str | None,
        contexts: list[str],
        references: Sequence[str] | None = None,
        context_ids: Sequence[str] | None = None,
        relevant_doc_ids: Sequence[str] | None = None,
    ) -> ScoreResult:
        candidates = [item for item in (references or []) if item.strip()]
        if not candidates and reference:
            candidates = [reference]
        if not candidates:
            return {"answer_correctness": None, "answer_exact_match": None}
        return {
            "answer_correctness": max(token_f1(answer, item) for item in candidates),
            "answer_exact_match": max(exact_match(answer, item) for item in candidates),
        }


class LexicalGroundednessScorer(Scorer):
    """Deterministic support heuristic for offline and regression evaluation.

    This scorer is intentionally lexical and must not be presented as a
    semantic judge. It provides a reproducible lower-cost signal alongside
    optional model-based groundedness evaluation.
    """

    name = "lexical-groundedness"
    version = "1"

    def __init__(self, *, claim_support_threshold: float = 0.5) -> None:
        if not 0.0 <= claim_support_threshold <= 1.0:
            raise ValueError("claim_support_threshold must be between 0 and 1")
        self.claim_support_threshold = claim_support_threshold

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "claim_support_threshold": self.claim_support_threshold,
            "method": "normalized answer-token overlap with retrieved contexts",
        }

    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str | None,
        contexts: list[str],
        references: Sequence[str] | None = None,
        context_ids: Sequence[str] | None = None,
        relevant_doc_ids: Sequence[str] | None = None,
    ) -> ScoreResult:
        if not contexts:
            return {"groundedness": None, "unsupported_claim_rate": None}
        context_tokens = set(normalize_text(" ".join(contexts)).split())
        answer_tokens = normalize_text(answer).split()
        if not answer_tokens:
            return {"groundedness": 0.0, "unsupported_claim_rate": 0.0}

        groundedness = sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)
        claims = [
            normalize_text(claim)
            for claim in re.split(r"(?:[.!?;]+|\n+)", answer)
            if normalize_text(claim)
        ]
        unsupported = 0
        for claim in claims:
            tokens = claim.split()
            overlap = sum(token in context_tokens for token in tokens) / len(tokens)
            unsupported += overlap < self.claim_support_threshold
        unsupported_rate = unsupported / len(claims) if claims else 0.0
        return {
            "groundedness": groundedness,
            "unsupported_claim_rate": unsupported_rate,
        }


class CitationScorer(Scorer):
    """Score ``[Document N]`` citations against retrieved relevance labels."""

    name = "document-citations"
    version = "1"

    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str | None,
        contexts: list[str],
        references: Sequence[str] | None = None,
        context_ids: Sequence[str] | None = None,
        relevant_doc_ids: Sequence[str] | None = None,
    ) -> ScoreResult:
        if not context_ids:
            return {"citation_precision": None, "citation_recall": None}
        cited_positions = list(
            dict.fromkeys(
                int(value)
                for value in re.findall(r"\[\s*document\s+(\d+)\s*\]", answer, re.IGNORECASE)
            )
        )
        relevant = set(relevant_doc_ids or [])
        cited_ids = {
            context_ids[position - 1]
            for position in cited_positions
            if 1 <= position <= len(context_ids)
        }
        supported_citations = cited_ids & relevant
        relevant_in_context = set(context_ids) & relevant
        precision = len(supported_citations) / len(cited_positions) if cited_positions else None
        recall = (
            len(supported_citations) / len(relevant_in_context) if relevant_in_context else None
        )
        return {
            "citation_precision": precision,
            "citation_recall": recall,
        }


class JudgeBackend(ABC):
    name: str

    @abstractmethod
    def evaluate(self, prompt: str) -> Mapping[str, Any]: ...

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name}


class CallableJudgeBackend(JudgeBackend):
    """Adapter for an in-process local or self-hosted judge implementation."""

    name = "callable"

    def __init__(
        self,
        function: Callable[[str], Mapping[str, Any]],
        *,
        model_name: str = "local-callable",
        model_revision: str | None = None,
    ) -> None:
        self.function = function
        self.model_name = model_name
        self.model_revision = model_revision

    def evaluate(self, prompt: str) -> Mapping[str, Any]:
        return self.function(prompt)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "model": self.model_name,
            "revision": self.model_revision,
        }


def _parse_json_object(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("Judge response did not contain a JSON object") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


class OpenAICompatibleJudgeBackend(JudgeBackend):
    """Judge backend for local vLLM/SGLang or hosted compatible endpoints."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local",
        model_revision: str | None = None,
        api_key_env: str = "JUDGE_API_KEY",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the RAG extras to use this judge backend") from exc
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url
        self.model = model
        self.model_revision = model_revision
        self.api_key_env = api_key_env

    def evaluate(self, prompt: str) -> Mapping[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return _parse_json_object(response.choices[0].message.content or "")

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "base_url": self.base_url,
            "model": self.model,
            "model_revision": self.model_revision,
            "api_key_env": self.api_key_env,
            "temperature": 0,
        }


class LLMJudgeScorer(Scorer):
    name = "llm-judge"
    version = "rag-judge-v1"

    def __init__(self, backend: JudgeBackend) -> None:
        self.backend = backend

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "backend": self.backend.metadata(),
            "output_scale": "0..1",
        }

    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str | None,
        contexts: list[str],
        references: Sequence[str] | None = None,
        context_ids: Sequence[str] | None = None,
        relevant_doc_ids: Sequence[str] | None = None,
    ) -> ScoreResult:
        context_labels = list(context_ids or [])
        payload = {
            "question": question,
            "answer": answer,
            "references": list(references or ([reference] if reference else [])),
            "contexts": [
                {
                    "id": context_labels[index]
                    if index < len(context_labels)
                    else f"document-{index + 1}",
                    "text": text,
                }
                for index, text in enumerate(contexts)
            ],
        }
        prompt = (
            "You are evaluating a retrieval-augmented answer. Return only one JSON object with "
            "numeric fields correctness, groundedness, and unsupported_claim_rate, each from 0 to 1, "
            "plus a short rationale string. Correctness compares with the references. Groundedness "
            "measures whether answer claims are supported by contexts. unsupported_claim_rate is the "
            "fraction of claims not supported by contexts.\n\n"
            f"Evaluation input:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        judged = self.backend.evaluate(prompt)
        output: ScoreResult = {}
        for source, destination in (
            ("correctness", "judge_correctness"),
            ("groundedness", "judge_groundedness"),
            ("unsupported_claim_rate", "judge_unsupported_claim_rate"),
        ):
            if source not in judged:
                raise ValueError(f"Judge response is missing required field {source!r}")
            try:
                value = float(judged[source])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Judge field {source!r} must be numeric") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Judge field {source!r} must be between 0 and 1")
            output[destination] = value
        if "rationale" in judged:
            output["judge_rationale"] = str(judged["rationale"])
        return output


class CompositeScorer(Scorer):
    name = "composite"

    def __init__(self, scorers: Sequence[Scorer]) -> None:
        self.scorers = list(scorers)

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "scorers": [scorer.metadata() for scorer in self.scorers]}

    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str | None,
        contexts: list[str],
        references: Sequence[str] | None = None,
        context_ids: Sequence[str] | None = None,
        relevant_doc_ids: Sequence[str] | None = None,
    ) -> ScoreResult:
        combined: ScoreResult = {}
        for scorer in self.scorers:
            values = scorer.score(
                question=question,
                answer=answer,
                reference=reference,
                contexts=contexts,
                references=references,
                context_ids=context_ids,
                relevant_doc_ids=relevant_doc_ids,
            )
            duplicates = set(combined) & set(values)
            if duplicates:
                raise ValueError(f"Scorers emitted duplicate metrics: {sorted(duplicates)}")
            combined.update(values)
        return combined


def build_scorer(config: Mapping[str, Any] | None = None) -> Scorer:
    """Build deterministic scorers plus an optional reproducible judge backend."""
    values = dict(config or {})
    if values.get("mode") == "disabled":
        return NoOpScorer()

    scorers: list[Scorer] = [DeterministicCorrectnessScorer()]
    lexical = values.get("lexical_groundedness", {})
    if lexical is not False:
        lexical_values = dict(lexical) if isinstance(lexical, Mapping) else {}
        scorers.append(
            LexicalGroundednessScorer(
                claim_support_threshold=float(lexical_values.get("claim_support_threshold", 0.5))
            )
        )

    if bool(values.get("citation_metrics", False)):
        scorers.append(CitationScorer())

    judge = values.get("judge")
    if isinstance(judge, Mapping) and judge.get("kind", "disabled") != "disabled":
        kind = str(judge.get("kind"))
        if kind != "openai-compatible":
            raise ValueError(f"Unsupported configured judge backend: {kind!r}")
        api_key_env = str(judge.get("api_key_env", "JUDGE_API_KEY"))
        backend = OpenAICompatibleJudgeBackend(
            base_url=str(judge["base_url"]),
            model=str(judge["model"]),
            api_key=os.getenv(api_key_env, "local"),
            model_revision=(
                str(judge["model_revision"]) if judge.get("model_revision") is not None else None
            ),
            api_key_env=api_key_env,
        )
        scorers.append(LLMJudgeScorer(backend))
    return CompositeScorer(scorers)
