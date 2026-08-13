from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rag_adapt_lab.config import validate_hf_model_config
from rag_adapt_lab.data.io import write_jsonl
from rag_adapt_lab.data.schema import Document, EvalExample
from rag_adapt_lab.evaluation.generation import exact_match, token_f1
from rag_adapt_lab.evaluation.retrieval import RetrievalMetrics, evaluate_rankings
from rag_adapt_lab.evaluation.scorers import Scorer
from rag_adapt_lab.evaluation.statistics import (
    aggregate_prediction_rows,
    paired_bootstrap_delta,
)
from rag_adapt_lab.generation.base import GenerationResult, Generator
from rag_adapt_lab.generation.prompts import (
    rag_prompt_provenance,
)
from rag_adapt_lab.generation.transformers import TransformersGenerator
from rag_adapt_lab.provenance import (
    BENCHMARK_SCHEMA_VERSION,
    AdapterVerification,
    artifact_sha256,
    file_sha256,
    validate_adapter_provenance,
)
from rag_adapt_lab.recipes.plan import RECIPE_RETRIEVAL, BenchmarkJob
from rag_adapt_lab.retrieval.base import RetrievalResult, Retriever
from rag_adapt_lab.tracking.base import Tracker
from rag_adapt_lab.tracking.null import NullTracker
from rag_adapt_lab.training.controls import training_control_differences

from .report import render_markdown_report

COMPARISON_PAIRS = (
    ("base", "rag"),
    ("rag", "sft-rag"),
    ("rag", "raft-rag"),
    ("sft-rag", "raft-rag"),
)


class GeneratorFactory(Protocol):
    def create(self, adapter_path: str | Path | None) -> Generator: ...


@dataclass(slots=True)
class TransformersGeneratorFactory:
    model_config: Mapping[str, Any]
    load_in_4bit: bool = False
    seed: int = 42
    allow_unverified_adapter: bool = False

    def create(self, adapter_path: str | Path | None) -> Generator:
        return TransformersGenerator(
            model_config=self.model_config,
            adapter_path=adapter_path,
            load_in_4bit=self.load_in_4bit,
            seed=self.seed,
            allow_unverified_adapter=self.allow_unverified_adapter,
        )


@dataclass(frozen=True, slots=True)
class CachedRetrieval:
    results: tuple[RetrievalResult, ...]
    latency_s: float


def _path_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    if source.is_file():
        return file_sha256(source)
    return artifact_sha256(source)


def _reference_answers(example: EvalExample) -> list[str]:
    configured = example.metadata.get("reference_answers")
    if isinstance(configured, list):
        references = [str(item).strip() for item in configured if str(item).strip()]
        if references:
            return list(dict.fromkeys(references))
    return [example.reference_answer] if example.reference_answer else []


def _comparison_seed(seed: int, pair: str, metric: str) -> int:
    digest = hashlib.sha256(f"{seed}:{pair}:{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def compose_inference_e2e_latency(
    generated: GenerationResult,
    *,
    retrieval_latency_s: float,
) -> float | None:
    """Compose retrieval-through-decode latency without scoring or judge time."""
    post_retrieval_latency = generated.inference_e2e_latency_s
    if post_retrieval_latency is None:
        stages = [
            generated.prompt_build_latency_s,
            generated.chat_template_latency_s,
            generated.tokenization_latency_s,
            generated.device_transfer_latency_s,
            generated.model_generate_latency_s,
            generated.decode_latency_s,
        ]
        available = [value for value in stages if value is not None]
        post_retrieval_latency = sum(available) if available else None
    return (
        post_retrieval_latency + retrieval_latency_s
        if post_retrieval_latency is not None
        else None
    )


class BenchmarkRunner:
    """Execute a controlled Base/RAG/SFT+RAG/RAFT+RAG comparison."""

    def __init__(
        self,
        *,
        jobs: Sequence[BenchmarkJob],
        model_config: Mapping[str, Any],
        documents: Sequence[Document],
        examples: Sequence[EvalExample],
        retriever: Retriever,
        retriever_config: Mapping[str, Any],
        generator_factory: GeneratorFactory,
        scorer: Scorer,
        output_dir: str | Path,
        top_k: int,
        bootstrap_samples: int = 10_000,
        seed: int = 42,
        warmup_examples: int = 1,
        tracker: Tracker | None = None,
        generator_config: Mapping[str, Any] | None = None,
        model_config_path: str | Path | None = None,
        documents_path: str | Path | None = None,
        eval_path: str | Path | None = None,
        allow_unverified_adapter: bool = False,
        allow_unmatched_training_controls: bool = False,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if warmup_examples < 0:
            raise ValueError("warmup_examples must be non-negative")
        if not jobs:
            raise ValueError("At least one benchmark recipe is required")
        names = [job.recipe for job in jobs]
        if len(names) != len(set(names)):
            raise ValueError("Benchmark recipes must be unique")
        contracts = {(job.model_config, job.documents, job.eval_set) for job in jobs}
        if len(contracts) != 1:
            raise ValueError("All benchmark jobs must share one model, corpus, and evaluation set")
        for job in jobs:
            if job.recipe not in RECIPE_RETRIEVAL:
                raise ValueError(f"Unknown benchmark recipe: {job.recipe!r}")
            if job.use_retrieval != RECIPE_RETRIEVAL[job.recipe]:
                raise ValueError(f"Recipe {job.recipe!r} has an invalid retrieval setting")
            if job.recipe in {"sft-rag", "raft-rag"} and not job.adapter_path:
                raise ValueError(f"Recipe {job.recipe!r} requires an adapter")
            if job.recipe in {"base", "rag"} and job.adapter_path:
                raise ValueError(f"Recipe {job.recipe!r} must not load an adapter")

        self.jobs = list(jobs)
        self.model_config = dict(model_config)
        self.model_id, self.model_revision = validate_hf_model_config(self.model_config)
        self.chat_template_kwargs = dict(self.model_config.get("chat_template_kwargs", {}))
        self.prompt_provenance = rag_prompt_provenance()
        self.documents = list(documents)
        self.examples = list(examples)
        self.retriever = retriever
        self.retriever_config = dict(retriever_config)
        self.generator_factory = generator_factory
        self.scorer = scorer
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self.warmup_examples = warmup_examples
        self.tracker = tracker or NullTracker()
        self.generator_config = dict(generator_config or {})
        self.model_config_path = Path(model_config_path) if model_config_path else None
        self.documents_path = Path(documents_path) if documents_path else None
        self.eval_path = Path(eval_path) if eval_path else None
        self.eval_sha256 = file_sha256(self.eval_path) if self.eval_path is not None else None
        self.allow_unverified_adapter = allow_unverified_adapter
        self.allow_unmatched_training_controls = allow_unmatched_training_controls
        self.adapter_verifications: dict[str, AdapterVerification] = {}
        self.provenance_warnings: list[str] = []
        self.training_controls_matched: bool | None = None
        self.training_control_mismatches: list[str] = []

        if not self.documents:
            raise ValueError("Benchmark corpus must contain at least one document")
        if not self.examples:
            raise ValueError("Benchmark evaluation set must contain at least one example")
        example_ids = {example.id for example in self.examples}
        if len(example_ids) != len(self.examples):
            raise ValueError("Evaluation example IDs must be unique")
        document_ids = {document.id for document in self.documents}
        if len(document_ids) != len(self.documents):
            raise ValueError("Document IDs must be unique")
        missing = sorted(
            {
                doc_id
                for example in self.examples
                for doc_id in example.relevant_doc_ids
                if doc_id not in document_ids
            }
        )
        if missing:
            raise ValueError(f"Evaluation set references missing documents: {missing[:10]}")
        if (
            any(job.recipe in {"sft-rag", "raft-rag"} for job in self.jobs)
            and self.eval_sha256 is None
            and not self.allow_unverified_adapter
        ):
            raise ValueError(
                "eval_path is required to verify adapted benchmark conditions against the "
                "current held-out evaluation file"
            )
        self._validate_adapters()

    def _validate_adapters(self) -> None:
        expected_prompt = {
            **self.prompt_provenance,
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        configured_adapters = {
            job.recipe: Path(job.adapter_path)
            for job in self.jobs
            if job.recipe in {"sft-rag", "raft-rag"} and job.adapter_path is not None
        }
        configured_sft = configured_adapters.get("sft-rag")
        configured_raft = configured_adapters.get("raft-rag")
        if (
            configured_sft is not None
            and configured_raft is not None
            and configured_sft.resolve() == configured_raft.resolve()
        ):
            message = "SFT and RAFT conditions use the same adapter artifact"
            if not self.allow_unverified_adapter:
                raise ValueError(message)
            self.provenance_warnings.append(message)
        for job in self.jobs:
            if job.recipe not in {"sft-rag", "raft-rag"}:
                continue
            assert job.adapter_path is not None
            verification = validate_adapter_provenance(
                job.adapter_path,
                model_id=self.model_id,
                model_revision=self.model_revision,
                expected_mode="sft" if job.recipe == "sft-rag" else "raft",
                expected_prompt=expected_prompt,
                held_out_evaluation_sha256=self.eval_sha256,
                allow_unverified=self.allow_unverified_adapter,
            )
            self.adapter_verifications[job.recipe] = verification
            self.provenance_warnings.extend(
                f"{job.recipe}: {message}" for message in verification.warnings
            )

        sft = self.adapter_verifications.get("sft-rag")
        raft = self.adapter_verifications.get("raft-rag")
        same_hash = bool(
            sft is not None
            and raft is not None
            and sft.artifact_sha256
            and sft.artifact_sha256 == raft.artifact_sha256
        )
        if same_hash:
            message = "SFT and RAFT conditions use the same adapter artifact"
            if not self.allow_unverified_adapter:
                raise ValueError(message)
            if message not in self.provenance_warnings:
                self.provenance_warnings.append(message)

        if sft is not None and raft is not None and sft.verified and raft.verified:
            mismatched_partitions = [
                name
                for name, sft_fingerprint, raft_fingerprint in (
                    (
                        "training",
                        sft.training_source_fingerprint,
                        raft.training_source_fingerprint,
                    ),
                    (
                        "validation",
                        sft.validation_source_fingerprint,
                        raft.validation_source_fingerprint,
                    ),
                )
                if sft_fingerprint != raft_fingerprint
            ]
            if mismatched_partitions:
                message = (
                    "SFT and RAFT conditions use different underlying source "
                    f"partitions: {', '.join(mismatched_partitions)}"
                )
                if not self.allow_unverified_adapter:
                    raise ValueError(message)
                self.provenance_warnings.append(message)

            self.training_controls_matched = (
                sft.training_control_sha256 == raft.training_control_sha256
            )
            if not self.training_controls_matched:
                self.training_control_mismatches = training_control_differences(
                    sft.training_controls or {},
                    raft.training_controls or {},
                )
                detail = "; ".join(self.training_control_mismatches[:10])
                message = "SFT and RAFT training controls differ"
                if detail:
                    message = f"{message}: {detail}"
                if not self.allow_unmatched_training_controls:
                    raise ValueError(message)
                self.provenance_warnings.append(
                    "Confounded comparison: adaptation training controls differ. " + detail
                )

        if self.provenance_warnings:
            for message in self.provenance_warnings:
                category = (
                    "CONFOUNDED COMPARISON"
                    if message.startswith("Confounded comparison:")
                    else "UNVERIFIED ADAPTER PROVENANCE"
                )
                warnings.warn(
                    f"{category}: {message}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _retrieve_once(self) -> tuple[dict[str, CachedRetrieval], RetrievalMetrics]:
        self.retriever.index(self.documents)
        document_ids = {document.id for document in self.documents}
        cache: dict[str, CachedRetrieval] = {}
        for example in self.examples:
            started = time.perf_counter()
            results = self.retriever.search(example.question, top_k=self.top_k)
            elapsed = time.perf_counter() - started
            unknown = sorted(
                {result.document.id for result in results if result.document.id not in document_ids}
            )
            if unknown:
                raise ValueError(
                    f"Retriever returned documents outside the fixed corpus: {unknown[:10]}"
                )
            cache[example.id] = CachedRetrieval(tuple(results[: self.top_k]), elapsed)
        rankings = {
            example_id: [result.document.id for result in cached.results]
            for example_id, cached in cache.items()
        }
        metrics = evaluate_rankings(self.examples, rankings, top_k=self.top_k)
        return cache, metrics

    def _configuration(self) -> dict[str, Any]:
        configuration: dict[str, Any] = {
            "model": {
                "model_id": self.model_id,
                "revision": self.model_revision,
                "generation": dict(self.model_config.get("generation", {})),
                "chat_template_kwargs": self.chat_template_kwargs,
                "thinking_enabled": self.chat_template_kwargs.get("enable_thinking") is True,
                "condition": (
                    "thinking-enabled"
                    if self.chat_template_kwargs.get("enable_thinking") is True
                    else "concise-non-thinking"
                ),
                "trust_remote_code": False,
            },
            "retriever": {**self.retriever_config, "top_k": self.top_k},
            "prompt": self.prompt_provenance,
            "generator": self.generator_config,
            "scorer": self.scorer.metadata(),
            "seed": self.seed,
            "bootstrap_samples": self.bootstrap_samples,
            "held_out_examples": len(self.examples),
            "documents": len(self.documents),
        }
        files = {
            "model_config": self.model_config_path,
            "documents": self.documents_path,
            "eval_set": self.eval_path,
        }
        configuration["inputs"] = {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in files.items()
            if path is not None
        }
        return configuration

    def _run_recipe(
        self,
        job: BenchmarkJob,
        generator: Generator,
        retrieval_cache: Mapping[str, CachedRetrieval],
        retrieval_metrics: RetrievalMetrics,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        for example in self.examples[: self.warmup_examples]:
            warmup_results = retrieval_cache[example.id].results
            contexts = (
                [result.document.text for result in warmup_results] if job.use_retrieval else []
            )
            generator.generate(question=example.question, contexts=contexts)
        generator.reset_runtime_metrics()
        rows: list[dict[str, Any]] = []
        for example in self.examples:
            cached = retrieval_cache[example.id]
            retrieved = list(cached.results)
            contexts = [result.document.text for result in retrieved] if job.use_retrieval else []
            generated = generator.generate(question=example.question, contexts=contexts)
            references = _reference_answers(example)
            em = max((exact_match(generated.text, value) for value in references), default=None)
            f1 = max((token_f1(generated.text, value) for value in references), default=None)
            scoring_started = time.perf_counter()
            scores = self.scorer.score(
                question=example.question,
                answer=generated.text,
                reference=example.reference_answer,
                references=references,
                contexts=contexts,
                context_ids=[result.document.id for result in retrieved]
                if job.use_retrieval
                else [],
                relevant_doc_ids=example.relevant_doc_ids,
            )
            total_scoring_latency = time.perf_counter() - scoring_started
            judge_latency_value = scores.get("judge_latency_s")
            judge_latency = (
                float(judge_latency_value)
                if isinstance(judge_latency_value, (int, float))
                and not isinstance(judge_latency_value, bool)
                else 0.0
            )
            scoring_latency = max(0.0, total_scoring_latency - judge_latency)
            model_generate_latency = generated.model_generate_latency_s
            output_tokens_per_second = (
                generated.output_tokens / model_generate_latency
                if generated.output_tokens is not None
                and model_generate_latency is not None
                and model_generate_latency > 0
                else None
            )
            total_tokens_per_second = (
                (generated.prompt_tokens + generated.output_tokens) / model_generate_latency
                if generated.prompt_tokens is not None
                and generated.output_tokens is not None
                and model_generate_latency is not None
                and model_generate_latency > 0
                else None
            )
            inference_e2e_latency = compose_inference_e2e_latency(
                generated,
                retrieval_latency_s=cached.latency_s if job.use_retrieval else 0.0,
            )
            rows.append(
                {
                    "id": example.id,
                    "recipe": job.recipe,
                    "question": example.question,
                    "reference": example.reference_answer,
                    "references": references,
                    "prediction": generated.text,
                    "raw_prediction": generated.raw_text,
                    "reasoning": generated.reasoning,
                    "exact_match": em,
                    "token_f1": f1,
                    "scores": scores,
                    "retrieval_used": job.use_retrieval,
                    "retrieved_doc_ids": [result.document.id for result in retrieved],
                    "retrieval_results": [
                        {
                            "doc_id": result.document.id,
                            "rank": result.rank,
                            "score": result.score,
                        }
                        for result in retrieved
                    ],
                    "retrieval_latency_s": cached.latency_s,
                    "prompt_tokens": generated.prompt_tokens,
                    "output_tokens": generated.output_tokens,
                    "reasoning_tokens": generated.reasoning_tokens,
                    "answer_tokens": generated.answer_tokens,
                    "thinking_boundary_token_id": generated.thinking_boundary_token_id,
                    "thinking_boundary_found": generated.thinking_boundary_found,
                    "thinking_protocol_violation": generated.thinking_protocol_violation,
                    "prompt_build_latency_s": generated.prompt_build_latency_s,
                    "chat_template_latency_s": generated.chat_template_latency_s,
                    "tokenization_latency_s": generated.tokenization_latency_s,
                    "device_transfer_latency_s": generated.device_transfer_latency_s,
                    "model_generate_latency_s": model_generate_latency,
                    "decode_latency_s": generated.decode_latency_s,
                    "inference_e2e_latency_s": inference_e2e_latency,
                    "scoring_latency_s": scoring_latency,
                    "judge_latency_s": judge_latency if "judge_status" in scores else None,
                    "output_tokens_per_model_generate_second": output_tokens_per_second,
                    "total_tokens_per_model_generate_second": total_tokens_per_second,
                    "batch_size": generated.batch_size,
                    "generation_mode": generated.generation_mode,
                    # Schema-v1 aliases; remove in the next major schema revision.
                    "latency_s": model_generate_latency,
                    "end_to_end_latency_s": inference_e2e_latency,
                    "tokens_per_second": output_tokens_per_second,
                }
            )
        metrics = aggregate_prediction_rows(rows)
        metrics["peak_allocated_vram_gb"] = generator.peak_allocated_vram_gb()
        metrics["peak_reserved_vram_gb"] = generator.peak_reserved_vram_gb()
        metrics["peak_gpu_vram_gb"] = metrics["peak_allocated_vram_gb"]
        if job.use_retrieval:
            metrics.update(retrieval_metrics.as_dict())
        return rows, metrics

    def _comparisons(
        self, rows_by_recipe: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> dict[str, Any]:
        comparisons: dict[str, Any] = {}

        def numeric_ids(recipe: str, metric: str) -> set[str]:
            identifiers: set[str] = set()
            for row in rows_by_recipe[recipe]:
                value = row.get(metric)
                scores = row.get("scores", {})
                if value is None and isinstance(scores, Mapping):
                    value = scores.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    identifiers.add(str(row["id"]))
            return identifiers

        def has_numeric_pairs(baseline: str, candidate: str, metric: str) -> bool:
            return bool(numeric_ids(baseline, metric) & numeric_ids(candidate, metric))

        for baseline, candidate in COMPARISON_PAIRS:
            if baseline not in rows_by_recipe or candidate not in rows_by_recipe:
                continue
            pair = f"{baseline}->{candidate}"
            metrics = [
                metric
                for metric in ("exact_match", "token_f1")
                if has_numeric_pairs(baseline, candidate, metric)
            ]
            for optional in (
                "reference_overlap",
                "lexical_groundedness",
                "lexical_unsupported_claim_rate",
                "judge_correctness",
                "judge_groundedness",
                "judge_unsupported_claim_rate",
                "citation_precision",
                "citation_recall",
            ):
                if has_numeric_pairs(baseline, candidate, optional):
                    metrics.append(optional)
            comparisons[pair] = {
                metric: paired_bootstrap_delta(
                    rows_by_recipe[baseline],
                    rows_by_recipe[candidate],
                    metric=metric,
                    samples=self.bootstrap_samples,
                    seed=_comparison_seed(self.seed, pair, metric),
                )
                for metric in metrics
            }
            has_unverified_adapter_warning = any(
                not warning.startswith("Confounded comparison:")
                for warning in getattr(self, "provenance_warnings", [])
            )
            if has_unverified_adapter_warning and (
                baseline in {"sft-rag", "raft-rag"}
                or candidate in {"sft-rag", "raft-rag"}
            ):
                for result in comparisons[pair].values():
                    result["decision_eligible"] = False
                    result["statistically_significant"] = False
                    result["status"] = "unverified_adapter_provenance"
            if pair == "sft-rag->raft-rag" and self.training_controls_matched is False:
                for result in comparisons[pair].values():
                    result["decision_eligible"] = False
                    result["statistically_significant"] = False
                    result["status"] = "confounded_training_controls"
        return comparisons

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir = self.output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        configuration = self._configuration()
        self.tracker.start_run(name="benchmark", config=configuration)
        generator: Generator | None = None
        try:
            retrieval_cache, retrieval_metrics = self._retrieve_once()
            rows_by_recipe: dict[str, list[dict[str, Any]]] = {}
            recipes: dict[str, Any] = {}
            current_adapter: str | None | object = object()
            all_rows: list[dict[str, Any]] = []
            for job in self.jobs:
                if job.adapter_path != current_adapter:
                    if generator is not None:
                        generator.close()
                    generator = self.generator_factory.create(job.adapter_path)
                    current_adapter = job.adapter_path
                assert generator is not None
                rows, metrics = self._run_recipe(
                    job,
                    generator,
                    retrieval_cache,
                    retrieval_metrics,
                )
                rows_by_recipe[job.recipe] = rows
                all_rows.extend(rows)
                recipe_path = predictions_dir / f"{job.recipe}.jsonl"
                write_jsonl(recipe_path, rows)
                recipes[job.recipe] = {
                    "adapter_path": job.adapter_path,
                    "adapter_sha256": (
                        self.adapter_verifications[job.recipe].artifact_sha256
                        if job.recipe in self.adapter_verifications
                        else _path_sha256(job.adapter_path)
                    ),
                    "adapter_provenance": (
                        self.adapter_verifications[job.recipe].as_dict()
                        if job.recipe in self.adapter_verifications
                        else None
                    ),
                    "retrieval_enabled": job.use_retrieval,
                    "predictions": str(recipe_path),
                    "metrics": metrics,
                }
                self.tracker.log(
                    {
                        f"{job.recipe}/{key}": value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    }
                )

            write_jsonl(self.output_dir / "predictions.jsonl", all_rows)
            comparisons = self._comparisons(rows_by_recipe)
            summary = {
                "schema_name": "benchmark-summary",
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "configuration": configuration,
                "provenance": {
                    "verified": not self.provenance_warnings
                    and all(item.verified for item in self.adapter_verifications.values()),
                    "confounded": self.training_controls_matched is False,
                    "allow_unverified_adapter": self.allow_unverified_adapter,
                    "allow_unmatched_training_controls": (
                        self.allow_unmatched_training_controls
                    ),
                    "training_controls_matched": self.training_controls_matched,
                    "training_control_mismatches": self.training_control_mismatches,
                    "warnings": self.provenance_warnings,
                },
                "retrieval_metrics": retrieval_metrics.as_dict(),
                "recipes": recipes,
                "comparisons": comparisons,
            }
            comparison_metrics = {
                f"comparison/{pair}/{metric}/{field}": value
                for pair, metric_values in comparisons.items()
                for metric, result in metric_values.items()
                for field, value in result.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if comparison_metrics:
                self.tracker.log(comparison_metrics)
            summary_path = self.output_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            report_path = self.output_dir / "report.md"
            report_path.write_text(render_markdown_report(summary), encoding="utf-8")
            self.tracker.log_artifact(
                summary_path, name="benchmark-summary", artifact_type="evaluation"
            )
            self.tracker.log_artifact(report_path, name="benchmark-report", artifact_type="report")
            self.tracker.log_artifact(
                self.output_dir / "predictions.jsonl",
                name="benchmark-predictions",
                artifact_type="predictions",
            )
            return summary
        finally:
            try:
                if generator is not None:
                    generator.close()
            finally:
                self.tracker.finish()
