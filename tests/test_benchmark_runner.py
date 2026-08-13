import json
from pathlib import Path

import pytest

from rag_adapt_lab.benchmark.runner import BenchmarkRunner
from rag_adapt_lab.data.schema import Document, EvalExample
from rag_adapt_lab.evaluation.scorers import build_scorer
from rag_adapt_lab.generation.base import GenerationResult, Generator
from rag_adapt_lab.generation.prompts import rag_prompt_provenance
from rag_adapt_lab.provenance import (
    ADAPTER_MANIFEST_SCHEMA_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    artifact_sha256,
    canonical_sha256,
    file_sha256,
)
from rag_adapt_lab.recipes.plan import build_plan
from rag_adapt_lab.retrieval.base import RetrievalResult, Retriever
from rag_adapt_lab.training.controls import normalize_training_controls

TEST_TRAINING_CONTROLS = normalize_training_controls({}, has_validation=True)


class StaticRetriever(Retriever):
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.search_calls = 0

    def index(self, documents: list[Document]) -> None:
        self.documents = documents

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.search_calls += 1
        target = "alpha" if "alpha" in query else "beta"
        ranked = sorted(self.documents, key=lambda document: document.id != target)
        return [
            RetrievalResult(document=document, score=1.0 / rank, rank=rank)
            for rank, document in enumerate(ranked[:top_k], start=1)
        ]


class FakeGenerator(Generator):
    def __init__(self, adapter_path: str | None) -> None:
        self.adapter_path = adapter_path
        self.closed = False

    def generate(self, *, question: str, contexts: list[str] | None = None) -> GenerationResult:
        expected = "alpha" if "alpha" in question else "beta"
        if self.adapter_path and "raft" in self.adapter_path:
            answer = expected
        elif contexts:
            answer = contexts[0]
        else:
            answer = "wrong"
        return GenerationResult(
            text=answer,
            prompt_tokens=5,
            output_tokens=1,
            latency_s=0.01,
        )

    def peak_memory_gb(self) -> float:
        return 1.25

    def close(self) -> None:
        self.closed = True


class FakeGeneratorFactory:
    def __init__(self) -> None:
        self.created: list[str | None] = []

    def create(self, adapter_path: str | Path | None) -> Generator:
        value = str(adapter_path) if adapter_path is not None else None
        self.created.append(value)
        return FakeGenerator(value)


def test_benchmark_executes_matrix_and_writes_reports(tmp_path: Path) -> None:
    documents = [Document(id="alpha", text="alpha"), Document(id="beta", text="beta")]
    examples = [
        EvalExample(
            id="q-alpha",
            question="find alpha",
            reference_answer="alpha",
            relevant_doc_ids=["alpha"],
        ),
        EvalExample(
            id="q-beta",
            question="find beta",
            reference_answer="beta",
            relevant_doc_ids=["beta"],
        ),
    ]
    jobs = build_plan(
        recipes=["base", "rag", "sft-rag", "raft-rag"],
        model_config="model.yaml",
        documents="documents.jsonl",
        eval_set="eval.jsonl",
        adapters={"sft-rag": "sft-adapter", "raft-rag": "raft-adapter"},
    )
    retriever = StaticRetriever()
    factory = FakeGeneratorFactory()
    runner = BenchmarkRunner(
        jobs=jobs,
        model_config={
            "model_id": "test/model",
            "revision": "0" * 40,
            "trust_remote_code": False,
            "generation": {"max_new_tokens": 8, "do_sample": False},
        },
        documents=documents,
        examples=examples,
        retriever=retriever,
        retriever_config={"kind": "static"},
        generator_factory=factory,
        scorer=build_scorer(),
        output_dir=tmp_path,
        top_k=2,
        bootstrap_samples=100,
        seed=3,
        allow_unverified_adapter=True,
    )
    summary = runner.run()

    assert retriever.search_calls == len(examples)  # One shared retrieval pass.
    assert factory.created == [None, "sft-adapter", "raft-adapter"]
    assert set(summary["recipes"]) == {"base", "rag", "sft-rag", "raft-rag"}
    assert summary["configuration"]["prompt"]["version"] == "4"
    assert summary["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert summary["provenance"]["verified"] is False
    assert summary["retrieval_metrics"]["retrieval/evaluated"] == len(examples)
    assert "base->rag" in summary["comparisons"]
    assert "rag->raft-rag" in summary["comparisons"]
    assert (
        summary["comparisons"]["rag->raft-rag"]["token_f1"]["status"]
        == "unverified_adapter_provenance"
    )
    assert (
        summary["comparisons"]["rag->raft-rag"]["token_f1"]["decision_eligible"]
        is False
    )
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert "RAFT + RAG" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "UNVERIFIED PROVENANCE" in (tmp_path / "report.md").read_text(encoding="utf-8")

    rows = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(examples) * 4
    assert all(row["retrieved_doc_ids"] for row in rows)
    base_rows = [row for row in rows if row["recipe"] == "base"]
    assert all(row["retrieval_used"] is False for row in base_rows)
    assert all(row["tokens_per_second"] == 100.0 for row in rows)
    assert all(row["model_generate_latency_s"] == 0.01 for row in rows)
    assert all(row["judge_latency_s"] is None for row in rows)


def test_benchmark_rejects_identical_sft_and_raft_adapter_paths(tmp_path: Path) -> None:
    adapter = tmp_path / "same-adapter"
    adapter.mkdir()
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"id":"q","question":"q"}\n', encoding="utf-8")
    jobs = build_plan(
        recipes=["sft-rag", "raft-rag"],
        model_config="model.yaml",
        documents="documents.jsonl",
        eval_set="eval.jsonl",
        adapters={"sft-rag": adapter, "raft-rag": adapter},
    )
    with pytest.raises(ValueError, match="same adapter artifact"):
        BenchmarkRunner(
            jobs=jobs,
            model_config={
                "model_id": "test/model",
                "revision": "0" * 40,
                "generation": {"max_new_tokens": 8, "do_sample": False},
            },
            documents=[Document(id="doc", text="doc")],
            examples=[EvalExample(id="q", question="q", relevant_doc_ids=["doc"])],
            retriever=StaticRetriever(),
            retriever_config={"kind": "static"},
            generator_factory=FakeGeneratorFactory(),
            scorer=build_scorer(),
            output_dir=tmp_path / "output",
            top_k=1,
            eval_path=eval_path,
        )


def test_benchmark_rejects_distinct_paths_with_identical_adapter_hashes(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"id":"q","question":"q"}\n', encoding="utf-8")
    adapters: dict[str, Path] = {}
    for mode in ("sft", "raft"):
        adapter = tmp_path / mode
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "test/model"}), encoding="utf-8"
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"identical-weights")
        manifest = {
            "schema_name": "raglab-adapter-manifest",
            "schema_version": ADAPTER_MANIFEST_SCHEMA_VERSION,
            "model": {"model_id": "test/model", "revision": "0" * 40},
            "recipe": f"test-{mode}",
            "adaptation_mode": mode,
            "training_prompt": rag_prompt_provenance(),
            "chat_template_kwargs": {},
            "training_dataset_fingerprint": "1" * 64,
            "validation_dataset_fingerprint": "2" * 64,
            "training_source_fingerprint": "4" * 64,
            "validation_source_fingerprint": "5" * 64,
            "held_out_evaluation_sha256": file_sha256(eval_path),
            "training_configuration_sha256": "3" * 64,
            "training_controls": TEST_TRAINING_CONTROLS,
            "training_control_sha256": canonical_sha256(TEST_TRAINING_CONTROLS),
            "adapter_artifact_sha256": artifact_sha256(adapter),
            "best_checkpoint": None,
            "best_validation_metric": None,
        }
        (adapter / "raglab_adapter_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        adapters[mode] = adapter
    jobs = build_plan(
        recipes=["sft-rag", "raft-rag"],
        model_config="model.yaml",
        documents="documents.jsonl",
        eval_set="eval.jsonl",
        adapters={"sft-rag": adapters["sft"], "raft-rag": adapters["raft"]},
    )
    with pytest.raises(ValueError, match="same adapter artifact"):
        BenchmarkRunner(
            jobs=jobs,
            model_config={
                "model_id": "test/model",
                "revision": "0" * 40,
                "generation": {"max_new_tokens": 8, "do_sample": False},
            },
            documents=[Document(id="doc", text="doc")],
            examples=[EvalExample(id="q", question="q", relevant_doc_ids=["doc"])],
            retriever=StaticRetriever(),
            retriever_config={"kind": "static"},
            generator_factory=FakeGeneratorFactory(),
            scorer=build_scorer(),
            output_dir=tmp_path / "output",
            top_k=1,
            eval_path=eval_path,
        )


def test_comparisons_skip_optional_metrics_without_numeric_pairs() -> None:
    runner = object.__new__(BenchmarkRunner)
    runner.bootstrap_samples = 20
    runner.seed = 7
    comparisons = runner._comparisons(
        {
            "base": [
                {"id": "a", "exact_match": 0.0, "token_f1": 0.0,
                 "scores": {"judge_correctness": 0.2}},
                {"id": "b", "exact_match": 0.0, "token_f1": 0.0, "scores": {}},
            ],
            "rag": [
                {"id": "a", "exact_match": 1.0, "token_f1": 1.0, "scores": {}},
                {"id": "b", "exact_match": 1.0, "token_f1": 1.0,
                 "scores": {"judge_correctness": 0.8}},
            ],
        }
    )
    assert set(comparisons["base->rag"]) == {"exact_match", "token_f1"}


def test_benchmark_rejects_mismatched_sft_and_raft_source_partitions(
    tmp_path: Path,
) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"id":"q","question":"q"}\n', encoding="utf-8")
    adapters: dict[str, Path] = {}
    for mode, source_fingerprint in (("sft", "4" * 64), ("raft", "6" * 64)):
        adapter = tmp_path / mode
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "test/model"}), encoding="utf-8"
        )
        (adapter / "adapter_model.safetensors").write_bytes(mode.encode())
        manifest = {
            "schema_name": "raglab-adapter-manifest",
            "schema_version": ADAPTER_MANIFEST_SCHEMA_VERSION,
            "model": {"model_id": "test/model", "revision": "0" * 40},
            "recipe": f"test-{mode}",
            "adaptation_mode": mode,
            "training_prompt": rag_prompt_provenance(),
            "chat_template_kwargs": {},
            "training_dataset_fingerprint": "1" * 64,
            "validation_dataset_fingerprint": "2" * 64,
            "training_source_fingerprint": source_fingerprint,
            "validation_source_fingerprint": "5" * 64,
            "held_out_evaluation_sha256": file_sha256(eval_path),
            "training_configuration_sha256": "3" * 64,
            "training_controls": TEST_TRAINING_CONTROLS,
            "training_control_sha256": canonical_sha256(TEST_TRAINING_CONTROLS),
            "adapter_artifact_sha256": artifact_sha256(adapter),
            "best_checkpoint": None,
            "best_validation_metric": None,
        }
        (adapter / "raglab_adapter_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        adapters[mode] = adapter

    jobs = build_plan(
        recipes=["sft-rag", "raft-rag"],
        model_config="model.yaml",
        documents="documents.jsonl",
        eval_set="eval.jsonl",
        adapters={"sft-rag": adapters["sft"], "raft-rag": adapters["raft"]},
    )
    with pytest.raises(ValueError, match="different underlying source partitions"):
        BenchmarkRunner(
            jobs=jobs,
            model_config={
                "model_id": "test/model",
                "revision": "0" * 40,
                "generation": {"max_new_tokens": 8, "do_sample": False},
            },
            documents=[Document(id="doc", text="doc")],
            examples=[EvalExample(id="q", question="q", relevant_doc_ids=["doc"])],
            retriever=StaticRetriever(),
            retriever_config={"kind": "static"},
            generator_factory=FakeGeneratorFactory(),
            scorer=build_scorer(),
            output_dir=tmp_path / "output",
            top_k=1,
            eval_path=eval_path,
        )
