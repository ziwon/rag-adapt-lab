import copy
import json
from pathlib import Path

import pytest

from rag_adapt_lab.benchmark.runner import BenchmarkRunner
from rag_adapt_lab.data.schema import Document, EvalExample
from rag_adapt_lab.evaluation.scorers import NoOpScorer
from rag_adapt_lab.generation.base import GenerationResult, Generator
from rag_adapt_lab.generation.prompts import rag_prompt_provenance
from rag_adapt_lab.provenance import artifact_sha256, canonical_sha256, file_sha256
from rag_adapt_lab.recipes.plan import build_plan
from rag_adapt_lab.retrieval.base import RetrievalResult, Retriever
from rag_adapt_lab.training.controls import (
    normalize_training_controls,
    training_control_sha256,
)


class OneDocumentRetriever(Retriever):
    def __init__(self) -> None:
        self.document: Document | None = None

    def index(self, documents: list[Document]) -> None:
        self.document = documents[0]

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        assert self.document is not None
        return [RetrievalResult(document=self.document, score=1.0, rank=1)]


class ConstantGenerator(Generator):
    def generate(self, *, question: str, contexts: list[str] | None = None) -> GenerationResult:
        return GenerationResult(text="answer", output_tokens=1, model_generate_latency_s=0.01)


class ConstantGeneratorFactory:
    def create(self, adapter_path: str | Path | None) -> Generator:
        return ConstantGenerator()


def base_training_config() -> dict[str, object]:
    return {
        "load_in_4bit": False,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["v_proj", "q_proj"],
        "learning_rate": 0.0001,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 512,
        "seed": 7,
        "data_seed": 11,
    }


def write_adapter(
    path: Path,
    *,
    mode: str,
    evaluation: Path,
    controls: dict[str, object],
) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "test/model"}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(f"weights-{mode}".encode())
    manifest = {
        "schema_name": "raglab-adapter-manifest",
        "schema_version": 3,
        "model": {"model_id": "test/model", "revision": "0" * 40},
        "recipe": f"test-{mode}",
        "adaptation_mode": mode,
        "training_prompt": rag_prompt_provenance(),
        "chat_template_kwargs": {},
        "training_dataset_fingerprint": "1" * 64,
        "validation_dataset_fingerprint": "2" * 64,
        "training_source_fingerprint": "4" * 64,
        "validation_source_fingerprint": "5" * 64,
        "held_out_evaluation_sha256": file_sha256(evaluation),
        "training_configuration_sha256": "3" * 64,
        "training_controls": controls,
        "training_control_sha256": canonical_sha256(controls),
        "adapter_artifact_sha256": artifact_sha256(path),
        "best_checkpoint": None,
        "best_validation_metric": None,
    }
    (path / "raglab_adapter_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def make_runner(
    tmp_path: Path,
    *,
    sft_controls: dict[str, object],
    raft_controls: dict[str, object],
    allow_mismatch: bool = False,
) -> BenchmarkRunner:
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text(
        '{"id":"q","question":"question","reference_answer":"answer","relevant_doc_ids":["d"]}\n',
        encoding="utf-8",
    )
    sft = tmp_path / "sft"
    raft = tmp_path / "raft"
    write_adapter(sft, mode="sft", evaluation=evaluation, controls=sft_controls)
    write_adapter(raft, mode="raft", evaluation=evaluation, controls=raft_controls)
    jobs = build_plan(
        recipes=["sft-rag", "raft-rag"],
        model_config="model.yaml",
        documents="documents.jsonl",
        eval_set=evaluation,
        adapters={"sft-rag": sft, "raft-rag": raft},
    )
    return BenchmarkRunner(
        jobs=jobs,
        model_config={
            "model_id": "test/model",
            "revision": "0" * 40,
            "generation": {"max_new_tokens": 8, "do_sample": False},
        },
        documents=[Document(id="d", text="answer")],
        examples=[
            EvalExample(
                id="q",
                question="question",
                reference_answer="answer",
                relevant_doc_ids=["d"],
            )
        ],
        retriever=OneDocumentRetriever(),
        retriever_config={"kind": "test"},
        generator_factory=ConstantGeneratorFactory(),
        scorer=NoOpScorer(),
        output_dir=tmp_path / "benchmark",
        top_k=1,
        bootstrap_samples=20,
        warmup_examples=0,
        eval_path=evaluation,
        allow_unmatched_training_controls=allow_mismatch,
    )


def test_normalized_controls_ignore_paths_tracking_and_recipe_mode() -> None:
    first = base_training_config()
    first.update({"output_dir": "one", "tracking": {"backend": "wandb"}, "mode": "sft"})
    second = base_training_config()
    second.update({"output_dir": "two", "tracking": {"backend": "none"}, "mode": "raft"})
    first_controls = normalize_training_controls(first, has_validation=True)
    second_controls = normalize_training_controls(second, has_validation=True)
    assert first_controls == second_controls
    assert training_control_sha256(first_controls) == training_control_sha256(second_controls)


@pytest.mark.parametrize(
    ("field", "value", "mismatch_name"),
    [
        ("lora_r", 64, "adapter.rank"),
        ("learning_rate", 0.0005, "optimization.learning_rate"),
        ("num_train_epochs", 5, "optimization.num_train_epochs"),
        ("gradient_accumulation_steps", 8, "batching.effective_batch_size"),
    ],
)
def test_benchmark_rejects_unmatched_training_controls(
    tmp_path: Path,
    field: str,
    value: object,
    mismatch_name: str,
) -> None:
    baseline = normalize_training_controls(base_training_config(), has_validation=True)
    changed_config = copy.deepcopy(base_training_config())
    changed_config[field] = value
    changed = normalize_training_controls(changed_config, has_validation=True)
    with pytest.raises(ValueError, match=mismatch_name):
        make_runner(tmp_path, sft_controls=baseline, raft_controls=changed)


def test_training_control_override_marks_report_confounded(tmp_path: Path) -> None:
    baseline = normalize_training_controls(base_training_config(), has_validation=True)
    changed_config = base_training_config()
    changed_config["lora_r"] = 64
    changed = normalize_training_controls(changed_config, has_validation=True)
    runner = make_runner(
        tmp_path,
        sft_controls=baseline,
        raft_controls=changed,
        allow_mismatch=True,
    )
    summary = runner.run()
    comparison = summary["comparisons"]["sft-rag->raft-rag"]["token_f1"]
    assert summary["provenance"]["confounded"] is True
    assert summary["provenance"]["verified"] is False
    assert comparison["decision_eligible"] is False
    assert comparison["status"] == "confounded_training_controls"
    report = (tmp_path / "benchmark" / "report.md").read_text(encoding="utf-8")
    assert "confounded: adaptation training controls differ" in report
    assert "Confounded comparison: adaptation training controls differ" in report
