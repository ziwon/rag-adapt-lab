import copy
import json
from pathlib import Path

import pytest

from rag_adapt_lab.data.schema import Document, EvalExample
from rag_adapt_lab.provenance import (
    ADAPTER_MANIFEST_SCHEMA_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    TRAINING_MANIFEST_SCHEMA_VERSION,
)
from rag_adapt_lab.schema_validation import load_artifact_schema, validate_artifact_schema
from rag_adapt_lab.training.controls import (
    normalize_training_controls,
    validate_training_controls,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_document_minimum_contract() -> None:
    doc = Document(id="d1", text="hello")
    assert doc.id == "d1"
    assert doc.metadata == {}


def test_eval_evidence_ids_are_merged() -> None:
    row = EvalExample.model_validate(
        {
            "id": "q1",
            "question": "question",
            "relevant_doc_ids": ["d1"],
            "evidence": [{"doc_id": "d2", "text": "evidence"}],
        }
    )
    assert row.relevant_doc_ids == ["d1", "d2"]


def test_machine_readable_manifest_schemas_match_runtime_versions() -> None:
    expected = {
        "adapter-manifest-v3.schema.json": ADAPTER_MANIFEST_SCHEMA_VERSION,
        "training-manifest-v3.schema.json": TRAINING_MANIFEST_SCHEMA_VERSION,
        "benchmark-summary-v3.schema.json": BENCHMARK_SCHEMA_VERSION,
    }
    for filename, version in expected.items():
        documented_schema = json.loads(
            (PROJECT_ROOT / "docs" / "schemas" / filename).read_text(encoding="utf-8")
        )
        packaged_schema = load_artifact_schema(filename)
        assert documented_schema == packaged_schema
        assert packaged_schema["properties"]["schema_version"]["const"] == version


def test_canonical_training_controls_schema_is_shared_and_documented() -> None:
    packaged = load_artifact_schema("training-controls-v1.schema.json")
    documented = json.loads(
        (PROJECT_ROOT / "docs/schemas/training-controls-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert documented == packaged
    for manifest_name in (
        "adapter-manifest-v3.schema.json",
        "training-manifest-v3.schema.json",
    ):
        assert (
            load_artifact_schema(manifest_name)["properties"]["training_controls"]
            == {"$ref": "training-controls-v1.schema.json"}
        )


def test_all_current_artifact_schemas_have_distinct_identities() -> None:
    expected = {
        "adapter-manifest-v3.schema.json": ("raglab-adapter-manifest", 3),
        "training-manifest-v3.schema.json": ("raglab-training-manifest", 3),
        "benchmark-summary-v3.schema.json": ("benchmark-summary", 3),
        "raft-partition-manifest-v1.schema.json": ("raft-partition-manifest", 1),
        "squad-paired-evaluation-v1.schema.json": ("squad-paired-evaluation", 1),
        "benchmark-plan-v1.schema.json": ("benchmark-plan", 1),
    }
    for filename, (name, version) in expected.items():
        schema = load_artifact_schema(filename)
        assert schema["properties"]["schema_name"]["const"] == name
        assert schema["properties"]["schema_version"]["const"] == version
        assert json.loads(
            (PROJECT_ROOT / "docs" / "schemas" / filename).read_text(encoding="utf-8")
        ) == schema


def test_representative_standalone_summary_validates_its_own_schema() -> None:
    summary = {
        "schema_name": "squad-paired-evaluation",
        "schema_version": 1,
        "retrieval": {},
        "base_rag": {},
        "base_oracle": {},
        "tuned_rag": {},
        "tuned_oracle": {},
        "delta_rag": {},
        "delta_oracle": {},
        "paired_bootstrap": {},
        "peak_allocated_vram_gb": 1.0,
        "peak_reserved_vram_gb": 2.0,
        "model_id": "test/model",
        "model_revision": "0" * 40,
        "chat_template_kwargs": {},
        "thinking_enabled": False,
        "seed": 42,
    }
    validate_artifact_schema(summary, "squad-paired-evaluation-v1.schema.json")


def test_representative_training_manifest_validates_nested_contract() -> None:
    digest = "a" * 64
    manifest = {
        "schema_name": "raglab-training-manifest",
        "schema_version": 3,
        "recipe": "test-sft",
        "mode": "sft",
        "adaptation_mode": "sft",
        "model": {"model_id": "test/model", "revision": "0" * 40},
        "training_prompt": {
            "name": "rag-user-prompt",
            "version": "4",
            "template_sha256": digest,
        },
        "chat_template_kwargs": {},
        "training_config": {},
        "training_configuration_sha256": digest,
        "training_controls": normalize_training_controls({}, has_validation=True),
        "training_control_sha256": digest,
        "training_dataset_fingerprint": digest,
        "validation_dataset_fingerprint": digest,
        "training_source_fingerprint": digest,
        "validation_source_fingerprint": digest,
        "held_out_evaluation_sha256": digest,
        "configuration_files": {},
        "loss_masking": {
            "strategy": "completion-only",
            "prompt_tokens_in_loss": False,
            "trl_completion_only_loss": True,
        },
        "split": {
            "strategy": "grouped",
            "group_by": ["normalized_question"],
            "group_counts": {},
            "corpus_policy": "shared-corpus",
            "partition_fingerprints": {},
            "document_overlap_count": 0,
            "question_overlap_count": 0,
            "negative_mining_scope": "not-applicable",
            "seed": 42,
            "train_examples": 2,
            "validation_examples": 1,
        },
        "best_checkpoint": None,
        "best_validation_metric": None,
        "metric_for_best_model": "eval_loss",
        "train_metrics": {},
        "validation_metrics": {},
        "adapter_path": "outputs/adapter",
        "adapter_artifact_sha256": digest,
    }
    validate_artifact_schema(manifest, "training-manifest-v3.schema.json")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda controls: controls.update(adapter={"rank": 8}),
        lambda controls: controls["adapter"].update(rank=0),
        lambda controls: controls["adapter"].update(dropout=1.1),
        lambda controls: controls["quantization"].update(type="int8"),
        lambda controls: controls.pop("seeds"),
        lambda controls: controls.pop("checkpoint_selection"),
        lambda controls: controls["adapter"].update(unknown_weight_control=True),
    ],
)
def test_incomplete_or_invalid_training_controls_fail_json_schema(
    mutation: object,
) -> None:
    controls = normalize_training_controls({}, has_validation=True)
    mutation(controls)  # type: ignore[operator]
    with pytest.raises(ValueError, match="training-controls-v1"):
        validate_artifact_schema(controls, "training-controls-v1.schema.json")


def test_effective_batch_size_relationship_is_runtime_validated() -> None:
    controls = copy.deepcopy(normalize_training_controls({}, has_validation=True))
    controls["batching"]["effective_batch_size"] += 1
    validate_artifact_schema(controls, "training-controls-v1.schema.json")
    with pytest.raises(ValueError, match="effective_batch_size"):
        validate_training_controls(controls)
