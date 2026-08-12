import json
from pathlib import Path

from rag_adapt_lab.data.schema import Document, EvalExample
from rag_adapt_lab.provenance import (
    ADAPTER_MANIFEST_SCHEMA_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    TRAINING_MANIFEST_SCHEMA_VERSION,
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
        schema = json.loads(
            (PROJECT_ROOT / "docs" / "schemas" / filename).read_text(encoding="utf-8")
        )
        assert schema["properties"]["schema_version"]["const"] == version
