from rag_adapt_lab.data.schema import Document, EvalExample


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
