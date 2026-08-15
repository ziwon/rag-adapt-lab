import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_adapt_lab.data.io import load_raft, write_raft_jsonl
from rag_adapt_lab.data.raft import build_raft_contexts, build_raft_examples
from rag_adapt_lab.data.schema import Document, EvalExample, RAFTExample
from rag_adapt_lab.retrieval.base import RetrievalResult, Retriever


class RankedRetriever(Retriever):
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def index(self, documents: list[Document]) -> None:
        self.documents = documents

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(document=document, score=1.0 / rank, rank=rank)
            for rank, document in enumerate(self.documents[:top_k], start=1)
        ]


def raft_row() -> dict[str, object]:
    return {
        "id": "q42",
        "question": "Which document answers the question?",
        "answer": "alpha",
        "contexts": [
            {"doc_id": "alpha", "text": "alpha answer", "relevant": True},
            {"doc_id": "beta", "text": "beta answer", "relevant": False},
        ],
        "evidence_doc_ids": ["alpha"],
    }


def test_valid_raft_evidence_metadata() -> None:
    row = RAFTExample.model_validate(raft_row())
    assert row.evidence_doc_ids == ["alpha"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row.update(contexts=[]),
            "contexts must contain at least one item",
        ),
        (
            lambda row: row.update(evidence_doc_ids=[]),
            "evidence_doc_ids must not be empty",
        ),
        (
            lambda row: row.update(evidence_doc_ids=["missing"]),
            "evidence document IDs missing from contexts=['missing']",
        ),
        (
            lambda row: row.update(evidence_doc_ids=["beta"]),
            "missing evidence document IDs=['alpha']",
        ),
        (
            lambda row: row["contexts"][0].update(relevant=False),  # type: ignore[index,union-attr]
            "evidence contexts marked relevant=false=['alpha']",
        ),
        (
            lambda row: row["contexts"][1].update(relevant=True),  # type: ignore[index,union-attr]
            "unexpected relevant document IDs=['beta']",
        ),
        (
            lambda row: row["contexts"].append(  # type: ignore[union-attr]
                {"doc_id": "alpha", "text": "duplicate", "relevant": True}
            ),
            "duplicated context IDs=['alpha']",
        ),
        (
            lambda row: row.update(evidence_doc_ids=["alpha", "alpha"]),
            "duplicated evidence IDs=['alpha']",
        ),
        (
            lambda row: [
                context.update(relevant=False)  # type: ignore[union-attr]
                for context in row["contexts"]  # type: ignore[union-attr]
            ],
            "no contexts are marked relevant=true",
        ),
    ],
)
def test_invalid_raft_evidence_metadata_is_rejected(
    mutation: object,
    message: str,
) -> None:
    row = raft_row()
    mutation(row)  # type: ignore[operator]
    with pytest.raises(ValidationError, match=message.replace("[", r"\[").replace("]", r"\]")):
        RAFTExample.model_validate(row)


def test_multiple_positive_contexts_are_valid() -> None:
    row = raft_row()
    row["contexts"] = [  # type: ignore[assignment]
        *row["contexts"],  # type: ignore[misc]
        {"doc_id": "gamma", "text": "supporting answer", "relevant": True},
    ]
    row["evidence_doc_ids"] = ["gamma", "alpha"]
    validated = RAFTExample.model_validate(row)
    assert {context.doc_id for context in validated.contexts if context.relevant} == {
        "alpha",
        "gamma",
    }


def test_invalid_external_raft_jsonl_is_rejected(tmp_path: Path) -> None:
    row = raft_row()
    row["evidence_doc_ids"] = ["beta"]
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="q42.*inconsistent evidence metadata"):
        load_raft(source)


def test_invalid_raft_jsonl_is_rejected_before_writing(tmp_path: Path) -> None:
    row = raft_row()
    row["evidence_doc_ids"] = ["beta"]
    with pytest.raises(ValidationError, match="q42.*inconsistent evidence metadata"):
        write_raft_jsonl(tmp_path / "invalid.jsonl", [row])


def test_build_raft_contexts_marks_disjoint_evidence() -> None:
    contexts = build_raft_contexts(
        positive_documents=[Document(id="alpha", text="alpha")],
        negative_documents=[Document(id="beta", text="beta")],
    )
    assert [(context.doc_id, context.relevant) for context in contexts] == [
        ("alpha", True),
        ("beta", False),
    ]


def test_build_raft_examples() -> None:
    docs = [
        Document(id="a", text="oracle"),
        Document(id="b", text="negative one"),
        Document(id="c", text="negative two"),
    ]
    eval_rows = [
        EvalExample(
            id="q1",
            question="what?",
            reference_answer="answer",
            relevant_doc_ids=["a"],
        )
    ]
    rows = build_raft_examples(docs, eval_rows, distractors=2, seed=1)
    assert len(rows) == 1
    assert rows[0].evidence_doc_ids == ["a"]
    assert sum(ctx.relevant for ctx in rows[0].contexts) == 1
    assert len(rows[0].contexts) == 3


def test_hard_negatives_exclude_positive_documents_and_are_deterministic() -> None:
    positive = Document(id="positive", text="the answer")
    hard = Document(id="hard", text="plausible but wrong")
    easy = Document(id="easy", text="unrelated")
    example = EvalExample(
        id="train-1",
        question="What is the answer?",
        reference_answer="the answer",
        relevant_doc_ids=["positive"],
    )
    retriever = RankedRetriever([positive, hard, easy])

    first = build_raft_examples(
        [positive, hard, easy],
        [example],
        distractors=2,
        seed=7,
        negative_strategy="bm25-hard-negative",
        negative_retriever=retriever,
        candidate_pool_size=3,
    )[0]
    second = build_raft_examples(
        [positive, hard, easy],
        [example],
        distractors=2,
        seed=7,
        negative_strategy="bm25-hard-negative",
        negative_retriever=retriever,
        candidate_pool_size=3,
    )[0]

    distractor_ids = [context.doc_id for context in first.contexts if not context.relevant]
    assert set(distractor_ids) == {"hard", "easy"}
    assert "positive" not in distractor_ids
    assert first == second
    assert first.metadata["negative_mining"]["strategy"] == "bm25-hard-negative"
