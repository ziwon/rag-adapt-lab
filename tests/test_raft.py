from rag_adapt_lab.data.raft import build_raft_examples
from rag_adapt_lab.data.schema import Document, EvalExample


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
