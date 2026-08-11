from rag_adapt_lab.data.raft import build_raft_examples
from rag_adapt_lab.data.schema import Document, EvalExample
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
