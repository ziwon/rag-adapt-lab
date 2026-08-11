import pytest

from rag_adapt_lab.data.schema import Document, EvalExample
from rag_adapt_lab.evaluation.retrieval import evaluate_retriever
from rag_adapt_lab.retrieval.base import RetrievalResult, Retriever


class StaticRetriever(Retriever):
    def __init__(self, ranked: list[Document]) -> None:
        self.ranked = ranked

    def index(self, documents: list[Document]) -> None:
        pass

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(document=doc, score=1.0 / rank, rank=rank)
            for rank, doc in enumerate(self.ranked[:top_k], start=1)
        ]


def test_retrieval_metrics() -> None:
    a = Document(id="a", text="a")
    b = Document(id="b", text="b")
    retriever = StaticRetriever([b, a])
    examples = [EvalExample(id="q", question="q", relevant_doc_ids=["a"])]
    metrics = evaluate_retriever(retriever, examples, top_k=2)
    assert metrics.recall_at_k == 1.0
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.mrr == 0.5
    assert 0.0 <= metrics.ndcg_at_k <= 1.0


def test_duplicate_retrieval_results_are_rejected() -> None:
    relevant = Document(id="a", text="a")
    retriever = StaticRetriever([relevant, relevant])
    examples = [EvalExample(id="q", question="q", relevant_doc_ids=["a"])]

    with pytest.raises(ValueError, match="duplicate document IDs"):
        evaluate_retriever(retriever, examples, top_k=2)
