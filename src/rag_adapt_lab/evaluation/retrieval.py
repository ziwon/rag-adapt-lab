from __future__ import annotations

import math
from dataclasses import dataclass

from rag_adapt_lab.data.schema import EvalExample
from rag_adapt_lab.retrieval.base import Retriever


@dataclass(slots=True)
class RetrievalMetrics:
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float
    evaluated: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "retrieval/recall_at_k": self.recall_at_k,
            "retrieval/hit_rate_at_k": self.hit_rate_at_k,
            "retrieval/mrr": self.mrr,
            "retrieval/ndcg_at_k": self.ndcg_at_k,
            "retrieval/evaluated": self.evaluated,
        }


def evaluate_retriever(
    retriever: Retriever,
    examples: list[EvalExample],
    *,
    top_k: int = 5,
) -> RetrievalMetrics:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    recalls: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for example in examples:
        relevant = set(example.relevant_doc_ids)
        if not relevant:
            continue
        results = retriever.search(example.question, top_k=top_k)
        ranked_ids = [item.document.id for item in results[:top_k]]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for doc_id in ranked_ids:
            if doc_id in seen:
                duplicates.add(doc_id)
            seen.add(doc_id)
        if duplicates:
            raise ValueError(
                f"Retriever returned duplicate document IDs for example {example.id!r}: "
                f"{sorted(duplicates)}"
            )
        found = [doc_id for doc_id in ranked_ids if doc_id in relevant]

        recalls.append(len(set(found)) / len(relevant))
        hits.append(1.0 if found else 0.0)

        rr = 0.0
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        dcg = sum(
            (1.0 / math.log2(rank + 1))
            for rank, doc_id in enumerate(ranked_ids, start=1)
            if doc_id in relevant
        )
        ideal_hits = min(len(relevant), top_k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    n = len(recalls)
    if n == 0:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0)
    return RetrievalMetrics(
        recall_at_k=sum(recalls) / n,
        hit_rate_at_k=sum(hits) / n,
        mrr=sum(reciprocal_ranks) / n,
        ndcg_at_k=sum(ndcgs) / n,
        evaluated=n,
    )
