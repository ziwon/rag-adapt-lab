from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal

from rag_adapt_lab.retrieval.base import Retriever
from rag_adapt_lab.retrieval.bm25 import BM25Retriever

from .schema import Document, EvalExample, RAFTContext, RAFTExample

NegativeStrategy = Literal["random", "bm25-hard-negative"]


@dataclass(frozen=True, slots=True)
class MinedNegative:
    document: Document
    rank: int | None = None
    score: float | None = None


def _example_rng(seed: int, example_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _random_negatives(
    documents: list[Document],
    *,
    relevant_ids: set[str],
    count: int,
    rng: random.Random,
) -> list[MinedNegative]:
    candidates = [document for document in documents if document.id not in relevant_ids]
    return [
        MinedNegative(document=document)
        for document in rng.sample(candidates, k=min(count, len(candidates)))
    ]


def _retrieved_negatives(
    retriever: Retriever,
    *,
    question: str,
    relevant_ids: set[str],
    count: int,
    document_count: int,
    candidate_pool_size: int,
    allowed_ids: set[str],
) -> list[MinedNegative]:
    top_n = min(document_count, max(candidate_pool_size, count + len(relevant_ids)))
    results = retriever.search(question, top_k=top_n)
    negatives = [
        result
        for result in results
        if result.document.id in allowed_ids and result.document.id not in relevant_ids
    ]
    if len(negatives) < count and top_n < document_count:
        results = retriever.search(question, top_k=document_count)
        negatives = [
            result
            for result in results
            if result.document.id in allowed_ids and result.document.id not in relevant_ids
        ]
    unique_negatives = []
    seen: set[str] = set()
    for result in negatives:
        if result.document.id in seen:
            continue
        seen.add(result.document.id)
        unique_negatives.append(result)
    return [
        MinedNegative(document=result.document, rank=result.rank, score=result.score)
        for result in unique_negatives[:count]
    ]


def build_raft_examples(
    documents: list[Document],
    examples: list[EvalExample],
    *,
    distractors: int = 2,
    seed: int = 42,
    negative_strategy: NegativeStrategy = "random",
    negative_retriever: Retriever | None = None,
    candidate_pool_size: int = 20,
) -> list[RAFTExample]:
    if distractors < 0:
        raise ValueError("distractors must be non-negative")
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be positive")
    if negative_strategy not in {"random", "bm25-hard-negative"}:
        raise ValueError(f"Unsupported negative strategy: {negative_strategy!r}")
    if not documents:
        raise ValueError("RAFT preparation requires at least one document")

    documents_by_id = {document.id: document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("Document IDs must be unique")

    if negative_strategy == "bm25-hard-negative":
        negative_retriever = negative_retriever or BM25Retriever()
        negative_retriever.index(documents)

    rows: list[RAFTExample] = []
    for example in examples:
        if not example.reference_answer or not example.reference_answer.strip():
            raise ValueError(f"RAFT example {example.id!r} has no reference answer")
        if not example.relevant_doc_ids:
            raise ValueError(f"RAFT example {example.id!r} has no relevant documents")

        relevant_ids = list(dict.fromkeys(example.relevant_doc_ids))
        relevant_set = set(relevant_ids)
        missing = [doc_id for doc_id in relevant_ids if doc_id not in documents_by_id]
        if missing:
            raise ValueError(f"RAFT example {example.id!r} references missing documents: {missing}")

        contexts = [
            RAFTContext(doc_id=doc_id, text=documents_by_id[doc_id].text, relevant=True)
            for doc_id in relevant_ids
        ]
        rng = _example_rng(seed, example.id)
        if negative_strategy == "random":
            sampled = _random_negatives(
                documents,
                relevant_ids=relevant_set,
                count=distractors,
                rng=rng,
            )
        else:
            assert negative_retriever is not None
            sampled = _retrieved_negatives(
                negative_retriever,
                question=example.question,
                relevant_ids=relevant_set,
                count=distractors,
                document_count=len(documents),
                candidate_pool_size=candidate_pool_size,
                allowed_ids=set(documents_by_id),
            )
        contexts.extend(
            RAFTContext(
                doc_id=negative.document.id,
                text=negative.document.text,
                relevant=False,
            )
            for negative in sampled
        )
        rng.shuffle(contexts)
        metadata = dict(example.metadata)
        metadata["negative_mining"] = {
            "strategy": negative_strategy,
            "seed": seed,
            "candidate_pool_size": candidate_pool_size,
            "distractors": [
                {
                    "doc_id": negative.document.id,
                    "rank": negative.rank,
                    "score": negative.score,
                }
                for negative in sampled
            ],
        }
        rows.append(
            RAFTExample(
                id=example.id,
                question=example.question,
                answer=example.reference_answer,
                contexts=contexts,
                evidence_doc_ids=relevant_ids,
                metadata=metadata,
            )
        )
    return rows
