from __future__ import annotations

import random

from .schema import Document, EvalExample, RAFTContext, RAFTExample


def build_raft_examples(
    documents: list[Document],
    examples: list[EvalExample],
    *,
    distractors: int = 2,
    seed: int = 42,
) -> list[RAFTExample]:
    if distractors < 0:
        raise ValueError("distractors must be non-negative")

    documents_by_id = {document.id: document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("Document IDs must be unique")

    rng = random.Random(seed)
    rows: list[RAFTExample] = []
    for example in examples:
        if not example.reference_answer or not example.reference_answer.strip():
            raise ValueError(f"RAFT example {example.id!r} has no reference answer")
        if not example.relevant_doc_ids:
            raise ValueError(f"RAFT example {example.id!r} has no relevant documents")

        relevant_ids = list(dict.fromkeys(example.relevant_doc_ids))
        missing = [doc_id for doc_id in relevant_ids if doc_id not in documents_by_id]
        if missing:
            raise ValueError(f"RAFT example {example.id!r} references missing documents: {missing}")

        contexts = [
            RAFTContext(doc_id=doc_id, text=documents_by_id[doc_id].text, relevant=True)
            for doc_id in relevant_ids
        ]
        candidates = [document for document in documents if document.id not in relevant_ids]
        sampled = rng.sample(candidates, k=min(distractors, len(candidates)))
        contexts.extend(
            RAFTContext(doc_id=document.id, text=document.text, relevant=False)
            for document in sampled
        )
        rng.shuffle(contexts)
        rows.append(
            RAFTExample(
                id=example.id,
                question=example.question,
                answer=example.reference_answer,
                contexts=contexts,
                evidence_doc_ids=relevant_ids,
                metadata=example.metadata,
            )
        )
    return rows
