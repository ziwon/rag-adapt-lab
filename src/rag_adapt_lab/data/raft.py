from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag_adapt_lab.retrieval.base import Retriever
from rag_adapt_lab.retrieval.bm25 import BM25Retriever
from rag_adapt_lab.schema_validation import validate_artifact_schema

from .schema import Document, EvalExample, RAFTContext, RAFTExample
from .splitting import CorpusPolicy, SplitStrategy, partition_audit, split_rows

NegativeStrategy = Literal["random", "bm25-hard-negative"]


def validate_distinct_output_paths(
    output: Path,
    validation_output: Path,
    manifest_output: Path,
) -> None:
    """Reject RAFT destinations that would overwrite one another."""
    targets = {
        "--output": output.resolve(),
        "--validation-output": validation_output.resolve(),
        "--manifest-output": manifest_output.resolve(),
    }
    duplicates = [
        f"{left} and {right}"
        for index, (left, left_path) in enumerate(targets.items())
        for right, right_path in list(targets.items())[index + 1 :]
        if left_path == right_path
    ]
    if duplicates:
        raise ValueError("RAFT output paths must be distinct: " + ", ".join(duplicates))


@dataclass(frozen=True, slots=True)
class MinedNegative:
    document: Document
    rank: int | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RAFTPartitions:
    train_rows: list[RAFTExample]
    validation_rows: list[RAFTExample]
    manifest: dict[str, Any]


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


def build_raft_contexts(
    *,
    positive_documents: Sequence[Document],
    negative_documents: Sequence[Document],
) -> list[RAFTContext]:
    """Build unambiguous RAFT evidence metadata from disjoint document groups."""
    positive_ids = [document.id for document in positive_documents]
    negative_ids = [document.id for document in negative_documents]
    if not positive_ids:
        raise ValueError("RAFT contexts require at least one positive document")
    duplicate_ids = sorted(
        {
            doc_id
            for doc_id in [*positive_ids, *negative_ids]
            if [*positive_ids, *negative_ids].count(doc_id) > 1
        }
    )
    if duplicate_ids:
        raise ValueError(f"RAFT context document IDs must be unique: {duplicate_ids}")
    return [
        *(
            RAFTContext(doc_id=document.id, text=document.text, relevant=True)
            for document in positive_documents
        ),
        *(
            RAFTContext(doc_id=document.id, text=document.text, relevant=False)
            for document in negative_documents
        ),
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
    mining_scope: str = "global",
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
        if mining_scope != "global" and len(sampled) < distractors:
            raise ValueError(
                f"RAFT example {example.id!r} requested {distractors} distractors but the "
                f"{mining_scope} document pool provides only {len(sampled)}"
            )
        contexts = build_raft_contexts(
            positive_documents=[documents_by_id[doc_id] for doc_id in relevant_ids],
            negative_documents=[negative.document for negative in sampled],
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
            "scope": mining_scope,
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


def _partition_document_pools(
    documents: list[Document],
    train_examples: list[EvalExample],
    validation_examples: list[EvalExample],
    *,
    corpus_policy: CorpusPolicy,
    validation_ratio: float,
    seed: int,
) -> tuple[list[Document], list[Document]]:
    if corpus_policy == "shared-corpus":
        return list(documents), list(documents)

    train_positive = {
        doc_id for example in train_examples for doc_id in example.relevant_doc_ids
    }
    validation_positive = {
        doc_id for example in validation_examples for doc_id in example.relevant_doc_ids
    }
    overlap = train_positive & validation_positive
    if overlap:
        raise ValueError(
            "document-disjoint split is impossible because positive documents cross partitions: "
            f"{sorted(overlap)[:10]}"
        )
    known = {document.id for document in documents}
    missing = sorted((train_positive | validation_positive) - known)
    if missing:
        raise ValueError(f"Split examples reference missing documents: {missing[:10]}")

    neutral = [
        document
        for document in documents
        if document.id not in train_positive and document.id not in validation_positive
    ]
    neutral.sort(
        key=lambda document: hashlib.sha256(f"{seed}:{document.id}".encode()).hexdigest()
    )
    validation_neutral_count = round(len(neutral) * validation_ratio)
    validation_neutral = {document.id for document in neutral[:validation_neutral_count]}
    train_ids = train_positive | {document.id for document in neutral if document.id not in validation_neutral}
    validation_ids = validation_positive | validation_neutral
    train_pool = [document for document in documents if document.id in train_ids]
    validation_pool = [document for document in documents if document.id in validation_ids]
    if not train_pool or not validation_pool:
        raise ValueError(
            "document-disjoint corpus allocation produced an empty document partition"
        )
    return train_pool, validation_pool


def build_raft_partitions(
    documents: list[Document],
    examples: list[EvalExample],
    *,
    validation_ratio: float = 0.1,
    seed: int = 42,
    split_strategy: SplitStrategy = "grouped",
    group_by: tuple[str, ...] = ("normalized_question",),
    corpus_policy: CorpusPolicy = "shared-corpus",
    distractors: int = 2,
    negative_strategy: NegativeStrategy = "random",
    candidate_pool_size: int = 20,
) -> RAFTPartitions:
    """Split labeled QA first, then mine negatives within each allowed partition."""
    raw_rows = [example.model_dump(mode="json") for example in examples]
    split = split_rows(
        raw_rows,
        validation_ratio=validation_ratio,
        seed=seed,
        strategy=split_strategy,
        group_by=group_by,
        corpus_policy=corpus_policy,
    )
    train_examples = [EvalExample.model_validate(row) for row in split.train_rows]
    validation_examples = [EvalExample.model_validate(row) for row in split.validation_rows]
    train_pool, validation_pool = _partition_document_pools(
        documents,
        train_examples,
        validation_examples,
        corpus_policy=corpus_policy,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    train_rows = build_raft_examples(
        train_pool,
        train_examples,
        distractors=distractors,
        seed=seed,
        negative_strategy=negative_strategy,
        candidate_pool_size=candidate_pool_size,
        mining_scope="train-partition-only",
    )
    validation_rows = build_raft_examples(
        validation_pool,
        validation_examples,
        distractors=distractors,
        seed=seed,
        negative_strategy=negative_strategy,
        candidate_pool_size=candidate_pool_size,
        mining_scope="validation-partition-only",
    )
    rendered_train = [row.model_dump(mode="json") for row in train_rows]
    rendered_validation = [row.model_dump(mode="json") for row in validation_rows]
    audit = partition_audit(rendered_train, rendered_validation)
    if corpus_policy == "document-disjoint" and audit.document_overlap_count:
        raise ValueError("document-disjoint RAFT mining leaked documents across partitions")
    manifest = {
        "schema_name": "raft-partition-manifest",
        "schema_version": 1,
        **split.metadata(),
        "negative_mining_scope": "split-before-mining",
        "negative_strategy": negative_strategy,
        "candidate_pool_size": candidate_pool_size,
        "distractors": distractors,
        "document_pools": {
            "train_count": len(train_pool),
            "validation_count": len(validation_pool),
            "overlap_count": len(
                {document.id for document in train_pool}
                & {document.id for document in validation_pool}
            ),
        },
        **audit.as_dict(),
    }
    validate_artifact_schema(manifest, "raft-partition-manifest-v1.schema.json")
    return RAFTPartitions(train_rows, validation_rows, manifest)
