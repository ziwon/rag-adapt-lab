#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from rag_adapt_lab.data.io import write_jsonl
from rag_adapt_lab.data.raft import build_raft_examples
from rag_adapt_lab.data.schema import Document, EvalExample
from rag_adapt_lab.data.validation import ensure_disjoint_qa_splits

DATASET_ID = "rajpurkar/squad"
DATASET_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"


def context_id(prefix: str, context: str) -> str:
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def select_unique_contexts(
    dataset: Dataset,
    *,
    count: int,
    seed: int,
    max_context_chars: int,
    excluded_contexts: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    excluded = excluded_contexts or set()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in indices:
        row = dataset[index]
        context = " ".join(row["context"].split())
        answers = row["answers"]["text"]
        if (
            context in excluded
            or context in seen
            or not answers
            or len(context) > max_context_chars
        ):
            continue
        selected.append({**row, "context": context, "answer": answers[0].strip()})
        seen.add(context)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Could only select {len(selected)} of {count} requested records")


def make_documents(rows: list[dict[str, Any]], prefix: str) -> list[Document]:
    return [
        Document(
            id=context_id(prefix, row["context"]),
            text=row["context"],
            metadata={
                "dataset": DATASET_ID,
                "source_id": row["id"],
                "title": row["title"],
            },
        )
        for row in rows
    ]


def make_examples(rows: list[dict[str, Any]], prefix: str) -> list[EvalExample]:
    return [
        EvalExample(
            id=f"{prefix}-{row['id']}",
            question=row["question"],
            reference_answer=row["answer"],
            relevant_doc_ids=[context_id(f"{prefix}-doc", row["context"])],
            metadata={
                "dataset": DATASET_ID,
                "title": row["title"],
                "reference_answers": list(
                    dict.fromkeys(
                        answer.strip() for answer in row["answers"]["text"] if answer.strip()
                    )
                ),
            },
        )
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=256)
    parser.add_argument("--eval-examples", type=int, default=40)
    parser.add_argument("--eval-documents", type=int, default=100)
    parser.add_argument("--distractors", type=int, default=1)
    parser.add_argument(
        "--negative-strategy",
        choices=("random", "bm25-hard-negative"),
        default="bm25-hard-negative",
    )
    parser.add_argument("--hard-negative-candidates", type=int, default=20)
    parser.add_argument("--max-context-chars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_documents < args.eval_examples:
        raise ValueError("eval-documents must be at least eval-examples")

    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        cache_dir=str(args.cache_dir),
    )
    train_rows = select_unique_contexts(
        dataset["train"],
        count=args.train_examples,
        seed=args.seed,
        max_context_chars=args.max_context_chars,
    )
    train_contexts = {row["context"] for row in train_rows}
    validation_rows = select_unique_contexts(
        dataset["validation"],
        count=args.eval_documents,
        seed=args.seed + 1,
        max_context_chars=args.max_context_chars,
        excluded_contexts=train_contexts,
    )

    train_documents = make_documents(train_rows, "train-doc")
    train_examples = make_examples(train_rows, "train")
    raft_rows = build_raft_examples(
        train_documents,
        train_examples,
        distractors=args.distractors,
        seed=args.seed,
        negative_strategy=args.negative_strategy,
        candidate_pool_size=args.hard_negative_candidates,
    )
    sft_rows = [
        {
            "id": example.id,
            "instruction": "Answer the question accurately and return only the concise answer.",
            "input": example.question,
            "output": example.reference_answer,
            "metadata": example.metadata,
        }
        for example in train_examples
    ]
    eval_documents = make_documents(validation_rows, "eval-doc")
    eval_examples = make_examples(validation_rows[: args.eval_examples], "eval")

    ensure_disjoint_qa_splits(train_examples, eval_examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_documents.jsonl", train_documents)
    write_jsonl(args.output_dir / "sft_train.jsonl", sft_rows)
    write_jsonl(args.output_dir / "raft_train.jsonl", raft_rows)
    write_jsonl(args.output_dir / "eval_documents.jsonl", eval_documents)
    write_jsonl(args.output_dir / "eval.jsonl", eval_examples)
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "license": "cc-by-sa-4.0",
        "seed": args.seed,
        "train_examples": len(raft_rows),
        "sft_examples": len(sft_rows),
        "eval_examples": len(eval_examples),
        "eval_documents": len(eval_documents),
        "distractors_per_train_example": args.distractors,
        "negative_strategy": args.negative_strategy,
        "hard_negative_candidates": args.hard_negative_candidates,
        "max_context_chars": args.max_context_chars,
        "train_eval_context_overlap": len(
            train_contexts & {row["context"] for row in validation_rows}
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
