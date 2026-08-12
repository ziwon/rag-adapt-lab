from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .validation import normalize_question

SplitStrategy = Literal["row", "grouped"]
CorpusPolicy = Literal["shared-corpus", "document-disjoint"]


@dataclass(frozen=True, slots=True)
class PartitionAudit:
    train_fingerprint: str
    validation_fingerprint: str
    question_overlap_count: int
    document_overlap_count: int
    train_document_count: int
    validation_document_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_fingerprints": {
                "train": self.train_fingerprint,
                "validation": self.validation_fingerprint,
            },
            "question_overlap_count": self.question_overlap_count,
            "document_overlap_count": self.document_overlap_count,
            "train_document_count": self.train_document_count,
            "validation_document_count": self.validation_document_count,
        }


@dataclass(frozen=True, slots=True)
class PartitionSplit:
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    method: str
    seed: int
    validation_ratio: float
    strategy: SplitStrategy = "row"
    group_by: tuple[str, ...] = ()
    corpus_policy: CorpusPolicy = "shared-corpus"
    train_group_count: int = 0
    validation_group_count: int = 0
    audit: PartitionAudit | None = None
    negative_mining_scope: str = "not-applicable"

    def metadata(self) -> dict[str, Any]:
        output = {
            "method": self.method,
            "strategy": self.strategy,
            "seed": self.seed,
            "validation_ratio": self.validation_ratio,
            "actual_validation_ratio": (
                len(self.validation_rows) / (len(self.train_rows) + len(self.validation_rows))
                if self.train_rows or self.validation_rows
                else 0.0
            ),
            "group_by": list(self.group_by),
            "group_counts": {
                "train": self.train_group_count,
                "validation": self.validation_group_count,
            },
            "corpus_policy": self.corpus_policy,
            "negative_mining_scope": self.negative_mining_scope,
        }
        if self.audit is not None:
            output.update(self.audit.as_dict())
        return output


def rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(row) for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_partition_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash representation-independent source identities for SFT/RAFT pairing.

    Full dataset fingerprints intentionally differ after RAFT context mining.
    This fingerprint uses the stable source ID, normalized question, and target
    answer so two adaptation modes can prove that they were trained and
    validated on the same underlying examples and labels.
    """
    identities: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row in rows:
        source_id = str(row.get("id", "")).strip()
        question = normalize_question(row_question(row))
        answer = str(
            row.get("answer", row.get("output", row.get("reference_answer", "")))
        ).strip()
        if not source_id or not question or not answer:
            raise ValueError(
                "Verifiable training partitions require non-empty source IDs, questions, "
                "and answers"
            )
        if source_id in seen_ids:
            raise ValueError(f"Training partition contains duplicate source ID {source_id!r}")
        seen_ids.add(source_id)
        identities.append(
            {"id": source_id, "normalized_question": question, "answer": answer}
        )
    return rows_fingerprint(
        sorted(identities, key=lambda item: (item["id"], item["normalized_question"]))
    )


def row_question(row: Mapping[str, Any]) -> str:
    return str(row.get("question", row.get("input", "")))


def _nested_value(row: Mapping[str, Any], field_name: str) -> Any:
    value: Any = row
    for part in field_name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Grouping field {field_name!r} is missing from a training row")
        value = value[part]
    return value


def _stable_values(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    output = [
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)
        for item in values
    ]
    if not output or any(item in {'null', '""'} for item in output):
        raise ValueError("Grouping fields must contain non-empty values")
    return sorted(set(output))


def _group_tokens(row: Mapping[str, Any], fields: Sequence[str]) -> set[tuple[str, str]]:
    tokens: set[tuple[str, str]] = set()
    effective_fields = list(dict.fromkeys(["normalized_question", *fields]))
    for field_name in effective_fields:
        if field_name == "normalized_question":
            question = normalize_question(row_question(row))
            if not question:
                raise ValueError("Grouped splitting requires a non-empty question")
            values = [question]
        else:
            values = _stable_values(_nested_value(row, field_name))
        tokens.update((field_name, value) for value in values)
    return tokens


def positive_document_ids(row: Mapping[str, Any]) -> set[str]:
    ids = {str(value) for value in row.get("relevant_doc_ids", []) if str(value)}
    ids.update(str(value) for value in row.get("evidence_doc_ids", []) if str(value))
    for evidence in row.get("evidence", []):
        if isinstance(evidence, Mapping) and evidence.get("doc_id"):
            ids.add(str(evidence["doc_id"]))
    for context in row.get("contexts", []):
        if (
            isinstance(context, Mapping)
            and context.get("relevant") is True
            and context.get("doc_id")
        ):
            ids.add(str(context["doc_id"]))
    return ids


def all_document_ids(row: Mapping[str, Any]) -> set[str]:
    ids = positive_document_ids(row)
    for context in row.get("contexts", []):
        if isinstance(context, Mapping) and context.get("doc_id"):
            ids.add(str(context["doc_id"]))
    return ids


def partition_audit(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> PartitionAudit:
    train_questions = {normalize_question(row_question(row)) for row in train_rows}
    validation_questions = {normalize_question(row_question(row)) for row in validation_rows}
    train_questions.discard("")
    validation_questions.discard("")
    train_documents = set().union(*(all_document_ids(row) for row in train_rows)) if train_rows else set()
    validation_documents = (
        set().union(*(all_document_ids(row) for row in validation_rows))
        if validation_rows
        else set()
    )
    return PartitionAudit(
        train_fingerprint=rows_fingerprint(train_rows),
        validation_fingerprint=rows_fingerprint(validation_rows),
        question_overlap_count=len(train_questions & validation_questions),
        document_overlap_count=len(train_documents & validation_documents),
        train_document_count=len(train_documents),
        validation_document_count=len(validation_documents),
    )


def enforce_partition_policy(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    corpus_policy: CorpusPolicy,
    strategy: SplitStrategy = "row",
    group_by: Sequence[str] = (),
) -> PartitionAudit:
    if corpus_policy not in {"shared-corpus", "document-disjoint"}:
        raise ValueError(f"Unsupported corpus policy: {corpus_policy!r}")
    audit = partition_audit(train_rows, validation_rows)
    if audit.question_overlap_count:
        raise ValueError(
            "Training and validation partitions share normalized questions; use grouped splitting"
        )
    if strategy not in {"row", "grouped"}:
        raise ValueError(f"Unsupported split strategy: {strategy!r}")
    if strategy == "grouped":
        train_tokens = set().union(*(_group_tokens(row, group_by) for row in train_rows)) \
            if train_rows else set()
        validation_tokens = set().union(
            *(_group_tokens(row, group_by) for row in validation_rows)
        ) if validation_rows else set()
        overlaps = sorted(train_tokens & validation_tokens)
        if overlaps:
            details = [f"{field}={value}" for field, value in overlaps[:10]]
            raise ValueError(
                "Training and validation partitions share configured grouping values: "
                + ", ".join(details)
            )
    if corpus_policy == "document-disjoint" and audit.document_overlap_count:
        raise ValueError(
            "document-disjoint policy was violated: training and validation documents overlap"
        )
    return audit


def _connected_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
    corpus_policy: CorpusPolicy,
) -> list[list[int]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        tokens = _group_tokens(row, group_by)
        if corpus_policy == "document-disjoint":
            tokens.update(("positive_document", value) for value in positive_document_ids(row))
        for token in tokens:
            if token in owners:
                union(index, owners[token])
            else:
                owners[token] = index

    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda indices: min(indices))


def count_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy: SplitStrategy,
    group_by: Sequence[str],
    corpus_policy: CorpusPolicy,
) -> int:
    if strategy == "row":
        return len(rows)
    return len(_connected_groups(rows, group_by=group_by, corpus_policy=corpus_policy))


def split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
    strategy: SplitStrategy = "row",
    group_by: Sequence[str] = (),
    corpus_policy: CorpusPolicy = "shared-corpus",
) -> PartitionSplit:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if strategy not in {"row", "grouped"}:
        raise ValueError(f"Unsupported split strategy: {strategy!r}")
    if corpus_policy not in {"shared-corpus", "document-disjoint"}:
        raise ValueError(f"Unsupported corpus policy: {corpus_policy!r}")
    copied = [dict(row) for row in rows]
    if validation_ratio == 0.0:
        audit = partition_audit(copied, [])
        return PartitionSplit(
            copied,
            [],
            "none",
            seed,
            validation_ratio,
            strategy,
            tuple(group_by),
            corpus_policy,
            count_groups(
                copied,
                strategy=strategy,
                group_by=group_by,
                corpus_policy=corpus_policy,
            ),
            0,
            audit,
        )
    if len(copied) < 2:
        raise ValueError("At least two rows are required for a validation split")

    if strategy == "row":
        groups = [[index] for index in range(len(copied))]
    else:
        groups = _connected_groups(copied, group_by=group_by, corpus_policy=corpus_policy)
    if len(groups) < 2:
        raise ValueError(
            "Grouped split is impossible because all rows belong to one connected group"
        )

    target = max(1, round(len(copied) * validation_ratio))
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    validation_groups: set[int] = set()
    validation_rows_count = 0
    for group_index in order:
        if len(validation_groups) >= len(groups) - 1:
            break
        validation_groups.add(group_index)
        validation_rows_count += len(groups[group_index])
        if validation_rows_count >= target:
            break
    validation_indices = {
        index
        for group_index in validation_groups
        for index in groups[group_index]
    }
    train = [row for index, row in enumerate(copied) if index not in validation_indices]
    validation = [row for index, row in enumerate(copied) if index in validation_indices]
    audit = enforce_partition_policy(
        train,
        validation,
        corpus_policy=corpus_policy,
        strategy=strategy,
        group_by=group_by,
    )
    return PartitionSplit(
        train,
        validation,
        "deterministic-grouped-split" if strategy == "grouped" else "deterministic-row-split",
        seed,
        validation_ratio,
        strategy,
        tuple(dict.fromkeys(["normalized_question", *group_by])) if strategy == "grouped" else (),
        corpus_policy,
        len(groups) - len(validation_groups),
        len(validation_groups),
        audit,
    )
