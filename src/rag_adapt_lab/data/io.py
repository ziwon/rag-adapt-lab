from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .schema import Document, EvalExample, RAFTExample, Record, SFTExample

T = TypeVar("T", bound=Record)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {source}:{line_number}")
            rows.append(value)
    return rows


def _load_records(path: str | Path, model: type[T]) -> list[T]:
    source = Path(path)
    records: list[T] = []
    for line_number, row in enumerate(read_jsonl(source), start=1):
        try:
            records.append(model.model_validate(row))
        except ValidationError as exc:
            raise ValueError(f"Invalid record in {source}:{line_number}: {exc}") from exc

    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        record_id = str(record.id)
        if record_id in seen:
            duplicates.add(record_id)
        seen.add(record_id)
    if duplicates:
        raise ValueError(f"Duplicate IDs in {source}: {sorted(duplicates)[:10]}")
    return records


def load_documents(path: str | Path) -> list[Document]:
    return _load_records(path, Document)


def load_eval(path: str | Path) -> list[EvalExample]:
    return _load_records(path, EvalExample)


def load_qa_examples(path: str | Path) -> list[EvalExample]:
    """Load labeled QA rows used as either training source data or evaluation data."""
    return _load_records(path, EvalExample)


def load_sft(path: str | Path) -> list[SFTExample]:
    return _load_records(path, SFTExample)


def load_raft(path: str | Path) -> list[RAFTExample]:
    return _load_records(path, RAFTExample)


def write_jsonl(
    path: str | Path,
    rows: Iterable[BaseModel | Mapping[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            value = row.model_dump(mode="json") if isinstance(row, BaseModel) else dict(row)
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_raft_jsonl(
    path: str | Path,
    rows: Iterable[RAFTExample | Mapping[str, Any]],
) -> None:
    """Validate RAFT invariants immediately before persisting prepared data."""
    validated = [
        row if isinstance(row, RAFTExample) else RAFTExample.model_validate(dict(row))
        for row in rows
    ]
    write_jsonl(path, validated)
