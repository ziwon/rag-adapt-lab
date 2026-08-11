from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rag_adapt_lab.data.validation import normalize_question

from .formatting import (
    format_raft_prompt,
    format_raft_user_prompt,
    format_sft_prompt,
    format_sft_user_prompt,
)

TrainingMode = Literal["sft", "raft"]


@dataclass(frozen=True, slots=True)
class TrainingSplit:
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    method: str
    seed: int
    validation_ratio: float


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id", "")).strip()


def _row_prompt(row: Mapping[str, Any]) -> str:
    value = row.get("question", row.get("input", ""))
    return normalize_question(str(value))


def ensure_disjoint_training_rows(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> None:
    left_ids = {_row_id(row) for row in left if _row_id(row)}
    right_ids = {_row_id(row) for row in right if _row_id(row)}
    duplicate_ids = sorted(left_ids & right_ids)
    left_prompts = {_row_prompt(row) for row in left if _row_prompt(row)}
    right_prompts = {_row_prompt(row) for row in right if _row_prompt(row)}
    duplicate_prompts = sorted(left_prompts & right_prompts)
    if duplicate_ids or duplicate_prompts:
        details: list[str] = []
        if duplicate_ids:
            details.append(f"IDs {duplicate_ids[:10]}")
        if duplicate_prompts:
            details.append(f"normalized prompts {duplicate_prompts[:5]}")
        raise ValueError(f"{left_name} overlaps {right_name}: " + "; ".join(details))


def deterministic_training_split(
    rows: Sequence[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
) -> TrainingSplit:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    copied = [dict(row) for row in rows]
    if validation_ratio == 0.0:
        return TrainingSplit(copied, [], "none", seed, validation_ratio)
    if len(copied) < 2:
        raise ValueError("At least two rows are required for a validation split")
    validation_count = max(1, round(len(copied) * validation_ratio))
    validation_count = min(validation_count, len(copied) - 1)
    indices = list(range(len(copied)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_count])
    train = [row for index, row in enumerate(copied) if index not in validation_indices]
    validation = [row for index, row in enumerate(copied) if index in validation_indices]
    ensure_disjoint_training_rows(
        train,
        validation,
        left_name="training split",
        right_name="validation split",
    )
    return TrainingSplit(train, validation, "deterministic-split", seed, validation_ratio)


def prompt_completion_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: TrainingMode,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    """Create TRL prompt-completion rows so prompt tokens can be masked."""
    if mode not in {"sft", "raft"}:
        raise ValueError(f"Unsupported training mode: {mode!r}")
    records: list[dict[str, Any]] = []
    for row in rows:
        if mode == "raft":
            user_prompt = format_raft_user_prompt(dict(row))
            plain_prompt = format_raft_prompt(dict(row))
            completion = str(row.get("answer", ""))
        else:
            user_prompt = format_sft_user_prompt(dict(row))
            plain_prompt = format_sft_prompt(dict(row))
            completion = str(row.get("output", row.get("answer", "")))
        if not completion.strip():
            raise ValueError(f"Training row {_row_id(row)!r} has an empty completion")
        if use_chat_template:
            records.append(
                {
                    "prompt": [{"role": "user", "content": user_prompt}],
                    "completion": [{"role": "assistant", "content": completion}],
                }
            )
        else:
            records.append({"prompt": plain_prompt, "completion": completion})
    return records


def split_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(row) for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
