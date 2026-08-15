from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from rag_adapt_lab.data.schema import RAFTExample
from rag_adapt_lab.data.splitting import (
    CorpusPolicy,
    PartitionSplit,
    SplitStrategy,
    rows_fingerprint,
    split_rows,
)
from rag_adapt_lab.data.validation import normalize_question

from .formatting import (
    format_raft_prompt,
    format_raft_user_prompt,
    format_sft_prompt,
    format_sft_user_prompt,
)

TrainingMode = Literal["sft", "raft"]


class ChatTemplateRenderer(Protocol):
    chat_template: str | None
    eos_token: str | None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str: ...


TrainingSplit = PartitionSplit


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
    return split_rows(
        rows,
        validation_ratio=validation_ratio,
        seed=seed,
        strategy="row",
    )


def configured_training_split(
    rows: Sequence[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
    strategy: SplitStrategy = "row",
    group_by: Sequence[str] = (),
    corpus_policy: CorpusPolicy = "shared-corpus",
) -> TrainingSplit:
    return split_rows(
        rows,
        validation_ratio=validation_ratio,
        seed=seed,
        strategy=strategy,
        group_by=group_by,
        corpus_policy=corpus_policy,
    )


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
    for source_row in rows:
        row: Mapping[str, Any] = source_row
        validated_raft: RAFTExample | None = None
        if mode == "raft" or "contexts" in row or "evidence_doc_ids" in row:
            validated_raft = RAFTExample.model_validate(dict(row))
            row = validated_raft.model_dump(mode="json")
        if mode == "raft":
            assert validated_raft is not None
            user_prompt = format_raft_user_prompt(dict(row))
            plain_prompt = format_raft_prompt(dict(row))
            completion = validated_raft.answer
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


def render_chat_prompt_completions(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: ChatTemplateRenderer,
    chat_template_kwargs: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Render chat records explicitly while retaining a completion-only boundary."""
    if not tokenizer.chat_template:
        raise ValueError("use_chat_template=true requires a tokenizer chat template")
    rendered: list[dict[str, str]] = []
    for record in records:
        prompt_messages = record.get("prompt")
        completion_messages = record.get("completion")
        if not isinstance(prompt_messages, list) or not isinstance(completion_messages, list):
            raise ValueError("Expected conversational prompt/completion records")
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if len(completion_messages) != 1 or not isinstance(completion_messages[0], Mapping):
            raise ValueError("Expected exactly one assistant completion message")
        completion = str(completion_messages[0].get("content", ""))
        if not completion.strip():
            raise ValueError("Assistant completion must not be empty")
        rendered.append(
            {
                "prompt": prompt,
                "completion": completion + (tokenizer.eos_token or ""),
            }
        )
    return rendered


def split_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return rows_fingerprint(rows)
