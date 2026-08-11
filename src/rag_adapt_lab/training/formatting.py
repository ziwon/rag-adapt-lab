from __future__ import annotations

from typing import Any

from rag_adapt_lab.generation.prompts import format_rag_user_prompt


def format_sft_row(row: dict[str, Any]) -> str:
    instruction = row.get("instruction", "Answer accurately.")
    user_input = row.get("input", row.get("question", ""))
    output = row.get("output", row.get("answer", ""))
    return f"### Instruction\n{instruction}\n\n### Input\n{user_input}\n\n### Response\n{output}"


def format_raft_user_prompt(row: dict[str, Any]) -> str:
    return format_rag_user_prompt(
        question=str(row.get("question", "")),
        contexts=[str(item.get("text", "")) for item in row.get("contexts", [])],
    )


def format_raft_prompt(row: dict[str, Any]) -> str:
    return f"{format_raft_user_prompt(row)}\n\n### Response\n"


def format_raft_row(row: dict[str, Any]) -> str:
    return f"{format_raft_prompt(row)}{row.get('answer', '')}"
