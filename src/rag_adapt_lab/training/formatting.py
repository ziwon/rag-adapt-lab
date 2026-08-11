from __future__ import annotations

from typing import Any


def format_sft_row(row: dict[str, Any]) -> str:
    instruction = row.get("instruction", "Answer accurately.")
    user_input = row.get("input", row.get("question", ""))
    output = row.get("output", row.get("answer", ""))
    return (
        "### Instruction\n"
        f"{instruction}\n\n"
        "### Input\n"
        f"{user_input}\n\n"
        "### Response\n"
        f"{output}"
    )


def format_raft_row(row: dict[str, Any]) -> str:
    contexts = row.get("contexts", [])
    rendered_contexts = []
    for i, item in enumerate(contexts, start=1):
        marker = "relevant" if item.get("relevant") else "distractor"
        rendered_contexts.append(f"[Document {i} | {marker}]\n{item.get('text', '')}")
    context_block = "\n\n".join(rendered_contexts)
    return (
        "### Instruction\n"
        "Answer the question using relevant evidence from the provided documents. "
        "Ignore distractor documents.\n\n"
        f"### Documents\n{context_block}\n\n"
        f"### Question\n{row.get('question', '')}\n\n"
        f"### Response\n{row.get('answer', '')}"
    )
