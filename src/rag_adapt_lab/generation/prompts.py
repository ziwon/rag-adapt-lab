from __future__ import annotations

from collections.abc import Sequence


def format_rag_user_prompt(*, question: str, contexts: Sequence[str]) -> str:
    """Build the shared retrieval prompt used for adaptation and inference."""
    if not contexts:
        return question
    context_block = "\n\n".join(
        f"[Document {index}]\n{text}" for index, text in enumerate(contexts, start=1)
    )
    return (
        "### Instruction\n"
        "Answer the question using relevant evidence from the provided documents. "
        "Ignore irrelevant documents and return only the concise answer.\n\n"
        f"### Documents\n{context_block}\n\n"
        f"### Question\n{question}"
    )
