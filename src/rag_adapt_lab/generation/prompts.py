from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

RAG_PROMPT_NAME = "rag-user-prompt"
RAG_PROMPT_VERSION = "3"


def format_rag_user_prompt(*, question: str, contexts: Sequence[str]) -> str:
    """Build the fixed prompt used by Base, RAG, and RAFT adaptation.

    The template remains identical across benchmark conditions. Only the
    contents of the documents section change, which avoids confounding the
    Base -> RAG comparison with a different instruction.
    """
    context_block = (
        "\n\n".join(f"[Document {index}]\n{text}" for index, text in enumerate(contexts, start=1))
        if contexts
        else "(no documents provided)"
    )
    return (
        "### Instruction\n"
        "Answer the question accurately and return only the concise answer. "
        "When documents are provided, use relevant evidence from them and ignore "
        "irrelevant documents.\n\n"
        f"### Question\n{question}\n\n"
        f"### Documents\n{context_block}"
    )


def rag_prompt_provenance() -> dict[str, Any]:
    """Return a stable identity for the model-facing training/inference prompt."""
    signature = format_rag_user_prompt(
        question="__QUESTION__",
        contexts=["__DOCUMENT_1__", "__DOCUMENT_2__"],
    )
    return {
        "name": RAG_PROMPT_NAME,
        "version": RAG_PROMPT_VERSION,
        "template_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
    }
