from __future__ import annotations

from collections.abc import Sequence

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
