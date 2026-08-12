from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

RAG_PROMPT_NAME = "rag-user-prompt"
RAG_PROMPT_VERSION = "4"


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
    # Cover every rendering branch.  Hashing only the multi-document branch
    # would let the SFT/no-document prompt change without changing provenance.
    renderings = [
        format_rag_user_prompt(question="__QUESTION__", contexts=contexts)
        for contexts in (
            [],
            ["__DOCUMENT_1__"],
            ["__DOCUMENT_1__", "__DOCUMENT_2__"],
        )
    ]
    signature = json.dumps(
        renderings,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "name": RAG_PROMPT_NAME,
        "version": RAG_PROMPT_VERSION,
        "template_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
    }
