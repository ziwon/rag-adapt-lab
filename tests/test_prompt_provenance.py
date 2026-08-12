from rag_adapt_lab.generation import prompts


def test_prompt_provenance_hash_covers_empty_document_rendering(monkeypatch) -> None:
    before = prompts.rag_prompt_provenance()["template_sha256"]
    original = prompts.format_rag_user_prompt

    def changed(*, question: str, contexts: list[str]) -> str:
        rendered = original(question=question, contexts=contexts)
        return rendered + ("\nEMPTY-BRANCH-CHANGE" if not contexts else "")

    monkeypatch.setattr(prompts, "format_rag_user_prompt", changed)
    assert prompts.rag_prompt_provenance()["template_sha256"] != before
