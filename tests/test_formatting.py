from rag_adapt_lab.generation.prompts import format_rag_user_prompt
from rag_adapt_lab.training.formatting import (
    format_raft_prompt,
    format_raft_row,
    format_raft_user_prompt,
    format_sft_user_prompt,
)


def test_raft_prompt_does_not_expose_oracle_relevance() -> None:
    row = {
        "id": "raft-1",
        "question": "Which document matters?",
        "answer": "the first",
        "contexts": [
            {"doc_id": "oracle", "text": "oracle", "relevant": True},
            {"doc_id": "noise", "text": "noise", "relevant": False},
        ],
        "evidence_doc_ids": ["oracle"],
    }
    prompt = format_raft_prompt(row)
    assert "| relevant" not in prompt.lower()
    assert "| distractor" not in prompt.lower()
    assert "[Document 1]" in prompt
    assert prompt.endswith("### Response\n")
    assert format_raft_row(row) == f"{prompt}the first"


def test_training_and_inference_share_the_same_rag_prompt() -> None:
    row = {
        "id": "raft-2",
        "question": "What happened?",
        "answer": "first",
        "contexts": [
            {"doc_id": "first", "text": "first", "relevant": True},
            {"doc_id": "second", "text": "second", "relevant": False},
        ],
        "evidence_doc_ids": ["first"],
    }
    expected = format_rag_user_prompt(
        question="What happened?",
        contexts=["first", "second"],
    )
    assert format_raft_user_prompt(row) == expected
    base_prompt = format_rag_user_prompt(question="No retrieval", contexts=[])
    assert "### Instruction" in base_prompt
    assert "(no documents provided)" in base_prompt
    assert "### Question\nNo retrieval" in base_prompt
    assert base_prompt.endswith("### Documents\n(no documents provided)")
    assert format_sft_user_prompt({"input": "No retrieval"}) == base_prompt


def test_sft_and_raft_differ_only_by_supplied_contexts() -> None:
    question = "What happened?"
    sft = format_sft_user_prompt({"input": question, "instruction": "Ignore this variant"})
    raft = format_raft_user_prompt(
        {
            "id": "raft-3",
            "question": question,
            "answer": "positive",
            "contexts": [
                {"doc_id": "positive", "text": "positive", "relevant": True},
                {"doc_id": "distractor", "text": "distractor", "relevant": False},
            ],
            "evidence_doc_ids": ["positive"],
        }
    )
    assert sft == format_rag_user_prompt(question=question, contexts=[])
    assert raft == format_rag_user_prompt(
        question=question,
        contexts=["positive", "distractor"],
    )
    assert '"relevant"' not in raft.casefold()
    assert "oracle" not in raft.casefold()
