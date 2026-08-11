from rag_adapt_lab.generation.prompts import format_rag_user_prompt
from rag_adapt_lab.training.formatting import (
    format_raft_prompt,
    format_raft_row,
    format_raft_user_prompt,
)


def test_raft_prompt_does_not_expose_oracle_relevance() -> None:
    row = {
        "question": "Which document matters?",
        "answer": "the first",
        "contexts": [
            {"text": "oracle", "relevant": True},
            {"text": "noise", "relevant": False},
        ],
    }
    prompt = format_raft_prompt(row)
    assert "| relevant" not in prompt.lower()
    assert "| distractor" not in prompt.lower()
    assert "[Document 1]" in prompt
    assert prompt.endswith("### Response\n")
    assert format_raft_row(row) == f"{prompt}the first"


def test_training_and_inference_share_the_same_rag_prompt() -> None:
    row = {
        "question": "What happened?",
        "contexts": [{"text": "first"}, {"text": "second"}],
    }
    expected = format_rag_user_prompt(
        question="What happened?",
        contexts=["first", "second"],
    )
    assert format_raft_user_prompt(row) == expected
    assert format_rag_user_prompt(question="No retrieval", contexts=[]) == "No retrieval"
