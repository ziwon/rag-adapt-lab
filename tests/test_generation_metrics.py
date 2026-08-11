import pytest

from rag_adapt_lab.evaluation.generation import exact_match, token_f1
from rag_adapt_lab.generation.transformers import parse_thinking_output


def test_exact_match_normalizes_case_and_punctuation() -> None:
    assert exact_match("Hello, World!", "hello world") == 1.0


def test_token_f1_partial_overlap() -> None:
    score = token_f1("a b", "a c")
    assert 0.0 < score < 1.0


def test_thinking_parser_scores_only_the_final_answer() -> None:
    reasoning, answer = parse_thinking_output(
        "<think>Long private reasoning with a wrong guess.</think> Seoul",
        thinking_enabled=True,
    )
    assert reasoning == "Long private reasoning with a wrong guess."
    assert answer == "Seoul"
    assert exact_match(answer, "Seoul") == 1.0


def test_thinking_parser_rejects_ambiguous_blocks() -> None:
    with pytest.raises(ValueError, match="multiple reasoning blocks"):
        parse_thinking_output(
            "<think>one</think><think>two</think>answer",
            thinking_enabled=True,
        )
