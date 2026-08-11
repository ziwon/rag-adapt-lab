from rag_adapt_lab.evaluation.generation import exact_match, token_f1


def test_exact_match_normalizes_case_and_punctuation() -> None:
    assert exact_match("Hello, World!", "hello world") == 1.0


def test_token_f1_partial_overlap() -> None:
    score = token_f1("a b", "a c")
    assert 0.0 < score < 1.0
