import pytest

from rag_adapt_lab.evaluation.generation import exact_match, token_f1
from rag_adapt_lab.generation.thinking import parse_thinking_tokens


class SpecialThinkingTokenizer:
    unk_token_id = 0
    pad_token_id = 1
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "reason": 10,
        "wrong": 11,
        "Seoul": 20,
        "<think>": 98,
        "</think>": 99,
    }

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocabulary.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [self.convert_tokens_to_ids(text)]

    def decode(self, token_ids: list[int] | tuple[int, ...], *, skip_special_tokens: bool) -> str:
        reverse = {value: key for key, value in self.vocabulary.items()}
        special = {0, 1, 98, 99}
        return " ".join(
            reverse[value]
            for value in token_ids
            if not skip_special_tokens or value not in special
        )


class SharedPadEOSThinkingTokenizer(SpecialThinkingTokenizer):
    eos_token_id = 1


def test_exact_match_normalizes_case_and_punctuation() -> None:
    assert exact_match("Hello, World!", "hello world") == 1.0


def test_token_f1_partial_overlap() -> None:
    score = token_f1("a b", "a c")
    assert 0.0 < score < 1.0


def test_thinking_parser_scores_only_the_final_answer() -> None:
    tokenizer = SpecialThinkingTokenizer()
    parsed = parse_thinking_tokens(
        [10, 11, 99, 20, 1],
        tokenizer=tokenizer,
        thinking_enabled=True,
    )
    assert parsed.reasoning == "reason wrong"
    assert parsed.answer == "Seoul"
    assert parsed.raw_text == "reason wrong </think> Seoul"
    assert parsed.boundary_token_id == 99
    assert parsed.boundary_found is True
    assert parsed.reasoning_tokens == 2
    assert parsed.answer_tokens == 1
    assert parsed.output_tokens == 4
    assert exact_match(parsed.answer, "Seoul") == 1.0


def test_token_counts_keep_one_generated_eos_but_exclude_batch_padding() -> None:
    parsed = parse_thinking_tokens(
        [10, 99, 20, 1, 1],
        tokenizer=SharedPadEOSThinkingTokenizer(),
        thinking_enabled=True,
    )
    assert parsed.answer == "Seoul"
    assert parsed.answer_token_ids == (20, 1)
    assert parsed.answer_tokens == 2
    assert parsed.output_tokens == 4


def test_special_boundary_survives_when_normal_decode_hides_it() -> None:
    tokenizer = SpecialThinkingTokenizer()
    assert tokenizer.decode([10, 99, 20], skip_special_tokens=True) == "reason Seoul"
    parsed = parse_thinking_tokens(
        [10, 99, 20], tokenizer=tokenizer, thinking_enabled=True
    )
    assert parsed.reasoning == "reason"
    assert parsed.answer == "Seoul"


def test_thinking_parser_does_not_require_generated_opening_boundary() -> None:
    parsed = parse_thinking_tokens(
        [10, 99, 20],
        tokenizer=SpecialThinkingTokenizer(),
        thinking_enabled=True,
    )
    assert parsed.boundary_found is True


def test_thinking_parser_accepts_one_generated_opening_boundary() -> None:
    parsed = parse_thinking_tokens(
        [98, 10, 99, 20],
        tokenizer=SpecialThinkingTokenizer(),
        thinking_enabled=True,
    )
    assert parsed.reasoning == "reason"
    assert parsed.answer == "Seoul"
    assert parsed.reasoning_tokens == 1
    assert parsed.answer_tokens == 1
    assert parsed.output_tokens == 4


def test_thinking_parser_rejects_missing_closing_boundary() -> None:
    with pytest.raises(ValueError, match="missing the </think> boundary"):
        parse_thinking_tokens(
            [10, 20],
            tokenizer=SpecialThinkingTokenizer(),
            thinking_enabled=True,
        )


def test_thinking_parser_rejects_ambiguous_blocks() -> None:
    with pytest.raises(ValueError, match="multiple </think> boundaries"):
        parse_thinking_tokens(
            [10, 99, 11, 99, 20],
            tokenizer=SpecialThinkingTokenizer(),
            thinking_enabled=True,
        )


def test_thinking_parser_rejects_empty_final_answer() -> None:
    with pytest.raises(ValueError, match="final answer is empty"):
        parse_thinking_tokens(
            [10, 99, 1],
            tokenizer=SpecialThinkingTokenizer(),
            thinking_enabled=True,
        )


@pytest.mark.parametrize("boundary", [98, 99])
def test_thinking_parser_rejects_boundaries_when_thinking_is_disabled(boundary: int) -> None:
    with pytest.raises(ValueError, match="thinking-disabled"):
        parse_thinking_tokens(
            [10, boundary, 20],
            tokenizer=SpecialThinkingTokenizer(),
            thinking_enabled=False,
        )
