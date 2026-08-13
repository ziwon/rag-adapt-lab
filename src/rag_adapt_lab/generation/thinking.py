from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

THINKING_OPEN_TOKEN = "<think>"
THINKING_CLOSE_TOKEN = "</think>"


class ThinkingTokenizer(Protocol):
    unk_token_id: int | None
    pad_token_id: int | None
    eos_token_id: int | None

    def get_vocab(self) -> Mapping[str, int]: ...

    def convert_tokens_to_ids(self, token: str) -> int | None: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ThinkingTokenParse:
    raw_text: str
    reasoning: str | None
    answer: str
    reasoning_token_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    boundary_token_id: int | None
    boundary_found: bool
    thinking_protocol_violation: str | None = None

    @property
    def output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def reasoning_tokens(self) -> int:
        return len(self.reasoning_token_ids)

    @property
    def answer_tokens(self) -> int:
        return len(self.answer_token_ids)


def resolve_special_token_id(tokenizer: ThinkingTokenizer, token: str) -> int | None:
    """Resolve a special-token ID without assuming a model-family constant."""
    vocabulary = tokenizer.get_vocab()
    if token in vocabulary:
        return int(vocabulary[token])

    converted = tokenizer.convert_tokens_to_ids(token)
    if converted is not None and converted != tokenizer.unk_token_id:
        encoded = tuple(int(value) for value in tokenizer.encode(token, add_special_tokens=False))
        if encoded == (int(converted),):
            return int(converted)

    encoded = tuple(int(value) for value in tokenizer.encode(token, add_special_tokens=False))
    if len(encoded) == 1 and encoded[0] != tokenizer.unk_token_id:
        return encoded[0]
    return None


def _trim_trailing_padding(
    token_ids: Sequence[int],
    *,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> tuple[int, ...]:
    values = [int(value) for value in token_ids]
    if pad_token_id is not None:
        minimum_trailing_tokens = 1 if pad_token_id == eos_token_id else 0
        trailing_tokens = 0
        for value in reversed(values):
            if value != pad_token_id:
                break
            trailing_tokens += 1
        for _ in range(max(0, trailing_tokens - minimum_trailing_tokens)):
            values.pop()
    return tuple(values)


def parse_thinking_tokens(
    token_ids: Sequence[int],
    *,
    tokenizer: ThinkingTokenizer,
    thinking_enabled: bool,
) -> ThinkingTokenParse:
    """Split reasoning from the answer using generated token IDs.

    Qwen3's chat template can introduce the opening thinking boundary before
    generation begins, so a valid generated completion contains exactly one
    closing boundary and does not need to contain an opening token.
    """
    output_ids = _trim_trailing_padding(
        token_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    close_id = resolve_special_token_id(tokenizer, THINKING_CLOSE_TOKEN)
    open_id = resolve_special_token_id(tokenizer, THINKING_OPEN_TOKEN)
    close_positions = (
        [index for index, token_id in enumerate(output_ids) if token_id == close_id]
        if close_id is not None
        else []
    )
    open_positions = (
        [index for index, token_id in enumerate(output_ids) if token_id == open_id]
        if open_id is not None
        else []
    )
    raw_text = tokenizer.decode(output_ids, skip_special_tokens=False).strip()

    if thinking_enabled:
        if close_id is None:
            raise ValueError(
                "Thinking protocol requires tokenizer support for the </think> boundary token"
            )
        if not close_positions:
            raise ValueError(
                "Thinking protocol failure: generated output is missing the </think> boundary"
            )
        if len(close_positions) != 1:
            raise ValueError(
                "Thinking protocol is ambiguous: generated output contains multiple "
                "</think> boundaries"
            )
        if len(open_positions) > 1 or (
            open_positions and (open_positions[0] != 0 or open_positions[0] > close_positions[-1])
        ):
            raise ValueError(
                "Thinking protocol is ambiguous: generated output contains invalid "
                "<think> boundaries"
            )
        boundary_index = close_positions[-1]
        reasoning_start = 1 if open_positions else 0
        reasoning_ids = output_ids[reasoning_start:boundary_index]
        answer_ids = output_ids[boundary_index + 1 :]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        if not answer or not answer_ids:
            raise ValueError("Thinking protocol failure: final answer is empty after </think>")
        return ThinkingTokenParse(
            raw_text=raw_text,
            reasoning=tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip(),
            answer=answer,
            reasoning_token_ids=reasoning_ids,
            answer_token_ids=answer_ids,
            output_token_ids=output_ids,
            boundary_token_id=close_id,
            boundary_found=True,
        )

    unexpected = sorted(set(open_positions + close_positions))
    if unexpected:
        raise ValueError(
            "Thinking protocol violation: thinking-boundary token appeared in a "
            "thinking-disabled condition"
        )
    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return ThinkingTokenParse(
        raw_text=raw_text,
        reasoning=None,
        answer=answer,
        reasoning_token_ids=(),
        answer_token_ids=output_ids,
        output_token_ids=output_ids,
        boundary_token_id=close_id,
        boundary_found=False,
    )
