from __future__ import annotations

from collections.abc import Sequence

from .schema import EvalExample


def normalize_question(question: str) -> str:
    """Normalize a question enough to catch accidental exact split reuse."""
    return " ".join(question.casefold().split())


def ensure_disjoint_qa_splits(
    training_examples: Sequence[EvalExample],
    held_out_examples: Sequence[EvalExample],
) -> None:
    """Reject training/evaluation overlap by record ID or normalized question."""
    training_ids = {example.id for example in training_examples}
    held_out_ids = {example.id for example in held_out_examples}
    overlapping_ids = sorted(training_ids & held_out_ids)

    training_questions = {
        normalize_question(example.question): example.id for example in training_examples
    }
    held_out_questions = {
        normalize_question(example.question): example.id for example in held_out_examples
    }
    overlapping_questions = sorted(set(training_questions) & set(held_out_questions))

    problems: list[str] = []
    if overlapping_ids:
        problems.append(f"record IDs {overlapping_ids[:10]}")
    if overlapping_questions:
        examples = [
            f"{training_questions[question]!r}/{held_out_questions[question]!r}: {question!r}"
            for question in overlapping_questions[:5]
        ]
        problems.append(f"normalized questions {examples}")
    if problems:
        raise ValueError(
            "Training data overlaps the held-out evaluation split by " + "; ".join(problems)
        )
