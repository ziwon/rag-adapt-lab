import pytest

from rag_adapt_lab.data.schema import EvalExample
from rag_adapt_lab.data.validation import ensure_disjoint_qa_splits


def example(identifier: str, question: str) -> EvalExample:
    return EvalExample(id=identifier, question=question)


def test_disjoint_qa_splits_are_accepted() -> None:
    ensure_disjoint_qa_splits(
        [example("train-1", "A training question")],
        [example("eval-1", "A held-out question")],
    )


@pytest.mark.parametrize(
    ("training", "held_out", "message"),
    [
        (example("same", "training"), example("same", "evaluation"), "record IDs"),
        (
            example("train-1", "  What IS BM25? "),
            example("eval-1", "what is bm25?"),
            "normalized questions",
        ),
    ],
)
def test_overlapping_qa_splits_are_rejected(
    training: EvalExample,
    held_out: EvalExample,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ensure_disjoint_qa_splits([training], [held_out])
