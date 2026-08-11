import pytest

from rag_adapt_lab.evaluation.scorers import (
    CallableJudgeBackend,
    CitationScorer,
    CompositeScorer,
    DeterministicCorrectnessScorer,
    LexicalGroundednessScorer,
    LLMJudgeScorer,
    NoOpScorer,
)


def test_deterministic_correctness_and_groundedness() -> None:
    scorer = CompositeScorer([DeterministicCorrectnessScorer(), LexicalGroundednessScorer()])
    result = scorer.score(
        question="Where?",
        answer="Seoul is in Korea.",
        reference="Seoul is in Korea",
        references=["Seoul is in Korea"],
        contexts=["Seoul is the capital city in Korea."],
    )
    assert result["answer_correctness"] == 1.0
    assert result["answer_exact_match"] == 1.0
    assert 0.0 < result["groundedness"] <= 1.0
    assert 0.0 <= result["unsupported_claim_rate"] <= 1.0


def test_noop_scorer_is_disabled() -> None:
    assert NoOpScorer().score(question="q", answer="a", reference=None, contexts=[]) == {}


def test_citation_precision_and_recall_use_retrieved_document_positions() -> None:
    result = CitationScorer().score(
        question="q",
        answer="Supported [Document 1], unsupported [Document 2].",
        reference="a",
        contexts=["supported", "distractor", "another relevant passage"],
        context_ids=["relevant-1", "noise", "relevant-2"],
        relevant_doc_ids=["relevant-1", "relevant-2"],
    )
    assert result["citation_precision"] == 0.5
    assert result["citation_recall"] == 0.5


def test_callable_judge_backend_is_pluggable_and_versioned() -> None:
    backend = CallableJudgeBackend(
        lambda prompt: {
            "correctness": 0.9,
            "groundedness": 0.8,
            "unsupported_claim_rate": 0.1,
            "rationale": "supported",
        },
        model_name="local-test-judge",
        model_revision="0" * 40,
    )
    scorer = LLMJudgeScorer(backend)
    result = scorer.score(
        question="q",
        answer="a",
        reference="a",
        contexts=["a"],
    )
    assert result["judge_correctness"] == 0.9
    assert scorer.metadata()["version"] == "rag-judge-v1"
    assert scorer.metadata()["backend"]["revision"] == "0" * 40


def test_judge_rejects_scores_outside_unit_interval() -> None:
    scorer = LLMJudgeScorer(
        CallableJudgeBackend(
            lambda prompt: {
                "correctness": 2,
                "groundedness": 0,
                "unsupported_claim_rate": 0,
            }
        )
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        scorer.score(question="q", answer="a", reference="a", contexts=["a"])
