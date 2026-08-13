import pytest

from rag_adapt_lab.evaluation.statistics import (
    aggregate_prediction_rows,
    paired_bootstrap_delta,
    percentile,
)


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([0.0, 10.0], 0.95) == 9.5


def test_paired_bootstrap_is_deterministic() -> None:
    baseline = [{"id": "a", "token_f1": 0.0}, {"id": "b", "token_f1": 0.5}]
    candidate = [{"id": "a", "token_f1": 1.0}, {"id": "b", "token_f1": 1.0}]
    first = paired_bootstrap_delta(baseline, candidate, metric="token_f1", samples=200, seed=5)
    second = paired_bootstrap_delta(baseline, candidate, metric="token_f1", samples=200, seed=5)
    assert first == second
    assert first["delta"] == 0.75
    assert first["ci95_low"] > 0


def test_paired_bootstrap_rejects_unpaired_examples() -> None:
    with pytest.raises(ValueError, match="example IDs differ"):
        paired_bootstrap_delta(
            [{"id": "a", "exact_match": 0}],
            [{"id": "b", "exact_match": 1}],
            metric="exact_match",
        )


def test_paired_bootstrap_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate example IDs"):
        paired_bootstrap_delta(
            [{"id": "a", "token_f1": 0.0}, {"id": "a", "token_f1": 0.5}],
            [{"id": "a", "token_f1": 1.0}],
            metric="token_f1",
        )


def test_judge_aggregate_records_full_coverage_and_cache_counts() -> None:
    rows = [
        {
            "id": str(index),
            "exact_match": 1.0,
            "token_f1": 1.0,
            "scores": {
                "judge_status": "ok",
                "judge_correctness": 0.8,
                "judge_groundedness": 0.9,
                "judge_unsupported_claim_rate": 0.1,
                "judge_cache_hit": index == 0,
            },
        }
        for index in range(4)
    ]
    summary = aggregate_prediction_rows(rows, minimum_judge_metric_coverage=0.8)
    correctness = summary["judge_metrics"]["judge_correctness"]
    assert correctness["total_evaluation_examples"] == 4
    assert correctness["numeric_examples"] == 4
    assert correctness["judge_successes"] == 4
    assert correctness["judge_failures"] == 0
    assert correctness["coverage"] == 1.0
    assert correctness["cache_hits"] == 1
    assert correctness["cache_misses"] == 3
    assert correctness["status"] == "ok"


def test_partial_judge_coverage_preserves_deterministic_metrics() -> None:
    rows = [
        {
            "id": "ok",
            "exact_match": 1.0,
            "token_f1": 1.0,
            "scores": {
                "judge_status": "ok",
                "judge_correctness": 0.8,
                "judge_cache_hit": False,
            },
        },
        {
            "id": "failed",
            "exact_match": 0.0,
            "token_f1": 0.5,
            "scores": {"judge_status": "error", "judge_cache_hit": False},
        },
    ]
    summary = aggregate_prediction_rows(rows, minimum_judge_metric_coverage=0.8)
    assert summary["exact_match"] == 0.5
    assert summary["token_f1"] == 0.75
    assert summary["judge_correctness"] == 0.8
    assert summary["judge_metrics"]["judge_correctness"]["numeric_examples"] == 1
    assert summary["judge_metrics"]["judge_correctness"]["coverage"] == 0.5
    assert summary["judge_metrics"]["judge_correctness"]["status"] == "insufficient_coverage"


def test_disjoint_judge_successes_return_explicit_zero_paired_coverage() -> None:
    baseline = [
        {"id": "a", "scores": {"judge_correctness": 0.2}},
        {"id": "b", "scores": {}},
    ]
    candidate = [
        {"id": "a", "scores": {}},
        {"id": "b", "scores": {"judge_correctness": 0.8}},
    ]
    result = paired_bootstrap_delta(
        baseline,
        candidate,
        metric="judge_correctness",
        minimum_coverage=0.8,
        minimum_paired_examples=1,
    )
    assert result["baseline_numeric_examples"] == 1
    assert result["candidate_numeric_examples"] == 1
    assert result["paired_examples"] == 0
    assert result["paired_coverage"] == 0.0
    assert result["dropped_example_ids"] == ["a", "b"]
    assert result["status"] == "insufficient_coverage"


def test_paired_sample_threshold_suppresses_statistical_decision() -> None:
    baseline = [{"id": str(index), "scores": {"judge_correctness": 0.0}} for index in range(2)]
    candidate = [{"id": str(index), "scores": {"judge_correctness": 1.0}} for index in range(2)]
    result = paired_bootstrap_delta(
        baseline,
        candidate,
        metric="judge_correctness",
        samples=100,
        minimum_coverage=0.8,
        minimum_paired_examples=3,
    )
    assert result["ci95_low"] > 0
    assert result["ci_excludes_zero"] is True
    assert result["statistically_significant"] is False
    assert result["decision_eligible"] is False
    assert result["status"] == "insufficient_coverage"
