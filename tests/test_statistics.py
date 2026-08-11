import pytest

from rag_adapt_lab.evaluation.statistics import paired_bootstrap_delta, percentile


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
