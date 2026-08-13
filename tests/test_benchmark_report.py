from rag_adapt_lab.benchmark.report import render_markdown_report


def test_report_separates_metric_meanings_and_renders_paired_coverage() -> None:
    report = render_markdown_report(
        {
            "recipes": {
                "rag": {
                    "metrics": {
                        "examples": 40,
                        "exact_match": 0.4,
                        "token_f1": 0.6,
                        "reference_overlap": 0.6,
                        "lexical_groundedness": 0.7,
                        "judge_correctness": 0.8,
                        "judge_groundedness": 0.9,
                        "judge_examples": 40,
                        "judge_successes": 35,
                        "judge_failures": 5,
                        "judge_total_evaluation_examples": 40,
                        "judge_metrics": {
                            "judge_correctness": {
                                "numeric_examples": 35,
                                "total_evaluation_examples": 40,
                            }
                        },
                    }
                }
            },
            "comparisons": {
                "base->rag": {
                    "judge_correctness": {
                        "delta": 0.08,
                        "ci95_low": 0.02,
                        "ci95_high": 0.14,
                        "statistically_significant": True,
                        "paired_examples": 33,
                        "total_examples": 40,
                        "paired_coverage": 0.825,
                        "status": "ok",
                    }
                }
            },
            "retrieval_metrics": {},
            "configuration": {},
            "provenance": {},
        }
    )
    assert "Reference overlap" in report
    assert "Judge correctness" in report
    assert "Lexical groundedness" in report
    assert "Judge groundedness" in report
    assert "35/40 (87.5%)" in report
    assert "paired examples: 33/40" in report
    assert "paired coverage: 82.5%" in report
    assert "Correctness uses the configured judge" not in report


def test_report_does_not_relabel_overlap_when_judge_is_disabled() -> None:
    report = render_markdown_report(
        {
            "recipes": {
                "base": {
                    "metrics": {
                        "token_f1": 0.5,
                        "reference_overlap": 0.5,
                        "judge_examples": 0,
                        "judge_metrics": {"judge_correctness": {}},
                    }
                }
            },
            "comparisons": {},
            "retrieval_metrics": {},
            "configuration": {},
            "provenance": {},
        }
    )
    assert "optional LLM judge was disabled" in report
    assert "same normalization as Token F1" in report
