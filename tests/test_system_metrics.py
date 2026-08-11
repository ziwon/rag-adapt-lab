import pytest

from rag_adapt_lab.benchmark.runner import compose_inference_e2e_latency
from rag_adapt_lab.evaluation.statistics import aggregate_prediction_rows, percentile
from rag_adapt_lab.generation.base import GenerationResult


def test_inference_e2e_includes_retrieval_through_decode_but_not_scoring() -> None:
    result = GenerationResult(
        text="answer",
        prompt_build_latency_s=0.01,
        chat_template_latency_s=0.02,
        tokenization_latency_s=0.03,
        device_transfer_latency_s=0.04,
        model_generate_latency_s=0.50,
        decode_latency_s=0.05,
    )
    measured = compose_inference_e2e_latency(result, retrieval_latency_s=0.10)
    assert measured == pytest.approx(0.75)


def test_latency_aggregation_uses_precise_names_and_deprecated_aliases() -> None:
    rows = [
        {
            "exact_match": 1.0,
            "token_f1": 1.0,
            "retrieval_used": True,
            "retrieval_latency_s": 0.1,
            "prompt_build_latency_s": 0.01,
            "chat_template_latency_s": 0.02,
            "tokenization_latency_s": 0.03,
            "device_transfer_latency_s": 0.04,
            "model_generate_latency_s": latency,
            "decode_latency_s": 0.05,
            "inference_e2e_latency_s": latency + 0.25,
            "scoring_latency_s": 0.02,
            "judge_latency_s": 4.0,
            "prompt_tokens": 10,
            "output_tokens": 2,
            "reasoning_tokens": 0,
            "answer_tokens": 2,
            "batch_size": 1,
            "generation_mode": "sequential",
            "scores": {"judge_status": "ok", "judge_cache_hit": False},
        }
        for latency in (1.0, 2.0, 3.0)
    ]
    summary = aggregate_prediction_rows(rows)
    assert summary["model_generate_latency_p50_s"] == 2.0
    assert summary["inference_e2e_latency_p95_s"] == pytest.approx(3.15)
    assert summary["judge_latency_mean_s"] == 4.0
    assert summary["inference_e2e_latency_p50_s"] == 2.25
    assert summary["latency_p50_s"] == summary["model_generate_latency_p50_s"]
    assert summary["end_to_end_latency_p50_s"] == summary["inference_e2e_latency_p50_s"]
    assert summary["batch_size"] == 1
    assert summary["generation_mode"] == "sequential"


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
