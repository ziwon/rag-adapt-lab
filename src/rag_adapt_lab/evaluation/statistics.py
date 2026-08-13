from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean_numeric(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [
        float(row[metric])
        for row in rows
        if isinstance(row.get(metric), (int, float)) and not isinstance(row.get(metric), bool)
    ]
    return statistics.fmean(values) if values else None


def aggregate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_judge_metric_coverage: float = 0.8,
) -> dict[str, Any]:
    if not 0.0 <= minimum_judge_metric_coverage <= 1.0:
        raise ValueError("minimum_judge_metric_coverage must be between 0 and 1")

    def numeric_values(field: str, *, retrieval_only: bool = False) -> list[float]:
        return [
            float(row[field])
            for row in rows
            if (not retrieval_only or row.get("retrieval_used") is True)
            and isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
        ]

    stage_fields = (
        "retrieval_latency_s",
        "prompt_build_latency_s",
        "chat_template_latency_s",
        "tokenization_latency_s",
        "device_transfer_latency_s",
        "model_generate_latency_s",
        "decode_latency_s",
        "inference_e2e_latency_s",
        "scoring_latency_s",
        "judge_latency_s",
    )
    token_timings = [
        (float(row["output_tokens"]), float(row["model_generate_latency_s"]))
        for row in rows
        if isinstance(row.get("output_tokens"), (int, float))
        and isinstance(row.get("model_generate_latency_s"), (int, float))
        and float(row["model_generate_latency_s"]) > 0
    ]
    total_token_timings = [
        (
            float(row["prompt_tokens"]) + float(row["output_tokens"]),
            float(row["model_generate_latency_s"]),
        )
        for row in rows
        if isinstance(row.get("prompt_tokens"), (int, float))
        and isinstance(row.get("output_tokens"), (int, float))
        and isinstance(row.get("model_generate_latency_s"), (int, float))
        and float(row["model_generate_latency_s"]) > 0
    ]
    summary: dict[str, Any] = {
        "examples": len(rows),
        "exact_match": mean_numeric(rows, "exact_match"),
        "token_f1": mean_numeric(rows, "token_f1"),
        "output_tokens_per_model_generate_second": (
            sum(tokens for tokens, _ in token_timings) / sum(timing for _, timing in token_timings)
            if token_timings
            else None
        ),
        "total_tokens_per_model_generate_second": (
            sum(tokens for tokens, _ in total_token_timings)
            / sum(timing for _, timing in total_token_timings)
            if total_token_timings
            else None
        ),
        "batch_size": (
            int(rows[0]["batch_size"])
            if rows and isinstance(rows[0].get("batch_size"), (int, float))
            else None
        ),
        "generation_mode": (
            str(rows[0]["generation_mode"]) if rows and rows[0].get("generation_mode") else None
        ),
        "prompt_tokens_total": sum(
            int(row["prompt_tokens"])
            for row in rows
            if isinstance(row.get("prompt_tokens"), (int, float))
        ),
        "output_tokens_total": sum(
            int(row["output_tokens"])
            for row in rows
            if isinstance(row.get("output_tokens"), (int, float))
        ),
        "reasoning_tokens_total": sum(
            int(row["reasoning_tokens"])
            for row in rows
            if isinstance(row.get("reasoning_tokens"), (int, float))
        ),
        "answer_tokens_total": sum(
            int(row["answer_tokens"])
            for row in rows
            if isinstance(row.get("answer_tokens"), (int, float))
        ),
    }
    for field in stage_fields:
        values = numeric_values(field, retrieval_only=field == "retrieval_latency_s")
        prefix = field.removesuffix("_s")
        summary[f"{prefix}_mean_s"] = statistics.fmean(values) if values else None
        summary[f"{prefix}_p50_s"] = percentile(values, 0.50) if values else None
        summary[f"{prefix}_p95_s"] = percentile(values, 0.95) if values else None

    judge_rows = [row for row in rows if "judge_status" in row.get("scores", {})]
    if judge_rows:
        judge_successes = sum(row["scores"].get("judge_status") == "ok" for row in judge_rows)
        judge_failures = len(judge_rows) - judge_successes
        cache_hits = sum(row["scores"].get("judge_cache_hit") is True for row in judge_rows)
        summary.update(
            {
                "judge_examples": len(judge_rows),
                "judge_successes": judge_successes,
                "judge_failures": judge_failures,
                "judge_coverage": judge_successes / len(rows) if rows else 0.0,
                "judge_failure_rate": judge_failures / len(judge_rows),
                "judge_cache_hits": cache_hits,
                "judge_cache_misses": len(judge_rows) - cache_hits,
                "judge_total_evaluation_examples": len(rows),
            }
        )
    else:
        summary.update(
            {
                "judge_examples": 0,
                "judge_successes": 0,
                "judge_failures": 0,
                "judge_coverage": None,
                "judge_failure_rate": None,
                "judge_cache_hits": 0,
                "judge_cache_misses": 0,
                "judge_total_evaluation_examples": len(rows),
            }
        )

    # Schema-v1 aliases. They are intentionally documented as deprecated.
    summary["latency_mean_s"] = summary["model_generate_latency_mean_s"]
    summary["latency_p50_s"] = summary["model_generate_latency_p50_s"]
    summary["latency_p95_s"] = summary["model_generate_latency_p95_s"]
    summary["end_to_end_latency_p50_s"] = summary["inference_e2e_latency_p50_s"]
    summary["end_to_end_latency_p95_s"] = summary["inference_e2e_latency_p95_s"]
    summary["tokens_per_second"] = summary["output_tokens_per_model_generate_second"]
    excluded = {
        "exact_match",
        "token_f1",
        "latency_s",
        "end_to_end_latency_s",
        "tokens_per_second",
        "prompt_tokens",
        "output_tokens",
        "retrieval_latency_s",
        "judge_latency_s",
        "judge_attempts",
        "judge_cache_hit",
    }
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.get("scores", {}).items()
            if key not in excluded
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    for metric in metric_names:
        values = [
            float(row["scores"][metric])
            for row in rows
            if isinstance(row.get("scores", {}).get(metric), (int, float))
            and not isinstance(row.get("scores", {}).get(metric), bool)
        ]
        summary[metric] = statistics.fmean(values) if values else None
    judge_metric_names = (
        "judge_correctness",
        "judge_groundedness",
        "judge_unsupported_claim_rate",
    )
    judge_metrics: dict[str, Any] = {}
    for metric in judge_metric_names:
        numeric_rows = [
            row
            for row in rows
            if isinstance(row.get("scores", {}).get(metric), (int, float))
            and not isinstance(row.get("scores", {}).get(metric), bool)
        ]
        coverage = len(numeric_rows) / len(rows) if rows and judge_rows else None
        judge_metrics[metric] = {
            "total_evaluation_examples": len(rows),
            "numeric_examples": len(numeric_rows),
            "judge_successes": summary["judge_successes"],
            "judge_failures": summary["judge_failures"],
            "coverage": coverage,
            "cache_hits": summary["judge_cache_hits"],
            "cache_misses": summary["judge_cache_misses"],
            "minimum_coverage": minimum_judge_metric_coverage,
            "status": (
                "not_applicable"
                if not judge_rows
                else (
                    "ok"
                    if coverage is not None and coverage >= minimum_judge_metric_coverage
                    else "insufficient_coverage"
                )
            ),
        }
    summary["judge_metrics"] = judge_metrics
    return summary


def paired_bootstrap_delta(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    samples: int = 10_000,
    seed: int = 42,
    minimum_coverage: float = 0.0,
    minimum_paired_examples: int = 1,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1")
    if minimum_paired_examples < 1:
        raise ValueError("minimum_paired_examples must be positive")

    def metric_value(row: Mapping[str, Any]) -> float | None:
        value = row.get(metric)
        if value is None and isinstance(row.get("scores"), Mapping):
            value = row["scores"].get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return float(value)

    baseline_ids = [str(row["id"]) for row in baseline_rows]
    candidate_ids = [str(row["id"]) for row in candidate_rows]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("Cannot pair rows: baseline contains duplicate example IDs")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Cannot pair rows: candidate contains duplicate example IDs")
    baseline = {str(row["id"]): metric_value(row) for row in baseline_rows}
    candidate = {str(row["id"]): metric_value(row) for row in candidate_rows}
    if baseline.keys() != candidate.keys():
        raise ValueError(f"Cannot pair metric {metric!r}: example IDs differ")
    baseline_numeric_ids = {row_id for row_id, value in baseline.items() if value is not None}
    candidate_numeric_ids = {row_id for row_id, value in candidate.items() if value is not None}
    paired_ids = sorted(baseline_numeric_ids & candidate_numeric_ids)
    dropped_ids = sorted(set(baseline) - set(paired_ids))
    total_examples = len(baseline)
    paired_coverage = len(paired_ids) / total_examples if total_examples else 0.0
    diagnostics: dict[str, Any] = {
        "total_examples": total_examples,
        "baseline_total_examples": len(baseline_rows),
        "candidate_total_examples": len(candidate_rows),
        "baseline_numeric_examples": len(baseline_numeric_ids),
        "candidate_numeric_examples": len(candidate_numeric_ids),
        "paired_examples": len(paired_ids),
        "paired_coverage": paired_coverage,
        "dropped_examples": len(dropped_ids),
        "dropped_example_ids": dropped_ids,
        "minimum_coverage": minimum_coverage,
        "minimum_paired_examples": minimum_paired_examples,
    }
    if not paired_ids:
        return {
            **diagnostics,
            "examples": 0,
            "delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_samples": samples,
            "method": "paired-percentile-bootstrap",
            "ci_excludes_zero": False,
            "statistically_significant": False,
            "decision_eligible": False,
            "status": "insufficient_coverage",
        }
    deltas = [
        candidate[row_id] - baseline[row_id]  # type: ignore[operator]
        for row_id in paired_ids
    ]

    # Chunked NumPy sampling keeps the default 10k bootstrap practical for
    # evaluation sets with thousands of examples without allocating an
    # unbounded samples-by-examples matrix.
    rng = np.random.default_rng(seed)
    delta_values = np.asarray(deltas, dtype=np.float64)
    chunk_size = max(1, min(samples, 1_000_000 // len(deltas)))
    means: list[float] = []
    for offset in range(0, samples, chunk_size):
        current = min(chunk_size, samples - offset)
        indices = rng.integers(0, len(deltas), size=(current, len(deltas)))
        means.extend(np.mean(delta_values[indices], axis=1).tolist())
    low = percentile(means, 0.025)
    high = percentile(means, 0.975)
    coverage_sufficient = (
        paired_coverage >= minimum_coverage
        and len(deltas) >= minimum_paired_examples
    )
    ci_excludes_zero = low > 0.0 or high < 0.0
    return {
        **diagnostics,
        "examples": len(deltas),
        "delta": statistics.fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_samples": samples,
        "method": "paired-percentile-bootstrap",
        "ci_excludes_zero": ci_excludes_zero,
        "statistically_significant": ci_excludes_zero and coverage_sufficient,
        "decision_eligible": coverage_sufficient,
        "status": "ok" if coverage_sufficient else "insufficient_coverage",
    }
