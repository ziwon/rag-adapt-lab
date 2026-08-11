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


def aggregate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    latencies = [
        float(row["latency_s"]) for row in rows if isinstance(row.get("latency_s"), (int, float))
    ]
    end_to_end = [
        float(row["end_to_end_latency_s"])
        for row in rows
        if isinstance(row.get("end_to_end_latency_s"), (int, float))
    ]
    retrieval_latencies = [
        float(row["retrieval_latency_s"])
        for row in rows
        if row.get("retrieval_used") is True
        and isinstance(row.get("retrieval_latency_s"), (int, float))
    ]
    token_timings = [
        (float(row["output_tokens"]), float(row["latency_s"]))
        for row in rows
        if isinstance(row.get("output_tokens"), (int, float))
        and isinstance(row.get("latency_s"), (int, float))
        and float(row["latency_s"]) > 0
    ]
    summary: dict[str, float | int | None] = {
        "examples": len(rows),
        "exact_match": mean_numeric(rows, "exact_match"),
        "token_f1": mean_numeric(rows, "token_f1"),
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.50) if latencies else None,
        "latency_p95_s": percentile(latencies, 0.95) if latencies else None,
        "retrieval_latency_mean_s": (
            statistics.fmean(retrieval_latencies) if retrieval_latencies else None
        ),
        "retrieval_latency_p50_s": (
            percentile(retrieval_latencies, 0.50) if retrieval_latencies else None
        ),
        "retrieval_latency_p95_s": (
            percentile(retrieval_latencies, 0.95) if retrieval_latencies else None
        ),
        "end_to_end_latency_p50_s": percentile(end_to_end, 0.50) if end_to_end else None,
        "end_to_end_latency_p95_s": percentile(end_to_end, 0.95) if end_to_end else None,
        "tokens_per_second": (
            sum(tokens for tokens, _ in token_timings) / sum(timing for _, timing in token_timings)
            if token_timings
            else None
        ),
        "prompt_tokens_total": sum(
            int(row["prompt_tokens"])
            for row in rows
            if isinstance(row.get("prompt_tokens"), (int, float))
        ),
        "output_tokens_total": sum(int(tokens) for tokens, _ in token_timings),
    }
    excluded = {
        "exact_match",
        "token_f1",
        "latency_s",
        "end_to_end_latency_s",
        "tokens_per_second",
        "prompt_tokens",
        "output_tokens",
        "retrieval_latency_s",
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
    return summary


def paired_bootstrap_delta(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int | bool | str]:
    if samples < 1:
        raise ValueError("samples must be positive")

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
    deltas = [
        candidate[row_id] - baseline[row_id]  # type: ignore[operator]
        for row_id in sorted(baseline)
        if baseline[row_id] is not None and candidate[row_id] is not None
    ]
    if not deltas:
        raise ValueError(f"Cannot pair metric {metric!r}: no numeric paired values")

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
    return {
        "examples": len(deltas),
        "delta": statistics.fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_samples": samples,
        "method": "paired-percentile-bootstrap",
        "statistically_significant": low > 0.0 or high < 0.0,
    }
