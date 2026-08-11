from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DISPLAY_NAMES = {
    "base": "Base",
    "rag": "RAG",
    "sft-rag": "SFT + RAG",
    "raft-rag": "RAFT + RAG",
}


def _number(value: object, *, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{float(value):.{digits}f}"


def _preferred_metric(metrics: Mapping[str, Any], judge: str, fallback: str) -> object:
    return metrics.get(judge) if metrics.get(judge) is not None else metrics.get(fallback)


def _with_unit(value: object, unit: str, *, digits: int = 3) -> str:
    rendered = _number(value, digits=digits)
    return rendered if rendered == "—" else f"{rendered}{unit}"


def _decision(result: Mapping[str, Any]) -> str:
    delta = result.get("delta")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        return "could not be estimated"
    if not result.get("statistically_significant"):
        return "not statistically resolved (the 95% CI includes zero)"
    if delta > 0:
        return "a statistically supported improvement"
    if delta < 0:
        return "a statistically supported regression"
    return "no measured change"


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    """Render a stable, concise benchmark comparison from summary.json data."""
    lines = [
        "# RAG adaptation benchmark report",
        "",
        "| Recipe | EM | Token F1 | Correctness | Groundedness | p50 Latency | p95 Latency | Tokens/s | Peak VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    recipes = summary.get("recipes", {})
    for recipe, values in recipes.items():
        metrics = values.get("metrics", {})
        correctness = _preferred_metric(metrics, "judge_correctness", "answer_correctness")
        groundedness = _preferred_metric(metrics, "judge_groundedness", "groundedness")
        lines.append(
            "| {name} | {em} | {f1} | {correctness} | {groundedness} | {p50} | "
            "{p95} | {throughput} | {vram} |".format(
                name=DISPLAY_NAMES.get(recipe, recipe),
                em=_number(metrics.get("exact_match")),
                f1=_number(metrics.get("token_f1")),
                correctness=_number(correctness),
                groundedness=_number(groundedness),
                p50=_with_unit(metrics.get("end_to_end_latency_p50_s"), "s"),
                p95=_with_unit(metrics.get("end_to_end_latency_p95_s"), "s"),
                throughput=_number(metrics.get("tokens_per_second"), digits=1),
                vram=_with_unit(metrics.get("peak_gpu_vram_gb"), " GB", digits=2),
            )
        )

    comparisons = summary.get("comparisons", {})
    lines.extend(
        [
            "",
            "Correctness uses the configured judge when enabled and otherwise reports the "
            "deterministic reference token-overlap score. Groundedness is lexical unless a judge "
            "metric is configured.",
            "",
            "## Decision summary",
            "",
        ]
    )
    decision_labels = {
        "base->rag": "Retrieval over Base",
        "rag->sft-rag": "SFT after RAG",
        "rag->raft-rag": "RAFT after RAG",
        "sft-rag->raft-rag": "RAFT over ordinary SFT",
    }
    decisions = 0
    for comparison, label in decision_labels.items():
        token_f1_result = comparisons.get(comparison, {}).get("token_f1")
        if isinstance(token_f1_result, Mapping):
            lines.append(
                f"- **{label}:** {_decision(token_f1_result)}; Token F1 delta "
                f"{_number(token_f1_result.get('delta'))}, 95% CI "
                f"[{_number(token_f1_result.get('ci95_low'))}, "
                f"{_number(token_f1_result.get('ci95_high'))}]."
            )
            decisions += 1
    if decisions == 0:
        lines.append("No paired Token F1 comparisons were available.")

    retrieval = summary.get("retrieval_metrics", {})
    lines.extend(["", "## Shared retrieval quality", ""])
    lines.append(
        "Recall@K {recall}; hit rate@K {hit}; MRR {mrr}; nDCG@K {ndcg} "
        "across {evaluated} labeled examples.".format(
            recall=_number(retrieval.get("retrieval/recall_at_k")),
            hit=_number(retrieval.get("retrieval/hit_rate_at_k")),
            mrr=_number(retrieval.get("retrieval/mrr")),
            ndcg=_number(retrieval.get("retrieval/ndcg_at_k")),
            evaluated=retrieval.get("retrieval/evaluated", 0),
        )
    )

    lines.extend(["", "## Paired comparisons", ""])
    if not comparisons:
        lines.append("No requested recipe pairs were available.")
    for comparison, values in comparisons.items():
        baseline, candidate = comparison.split("->", maxsplit=1)
        lines.append(
            f"### {DISPLAY_NAMES.get(baseline, baseline)} → {DISPLAY_NAMES.get(candidate, candidate)}"
        )
        lines.append("")
        for metric, result in values.items():
            significant = "yes" if result.get("statistically_significant") else "no"
            lines.append(
                f"- `{metric}` delta: {_number(result.get('delta'))}; "
                f"95% bootstrap CI [{_number(result.get('ci95_low'))}, "
                f"{_number(result.get('ci95_high'))}]; significant: {significant}."
            )
        lines.append("")

    config = summary.get("configuration", {})
    lines.extend(
        [
            "## Reproducibility contract",
            "",
            f"- Model: `{config.get('model', {}).get('model_id', 'unknown')}` at "
            f"`{config.get('model', {}).get('revision', 'unknown')}`.",
            f"- Retriever: `{config.get('retriever', {}).get('kind', 'unknown')}` with "
            f"`top_k={config.get('retriever', {}).get('top_k', 'unknown')}`.",
            f"- Prompt version: `{config.get('prompt', {}).get('version', 'unknown')}`.",
            f"- Bootstrap samples: `{config.get('bootstrap_samples', 'unknown')}`.",
            "- LLM-judge scores are complementary; exact match and token F1 remain canonical "
            "deterministic metrics.",
            "",
        ]
    )
    return "\n".join(lines)
