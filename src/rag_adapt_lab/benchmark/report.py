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


def _with_unit(value: object, unit: str, *, digits: int = 3) -> str:
    rendered = _number(value, digits=digits)
    return rendered if rendered == "—" else f"{rendered}{unit}"


def _decision(result: Mapping[str, Any]) -> str:
    status = result.get("status")
    if status == "confounded_training_controls":
        return "confounded: adaptation training controls differ"
    if status == "unverified_adapter_provenance":
        return "not decision-eligible because adapter provenance is unverified"
    if status == "insufficient_coverage" or result.get("decision_eligible") is False:
        return "not decision-eligible because coverage requirements were not met"
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
        "| Recipe | EM | Token F1 | Reference overlap | Judge correctness | Lexical groundedness | Judge groundedness | Judge coverage | p50 Inference | p95 Inference | Output tok/s | Allocated VRAM | Reserved VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    recipes = summary.get("recipes", {})
    for recipe, values in recipes.items():
        metrics = values.get("metrics", {})
        judge_metric = metrics.get("judge_metrics", {}).get("judge_correctness", {})
        numeric_judge = int(judge_metric.get("numeric_examples", 0))
        total_examples = int(judge_metric.get("total_evaluation_examples", 0))
        judge_coverage = (
            f"{numeric_judge}/{total_examples} ({numeric_judge / total_examples:.1%})"
            if metrics.get("judge_examples", 0) and total_examples
            else "—"
        )
        lines.append(
            "| {name} | {em} | {f1} | {overlap} | {judge_correctness} | "
            "{lexical_groundedness} | {judge_groundedness} | {judge_coverage} | "
            "{p50} | {p95} | {throughput} | {allocated} | {reserved} |".format(
                name=DISPLAY_NAMES.get(recipe, recipe),
                em=_number(metrics.get("exact_match")),
                f1=_number(metrics.get("token_f1")),
                overlap=_number(metrics.get("reference_overlap")),
                judge_correctness=_number(metrics.get("judge_correctness")),
                lexical_groundedness=_number(metrics.get("lexical_groundedness")),
                judge_groundedness=_number(metrics.get("judge_groundedness")),
                judge_coverage=judge_coverage,
                p50=_with_unit(metrics.get("inference_e2e_latency_p50_s"), "s"),
                p95=_with_unit(metrics.get("inference_e2e_latency_p95_s"), "s"),
                throughput=_number(
                    metrics.get("output_tokens_per_model_generate_second"), digits=1
                ),
                allocated=_with_unit(metrics.get("peak_allocated_vram_gb"), " GB", digits=2),
                reserved=_with_unit(metrics.get("peak_reserved_vram_gb"), " GB", digits=2),
            )
        )

    comparisons = summary.get("comparisons", {})
    lines.extend(
        [
            "",
            "`reference_overlap` is a deterministic reference-token overlap score and uses the "
            "same normalization as Token F1; it is not a semantic judge. Lexical and judge "
            "groundedness remain separate. Judge time is excluded from inference latency.",
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

    lines.extend(["", "## Judge reliability", ""])
    judge_rows = [
        values.get("metrics", {})
        for values in recipes.values()
        if values.get("metrics", {}).get("judge_examples", 0)
    ]
    if not judge_rows:
        lines.append("The optional LLM judge was disabled; canonical EM and Token F1 are unaffected.")
    else:
        total = sum(int(metrics.get("judge_total_evaluation_examples", 0)) for metrics in judge_rows)
        failures = sum(int(metrics.get("judge_failures", 0)) for metrics in judge_rows)
        successes = sum(int(metrics.get("judge_successes", 0)) for metrics in judge_rows)
        lines.append(
            f"Judge coverage: {successes}/{total} "
            f"({_number(successes / total if total else None)}); failures: {failures}."
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
            paired = int(result.get("paired_examples", result.get("examples", 0)))
            total = int(result.get("total_examples", result.get("examples", 0)))
            coverage = result.get("paired_coverage")
            status = result.get("status", "ok")
            lines.append(
                f"- `{metric}` delta: {_number(result.get('delta'))}; "
                f"95% bootstrap CI [{_number(result.get('ci95_low'))}, "
                f"{_number(result.get('ci95_high'))}]; significant: {significant}; "
                f"paired examples: {paired}/{total}; paired coverage: "
                f"{_with_unit(100 * coverage if isinstance(coverage, (int, float)) else None, '%', digits=1)}; "
                f"status: `{status}`."
            )
        lines.append("")

    config = summary.get("configuration", {})
    provenance = summary.get("provenance", {})
    lines.extend(
        [
            "## Reproducibility contract",
            "",
            f"- Model: `{config.get('model', {}).get('model_id', 'unknown')}` at "
            f"`{config.get('model', {}).get('revision', 'unknown')}`.",
            f"- Retriever: `{config.get('retriever', {}).get('kind', 'unknown')}` with "
            f"`top_k={config.get('retriever', {}).get('top_k', 'unknown')}`.",
            f"- Prompt version: `{config.get('prompt', {}).get('version', 'unknown')}`.",
            f"- Chat-template arguments: `{config.get('model', {}).get('chat_template_kwargs', {})}`.",
            f"- Model condition: `{config.get('model', {}).get('condition', 'unknown')}`.",
            f"- Adapter provenance verified: `{provenance.get('verified', True)}`; legacy override: "
            f"`{provenance.get('allow_unverified_adapter', False)}`.",
            f"- Adaptation training controls matched: "
            f"`{provenance.get('training_controls_matched')}`; confounded-control override: "
            f"`{provenance.get('allow_unmatched_training_controls', False)}`.",
            f"- Bootstrap samples: `{config.get('bootstrap_samples', 'unknown')}`.",
            "- LLM-judge scores are complementary; exact match and token F1 remain canonical "
            "deterministic metrics.",
            "",
        ]
    )
    for warning in provenance.get("warnings", []):
        if str(warning).startswith("Confounded comparison:"):
            lines.append(f"- **CONFOUNDED COMPARISON:** {warning}")
        else:
            lines.append(f"- **UNVERIFIED PROVENANCE:** {warning}")
    lines.append("")
    return "\n".join(lines)
