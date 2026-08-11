from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag_adapt_lab.data.io import load_documents, load_eval, write_jsonl
from rag_adapt_lab.data.raft import build_raft_examples
from rag_adapt_lab.evaluation.retrieval import evaluate_retriever
from rag_adapt_lab.recipes.plan import build_plan
from rag_adapt_lab.retrieval.bm25 import BM25Retriever

app = typer.Typer(help="Domain-neutral RAG adaptation research harness.", no_args_is_help=True)
console = Console()


@app.command("validate-data")
def validate_data(
    documents: Path = typer.Option(..., exists=True, readable=True),
    eval_set: Path = typer.Option(..., "--eval-set", exists=True, readable=True),
) -> None:
    docs = load_documents(documents)
    eval_rows = load_eval(eval_set)
    doc_ids = {doc.id for doc in docs}
    missing = sorted({doc_id for row in eval_rows for doc_id in row.relevant_doc_ids if doc_id not in doc_ids})
    if missing:
        raise typer.BadParameter(f"Eval set references missing document IDs: {missing[:10]}")
    console.print(f"[green]OK[/green] {len(docs)} documents, {len(eval_rows)} eval examples")


@app.command("prepare-raft")
def prepare_raft(
    documents: Path = typer.Option(..., exists=True, readable=True),
    eval_set: Path = typer.Option(..., "--eval-set", exists=True, readable=True),
    output: Path = typer.Option(...),
    distractors: int = typer.Option(2, min=0, max=20),
    seed: int = typer.Option(42),
) -> None:
    rows = build_raft_examples(
        load_documents(documents),
        load_eval(eval_set),
        distractors=distractors,
        seed=seed,
    )
    write_jsonl(output, rows)
    console.print(f"[green]Wrote[/green] {len(rows)} RAFT examples to {output}")


@app.command("eval-retrieval")
def eval_retrieval(
    documents: Path = typer.Option(..., exists=True, readable=True),
    eval_set: Path = typer.Option(..., "--eval-set", exists=True, readable=True),
    retriever: str = typer.Option("bm25"),
    top_k: int = typer.Option(5, min=1, max=100),
) -> None:
    docs = load_documents(documents)
    examples = load_eval(eval_set)
    if retriever != "bm25":
        raise typer.BadParameter("The CLI v0.1 wires BM25 directly. Dense/hybrid are extension backends.")
    backend = BM25Retriever()
    backend.index(docs)
    metrics = evaluate_retriever(backend, examples, top_k=top_k)
    table = Table(title=f"Retrieval evaluation (top_k={top_k})")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.as_dict().items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        table.add_row(key, rendered)
    console.print(table)


@app.command("train")
def train(
    config: Path = typer.Option(..., exists=True, readable=True),
    train_file: Path = typer.Option(..., "--train-file", exists=True, readable=True),
) -> None:
    from rag_adapt_lab.training.qlora import train_qlora

    output = train_qlora(recipe_config=config, train_file=train_file)
    console.print(f"[green]Training complete.[/green] Adapter: {output}")


@app.command("benchmark")
def benchmark(
    recipes: str = typer.Option("base,rag,sft-rag,raft-rag"),
    model_config: Path = typer.Option(..., "--model-config", exists=True, readable=True),
    documents: Path = typer.Option(..., exists=True, readable=True),
    eval_set: Path = typer.Option(..., "--eval-set", exists=True, readable=True),
    output: Path | None = typer.Option(None),
) -> None:
    requested = [item.strip() for item in recipes.split(",") if item.strip()]
    allowed = {"base", "rag", "sft-rag", "raft-rag"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise typer.BadParameter(f"Unknown recipes: {unknown}")
    # Validate the shared contract before producing the plan.
    load_documents(documents)
    load_eval(eval_set)
    jobs = build_plan(
        recipes=requested,
        model_config=model_config,
        documents=documents,
        eval_set=eval_set,
    )
    payload = {"version": 1, "jobs": [job.as_dict() for job in jobs]}
    rendered = json.dumps(payload, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        console.print(f"[green]Benchmark plan written:[/green] {output}")
    else:
        console.print(rendered)
    console.print(
        "[yellow]Note:[/yellow] v0.1 benchmark creates a validated execution plan. "
        "Attach your generation runner and judge plugin to execute the complete matrix."
    )


if __name__ == "__main__":
    app()
