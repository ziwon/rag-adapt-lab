from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag_adapt_lab.config import load_yaml, validate_hf_model_config
from rag_adapt_lab.data.io import load_documents, load_eval, load_qa_examples, write_jsonl
from rag_adapt_lab.data.raft import build_raft_examples
from rag_adapt_lab.data.validation import ensure_disjoint_qa_splits
from rag_adapt_lab.evaluation.retrieval import evaluate_retriever
from rag_adapt_lab.evaluation.scorers import build_scorer
from rag_adapt_lab.generation.prompts import RAG_PROMPT_NAME, RAG_PROMPT_VERSION
from rag_adapt_lab.recipes.plan import build_plan
from rag_adapt_lab.retrieval.bm25 import BM25Retriever
from rag_adapt_lab.retrieval.factory import create_retriever
from rag_adapt_lab.tracking.base import Tracker

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
    missing = sorted(
        {doc_id for row in eval_rows for doc_id in row.relevant_doc_ids if doc_id not in doc_ids}
    )
    if missing:
        raise typer.BadParameter(f"Eval set references missing document IDs: {missing[:10]}")
    console.print(f"[green]OK[/green] {len(docs)} documents, {len(eval_rows)} eval examples")


@app.command("prepare-raft")
def prepare_raft(
    documents: Path = typer.Option(..., exists=True, readable=True),
    training_set: Path = typer.Option(
        ...,
        "--training-set",
        exists=True,
        readable=True,
        help="Labeled QA examples reserved for training.",
    ),
    held_out_eval: Path | None = typer.Option(
        None,
        "--held-out-eval",
        exists=True,
        readable=True,
        help="Optional held-out split checked for overlapping IDs and questions.",
    ),
    output: Path = typer.Option(...),
    distractors: int = typer.Option(2, min=0, max=20),
    seed: int = typer.Option(42),
    negative_strategy: str = typer.Option(
        "random",
        "--negative-strategy",
        help="Distractor strategy: random or bm25-hard-negative.",
    ),
    candidate_pool_size: int = typer.Option(20, min=1),
) -> None:
    training_examples = load_qa_examples(training_set)
    if held_out_eval is not None:
        try:
            ensure_disjoint_qa_splits(training_examples, load_eval(held_out_eval))
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--training-set") from exc
    if negative_strategy not in {"random", "bm25-hard-negative"}:
        raise typer.BadParameter(
            "Expected random or bm25-hard-negative", param_hint="--negative-strategy"
        )
    rows = build_raft_examples(
        load_documents(documents),
        training_examples,
        distractors=distractors,
        seed=seed,
        negative_strategy=negative_strategy,  # type: ignore[arg-type]
        candidate_pool_size=candidate_pool_size,
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
        raise typer.BadParameter(
            "The CLI v0.1 wires BM25 directly. Dense/hybrid are extension backends."
        )
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
    validation_file: Path | None = typer.Option(
        None, "--validation-file", exists=True, readable=True
    ),
    held_out_eval: Path | None = typer.Option(
        None,
        "--held-out-eval",
        exists=True,
        readable=True,
        help="Benchmark evaluation set used only for leakage checks.",
    ),
) -> None:
    from rag_adapt_lab.training.qlora import train_qlora

    output = train_qlora(
        recipe_config=config,
        train_file=train_file,
        validation_file=validation_file,
        held_out_eval_file=held_out_eval,
    )
    console.print(f"[green]Training complete.[/green] Adapter: {output}")


@app.command("benchmark")
def benchmark(
    recipes: str = typer.Option("base,rag,sft-rag,raft-rag"),
    model_config: Path = typer.Option(..., "--model-config", exists=True, readable=True),
    documents: Path = typer.Option(..., exists=True, readable=True),
    eval_set: Path = typer.Option(..., "--eval-set", exists=True, readable=True),
    retriever_config: Path | None = typer.Option(
        None, "--retriever-config", exists=True, readable=True
    ),
    scorer_config: Path | None = typer.Option(None, "--scorer-config", exists=True, readable=True),
    sft_adapter: Path | None = typer.Option(None, "--sft-adapter"),
    raft_adapter: Path | None = typer.Option(None, "--raft-adapter"),
    output_dir: Path = typer.Option(Path("outputs/benchmark"), "--output-dir"),
    plan_output: Path | None = typer.Option(None, "--plan-output", "--output"),
    top_k: int | None = typer.Option(None, min=1, max=100),
    bootstrap_samples: int = typer.Option(10_000, min=1),
    seed: int = typer.Option(42),
    warmup_examples: int = typer.Option(1, min=0),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit"),
    tracking_backend: str = typer.Option("none", "--tracking-backend"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    requested = [item.strip() for item in recipes.split(",") if item.strip()]
    allowed = {"base", "rag", "sft-rag", "raft-rag"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise typer.BadParameter(f"Unknown recipes: {unknown}")
    model_values = load_yaml(model_config)
    validate_hf_model_config(model_values)
    loaded_documents = load_documents(documents)
    loaded_examples = load_eval(eval_set)
    if not loaded_documents:
        raise typer.BadParameter("Corpus is empty", param_hint="--documents")
    if not loaded_examples:
        raise typer.BadParameter("Evaluation set is empty", param_hint="--eval-set")
    document_ids = {document.id for document in loaded_documents}
    missing = sorted(
        {
            doc_id
            for example in loaded_examples
            for doc_id in example.relevant_doc_ids
            if doc_id not in document_ids
        }
    )
    if missing:
        raise typer.BadParameter(
            f"Evaluation set references missing documents: {missing[:10]}",
            param_hint="--eval-set",
        )
    retriever_values: dict[str, object] = (
        load_yaml(retriever_config)
        if retriever_config is not None
        else {"kind": "bm25", "top_k": 5}
    )
    scorer_values = load_yaml(scorer_config) if scorer_config is not None else None
    configured_top_k = retriever_values.get("top_k", 5)
    if not isinstance(configured_top_k, (int, str)):
        raise typer.BadParameter("Retriever top_k must be an integer", param_hint="--top-k")
    resolved_top_k = top_k or int(configured_top_k)
    try:
        create_retriever(retriever_values)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--retriever-config") from exc
    adapters = {"sft-rag": sft_adapter, "raft-rag": raft_adapter}
    try:
        jobs = build_plan(
            recipes=requested,
            model_config=model_config,
            documents=documents,
            eval_set=eval_set,
            adapters=adapters,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--recipes") from exc
    payload = {
        "version": 2,
        "fixed_contract": {
            "model_config": str(model_config),
            "documents": str(documents),
            "eval_set": str(eval_set),
            "retriever": retriever_values,
            "scorer": scorer_values or {"mode": "default"},
            "top_k": resolved_top_k,
            "seed": seed,
            "generation": model_values.get("generation", {}),
            "prompt": {"name": RAG_PROMPT_NAME, "version": RAG_PROMPT_VERSION},
            "bootstrap_samples": bootstrap_samples,
            "warmup_examples": warmup_examples,
            "load_in_4bit": load_in_4bit,
        },
        "jobs": [job.as_dict() for job in jobs],
    }
    rendered = json.dumps(payload, indent=2)
    if plan_output:
        plan_output.parent.mkdir(parents=True, exist_ok=True)
        plan_output.write_text(rendered + "\n", encoding="utf-8")
        console.print(f"[green]Benchmark plan written:[/green] {plan_output}")
    if dry_run:
        console.print(rendered)
        return

    for recipe, adapter in adapters.items():
        if recipe in requested and adapter is None:
            raise typer.BadParameter(
                f"{recipe} requires an adapter path", param_hint=f"--{recipe.split('-')[0]}-adapter"
            )
        if recipe in requested and adapter is not None and not adapter.exists():
            raise typer.BadParameter(
                f"Adapter path does not exist: {adapter}",
                param_hint=f"--{recipe.split('-')[0]}-adapter",
            )

    from rag_adapt_lab.benchmark.runner import BenchmarkRunner, TransformersGeneratorFactory
    from rag_adapt_lab.tracking.null import NullTracker
    from rag_adapt_lab.tracking.wandb import WandbTracker

    tracker: Tracker
    if tracking_backend == "none":
        tracker = NullTracker()
    elif tracking_backend == "wandb":
        tracker = WandbTracker()
    else:
        raise typer.BadParameter("Expected none or wandb", param_hint="--tracking-backend")

    runner = BenchmarkRunner(
        jobs=jobs,
        model_config=model_values,
        documents=loaded_documents,
        examples=loaded_examples,
        retriever=create_retriever(retriever_values),
        retriever_config=retriever_values,
        generator_factory=TransformersGeneratorFactory(
            model_config=model_values,
            load_in_4bit=load_in_4bit,
            seed=seed,
        ),
        scorer=build_scorer(scorer_values),
        output_dir=output_dir,
        top_k=resolved_top_k,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        warmup_examples=warmup_examples,
        tracker=tracker,
        generator_config={
            "backend": "transformers",
            "load_in_4bit": load_in_4bit,
            "paired_seed_schedule": True,
            "warmup_examples": warmup_examples,
        },
        model_config_path=model_config,
        documents_path=documents,
        eval_path=eval_set,
    )
    runner.run()
    console.print(f"[green]Benchmark complete:[/green] {output_dir / 'summary.json'}")
    console.print(f"[green]Report:[/green] {output_dir / 'report.md'}")


if __name__ == "__main__":
    app()
