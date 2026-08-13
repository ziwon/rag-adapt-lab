#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_adapt_lab.config import (
    effective_chat_template_kwargs,
    effective_generation_config,
    load_yaml,
    validate_hf_model_config,
    validate_thinking_configuration,
)
from rag_adapt_lab.data.io import load_documents, load_eval, write_jsonl
from rag_adapt_lab.evaluation.generation import exact_match, normalize_text, token_f1
from rag_adapt_lab.evaluation.retrieval import evaluate_retriever
from rag_adapt_lab.evaluation.statistics import paired_bootstrap_delta
from rag_adapt_lab.generation.prompts import format_rag_user_prompt, rag_prompt_provenance
from rag_adapt_lab.generation.thinking import parse_thinking_tokens
from rag_adapt_lab.provenance import file_sha256, validate_adapter_provenance
from rag_adapt_lab.retrieval.bm25 import BM25Retriever
from rag_adapt_lab.schema_validation import validate_artifact_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "examples": len(rows),
        "exact_match": statistics.fmean(row["exact_match"] for row in rows),
        "token_f1": statistics.fmean(row["token_f1"] for row in rows),
        "answer_containment": statistics.fmean(row["answer_containment"] for row in rows),
        "model_generate_latency_s_per_example": statistics.fmean(
            row["model_generate_latency_s"] for row in rows
        ),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "reasoning_tokens": sum(row["reasoning_tokens"] for row in rows),
        "answer_tokens": sum(row["answer_tokens"] for row in rows),
    }


def paired_condition_delta(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    metric: str,
    samples: int = 20_000,
    seed: int = 42,
) -> dict[str, float | int | bool | str]:
    baseline = [row for row in rows if row["condition"] == condition and row["model"] == "base"]
    candidate = [row for row in rows if row["condition"] == condition and row["model"] == "tuned"]
    return paired_bootstrap_delta(
        baseline,
        candidate,
        metric=metric,
        samples=samples,
        seed=seed,
    )


def run_generation(
    *,
    model: Any,
    tokenizer: Any,
    prepared: list[dict[str, Any]],
    condition: str,
    label: str,
    batch_size: int,
    generation_config: dict[str, Any],
    chat_template_kwargs: dict[str, Any],
    thinking_enabled: bool,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row[f"{condition}_user_prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            for row in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        batch_index = start // batch_size
        torch.manual_seed(seed + batch_index)
        torch.cuda.manual_seed_all(seed + batch_index)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **generation_config,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        per_example_latency = elapsed / len(batch)
        for row, generated_ids in zip(batch, generated.tolist(), strict=True):
            parsed = parse_thinking_tokens(
                generated_ids,
                tokenizer=tokenizer,
                thinking_enabled=thinking_enabled,
            )
            prediction = parsed.answer
            references = row["references"]
            normalized_prediction = normalize_text(prediction)
            results.append(
                {
                    "id": row["id"],
                    "model": label,
                    "condition": condition,
                    "question": row["question"],
                    "reference": row["reference"],
                    "references": references,
                    "prediction": prediction,
                    "raw_prediction": parsed.raw_text,
                    "reasoning": parsed.reasoning,
                    "output_tokens": parsed.output_tokens,
                    "reasoning_tokens": parsed.reasoning_tokens,
                    "answer_tokens": parsed.answer_tokens,
                    "thinking_boundary_token_id": parsed.boundary_token_id,
                    "thinking_boundary_found": parsed.boundary_found,
                    "thinking_protocol_violation": parsed.thinking_protocol_violation,
                    "exact_match": max(exact_match(prediction, ref) for ref in references),
                    "token_f1": max(token_f1(prediction, ref) for ref in references),
                    "answer_containment": float(
                        any(
                            normalize_text(ref) in normalized_prediction
                            for ref in references
                            if normalize_text(ref)
                        )
                    ),
                    "model_generate_latency_s": per_example_latency,
                    "latency_s": per_example_latency,
                    "retrieved_doc_ids": row["retrieved_doc_ids"],
                    "retrieval_hit": row["retrieval_hit"],
                }
            )
    return results


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")

    model_config = load_yaml(args.model_config)
    model_id, model_revision = validate_hf_model_config(model_config)
    chat_template_kwargs = effective_chat_template_kwargs(model_config)
    thinking_enabled = validate_thinking_configuration(model_config)
    generation_config = effective_generation_config(
        model_config,
        max_new_tokens=args.max_new_tokens,
    )
    validate_adapter_provenance(
        args.adapter,
        model_id=model_id,
        model_revision=model_revision,
        expected_prompt={
            **rag_prompt_provenance(),
            "chat_template_kwargs": chat_template_kwargs,
        },
        held_out_evaluation_sha256=file_sha256(args.eval_set),
    )
    documents = load_documents(args.documents)
    examples = load_eval(args.eval_set)
    documents_by_id = {document.id: document for document in documents}

    retriever = BM25Retriever()
    retriever.index(documents)
    retrieval_metrics = evaluate_retriever(retriever, examples, top_k=args.top_k)
    prepared: list[dict[str, Any]] = []
    for example in examples:
        retrieved = retriever.search(example.question, top_k=args.top_k)
        retrieved_ids = [result.document.id for result in retrieved]
        retrieved_contexts = [result.document.text for result in retrieved]
        oracle_contexts = [documents_by_id[doc_id].text for doc_id in example.relevant_doc_ids]
        references = example.metadata.get("reference_answers") or [example.reference_answer or ""]
        prepared.append(
            {
                "id": example.id,
                "question": example.question,
                "reference": example.reference_answer or "",
                "references": references,
                "retrieved_doc_ids": retrieved_ids,
                "retrieval_hit": bool(set(retrieved_ids) & set(example.relevant_doc_ids)),
                "rag_user_prompt": format_rag_user_prompt(
                    question=example.question,
                    contexts=retrieved_contexts,
                ),
                "oracle_user_prompt": format_rag_user_prompt(
                    question=example.question,
                    contexts=oracle_contexts,
                ),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        device_map="auto",
    )
    model.eval()
    model_generation_config = model.generation_config
    if model_generation_config is not None and not bool(
        generation_config.get("do_sample", False)
    ):
        model_generation_config.do_sample = False
        model_generation_config.temperature = None
        model_generation_config.top_p = None
        model_generation_config.top_k = None
    torch.cuda.reset_peak_memory_stats()

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"retrieval": retrieval_metrics.as_dict()}
    for condition in ("rag", "oracle"):
        rows = run_generation(
            model=model,
            tokenizer=tokenizer,
            prepared=prepared,
            condition=condition,
            label="base",
            batch_size=args.batch_size,
            generation_config=generation_config,
            chat_template_kwargs=chat_template_kwargs,
            thinking_enabled=thinking_enabled,
            seed=args.seed,
        )
        all_rows.extend(rows)
        summary[f"base_{condition}"] = aggregate(rows)

    tuned_model = PeftModel.from_pretrained(model, args.adapter)
    tuned_model.eval()
    for condition in ("rag", "oracle"):
        rows = run_generation(
            model=tuned_model,
            tokenizer=tokenizer,
            prepared=prepared,
            condition=condition,
            label="tuned",
            batch_size=args.batch_size,
            generation_config=generation_config,
            chat_template_kwargs=chat_template_kwargs,
            thinking_enabled=thinking_enabled,
            seed=args.seed,
        )
        all_rows.extend(rows)
        summary[f"tuned_{condition}"] = aggregate(rows)

    summary["delta_rag"] = {
        metric: summary["tuned_rag"][metric] - summary["base_rag"][metric]
        for metric in ("exact_match", "token_f1", "answer_containment")
    }
    summary["delta_oracle"] = {
        metric: summary["tuned_oracle"][metric] - summary["base_oracle"][metric]
        for metric in ("exact_match", "token_f1", "answer_containment")
    }
    summary["paired_bootstrap"] = {
        condition: {
            metric: paired_condition_delta(
                all_rows,
                condition=condition,
                metric=metric,
                seed=42 + offset,
            )
            for offset, metric in enumerate(("exact_match", "token_f1"))
        }
        for condition in ("rag", "oracle")
    }
    summary["schema_name"] = "squad-paired-evaluation"
    summary["schema_version"] = 1
    summary["peak_allocated_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    summary["peak_reserved_vram_gb"] = torch.cuda.max_memory_reserved() / 1024**3
    summary["model_id"] = model_config["model_id"]
    summary["model_revision"] = model_config["revision"]
    summary["chat_template_kwargs"] = chat_template_kwargs
    summary["thinking_enabled"] = thinking_enabled
    summary["seed"] = args.seed
    validate_artifact_schema(summary, "squad-paired-evaluation-v1.schema.json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
