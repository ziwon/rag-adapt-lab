from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from rag_adapt_lab.config import (
    effective_chat_template_kwargs,
    load_yaml,
    resolve_relative,
    validate_hf_model_config,
)
from rag_adapt_lab.data.io import load_eval, read_jsonl
from rag_adapt_lab.data.splitting import (
    count_groups,
    enforce_partition_policy,
    source_partition_fingerprint,
)
from rag_adapt_lab.generation.prompts import rag_prompt_provenance
from rag_adapt_lab.provenance import (
    ADAPTER_MANIFEST_FILENAME,
    ADAPTER_MANIFEST_SCHEMA_VERSION,
    TRAINING_MANIFEST_SCHEMA_VERSION,
    artifact_sha256,
    canonical_sha256,
    file_sha256,
)
from rag_adapt_lab.schema_validation import validate_artifact_schema

from .controls import normalize_training_controls, training_control_sha256
from .data import (
    TrainingSplit,
    configured_training_split,
    ensure_disjoint_training_rows,
    prompt_completion_records,
    render_chat_prompt_completions,
    split_fingerprint,
)


def _file_record(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def build_sft_config_values(
    training_cfg: dict[str, Any],
    *,
    output_dir: Path,
    report_to_wandb: bool,
    has_validation: bool,
) -> dict[str, Any]:
    eval_strategy = str(training_cfg.get("eval_strategy", "steps"))
    eval_steps = int(training_cfg.get("eval_steps", 50))
    save_steps = int(training_cfg.get("save_steps", eval_steps))
    if has_validation and eval_strategy not in {"steps", "epoch"}:
        raise ValueError("eval_strategy must be 'steps' or 'epoch' when validation is enabled")
    if eval_steps < 1:
        raise ValueError("eval_steps must be positive")
    if save_steps < 1:
        raise ValueError("save_steps must be positive")
    if has_validation and eval_strategy == "steps" and save_steps % eval_steps != 0:
        raise ValueError(
            "save_steps must be a multiple of eval_steps when selecting the best model"
        )

    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-4)),
        "optim": str(training_cfg.get("optim", "adamw_torch")),
        "optim_args": str(training_cfg.get("optim_args", "")),
        "adam_beta1": float(training_cfg.get("adam_beta1", 0.9)),
        "adam_beta2": float(training_cfg.get("adam_beta2", 0.999)),
        "adam_epsilon": float(training_cfg.get("adam_epsilon", 1e-8)),
        "lr_scheduler_type": str(training_cfg.get("lr_scheduler_type", "linear")),
        "lr_scheduler_kwargs": dict(training_cfg.get("lr_scheduler_kwargs", {})),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(training_cfg.get("max_grad_norm", 1.0)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 2)),
        "max_steps": int(training_cfg.get("max_steps", -1)),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 16)),
        "max_length": int(training_cfg.get("max_seq_length", 2048)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 0)),
        "logging_steps": int(training_cfg.get("logging_steps", 5)),
        "save_strategy": eval_strategy if has_validation else "steps",
        "save_steps": save_steps,
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": dict(
            training_cfg.get("gradient_checkpointing_kwargs", {})
        ),
        "bf16": bool(training_cfg.get("bf16", True)),
        "fp16": bool(training_cfg.get("fp16", False)),
        "tf32": bool(training_cfg.get("tf32", False)),
        "seed": int(training_cfg.get("seed", 42)),
        "data_seed": int(training_cfg.get("data_seed", training_cfg.get("seed", 42))),
        "label_smoothing_factor": float(training_cfg.get("label_smoothing_factor", 0.0)),
        "report_to": ["wandb"] if report_to_wandb else [],
        "completion_only_loss": True,
        "include_tokens_per_second": True,
    }
    if has_validation:
        metric = str(training_cfg.get("metric_for_best_model", "eval_loss"))
        values.update(
            {
                "eval_strategy": eval_strategy,
                "eval_steps": eval_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": metric,
                "greater_is_better": bool(
                    training_cfg.get("greater_is_better", not metric.endswith("loss"))
                ),
            }
        )
    else:
        values["eval_strategy"] = "no"
    return values


def require_verifiable_training_prompt(training_cfg: Mapping[str, Any]) -> None:
    if training_cfg.get("use_chat_template") is not True:
        raise ValueError(
            "Verifiable adapter training requires use_chat_template=true so the training "
            "prompt exactly matches benchmark inference"
        )


def _load_training_split(
    *,
    train_file: str | Path,
    validation_file: str | Path | None,
    held_out_eval_file: str | Path | None,
    validation_ratio: float,
    seed: int,
    split_config: Mapping[str, Any] | None = None,
) -> TrainingSplit:
    train_path = Path(train_file).resolve()
    rows = read_jsonl(train_path)
    if not rows:
        raise ValueError(f"Training file is empty: {train_file}")

    configured_split = dict(split_config or {})
    resolved_validation_ratio = float(
        configured_split.get("validation_ratio", validation_ratio)
    )
    resolved_seed = int(configured_split.get("seed", seed))
    if not 0.0 <= resolved_validation_ratio < 1.0:
        raise ValueError("split.validation_ratio must be in [0, 1)")
    strategy = str(configured_split.get("strategy", "row"))
    if strategy not in {"row", "grouped"}:
        raise ValueError("split.strategy must be 'row' or 'grouped'")
    group_by_value = configured_split.get("group_by", [])
    if not isinstance(group_by_value, list) or not all(
        isinstance(value, str) for value in group_by_value
    ):
        raise ValueError("split.group_by must be a list of field names")
    group_by = tuple(group_by_value)
    recorded_group_by = (
        tuple(dict.fromkeys(["normalized_question", *group_by]))
        if strategy == "grouped"
        else ()
    )
    corpus_policy = str(configured_split.get("corpus_policy", "shared-corpus"))
    if corpus_policy not in {"shared-corpus", "document-disjoint"}:
        raise ValueError(
            "split.corpus_policy must be 'shared-corpus' or 'document-disjoint'"
        )

    if validation_file is not None:
        validation_path = Path(validation_file).resolve()
        if validation_path == train_path:
            raise ValueError("Training and validation files must be different")
        validation_rows = read_jsonl(validation_path)
        if not validation_rows:
            raise ValueError(f"Validation file is empty: {validation_file}")
        ensure_disjoint_training_rows(
            rows,
            validation_rows,
            left_name="training data",
            right_name="validation data",
        )
        audit = enforce_partition_policy(
            rows,
            validation_rows,
            corpus_policy=corpus_policy,  # type: ignore[arg-type]
            strategy=strategy,  # type: ignore[arg-type]
            group_by=group_by,
        )
        split = TrainingSplit(
            rows,
            validation_rows,
            "explicit-files",
            resolved_seed,
            len(validation_rows) / (len(rows) + len(validation_rows)),
            strategy,  # type: ignore[arg-type]
            recorded_group_by,
            corpus_policy,  # type: ignore[arg-type]
            count_groups(
                rows,
                strategy=strategy,  # type: ignore[arg-type]
                group_by=group_by,
                corpus_policy=corpus_policy,  # type: ignore[arg-type]
            ),
            count_groups(
                validation_rows,
                strategy=strategy,  # type: ignore[arg-type]
                group_by=group_by,
                corpus_policy=corpus_policy,  # type: ignore[arg-type]
            ),
            audit,
        )
    else:
        split = configured_training_split(
            rows,
            validation_ratio=resolved_validation_ratio,
            seed=resolved_seed,
            strategy=strategy,  # type: ignore[arg-type]
            group_by=group_by,
            corpus_policy=corpus_policy,  # type: ignore[arg-type]
        )

    if held_out_eval_file is not None:
        held_out_path = Path(held_out_eval_file).resolve()
        if validation_file is not None and held_out_path == Path(validation_file).resolve():
            raise ValueError("Held-out benchmark evaluation cannot be used as training validation")
        held_out_rows = [example.model_dump(mode="json") for example in load_eval(held_out_path)]
        ensure_disjoint_training_rows(
            split.train_rows,
            held_out_rows,
            left_name="training data",
            right_name="held-out benchmark evaluation",
        )
        ensure_disjoint_training_rows(
            split.validation_rows,
            held_out_rows,
            left_name="training validation",
            right_name="held-out benchmark evaluation",
        )
    mining_scopes = {
        str(row.get("metadata", {}).get("negative_mining", {}).get("scope"))
        for row in [*split.train_rows, *split.validation_rows]
        if isinstance(row.get("metadata"), Mapping)
        and isinstance(row.get("metadata", {}).get("negative_mining"), Mapping)
        and row.get("metadata", {}).get("negative_mining", {}).get("scope")
    }
    if {"train-partition-only", "validation-partition-only"} <= mining_scopes:
        mining_scope = "split-before-mining"
    elif mining_scopes:
        mining_scope = ",".join(sorted(mining_scopes))
    else:
        mining_scope = "not-applicable"
    return replace(split, negative_mining_scope=mining_scope)


def train_qlora(
    *,
    recipe_config: str | Path,
    train_file: str | Path,
    validation_file: str | Path | None = None,
    held_out_eval_file: str | Path | None = None,
) -> Path:
    """Run a single-GPU LoRA/QLoRA job with isolated validation data."""
    if held_out_eval_file is None:
        raise ValueError(
            "--held-out-eval is required to create a benchmark-verifiable adapter manifest"
        )
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install training extras: pip install -e '.[train]'") from exc

    recipe_path = Path(recipe_config)
    recipe = load_yaml(recipe_path)
    model_path = resolve_relative(recipe_path, recipe["model"])
    training_path = resolve_relative(recipe_path, recipe["training"]["config"])
    model_cfg = load_yaml(model_path)
    training_cfg = load_yaml(training_path)
    model_id, model_revision = validate_hf_model_config(model_cfg)
    chat_template_kwargs = effective_chat_template_kwargs(model_cfg)
    if chat_template_kwargs.get("enable_thinking") is True:
        raise ValueError(
            "Thinking-enabled adapter training is not supported by the current answer-only "
            "completion schema; use the benchmark's non-thinking condition or provide an "
            "externally trained, schema-v3 verified thinking adapter"
        )
    mode = recipe["training"].get("mode", "sft")
    if mode not in {"sft", "raft"}:
        raise ValueError(f"Unsupported training mode: {mode!r}")
    output_dir = Path(recipe.get("output_dir", f"outputs/{recipe['name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(training_cfg.get("seed", 42))
    split = _load_training_split(
        train_file=train_file,
        validation_file=validation_file,
        held_out_eval_file=held_out_eval_file,
        validation_ratio=float(training_cfg.get("validation_split_ratio", 0.1)),
        seed=seed,
        split_config=(
            training_cfg.get("split") if isinstance(training_cfg.get("split"), Mapping) else None
        ),
    )
    require_verifiable_training_prompt(training_cfg)
    use_chat_template = True
    train_fingerprint = split_fingerprint(split.train_rows)
    validation_fingerprint = split_fingerprint(split.validation_rows)
    training_source_fingerprint = source_partition_fingerprint(split.train_rows)
    validation_source_fingerprint = source_partition_fingerprint(split.validation_rows)
    train_records = prompt_completion_records(
        split.train_rows,
        mode=mode,
        use_chat_template=use_chat_template,
    )
    validation_records = prompt_completion_records(
        split.validation_rows,
        mode=mode,
        use_chat_template=use_chat_template,
    )
    quantization_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype_name = str(training_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    if compute_dtype_name not in quantization_dtypes:
        raise ValueError("bnb_4bit_compute_dtype must be bfloat16, float16, or float32")
    compute_dtype = quantization_dtypes[compute_dtype_name]
    quant_cfg = None
    if bool(training_cfg.get("load_in_4bit", True)):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=training_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(training_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_chat_template:
        train_records = render_chat_prompt_completions(
            train_records,
            tokenizer=tokenizer,
            chat_template_kwargs=chat_template_kwargs,
        )
        validation_records = render_chat_prompt_completions(
            validation_records,
            tokenizer=tokenizer,
            chat_template_kwargs=chat_template_kwargs,
        )
    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(validation_records) if validation_records else None
    training_controls = normalize_training_controls(
        training_cfg,
        has_validation=eval_dataset is not None,
    )
    control_sha256 = training_control_sha256(training_controls)

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "trust_remote_code": False,
        "device_map": "auto",
        "dtype": dtype_by_name.get(model_cfg.get("torch_dtype", "bfloat16")),
    }
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=int(training_cfg.get("lora_r", 16)),
        lora_alpha=int(training_cfg.get("lora_alpha", 32)),
        lora_dropout=float(training_cfg.get("lora_dropout", 0.05)),
        bias=str(training_cfg.get("lora_bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=training_cfg.get("target_modules", "all-linear"),
    )

    sft_values = build_sft_config_values(
        training_cfg,
        output_dir=output_dir,
        report_to_wandb=recipe.get("tracking", {}).get("backend") == "wandb",
        has_validation=eval_dataset is not None,
    )
    callbacks: list[Any] = []
    patience = int(training_cfg.get("early_stopping_patience", 0))
    if eval_dataset is not None and patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=float(training_cfg.get("early_stopping_threshold", 0.0)),
            )
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_cfg,
        args=SFTConfig(**sft_values),
        callbacks=callbacks,
    )
    result = trainer.train()
    trainer.save_metrics("train", result.metrics)
    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    if eval_metrics:
        trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    prompt_provenance = rag_prompt_provenance()
    held_out_evaluation_sha256 = file_sha256(held_out_eval_file)
    training_configuration_sha256 = canonical_sha256(
        {
            "recipe": recipe,
            "model": model_cfg,
            "training": training_cfg,
            "chat_template_kwargs": chat_template_kwargs,
            "prompt": prompt_provenance,
        }
    )
    adapter_artifact_sha256 = artifact_sha256(adapter_path)

    manifest = {
        "schema_name": "raglab-training-manifest",
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "recipe": recipe.get("name"),
        "mode": mode,
        "adaptation_mode": mode,
        "model": {"model_id": model_id, "revision": model_revision},
        "chat_template_kwargs": chat_template_kwargs,
        "training_prompt": prompt_provenance,
        "training_config": training_cfg,
        "training_configuration_sha256": training_configuration_sha256,
        "training_controls": training_controls,
        "training_control_sha256": control_sha256,
        "training_dataset_fingerprint": train_fingerprint,
        "validation_dataset_fingerprint": validation_fingerprint,
        "training_source_fingerprint": training_source_fingerprint,
        "validation_source_fingerprint": validation_source_fingerprint,
        "held_out_evaluation_sha256": held_out_evaluation_sha256,
        "configuration_files": {
            "recipe": _file_record(recipe_path),
            "model": _file_record(model_path),
            "training": _file_record(training_path),
        },
        "loss_masking": {
            "strategy": "completion-only",
            "prompt_tokens_in_loss": False,
            "trl_completion_only_loss": True,
        },
        "split": {
            **split.metadata(),
            "train_examples": len(split.train_rows),
            "validation_examples": len(split.validation_rows),
            "train_ids": [str(row.get("id", "")) for row in split.train_rows],
            "validation_ids": [str(row.get("id", "")) for row in split.validation_rows],
            "train_fingerprint": train_fingerprint,
            "validation_fingerprint": validation_fingerprint,
            "train_file": str(train_file),
            "validation_file": str(validation_file) if validation_file else None,
            "held_out_eval_file": str(held_out_eval_file) if held_out_eval_file else None,
            "source_files": {
                "train": _file_record(train_file),
                "validation": _file_record(validation_file),
                "held_out_eval": _file_record(held_out_eval_file),
            },
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
        "metric_for_best_model": sft_values.get("metric_for_best_model"),
        "train_metrics": result.metrics,
        "validation_metrics": eval_metrics,
        "adapter_path": str(adapter_path),
        "adapter_artifact_sha256": adapter_artifact_sha256,
    }
    adapter_manifest = {
        "schema_name": "raglab-adapter-manifest",
        "schema_version": ADAPTER_MANIFEST_SCHEMA_VERSION,
        "model": {"model_id": model_id, "revision": model_revision},
        "recipe": recipe.get("name"),
        "adaptation_mode": mode,
        "training_prompt": prompt_provenance,
        "chat_template_kwargs": chat_template_kwargs,
        "training_dataset_fingerprint": train_fingerprint,
        "validation_dataset_fingerprint": validation_fingerprint,
        "training_source_fingerprint": training_source_fingerprint,
        "validation_source_fingerprint": validation_source_fingerprint,
        "held_out_evaluation_sha256": held_out_evaluation_sha256,
        "training_configuration_sha256": training_configuration_sha256,
        "training_controls": training_controls,
        "training_control_sha256": control_sha256,
        "adapter_artifact_sha256": adapter_artifact_sha256,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
    }
    validate_artifact_schema(manifest, "training-manifest-v3.schema.json")
    validate_artifact_schema(adapter_manifest, "adapter-manifest-v3.schema.json")
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (adapter_path / ADAPTER_MANIFEST_FILENAME).write_text(
        json.dumps(adapter_manifest, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    return adapter_path
