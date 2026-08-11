from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag_adapt_lab.config import load_yaml, resolve_relative, validate_hf_model_config
from rag_adapt_lab.data.io import load_eval, read_jsonl

from .data import (
    TrainingSplit,
    deterministic_training_split,
    ensure_disjoint_training_rows,
    prompt_completion_records,
    split_fingerprint,
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": _file_sha256(resolved)}


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
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 2)),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 16)),
        "max_length": int(training_cfg.get("max_seq_length", 2048)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "logging_steps": int(training_cfg.get("logging_steps", 5)),
        "save_strategy": eval_strategy if has_validation else "steps",
        "save_steps": save_steps,
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "bf16": bool(training_cfg.get("bf16", True)),
        "seed": int(training_cfg.get("seed", 42)),
        "data_seed": int(training_cfg.get("seed", 42)),
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


def _load_training_split(
    *,
    train_file: str | Path,
    validation_file: str | Path | None,
    held_out_eval_file: str | Path | None,
    validation_ratio: float,
    seed: int,
) -> TrainingSplit:
    train_path = Path(train_file).resolve()
    rows = read_jsonl(train_path)
    if not rows:
        raise ValueError(f"Training file is empty: {train_file}")

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
        split = TrainingSplit(
            rows,
            validation_rows,
            "explicit-files",
            seed,
            len(validation_rows) / (len(rows) + len(validation_rows)),
        )
    else:
        split = deterministic_training_split(
            rows,
            validation_ratio=validation_ratio,
            seed=seed,
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
    return split


def train_qlora(
    *,
    recipe_config: str | Path,
    train_file: str | Path,
    validation_file: str | Path | None = None,
    held_out_eval_file: str | Path | None = None,
) -> Path:
    """Run a single-GPU LoRA/QLoRA job with isolated validation data."""
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
    )
    use_chat_template = bool(training_cfg.get("use_chat_template", False))
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
    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(validation_records) if validation_records else None

    compute_dtype = (
        torch.bfloat16
        if training_cfg.get("bnb_4bit_compute_dtype") == "bfloat16"
        else torch.float16
    )
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
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=training_cfg.get("target_modules", "all-linear"),
    )

    sft_values = build_sft_config_values(
        training_cfg,
        output_dir=output_dir,
        report_to_wandb=recipe.get("tracking", {}).get("backend") == "wandb",
        has_validation=eval_dataset is not None,
    )
    callbacks = []
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

    manifest = {
        "schema_version": 1,
        "recipe": recipe.get("name"),
        "mode": mode,
        "model": {"model_id": model_id, "revision": model_revision},
        "training_config": training_cfg,
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
            "method": split.method,
            "seed": split.seed,
            "validation_ratio": split.validation_ratio,
            "train_examples": len(split.train_rows),
            "validation_examples": len(split.validation_rows),
            "train_ids": [str(row.get("id", "")) for row in split.train_rows],
            "validation_ids": [str(row.get("id", "")) for row in split.validation_rows],
            "train_fingerprint": split_fingerprint(split.train_rows),
            "validation_fingerprint": split_fingerprint(split.validation_rows),
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
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (adapter_path / "raglab_adapter_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"model_id": model_id, "revision": model_revision},
                "recipe": recipe.get("name"),
                "mode": mode,
                "train_fingerprint": split_fingerprint(split.train_rows),
                "validation_fingerprint": split_fingerprint(split.validation_rows),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_validation_metric": trainer.state.best_metric,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return adapter_path
