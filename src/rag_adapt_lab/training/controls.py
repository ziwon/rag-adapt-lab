from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag_adapt_lab.provenance import canonical_sha256


def _normalized_target_modules(value: Any) -> str | list[str]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("target_modules must not be empty")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        modules = sorted({str(item).strip() for item in value if str(item).strip()})
        if not modules:
            raise ValueError("target_modules must not be empty")
        return modules
    raise ValueError("target_modules must be a string or sequence of strings")


def normalize_training_controls(
    training_cfg: Mapping[str, Any],
    *,
    has_validation: bool,
) -> dict[str, Any]:
    """Normalize only controls that can affect learned weights or selection.

    Paths, recipe names, tracking, adaptation mode, and dataset representation
    are intentionally excluded so SFT and RAFT can be compared as treatments.
    """
    seed = int(training_cfg.get("seed", 42))
    data_seed = int(training_cfg.get("data_seed", seed))
    per_device_batch = int(training_cfg.get("per_device_train_batch_size", 1))
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 16))
    if per_device_batch < 1 or accumulation < 1:
        raise ValueError("Training batch size and gradient accumulation must be positive")
    load_in_4bit = bool(training_cfg.get("load_in_4bit", True))
    adaptation_method = "qlora" if load_in_4bit else "lora"
    configured_method = training_cfg.get("method")
    if configured_method is not None and str(configured_method).casefold() != adaptation_method:
        raise ValueError(
            f"Training method {configured_method!r} conflicts with load_in_4bit={load_in_4bit}"
        )
    compute_dtype = str(training_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    if compute_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("bnb_4bit_compute_dtype must be bfloat16, float16, or float32")
    eval_strategy = str(training_cfg.get("eval_strategy", "steps")) if has_validation else "no"
    eval_steps = int(training_cfg.get("eval_steps", 50)) if has_validation else None
    save_steps = int(training_cfg.get("save_steps", eval_steps or 50))
    metric = str(training_cfg.get("metric_for_best_model", "eval_loss"))
    greater_is_better = bool(
        training_cfg.get("greater_is_better", not metric.endswith("loss"))
    )
    target_modules = _normalized_target_modules(
        training_cfg.get("target_modules", "all-linear")
    )

    return {
        "adaptation_method": adaptation_method,
        "adapter": {
            "rank": int(training_cfg.get("lora_r", 16)),
            "alpha": int(training_cfg.get("lora_alpha", 32)),
            "dropout": float(training_cfg.get("lora_dropout", 0.05)),
            "target_modules": target_modules,
            "bias": str(training_cfg.get("lora_bias", "none")),
            "task_type": "CAUSAL_LM",
        },
        "optimization": {
            "learning_rate": float(training_cfg.get("learning_rate", 2e-4)),
            "optimizer": str(training_cfg.get("optim", "adamw_torch")),
            "optimizer_args": str(training_cfg.get("optim_args", "")),
            "adam_beta1": float(training_cfg.get("adam_beta1", 0.9)),
            "adam_beta2": float(training_cfg.get("adam_beta2", 0.999)),
            "adam_epsilon": float(training_cfg.get("adam_epsilon", 1e-8)),
            "scheduler": str(training_cfg.get("lr_scheduler_type", "linear")),
            "scheduler_kwargs": dict(training_cfg.get("lr_scheduler_kwargs", {})),
            "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
            "max_grad_norm": float(training_cfg.get("max_grad_norm", 1.0)),
            "num_train_epochs": float(training_cfg.get("num_train_epochs", 2)),
            "max_steps": int(training_cfg.get("max_steps", -1)),
        },
        "batching": {
            "per_device_train_batch_size": per_device_batch,
            "per_device_eval_batch_size": int(
                training_cfg.get("per_device_eval_batch_size", 1)
            ),
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size_per_device": per_device_batch * accumulation,
            "effective_batch_size": per_device_batch * accumulation,
        },
        "sequence": {
            "max_length": int(training_cfg.get("max_seq_length", 2048)),
        },
        "warmup": {
            "ratio": float(training_cfg.get("warmup_ratio", 0.03)),
            "steps": int(training_cfg.get("warmup_steps", 0)),
        },
        "gradient_checkpointing": {
            "enabled": bool(training_cfg.get("gradient_checkpointing", True)),
            "kwargs": dict(training_cfg.get("gradient_checkpointing_kwargs", {})),
        },
        "precision": {
            "bf16": bool(training_cfg.get("bf16", True)),
            "fp16": bool(training_cfg.get("fp16", False)),
            "tf32": bool(training_cfg.get("tf32", False)),
        },
        "quantization": {
            "load_in_4bit": load_in_4bit,
            "type": str(training_cfg.get("bnb_4bit_quant_type", "nf4"))
            if load_in_4bit
            else None,
            "double_quantization": bool(
                training_cfg.get("bnb_4bit_use_double_quant", True)
            )
            if load_in_4bit
            else None,
            "compute_dtype": compute_dtype
            if load_in_4bit
            else None,
        },
        "seeds": {"training": seed, "data": data_seed},
        "loss": {
            "completion_only": True,
            "label_smoothing_factor": float(
                training_cfg.get("label_smoothing_factor", 0.0)
            ),
        },
        "checkpoint_selection": {
            "validation_enabled": has_validation,
            "evaluation_strategy": eval_strategy,
            "evaluation_steps": eval_steps,
            "save_strategy": eval_strategy if has_validation else "steps",
            "save_steps": save_steps,
            "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
            "load_best_model_at_end": has_validation,
            "metric_for_best_model": metric if has_validation else None,
            "early_stopping_metric": metric if has_validation else None,
            "greater_is_better": greater_is_better if has_validation else None,
            "early_stopping_patience": int(
                training_cfg.get("early_stopping_patience", 0)
            )
            if has_validation
            else 0,
            "early_stopping_threshold": float(
                training_cfg.get("early_stopping_threshold", 0.0)
            )
            if has_validation
            else 0.0,
            "best_model_selection_policy": {
                "load_best_model_at_end": has_validation,
                "metric": metric if has_validation else None,
                "greater_is_better": greater_is_better if has_validation else None,
            },
        },
    }


def training_control_sha256(controls: Mapping[str, Any]) -> str:
    return canonical_sha256(controls)


def training_control_differences(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[str]:
    """Return dotted fields whose normalized control values differ."""
    differences: list[str] = []

    def compare(left: Any, right: Any, prefix: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                compare(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key))
            return
        if left != right:
            differences.append(f"{prefix}: {left!r} != {right!r}")

    compare(baseline, candidate, "")
    return differences
