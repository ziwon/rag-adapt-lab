from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_adapt_lab.config import load_yaml, resolve_relative
from rag_adapt_lab.data.io import read_jsonl

from .formatting import format_raft_row, format_sft_row


def train_qlora(*, recipe_config: str | Path, train_file: str | Path) -> Path:
    """Run a single-GPU QLoRA SFT/RAFT training job with TRL.

    This function intentionally keeps the surface small. Advanced distributed or
    model-specific optimization belongs in later backends.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install training extras: pip install -e '.[train]'") from exc

    recipe_path = Path(recipe_config)
    recipe = load_yaml(recipe_path)
    model_path = resolve_relative(recipe_path, recipe["model"])
    training_path = resolve_relative(recipe_path, recipe["training"]["config"])
    model_cfg = load_yaml(model_path)
    training_cfg = load_yaml(training_path)
    mode = recipe["training"].get("mode", "sft")
    output_dir = Path(recipe.get("output_dir", f"outputs/{recipe['name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(train_file)
    formatter = format_raft_row if mode == "raft" else format_sft_row
    dataset = Dataset.from_dict({"text": [formatter(row) for row in rows]})

    compute_dtype = torch.bfloat16 if training_cfg.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(training_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=training_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(training_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_id"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        quantization_config=quant_cfg,
        device_map="auto",
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=int(training_cfg.get("lora_r", 16)),
        lora_alpha=int(training_cfg.get("lora_alpha", 32)),
        lora_dropout=float(training_cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=training_cfg.get("target_modules", "all-linear"),
    )

    args: dict[str, Any] = dict(
        output_dir=str(output_dir),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 16)),
        max_length=int(training_cfg.get("max_seq_length", 2048)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        save_steps=int(training_cfg.get("save_steps", 100)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        bf16=bool(training_cfg.get("bf16", True)),
        seed=int(training_cfg.get("seed", 42)),
        report_to=["wandb"] if recipe.get("tracking", {}).get("backend") == "wandb" else [],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_cfg,
        args=SFTConfig(**args),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    return output_dir / "adapter"
