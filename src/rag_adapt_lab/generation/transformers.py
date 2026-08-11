from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rag_adapt_lab.config import validate_hf_model_config

from .base import GenerationResult, Generator
from .prompts import format_rag_user_prompt


def validate_adapter_identity(
    adapter_path: str | Path,
    *,
    model_id: str,
    model_revision: str,
) -> None:
    """Reject adapters that declare a different base model or revision."""
    adapter = Path(adapter_path)
    peft_config_path = adapter / "adapter_config.json"
    if peft_config_path.is_file():
        peft_config = json.loads(peft_config_path.read_text(encoding="utf-8"))
        configured_base = peft_config.get("base_model_name_or_path")
        if configured_base and configured_base != model_id:
            raise ValueError(
                f"Adapter base model {configured_base!r} does not match benchmark model "
                f"{model_id!r}"
            )

    manifest_path = adapter / "raglab_adapter_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter_model = manifest.get("model", {})
    configured_id = adapter_model.get("model_id")
    configured_revision = adapter_model.get("revision")
    if configured_id != model_id or configured_revision != model_revision:
        raise ValueError(
            "Adapter identity does not match the immutable benchmark base: "
            f"adapter={configured_id}@{configured_revision}, "
            f"benchmark={model_id}@{model_revision}"
        )


class TransformersGenerator(Generator):
    """Single-GPU local generator with optional PEFT adapter loading."""

    SUPPORTED_GENERATION_FIELDS = {
        "do_sample",
        "early_stopping",
        "length_penalty",
        "max_new_tokens",
        "min_new_tokens",
        "no_repeat_ngram_size",
        "num_beams",
        "repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
        "typical_p",
    }

    def __init__(
        self,
        *,
        model_config: Mapping[str, Any],
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = False,
        seed: int = 42,
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Install training extras to use local Transformers generation"
            ) from exc

        self.torch = torch
        self.model_config = dict(model_config)
        model_id, revision = validate_hf_model_config(self.model_config)
        self.model_id = model_id
        self.model_revision = revision
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.generation_config = dict(self.model_config.get("generation", {}))
        unsupported_generation_fields = (
            set(self.generation_config) - self.SUPPORTED_GENERATION_FIELDS
        )
        if unsupported_generation_fields:
            raise ValueError(
                "Unsupported generation configuration fields: "
                f"{sorted(unsupported_generation_fields)}"
            )
        self.seed = seed
        self._generation_index = 0

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
        )
        self.tokenizer.padding_side = "left"
        # Question and highest-ranked documents occur first in prompt v3, so
        # right truncation retains the most decision-relevant input.
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": False,
            "device_map": "auto",
            "dtype": dtype_by_name.get(self.model_config.get("torch_dtype", "bfloat16")),
        }
        if self.model_config.get("attn_implementation"):
            model_kwargs["attn_implementation"] = self.model_config["attn_implementation"]
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype_by_name.get(
                    self.model_config.get("torch_dtype", "bfloat16"), torch.bfloat16
                ),
            )

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not bool(self.generation_config.get("do_sample", False)):
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            model.generation_config.typical_p = None
        if adapter_path is not None:
            adapter = Path(adapter_path)
            if not adapter.exists():
                raise FileNotFoundError(f"Adapter path does not exist: {adapter}")
            validate_adapter_identity(
                adapter,
                model_id=model_id,
                model_revision=revision,
            )
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        self.model = model

    def _synchronize(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def generate(self, *, question: str, contexts: list[str] | None = None) -> GenerationResult:
        user_prompt = format_rag_user_prompt(question=question, contexts=contexts or [])
        chat_template_kwargs = dict(self.model_config.get("chat_template_kwargs", {}))
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        else:
            prompt = user_prompt
        max_new_tokens = int(self.generation_config.get("max_new_tokens", 128))
        configured_length = int(
            self.model_config.get("max_seq_length", self.tokenizer.model_max_length)
        )
        if max_new_tokens < 1 or configured_length <= max_new_tokens:
            raise ValueError("max_seq_length must be greater than positive max_new_tokens")
        max_input_tokens = configured_length - max_new_tokens
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(self.model.device)

        do_sample = bool(self.generation_config.get("do_sample", False))
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        for name in (
            "early_stopping",
            "length_penalty",
            "min_new_tokens",
            "no_repeat_ngram_size",
            "num_beams",
            "repetition_penalty",
        ):
            if name in self.generation_config:
                generation_kwargs[name] = self.generation_config[name]
        if do_sample:
            for name in ("temperature", "top_p", "top_k", "typical_p"):
                if name in self.generation_config:
                    generation_kwargs[name] = self.generation_config[name]

        # Resetting this counter per recipe gives every paired condition the
        # same sequence of sampling seeds without requiring global RNG state.
        self.torch.manual_seed(self.seed + self._generation_index)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.seed + self._generation_index)
        self._generation_index += 1
        self._synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        self._synchronize()
        latency = time.perf_counter() - started
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated = outputs[:, prompt_tokens:]
        output_tokens = int(generated.shape[1])
        text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
        )

    def reset_runtime_metrics(self) -> None:
        self._generation_index = 0
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def peak_memory_gb(self) -> float | None:
        if not self.torch.cuda.is_available():
            return None
        return float(self.torch.cuda.max_memory_allocated() / 1024**3)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
