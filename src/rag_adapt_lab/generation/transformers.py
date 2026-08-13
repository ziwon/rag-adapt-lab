from __future__ import annotations

import gc
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rag_adapt_lab.config import (
    effective_chat_template_kwargs,
    validate_hf_model_config,
    validate_thinking_configuration,
)
from rag_adapt_lab.provenance import AdapterVerification, validate_adapter_provenance

from .base import GenerationResult, Generator
from .prompts import format_rag_user_prompt
from .thinking import parse_thinking_tokens


def validate_adapter_identity(
    adapter_path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    expected_mode: str | None = None,
    expected_prompt: Mapping[str, Any] | None = None,
    held_out_evaluation_sha256: str | None = None,
    allow_unverified: bool = False,
) -> AdapterVerification:
    """Compatibility wrapper around fail-closed adapter provenance validation."""
    if expected_mode not in {None, "sft", "raft"}:
        raise ValueError(f"Unsupported expected adapter mode: {expected_mode!r}")
    return validate_adapter_provenance(
        adapter_path,
        model_id=model_id,
        model_revision=model_revision,
        expected_mode=expected_mode,  # type: ignore[arg-type]
        expected_prompt=expected_prompt,
        held_out_evaluation_sha256=held_out_evaluation_sha256,
        allow_unverified=allow_unverified,
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
        allow_unverified_adapter: bool = False,
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
        self.chat_template_kwargs = effective_chat_template_kwargs(self.model_config)
        self.thinking_enabled = validate_thinking_configuration(self.model_config)
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
        # Question and highest-ranked documents occur first in prompt v4, so
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

        model: Any = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
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
                allow_unverified=allow_unverified_adapter,
            )
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        self.model: Any = model

    def _synchronize(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def generate(self, *, question: str, contexts: list[str] | None = None) -> GenerationResult:
        inference_started = time.perf_counter()
        stage_started = time.perf_counter()
        user_prompt = format_rag_user_prompt(question=question, contexts=contexts or [])
        prompt_build_latency = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        else:
            prompt = user_prompt
        chat_template_latency = time.perf_counter() - stage_started
        max_new_tokens = int(self.generation_config.get("max_new_tokens", 64))
        configured_length = int(
            self.model_config.get("max_seq_length", self.tokenizer.model_max_length)
        )
        if max_new_tokens < 1 or configured_length <= max_new_tokens:
            raise ValueError("max_seq_length must be greater than positive max_new_tokens")
        max_input_tokens = configured_length - max_new_tokens
        stage_started = time.perf_counter()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        tokenization_latency = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        inputs = inputs.to(self.model.device)
        self._synchronize()
        device_transfer_latency = time.perf_counter() - stage_started

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
        model_generate_latency = time.perf_counter() - started
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated = outputs[:, prompt_tokens:]
        stage_started = time.perf_counter()
        parsed = parse_thinking_tokens(
            generated[0].tolist(),
            tokenizer=self.tokenizer,
            thinking_enabled=self.thinking_enabled,
        )
        decode_latency = time.perf_counter() - stage_started
        inference_e2e_latency = time.perf_counter() - inference_started
        return GenerationResult(
            text=parsed.answer,
            raw_text=parsed.raw_text,
            reasoning=parsed.reasoning,
            prompt_tokens=prompt_tokens,
            output_tokens=parsed.output_tokens,
            reasoning_tokens=parsed.reasoning_tokens,
            answer_tokens=parsed.answer_tokens,
            thinking_boundary_token_id=parsed.boundary_token_id,
            thinking_boundary_found=parsed.boundary_found,
            thinking_protocol_violation=parsed.thinking_protocol_violation,
            prompt_build_latency_s=prompt_build_latency,
            chat_template_latency_s=chat_template_latency,
            tokenization_latency_s=tokenization_latency,
            device_transfer_latency_s=device_transfer_latency,
            model_generate_latency_s=model_generate_latency,
            decode_latency_s=decode_latency,
            inference_e2e_latency_s=inference_e2e_latency,
        )

    def reset_runtime_metrics(self) -> None:
        self._generation_index = 0
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def peak_memory_gb(self) -> float | None:
        """Deprecated allocated-memory alias retained for older callers."""
        return self.peak_allocated_vram_gb()

    def peak_allocated_vram_gb(self) -> float | None:
        if not self.torch.cuda.is_available():
            return None
        return float(self.torch.cuda.max_memory_allocated() / 1024**3)

    def peak_reserved_vram_gb(self) -> float | None:
        if not self.torch.cuda.is_available():
            return None
        return float(self.torch.cuda.max_memory_reserved() / 1024**3)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
