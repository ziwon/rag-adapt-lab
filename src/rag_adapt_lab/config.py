from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_HF_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_relative(config_path: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = Path(config_path).resolve().parent / path
    if candidate.exists():
        return candidate
    return path


def require_pinned_hf_revision(revision: object, *, model_id: str) -> str:
    """Return a Hugging Face commit SHA or reject a mutable model revision."""
    if not isinstance(revision, str) or _HF_COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError(
            f"Model {model_id!r} must use an immutable 40-character Hugging Face commit SHA; "
            f"got revision {revision!r}"
        )
    return revision


def validate_hf_model_config(config: dict[str, Any]) -> tuple[str, str]:
    """Validate the reproducibility and remote-code policy for a model config."""
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Model config must contain a non-empty model_id")
    revision = require_pinned_hf_revision(config.get("revision"), model_id=model_id)
    trust_remote_code = config.get("trust_remote_code", False)
    if not isinstance(trust_remote_code, bool):
        raise ValueError("trust_remote_code must be a boolean")
    if trust_remote_code:
        raise ValueError(
            f"Model {model_id!r} requests trust_remote_code=true; remote model code is disabled"
        )
    validate_thinking_configuration(config)
    return model_id, revision


def is_qwen3_model(config: Mapping[str, Any]) -> bool:
    """Return whether a model config identifies the Qwen3 model family."""
    return "qwen3" in str(config.get("model_id", "")).casefold()


def effective_chat_template_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the exact kwargs applied to a model chat template.

    Qwen3 has a behavior-changing ``enable_thinking`` template argument.  An
    implicit default is not a reproducible benchmark condition, so Qwen3
    configurations must always specify it.
    """
    raw = config.get("chat_template_kwargs", {})
    if not isinstance(raw, Mapping):
        raise ValueError("chat_template_kwargs must be a mapping")
    values = dict(raw)
    if is_qwen3_model(config) and "enable_thinking" not in values:
        raise ValueError(
            "Qwen3 model configs must explicitly set "
            "chat_template_kwargs.enable_thinking to true or false"
        )
    if "enable_thinking" in values and not isinstance(values["enable_thinking"], bool):
        raise ValueError("chat_template_kwargs.enable_thinking must be a boolean")
    if values.get("enable_thinking") is True and not is_qwen3_model(config):
        raise ValueError("enable_thinking=true is only supported for explicit Qwen3 configs")
    return values


def validate_thinking_configuration(config: Mapping[str, Any]) -> bool:
    """Validate that thinking and decoding settings form one explicit condition."""
    chat_kwargs = effective_chat_template_kwargs(config)
    thinking_enabled = chat_kwargs.get("enable_thinking") is True
    generation = config.get("generation", {})
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be a mapping")
    do_sample = generation.get("do_sample", False)
    if not isinstance(do_sample, bool):
        raise ValueError("generation.do_sample must be a boolean")
    try:
        max_new_tokens = int(generation.get("max_new_tokens", 64))
    except (TypeError, ValueError) as exc:
        raise ValueError("generation.max_new_tokens must be an integer") from exc
    if max_new_tokens < 1:
        raise ValueError("generation.max_new_tokens must be positive")

    if is_qwen3_model(config) and not thinking_enabled and do_sample:
        raise ValueError(
            "The default concise Qwen3 condition requires deterministic generation "
            "with do_sample=false"
        )
    if thinking_enabled:
        if not do_sample:
            raise ValueError(
                "Qwen3 thinking mode is a separate sampled condition and requires do_sample=true"
            )
        temperature = generation.get("temperature")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("Qwen3 thinking mode requires a positive generation.temperature")
        if float(temperature) <= 0:
            raise ValueError("Qwen3 thinking mode requires a positive generation.temperature")
        if max_new_tokens < 128:
            raise ValueError(
                "Qwen3 thinking mode requires max_new_tokens>=128 so the final answer is not "
                "systematically truncated"
            )
    return thinking_enabled


def effective_generation_config(
    config: Mapping[str, Any],
    *,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Return generation settings after validating any CLI token override."""
    raw = config.get("generation", {})
    if not isinstance(raw, Mapping):
        raise ValueError("generation must be a mapping")
    generation = dict(raw)
    if max_new_tokens is not None:
        generation["max_new_tokens"] = max_new_tokens
    generation.setdefault("max_new_tokens", 64)
    generation.setdefault("do_sample", False)
    effective = dict(config)
    effective["generation"] = generation
    validate_thinking_configuration(effective)
    return generation
