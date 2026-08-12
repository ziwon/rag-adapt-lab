from pathlib import Path

import pytest
import yaml

from rag_adapt_lab.config import (
    effective_chat_template_kwargs,
    effective_generation_config,
    validate_hf_model_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_model_configs_pin_revisions_and_disable_remote_code() -> None:
    for path in sorted((PROJECT_ROOT / "configs" / "models").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_id, revision = validate_hf_model_config(config)
        assert model_id == config["model_id"]
        assert revision == config["revision"]
        if "Qwen3" in model_id:
            assert effective_chat_template_kwargs(config) == {"enable_thinking": False}
            assert config["generation"] == {"max_new_tokens": 64, "do_sample": False}


def test_training_configs_use_model_chat_templates() -> None:
    for path in sorted((PROJECT_ROOT / "configs" / "training").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["use_chat_template"] is True, path


def test_mutable_model_revision_is_rejected() -> None:
    with pytest.raises(ValueError, match="immutable 40-character"):
        validate_hf_model_config(
            {"model_id": "organization/model", "revision": "main", "trust_remote_code": False}
        )


def test_remote_model_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="remote model code is disabled"):
        validate_hf_model_config(
            {
                "model_id": "organization/model",
                "revision": "0" * 40,
                "trust_remote_code": True,
            }
        )


def test_qwen3_requires_an_explicit_thinking_condition() -> None:
    with pytest.raises(ValueError, match="explicitly set"):
        validate_hf_model_config(
            {
                "model_id": "Qwen/Qwen3-Test",
                "revision": "0" * 40,
                "generation": {"max_new_tokens": 64, "do_sample": False},
            }
        )


def test_qwen3_thinking_rejects_ambiguous_greedy_decoding() -> None:
    with pytest.raises(ValueError, match="requires do_sample=true"):
        validate_hf_model_config(
            {
                "model_id": "Qwen/Qwen3-Test",
                "revision": "0" * 40,
                "chat_template_kwargs": {"enable_thinking": True},
                "generation": {"max_new_tokens": 256, "do_sample": False},
            }
        )


def test_qwen3_thinking_accepts_an_explicit_sampled_condition() -> None:
    config = {
        "model_id": "Qwen/Qwen3-Test",
        "revision": "0" * 40,
        "chat_template_kwargs": {"enable_thinking": True},
        "generation": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    }
    assert validate_hf_model_config(config) == ("Qwen/Qwen3-Test", "0" * 40)


def test_thinking_token_override_is_revalidated() -> None:
    config = {
        "model_id": "Qwen/Qwen3-Test",
        "revision": "0" * 40,
        "chat_template_kwargs": {"enable_thinking": True},
        "generation": {"max_new_tokens": 256, "do_sample": True, "temperature": 0.6},
    }
    with pytest.raises(ValueError, match="max_new_tokens>=128"):
        effective_generation_config(config, max_new_tokens=64)
