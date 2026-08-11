from pathlib import Path

import pytest
import yaml

from rag_adapt_lab.config import validate_hf_model_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_model_configs_pin_revisions_and_disable_remote_code() -> None:
    for path in sorted((PROJECT_ROOT / "configs" / "models").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_id, revision = validate_hf_model_config(config)
        assert model_id == config["model_id"]
        assert revision == config["revision"]


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
