import json
from pathlib import Path

import pytest

from rag_adapt_lab.generation.transformers import validate_adapter_identity


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_adapter_identity_accepts_exact_pinned_base(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": "test/model"})
    write_json(
        tmp_path / "raglab_adapter_manifest.json",
        {"model": {"model_id": "test/model", "revision": "a" * 40}},
    )
    validate_adapter_identity(tmp_path, model_id="test/model", model_revision="a" * 40)


def test_adapter_identity_rejects_wrong_revision(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": "test/model"})
    write_json(
        tmp_path / "raglab_adapter_manifest.json",
        {"model": {"model_id": "test/model", "revision": "b" * 40}},
    )
    with pytest.raises(ValueError, match="immutable benchmark base"):
        validate_adapter_identity(tmp_path, model_id="test/model", model_revision="a" * 40)


def test_legacy_adapter_rejects_wrong_base_model(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": "other/model"})
    with pytest.raises(ValueError, match="does not match benchmark model"):
        validate_adapter_identity(tmp_path, model_id="test/model", model_revision="a" * 40)
