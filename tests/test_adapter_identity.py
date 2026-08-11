import json
from pathlib import Path

import pytest

from rag_adapt_lab.generation.prompts import rag_prompt_provenance
from rag_adapt_lab.generation.transformers import validate_adapter_identity
from rag_adapt_lab.provenance import (
    ADAPTER_MANIFEST_SCHEMA_VERSION,
    artifact_sha256,
)

MODEL_ID = "test/model"
MODEL_REVISION = "a" * 40
EVAL_HASH = "e" * 64


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_adapter_manifest(
    path: Path,
    *,
    mode: str = "sft",
    revision: str = MODEL_REVISION,
    prompt_version: str | None = None,
    eval_hash: str = EVAL_HASH,
) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "adapter_config.json", {"base_model_name_or_path": MODEL_ID})
    (path / "adapter_model.safetensors").write_bytes(f"weights-{mode}".encode())
    prompt = rag_prompt_provenance()
    if prompt_version is not None:
        prompt["version"] = prompt_version
    manifest: dict[str, object] = {
        "schema_version": ADAPTER_MANIFEST_SCHEMA_VERSION,
        "model": {"model_id": MODEL_ID, "revision": revision},
        "adaptation_mode": mode,
        "training_prompt": prompt,
        "chat_template_kwargs": {},
        "training_dataset_fingerprint": "1" * 64,
        "validation_dataset_fingerprint": "2" * 64,
        "held_out_evaluation_sha256": eval_hash,
        "training_configuration_sha256": "3" * 64,
        "adapter_artifact_sha256": artifact_sha256(path),
    }
    write_json(path / "raglab_adapter_manifest.json", manifest)
    return manifest


def validate(path: Path, **kwargs: object) -> object:
    return validate_adapter_identity(
        path,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        expected_mode=kwargs.get("expected_mode", "sft"),  # type: ignore[arg-type]
        expected_prompt={**rag_prompt_provenance(), "chat_template_kwargs": {}},
        held_out_evaluation_sha256=EVAL_HASH,
        allow_unverified=bool(kwargs.get("allow_unverified", False)),
    )


@pytest.mark.parametrize("mode", ["sft", "raft"])
def test_adapter_identity_accepts_complete_verified_manifest(tmp_path: Path, mode: str) -> None:
    write_adapter_manifest(tmp_path, mode=mode)
    result = validate(tmp_path, expected_mode=mode)
    assert result.verified is True
    assert result.adaptation_mode == mode


def test_adapter_identity_rejects_wrong_adaptation_mode(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path, mode="raft")
    with pytest.raises(ValueError, match="adaptation mode"):
        validate(tmp_path)


def test_adapter_identity_rejects_wrong_revision(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path, revision="b" * 40)
    with pytest.raises(ValueError, match="immutable benchmark base"):
        validate(tmp_path)


def test_adapter_identity_rejects_wrong_prompt_version(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path, prompt_version="old")
    with pytest.raises(ValueError, match="prompt version"):
        validate(tmp_path)


def test_adapter_identity_rejects_wrong_evaluation_hash(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path, eval_hash="f" * 64)
    with pytest.raises(ValueError, match="evaluation hash"):
        validate(tmp_path)


def test_adapter_identity_fails_closed_without_manifest(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": MODEL_ID})
    with pytest.raises(ValueError, match="no verifiable"):
        validate(tmp_path)


def test_legacy_override_marks_missing_manifest_unverified(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": MODEL_ID})
    (tmp_path / "adapter_model.safetensors").write_bytes(b"legacy")
    result = validate(tmp_path, allow_unverified=True)
    assert result.verified is False
    assert result.warnings


def test_legacy_adapter_still_rejects_wrong_base_model(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": "other/model"})
    with pytest.raises(ValueError, match="does not match benchmark model"):
        validate(tmp_path, allow_unverified=True)
