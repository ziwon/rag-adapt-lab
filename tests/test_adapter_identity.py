import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_adapt_lab.generation.prompts import rag_prompt_provenance
from rag_adapt_lab.generation.transformers import validate_adapter_identity
from rag_adapt_lab.provenance import (
    ADAPTER_MANIFEST_SCHEMA_VERSION,
    artifact_sha256,
    canonical_sha256,
)
from rag_adapt_lab.training.controls import (
    normalize_training_controls,
    peft_lora_config_kwargs,
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
    training_config: dict[str, object] | None = None,
    peft_mutation: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    prompt = rag_prompt_provenance()
    if prompt_version is not None:
        prompt["version"] = prompt_version
    training_controls = normalize_training_controls(
        training_config or {}, has_validation=True
    )
    peft_config = peft_lora_config_kwargs(
        training_controls,
        model_id=MODEL_ID,
        revision=revision,
    )
    if peft_mutation is not None:
        peft_mutation(peft_config)
    write_json(
        path / "adapter_config.json",
        peft_config,
    )
    (path / "adapter_model.safetensors").write_bytes(f"weights-{mode}".encode())
    manifest: dict[str, object] = {
        "schema_name": "raglab-adapter-manifest",
        "schema_version": ADAPTER_MANIFEST_SCHEMA_VERSION,
        "model": {"model_id": MODEL_ID, "revision": revision},
        "recipe": f"test-{mode}",
        "adaptation_mode": mode,
        "training_prompt": prompt,
        "chat_template_kwargs": {},
        "training_dataset_fingerprint": "1" * 64,
        "validation_dataset_fingerprint": "2" * 64,
        "training_source_fingerprint": "4" * 64,
        "validation_source_fingerprint": "5" * 64,
        "held_out_evaluation_sha256": eval_hash,
        "training_configuration_sha256": "3" * 64,
        "training_controls": training_controls,
        "training_control_sha256": canonical_sha256(training_controls),
        "adapter_artifact_sha256": artifact_sha256(path),
        "best_checkpoint": None,
        "best_validation_metric": None,
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
    assert result.training_source_fingerprint == "4" * 64


def test_adapter_identity_rejects_wrong_adaptation_mode(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path, mode="raft")
    with pytest.raises(ValueError, match="adaptation mode"):
        validate(tmp_path, allow_unverified=True)


def test_adapter_identity_rejects_wrong_revision(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path, revision="b" * 40)
    with pytest.raises(ValueError, match="immutable benchmark base"):
        validate(tmp_path, allow_unverified=True)


def test_adapter_identity_rejects_wrong_prompt_version(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path, prompt_version="old")
    with pytest.raises(ValueError, match="prompt version"):
        validate(tmp_path, allow_unverified=True)


def test_adapter_identity_rejects_wrong_evaluation_hash(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path, eval_hash="f" * 64)
    with pytest.raises(ValueError, match="evaluation hash"):
        validate(tmp_path, allow_unverified=True)


def test_adapter_identity_fails_closed_without_manifest(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": MODEL_ID})
    with pytest.raises(ValueError, match="no verifiable"):
        validate(tmp_path)


def test_legacy_override_marks_missing_manifest_unverified(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": MODEL_ID})
    (tmp_path / "adapter_model.safetensors").write_bytes(b"legacy")
    result = validate(tmp_path, allow_unverified=True)
    assert result.verified is False
    assert result.status == "legacy_manifest_unavailable"
    assert result.reason_code == "manifest_missing"
    assert "training_controls" in result.unchecked_fields
    assert result.warnings


def test_legacy_adapter_still_rejects_wrong_base_model(tmp_path: Path) -> None:
    write_json(tmp_path / "adapter_config.json", {"base_model_name_or_path": "other/model"})
    with pytest.raises(ValueError, match="does not match benchmark model"):
        validate(tmp_path, allow_unverified=True)


def test_legacy_v2_manifest_requires_override_and_records_reason(tmp_path: Path) -> None:
    write_json(
        tmp_path / "adapter_config.json",
        {"base_model_name_or_path": MODEL_ID, "revision": MODEL_REVISION},
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"legacy-v2")
    manifest = {
        "schema_version": 2,
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "adaptation_mode": "sft",
        "training_prompt": rag_prompt_provenance(),
        "chat_template_kwargs": {},
        "training_dataset_fingerprint": "1" * 64,
        "validation_dataset_fingerprint": "2" * 64,
        "held_out_evaluation_sha256": EVAL_HASH,
        "training_configuration_sha256": "3" * 64,
        "adapter_artifact_sha256": artifact_sha256(tmp_path),
    }
    write_json(tmp_path / "raglab_adapter_manifest.json", manifest)
    with pytest.raises(ValueError, match="schema v2 lacks"):
        validate(tmp_path)
    result = validate(tmp_path, allow_unverified=True)
    assert result.status == "unverified_legacy_provenance"
    assert result.reason_code == "legacy_training_controls_unavailable"
    assert result.adaptation_mode == "sft"


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("r", 64, "adapter.rank"),
        ("lora_alpha", 128, "adapter.alpha"),
        ("lora_dropout", 0.2, "adapter.dropout"),
        ("bias", "all", "adapter.bias"),
        ("target_modules", ["q_proj"], "adapter.target_modules"),
        ("task_type", "SEQ_CLS", "adapter.task_type"),
    ],
)
def test_manifest_is_cross_validated_against_peft_config(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    write_adapter_manifest(
        tmp_path,
        training_config={"target_modules": ["q_proj", "v_proj"]},
        peft_mutation=lambda config: config.update({field: value}),
    )
    with pytest.raises(ValueError, match=expected):
        validate(tmp_path, allow_unverified=True)


def test_peft_target_module_order_is_normalized(tmp_path: Path) -> None:
    write_adapter_manifest(
        tmp_path,
        training_config={"target_modules": ["q_proj", "v_proj"]},
        peft_mutation=lambda config: config.update(
            target_modules=["v_proj", "q_proj"]
        ),
    )
    assert validate(tmp_path).verified is True


def test_current_manifest_rejects_peft_base_model_mismatch(tmp_path: Path) -> None:
    write_adapter_manifest(
        tmp_path,
        peft_mutation=lambda config: config.update(
            base_model_name_or_path="other/model"
        ),
    )
    with pytest.raises(ValueError, match="does not match benchmark model"):
        validate(tmp_path, allow_unverified=True)


def test_current_manifest_rejects_recorded_peft_revision_mismatch(
    tmp_path: Path,
) -> None:
    write_adapter_manifest(
        tmp_path,
        peft_mutation=lambda config: config.update(revision="b" * 40),
    )
    with pytest.raises(ValueError, match="immutable benchmark base"):
        validate(tmp_path, allow_unverified=True)


def test_missing_optional_peft_defaults_are_normalized(tmp_path: Path) -> None:
    optional = {
        "modules_to_save",
        "use_rslora",
        "use_dora",
        "rank_pattern",
        "alpha_pattern",
        "layers_to_transform",
        "layers_pattern",
    }
    write_adapter_manifest(
        tmp_path,
        peft_mutation=lambda config: [config.pop(field, None) for field in optional],
    )
    assert validate(tmp_path).verified is True


def test_malformed_peft_config_is_never_overridden(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="PEFT config is unreadable"):
        validate(tmp_path, allow_unverified=True)


def test_malformed_schema_v3_is_never_overridden(tmp_path: Path) -> None:
    manifest = write_adapter_manifest(tmp_path)
    manifest.pop("training_controls")
    write_json(tmp_path / "raglab_adapter_manifest.json", manifest)
    with pytest.raises(ValueError, match="does not conform"):
        validate(tmp_path, allow_unverified=True)


def test_artifact_modification_is_never_overridden(tmp_path: Path) -> None:
    write_adapter_manifest(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact hash does not match"):
        validate(tmp_path, allow_unverified=True)


def test_training_control_hash_mismatch_is_never_overridden(tmp_path: Path) -> None:
    manifest = write_adapter_manifest(tmp_path)
    manifest["training_control_sha256"] = "f" * 64
    write_json(tmp_path / "raglab_adapter_manifest.json", manifest)
    with pytest.raises(ValueError, match="training-control hash"):
        validate(tmp_path, allow_unverified=True)
