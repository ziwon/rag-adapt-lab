from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag_adapt_lab.schema_validation import validate_artifact_schema
from rag_adapt_lab.training.controls import (
    training_control_sha256,
    validate_training_controls,
)

ADAPTER_MANIFEST_FILENAME = "raglab_adapter_manifest.json"
ADAPTER_MANIFEST_SCHEMA_VERSION = 3
TRAINING_MANIFEST_SCHEMA_VERSION = 3
BENCHMARK_SCHEMA_VERSION = 3

AdaptationMode = Literal["sft", "raft"]
_SHA256_HEX_LENGTH = 64


class AdapterProvenanceError(ValueError):
    """Base class for adapter identity, provenance, and integrity failures."""


class LegacyManifestUnavailable(AdapterProvenanceError):
    """The adapter predates the manifest contract and has no manifest."""


class UnsupportedLegacyManifest(AdapterProvenanceError):
    """A recognized legacy manifest cannot prove the current contract."""


class ProvenanceMismatchError(AdapterProvenanceError):
    """Recorded provenance contradicts the requested benchmark contract."""


class ArtifactIntegrityError(AdapterProvenanceError):
    """Adapter files no longer match the immutable artifact identity."""


class CurrentManifestSchemaError(AdapterProvenanceError):
    """A present manifest violates its declared machine-readable schema."""


class PeftConfigurationError(AdapterProvenanceError):
    """The persisted PEFT configuration is missing, unreadable, or contradictory."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_sha256(path: str | Path) -> str:
    """Hash adapter artifacts without creating a manifest self-reference."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {source}")
    if source.is_file():
        return file_sha256(source)
    digest = hashlib.sha256()
    children = sorted(
        item
        for item in source.rglob("*")
        if item.is_file() and item.name != ADAPTER_MANIFEST_FILENAME
    )
    if not children:
        raise ValueError(f"Adapter directory contains no hashable artifacts: {source}")
    for child in children:
        digest.update(str(child.relative_to(source)).encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AdapterVerification:
    path: str
    verified: bool
    artifact_sha256: str | None
    adaptation_mode: str | None
    training_source_fingerprint: str | None = None
    validation_source_fingerprint: str | None = None
    training_control_sha256: str | None = None
    training_controls: Mapping[str, Any] | None = None
    status: str = "verified"
    reason_code: str | None = None
    unchecked_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    manifest: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "verified": self.verified,
            "artifact_sha256": self.artifact_sha256,
            "adaptation_mode": self.adaptation_mode,
            "training_source_fingerprint": self.training_source_fingerprint,
            "validation_source_fingerprint": self.validation_source_fingerprint,
            "training_control_sha256": self.training_control_sha256,
            "training_controls": dict(self.training_controls)
            if self.training_controls is not None
            else None,
            "status": self.status,
            "reason_code": self.reason_code,
            "unchecked_fields": list(self.unchecked_fields),
            "warnings": list(self.warnings),
            "manifest_schema_version": (
                self.manifest.get("schema_version") if self.manifest is not None else None
            ),
            "manifest_schema_name": (
                self.manifest.get("schema_name") if self.manifest is not None else None
            ),
        }


def _normalize_string_collection(
    value: Any,
    *,
    field: str,
    allow_string: bool,
) -> str | list[str] | None:
    if value is None:
        return None
    if allow_string and isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise PeftConfigurationError(f"PEFT field {field!r} must not be blank")
        return normalized
    if not isinstance(value, (list, tuple, set)):
        raise PeftConfigurationError(f"PEFT field {field!r} must be a list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise PeftConfigurationError(
            f"PEFT field {field!r} must contain only non-empty strings"
        )
    normalized_values = sorted({item.strip() for item in value})
    if not normalized_values:
        raise PeftConfigurationError(f"PEFT field {field!r} must not be empty")
    return normalized_values


def _normalize_integer_pattern(value: Any, *, field: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PeftConfigurationError(f"PEFT field {field!r} must be an object")
    if not all(
        isinstance(key, str)
        and key
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 1
        for key, item in value.items()
    ):
        raise PeftConfigurationError(
            f"PEFT field {field!r} requires non-empty keys and positive values"
        )
    normalized = {key: item for key, item in value.items()}
    return dict(sorted(normalized.items()))


def normalize_peft_adapter_config(
    peft_config: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Create a canonical, comparison-safe view of a persisted PEFT config."""
    model_id = peft_config.get("base_model_name_or_path")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PeftConfigurationError(
            "PEFT adapter_config.json requires base_model_name_or_path"
        )
    required = ("peft_type", "task_type", "r", "lora_alpha", "lora_dropout", "bias", "target_modules")
    missing = [field for field in required if field not in peft_config]
    if require_complete and missing:
        raise PeftConfigurationError(
            "PEFT adapter_config.json is missing required fields: " + ", ".join(missing)
        )

    adapter: dict[str, Any] = {}
    field_mappings = {
        "peft_type": "peft_type",
        "task_type": "task_type",
        "r": "rank",
        "lora_alpha": "alpha",
        "lora_dropout": "dropout",
        "bias": "bias",
    }
    for peft_field, normalized_field in field_mappings.items():
        if peft_field not in peft_config:
            continue
        value = peft_config[peft_field]
        if normalized_field in {"rank", "alpha"}:
            if not isinstance(value, int) or isinstance(value, bool):
                raise PeftConfigurationError(
                    f"PEFT field {peft_field!r} must be an integer"
                )
        elif normalized_field == "dropout":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PeftConfigurationError(
                    f"PEFT field {peft_field!r} must be numeric"
                )
            value = float(value)
        elif normalized_field in {"peft_type", "task_type"}:
            if not isinstance(value, str) or not value.strip():
                raise PeftConfigurationError(
                    f"PEFT field {peft_field!r} must be a non-empty string"
                )
            value = value.upper()
        elif normalized_field == "bias":
            if not isinstance(value, str) or not value.strip():
                raise PeftConfigurationError(
                    "PEFT field 'bias' must be a non-empty string"
                )
            value = value.lower()
        adapter[normalized_field] = value
    if "target_modules" in peft_config:
        adapter["target_modules"] = _normalize_string_collection(
            peft_config["target_modules"], field="target_modules", allow_string=True
        )

    optional_defaults: dict[str, Any] = {
        "modules_to_save": None,
        "use_rslora": False,
        "use_dora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "layers_to_transform": None,
        "layers_pattern": None,
    }
    for field, default in optional_defaults.items():
        if not require_complete and field not in peft_config:
            continue
        value = peft_config.get(field, default)
        if field == "modules_to_save":
            value = _normalize_string_collection(
                value, field=field, allow_string=False
            )
        elif field in {"rank_pattern", "alpha_pattern"}:
            value = _normalize_integer_pattern(value, field=field)
        elif field == "layers_to_transform" and value is not None:
            if isinstance(value, int):
                if isinstance(value, bool) or value < 0:
                    raise PeftConfigurationError(
                        "PEFT field 'layers_to_transform' must contain non-negative integers"
                    )
                value = value
            elif isinstance(value, (list, tuple, set)):
                if not all(
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    and item >= 0
                    for item in value
                ):
                    raise PeftConfigurationError(
                        "PEFT field 'layers_to_transform' must contain non-negative integers"
                    )
                value = sorted(set(value))
            else:
                raise PeftConfigurationError(
                    "PEFT field 'layers_to_transform' must be an integer or list"
                )
        elif field == "layers_pattern":
            value = _normalize_string_collection(
                value, field=field, allow_string=True
            )
        elif field in {"use_rslora", "use_dora"}:
            if not isinstance(value, bool):
                raise PeftConfigurationError(f"PEFT field {field!r} must be boolean")
        adapter[field] = value

    revision = peft_config.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise PeftConfigurationError("PEFT field 'revision' must be a string or null")
    return {
        "model": {
            "model_id": model_id.strip(),
            "revision": revision.strip() if isinstance(revision, str) and revision.strip() else None,
        },
        "adapter": adapter,
    }


def _required_string(manifest: Mapping[str, Any], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Adapter manifest field {field!r} must be a non-empty string")
    return value


def _required_sha256(manifest: Mapping[str, Any], field: str) -> str:
    value = _required_string(manifest, field)
    if len(value) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Adapter manifest field {field!r} must be a lowercase SHA-256 hex digest")
    return value


def _assert_manifest_peft_agreement(
    manifest: Mapping[str, Any],
    peft_view: Mapping[str, Any],
) -> None:
    manifest_model = manifest["model"]
    manifest_adapter = manifest["training_controls"]["adapter"]
    peft_model = peft_view["model"]
    peft_adapter = peft_view["adapter"]
    mismatches: list[str] = []
    if manifest_model["model_id"] != peft_model["model_id"]:
        mismatches.append(
            f"model.model_id: manifest={manifest_model['model_id']!r}, "
            f"peft={peft_model['model_id']!r}"
        )
    if (
        peft_model.get("revision") is not None
        and manifest_model["revision"] != peft_model["revision"]
    ):
        mismatches.append(
            f"model.revision: manifest={manifest_model['revision']!r}, "
            f"peft={peft_model['revision']!r}"
        )
    for field in (
        "peft_type",
        "task_type",
        "rank",
        "alpha",
        "dropout",
        "bias",
        "target_modules",
        "modules_to_save",
        "use_rslora",
        "use_dora",
        "rank_pattern",
        "alpha_pattern",
        "layers_to_transform",
        "layers_pattern",
    ):
        if manifest_adapter[field] != peft_adapter[field]:
            mismatches.append(
                f"adapter.{field}: manifest={manifest_adapter[field]!r}, "
                f"peft={peft_adapter[field]!r}"
            )
    if mismatches:
        raise PeftConfigurationError(
            "Adapter manifest contradicts adapter_config.json:\n" + "\n".join(mismatches)
        )


def _verify_adapter_manifest(
    adapter: Path,
    manifest: Mapping[str, Any],
    *,
    peft_view: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    expected_mode: AdaptationMode | None,
    expected_prompt: Mapping[str, Any] | None,
    held_out_evaluation_sha256: str | None,
) -> AdapterVerification:
    try:
        validate_artifact_schema(manifest, "adapter-manifest-v3.schema.json")
    except ValueError as exc:
        raise CurrentManifestSchemaError(str(exc)) from exc
    adapter_model = manifest.get("model")
    if not isinstance(adapter_model, Mapping):
        raise ValueError("Adapter manifest field 'model' must be a mapping")
    configured_id = adapter_model.get("model_id")
    configured_revision = adapter_model.get("revision")
    if configured_id != model_id or configured_revision != model_revision:
        raise ProvenanceMismatchError(
            "Adapter identity does not match the immutable benchmark base: "
            f"adapter={configured_id}@{configured_revision}, "
            f"benchmark={model_id}@{model_revision}"
        )
    if manifest.get("schema_version") != ADAPTER_MANIFEST_SCHEMA_VERSION:
        raise CurrentManifestSchemaError(
            "Adapter manifest schema is unsupported; expected "
            f"{ADAPTER_MANIFEST_SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )

    mode = _required_string(manifest, "adaptation_mode")
    if mode not in {"sft", "raft"}:
        raise CurrentManifestSchemaError(
            f"Adapter manifest has unsupported adaptation_mode {mode!r}"
        )
    if expected_mode is not None and mode != expected_mode:
        raise ProvenanceMismatchError(
            f"Adapter adaptation mode {mode!r} does not match expected mode {expected_mode!r}"
        )

    prompt = manifest.get("training_prompt")
    if not isinstance(prompt, Mapping):
        raise ValueError("Adapter manifest field 'training_prompt' must be a mapping")
    for field in ("name", "version", "template_sha256"):
        _required_string(prompt, field)
    _required_sha256(prompt, "template_sha256")
    if expected_prompt is not None:
        for field in ("name", "version", "template_sha256"):
            if prompt.get(field) != expected_prompt.get(field):
                raise ProvenanceMismatchError(
                    f"Adapter training prompt {field} {prompt.get(field)!r} does not match "
                    f"benchmark contract {expected_prompt.get(field)!r}"
                )

    for field in (
        "training_dataset_fingerprint",
        "validation_dataset_fingerprint",
        "training_source_fingerprint",
        "validation_source_fingerprint",
        "held_out_evaluation_sha256",
        "training_configuration_sha256",
        "training_control_sha256",
        "adapter_artifact_sha256",
    ):
        _required_sha256(manifest, field)
    chat_kwargs = manifest.get("chat_template_kwargs")
    if not isinstance(chat_kwargs, Mapping):
        raise ValueError("Adapter manifest field 'chat_template_kwargs' must be a mapping")
    if expected_prompt is not None and "chat_template_kwargs" in expected_prompt:
        if dict(chat_kwargs) != dict(expected_prompt["chat_template_kwargs"]):
            raise ProvenanceMismatchError(
                "Adapter chat-template arguments do not match the benchmark contract"
            )
    training_controls = manifest.get("training_controls")
    if not isinstance(training_controls, Mapping) or not training_controls:
        raise CurrentManifestSchemaError(
            "Adapter manifest field 'training_controls' must be a non-empty mapping"
        )
    try:
        validate_training_controls(training_controls)
    except ValueError as exc:
        raise CurrentManifestSchemaError(
            f"Adapter manifest has invalid training_controls: {exc}"
        ) from exc
    if training_control_sha256(training_controls) != manifest["training_control_sha256"]:
        raise ArtifactIntegrityError(
            "Adapter training-control hash does not match normalized controls"
        )
    _assert_manifest_peft_agreement(manifest, peft_view)

    recorded_eval_hash = manifest["held_out_evaluation_sha256"]
    if (
        held_out_evaluation_sha256 is not None
        and recorded_eval_hash != held_out_evaluation_sha256
    ):
        raise ProvenanceMismatchError(
            "Adapter leakage-check evaluation hash does not match the current held-out "
            f"evaluation file: adapter={recorded_eval_hash}, "
            f"benchmark={held_out_evaluation_sha256}"
        )

    computed_artifact_hash = artifact_sha256(adapter)
    if manifest["adapter_artifact_sha256"] != computed_artifact_hash:
        raise ArtifactIntegrityError(
            "Adapter artifact hash does not match its manifest; the adapter may have been changed"
        )
    return AdapterVerification(
        path=str(adapter),
        verified=True,
        artifact_sha256=computed_artifact_hash,
        adaptation_mode=mode,
        training_source_fingerprint=manifest["training_source_fingerprint"],
        validation_source_fingerprint=manifest["validation_source_fingerprint"],
        training_control_sha256=manifest["training_control_sha256"],
        training_controls=training_controls,
        manifest=manifest,
    )


def _verify_legacy_v2_manifest(
    adapter: Path,
    manifest: Mapping[str, Any],
    *,
    peft_view: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    expected_mode: AdaptationMode | None,
    expected_prompt: Mapping[str, Any] | None,
    held_out_evaluation_sha256: str | None,
) -> AdapterVerification:
    try:
        validate_artifact_schema(manifest, "adapter-manifest-v2.schema.json")
    except ValueError as exc:
        raise CurrentManifestSchemaError(
            f"Legacy adapter manifest is malformed: {exc}"
        ) from exc
    adapter_model = manifest["model"]
    if (
        adapter_model["model_id"] != model_id
        or adapter_model["revision"] != model_revision
    ):
        raise ProvenanceMismatchError(
            "Adapter identity does not match the immutable benchmark base: "
            f"adapter={adapter_model['model_id']}@{adapter_model['revision']}, "
            f"benchmark={model_id}@{model_revision}"
        )
    if adapter_model["model_id"] != peft_view["model"]["model_id"]:
        raise PeftConfigurationError(
            "Legacy adapter manifest base model contradicts adapter_config.json"
        )
    peft_revision = peft_view["model"].get("revision")
    if peft_revision is not None and adapter_model["revision"] != peft_revision:
        raise PeftConfigurationError(
            "Legacy adapter manifest revision contradicts adapter_config.json"
        )
    mode = str(manifest["adaptation_mode"])
    if expected_mode is not None and mode != expected_mode:
        raise ProvenanceMismatchError(
            f"Adapter adaptation mode {mode!r} does not match expected mode {expected_mode!r}"
        )
    prompt = manifest["training_prompt"]
    if expected_prompt is not None:
        for field in ("name", "version", "template_sha256"):
            if prompt[field] != expected_prompt.get(field):
                raise ProvenanceMismatchError(
                    f"Adapter training prompt {field} {prompt[field]!r} does not match "
                    f"benchmark contract {expected_prompt.get(field)!r}"
                )
        if dict(manifest["chat_template_kwargs"]) != dict(
            expected_prompt.get("chat_template_kwargs", {})
        ):
            raise ProvenanceMismatchError(
                "Adapter chat-template arguments do not match the benchmark contract"
            )
    if (
        held_out_evaluation_sha256 is not None
        and manifest["held_out_evaluation_sha256"] != held_out_evaluation_sha256
    ):
        raise ProvenanceMismatchError(
            "Adapter leakage-check evaluation hash does not match the current held-out "
            "evaluation file"
        )
    computed_artifact_hash = artifact_sha256(adapter)
    if manifest["adapter_artifact_sha256"] != computed_artifact_hash:
        raise ArtifactIntegrityError(
            "Adapter artifact hash does not match its manifest; the adapter may have been changed"
        )
    warning = (
        "Adapter uses schema v2: source-partition and normalized training-control "
        "provenance are unavailable."
    )
    return AdapterVerification(
        path=str(adapter),
        verified=False,
        artifact_sha256=computed_artifact_hash,
        adaptation_mode=mode,
        status="unverified_legacy_provenance",
        reason_code="legacy_training_controls_unavailable",
        unchecked_fields=(
            "training_source_fingerprint",
            "validation_source_fingerprint",
            "training_controls",
        ),
        warnings=(warning,),
        manifest=manifest,
    )


def validate_adapter_provenance(
    adapter_path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    expected_mode: AdaptationMode | None = None,
    expected_prompt: Mapping[str, Any] | None = None,
    held_out_evaluation_sha256: str | None = None,
    allow_unverified: bool = False,
) -> AdapterVerification:
    """Validate adapter identity; override only unavailable legacy provenance."""
    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter}")
    if not adapter.is_dir():
        raise PeftConfigurationError(f"Adapter path must be a directory: {adapter}")

    peft_config_path = adapter / "adapter_config.json"
    if not peft_config_path.is_file():
        raise PeftConfigurationError(
            f"Adapter has no PEFT adapter_config.json: {peft_config_path}"
        )
    try:
        peft_config = json.loads(peft_config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PeftConfigurationError(
            f"Adapter PEFT config is unreadable: {peft_config_path}"
        ) from exc
    if not isinstance(peft_config, Mapping):
        raise PeftConfigurationError("Adapter PEFT config must be a JSON object")
    partial_peft_view = normalize_peft_adapter_config(
        peft_config, require_complete=False
    )
    configured_base = partial_peft_view["model"]["model_id"]
    if configured_base != model_id:
        raise ProvenanceMismatchError(
            f"Adapter base model {configured_base!r} does not match benchmark model "
            f"{model_id!r}"
        )
    configured_revision = partial_peft_view["model"].get("revision")
    if configured_revision is not None and configured_revision != model_revision:
        raise ProvenanceMismatchError(
            "Adapter identity does not match the immutable benchmark base: "
            f"PEFT revision={configured_revision!r}, benchmark revision={model_revision!r}"
        )

    manifest_path = adapter / ADAPTER_MANIFEST_FILENAME
    if not manifest_path.is_file():
        error = LegacyManifestUnavailable(
            f"Adapter has no verifiable {ADAPTER_MANIFEST_FILENAME}; "
            "use --allow-unverified-adapter only for an explicitly labeled legacy run"
        )
        if not allow_unverified:
            raise error
        return AdapterVerification(
            path=str(adapter),
            verified=False,
            artifact_sha256=artifact_sha256(adapter),
            adaptation_mode=None,
            status="legacy_manifest_unavailable",
            reason_code="manifest_missing",
            unchecked_fields=(
                "adaptation_mode",
                "training_prompt",
                "held_out_evaluation_sha256",
                "source_partitions",
                "training_controls",
            ),
            warnings=("Adapter has no schema-v3 provenance manifest.",),
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CurrentManifestSchemaError(
            f"Adapter manifest is unreadable: {manifest_path}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise CurrentManifestSchemaError("Adapter manifest must be a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version == ADAPTER_MANIFEST_SCHEMA_VERSION:
        peft_view = normalize_peft_adapter_config(peft_config, require_complete=True)
        return _verify_adapter_manifest(
            adapter,
            raw,
            peft_view=peft_view,
            model_id=model_id,
            model_revision=model_revision,
            expected_mode=expected_mode,
            expected_prompt=expected_prompt,
            held_out_evaluation_sha256=held_out_evaluation_sha256,
        )
    if schema_version == 2:
        result = _verify_legacy_v2_manifest(
            adapter,
            raw,
            peft_view=partial_peft_view,
            model_id=model_id,
            model_revision=model_revision,
            expected_mode=expected_mode,
            expected_prompt=expected_prompt,
            held_out_evaluation_sha256=held_out_evaluation_sha256,
        )
        if not allow_unverified:
            raise UnsupportedLegacyManifest(
                "Adapter schema v2 lacks source-partition and training-control provenance; "
                "use --allow-unverified-adapter only for a labeled legacy run"
            )
        return result
    raise CurrentManifestSchemaError(
        f"Adapter manifest schema_version {schema_version!r} is not a recognized contract"
    )
