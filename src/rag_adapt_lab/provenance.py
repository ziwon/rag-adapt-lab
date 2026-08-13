from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag_adapt_lab.schema_validation import validate_artifact_schema

ADAPTER_MANIFEST_FILENAME = "raglab_adapter_manifest.json"
ADAPTER_MANIFEST_SCHEMA_VERSION = 3
TRAINING_MANIFEST_SCHEMA_VERSION = 3
BENCHMARK_SCHEMA_VERSION = 3

AdaptationMode = Literal["sft", "raft"]
_SHA256_HEX_LENGTH = 64
_REQUIRED_TRAINING_CONTROL_PATHS = (
    "adaptation_method",
    "adapter.rank",
    "adapter.alpha",
    "adapter.dropout",
    "adapter.target_modules",
    "adapter.bias",
    "adapter.task_type",
    "optimization.learning_rate",
    "optimization.optimizer",
    "optimization.optimizer_args",
    "optimization.adam_beta1",
    "optimization.adam_beta2",
    "optimization.adam_epsilon",
    "optimization.scheduler",
    "optimization.scheduler_kwargs",
    "optimization.weight_decay",
    "optimization.max_grad_norm",
    "optimization.num_train_epochs",
    "optimization.max_steps",
    "batching.per_device_train_batch_size",
    "batching.per_device_eval_batch_size",
    "batching.gradient_accumulation_steps",
    "batching.effective_batch_size",
    "batching.effective_batch_size_per_device",
    "sequence.max_length",
    "warmup.ratio",
    "warmup.steps",
    "gradient_checkpointing.enabled",
    "gradient_checkpointing.kwargs",
    "precision.bf16",
    "precision.fp16",
    "precision.tf32",
    "quantization.load_in_4bit",
    "quantization.type",
    "quantization.double_quantization",
    "quantization.compute_dtype",
    "seeds.training",
    "seeds.data",
    "loss.completion_only",
    "loss.label_smoothing_factor",
    "checkpoint_selection.validation_enabled",
    "checkpoint_selection.evaluation_strategy",
    "checkpoint_selection.evaluation_steps",
    "checkpoint_selection.save_strategy",
    "checkpoint_selection.save_steps",
    "checkpoint_selection.save_total_limit",
    "checkpoint_selection.load_best_model_at_end",
    "checkpoint_selection.metric_for_best_model",
    "checkpoint_selection.greater_is_better",
    "checkpoint_selection.early_stopping_metric",
    "checkpoint_selection.early_stopping_patience",
    "checkpoint_selection.early_stopping_threshold",
    "checkpoint_selection.best_model_selection_policy",
)


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
            "warnings": list(self.warnings),
            "manifest_schema_version": (
                self.manifest.get("schema_version") if self.manifest is not None else None
            ),
            "manifest_schema_name": (
                self.manifest.get("schema_name") if self.manifest is not None else None
            ),
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


def _validate_training_controls(controls: Mapping[str, Any]) -> None:
    missing: list[str] = []
    for path in _REQUIRED_TRAINING_CONTROL_PATHS:
        value: Any = controls
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                missing.append(path)
                break
            value = value[part]
    if missing:
        raise ValueError(
            "Adapter training_controls is missing normalized fields: "
            + ", ".join(missing)
        )
    if controls["adaptation_method"] not in {"lora", "qlora"}:
        raise ValueError("Adapter training_controls has an invalid adaptation_method")
    try:
        adapter = controls["adapter"]
        optimization = controls["optimization"]
        batching = controls["batching"]
        sequence = controls["sequence"]
        rank = int(adapter["rank"])
        alpha = int(adapter["alpha"])
        dropout = float(adapter["dropout"])
        learning_rate = float(optimization["learning_rate"])
        per_device_batch = int(batching["per_device_train_batch_size"])
        accumulation = int(batching["gradient_accumulation_steps"])
        effective_batch = int(batching["effective_batch_size"])
        effective_batch_per_device = int(batching["effective_batch_size_per_device"])
        max_length = int(sequence["max_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Adapter training_controls contains invalid normalized values") from exc
    if rank < 1 or alpha < 1 or not 0.0 <= dropout <= 1.0:
        raise ValueError("Adapter training_controls contains invalid LoRA controls")
    if learning_rate <= 0 or per_device_batch < 1 or accumulation < 1 or max_length < 1:
        raise ValueError("Adapter training_controls contains invalid optimization controls")
    expected_batch = per_device_batch * accumulation
    if effective_batch != expected_batch or effective_batch_per_device != expected_batch:
        raise ValueError(
            "Adapter training_controls effective_batch_size does not match batch size "
            "and gradient accumulation"
        )


def _verify_adapter_manifest(
    adapter: Path,
    manifest: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
    expected_mode: AdaptationMode | None,
    expected_prompt: Mapping[str, Any] | None,
    held_out_evaluation_sha256: str | None,
) -> AdapterVerification:
    validate_artifact_schema(manifest, "adapter-manifest-v3.schema.json")
    adapter_model = manifest.get("model")
    if not isinstance(adapter_model, Mapping):
        raise ValueError("Adapter manifest field 'model' must be a mapping")
    configured_id = adapter_model.get("model_id")
    configured_revision = adapter_model.get("revision")
    if configured_id != model_id or configured_revision != model_revision:
        raise ValueError(
            "Adapter identity does not match the immutable benchmark base: "
            f"adapter={configured_id}@{configured_revision}, "
            f"benchmark={model_id}@{model_revision}"
        )
    if manifest.get("schema_version") != ADAPTER_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Adapter manifest schema is unsupported; expected "
            f"{ADAPTER_MANIFEST_SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )

    mode = _required_string(manifest, "adaptation_mode")
    if mode not in {"sft", "raft"}:
        raise ValueError(f"Adapter manifest has unsupported adaptation_mode {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(
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
                raise ValueError(
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
            raise ValueError("Adapter chat-template arguments do not match the benchmark contract")
    training_controls = manifest.get("training_controls")
    if not isinstance(training_controls, Mapping) or not training_controls:
        raise ValueError("Adapter manifest field 'training_controls' must be a non-empty mapping")
    _validate_training_controls(training_controls)
    if canonical_sha256(training_controls) != manifest["training_control_sha256"]:
        raise ValueError("Adapter training-control hash does not match normalized controls")

    recorded_eval_hash = manifest["held_out_evaluation_sha256"]
    if (
        held_out_evaluation_sha256 is not None
        and recorded_eval_hash != held_out_evaluation_sha256
    ):
        raise ValueError(
            "Adapter leakage-check evaluation hash does not match the current held-out "
            f"evaluation file: adapter={recorded_eval_hash}, "
            f"benchmark={held_out_evaluation_sha256}"
        )

    computed_artifact_hash = artifact_sha256(adapter)
    if manifest["adapter_artifact_sha256"] != computed_artifact_hash:
        raise ValueError(
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
    """Validate an adapter manifest, failing closed unless explicitly overridden."""
    adapter = Path(adapter_path)
    if not adapter.exists():
        message = f"Adapter path does not exist: {adapter}"
        if not allow_unverified:
            raise FileNotFoundError(message)
        return AdapterVerification(
            str(adapter),
            False,
            None,
            None,
            warnings=(message,),
        )

    peft_config_path = adapter / "adapter_config.json"
    if peft_config_path.is_file():
        try:
            peft_config = json.loads(peft_config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Adapter PEFT config is unreadable: {peft_config_path}") from exc
        configured_base = peft_config.get("base_model_name_or_path")
        if configured_base and configured_base != model_id:
            raise ValueError(
                f"Adapter base model {configured_base!r} does not match benchmark model "
                f"{model_id!r}"
            )

    manifest_path = adapter / ADAPTER_MANIFEST_FILENAME
    try:
        if not manifest_path.is_file():
            raise ValueError(
                f"Adapter has no verifiable {ADAPTER_MANIFEST_FILENAME}; "
                "use --allow-unverified-adapter only for an explicitly labeled legacy run"
            )
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("Adapter manifest must be a JSON object")
        return _verify_adapter_manifest(
            adapter,
            raw,
            model_id=model_id,
            model_revision=model_revision,
            expected_mode=expected_mode,
            expected_prompt=expected_prompt,
            held_out_evaluation_sha256=held_out_evaluation_sha256,
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        if not allow_unverified:
            if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError):
                raise
            raise ValueError(f"Adapter manifest is unreadable: {manifest_path}") from exc
        artifact_hash: str | None
        try:
            artifact_hash = artifact_sha256(adapter)
        except (FileNotFoundError, ValueError, OSError):
            artifact_hash = None
        return AdapterVerification(
            path=str(adapter),
            verified=False,
            artifact_sha256=artifact_hash,
            adaptation_mode=None,
            warnings=(str(exc),),
        )
