from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def load_artifact_schema(filename: str) -> dict[str, Any]:
    resource = files("rag_adapt_lab.schemas").joinpath(filename)
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Artifact schema {filename!r} must be a JSON object")
    Draft202012Validator.check_schema(value)
    return value


def validate_artifact_schema(artifact: Any, filename: str) -> None:
    """Fail before publishing an artifact that violates its declared contract."""
    schema = load_artifact_schema(filename)
    try:
        Draft202012Validator(schema).validate(artifact)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ValueError(
            f"Artifact does not conform to {filename} at {location}: {exc.message}"
        ) from exc
