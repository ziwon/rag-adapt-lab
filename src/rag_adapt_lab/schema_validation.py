from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource


def load_artifact_schema(filename: str) -> dict[str, Any]:
    resource = files("rag_adapt_lab.schemas").joinpath(filename)
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Artifact schema {filename!r} must be a JSON object")
    Draft202012Validator.check_schema(value)
    return value


@lru_cache(maxsize=1)
def _artifact_schema_registry() -> Registry[Any]:
    registry: Registry[Any] = Registry()
    for resource in files("rag_adapt_lab.schemas").iterdir():
        if not resource.name.endswith(".schema.json"):
            continue
        schema = json.loads(resource.read_text(encoding="utf-8"))
        if isinstance(schema, dict) and isinstance(schema.get("$id"), str):
            registry = registry.with_resource(
                schema["$id"], Resource.from_contents(schema)
            )
    return registry


def validate_artifact_schema(artifact: Any, filename: str) -> None:
    """Fail before publishing an artifact that violates its declared contract."""
    schema = load_artifact_schema(filename)
    try:
        Draft202012Validator(schema, registry=_artifact_schema_registry()).validate(artifact)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ValueError(
            f"Artifact does not conform to {filename} at {location}: {exc.message}"
        ) from exc
