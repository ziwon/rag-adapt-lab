from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RECIPE_RETRIEVAL = {
    "base": False,
    "rag": True,
    "sft-rag": True,
    "raft-rag": True,
}


@dataclass(slots=True)
class BenchmarkJob:
    recipe: str
    model_config: str
    documents: str
    eval_set: str
    use_retrieval: bool
    adapter_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_plan(
    *,
    recipes: list[str],
    model_config: str | Path,
    documents: str | Path,
    eval_set: str | Path,
    adapters: Mapping[str, str | Path | None] | None = None,
) -> list[BenchmarkJob]:
    if not recipes:
        raise ValueError("At least one benchmark recipe is required")
    unknown = sorted(set(recipes) - set(RECIPE_RETRIEVAL))
    if unknown:
        raise ValueError(f"Unknown recipes: {unknown}")
    if len(recipes) != len(set(recipes)):
        raise ValueError("Benchmark recipes must be unique")
    adapter_paths = adapters or {}
    return [
        BenchmarkJob(
            recipe=recipe,
            model_config=str(model_config),
            documents=str(documents),
            eval_set=str(eval_set),
            use_retrieval=RECIPE_RETRIEVAL[recipe],
            adapter_path=(
                str(adapter_paths[recipe]) if adapter_paths.get(recipe) is not None else None
            ),
        )
        for recipe in recipes
    ]
