from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class BenchmarkJob:
    recipe: str
    model_config: str
    documents: str
    eval_set: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def build_plan(
    *, recipes: list[str], model_config: str | Path, documents: str | Path, eval_set: str | Path
) -> list[BenchmarkJob]:
    return [
        BenchmarkJob(
            recipe=recipe,
            model_config=str(model_config),
            documents=str(documents),
            eval_set=str(eval_set),
        )
        for recipe in recipes
    ]
