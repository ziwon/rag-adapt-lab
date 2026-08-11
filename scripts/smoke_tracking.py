#!/usr/bin/env python
from __future__ import annotations

import json
import os


def require_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required. Configure W&B at http://localhost:8080, then add "
            "the value to .env.compose."
        )
    return value


def main() -> None:
    import wandb
    import weave

    base_url = require_environment("WANDB_BASE_URL")
    api_key = require_environment("WANDB_API_KEY")
    project = os.getenv("WANDB_PROJECT", "rag-adapt-lab")
    weave_project = os.getenv("WEAVE_PROJECT", project)

    wandb.login(host=base_url, key=api_key, relogin=True)
    run = wandb.init(project=project, name="compose-tracking-smoke")
    run.log({"compose/healthy": 1})
    run.finish()

    weave.init(weave_project)

    @weave.op()
    def compose_trace(value: str) -> str:
        return f"tracked:{value}"

    traced_result = compose_trace("ok")
    weave.finish()
    print(
        json.dumps(
            {
                "wandb_base_url": base_url,
                "wandb_project": project,
                "weave_project": weave_project,
                "weave_result": traced_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
