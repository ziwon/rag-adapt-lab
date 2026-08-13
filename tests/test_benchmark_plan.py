import json
from pathlib import Path

from typer.testing import CliRunner

from rag_adapt_lab.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_qwen3_benchmark_plan_records_effective_non_thinking_args(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--recipes",
            "base",
            "--model-config",
            str(PROJECT_ROOT / "configs/models/qwen3-4b.yaml"),
            "--documents",
            str(PROJECT_ROOT / "examples/demo/documents.jsonl"),
            "--eval-set",
            str(PROJECT_ROOT / "examples/demo/eval.jsonl"),
            "--plan-output",
            str(plan),
            "--plan-only",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(plan.read_text(encoding="utf-8"))
    assert payload["schema_name"] == "benchmark-plan"
    assert payload["schema_version"] == 1
    assert payload["validation"]["validation_level"] == "structural"
    assert payload["validation"]["adapter_provenance_validated"] is False
    assert payload["fixed_contract"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert payload["fixed_contract"]["generation"] == {
        "max_new_tokens": 64,
        "do_sample": False,
    }
    assert payload["fixed_contract"]["prompt"]["template_sha256"]
