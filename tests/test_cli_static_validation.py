import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from rich.text import Text
from typer.testing import CliRunner

from rag_adapt_lab.cli import app
from rag_adapt_lab.generation.prompts import rag_prompt_provenance
from rag_adapt_lab.provenance import artifact_sha256, canonical_sha256, file_sha256
from rag_adapt_lab.training.controls import normalize_training_controls


def write_yaml(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def write_adapter(
    path: Path,
    *,
    mode: str,
    eval_path: Path,
    controls: dict[str, object],
    weight_bytes: bytes | None = None,
) -> dict[str, object]:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "test/model"}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(weight_bytes or mode.encode())
    manifest: dict[str, object] = {
        "schema_name": "raglab-adapter-manifest",
        "schema_version": 3,
        "model": {"model_id": "test/model", "revision": "0" * 40},
        "recipe": f"test-{mode}",
        "adaptation_mode": mode,
        "training_prompt": rag_prompt_provenance(),
        "chat_template_kwargs": {},
        "training_dataset_fingerprint": "1" * 64,
        "validation_dataset_fingerprint": "2" * 64,
        "training_source_fingerprint": "4" * 64,
        "validation_source_fingerprint": "5" * 64,
        "held_out_evaluation_sha256": file_sha256(eval_path),
        "training_configuration_sha256": "3" * 64,
        "training_controls": controls,
        "training_control_sha256": canonical_sha256(controls),
        "adapter_artifact_sha256": artifact_sha256(path),
        "best_checkpoint": None,
        "best_validation_metric": None,
    }
    (path / "raglab_adapter_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def static_fixture(
    tmp_path: Path,
    *,
    identical_artifacts: bool = False,
) -> tuple[list[str], dict[str, dict[str, object]]]:
    model = tmp_path / "model.yaml"
    documents = tmp_path / "documents.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    write_yaml(
        model,
        {
            "model_id": "test/model",
            "revision": "0" * 40,
            "trust_remote_code": False,
            "generation": {"max_new_tokens": 8, "do_sample": False},
        },
    )
    documents.write_text('{"id":"d","text":"answer"}\n', encoding="utf-8")
    evaluation.write_text(
        '{"id":"q","question":"question","reference_answer":"answer","relevant_doc_ids":["d"]}\n',
        encoding="utf-8",
    )
    controls = normalize_training_controls(
        {
            "load_in_4bit": False,
            "lora_r": 8,
            "learning_rate": 0.0001,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
        has_validation=True,
    )
    common_weights = b"identical" if identical_artifacts else None
    paths = {"sft": tmp_path / "sft", "raft": tmp_path / "raft"}
    manifests = {
        mode: write_adapter(
            paths[mode],
            mode=mode,
            eval_path=evaluation,
            controls=controls,
            weight_bytes=common_weights,
        )
        for mode in ("sft", "raft")
    }
    args = [
        "benchmark",
        "--recipes",
        "base,rag,sft-rag,raft-rag",
        "--model-config",
        str(model),
        "--documents",
        str(documents),
        "--eval-set",
        str(evaluation),
        "--sft-adapter",
        str(paths["sft"]),
        "--raft-adapter",
        str(paths["raft"]),
        "--output-dir",
        str(tmp_path / "output"),
        "--dry-run",
    ]
    return args, manifests


def rewrite_manifest(path: Path, manifest: dict[str, object]) -> None:
    (path / "raglab_adapter_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def plain_output(value: str) -> str:
    plain = Text.from_ansi(value).plain
    return " ".join(plain.replace("│", "").split())


def test_dry_run_performs_full_static_protocol_validation(tmp_path: Path) -> None:
    args, _ = static_fixture(tmp_path)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["validation"] == {
        "validation_level": "static-protocol",
        "model_weights_loaded": False,
        "adapter_provenance_validated": True,
        "training_controls_matched": True,
        "scorer_configuration_validated": True,
        "ready_for_execution": True,
    }


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda manifests: manifests["sft"].update(adaptation_mode="raft"), "adaptation mode"),
        (
            lambda manifests: manifests["sft"]["training_prompt"].update(version="old"),
            "prompt version",
        ),
        (
            lambda manifests: manifests["sft"].update(held_out_evaluation_sha256="f" * 64),
            "evaluation hash",
        ),
        (
            lambda manifests: manifests["raft"].update(training_source_fingerprint="6" * 64),
            "different underlying source partitions",
        ),
    ],
)
def test_dry_run_fails_protocol_mismatches(
    tmp_path: Path,
    mutation: Callable[[dict[str, dict[str, object]]], None],
    expected: str,
) -> None:
    args, manifests = static_fixture(tmp_path)
    mutation(manifests)
    rewrite_manifest(tmp_path / "sft", manifests["sft"])
    rewrite_manifest(tmp_path / "raft", manifests["raft"])
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert expected in plain_output(result.output)


def test_dry_run_fails_identical_adapter_artifacts(tmp_path: Path) -> None:
    args, _ = static_fixture(tmp_path, identical_artifacts=True)
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "same adapter artifact" in plain_output(result.output)


def test_dry_run_fails_changed_adapter_artifact_hash(tmp_path: Path) -> None:
    args, _ = static_fixture(tmp_path)
    (tmp_path / "sft" / "adapter_model.safetensors").write_bytes(b"changed-after-manifest")
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "artifact hash does not match" in plain_output(result.output)


def test_dry_run_fails_unmatched_training_controls(tmp_path: Path) -> None:
    args, manifests = static_fixture(tmp_path)
    controls = dict(manifests["raft"]["training_controls"])
    controls["adapter"] = {**controls["adapter"], "rank": 64}
    manifests["raft"]["training_controls"] = controls
    manifests["raft"]["training_control_sha256"] = canonical_sha256(controls)
    rewrite_manifest(tmp_path / "raft", manifests["raft"])
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "training controls differ" in plain_output(result.output)


def test_dry_run_fails_missing_adapter(tmp_path: Path) -> None:
    args, _ = static_fixture(tmp_path)
    missing_index = args.index("--sft-adapter") + 1
    args[missing_index] = str(tmp_path / "missing")
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "Adapter path does not exist" in plain_output(result.output)


def test_dry_run_fails_invalid_judge_without_contacting_endpoint(tmp_path: Path) -> None:
    args, _ = static_fixture(tmp_path)
    scorer = tmp_path / "scorer.yaml"
    write_yaml(
        scorer,
        {"judge": {"kind": "openai-compatible", "base_url": "http://127.0.0.1:1"}},
    )
    args.extend(["--scorer-config", str(scorer)])
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "non-empty model" in plain_output(result.output)


def test_dry_run_fails_invalid_output_destination(tmp_path: Path) -> None:
    args, _ = static_fixture(tmp_path)
    output = Path(args[args.index("--output-dir") + 1])
    output.write_text("not a directory", encoding="utf-8")
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "output path is not a directory" in plain_output(result.output)


def test_plan_only_allows_future_adapter_paths_and_labels_scope(tmp_path: Path) -> None:
    args, _ = static_fixture(tmp_path)
    args[args.index("--dry-run")] = "--plan-only"
    args[args.index("--sft-adapter") + 1] = str(tmp_path / "future-sft")
    args[args.index("--raft-adapter") + 1] = str(tmp_path / "future-raft")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["validation"]["validation_level"] == "structural"
    assert payload["validation"]["adapter_provenance_validated"] is False
    assert payload["validation"]["ready_for_execution"] is False
