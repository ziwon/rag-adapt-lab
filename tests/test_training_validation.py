from pathlib import Path

import pytest

from rag_adapt_lab.training.data import (
    deterministic_training_split,
    ensure_disjoint_training_rows,
    prompt_completion_records,
)
from rag_adapt_lab.training.qlora import _load_training_split as load_training_split
from rag_adapt_lab.training.qlora import build_sft_config_values


def rows(count: int = 10) -> list[dict[str, str]]:
    return [
        {"id": f"row-{index}", "question": f"Question {index}", "answer": f"Answer {index}"}
        for index in range(count)
    ]


def test_deterministic_validation_split_is_stable_and_disjoint() -> None:
    first = deterministic_training_split(rows(), validation_ratio=0.2, seed=11)
    second = deterministic_training_split(rows(), validation_ratio=0.2, seed=11)
    assert first == second
    assert len(first.train_rows) == 8
    assert len(first.validation_rows) == 2
    assert {row["id"] for row in first.train_rows}.isdisjoint(
        row["id"] for row in first.validation_rows
    )


def test_held_out_overlap_is_rejected_for_training_rows() -> None:
    with pytest.raises(ValueError, match="held-out benchmark"):
        ensure_disjoint_training_rows(
            [{"id": "train", "question": "What is RAG?"}],
            [{"id": "eval", "question": " what IS rag? "}],
            left_name="training data",
            right_name="held-out benchmark evaluation",
        )


def test_prompt_completion_records_keep_oracle_labels_out_of_prompt() -> None:
    records = prompt_completion_records(
        [
            {
                "id": "raft-1",
                "question": "Which evidence?",
                "answer": "the first",
                "contexts": [
                    {"text": "evidence", "relevant": True},
                    {"text": "noise", "relevant": False},
                ],
            }
        ],
        mode="raft",
        use_chat_template=False,
    )
    assert records[0]["completion"] == "the first"
    assert "| relevant" not in records[0]["prompt"].lower()
    assert "| distractor" not in records[0]["prompt"].lower()


def test_sft_configuration_uses_completion_only_loss_and_best_model() -> None:
    values = build_sft_config_values(
        {
            "eval_strategy": "steps",
            "eval_steps": 5,
            "save_steps": 10,
            "metric_for_best_model": "eval_loss",
        },
        output_dir=Path("outputs/test"),
        report_to_wandb=False,
        has_validation=True,
    )
    assert values["completion_only_loss"] is True
    assert values["load_best_model_at_end"] is True
    assert values["metric_for_best_model"] == "eval_loss"
    assert values["greater_is_better"] is False


def test_best_model_step_intervals_must_align() -> None:
    with pytest.raises(ValueError, match="multiple of eval_steps"):
        build_sft_config_values(
            {"eval_strategy": "steps", "eval_steps": 6, "save_steps": 10},
            output_dir=Path("outputs/test"),
            report_to_wandb=False,
            has_validation=True,
        )


def test_validation_cannot_be_the_held_out_benchmark_file(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "benchmark.jsonl"
    train_path.write_text(
        '{"id":"train","question":"training question","answer":"answer"}\n',
        encoding="utf-8",
    )
    validation_path.write_text(
        '{"id":"eval","question":"evaluation question","reference_answer":"answer"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be used as training validation"):
        load_training_split(
            train_file=train_path,
            validation_file=validation_path,
            held_out_eval_file=validation_path,
            validation_ratio=0.1,
            seed=42,
        )


def test_validation_requires_an_evaluation_schedule() -> None:
    with pytest.raises(ValueError, match="eval_strategy"):
        build_sft_config_values(
            {"eval_strategy": "no"},
            output_dir=Path("outputs/test"),
            report_to_wandb=False,
            has_validation=True,
        )
