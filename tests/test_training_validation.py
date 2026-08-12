from pathlib import Path

import pytest

from rag_adapt_lab.training.data import (
    deterministic_training_split,
    ensure_disjoint_training_rows,
    prompt_completion_records,
    render_chat_prompt_completions,
)
from rag_adapt_lab.training.qlora import _load_training_split as load_training_split
from rag_adapt_lab.training.qlora import (
    build_sft_config_values,
    require_verifiable_training_prompt,
)


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


class FakeChatTokenizer:
    chat_template = "configured"
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        assert kwargs["enable_thinking"] is False
        assert kwargs["tokenize"] is False
        return f"USER:{conversation[0]['content']}\nASSISTANT:"


def test_explicit_chat_rendering_preserves_completion_only_boundary() -> None:
    records = prompt_completion_records(
        [{"id": "sft", "input": "Question?", "output": "Answer."}],
        mode="sft",
        use_chat_template=True,
    )
    rendered = render_chat_prompt_completions(
        records,
        tokenizer=FakeChatTokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    assert "Answer." not in rendered[0]["prompt"]
    assert rendered[0]["completion"] == "Answer.<eos>"
    assert "(no documents provided)" in rendered[0]["prompt"]


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
    assert "chat_template_kwargs" not in values  # TRL 0.24.0 has no such SFTConfig field.


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


def test_verifiable_training_requires_the_inference_chat_template() -> None:
    with pytest.raises(ValueError, match="use_chat_template=true"):
        require_verifiable_training_prompt({"use_chat_template": False})
    require_verifiable_training_prompt({"use_chat_template": True})


def test_explicit_validation_enforces_custom_group_fields(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(
        '{"id":"train","question":"train q","answer":"a",'
        '"metadata":{"thread":"shared"}}\n',
        encoding="utf-8",
    )
    validation_path.write_text(
        '{"id":"validation","question":"validation q","answer":"a",'
        '"metadata":{"thread":"shared"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="configured grouping values"):
        load_training_split(
            train_file=train_path,
            validation_file=validation_path,
            held_out_eval_file=None,
            validation_ratio=0.1,
            seed=42,
            split_config={"strategy": "grouped", "group_by": ["metadata.thread"]},
        )


def test_nested_split_ratio_and_seed_override_legacy_top_level_values(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        "".join(
            f'{{"id":"row-{index}","question":"q {index}","answer":"a"}}\n'
            for index in range(10)
        ),
        encoding="utf-8",
    )
    split = load_training_split(
        train_file=train_path,
        validation_file=None,
        held_out_eval_file=None,
        validation_ratio=0.1,
        seed=1,
        split_config={"validation_ratio": 0.3, "seed": 99},
    )
    assert split.validation_ratio == 0.3
    assert split.seed == 99
    assert len(split.validation_rows) == 3
