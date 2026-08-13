import json
from pathlib import Path

import pytest

from rag_adapt_lab.benchmark.report import render_markdown_report
from rag_adapt_lab.data.schema import Document
from rag_adapt_lab.generation.prompts import rag_prompt_provenance
from rag_adapt_lab.provenance import (
    ADAPTER_MANIFEST_SCHEMA_VERSION,
    artifact_sha256,
    canonical_sha256,
    file_sha256,
    validate_adapter_provenance,
)
from rag_adapt_lab.recipes.plan import build_plan
from rag_adapt_lab.retrieval.factory import create_retriever
from rag_adapt_lab.training.controls import normalize_training_controls
from rag_adapt_lab.training.data import (
    prompt_completion_records,
    render_chat_prompt_completions,
)
from rag_adapt_lab.training.qlora import build_sft_config_values

pytestmark = pytest.mark.integration


def test_real_cpu_dependency_contracts(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")

    retriever = create_retriever({"kind": "bm25"})
    retriever.index(
        [Document(id="alpha", text="alpha evidence"), Document(id="beta", text="beta noise")]
    )
    assert retriever.search("alpha", top_k=1)[0].document.id == "alpha"

    tokenizer_model = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            vocab={"<unk>": 0, "<eos>": 1, "Question": 2, "Answer": 3},
            unk_token="<unk>",
        )
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )
    conversational = prompt_completion_records(
        [{"id": "sft", "input": "Question", "output": "EXPECTED_COMPLETION"}],
        mode="sft",
        use_chat_template=True,
    )
    processed = render_chat_prompt_completions(
        conversational,
        tokenizer=tokenizer,
        chat_template_kwargs={},
    )
    dataset = datasets.Dataset.from_list(processed)
    assert dataset.column_names == ["prompt", "completion"]
    assert "EXPECTED_COMPLETION" not in dataset[0]["prompt"]

    values = build_sft_config_values(
        {
            "bf16": False,
            "gradient_checkpointing": False,
            "eval_strategy": "steps",
            "eval_steps": 1,
            "save_steps": 1,
        },
        output_dir=tmp_path / "trainer",
        report_to_wandb=False,
        has_validation=True,
    )
    sft_config = trl.SFTConfig(**values)
    assert sft_config.completion_only_loss is True
    assert sft_config.eval_strategy.value == "steps"
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        completion_only_loss=True,
    )
    batch = collator([{"input_ids": [2, 3], "completion_mask": [0, 1]}])
    assert batch["labels"][0].tolist() == [-100, 3]

    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text('{"id":"eval","question":"held out"}\n', encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "test/model"}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"real-artifact-bytes")
    training_controls = normalize_training_controls({}, has_validation=True)
    manifest = {
        "schema_name": "raglab-adapter-manifest",
        "schema_version": ADAPTER_MANIFEST_SCHEMA_VERSION,
        "model": {"model_id": "test/model", "revision": "a" * 40},
        "recipe": "test-sft",
        "adaptation_mode": "sft",
        "training_prompt": rag_prompt_provenance(),
        "chat_template_kwargs": {},
        "training_dataset_fingerprint": "1" * 64,
        "validation_dataset_fingerprint": "2" * 64,
        "training_source_fingerprint": "4" * 64,
        "validation_source_fingerprint": "5" * 64,
        "held_out_evaluation_sha256": file_sha256(evaluation),
        "training_configuration_sha256": "3" * 64,
        "training_controls": training_controls,
        "training_control_sha256": canonical_sha256(training_controls),
        "adapter_artifact_sha256": artifact_sha256(adapter),
        "best_checkpoint": None,
        "best_validation_metric": None,
    }
    (adapter / "raglab_adapter_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    verification = validate_adapter_provenance(
        adapter,
        model_id="test/model",
        model_revision="a" * 40,
        expected_mode="sft",
        expected_prompt={**rag_prompt_provenance(), "chat_template_kwargs": {}},
        held_out_evaluation_sha256=file_sha256(evaluation),
    )
    assert verification.verified is True

    plan = build_plan(
        recipes=["base", "rag"],
        model_config="model.yaml",
        documents="documents.jsonl",
        eval_set="eval.jsonl",
    )
    assert [job.recipe for job in plan] == ["base", "rag"]
    report = render_markdown_report(
        {
            "recipes": {
                "base": {"metrics": {"exact_match": 0.0, "token_f1": 0.0}},
                "rag": {"metrics": {"exact_match": 1.0, "token_f1": 1.0}},
            },
            "comparisons": {},
            "retrieval_metrics": {},
            "configuration": {},
            "provenance": {"verified": True},
        }
    )
    assert "Base" in report and "RAG" in report
