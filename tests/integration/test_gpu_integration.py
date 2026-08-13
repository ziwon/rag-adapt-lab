import json
import os
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_yaml(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


@pytest.mark.skipif(
    os.getenv("RAGLAB_RUN_GPU_INTEGRATION") != "1",
    reason="requires the explicit GPU integration workflow",
)
def test_tiny_real_lora_save_load_generation_memory_and_benchmark(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.fail("GPU integration was requested but CUDA is unavailable")

    from rag_adapt_lab.benchmark.runner import BenchmarkRunner, TransformersGeneratorFactory
    from rag_adapt_lab.data.io import load_documents, load_eval
    from rag_adapt_lab.evaluation.scorers import build_scorer
    from rag_adapt_lab.recipes.plan import build_plan
    from rag_adapt_lab.retrieval.factory import create_retriever
    from rag_adapt_lab.schema_validation import validate_artifact_schema
    from rag_adapt_lab.training.qlora import train_qlora

    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    vocabulary = {
        "<unk>": 0,
        "<eos>": 1,
        "user": 2,
        "assistant": 3,
        "alpha": 4,
        "beta": 5,
        "question": 6,
        "answer": 7,
    }
    tokenizer_model = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocabulary, unk_token="<unk>")
    )
    tokenizer_model.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    tokenizer.save_pretrained(model_dir)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    model.save_pretrained(model_dir)

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    model_config_path = config_dir / "model.yaml"
    training_config_path = config_dir / "training.yaml"
    write_yaml(
        model_config_path,
        {
            "model_id": str(model_dir),
            "revision": "0" * 40,
            "trust_remote_code": False,
            "torch_dtype": "float32",
            "max_seq_length": 64,
            "generation": {"max_new_tokens": 4, "do_sample": False},
        },
    )
    write_yaml(
        training_config_path,
        {
            "load_in_4bit": False,
            "use_chat_template": True,
            "validation_split_ratio": 0.25,
            "eval_strategy": "epoch",
            "eval_steps": 1,
            "save_steps": 1,
            "save_total_limit": 1,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "target_modules": ["c_attn"],
            "lora_r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "learning_rate": 0.001,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 64,
            "logging_steps": 1,
            "gradient_checkpointing": False,
            "bf16": False,
            "seed": 7,
            "split": {
                "strategy": "grouped",
                "group_by": ["normalized_question"],
                "corpus_policy": "shared-corpus",
            },
        },
    )

    documents_path = tmp_path / "documents.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(
        documents_path,
        [
            {"id": "alpha", "text": "alpha answer"},
            {"id": "beta", "text": "beta answer"},
        ],
    )
    write_jsonl(
        eval_path,
        [
            {
                "id": "held-out",
                "question": "alpha question",
                "reference_answer": "alpha",
                "relevant_doc_ids": ["alpha"],
            }
        ],
    )
    sft_train = tmp_path / "sft-train.jsonl"
    sft_validation = tmp_path / "sft-validation.jsonl"
    raft_train = tmp_path / "raft-train.jsonl"
    raft_validation = tmp_path / "raft-validation.jsonl"
    source_train = [
        {"id": "q1", "question": "question alpha one", "answer": "alpha"},
        {"id": "q2", "question": "question beta two", "answer": "beta"},
    ]
    source_validation = [
        {"id": "q3", "question": "question alpha validation", "answer": "alpha"}
    ]
    write_jsonl(
        sft_train,
        [
            {"id": row["id"], "input": row["question"], "output": row["answer"]}
            for row in source_train
        ],
    )
    write_jsonl(
        sft_validation,
        [
            {"id": row["id"], "input": row["question"], "output": row["answer"]}
            for row in source_validation
        ],
    )
    raft_contexts = [
        {"doc_id": "alpha", "text": "alpha answer", "relevant": True},
        {"doc_id": "beta", "text": "beta answer", "relevant": False},
    ]
    write_jsonl(
        raft_train,
        [
            {
                **row,
                "contexts": raft_contexts if row["answer"] == "alpha" else list(reversed(raft_contexts)),
                "evidence_doc_ids": [row["answer"]],
            }
            for row in source_train
        ],
    )
    write_jsonl(
        raft_validation,
        [
            {
                **row,
                "contexts": raft_contexts,
                "evidence_doc_ids": ["alpha"],
            }
            for row in source_validation
        ],
    )

    adapters: dict[str, Path] = {}
    for mode, train_file, validation_file in (
        ("sft", sft_train, sft_validation),
        ("raft", raft_train, raft_validation),
    ):
        recipe_path = config_dir / f"{mode}.yaml"
        write_yaml(
            recipe_path,
            {
                "name": f"tiny-{mode}",
                "model": str(model_config_path),
                "training": {
                    "enabled": True,
                    "mode": mode,
                    "config": str(training_config_path),
                },
                "output_dir": str(tmp_path / f"output-{mode}"),
                "tracking": {"backend": "null"},
            },
        )
        adapters[mode] = train_qlora(
            recipe_config=recipe_path,
            train_file=train_file,
            validation_file=validation_file,
            held_out_eval_file=eval_path,
        )
        assert (adapters[mode] / "adapter_model.safetensors").is_file()

    adapter_manifests = {
        mode: json.loads(
            (adapter / "raglab_adapter_manifest.json").read_text(encoding="utf-8")
        )
        for mode, adapter in adapters.items()
    }
    assert (
        adapter_manifests["sft"]["training_source_fingerprint"]
        == adapter_manifests["raft"]["training_source_fingerprint"]
    )
    assert (
        adapter_manifests["sft"]["validation_source_fingerprint"]
        == adapter_manifests["raft"]["validation_source_fingerprint"]
    )
    assert (
        adapter_manifests["sft"]["training_control_sha256"]
        == adapter_manifests["raft"]["training_control_sha256"]
    )
    for field in ("model", "training_prompt", "chat_template_kwargs", "training_controls"):
        assert adapter_manifests["sft"][field] == adapter_manifests["raft"][field]
    assert (
        adapter_manifests["sft"]["training_dataset_fingerprint"]
        != adapter_manifests["raft"]["training_dataset_fingerprint"]
    )
    assert (
        adapter_manifests["sft"]["adapter_artifact_sha256"]
        != adapter_manifests["raft"]["adapter_artifact_sha256"]
    )
    for mode, adapter_manifest in adapter_manifests.items():
        validate_artifact_schema(adapter_manifest, "adapter-manifest-v3.schema.json")
        training_manifest = json.loads(
            (tmp_path / f"output-{mode}" / "training_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        validate_artifact_schema(training_manifest, "training-manifest-v3.schema.json")

    documents = load_documents(documents_path)
    examples = load_eval(eval_path)
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    jobs = build_plan(
        recipes=["base", "rag", "sft-rag", "raft-rag"],
        model_config=model_config_path,
        documents=documents_path,
        eval_set=eval_path,
        adapters={"sft-rag": adapters["sft"], "raft-rag": adapters["raft"]},
    )
    runner = BenchmarkRunner(
        jobs=jobs,
        model_config=model_config,
        documents=documents,
        examples=examples,
        retriever=create_retriever({"kind": "bm25"}),
        retriever_config={"kind": "bm25"},
        generator_factory=TransformersGeneratorFactory(model_config=model_config, seed=7),
        scorer=build_scorer({"mode": "disabled"}),
        output_dir=tmp_path / "benchmark",
        top_k=1,
        bootstrap_samples=10,
        warmup_examples=0,
        model_config_path=model_config_path,
        documents_path=documents_path,
        eval_path=eval_path,
    )
    summary = runner.run()
    assert set(summary["recipes"]) == {"base", "rag", "sft-rag", "raft-rag"}
    assert summary["provenance"]["verified"] is True
    assert summary["provenance"]["training_controls_matched"] is True
    assert (tmp_path / "benchmark" / "summary.json").is_file()
    assert (tmp_path / "benchmark" / "report.md").is_file()
    assert (tmp_path / "benchmark" / "predictions.jsonl").is_file()
    validate_artifact_schema(summary, "benchmark-summary-v3.schema.json")
    for recipe in summary["recipes"].values():
        assert recipe["metrics"]["exact_match"] is not None
        assert recipe["metrics"]["token_f1"] is not None
        assert recipe["metrics"]["peak_allocated_vram_gb"] is not None
        assert recipe["metrics"]["peak_reserved_vram_gb"] is not None
        assert Path(recipe["predictions"]).is_file()
