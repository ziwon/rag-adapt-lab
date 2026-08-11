# Scaffold Status

## Implemented in v0.1

- Domain-neutral JSONL schemas for documents, evaluation records, SFT records, and RAFT records
- Dataset validation CLI
- Generic RAFT dataset builder using annotated oracle documents plus random distractors
- BM25 retrieval backend
- Dense retrieval backend interface/implementation
- Retrieval evaluation: Recall@K, Hit Rate@K, MRR, nDCG@K
- Deterministic generation metrics: normalized exact match and token F1
- QLoRA training scaffold using Transformers, TRL, PEFT, and bitsandbytes
- SFT and RAFT formatting paths through the same trainer
- W&B Models run/artifact tracker
- W&B Weave tracing wrapper
- Null tracking backend
- OpenAI-compatible generation runner for local vLLM/SGLang-style endpoints
- 16GB and 24GB GPU configuration profiles
- Qwen3 4B, 8B, and 14B example model configs
- Base / RAG / SFT+RAG / RAFT+RAG recipe configs
- Benchmark execution-plan CLI
- Unit tests and GitHub Actions CI scaffold
- Dockerfile for NVIDIA CUDA environments

## Deliberate extension points

These are intentionally not presented as complete production implementations in v0.1:

- end-to-end benchmark execution across all four recipes;
- dense/hybrid retriever wiring in the CLI;
- hard-negative mining for RAFT;
- local LLM-as-a-judge and domain-specific scorers;
- automatic W&B Artifact lineage across every CLI command;
- full Transformers generation runner;
- vLLM/SGLang server lifecycle management;
- reranking;
- production serving and distributed training.

The repository is structured so these capabilities can be added without changing the data contract or experiment protocol.

## Validation performed on the scaffold

- Python source compilation completed successfully.
- Unit tests completed successfully.
- CLI help entrypoint executed successfully through `PYTHONPATH=src`.
- Repository text was checked for Hangul; all scaffold content is in English.

GPU training was not executed in the build environment because no target RTX 5080/5090 CUDA runtime was available here. Validate CUDA/PyTorch/bitsandbytes compatibility on the target workstation before starting a full QLoRA run.
