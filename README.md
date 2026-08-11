# rag-adapt-lab

A domain-neutral, W&B-first research harness for answering a practical question:

> **When is plain RAG enough, and when does domain adaptation (SFT or RAFT) provide measurable value?**

<img width="1024" height="576" alt="ChatGPT Image Aug 12, 2026, 01_02_27 AM (1)" src="https://github.com/user-attachments/assets/018c71c8-a955-4eae-ae12-1f6bb5506915" />


`rag-adapt-lab` is designed for local and self-hosted LLM experiments on commodity NVIDIA GPUs. It compares four first-class recipes under a shared data contract and evaluation protocol:

1. **Base** — model only, no retrieval, no training
2. **RAG** — retrieval-augmented generation, no training
3. **SFT + RAG** — QLoRA/SFT domain adaptation followed by RAG
4. **RAFT + RAG** — retrieval-aware fine-tuning with positive evidence and distractors, followed by RAG

The project uses **Weights & Biases Models** for training/config/checkpoint/dataset lineage and **W&B Weave** for RAG/LLM tracing and application-level evaluation. Tracking is abstracted so the core experiment code remains usable without W&B.

## Why this project exists

Teams often jump directly from a general-purpose LLM to fine-tuning. That can be expensive and unnecessary. This repository makes the alternatives comparable with the same corpus, held-out questions, retrieval pipeline, model family, and metrics. Fine-tuning examples come from a separate labeled training split.

The intended output is not merely a higher score. It is a reproducible answer to questions such as:

- How much does retrieval improve over the base model?
- Does SFT add value after retrieval is already strong?
- Does RAFT improve evidence selection under distracting retrieval results?
- Does a smaller adapted model outperform a larger generic model in a narrow task?
- What is the quality/latency/VRAM trade-off for each recipe?

## Target hardware

The default profiles target single-GPU development and experimentation:

| Profile | Intended use | Suggested model tier |
|---|---|---|
| `rtx_16gb` | smoke tests, development, primary QLoRA | 4B–8B |
| `rtx_24gb` | primary experiments, longer context, larger adapters | 8B–14B |

The included Qwen configs are examples, not hard requirements. Their Hub revisions are immutable commit SHAs and remote model code is disabled. Any compatible causal LM can be added through config with the same safeguards.

## Repository layout

```text
rag-adapt-lab/
├── configs/
│   ├── hardware/
│   ├── models/
│   ├── recipes/
│   ├── retrievers/
│   └── training/
├── examples/demo/
├── src/rag_adapt_lab/
│   ├── data/
│   ├── evaluation/
│   ├── generation/
│   ├── recipes/
│   ├── retrieval/
│   ├── tracking/
│   └── training/
├── tests/
├── docker/
└── .github/workflows/
```

## Data contract

The core project is intentionally domain-neutral.

### `documents.jsonl`

Minimum required fields:

```json
{"id":"doc-001","text":"Document body...","metadata":{"source":"manual.pdf","language":"en"}}
```

Only `id` and `text` are required.

### `eval.jsonl`

```json
{"id":"q-001","question":"What is ...?","reference_answer":"...","relevant_doc_ids":["doc-001"]}
```

Optional fields include `evidence` and arbitrary metadata.

### `train_qa.jsonl` (for RAFT preparation)

The labeled RAFT source uses the same schema as `eval.jsonl`, but it must be a distinct training split:

```json
{"id":"train-q-001","question":"What is ...?","reference_answer":"...","relevant_doc_ids":["doc-001"]}
```

`raglab prepare-raft --held-out-eval ...` rejects reused record IDs and normalized questions so evaluation examples cannot accidentally become training examples.

### `sft.jsonl` (optional)

```json
{"id":"sft-001","instruction":"Answer using the domain terminology.","input":"Question...","output":"Expected answer..."}
```

If no explicit SFT file is available, you can create one from your own pipeline or use the RAFT builder as a starting point for retrieval-aware examples.

## Quick start

### 1. Create an environment

Python 3.11 is recommended.

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[rag,train,wandb,dev]'
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[rag,train,wandb,dev]'
```

> Install the PyTorch build that matches your CUDA driver first when needed. The project intentionally does not pin a CUDA-specific wheel URL.

### 2. Configure W&B

```bash
cp .env.example .env
export WANDB_PROJECT=rag-adapt-lab
export WANDB_ENTITY=<your-team-or-user>
wandb login
```

For Weave:

```bash
export WEAVE_PROJECT=rag-adapt-lab
```

All tracking can be disabled with:

```bash
export WANDB_MODE=disabled
```

### 3. Validate the demo data

```bash
raglab validate-data \
  --documents examples/demo/documents.jsonl \
  --eval-set examples/demo/eval.jsonl
```

### 4. Build a RAFT-style dataset

```bash
raglab prepare-raft \
  --documents examples/demo/documents.jsonl \
  --training-set examples/demo/train_qa.jsonl \
  --held-out-eval examples/demo/eval.jsonl \
  --output data/raft_train.jsonl \
  --distractors 2
```

This basic builder uses training-only relevant document annotations as oracle evidence and samples non-relevant documents as distractors. The relevance labels remain dataset metadata and are not rendered into the model prompt. For production research, replace the random sampler with BM25/dense hard negatives.

### 5. Run retrieval evaluation

```bash
raglab eval-retrieval \
  --documents examples/demo/documents.jsonl \
  --eval-set examples/demo/eval.jsonl \
  --retriever bm25 \
  --top-k 3
```

### 6. Train a QLoRA adapter

```bash
raglab train \
  --config configs/recipes/sft-rag.yaml \
  --train-file examples/demo/sft.jsonl
```

For RAFT:

```bash
raglab train \
  --config configs/recipes/raft-rag.yaml \
  --train-file data/raft_train.jsonl
```

### 7. Run the benchmark matrix

`benchmark` currently creates and validates an experiment plan. The model execution hooks are deliberately modular so you can attach Transformers, vLLM, SGLang, or an OpenAI-compatible endpoint without rewriting evaluation code.

```bash
raglab benchmark \
  --recipes base,rag,sft-rag,raft-rag \
  --model-config configs/models/qwen3-8b.yaml \
  --documents examples/demo/documents.jsonl \
  --eval-set examples/demo/eval.jsonl
```

### Public Hugging Face smoke experiment

The included SQuAD workflow pins the public dataset and base-model revisions, creates disjoint training/evaluation splits, trains a small LoRA adapter, and compares paired base/tuned predictions:

```bash
python scripts/prepare_hf_squad.py \
  --output-dir data/hf_squad_smoke \
  --cache-dir .cache/huggingface

raglab train \
  --config configs/recipes/hf-squad-raft-smoke.yaml \
  --train-file data/hf_squad_smoke/raft_train.jsonl

python scripts/evaluate_hf_squad.py \
  --model-config configs/models/qwen2.5-0.5b-instruct.yaml \
  --adapter outputs/hf-squad-raft-chat-smoke/adapter \
  --documents data/hf_squad_smoke/eval_documents.jsonl \
  --eval-set data/hf_squad_smoke/eval.jsonl \
  --output-dir outputs/hf-squad-raft-chat-smoke/evaluation
```

Training and evaluation require the `train` and `rag` extras; model evaluation requires CUDA.

### Reproducible local Compose stack

The repository includes a development Compose environment with pinned SeaweedFS and W&B Server
images, S3 bucket initialization, persistent caches, and GPU profiles for the lab and public-data
jobs:

```bash
cp .env.compose.example .env.compose
# Add WANDB_LICENSE, then:
docker compose --env-file .env.compose up -d --wait seaweedfs seaweed-init wandb
docker compose --env-file .env.compose --profile lab up -d --build --wait lab
```

Weave uses the same self-hosted W&B endpoint when the W&B license enables Weave; it is not a
separate standalone container. See [Local Compose environment](docs/local_compose.md) for setup,
service URLs, job commands, storage behavior, and the production deployment caveat.

## Experiment model

A recommended experiment is:

```text
              Same corpus / one held-out evaluation set
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
      Base                   RAG            Separate train split
                                                     │
                                         ┌───────────┴───────────┐
                                         │                       │
                                        SFT                     RAFT
                                      (QLoRA)                 (QLoRA)
                                         │                       │
                                         └───────────┬───────────┘
                                                     │
                                                    RAG
                                                     │
                                              Unified metrics
```

### Retrieval metrics

- Recall@K
- Hit Rate@K
- MRR
- nDCG@K

### Generation metrics

The core package provides deterministic building blocks and an interface for model-based scorers:

- exact match / normalized exact match
- token F1
- answer correctness (judge/plugin)
- groundedness / faithfulness (judge/plugin)
- citation precision / recall (plugin)
- unsupported-claim rate (plugin)

### System metrics

Recommended fields to log with every run:

- end-to-end latency
- retrieval latency
- generation latency
- prompt/output tokens
- tokens/sec
- peak GPU memory
- GPU utilization
- failure rate

## W&B architecture

### W&B Models

Use W&B runs and artifacts for:

```text
corpus:v3
   ↓
raft_dataset:v1
   ↓
qwen_adapter:v4
   ↓
evaluation:v7
```

Training runs should log the git commit, model revision, dataset artifact, recipe, LoRA config, optimizer, context length, seed, and GPU profile.

### Weave

Weave is intended for application-level traces:

```text
question
  ↓
retrieve → retrieved documents
  ↓
(optional reranker)
  ↓
prompt construction
  ↓
local/self-hosted LLM
  ↓
answer
  ↓
scorers / judge
```

The `WeaveTracer` wrapper in this repository is intentionally lightweight. It initializes a project and can decorate functions without making Weave mandatory for the rest of the codebase.

## Configuration philosophy

The repository follows four rules:

1. **Domain knowledge lives in data, not Python source.**
2. **RAFT is a data recipe, not a domain-specific trainer.**
3. **W&B-first does not mean W&B-hardcoded.**
4. **Retrieval, generation, training, scoring, and tracking have separate interfaces.**

## Extending the project

Good next additions after v0.1:

- Qwen3 Embedding / BGE-M3 dense retrievers
- hard-negative mining for RAFT
- hybrid retrieval and reranking
- Transformers generation runner
- vLLM and SGLang OpenAI-compatible runners
- local LLM-as-a-judge
- Weave evaluation datasets/scorers
- MLflow tracker/tracer backend
- benchmark report generation

Out of scope for v0.1:

- full-parameter fine-tuning
- RLHF/RL
- 30B+ training
- agent orchestration
- Graph RAG
- multimodal RAG
- distributed training
- custom CUDA kernels
- a custom vector database

## Reproducibility checklist

For every experiment, record:

- model ID and immutable commit revision
- tokenizer commit revision
- dataset artifact/version
- retrieval index configuration
- train/eval split seed
- QLoRA parameters
- max sequence length
- prompt template version
- generation parameters
- evaluator/scorer version
- GPU model and VRAM
- git commit

## Security and privacy

This repository is designed to work with private corpora, but W&B/Weave telemetry can contain prompts, retrieved text, outputs, or metadata. Before using sensitive data:

- review what fields are logged;
- redact or hash identifiers where appropriate;
- use organization-approved W&B deployment/settings;
- or disable external tracking and use the `NullTracker` until an approved backend is configured.

## License

Apache License 2.0. See [LICENSE](LICENSE).
