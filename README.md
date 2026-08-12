# rag-adapt-lab

A domain-neutral, reproducible research harness for answering a practical question:

> **When is plain RAG enough, and when does domain adaptation (SFT or RAFT) provide measurable value?**

<img width="1024" height="576" alt="ChatGPT Image Aug 12, 2026, 01_02_27 AM (1)" src="https://github.com/user-attachments/assets/018c71c8-a955-4eae-ae12-1f6bb5506915" />


`rag-adapt-lab` is designed for local and self-hosted LLM experiments on commodity NVIDIA GPUs. It compares four first-class recipes under a shared data contract and evaluation protocol:

1. **Base** — model only, no retrieval, no training
2. **RAG** — retrieval-augmented generation, no training
3. **SFT + RAG** — QLoRA/SFT domain adaptation followed by RAG
4. **RAFT + RAG** — retrieval-aware fine-tuning with positive evidence and distractors, followed by RAG

The project can use **Weights & Biases Models** for training/config/checkpoint/dataset lineage and **W&B Weave** for RAG/LLM tracing. Both are optional: the benchmark, metrics, predictions, confidence intervals, and reports work locally without a tracking service.

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

The default Qwen3 condition is deliberately concise and non-thinking: every shipped Qwen3 config
sets `chat_template_kwargs.enable_thinking: false`, `do_sample: false`, and
`max_new_tokens: 64`. Thinking mode is supported only as a separately labeled sampled condition;
its reasoning and final answer are parsed and counted independently, and EM/F1 use the final answer.

## Repository layout

```text
rag-adapt-lab/
├── configs/
│   ├── hardware/
│   ├── models/
│   ├── recipes/
│   ├── retrievers/
│   ├── scorers/
│   └── training/
├── examples/demo/
├── src/rag_adapt_lab/
│   ├── data/
│   ├── benchmark/
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
{"id":"sft-001","input":"Question...","output":"Expected answer..."}
```

The legacy `instruction` field remains accepted as data metadata, but schema v3 deliberately does not render per-row instruction variants: SFT must use the same benchmark prompt contract with an empty document list. If no explicit SFT file is available, derive it from the same pre-split labeled QA source used for RAFT.

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

### 2. Optionally configure W&B

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
  --validation-output data/raft_validation.jsonl \
  --split-config configs/splits/grouped-shared-corpus.yaml \
  --distractors 2 \
  --negative-strategy bm25-hard-negative \
  --candidate-pool-size 20 \
  --seed 42
```

`random` remains available for ablations. `bm25-hard-negative` retrieves the highest-ranked non-relevant documents and mixes them with the positive evidence. Positive IDs are always removed before distractor selection. With `--validation-output`, raw QA is group-split first and separate retrievers mine each partition. `shared-corpus` permits document reuse while keeping questions disjoint; `document-disjoint` partitions positives and distractor pools so no document crosses the boundary. The emitted manifest records group counts, fingerprints, overlap counts, corpus policy, mining scope, and seed. Relevance labels remain metadata and are never rendered into the prompt.

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
  --train-file examples/demo/sft.jsonl \
  --validation-file data/sft_validation.jsonl \
  --held-out-eval examples/demo/eval.jsonl
```

For RAFT:

```bash
raglab train \
  --config configs/recipes/raft-rag.yaml \
  --train-file data/raft_train.jsonl \
  --validation-file data/raft_validation.jsonl \
  --held-out-eval examples/demo/eval.jsonl
```

Training configs default to a deterministic group-aware validation split. Supplying the partition produced before RAFT mining with `--validation-file` is preferred. `--held-out-eval` is mandatory for verifiable adapters: it is used only for overlap and hash checks and is never passed to the trainer. The trainer evaluates at the configured interval, supports early stopping, reloads the best checkpoint, and writes schema-v3 training and adapter manifests. These include the base revision, adaptation mode, prompt identity/hash, effective chat-template args, representation-independent source-partition fingerprints, full dataset fingerprints, held-out hash, training-config hash, split policy/audit, and adapter artifact hash. When both adapted conditions are benchmarked, their source-partition fingerprints must match.

SFT and RAFT use exactly the same versioned `rag-user-prompt` builder. SFT supplies no documents; RAFT supplies positive evidence and distractors. For TRL 0.24.0, the tokenizer chat template is explicitly rendered with the model's effective kwargs before creating plain prompt/completion records. `completion_only_loss: true` therefore masks the entire prompt (including retrieved documents), leaving only assistant completion tokens as targets. Oracle relevance flags are excluded.

### 7. Run the benchmark matrix

Train the SFT and RAFT adapters first, then execute all four conditions against one held-out set:

```bash
raglab benchmark \
  --recipes base,rag,sft-rag,raft-rag \
  --model-config configs/models/qwen3-8b.yaml \
  --documents examples/demo/documents.jsonl \
  --eval-set examples/demo/eval.jsonl \
  --retriever-config configs/retrievers/bm25.yaml \
  --scorer-config configs/scorers/default.yaml \
  --sft-adapter outputs/qwen3-8b-sft-rag/adapter \
  --raft-adapter outputs/qwen3-8b-raft-rag/adapter \
  --top-k 3 \
  --bootstrap-samples 10000 \
  --output-dir outputs/demo-benchmark
```

The runner builds the retrieval index once, caches one ranking per evaluation example, and holds the model revision, prompt version, retrieval results, generation settings, and scorer versions fixed. Base and RAG share the same loaded base model; adapted recipes load their PEFT adapters on that same revision. A fixed per-example sampling seed schedule and unmeasured warm-up make paired quality and latency comparisons less confounded.

Adapter validation fails closed. SFT and RAFT manifests must declare the expected mode and match
the benchmark model revision, prompt contract, chat-template args, and current held-out file hash;
their recorded artifact hashes are recomputed, and identical SFT/RAFT artifacts are rejected. A
legacy adapter can be run only with `--allow-unverified-adapter`; the CLI emits a warning and both
`summary.json` and `report.md` label the experiment as unverified.

Outputs include:

- `predictions/<recipe>.jsonl` and combined `predictions.jsonl`, including retrieved IDs/scores, every inference stage, separate reasoning/answer token counts, deterministic scoring time, judge time/status, and all per-example scores;
- `summary.json`, containing aggregate metrics, configuration/input hashes, and paired percentile-bootstrap intervals;
- `report.md`, containing the recipe table, retrieval quality, decision-oriented comparisons, latency, throughput, and peak GPU VRAM.

Use `--dry-run --plan-output outputs/benchmark-plan.json` to validate and save a plan without loading a model. W&B logging is opt-in with `--tracking-backend wandb`; the default is local-only.

The default scorer combines deterministic `reference_overlap` with explicitly labeled lexical groundedness/unsupported-claim heuristics. An optional versioned LLM judge can target any OpenAI-compatible endpoint:

```bash
cp configs/scorers/openai-compatible.example.yaml configs/scorers/local-judge.yaml
export JUDGE_API_KEY=local
```

The judge treats the answer and retrieved text as untrusted serialized data under a separate system rubric. Connect/read timeouts, bounded retry/backoff, response-size limits, structured output, concurrency, deterministic cache, and strict/non-strict behavior are configurable. Non-strict failures are isolated per example and never remove EM/F1; reports include judge coverage and failure rate. The Python API also accepts `CallableJudgeBackend`. Set `mode: disabled` to disable plugin scores while retaining EM/F1.

### Public Hugging Face smoke experiment

The included SQuAD workflow pins the public dataset and base-model revisions, creates disjoint training/evaluation splits, trains ordinary SFT and RAFT LoRA adapters, and executes the paired four-recipe comparison:

```bash
python scripts/prepare_hf_squad.py \
  --output-dir data/hf_squad_smoke \
  --cache-dir .cache/huggingface

raglab train \
  --config configs/recipes/hf-squad-sft-smoke.yaml \
  --train-file data/hf_squad_smoke/sft_train.jsonl \
  --validation-file data/hf_squad_smoke/sft_validation.jsonl \
  --held-out-eval data/hf_squad_smoke/eval.jsonl

raglab train \
  --config configs/recipes/hf-squad-raft-smoke.yaml \
  --train-file data/hf_squad_smoke/raft_train.jsonl \
  --validation-file data/hf_squad_smoke/raft_validation.jsonl \
  --held-out-eval data/hf_squad_smoke/eval.jsonl

raglab benchmark \
  --recipes base,rag,sft-rag,raft-rag \
  --model-config configs/models/qwen2.5-0.5b-instruct.yaml \
  --documents data/hf_squad_smoke/eval_documents.jsonl \
  --eval-set data/hf_squad_smoke/eval.jsonl \
  --retriever-config configs/retrievers/bm25.yaml \
  --sft-adapter outputs/hf-squad-sft-chat-smoke/adapter \
  --raft-adapter outputs/hf-squad-raft-chat-smoke/adapter \
  --output-dir outputs/hf-squad-benchmark
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

The core package provides deterministic metrics and complementary scorer plugins:

- exact match / normalized exact match
- token F1
- reference answer correctness (deterministic token overlap)
- lexical groundedness / unsupported-claim rate (deterministic heuristic)
- answer correctness, groundedness, and unsupported-claim rate (optional LLM judge)
- citation precision / recall for `[Document N]` citations (optional deterministic plugin)

### System metrics

Recommended fields to log with every run:

- inference E2E latency (retrieval through decode, excluding scoring/judge)
- retrieval, prompt-build, chat-template, tokenization, transfer, generation, and decode latency
- deterministic scoring latency and separate judge latency
- prompt/output tokens
- reasoning/answer tokens for thinking-enabled conditions
- output and total tokens per model-generation second
- peak allocated and peak reserved GPU VRAM
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

Useful next additions:

- hybrid retrieval and reranking
- batched Transformers generation and vLLM/SGLang benchmark factories
- semantic claim decomposition for stronger local groundedness scoring
- judge calibration against human labels and inter-judge agreement reporting
- multiple-comparison corrections when many recipes or metrics are explored
- MLflow tracker/tracer backend

Current limitations:

- the CLI benchmark uses sequential, batch-size-one Transformers generation; alternative generator factories can be injected in Python, but vLLM/SGLang are not yet CLI-selectable;
- BM25 is the only named hard-negative strategy; dense and hybrid mining require an injected retriever;
- lexical groundedness is a reproducible heuristic, not semantic entailment, and LLM judges require task-specific human calibration;
- citation scoring recognizes explicit `[Document N]` references only and is disabled by default;
- the stock trainer selects from metrics emitted by TRL/Transformers (normally `eval_loss`); custom task metrics require extending the trainer;
- confidence intervals are unadjusted for multiple comparisons.
- the stock answer-only trainer fails closed for thinking-enabled adapter training; thinking-enabled base/RAG inference and externally trained schema-v3 adapters remain evaluable as separate conditions;
- the GPU workflow requires a compatible self-hosted runner and is not evidence that a particular local checkout has executed CUDA unless that job result is available;

## Validation layers

- **Unit tests:** core/dev dependencies; fast orchestration, failure, and statistical tests.
- **CPU integration:** pinned real BM25, Datasets, Transformers, TRL, PEFT manifest contracts, report generation, and Compose parsing without model downloads.
- **GPU integration:** scheduled/manual self-hosted workflow that constructs a local tiny model, trains distinct SFT/RAFT LoRA adapters, reloads them, generates, captures CUDA memory, and executes one benchmark matrix.

Passing only the unit job is not described as full validation. See
[schema-v3 migration notes](docs/migration_v3.md) for matched adaptation populations, prompt provenance, and judge-cache changes. The historical [schema-v2 notes](docs/migration_v2.md) remain available.

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
