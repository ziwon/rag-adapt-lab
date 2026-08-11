# Architecture

## Goal

`rag-adapt-lab` is a comparison harness, not an application framework. Its job is to make domain adaptation decisions measurable.

## Core interfaces

```text
Data Contract
   ├── Documents
   ├── Evaluation examples
   └── Optional SFT/RAFT examples
          │
          ▼
Retriever ───> Shared ranking cache ─────────────┐
                                                │
Generator ────────────────────────────────> Benchmark runner ───> Predictions
                                                │                    │
Trainer ───> versioned SFT/RAFT adapters ───────┘                    ▼
                                                               Metrics + paired CI
                                                                      │
                                             ┌────────────────────────┼───────────┐
                                             ▼                        ▼           ▼
                                        summary.json              report.md   Tracker
                                                │
                                      ┌─────────┴─────────┐
                                      │                   │
                                   W&B Models           Weave
```

## Separation of concerns

### Data

The data package validates domain-neutral JSONL records and prepares RAFT-style training samples. Negative mining is a strategy: random sampling is the control, BM25 hard-negative mining is built in, and any `Retriever` can be injected for dense or hybrid mining. Oracle relevance is retained in JSONL metadata but is stripped from model-facing prompts.

### Retrieval

Retrievers return ranked documents and scores. BM25 and pinned dense retrieval are available through the retriever factory; hybrid retrieval remains an explicit extension point. A benchmark indexes the corpus and retrieves each held-out question once, then reuses those exact rankings for every retrieval-enabled recipe.

### Generation

Generation is an interface. The benchmark's local Transformers runner loads the pinned base model and optional PEFT adapters, records token counts/timing/peak VRAM, and uses paired per-example seeds. The existing OpenAI-compatible runner remains suitable for vLLM/SGLang integrations; a factory can be injected without changing benchmark or evaluation code.

### Training

The QLoRA trainer uses TRL + PEFT + bitsandbytes and accepts ordinary SFT or RAFT records. It uses an explicit validation file or deterministic seeded split, checks both training partitions against an optional held-out benchmark file, evaluates on a configurable schedule, supports early stopping, and persists the best adapter plus a training manifest.

Training records use TRL's prompt/completion contract with completion-only loss. This masks prompt and retrieved-document tokens, leaving only assistant completion tokens as labels. Chat formatting uses conversational prompt/completion messages; it does not depend on template-specific assistant masks.

### Evaluation

Retrieval metrics, exact match, token F1, lexical groundedness, unsupported-claim rate, and optional citation metrics are deterministic. Model-based correctness/groundedness is complementary and uses a versioned scorer over a pluggable judge backend. OpenAI-compatible endpoints and in-process callable judges are supported; a no-op scorer disables all plugins while retaining canonical EM/F1.

The statistics layer pairs recipes by evaluation ID and produces deterministic percentile-bootstrap confidence intervals. The report layer consumes only `summary.json` data, so human-readable reporting stays separate from model execution.

### Tracking

`Tracker` is backend-neutral. `NullTracker` is the benchmark default and `WandbTracker` is opt-in. `WeaveTracer` remains separate because tracing and metric/artifact logging have different lifecycles.

## Why RAFT is modeled as data preparation

RAFT is most reusable when its semantics are represented in data:

- question;
- oracle/relevant evidence;
- distractors;
- answer;
- evidence IDs.

This makes the training backend replaceable and keeps domain logic out of trainer code. New negative-mining methods should implement or compose a retriever rather than introduce domain-specific trainers.

## Controlled benchmark contract

All jobs in one benchmark share:

- one held-out `EvalExample` sequence and one corpus hash;
- one immutable base model revision;
- one prompt name/version and generation configuration;
- one cached retrieval ranking per question and one `top_k`;
- one scorer configuration and bootstrap seed.

Only two recipe switches vary: whether retrieved contexts are supplied, and whether an SFT or RAFT adapter is loaded. Base and RAG reuse the same unadapted model instance. The runner rejects duplicate recipes, missing documents, unknown relevant IDs, and empty corpora/evaluation sets before generation.
