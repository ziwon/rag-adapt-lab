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
Retriever ────────┐
                  │
Generator ────────┼──> Recipe execution ───> Evaluation
                  │                              │
Trainer ──────────┘                              ▼
                                             Tracker
                                                │
                                      ┌─────────┴─────────┐
                                      │                   │
                                   W&B Models           Weave
```

## Separation of concerns

### Data

The data package validates domain-neutral JSONL records and prepares RAFT-style training samples.

### Retrieval

Retrievers return ranked documents and scores. BM25 is wired into the v0.1 CLI. Dense retrieval is implemented as a backend and hybrid retrieval is intentionally left as an extension point.

### Generation

Generation is an interface. The included OpenAI-compatible runner is suitable for vLLM/SGLang endpoints. A Transformers local runner can be added without touching evaluation logic.

### Training

The QLoRA trainer uses TRL + PEFT + bitsandbytes and accepts either ordinary SFT records or RAFT records.

### Evaluation

Retrieval metrics are deterministic. Generation scorers have a plugin interface so teams may choose a local judge, hosted judge, Weave scorer, or task-specific deterministic metric.

### Tracking

`Tracker` is backend-neutral. `WandbTracker` is the default integration and `NullTracker` supports disconnected/private development. `WeaveTracer` is separate because tracing and metric/artifact logging have different lifecycles.

## Why RAFT is modeled as data preparation

RAFT is most reusable when its semantics are represented in data:

- question;
- oracle/relevant evidence;
- distractors;
- answer;
- evidence IDs.

This makes the training backend replaceable and keeps domain logic out of trainer code. Future RAFT builders should add hard-negative mining rather than new domain-specific trainers.
