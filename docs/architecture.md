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

Generation is an interface. The local runner validates the explicit chat-template condition and
uses one shared token-level parser to split Qwen-style reasoning from the scored final answer at
the tokenizer-resolved special boundary. It records the raw output, split text/IDs-derived counts,
boundary identity/status, prompt construction, template, tokenization, transfer, generation,
decode, and allocated/reserved VRAM metrics. `inference_e2e_latency_s` covers retrieval through
decode and excludes all scoring.

### Training

The QLoRA trainer uses pinned TRL + PEFT + bitsandbytes and accepts ordinary SFT or RAFT records. Group-aware splitting operates on connected grouping keys and supports shared-corpus and document-disjoint policies. RAFT can split raw labeled QA before independently mining each partition. Both partitions are checked against a mandatory held-out benchmark file before a verifiable adapter is produced.

Both training modes use prompt v4: SFT passes an empty context list and RAFT passes evidence plus distractors. The provenance hash covers the empty-, one-, and multi-document rendering branches. Because TRL 0.24.0 does not accept chat-template kwargs in `SFTConfig`, the tokenizer template is explicitly rendered first with the recorded kwargs, then passed to TRL as plain prompt/completion text with completion-only loss. Verifiable training therefore requires `use_chat_template: true`.

Training and adapter manifests use schema v3. Both persist a canonical normalized training-control
object and SHA-256 digest. The benchmark recomputes adapter artifact hashes and validates model
revision, adaptation mode, prompt identity/hash, chat-template args, the exact held-out evaluation
hash, and matched source-partition and training-control fingerprints for SFT versus RAFT. Missing
or incompatible manifests fail closed unless the corresponding visibly recorded override is
enabled; a training-control override is labeled confounded rather than merely legacy-unverified.

### Evaluation

Retrieval metrics, exact match, token F1, `reference_overlap`, lexical groundedness, lexical unsupported-claim rate, and optional citation metrics are deterministic. Model-based scores remain complementary. Judge inputs are delimited untrusted data, endpoint failures are isolated unless strict mode is requested, and cache/retry/failure/latency metadata remains auditable. Aggregate judge metrics carry numeric/total coverage; paired statistics carry baseline, candidate, and intersection counts and become decision-ineligible below configured coverage/sample thresholds.

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

Only two recipe switches vary: whether retrieved contexts are supplied, and whether an SFT or RAFT
adapter is loaded. Base and RAG reuse the same unadapted model instance. SFT and RAFT must share
their source populations and every normalized learning control, differing only in data treatment.
The runner rejects duplicate recipes, missing documents, unknown relevant IDs, empty
corpora/evaluation sets, duplicate adapter artifacts, and confounded adapted comparisons before
generation.
