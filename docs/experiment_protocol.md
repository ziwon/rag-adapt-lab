# Experiment Protocol

## Canonical comparison

For a fixed model family and corpus, compare:

1. Base
2. RAG
3. SFT + RAG
4. RAFT + RAG

Hold the evaluation set, retrieval index, prompt template, and generation parameters constant unless the experiment explicitly studies one of those variables.

Run all recipes in one `raglab benchmark` invocation where possible. The runner caches retrieval once and pairs metrics by example ID. If a setting is intentionally changed, treat it as a separate experiment rather than another recipe in the same comparison.

## Minimum reporting table

| Field | Description |
|---|---|
| recipe | base / rag / sft-rag / raft-rag |
| model | base model ID and revision |
| adapter | adapter artifact/version if applicable |
| dataset | corpus and train/eval artifact versions |
| retriever | kind, model/version, top-k |
| recall@k | retrieval recall |
| mrr | mean reciprocal rank |
| ndcg@k | normalized discounted cumulative gain |
| exact_match | deterministic generation metric when applicable |
| token_f1 | deterministic generation metric when applicable |
| correctness | judge or task-specific metric |
| groundedness | judge or task-specific metric |
| latency_p50 | end-to-end latency |
| latency_p95 | end-to-end latency |
| peak_vram_gb | peak observed GPU memory |

`summary.json` is the source of truth; `report.md` is a deterministic rendering of it. Per-example JSONL must be retained so aggregate results and paired statistics can be audited.

## Statistical interpretation

Use paired deltas because every recipe answers the same held-out examples. The standard report includes Base → RAG, RAG → SFT + RAG, RAG → RAFT + RAG, and SFT + RAG → RAFT + RAG. Exact match and token F1 use seeded paired percentile-bootstrap 95% confidence intervals. A difference is marked statistically supported only when the interval excludes zero.

The interval describes uncertainty over the sampled evaluation examples; it does not establish external validity for another domain, corpus, or retrieval distribution. LLM-judge results remain secondary to deterministic metrics and should be calibrated against human ratings before driving a high-stakes conclusion.

## Recommended ablations

- random distractors vs hard negatives in RAFT;
- BM25 vs dense vs hybrid retrieval;
- 4B vs 8B vs 14B base models;
- LoRA rank and context length;
- number of retrieved documents;
- local judge vs frontier judge;
- with/without reasoning mode where supported.

## Leakage controls

Do not create training examples from the held-out evaluation answers. Always pass the benchmark file to both `prepare-raft --held-out-eval` and `train --held-out-eval`; both reject reused IDs and normalized questions. A training validation set must come from a separate file or a deterministic split of training data, never from the benchmark evaluation file.

For synthetic data generation, record the source model, prompt version, filtering rules, and whether evaluation documents were excluded. For hard-negative mining, index only the training corpus. Record the mining strategy, seed, candidate pool, selected document IDs, ranks, and scores.

## Latency protocol

The local runner performs a configurable warm-up outside the measured region, synchronizes CUDA around generation, and reports per-example generation and end-to-end latency. Retrieval time is counted only for retrieval-enabled recipes. Peak allocated GPU VRAM is reset and measured per recipe; tokens/sec is computed from generated tokens where token accounting is available. Run timing comparisons on an otherwise idle machine and repeat them when small differences matter.
