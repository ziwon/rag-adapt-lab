# Experiment Protocol

## Canonical comparison

For a fixed model family and corpus, compare:

1. Base
2. RAG
3. SFT + RAG
4. RAFT + RAG

Hold the evaluation set, retrieval index, prompt template, and generation parameters constant unless the experiment explicitly studies one of those variables.

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

## Recommended ablations

- random distractors vs hard negatives in RAFT;
- BM25 vs dense vs hybrid retrieval;
- 4B vs 8B vs 14B base models;
- LoRA rank and context length;
- number of retrieved documents;
- local judge vs frontier judge;
- with/without reasoning mode where supported.

## Leakage controls

Do not create training examples from the held-out evaluation answers. For synthetic data generation, record the source model, prompt version, filtering rules, and whether any evaluation documents were excluded.
