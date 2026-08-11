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
| reference_overlap | deterministic reference token overlap; not semantic correctness |
| judge_correctness | optional model-judge score with coverage/failure rate |
| lexical_groundedness | deterministic lexical support heuristic |
| judge_groundedness | optional model-judge groundedness |
| inference_e2e_latency_p50_s | retrieval through decoded answer; no scoring/judge |
| inference_e2e_latency_p95_s | retrieval through decoded answer; no scoring/judge |
| peak_allocated_vram_gb | maximum live allocated CUDA tensor memory |
| peak_reserved_vram_gb | maximum CUDA allocator reservation |

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

Do not create training examples from held-out answers. Always pass the benchmark file to both preparation and training. For RAFT, prefer grouped split-before-mining and pass the resulting train/validation files explicitly. Choose `shared-corpus` when document reuse is part of the task, or `document-disjoint` for source generalization. The latter partitions every positive and distractor pool before retrieval indexing.

For synthetic data generation, record the source model, prompt version, filtering rules, and whether evaluation documents were excluded. For hard-negative mining, index only the training corpus. Record the mining strategy, seed, candidate pool, selected document IDs, ranks, and scores.

## Latency protocol

The local runner performs an unmeasured warm-up and synchronizes CUDA around transfers and generation. Persist retrieval, prompt-build, chat-template, tokenization, device-transfer, model-generate, decode, inference-E2E, deterministic-scoring, and judge latency separately. Judge latency is never part of user-facing inference latency. Report allocated and reserved CUDA peaks, output/total-token throughput, batch size, and sequential/batched mode. Repeat timing runs on an idle machine when small differences matter.

## Adapter provenance gate

Schema-v2 adapter manifests are mandatory by default. Before loading, validate the immutable base,
expected SFT/RAFT mode, prompt name/version/hash, chat-template arguments, held-out file hash, and
recomputed adapter artifact hash. Distinct SFT and RAFT conditions must not resolve to the same
path or artifact hash. `--allow-unverified-adapter` is only a legacy escape hatch and invalidates a
clean causal interpretation; its warning must remain in both machine and Markdown reports.

## Thinking-mode protocol

The standard concise-QA condition disables Qwen3 thinking and uses greedy decoding with 64 new
tokens. A thinking-enabled run is a different experimental condition: use sampled thinking
decoding, reserve enough output tokens, persist reasoning and answer tokens separately, and score
only the parsed final answer. Never place thinking and non-thinking recipes in one adaptation-only
comparison.
