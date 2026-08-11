# Schema-v2 migration notes

## Adapter loading

Adapters without `raglab_adapter_manifest.json` schema v2 now fail closed. Re-train the adapter to
capture the prompt, split, held-out hash, configuration, and artifact identity. For a legacy-only
diagnostic run, pass `--allow-unverified-adapter`; the resulting summary and report are explicitly
marked unverified. Wrong base models still cannot be overridden.

SFT and RAFT paths must declare `adaptation_mode: sft` and `adaptation_mode: raft` respectively.
The same path or artifact hash cannot represent both conditions by default.

## Metric renames

Schema v2 uses precise names:

| Schema v1 | Schema v2 | Status |
|---|---|---|
| `latency_s` | `model_generate_latency_s` | v1 alias retained temporarily |
| `end_to_end_latency_s` | `inference_e2e_latency_s` | v1 alias retained temporarily |
| `tokens_per_second` | `output_tokens_per_model_generate_second` | v1 alias retained temporarily |
| `peak_gpu_vram_gb` | `peak_allocated_vram_gb` | v1 alias retained temporarily |
| `answer_correctness` | `reference_overlap` | renamed; no ambiguous alias |
| `groundedness` | `lexical_groundedness` | renamed; no ambiguous alias |
| `unsupported_claim_rate` | `lexical_unsupported_claim_rate` | renamed; no ambiguous alias |

`peak_reserved_vram_gb`, detailed inference stages, deterministic `scoring_latency_s`, and separate
`judge_latency_s` are new. Judge coverage/failure/cache counts are aggregate metrics.

## Prompt and Qwen3 behavior

SFT no longer uses a separate instruction/input wording. SFT and RAFT now share
`rag-user-prompt` v3, differing only in supplied contexts. All Qwen3 configs must explicitly set
`chat_template_kwargs.enable_thinking`; an omitted or incompatible thinking/generation condition
is rejected.

## Training splits

Training configs support `split.strategy`, `split.group_by`, and `split.corpus_policy`. Existing
row-level splitting remains callable for compatibility, but the shipped configs use grouped
splits. For strict RAFT isolation, generate explicit train/validation files with
`prepare-raft --validation-output ... --split-config ...` so mining occurs after partitioning.
