# Schema v3 migration

Schema v3 makes the SFT-versus-RAFT comparison population and prompt contract verifiable. It is intentionally fail-closed: schema-v2 adapters must be retrained or explicitly benchmarked with the unverified-adapter override.

## Adapter and training manifests

Fresh training writes `training_source_fingerprint` and `validation_source_fingerprint`. These hashes are computed from sorted source IDs, normalized questions, and target answers, so they remain equal across SFT and RAFT representations while the full dataset fingerprints continue to capture their different prompts and contexts. A canonical four-condition benchmark rejects verified SFT and RAFT adapters whose source fingerprints differ.

The final v0.2 contract also requires `training_controls` and
`training_control_sha256` in both training and adapter manifests. Early schema-v3 artifacts that
lack these fields—or the complete canonical adapter-control fields—no longer pass strict
validation: retrain them rather than inventing provenance.
The normalized object excludes paths, tracking, and the SFT/RAFT representation name but includes
all material LoRA/QLoRA, optimization, effective-batch, sequence, warmup, precision,
quantization, seed, early-stopping, and best-model-selection controls. A mismatch fails closed;
`--allow-unmatched-training-controls` produces a clearly confounded, decision-ineligible report.

The adapter-control section is now checked against the actual PEFT `adapter_config.json`. New
training records PEFT/task types, rank, alpha, dropout, bias, canonical target modules,
modules-to-save, RSLoRA/DoRA, rank/alpha patterns, and supported layer selectors. A schema-v3
manifest that contradicts those persisted settings is invalid even when its own digest is
internally consistent. Training and adapter manifests reference one shared
`training-controls-v1` JSON Schema; effective-batch arithmetic and adaptation-method versus
quantization agreement remain additional runtime semantic checks.

`--allow-unverified-adapter` is narrower than the original schema-v3 release. It permits a missing
manifest or a validated schema-v2 manifest whose historical source/control provenance is
unavailable. It does not permit malformed schema-v3 data or known model, revision, adaptation-mode,
prompt, held-out-evaluation, PEFT-configuration, training-control-hash, or artifact-hash
contradictions. Machine output records `status`, `reason_code`, and `unchecked_fields` for an
accepted legacy adapter. Artifact integrity is never overridable.

RAFT JSONL now has a model-level invariant: `evidence_doc_ids` must equal exactly the unique
`contexts[].doc_id` values marked `relevant=true`. Previously prepared rows that violate the
invariant must be regenerated; they cannot be fingerprinted or trained as valid RAFT data.

Prompt v4 hashes the empty-, one-, and multi-document rendering branches. Verifiable training now requires `use_chat_template: true`; a plain prompt-completion representation cannot claim the model-facing inference prompt identity.

Re-run training to produce schema-v3 manifests. Do not relabel an existing schema-v2 manifest: it lacks the evidence needed to establish these contracts.

## Split configuration

`training.split.validation_ratio` and `training.split.seed` take precedence over the legacy top-level validation ratio and seed. Explicit train/validation files are checked against every configured `group_by` field as well as normalized questions and the corpus policy.

## Judge operation

The persistent judge cache is SQLite and defaults to `.cache/raglab/judge-cache.sqlite3`. Choose a new path rather than pointing at a legacy JSON cache. Judge calls stream their response, stop when `max_response_bytes` is exceeded, request `max_completion_tokens`, and cap structured-output rationale length with `max_rationale_characters`.

Judge aggregates now expose numeric coverage per recipe and paired intersection coverage per
comparison. Configure `minimum_metric_coverage` and `minimum_paired_examples`; results below either
threshold retain diagnostics and raw rows but cannot support decision language.

## CLI and artifact identities

The former structural `--dry-run` behavior moved to `--plan-only`. `--dry-run` now validates every
static protocol condition without loading weights or contacting a judge. Automation that creates
plans before adapters exist must switch to `--plan-only` and check
`validation.adapter_provenance_validated: false`.

The standalone `scripts/evaluate_hf_squad.py` summary is now
`squad-paired-evaluation` schema v1, not `benchmark-summary` v3. RAFT partition manifests and
benchmark plans also have distinct identities. Generated manifests and summaries are validated at
runtime against packaged Draft 2020-12 JSON Schemas; the copies in `docs/schemas/` are the
human-facing mirrors.

## Thinking output

Text searches for `<think>` blocks have been removed. Generated token IDs are split with the
tokenizer-resolved `</think>` token, then reasoning and final-answer sequences are decoded and
counted independently. Thinking-mode output without exactly one valid closing boundary, and
non-thinking output containing either boundary token, now fails closed.
