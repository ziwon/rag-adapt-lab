# Schema v3 migration

Schema v3 makes the SFT-versus-RAFT comparison population and prompt contract verifiable. It is intentionally fail-closed: schema-v2 adapters must be retrained or explicitly benchmarked with the unverified-adapter override.

## Adapter and training manifests

Fresh training writes `training_source_fingerprint` and `validation_source_fingerprint`. These hashes are computed from sorted source IDs, normalized questions, and target answers, so they remain equal across SFT and RAFT representations while the full dataset fingerprints continue to capture their different prompts and contexts. A canonical four-condition benchmark rejects verified SFT and RAFT adapters whose source fingerprints differ.

Prompt v4 hashes the empty-, one-, and multi-document rendering branches. Verifiable training now requires `use_chat_template: true`; a plain prompt-completion representation cannot claim the model-facing inference prompt identity.

Re-run training to produce schema-v3 manifests. Do not relabel an existing schema-v2 manifest: it lacks the evidence needed to establish these contracts.

## Split configuration

`training.split.validation_ratio` and `training.split.seed` take precedence over the legacy top-level validation ratio and seed. Explicit train/validation files are checked against every configured `group_by` field as well as normalized questions and the corpus policy.

## Judge operation

The persistent judge cache is SQLite and defaults to `.cache/raglab/judge-cache.sqlite3`. Choose a new path rather than pointing at a legacy JSON cache. Judge calls stream their response, stop when `max_response_bytes` is exceeded, request `max_completion_tokens`, and cap structured-output rationale length with `max_rationale_characters`.
