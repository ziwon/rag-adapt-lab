# Validation Status

`rag-adapt-lab` is an executable single-GPU research harness rather than an orchestration-only
scaffold. It includes the Base/RAG/SFT+RAG/RAFT+RAG runner, paired bootstrap reporting,
hard-negative RAFT data preparation, validation-aware LoRA/QLoRA training, fail-closed adapter
provenance, detailed systems metrics, and optional robust judge scoring.

Validation is intentionally reported in layers:

- Unit CI installs `.[dev]` and exercises core contracts without optional ML dependencies.
- CPU integration CI installs the exact `.[rag,train,dev]` versions and exercises real BM25,
  Datasets, Transformers tokenizer formatting, TRL `SFTConfig`, completion-only configuration,
  manifest verification, benchmark planning/reporting, and Compose parsing.
- GPU integration is a separate scheduled/manual workflow for actual tiny LoRA training,
  adapter save/load, base-versus-adapter generation, CUDA memory measurement, and one benchmark.

Passing unit or CPU integration does not imply that CUDA was exercised. A checkout should claim
GPU end-to-end validation only when its GPU workflow or a documented local CUDA run completed.

Known extension points remain batched/vLLM generation, built-in dense/hybrid negative-mining
strategies, semantic claim decomposition, judge calibration against human labels, and
multiple-comparison corrections. Distributed training and production orchestration remain out of
scope.
