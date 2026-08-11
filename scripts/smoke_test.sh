#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m rag_adapt_lab.cli validate-data \
  --documents examples/demo/documents.jsonl \
  --eval-set examples/demo/eval.jsonl
PYTHONPATH=src python -m rag_adapt_lab.cli benchmark \
  --recipes base,rag \
  --model-config configs/models/qwen2.5-0.5b-instruct.yaml \
  --documents examples/demo/documents.jsonl \
  --eval-set examples/demo/eval.jsonl \
  --dry-run >/dev/null

echo "Core smoke test passed."
