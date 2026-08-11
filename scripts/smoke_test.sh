#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src tests
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m rag_adapt_lab.cli validate-data \
  --documents examples/demo/documents.jsonl \
  --eval-set examples/demo/eval.jsonl

echo "Core smoke test passed."
