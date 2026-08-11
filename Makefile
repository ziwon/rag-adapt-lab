.PHONY: install test lint format demo

install:
	uv pip install -e '.[rag,train,wandb,dev]'

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

demo:
	raglab validate-data --documents examples/demo/documents.jsonl --eval-set examples/demo/eval.jsonl
	raglab eval-retrieval --documents examples/demo/documents.jsonl --eval-set examples/demo/eval.jsonl --retriever bm25 --top-k 3
