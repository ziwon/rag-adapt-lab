.PHONY: install test test-unit test-integration lint format demo compose-config compose-up compose-down compose-lab

COMPOSE_ENV ?= .env.compose.example
COMPOSE = docker compose --env-file $(COMPOSE_ENV)

install:
	uv pip install -e '.[rag,train,wandb,dev]'

test:
	pytest

test-unit:
	pytest -m 'not integration and not gpu'

test-integration:
	pytest -m 'integration and not gpu'

lint:
	ruff check .

format:
	ruff format .

demo:
	raglab validate-data --documents examples/demo/documents.jsonl --eval-set examples/demo/eval.jsonl
	raglab eval-retrieval --documents examples/demo/documents.jsonl --eval-set examples/demo/eval.jsonl --retriever bm25 --top-k 3

compose-config:
	$(COMPOSE) config --quiet

compose-up:
	$(COMPOSE) up -d --wait seaweedfs seaweed-init wandb

compose-down:
	$(COMPOSE) down

compose-lab:
	$(COMPOSE) --profile lab up -d --build --wait lab
