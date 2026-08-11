# Contributing

Thank you for contributing to `rag-adapt-lab`.

## Design principles

Please keep changes aligned with these principles:

- Domain-specific knowledge belongs in datasets and config, not core code.
- Retrieval, generation, training, evaluation, and tracking should remain replaceable.
- Optional integrations must fail gracefully when their dependency is not installed.
- Benchmark recipes should be comparable under a shared evaluation protocol.
- GPU defaults should be conservative for 16GB and 24GB single-GPU systems.

## Development

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[rag,train,wandb,dev]'
pytest
ruff check .
```

## Pull requests

Please include:

- a concise problem statement;
- tests for new behavior;
- config examples for new backends;
- notes on VRAM/compute requirements for training changes.
