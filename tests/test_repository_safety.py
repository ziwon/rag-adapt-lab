from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_docker_context_excludes_secrets_data_and_model_artifacts() -> None:
    ignored = set((PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {
        ".env",
        ".env.*",
        "data/**",
        "outputs/**",
        "wandb/**",
        ".weave/**",
        "*.safetensors",
        "*.bin",
    } <= ignored
    assert "!.env.example" in ignored
    assert "data" not in ignored  # Do not exclude src/rag_adapt_lab/data from the image.
