from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_compose() -> dict[str, object]:
    return yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))


def test_compose_has_pinned_local_infrastructure() -> None:
    compose = load_compose()
    services = compose["services"]
    assert isinstance(services, dict)
    assert {"seaweedfs", "seaweed-init", "wandb", "lab", "tracking-smoke"} <= services.keys()

    for service_name in ("seaweedfs", "seaweed-init", "wandb"):
        service = services[service_name]
        assert isinstance(service, dict)
        image = service["image"]
        assert isinstance(image, str)
        assert "@sha256:" in image
        assert ":latest" not in image


def test_compose_wires_wandb_to_seaweedfs_without_public_bindings() -> None:
    compose = load_compose()
    services = compose["services"]
    assert isinstance(services, dict)
    wandb = services["wandb"]
    seaweedfs = services["seaweedfs"]
    assert isinstance(wandb, dict)
    assert isinstance(seaweedfs, dict)

    environment = wandb["environment"]
    assert isinstance(environment, dict)
    assert "seaweedfs:8333" in environment["BUCKET"]
    assert environment["BUCKET_PROXY"] == "true"
    assert environment["GORILLA_FILE_STORE_IS_PROXIED"] == "true"

    for service in (wandb, seaweedfs):
        ports = service["ports"]
        assert isinstance(ports, list)
        assert all(str(port).startswith("127.0.0.1:") for port in ports)


def test_application_image_uses_frozen_lockfile() -> None:
    dockerfile = (PROJECT_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "uv sync --frozen" in dockerfile
    assert "COPY pyproject.toml uv.lock" in dockerfile
    assert dockerfile.count("@sha256:") == 2
    assert (PROJECT_ROOT / "uv.lock").is_file()
