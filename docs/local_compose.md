# Local Compose environment

`compose.yaml` provides a repeatable, development-only environment for the public-data workflow:

```text
Browser / host CLI ───────► W&B Server (:8080)
                                │
GPU lab / jobs ────────────────┤ Models metrics and Weave SDK traces
        │                       │
        └───────────────────────┴──► SeaweedFS S3 (:8333)
                                      artifacts and object data
```

The default stack starts SeaweedFS, creates its S3 bucket idempotently, and starts the pinned
`wandb/local` development image. The application image and one-shot preparation, training,
evaluation, and tracking checks are exposed through Compose profiles.

## Scope and limitations

This is a local research stack, not a production W&B deployment. Current W&B Self-Managed
production guidance requires Kubernetes, MySQL 8.4, Redis 7, S3-compatible object storage, and a
W&B Server license. Self-managed Weave additionally requires a Weave-enabled license and
ClickHouse; Weave is enabled inside W&B Platform rather than deployed as an independent Compose
service. See the official [W&B Self-Managed overview](https://docs.wandb.ai/platform/hosting/self-managed/ref-arch)
and [self-managed Weave guide](https://docs.wandb.ai/weave/guides/platform/weave-self-managed).

The Compose file therefore does not invent an unsupported standalone Weave server. The Python
Weave SDK uses `WANDB_BASE_URL=http://wandb:8080` and sends traces to the W&B instance when the
instance's license enables Weave. Without that entitlement, W&B Models still works but the
`tracking-smoke` Weave portion will be rejected by the server.

SeaweedFS 4.41 supplies the S3-compatible development store. W&B is configured to proxy object
traffic, so its internal `seaweedfs:8333` endpoint never needs to resolve in the host browser.
SeaweedFS's S3 CORS origin is restricted to the local W&B URL. For SeaweedFS configuration details,
see the project's [official Docker examples](https://github.com/seaweedfs/seaweedfs/tree/4.41/docker).

## Prerequisites

- Docker Engine with Compose V2.
- NVIDIA Container Toolkit and a driver compatible with the locked PyTorch/CUDA environment for
  the `lab`, `train`, and `evaluate` GPU services.
- A W&B Server license for self-hosted W&B, and a Weave-enabled license to store Weave traces.
- Approximately 5 GB just for the W&B development image, plus model and Python dependency caches.

Check GPU container access before training:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

## Configure and start

```bash
cp .env.compose.example .env.compose
```

Edit `.env.compose` and set at least `WANDB_LICENSE`. The included S3 credentials are deliberately
local-only defaults; replace them if any port will be exposed beyond loopback.

Validate and start the infrastructure:

```bash
docker compose --env-file .env.compose config --quiet
docker compose --env-file .env.compose up -d --wait seaweedfs seaweed-init wandb
```

Equivalent Make targets are available:

```bash
make compose-config COMPOSE_ENV=.env.compose
make compose-up COMPOSE_ENV=.env.compose
```

Open these local endpoints:

- W&B: <http://localhost:8080>
- SeaweedFS S3 API: <http://localhost:8333>
- SeaweedFS filer UI: <http://localhost:8888>
- SeaweedFS master UI: <http://localhost:9333>

Finish W&B's first-run setup, create or join a team, and copy a user API key into
`WANDB_API_KEY` in `.env.compose`. For Weave, set `WEAVE_PROJECT` to the project path expected by
your W&B organization, commonly `entity/project`.

## Use the lab and jobs

Start an interactive GPU container:

```bash
docker compose --env-file .env.compose --profile lab up -d --build --wait lab
docker compose --env-file .env.compose exec lab bash
```

Run the public SQuAD workflow as explicit, restartable jobs. Preparation emits both ordinary SFT and hard-negative RAFT data. Train the two adapters against the same base revision and training source split, then execute the four-recipe benchmark:

```bash
docker compose --env-file .env.compose run --rm prepare
TRAIN_RECIPE=configs/recipes/hf-squad-sft-smoke.yaml \
  TRAIN_FILE=data/hf_squad_smoke/sft_train.jsonl \
  docker compose --env-file .env.compose run --rm train
TRAIN_RECIPE=configs/recipes/hf-squad-raft-smoke.yaml \
  TRAIN_FILE=data/hf_squad_smoke/raft_train.jsonl \
  docker compose --env-file .env.compose run --rm train
docker compose --env-file .env.compose run --rm benchmark
```

The `train` service passes the held-out evaluation file for leakage checking. The `benchmark` service writes `outputs/hf-squad-benchmark/summary.json`, `report.md`, and per-recipe predictions. Its tracker defaults to `none`; set `BENCHMARK_TRACKING_BACKEND=wandb` to log metrics and artifacts to the local W&B service. The older `evaluate` job remains as a focused Base-vs-RAFT/oracle diagnostic.

Verify W&B Models logging and a Weave trace after configuring the API key and licenses:

```bash
docker compose --env-file .env.compose run --rm tracking-smoke
```

`data/` and `outputs/` are bind-mounted so generated datasets, adapters, and evaluations remain
available on the host. The Hugging Face cache and service state use named volumes.

## Stop and inspect

```bash
docker compose --env-file .env.compose ps
docker compose --env-file .env.compose logs -f wandb
docker compose --env-file .env.compose down
```

`down` preserves the SeaweedFS, W&B, and Hugging Face volumes. Adding `--volumes` permanently
deletes that local state, so it is intentionally not part of the Make target.

## Reproducibility notes

- Third-party service images and the CUDA/uv build images are pinned by version and digest.
- Python direct dependencies are constrained to the versions used by the validated public run,
  and `uv.lock` pins the complete dependency graph with artifact hashes.
- The application image installs with `uv sync --frozen`; a changed `pyproject.toml` therefore
  requires an intentional `uv lock` update.
- Hugging Face model and dataset revisions remain immutable commit SHAs.
