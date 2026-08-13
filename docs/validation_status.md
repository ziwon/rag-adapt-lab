# Validation status

Validation date: 2026-08-13 (Asia/Seoul).

The final scientific-validity implementation and matched-source GPU fixture were validated at
commit `5f717412c1e43e2c8585d3ac9607b510fb3a206b`.

## Results

| Layer | Command | Status |
|---|---|---|
| Ruff | `uv run ruff check .` | passed |
| Unit | `uv run pytest -m 'not integration'` | passed, 123 tests |
| CPU integration | `uv run pytest -m 'integration and not gpu' -vv` | passed, 1 real dependency contract test |
| Compose | `docker compose --env-file .env.compose.example config --quiet` | passed |
| GPU integration | `RAGLAB_RUN_GPU_INTEGRATION=1 WANDB_MODE=disabled uv run pytest tests/integration/test_gpu_integration.py -m gpu -vv` | passed, 1 test |

GPU integration: passed locally at commit
`5f717412c1e43e2c8585d3ac9607b510fb3a206b` on an NVIDIA GeForce RTX 5080 (16,303 MiB reported),
driver 580.173.02, PyTorch 2.11.0+cu130, Python 3.11.13.

The GPU test created a tiny local causal LM and tokenizer, completed real SFT and RAFT LoRA
training from matched train/validation source rows, saved weight files and schema-v3 manifests,
asserted equal source/control fingerprints and unequal adapter artifact hashes, reloaded both
adapters, executed Base/RAG/SFT+RAG/RAFT+RAG, produced EM and Token F1, recorded non-null allocated
and reserved VRAM, wrote prediction/summary/report artifacts, and validated the generated schemas.

No GitHub Actions GPU run is claimed: the repository had zero registered self-hosted runners and
no historical `gpu-integration.yml` runs when checked. The successful evidence above is the
documented local CUDA run.

## Scope limits

- Token-level thinking parsing is unit validated with a tokenizer whose thinking tags disappear
  under `skip_special_tokens=True`.
- Qwen3 thinking-mode generation itself was not executed end to end on GPU and is not claimed as
  GPU validated.
- The successful GPU integration used the explicit concise, non-thinking tiny-model condition.
