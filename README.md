<div align="center">

# Pareton (SN10)

**Workload-Specific Inference Optimization. Every Improvement Raises the Baseline.**

[![Discord](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![Docs](https://img.shields.io/badge/docs-pareton.ai-blue)](https://pareton.ai)
[![X](https://img.shields.io/badge/X-@pareton__ai-000000?logo=x&logoColor=white)](https://x.com/pareton_ai)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[Website](https://pareton.ai) | [GitHub](https://github.com/pareton-ai) | [Discord](https://discord.gg/bittensor)

---

</div>

Pareton is a Bittensor subnet (SN10) that runs **inference-optimization campaigns**. Miners submit git (code) patches against a pinned vLLM baseline. Pareton validates patch integrity and allowed changes, builds patches in a reproducible container environment, and benchmarks real performance gains. Improvements that pass become the new floor for the next campaign.

## How It Works

1. **Campaigns** pin a baseline commit, base image digest, allowed/denied path globs, and a content-addressed workload trace. Once open, the manifest is frozen.
2. **Miners** author a git patch against that baseline, upload it via Pareton-presigned S3, and commit `campaign_id`, `baseline_commit`, `patch_hash`, and `retrieval_url` on-chain.
3. **The watcher** (`python -m worker.watcher`) scans SN10 for new commitments and writes submissions to Postgres. **The worker** (`python -m worker`) claims jobs and runs validation gates (identity, integrity, base-apply, surface).
4. **Hermetic build** applies the patch inside the pinned base image and pushes a content-addressed engine image to GHCR.
5. **Bench** (worker-wired) runs correctness, perf screen, and SLA checks. Use `--mock-bench` for an in-process path, or a rented GPU for the live path. On-chain scoring is still design-only.

## Layout

| Path                    | Role                                          |
| ----------------------- | --------------------------------------------- |
| `campaign/`             | Profiles, manifests, seed CLI                 |
| `chain/`                | Patch commitment parse + chain watcher/RPC    |
| `gate/`                 | Patch validation gates a–d                    |
| `builder/`              | Hermetic build + GHCR tagging                 |
| `bench/`                | Correctness / perf / SLA harness              |
| `gpu/`                  | GPU pod rent/provision/destroy + remote bench |
| `storage/`              | Pareton-presigned S3 uploads                  |
| `db/`                   | Neon schema + connection                      |
| `worker/`               | Job loop + chain watcher (`python -m worker.watcher`) |
| `api/`                  | HTTP API (campaigns, submissions, presign)    |
| `miner/commit_patch.py` | Miner commit CLI                              |
| `fixtures/`             | Synthetic campaign fixtures                   |
| `images/baseline/`      | Baseline Dockerfile                           |
| `ops/`                  | Deploy helpers (Vector, Axiom, GPU scripts)   |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
