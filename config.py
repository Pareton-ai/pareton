"""Environment-driven defaults for Pareton Stage 0."""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

NETUID: int = int(os.environ.get("PARETON_NETUID", "10"))
SUBTENSOR_NETWORK: str = os.environ.get("PARETON_NETWORK", "finney")
WALLET_NAME: str = os.environ.get("PARETON_WALLET_NAME", "default")
WALLET_HOTKEY: str = os.environ.get("PARETON_WALLET_HOTKEY", "default")

POLL_INTERVAL_S: int = int(os.environ.get("PARETON_POLL_INTERVAL_S", "30"))
CHAIN_RETRY_ATTEMPTS: int = int(os.environ.get("PARETON_CHAIN_RETRY_ATTEMPTS", "3"))
CHAIN_RETRY_DELAY_S: int = int(os.environ.get("PARETON_CHAIN_RETRY_DELAY_S", "30"))

# S3 (presigned patch uploads). AWS S3: leave PARETON_S3_ENDPOINT_URL empty.
S3_ENDPOINT_URL: str = os.environ.get("PARETON_S3_ENDPOINT_URL", "")
S3_ACCESS_KEY: str = os.environ.get("PARETON_S3_ACCESS_KEY", "")
S3_SECRET_KEY: str = os.environ.get("PARETON_S3_SECRET_KEY", "")
S3_BUCKET: str = os.environ.get("PARETON_S3_BUCKET", "pareton-s3")
S3_PREFIX: str = os.environ.get("PARETON_S3_PREFIX", "stage0")
S3_REGION: str = os.environ.get("PARETON_S3_REGION", "us-east-2")
S3_PUBLIC_BASE_URL: str = os.environ.get("PARETON_S3_PUBLIC_BASE_URL", "")
PRESIGN_EXPIRES_S: int = int(os.environ.get("PARETON_PRESIGN_EXPIRES_S", "3600"))

# Patch fetch bounds
PATCH_MAX_BYTES: int = int(
    os.environ.get("PARETON_PATCH_MAX_BYTES", str(5 * 1024 * 1024))
)
PATCH_FETCH_TIMEOUT_S: float = float(
    os.environ.get("PARETON_PATCH_FETCH_TIMEOUT_S", "30")
)
PATCH_FETCH_RETRIES: int = int(os.environ.get("PARETON_PATCH_FETCH_RETRIES", "3"))

# Builder / registry (GitHub org: Pareton-ai; GHCR namespaces are lowercase)
GHCR_OWNER: str = os.environ.get("PARETON_GHCR_OWNER", "pareton-ai")
GHCR_IMAGE: str = os.environ.get("PARETON_GHCR_IMAGE", "pareton-engine")
GHCR_BASELINE_IMAGE: str = os.environ.get(
    "PARETON_GHCR_BASELINE_IMAGE", "pareton-baseline"
)
GHCR_TOKEN: str = os.environ.get("PARETON_GHCR_TOKEN", "")
GHCR_USERNAME: str = os.environ.get("PARETON_GHCR_USERNAME", "")
# Mutable tag fallback for local/dev only. Production builds use campaign
# base_image_digest via builder.registry.baseline_build_image_ref.
BASE_IMAGE: str = os.environ.get(
    "PARETON_BASE_IMAGE", "ghcr.io/pareton-ai/pareton-baseline:v0"
)
# Hermetic vLLM builds are long; GHA A2b may need the higher end.
BUILD_TIMEOUT_S: int = int(os.environ.get("PARETON_BUILD_TIMEOUT_S", "7200"))
# Compile parallelism (OOM-safe default). Scale ≤ vCPUs and ≤ RAM/3GB on bigger hosts.
BUILD_MAX_JOBS: int = int(os.environ.get("PARETON_BUILD_MAX_JOBS", "1"))
WORK_DIR: Path = Path(
    os.environ.get("PARETON_WORK_DIR", str(REPO_ROOT / ".pareton-work"))
).resolve()
# Durable per-submission build logs: <BUILD_LOG_DIR>/<submission_id>/build.log.
# The GB-sized docker context lives under WORK_DIR and is deleted after build.
BUILD_LOG_DIR: Path = Path(
    os.environ.get("PARETON_BUILD_LOG_DIR", "/var/log/pareton/builds")
).resolve()

# Bench harness (engine lifecycle). Overridable per-call; defaults for fresh pods.
BENCH_HEALTH_TIMEOUT_S: float = float(
    os.environ.get("PARETON_BENCH_HEALTH_TIMEOUT_S", "600")
)
BENCH_HEALTH_POLL_S: float = float(os.environ.get("PARETON_BENCH_HEALTH_POLL_S", "2"))
BENCH_ENGINE_PORT: int = int(os.environ.get("PARETON_BENCH_ENGINE_PORT", "8000"))
BENCH_DOCKER_PULL_TIMEOUT_S: float = float(
    os.environ.get("PARETON_BENCH_DOCKER_PULL_TIMEOUT_S", "1800")
)
BENCH_DOCKER_CMD_TIMEOUT_S: float = float(
    os.environ.get("PARETON_BENCH_DOCKER_CMD_TIMEOUT_S", "120")
)
BENCH_HF_CACHE_DIR: Path = Path(
    os.environ.get(
        "PARETON_BENCH_HF_CACHE_DIR",
        str(Path.home() / ".cache" / "pareton" / "hf"),
    )
).expanduser()
# Host dir mounted at /root/.cache/sglang inside the engine container.
# Empty means no mount (laptop tests). GPU remote env sets /workspace/sglang-cache
# so FlashInfer autotune survives container restarts.
BENCH_ENGINE_CACHE_DIR: str = os.environ.get(
    "PARETON_BENCH_ENGINE_CACHE_DIR", ""
).strip()
# Z-calibrate needs correctness + perf_screen only. Skip sla_bench when set.
BENCH_SKIP_SLA: bool = os.environ.get("PARETON_BENCH_SKIP_SLA", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# GPU pod orchestration (WS-C)
TARGON_API_KEY: str = os.environ.get("PARETON_TARGON_API_KEY", "")
SHADEFORM_API_KEY: str = os.environ.get("PARETON_SHADEFORM_API_KEY", "")
LIUM_API_KEY: str = os.environ.get("PARETON_LIUM_API_KEY", "")
GPU_TTL_HOURS: float = float(os.environ.get("PARETON_GPU_TTL_HOURS", "2"))
GPU_MAX_HOURLY_CENTS: int = int(os.environ.get("PARETON_GPU_MAX_HOURLY_CENTS", "1000"))
GPU_STATE_DIR: Path = Path(
    os.environ.get(
        "PARETON_GPU_STATE_DIR",
        str(Path.home() / ".cache" / "pareton" / "gpu"),
    )
).expanduser()
GPU_VOLUME_GIB: int = int(os.environ.get("PARETON_GPU_VOLUME_GIB", "250"))
GPU_STATIC_SSH: str = os.environ.get("PARETON_GPU_STATIC_SSH", "")
GPU_SSH_KEY_PATH: str = os.environ.get("PARETON_GPU_SSH_KEY_PATH", "")

_PUBKEY_RE = re.compile(
    r"^(?:sk-)?(?:ssh|ecdsa)-[A-Za-z0-9@.-]+ [A-Za-z0-9+/]{40,}={0,3}(?: \S.*)?$"
)


def _parse_pubkeys(raw: str) -> list[str]:
    """Split newline-separated pubkeys; reject malformed lines loudly."""
    keys: list[str] = []
    for line in raw.splitlines():
        entry = line.strip()
        if not entry:
            continue
        if not _PUBKEY_RE.match(entry):
            raise ValueError(
                "PARETON_GPU_EXTRA_SSH_PUBKEYS has a malformed entry "
                f"(want 'ssh-<type> <base64> [comment]'): {entry[:40]!r}"
            )
        if entry not in keys:
            keys.append(entry)
    return keys


GPU_EXTRA_SSH_PUBKEYS: list[str] = _parse_pubkeys(
    os.environ.get("PARETON_GPU_EXTRA_SSH_PUBKEYS", "")
)
GHCR_USER: str = os.environ.get(
    "PARETON_GHCR_USER", os.environ.get("PARETON_GHCR_USERNAME", "")
)

# WS-D bench worker. Correctness thresholds are B7-calibrated (2026-08-03, below);
# the rest remain provisional until a realistic-trace re-calibration.
TRACE_MAX_BYTES: int = int(
    os.environ.get("PARETON_TRACE_MAX_BYTES", str(100 * 1024 * 1024))
)
BENCH_TIMEOUT_S: float = float(os.environ.get("PARETON_BENCH_TIMEOUT_S", "10800"))
ALLOW_MOCK_BENCH: bool = os.environ.get("PARETON_ALLOW_MOCK_BENCH", "") == "1"
# Correctness prompts and perf_screen requests have no env default on purpose:
# campaigns pin bench.correctness.num_prompts / bench.perf_screen.num_requests,
# and absent a pin the worker scores every request in the trace (PAR-65).
BENCH_CORRECTNESS_MAX_NEW_TOKENS: int = int(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MAX_NEW_TOKENS", "32")
)
# B7 Hopper calibration 2026-08-03 (engine 6abb5786…, safety_factor=2.0).
BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF", "0.0246")
)
BENCH_CORRECTNESS_MAX_ABS_LOGPROB_DIFF: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MAX_ABS_LOGPROB_DIFF", "0.164")
)
BENCH_CORRECTNESS_ARGMAX_MISMATCH_RATE: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_ARGMAX_MISMATCH_RATE", "0.001")
)
BENCH_PERF_CONCURRENCY: int = int(os.environ.get("PARETON_BENCH_PERF_CONCURRENCY", "2"))
# Screen-only smoke gate: single unrepeated measurement, so identical engines
# routinely land under 1.0 (B7 baseline-vs-baseline ratios: 0.970, 0.992, 0.999,
# 0.995, 1.089). The real speedup gate is cross_env.min_speedup_each on
# sla_bench.speedup (median-of-3). 0.95 keeps the screen from false-rejecting.
BENCH_PERF_MIN_THROUGHPUT_RATIO: float = float(
    os.environ.get("PARETON_BENCH_PERF_MIN_THROUGHPUT_RATIO", "0.95")
)
BENCH_SLA_REPETITIONS: int = int(os.environ.get("PARETON_BENCH_SLA_REPETITIONS", "3"))
BENCH_SLA_WARMUP_REQUESTS: int = int(
    os.environ.get("PARETON_BENCH_SLA_WARMUP_REQUESTS", "1")
)

# Dynamic workload sampling + z-score promotion calibration.
CHAIN_FINALITY_DEPTH: int = int(os.environ.get("PARETON_CHAIN_FINALITY_DEPTH", "1"))
# Minimum distinct baseline-vs-baseline samples for campaigns.calibration.
CALIB_MIN_SAMPLES: int = int(os.environ.get("PARETON_CALIB_MIN_SAMPLES", "3"))
# Soft guidance for how many pods / concurrent prepare dirs operators run.
CALIB_PODS: int = int(os.environ.get("PARETON_CALIB_PODS", "1"))
CALIB_SAMPLES_PER_POD: int = int(os.environ.get("PARETON_CALIB_SAMPLES_PER_POD", "50"))
GPU_PROVIDER: str = os.environ.get("PARETON_GPU_PROVIDER", "lium")
# Ordered fallback providers tried after GPU_PROVIDER on no-capacity or
# provision error. Comma-separated; empty string disables fallback.
GPU_PROVIDER_FALLBACKS: list[str] = [
    p.strip().lower()
    for p in os.environ.get("PARETON_GPU_PROVIDER_FALLBACKS", "shadeform").split(",")
    if p.strip()
]

# Per-submission fee the miner transfers to PAYMENT_RECIPIENT_ADDRESS before
# committing; > 0 makes a verified fee proof mandatory. Default off so local
# and mock runs need no payments: set 0.05 in the deployed env (.env.example)
# and keep it > 0 whenever a campaign is open to external miners.
SUBMISSION_FEE_TAO: float = float(os.environ.get("PARETON_SUBMISSION_FEE_TAO", "0.05"))
PAYMENT_RECIPIENT_ADDRESS: str = os.environ.get(
    "PARETON_PAYMENT_RECIPIENT_ADDRESS",
    "5CiieAa5nzSMbw4LPkh2hqv9rfMPZX9ZfEcSjh3SYWNBzk3K",
)

# Axiom observability (Vector log shipping). Ingest-only token; store in
# /opt/pareton/.env at deploy time, never in the repo.
AXIOM_TOKEN: str = os.environ.get("PARETON_AXIOM_TOKEN", "")
AXIOM_DATASET: str = os.environ.get("PARETON_AXIOM_DATASET", "pareton-prod")

API_ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("PARETON_ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]

DEFAULT_ALLOWED_PATHS: list[str] = ["vllm/**"]
DEFAULT_DENIED_PATHS: list[str] = [
    "tests/**",
    "benchmarks/**",
    ".github/**",
    "docker/**",
    "**/Dockerfile*",
    "**/pyproject.toml",
    "**/setup.py",
    "**/setup.cfg",
    "**/requirements*.txt",
    "**/CMakeLists.txt",
]
