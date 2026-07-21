"""Environment-driven defaults for Pareton Stage 0."""

from __future__ import annotations

import os
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
WORK_DIR: Path = Path(
    os.environ.get("PARETON_WORK_DIR", str(REPO_ROOT / ".pareton-work"))
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

# GPU pod orchestration (WS-C)
TARGON_API_KEY: str = os.environ.get("PARETON_TARGON_API_KEY", "")
SHADEFORM_API_KEY: str = os.environ.get("PARETON_SHADEFORM_API_KEY", "")
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
GHCR_USER: str = os.environ.get(
    "PARETON_GHCR_USER", os.environ.get("PARETON_GHCR_USERNAME", "")
)

# WS-D bench worker (pre-B7 calibration placeholders — not production thresholds)
TRACE_MAX_BYTES: int = int(
    os.environ.get("PARETON_TRACE_MAX_BYTES", str(100 * 1024 * 1024))
)
BENCH_TIMEOUT_S: float = float(os.environ.get("PARETON_BENCH_TIMEOUT_S", "10800"))
ALLOW_MOCK_BENCH: bool = os.environ.get("PARETON_ALLOW_MOCK_BENCH", "") == "1"
BENCH_CORRECTNESS_NUM_PROMPTS: int = int(
    os.environ.get("PARETON_BENCH_CORRECTNESS_NUM_PROMPTS", "8")
)
BENCH_CORRECTNESS_MAX_NEW_TOKENS: int = int(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MAX_NEW_TOKENS", "32")
)
BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MEAN_ABS_LOGPROB_DIFF", "0.005")
)
BENCH_CORRECTNESS_MAX_ABS_LOGPROB_DIFF: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MAX_ABS_LOGPROB_DIFF", "0.05")
)
BENCH_CORRECTNESS_ARGMAX_MISMATCH_RATE: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_ARGMAX_MISMATCH_RATE", "0.001")
)
BENCH_PERF_NUM_REQUESTS: int = int(
    os.environ.get("PARETON_BENCH_PERF_NUM_REQUESTS", "8")
)
BENCH_PERF_CONCURRENCY: int = int(os.environ.get("PARETON_BENCH_PERF_CONCURRENCY", "2"))
BENCH_PERF_MIN_THROUGHPUT_RATIO: float = float(
    os.environ.get("PARETON_BENCH_PERF_MIN_THROUGHPUT_RATIO", "1.0")
)
BENCH_SLA_REPETITIONS: int = int(os.environ.get("PARETON_BENCH_SLA_REPETITIONS", "3"))
BENCH_SLA_WARMUP_REQUESTS: int = int(
    os.environ.get("PARETON_BENCH_SLA_WARMUP_REQUESTS", "1")
)
GPU_PROVIDER: str = os.environ.get("PARETON_GPU_PROVIDER", "targon")

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
