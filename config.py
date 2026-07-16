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
GHCR_TOKEN: str = os.environ.get("PARETON_GHCR_TOKEN", "")
GHCR_USERNAME: str = os.environ.get("PARETON_GHCR_USERNAME", "")
BASE_IMAGE: str = os.environ.get(
    "PARETON_BASE_IMAGE", "ghcr.io/pareton-ai/pareton-baseline:v0"
)
BUILD_TIMEOUT_S: int = int(os.environ.get("PARETON_BUILD_TIMEOUT_S", "1800"))
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
