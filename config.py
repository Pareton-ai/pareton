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

# Tags the weight vector with our mechanism version. Other validators only
# trust-weight us while their key agrees with ours, so a bump is a coordinated
# announcement, never a deploy artifact: it is set by hand in the env file.
VERSION_KEY: int = int(os.environ.get("PARETON_VERSION_KEY", "2032"))
# Emission not claimed by a seated leader goes here.
BURN_UID: int = int(os.environ.get("PARETON_BURN_UID", "201"))
# How often pareton-weights recomputes. 360 blocks is about 72 minutes.
WEIGHTS_CADENCE_BLOCKS: int = int(
    os.environ.get("PARETON_WEIGHTS_CADENCE_BLOCKS", "360")
)
# Kill switch, not a dry-run default. Off means compute and store but never
# sign, so /v1/weights stays truthful while the chain call is paused.
WEIGHTS_ENABLED: bool = os.environ.get(
    "PARETON_WEIGHTS_ENABLED", "true"
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

POLL_INTERVAL_S: int = int(os.environ.get("PARETON_POLL_INTERVAL_S", "30"))
CHAIN_RETRY_ATTEMPTS: int = int(os.environ.get("PARETON_CHAIN_RETRY_ATTEMPTS", "3"))
CHAIN_RETRY_DELAY_S: int = int(os.environ.get("PARETON_CHAIN_RETRY_DELAY_S", "30"))

# Must stay well under the dashboard stale window (~60s) and PAR-48 reclaim.
JOB_HEARTBEAT_INTERVAL_S: float = float(
    os.environ.get("PARETON_JOB_HEARTBEAT_INTERVAL_S", "12")
)
BENCH_PHASE_POLL_S: float = float(os.environ.get("PARETON_BENCH_PHASE_POLL_S", "20"))

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
# Host dir mounted at the engine profile's cache_dir inside the container.
# Empty means no mount (laptop tests). GPU remote env sets /workspace/engine-cache
# so FlashInfer autotune survives container restarts.
BENCH_ENGINE_CACHE_DIR: str = os.environ.get(
    "PARETON_BENCH_ENGINE_CACHE_DIR", ""
).strip()

# GPU pod orchestration (WS-C)
TARGON_API_KEY: str = os.environ.get("PARETON_TARGON_API_KEY", "")
SHADEFORM_API_KEY: str = os.environ.get("PARETON_SHADEFORM_API_KEY", "")
LIUM_API_KEY: str = os.environ.get("PARETON_LIUM_API_KEY", "")
RUNPOD_API_KEY: str = os.environ.get("PARETON_RUNPOD_API_KEY", "")
RUNPOD_IMAGE: str = os.environ.get(
    "PARETON_RUNPOD_IMAGE",
    "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
)
# SECURE | COMMUNITY | ANY (prefer community when under budget).
RUNPOD_CLOUD: str = (
    os.environ.get("PARETON_RUNPOD_CLOUD", "ANY").strip().upper() or "ANY"
)
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

# Bench worker bounds.
TRACE_MAX_BYTES: int = int(
    os.environ.get("PARETON_TRACE_MAX_BYTES", str(100 * 1024 * 1024))
)
BENCH_TIMEOUT_S: float = float(os.environ.get("PARETON_BENCH_TIMEOUT_S", "10800"))
ALLOW_MOCK_BENCH: bool = os.environ.get("PARETON_ALLOW_MOCK_BENCH", "") == "1"
# Correctness prompts have no env default on purpose: campaigns pin
# bench.correctness.num_prompts, and absent a pin the worker scores every
# request in the trace (PAR-65).
#
# The bars below are absolute logprobs the shared scorer grades a captured
# output against. They are seed-time defaults only: campaign/seed.py copies
# them onto campaigns.bench.correctness.thresholds, and the harness reads them
# from the campaign, so editing these on a pod cannot move a live campaign.
BENCH_CORRECTNESS_MIN_MEAN_LOGPROB: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MIN_MEAN_LOGPROB", "-4.0")
)
BENCH_CORRECTNESS_MIN_TOKEN_LOGPROB: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MIN_TOKEN_LOGPROB", "-12.0")
)
# The min-token bar is applied to the k-th lowest scored position rather than
# the outright minimum, with k = ceil(quantile * positions). Scorer and
# candidate are separate instances of the same image, so numerical divergence
# can flip the argmax at one high-entropy position and have the scorer rate
# the candidate's own token near zero probability. At 0.001 that costs four
# positions in a 4097-position round and leaves the bar otherwise untouched.
BENCH_CORRECTNESS_MIN_TOKEN_QUANTILE: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MIN_TOKEN_QUANTILE", "0.001")
)
# Below this fraction of the candidate's streamed tokens, the scorer never saw
# the output: infrastructure, not a wrong answer.
BENCH_CORRECTNESS_MIN_COVERAGE_RATIO: float = float(
    os.environ.get("PARETON_BENCH_CORRECTNESS_MIN_COVERAGE_RATIO", "0.5")
)
BENCH_SLA_REPETITIONS: int = int(os.environ.get("PARETON_BENCH_SLA_REPETITIONS", "3"))

# Seed-time defaults for campaigns.emission_rule: what a campaign's leader is
# paid, decaying from start to floor over decay_blocks since the crown was won.
# Blocks, not days, because other validators recompute the same vector and the
# chain is the only clock we share; 201600 blocks is two weeks at 12s. Like the
# correctness bars, these are copied onto the campaign and signed into
# manifest_hash, so editing them cannot move a live campaign's pay schedule.
EMISSION_START_WEIGHT: float = float(
    os.environ.get("PARETON_EMISSION_START_WEIGHT", "0.10")
)
EMISSION_FLOOR_WEIGHT: float = float(
    os.environ.get("PARETON_EMISSION_FLOOR_WEIGHT", "0.02")
)
EMISSION_DECAY_BLOCKS: int = int(
    os.environ.get("PARETON_EMISSION_DECAY_BLOCKS", "201600")
)

# Round seeding reads the block this many behind the head, so the seed block is
# already settled and the watcher never blocks waiting for the tip to finalize.
CHAIN_FINALITY_DEPTH: int = int(os.environ.get("PARETON_CHAIN_FINALITY_DEPTH", "1"))

# Round creation. A round fires at ROUND_SIZE queued challengers, or when the
# oldest queued submission has waited ROUND_MAX_WAIT_S;
ROUND_SIZE: int = int(os.environ.get("PARETON_ROUND_SIZE", "5"))
ROUND_MAX_WAIT_S: int = int(os.environ.get("PARETON_ROUND_MAX_WAIT_S", "21600"))
ROUND_STALE_S: int = int(os.environ.get("PARETON_ROUND_STALE_S", "1800"))
ROUND_MAX_DURATION_S: int = int(os.environ.get("PARETON_ROUND_MAX_DURATION_S", "21600"))
OVERTAKE_EPSILON: float = float(os.environ.get("PARETON_OVERTAKE_EPSILON", "0.01"))
# An out-of-stock market defers a round instead of voiding it. The delay
# doubles per attempt, so a long outage costs a few provider calls per hour.
PROVISION_RETRY_BASE_S: int = int(
    os.environ.get("PARETON_PROVISION_RETRY_BASE_S", "300")
)
PROVISION_RETRY_MAX_S: int = int(
    os.environ.get("PARETON_PROVISION_RETRY_MAX_S", "3600")
)
# Drift is in the same units as the crown decision. The overtake moat is 0.01,
# so a round voids only when the machine moved five times that margin.
BASELINE_DRIFT_CEILING: float = float(
    os.environ.get("PARETON_BASELINE_DRIFT_CEILING", "0.05")
)

# Ordered GPU control-plane try list (first → last), preferred over the legacy
# primary + fallbacks pair. Targon is omitted while its inventory API is
# retired; the provider module stays for when it comes back.
_DEFAULT_GPU_PROVIDERS = ("lium", "shadeform")


def _parse_gpu_providers(raw: str) -> list[str]:
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _load_gpu_providers() -> list[str]:
    raw = os.environ.get("PARETON_GPU_PROVIDERS")
    if raw is not None:
        parsed = _parse_gpu_providers(raw)
        return parsed or list(_DEFAULT_GPU_PROVIDERS)
    # Legacy: PARETON_GPU_PROVIDER + PARETON_GPU_PROVIDER_FALLBACKS.
    if (
        "PARETON_GPU_PROVIDER" in os.environ
        or "PARETON_GPU_PROVIDER_FALLBACKS" in os.environ
    ):
        primary = (os.environ.get("PARETON_GPU_PROVIDER") or "lium").strip().lower()
        if not primary or primary == "auto":
            primary = "lium"
        if "PARETON_GPU_PROVIDER_FALLBACKS" in os.environ:
            fallbacks = _parse_gpu_providers(
                os.environ.get("PARETON_GPU_PROVIDER_FALLBACKS", "")
            )
        else:
            fallbacks = ["shadeform"]
        out: list[str] = []
        for name in [primary, *fallbacks]:
            if name and name != "auto" and name not in out:
                out.append(name)
        return out or list(_DEFAULT_GPU_PROVIDERS)
    return list(_DEFAULT_GPU_PROVIDERS)


GPU_PROVIDERS: list[str] = _load_gpu_providers()
# Back-compat aliases used by older call sites / docs.
GPU_PROVIDER: str = GPU_PROVIDERS[0] if GPU_PROVIDERS else "lium"
GPU_PROVIDER_FALLBACKS: list[str] = list(GPU_PROVIDERS[1:])

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
