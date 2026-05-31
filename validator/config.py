"""Environment-driven defaults for the validator process.

Paths, poll interval, wallet names, subnet id, and timeouts are read from
``CACHEON_*`` variables so one codebase can target testnet vs mainnet,
different machines, and dry-run mode without editing source.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NETUID: int = int(os.environ.get("CACHEON_NETUID", "14"))

SUBTENSOR_NETWORK: str = os.environ.get("CACHEON_NETWORK", "finney")
"""Bittensor network name. `finney` = mainnet, `test` = testnet, or an ws:// URL."""

WALLET_NAME: str = os.environ.get("CACHEON_WALLET_NAME", "default")
WALLET_HOTKEY: str = os.environ.get("CACHEON_WALLET_HOTKEY", "default")

POLL_INTERVAL_S: int = int(os.environ.get("CACHEON_POLL_INTERVAL_S", "600"))
"""Seconds to sleep when there's nothing new to evaluate. Docker eval takes
minutes; reacting faster than this buys nothing."""

CHAIN_RETRY_ATTEMPTS: int = int(os.environ.get("CACHEON_CHAIN_RETRY_ATTEMPTS", "3"))
CHAIN_RETRY_DELAY_S: int = int(os.environ.get("CACHEON_CHAIN_RETRY_DELAY_S", "30"))

STATE_DIR: Path = Path(
    os.environ.get("CACHEON_STATE_DIR", str(REPO_ROOT / "state-mainnet"))
).resolve()
"""Where local JSON state files live. Gitignored."""

DRY_RUN: bool = os.environ.get("CACHEON_DRY_RUN", "0") == "1"
"""When True, skip `subtensor.set_weights()` and do not run Docker eval.
Useful for testing the loop without touching the chain."""

VERSION_KEY: int = int(os.environ.get("CACHEON_VERSION_KEY", "29"))
"""Version tag passed as `version_key` to `subtensor.set_weights(...)`. Bump
whenever the scoring mechanism or evaluation rules change in a way that would
produce different winner selections on identical commits.

Yuma consensus only trust-weights validators that agree on the version, so
bumping this effectively rolls consensus to the new version once a quorum of
stake has upgraded."""

MODEL_VOLUME: str = os.environ.get("CACHEON_MODEL_VOLUME", "/models")
"""Host path mounted read-only into miner/baseline containers at ``/models``."""

MODEL_PATH: str = os.environ.get("CACHEON_MODEL_PATH", "/models")
"""Path to read model config.json inside the gpu-eval container (mounted model dir)."""

BASELINE_IMAGE: str = os.environ.get(
    "CACHEON_BASELINE_IMAGE", "vllm/vllm-openai:v0.22.0"
)
"""Pass 1"""
BASELINE_DIGEST: str = os.environ.get(
    "CACHEON_BASELINE_DIGEST",
    "sha256:0fec7ec5f3e6bc168e54899935fb0557da908a4832a1dbc88e2debcf2f889416",
)

SCORING_IMAGE: str = os.environ.get("CACHEON_SCORING_IMAGE", "vllm/vllm-openai:v0.9.2")
"""vLLM image used for teacher-forcing correctness (prompt_logprobs). Pinned to v0.9.2 for B200 (sm_100) support"""

GPU_COUNT: int = int(os.environ.get("CACHEON_GPU_COUNT", "8"))
"""Number of GPUs on the host. Set to 8 for 8x H200/B200/B300 (the standard eval tier)."""

# --------------------------------------------------------------------------- #
# GPU orchestration (auto-rent)
# --------------------------------------------------------------------------- #

AUTO_RENT: bool = os.environ.get("CACHEON_AUTO_RENT", "0") == "1"
"""When True, the validator automatically rents a GPU pod when challengers
are detected, runs eval, and tears it down."""

PREFERRED_PROVIDER: str = os.environ.get("CACHEON_PREFERRED_PROVIDER", "")
"""If set to 'lium' or 'targon', only that provider is used for GPU rental
even when both API keys are configured. Empty means cheapest-wins."""

LIUM_API_KEY: str = os.environ.get("LIUM_API_KEY", "")
TARGON_API_KEY: str = os.environ.get("TARGON_API_KEY", "")
SHADEFORM_API_KEY: str = os.environ.get("SHADEFORM_API_KEY", "")
TARGON_VOLUME_UID: str = os.environ.get("TARGON_VOLUME_UID", "")

MAX_HOURLY_PRICE: int = int(os.environ.get("CACHEON_MAX_HOURLY_PRICE", "2000"))
"""Maximum hourly price in US cents. Refuse to rent above this."""

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
"""Hugging Face token passed to the remote pod for model download."""

HIPPIUS_ACCESS_KEY: str = os.environ.get("HIPPIUS_ACCESS_KEY", "")
HIPPIUS_SECRET_KEY: str = os.environ.get("HIPPIUS_SECRET_KEY", "")
S3_BUCKET: str = os.environ.get("CACHEON_S3_BUCKET", "cacheon-validator")
S3_PREFIX: str = os.environ.get("CACHEON_S3_PREFIX", "state-mainnet")

SKIP_S3: bool = os.environ.get("CACHEON_SKIP_S3", "0") == "1"
"""When True, ``gpu_eval`` skips Hippius S3 download and upload (local pod testing)."""

# --------------------------------------------------------------------------- #
# Winner defender-advantage window
# --------------------------------------------------------------------------- #

WINNER_EPSILON_INITIAL: float = float(
    os.environ.get("CACHEON_WINNER_EPSILON_INITIAL", "0.01")
)
"""Fixed 1% moat: a challenger must strictly beat
`winner.score * (1 + WINNER_EPSILON_INITIAL)` to overtake the leader.

1% is small enough to not protect truly weak winners, large enough to swallow
float noise and discourage copycat submissions that match the winner
byte-for-byte (a byte-identical copy also trips the `duplicate_of_leader` DQ
path in `state.record_evaluation`; the epsilon covers near-duplicates /
scoring noise). Leader and runner-up are re-evaluated each round, so the moat
is always applied against a fresh score."""

# --------------------------------------------------------------------------- #
# Competition weight distribution
# --------------------------------------------------------------------------- #

WINNER_WEIGHT_SHARE: float = float(
    os.environ.get("CACHEON_WINNER_WEIGHT_SHARE", "0.80")
)
"""Fraction of the competition pool allocated to the winner."""

RUNNER_UP_WEIGHT_SHARE: float = float(
    os.environ.get("CACHEON_RUNNER_UP_WEIGHT_SHARE", "0.20")
)
"""Fraction of the competition pool allocated to the runner-up.
When no runner-up exists, the winner receives 100% of the pool."""

EMISSION_RAMP_START_BLOCK: int = 8_309_900
"""Mainnet block where the competition emission ramp begins (10% pool)."""

EMISSION_RAMP_END_BLOCK: int = 9_166_700
"""Mainnet block where the competition pool reaches 100% of emission."""

EMISSION_PRE_RAMP_FRAC: float = 0.02
"""Competition pool fraction before EMISSION_RAMP_START_BLOCK."""

EMISSION_RAMP_START_FRAC: float = 0.10
"""Competition pool fraction at EMISSION_RAMP_START_BLOCK."""

EMISSION_RAMP_END_FRAC: float = 1.0
"""Competition pool fraction at and after EMISSION_RAMP_END_BLOCK."""

_emission_override_raw = os.environ.get("CACHEON_EMISSION_FRAC_OVERRIDE", "").strip()
EMISSION_FRAC_OVERRIDE: float | None = (
    float(_emission_override_raw) if _emission_override_raw else None
)
"""When set, replaces the block-scheduled competition pool fraction.
Validator-operator env only; clamped to [0.0, 1.0] at use time."""

BURN_UID: int = int(os.environ.get("CACHEON_BURN_UID", "29"))
"""UID that receives emission not allocated to the competition pool.
Must not collide with the winner or runner-up UID; the weight builder
folds burn weight into the winner on collision."""

# --------------------------------------------------------------------------- #
# Teacher-forcing correctness gate
# --------------------------------------------------------------------------- #

MEAN_LOGPROB_THRESHOLD: float = float(
    os.environ.get("CACHEON_MEAN_LOGPROB_THRESHOLD", "-4.0")
)
"""Mean per-token logprob (from baseline scoring pass) below which a miner
is DQ'd. Legitimate cross-model outputs score -0.5 to -2.0; garbage scores
below -10."""

MIN_LOGPROB_THRESHOLD: float = float(
    os.environ.get("CACHEON_MIN_LOGPROB_THRESHOLD", "-12.0")
)
"""Floor logprob for any single token. Catches isolated garbage tokens."""

PASS1_MATCH_DQ_THRESHOLD: float = float(
    os.environ.get("CACHEON_PASS1_MATCH_DQ_THRESHOLD", "0.10")
)
"""Cheap pre-filter: aggregate token match vs baseline must be at or above
this fraction before teacher-forcing. Below threshold, miner is DQ'd without
scoring. Not the authoritative correctness gate (teacher-forcing is)."""

VLLM_COMPILE_CACHE_DIR: str = os.environ.get("CACHEON_VLLM_CACHE_DIR", "")
"""Host directory mounted into Pass 1 baseline container at /root/.cache/vllm.
Empty disables the mount. Auto-rent on Targon sets /workspace/vllm-cache;
set manually on persistent-volume Targon pods."""


def baseline_compile_cache_dir() -> str | None:
    """Return the compile-cache host path, or None when disabled."""
    path = VLLM_COMPILE_CACHE_DIR.strip()
    return path or None


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #

LOG_RETENTION_DAYS: int = int(os.environ.get("CACHEON_LOG_RETENTION_DAYS", "7"))
"""Delete log files in ``state/logs/`` older than this many days (by filename
timestamp). 0 disables pruning."""

# Submission payment gate (SUBMISSION_FEE_RAO=0 disables verification)
PAYMENT_ADDRESS = "5HKd9UwXBHdAouKvxL73beBhfoW8LyNuMVkUBisboC4CYU41"
SUBMISSION_FEE_RAO: int = int(os.environ.get("CACHEON_SUBMISSION_FEE_RAO", "0"))
