"""Durable Pareton SSH keypair under the GPU state dir."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from gpu.errors import GpuError
from gpu.registry import _state_dir

logger = logging.getLogger(__name__)

KEY_BASENAME = "pareton-gpu-ed25519"


def durable_key_paths(state_dir: Path | None = None) -> tuple[Path, Path]:
    root = _state_dir(state_dir)
    keys = root / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    try:
        keys.chmod(0o700)
    except OSError:
        pass
    priv = keys / KEY_BASENAME
    pub = keys / f"{KEY_BASENAME}.pub"
    return priv, pub


def ensure_durable_keypair(state_dir: Path | None = None) -> tuple[Path, Path]:
    """Return (private, public) paths; generate once if missing."""
    priv, pub = durable_key_paths(state_dir)
    if priv.is_file() and pub.is_file():
        try:
            priv.chmod(0o600)
        except OSError:
            pass
        return priv, pub
    if priv.is_file() and not pub.is_file():
        raise GpuError(f"private key exists without public key: {priv}")
    cmd = [
        "ssh-keygen",
        "-t",
        "ed25519",
        "-N",
        "",
        "-f",
        str(priv),
        "-C",
        "pareton-gpu",
    ]
    logger.info("generating durable Pareton SSH key at %s", priv)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
    except FileNotFoundError as exc:
        raise GpuError("ssh-keygen not found on PATH") from exc
    if proc.returncode != 0:
        raise GpuError(
            f"ssh-keygen failed: {(proc.stderr or proc.stdout).strip()[-500:]}"
        )
    try:
        priv.chmod(0o600)
    except OSError:
        pass
    return priv, pub


def read_public_key(state_dir: Path | None = None) -> str:
    _priv, pub = ensure_durable_keypair(state_dir)
    return pub.read_text(encoding="utf-8").strip()
