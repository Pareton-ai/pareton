"""SSH key discovery and loading for GPU providers."""

from __future__ import annotations

import os
from pathlib import Path

import paramiko

# Dedicated deploy key (no passphrase). Prefer over personal ~/.ssh/id_* keys.
_KEY_NAMES = (
    "cacheon-cpu-validator-mbp",
    "cacheon-cpu-validator",
    "id_ed25519",
    "id_rsa",
    "id_ecdsa",
)
_SSH_DIR = Path("~/.ssh").expanduser()


def _private_key_paths() -> list[Path]:
    paths: list[Path] = []
    if env := os.environ.get("CACHEON_SSH_KEY_PATH"):
        paths.append(Path(env).expanduser())
    paths.extend(_SSH_DIR / name for name in _KEY_NAMES)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_private_key(path: Path) -> paramiko.PKey:
    """Load a private key file, trying common paramiko key types."""
    for key_type in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return key_type.from_private_key_file(str(path))
        except (paramiko.SSHException, OSError):
            continue
    raise RuntimeError(f"Could not load private key from {path}")


def discover_ssh_private_key() -> tuple[Path, paramiko.PKey]:
    """Find the first loadable private key under ~/.ssh (or CACHEON_SSH_KEY_PATH)."""
    last_err: RuntimeError | None = None
    for path in _private_key_paths():
        if not path.is_file():
            continue
        try:
            return path, load_private_key(path)
        except RuntimeError as exc:
            last_err = exc
    if last_err:
        raise RuntimeError(f"No loadable SSH private key in {_SSH_DIR} ({last_err})")
    raise RuntimeError(f"No SSH private key found in {_SSH_DIR}")


def discover_ssh_public_key() -> tuple[Path, str]:
    """Return the public key paired with the discovered private key."""
    priv_path, _ = discover_ssh_private_key()
    pub_path = priv_path.with_name(f"{priv_path.name}.pub")
    if not pub_path.is_file():
        raise RuntimeError(f"No SSH public key at {pub_path}")
    return pub_path, pub_path.read_text().strip()
