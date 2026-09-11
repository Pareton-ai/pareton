"""Shared helpers for the standalone pareton-ops programs.

Installed to /usr/local/lib/pareton-ops/ops_common.py next to sync-config.py
and notify-deploy-failure.py. Standard library only: these programs must run
without the application venv, database, Vector, or Axiom (spec sections 5, 7).

Env-file parsing lives here so the deploy notifier and the config checker use
one implementation instead of two drifting copies.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    """UTC timestamp used across ops state files; overridable in tests."""
    override = os.environ.get("PARETON_TEST_NOW")
    if override:
        return override
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return None


def parse_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse a KEY=VALUE env file without executing a shell.

    Handles optional ``export`` prefixes and single/double quotes. Values may
    contain ``=``. Never raises for content problems; returns them instead so
    callers can report a category without echoing file contents.
    """
    values: dict[str, str] = {}
    problems: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except PermissionError:
        return {}, ["permission-denied"]
    except FileNotFoundError:
        return {}, ["missing"]
    except OSError as exc:
        return {}, [type(exc).__name__]
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            problems.append(f"line-{lineno}-no-assignment")
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            problems.append(f"line-{lineno}-empty-key")
            continue
        values[key] = value
    return values, problems


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    """Write via a same-directory temp file and rename (spec section 5.2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_json(path: Path) -> dict | None:
    try:
        with open(path) as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, data: dict) -> None:
    atomic_write(
        path, json.dumps(data, indent=2, sort_keys=True).encode() + b"\n", 0o600
    )


@contextmanager
def locked(lock_path: Path):
    """Advisory flock around a state file (spec section 7.3)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
