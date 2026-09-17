"""Validate that Docker cannot garbage-collect Pareton's ccache mounts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import config


def validate_daemon_gc_config(data: Any) -> None:
    """Require application-managed BuildKit GC on the dedicated builder host."""
    if not isinstance(data, dict):
        raise TypeError("Docker daemon config must be a JSON object")
    builder = data.get("builder")
    gc = builder.get("gc") if isinstance(builder, dict) else None
    if not isinstance(gc, dict) or gc.get("enabled") is not False:
        raise ValueError(
            "Docker builder GC must set builder.gc.enabled=false; "
            "Pareton cleanup is the sole BuildKit GC authority"
        )


def validate_daemon_gc_file(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Docker daemon config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Docker daemon config is invalid JSON: {path}: {exc}"
        ) from exc
    validate_daemon_gc_config(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=config.DOCKER_DAEMON_CONFIG_PATH,
        help="Docker daemon JSON to validate",
    )
    args = parser.parse_args(argv)
    try:
        validate_daemon_gc_file(args.config)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"unsafe Docker builder GC configuration: {exc}\n")
    print(f"Docker builder GC is application-managed: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
