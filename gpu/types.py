"""Dataclasses for GPU pod orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PodSpec:
    gpu_count: int = 1
    gpu_type: str | None = None
    max_hourly_cents: int = 1000
    ttl_hours: float = 2.0
    provider: str = "auto"
    force: bool = False
    manual: bool = False


@dataclass
class Offer:
    provider: str
    instance_id: str
    description: str
    hourly_price_cents: int
    gpu_count: int
    gpu_type: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SshTarget:
    host: str
    port: int
    user: str


@dataclass
class Pod:
    provider: str
    pod_id: str
    name: str
    ssh: SshTarget
    key_path: Path
    hourly_price_cents: int
    created_utc: datetime
    ttl_hours: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
