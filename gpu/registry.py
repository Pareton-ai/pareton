"""Local pod registry + name-carried TTL encode/parse."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from gpu.errors import GpuError

logger = logging.getLogger(__name__)

NAME_PREFIX = "pareton-gpu-"
_NAME_RE = re.compile(
    r"^pareton-gpu-"
    r"(?P<stamp>\d{8}-\d{6})-"
    r"(?P<ttl>[0-9]+(?:\.[0-9]+)?)h-"
    r"(?P<uid>[0-9a-f]{8})$"
)

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _state_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    try:
        import config as _cfg

        return (
            Path(
                getattr(
                    _cfg,
                    "GPU_STATE_DIR",
                    Path.home() / ".cache" / "pareton" / "gpu",
                )
            )
            .expanduser()
            .resolve()
        )
    except Exception:  # noqa: BLE001
        return (Path.home() / ".cache" / "pareton" / "gpu").resolve()


def encode_pod_name(
    *,
    ttl_hours: float,
    created: datetime | None = None,
    uid8: str | None = None,
) -> str:
    """Build pareton-gpu-<yyyymmdd-hhmmss>-<ttl_hours>h-<uuid8> (UTC)."""
    created = created or _utc_now()
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    else:
        created = created.astimezone(timezone.utc)
    stamp = created.strftime("%Y%m%d-%H%M%S")
    uid = (uid8 or uuid.uuid4().hex[:8]).lower()
    # Trim trailing .0 for integer hours in the name.
    ttl_s = f"{ttl_hours:g}"
    return f"{NAME_PREFIX}{stamp}-{ttl_s}h-{uid}"


def parse_pod_name(
    name: str, *, clock: Clock | None = None
) -> tuple[datetime, float, datetime] | None:
    """Return (created_utc, ttl_hours, deadline_utc) or None if not ours."""
    m = _NAME_RE.match(name.strip())
    if not m:
        return None
    created = datetime.strptime(m.group("stamp"), "%Y%m%d-%H%M%S").replace(
        tzinfo=timezone.utc
    )
    ttl_hours = float(m.group("ttl"))
    deadline = created + timedelta(hours=ttl_hours)
    return created, ttl_hours, deadline


def is_expired(name: str, *, clock: Clock | None = None) -> bool | None:
    """True if past deadline, False if unexpired, None if not a managed name."""
    parsed = parse_pod_name(name)
    if parsed is None:
        return None
    _created, _ttl, deadline = parsed
    now = (clock or _utc_now)()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now >= deadline


@dataclass
class RegistryEntry:
    provider: str
    pod_id: str
    name: str
    deadline: str  # ISO UTC
    hourly_price_cents: int
    volume_uid: str = ""
    volume_name: str = ""
    state: str = "active"  # active | destroy_failed
    key_path: str = ""
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    created_utc: str = ""
    ttl_hours: float = 2.0
    raw: dict[str, Any] | None = None


class PodRegistry:
    """Atomic JSON registry under state_dir/pods.json."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = _state_dir(state_dir)
        self.path = self.state_dir / "pods.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "keys").mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            bak = self.path.with_suffix(".json.corrupt")
            try:
                shutil.copy2(self.path, bak)
            except OSError:
                pass
            # Fail closed: empty registry would drop single-flight and allow a
            # second billable rent while the previous cloud workload may exist.
            raise GpuError(
                f"corrupt registry {self.path} (backed up to {bak.name}); "
                f"repair or remove it before provisioning: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise GpuError(
                f"corrupt registry {self.path}: top-level JSON must be a list"
            )
        return [x for x in data if isinstance(x, dict)]

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, self.path)

    def list(self) -> list[RegistryEntry]:
        out: list[RegistryEntry] = []
        for d in self._load():
            try:
                out.append(
                    RegistryEntry(
                        provider=str(d["provider"]),
                        pod_id=str(d["pod_id"]),
                        name=str(d["name"]),
                        deadline=str(d.get("deadline", "")),
                        hourly_price_cents=int(d.get("hourly_price_cents", 0)),
                        volume_uid=str(d.get("volume_uid", "")),
                        volume_name=str(d.get("volume_name", "")),
                        state=str(d.get("state", "active")),
                        key_path=str(d.get("key_path", "")),
                        ssh_host=str(d.get("ssh_host", "")),
                        ssh_port=int(d.get("ssh_port", 22)),
                        ssh_user=str(d.get("ssh_user", "")),
                        created_utc=str(d.get("created_utc", "")),
                        ttl_hours=float(d.get("ttl_hours", 2.0)),
                        raw=d.get("raw") if isinstance(d.get("raw"), dict) else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("skipping bad registry entry: %s", exc)
        return out

    def add(self, entry: RegistryEntry) -> None:
        entries = self._load()
        entries = [e for e in entries if e.get("name") != entry.name]
        entries.append(asdict(entry))
        self._save(entries)

    def update(self, entry: RegistryEntry) -> None:
        self.add(entry)

    def remove(self, name: str) -> None:
        entries = [e for e in self._load() if e.get("name") != name]
        self._save(entries)

    def get(self, name: str) -> RegistryEntry | None:
        for e in self.list():
            if e.name == name:
                return e
        return None

    def has_blocking_managed(self) -> RegistryEntry | None:
        """Return first active/destroy_failed non-static entry, else None."""
        for e in self.list():
            if e.provider == "static_ssh":
                continue
            if e.state in ("active", "destroy_failed"):
                return e
        return None

    @contextmanager
    def provision_lock(self) -> Iterator[None]:
        """Exclusive flock spanning single-flight check + rent + registry add."""
        import fcntl

        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_dir / "provision.lock"
        with lock_path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
