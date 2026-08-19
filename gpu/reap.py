"""Stateless-first TTL reap for workloads and volumes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from gpu.errors import DestroyError
from gpu.providers import configured_providers, get_provider
from gpu.registry import NAME_PREFIX, PodRegistry, is_expired
from gpu.types import Pod, SshTarget
from observability import events as obs

logger = logging.getLogger(__name__)


@dataclass
class ReapAction:
    kind: str  # workload | volume | registry_retry
    name: str
    id: str
    provider: str
    dry_run: bool
    destroyed: bool
    detail: str = ""


def _configured_cloud_providers(
    *, state_dir: Path | None, factory: Callable[..., Any] | None
) -> list[Any]:
    factory = factory or get_provider
    out = []
    names = [n for n in configured_providers() if n != "static_ssh"]
    # Always include known cloud providers so orphans from a removed list entry
    # are still visible when their API key is present.
    ordered = names + [
        n for n in ("lium", "shadeform", "runpod", "targon") if n not in names
    ]
    seen: set[str] = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        try:
            out.append(factory(name, state_dir=state_dir))
        except Exception as exc:  # noqa: BLE001 — missing key is fine
            logger.info("skip provider %s for reap: %s", name, exc)
    return out


def reap(
    *,
    dry_run: bool = False,
    state_dir: Path | None = None,
    registry: PodRegistry | None = None,
    provider_factory: Callable[..., Any] | None = None,
    clock=None,
) -> list[ReapAction]:
    """Destroy expired pt-* workloads/volumes; retry destroy_failed."""
    registry = registry or PodRegistry(state_dir)
    actions: list[ReapAction] = []
    providers = _configured_cloud_providers(
        state_dir=registry.state_dir, factory=provider_factory
    )

    for provider in providers:
        # Workloads
        try:
            pods = provider.list_pods()
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_pods failed for %s: %s", provider.name, exc)
            pods = []
        for pod in pods:
            if not pod.name.startswith(NAME_PREFIX):
                continue
            expired = is_expired(pod.name, clock=clock)
            if expired is not True:
                continue
            obs.pod_ttl_exceeded(pod=pod.name, provider=provider.name)
            actions.append(
                ReapAction(
                    kind="workload",
                    name=pod.name,
                    id=pod.pod_id,
                    provider=provider.name,
                    dry_run=dry_run,
                    destroyed=False,
                )
            )
            if dry_run:
                continue
            # list_pods often omits volume_uid; keep registry volume metadata for destroy.
            entry = registry.get(pod.name)
            if entry is not None and entry.volume_uid:
                raw = dict(pod.raw or {})
                if not raw.get("volume_uid"):
                    raw["volume_uid"] = entry.volume_uid
                    if entry.volume_name:
                        raw["volume_name"] = entry.volume_name
                    pod = replace(pod, raw=raw)
            try:
                provider.destroy(pod)
                actions[-1].destroyed = True
                registry.remove(pod.name)
            except DestroyError as exc:
                actions[-1].detail = str(exc)
                obs.destroy_failed(pod=pod.name, provider=provider.name, error=str(exc))
                entry = registry.get(pod.name)
                if entry is not None:
                    entry.state = "destroy_failed"
                    registry.update(entry)
                logger.error("reap destroy failed for %s: %s", pod.name, exc)

        # Volumes (may outlive workloads)
        try:
            volumes = provider.list_volumes()
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_volumes failed for %s: %s", provider.name, exc)
            volumes = []
        for vol in volumes:
            vname = str(vol.get("name", ""))
            vid = str(vol.get("id", ""))
            if not vname.startswith(NAME_PREFIX):
                continue
            expired = is_expired(vname, clock=clock)
            if expired is not True:
                continue
            actions.append(
                ReapAction(
                    kind="volume",
                    name=vname,
                    id=vid,
                    provider=provider.name,
                    dry_run=dry_run,
                    destroyed=False,
                )
            )
            if dry_run:
                continue
            # Reuse destroy with a synthetic pod carrying volume_uid.
            synthetic = Pod(
                provider=provider.name,
                pod_id=f"volume-only-{vid}",
                name=vname,
                ssh=SshTarget(host="", port=22, user=""),
                key_path=Path("/dev/null"),
                hourly_price_cents=0,
                created_utc=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ),
                ttl_hours=0.0,
                raw={"volume_uid": vid, "volume_name": vname},
            )
            try:
                # Prefer volume-only teardown if provider exposes it.
                if hasattr(provider, "_teardown_volume"):
                    provider._teardown_volume(vid, raise_on_fail=True)  # noqa: SLF001
                else:
                    provider.destroy(synthetic)
                actions[-1].destroyed = True
            except Exception as exc:  # noqa: BLE001
                actions[-1].detail = str(exc)
                logger.error("reap volume delete failed for %s: %s", vname, exc)

    # Registry destroy_failed retries
    for entry in registry.list():
        if entry.provider == "static_ssh":
            continue
        if entry.state != "destroy_failed":
            continue
        actions.append(
            ReapAction(
                kind="registry_retry",
                name=entry.name,
                id=entry.pod_id,
                provider=entry.provider,
                dry_run=dry_run,
                destroyed=False,
            )
        )
        if dry_run:
            continue
        try:
            provider = (provider_factory or get_provider)(
                entry.provider, state_dir=registry.state_dir
            )
            pod = Pod(
                provider=entry.provider,
                pod_id=entry.pod_id,
                name=entry.name,
                ssh=SshTarget(
                    host=entry.ssh_host or "ssh.deployments.targon.com",
                    port=entry.ssh_port or 22,
                    user=entry.ssh_user or entry.pod_id,
                ),
                key_path=Path(entry.key_path or "/dev/null"),
                hourly_price_cents=entry.hourly_price_cents,
                created_utc=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ),
                ttl_hours=entry.ttl_hours,
                raw={
                    "volume_uid": entry.volume_uid,
                    "volume_name": entry.volume_name,
                },
            )
            provider.destroy(pod)
            registry.remove(entry.name)
            actions[-1].destroyed = True
        except Exception as exc:  # noqa: BLE001
            actions[-1].detail = str(exc)
            logger.error("retry destroy_failed failed for %s: %s", entry.name, exc)

    return actions
