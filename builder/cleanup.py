"""Keep persistent Docker builder storage below its high watermark."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Iterable
from typing import Any

import config
from builder.lock import builder_storage_lock
from builder.registry import baseline_build_image_ref, baseline_engine_image_ref

logger = logging.getLogger(__name__)

_RETAIN_REPOSITORY = "pareton-retain"
_HEX_TAG = re.compile(r"^[0-9a-f]{64}$")


def _run(cmd: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def _image_id(ref: str) -> str | None:
    proc = _run(["docker", "image", "inspect", "--format", "{{.Id}}", ref])
    return proc.stdout.strip() if proc.returncode == 0 else None


def _list_refs(repository: str) -> set[str]:
    proc = _run(
        [
            "docker",
            "image",
            "ls",
            "--filter",
            f"reference={repository}:*",
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker image ls failed: {proc.stderr.strip()[:500]}")
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _retention_tags(campaigns: Iterable[Any]) -> dict[str, str]:
    """Map local retention tags to active campaign baseline digest refs."""
    tags: dict[str, str] = {}
    for campaign in campaigns:
        if str(getattr(campaign, "status", "")) not in {"draft", "open"}:
            continue
        campaign_id = str(getattr(campaign, "campaign_id", "")).lower()
        if not campaign_id:
            raise ValueError("persisted campaign is missing campaign_id")
        tags[f"{_RETAIN_REPOSITORY}:{campaign_id}-build"] = baseline_build_image_ref(
            campaign.base_image_digest
        )
        bench = getattr(campaign, "bench", None) or {}
        engine_digest = str(bench.get("baseline_engine_image_digest") or "").strip()
        if engine_digest:
            tags[f"{_RETAIN_REPOSITORY}:{campaign_id}-engine"] = (
                baseline_engine_image_ref(engine_digest)
            )
    return tags


def _reconcile_retention(campaigns: Iterable[Any], *, dry_run: bool) -> None:
    desired = _retention_tags(campaigns)
    existing = _list_refs(_RETAIN_REPOSITORY)
    for tag, source in desired.items():
        source_id = _image_id(source)
        if not source_id:
            logger.info("cleanup: protected image is not local: %s", source)
            continue
        if _image_id(tag) == source_id:
            continue
        logger.info("cleanup: retain %s as %s", source, tag)
        if not dry_run:
            proc = _run(["docker", "tag", source, tag])
            if proc.returncode != 0:
                raise RuntimeError(
                    f"cannot retain protected image {source}: "
                    f"{proc.stderr.strip()[:500]}"
                )
    for tag in sorted(existing - set(desired)):
        logger.info("cleanup: release %s", tag)
        if not dry_run:
            proc = _run(["docker", "image", "rm", tag])
            if proc.returncode != 0:
                raise RuntimeError(f"cannot release {tag}: {proc.stderr.strip()[:500]}")


def _candidate_refs() -> set[str]:
    repository = f"ghcr.io/{config.GHCR_OWNER}/{config.GHCR_IMAGE}"
    return {
        ref
        for ref in _list_refs(repository)
        if _HEX_TAG.fullmatch(ref.rpartition(":")[2])
    }


def _remove_candidates(*, dry_run: bool) -> int:
    refs = sorted(_candidate_refs())
    for ref in refs:
        logger.info("cleanup: remove local candidate %s", ref)
        if not dry_run:
            proc = _run(["docker", "image", "rm", ref])
            if proc.returncode != 0:
                logger.warning(
                    "cleanup: candidate removal failed for %s: %s",
                    ref,
                    proc.stderr.strip()[:500],
                )
    return len(refs)


def _used_percent(usage: Any) -> float:
    return 100.0 * usage.used / usage.total


def cleanup_once(
    campaigns: Iterable[Any], *, dry_run: bool = False, force: bool = False
) -> dict[str, Any]:
    """Reconcile protected tags and prune disposable Docker data under pressure."""
    high = config.BUILDER_CLEANUP_HIGH_WATER_PERCENT
    hard = config.BUILDER_CLEANUP_HARD_WATER_PERCENT
    if not 0 < high < hard < 100:
        raise ValueError("cleanup watermarks must satisfy 0 < high < hard < 100")

    before = shutil.disk_usage(config.BUILDER_DOCKER_ROOT)
    _reconcile_retention(campaigns, dry_run=dry_run)
    candidates_removed = _remove_candidates(dry_run=dry_run)
    pruned = force or _used_percent(before) >= high
    if pruned:
        min_free = max(1, int(before.total * (1 - high / 100)))
        commands = [
            ["docker", "image", "prune", "--force"],
            [
                "docker",
                "buildx",
                "prune",
                "--force",
                "--filter",
                "type!=exec.cachemount",
                "--min-free-space",
                f"{min_free}b",
            ],
        ]
        for cmd in commands:
            logger.info("cleanup: %s", " ".join(cmd))
            if dry_run:
                continue
            proc = _run(cmd, timeout=1800)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{' '.join(cmd[:3])} failed: {proc.stderr.strip()[:500]}"
                )

    after = shutil.disk_usage(config.BUILDER_DOCKER_ROOT)
    return {
        "dry_run": dry_run,
        "usage_before_percent": _used_percent(before),
        "usage_after_percent": _used_percent(after),
        "candidates_removed": candidates_removed,
        "pruned": pruned,
    }


def evict_candidate_image(image_tag: str) -> bool:
    """Remove one published candidate tag after its digest is stored."""
    repository = f"ghcr.io/{config.GHCR_OWNER}/{config.GHCR_IMAGE}"
    name, sep, tag = str(image_tag).rpartition(":")
    if not sep or name != repository or not _HEX_TAG.fullmatch(tag):
        raise ValueError(f"not a Pareton candidate image tag: {image_tag!r}")
    with builder_storage_lock(blocking=True):
        proc = _run(["docker", "image", "rm", image_tag])
    if proc.returncode == 0:
        return True
    message = proc.stderr or proc.stdout
    if "No such image" in message:
        return False
    raise RuntimeError(f"docker image rm failed for {image_tag}: {message[:500]}")


def _load_campaigns() -> list[Any]:
    from campaign.store import list_campaigns

    return list_campaigns()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        campaigns = _load_campaigns()
    except Exception as exc:  # noqa: BLE001 - no DB means no safe cleanup.
        logger.error("cleanup: campaign load failed; no Docker data changed: %s", exc)
        return 1
    try:
        with builder_storage_lock(blocking=False) as acquired:
            if not acquired:
                logger.info("cleanup: build storage lock is busy; skip this run")
                return 0
            result = cleanup_once(
                campaigns, dry_run=bool(args.dry_run), force=bool(args.force)
            )
    except Exception as exc:  # noqa: BLE001 - oneshot reports operational failure.
        logger.error("cleanup failed: %s", exc)
        return 1
    print(json.dumps(result, sort_keys=True))
    if (
        not args.dry_run
        and result["usage_after_percent"] >= config.BUILDER_CLEANUP_HARD_WATER_PERCENT
    ):
        logger.error("cleanup: Docker disk remains above hard watermark")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
