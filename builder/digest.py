"""Resolve registry RepoDigest for a pushed engine image tag."""

from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


def tag_ref_to_name(image_ref: str) -> str:
    """Strip mutable tag (or digest) to registry/name."""
    if "@" in image_ref:
        return image_ref.split("@", 1)[0]
    # tag form name:tag — keep ports like localhost:5000/foo:tag
    if image_ref.count(":") == 1:
        return image_ref.rsplit(":", 1)[0]
    # host:port/name:tag
    host, rest = image_ref.split("/", 1)
    if ":" in rest:
        return f"{host}/{rest.rsplit(':', 1)[0]}"
    return image_ref


def digest_pinned_ref(tag_or_name: str, digest: str) -> str:
    name = tag_ref_to_name(tag_or_name)
    d = digest.lower()
    if not d.startswith("sha256:"):
        d = f"sha256:{d}"
    return f"{name}@{d}"


def resolve_image_repo_digest(
    image_ref: str,
    *,
    timeout_s: float = 120.0,
) -> str | None:
    """Return sha256:<64 hex> from docker inspect RepoDigests, or None."""
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format", "{{json .RepoDigests}}", image_ref],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("docker inspect failed for %s: %s", image_ref, exc)
        return None
    if proc.returncode != 0:
        logger.warning(
            "docker inspect non-zero for %s: %s",
            image_ref,
            (proc.stderr or proc.stdout)[:500],
        )
        return None
    try:
        digests = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return None
    if not digests:
        return None
    ref = str(digests[0])
    if "@sha256:" in ref.lower():
        return "sha256:" + ref.rsplit("@sha256:", 1)[1].lower()
    if ref.lower().startswith("sha256:"):
        return ref.lower()
    return None


def image_ref_resolves(image_ref: str, *, timeout_s: float = 60.0) -> bool:
    """True when ``docker manifest inspect`` can see the image in the registry."""
    try:
        proc = subprocess.run(
            ["docker", "manifest", "inspect", image_ref],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("docker manifest inspect failed for %s: %s", image_ref, exc)
        return False
    if proc.returncode != 0:
        logger.warning(
            "docker manifest inspect non-zero for %s: %s",
            image_ref,
            (proc.stderr or proc.stdout)[:500],
        )
        return False
    return True


def mock_digest_from_patch_hash(patch_hash: str) -> str:
    """Synthetic sha256:<64 hex> derived from patch_hash for mock builds."""
    hex_part = patch_hash.lower().strip()
    if hex_part.startswith("sha256:"):
        hex_part = hex_part.split(":", 1)[1]
    hex_part = "".join(c for c in hex_part if c in "0123456789abcdef")
    if len(hex_part) < 64:
        hex_part = (hex_part + "0" * 64)[:64]
    else:
        hex_part = hex_part[:64]
    return f"sha256:{hex_part}"
