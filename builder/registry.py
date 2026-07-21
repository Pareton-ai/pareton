"""GHCR image naming helpers."""

from __future__ import annotations

import re

import config

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)


def engine_image_ref(patch_hash: str) -> str:
    """Return ghcr.io/<owner>/<image>:<tag> for a patch hash."""
    tag = patch_hash.lower()
    if tag.startswith("sha256:"):
        tag = tag.split(":", 1)[1]
    tag = tag[:64]
    return f"ghcr.io/{config.GHCR_OWNER}/{config.GHCR_IMAGE}:{tag}"


def short_tag(patch_hash: str) -> str:
    tag = patch_hash.lower()
    if tag.startswith("sha256:"):
        tag = tag.split(":", 1)[1]
    return tag[:12]


def normalize_digest(digest: str) -> str:
    """Return canonical lowercase sha256:<64 hex>."""
    s = digest.strip().lower()
    if "@sha256:" in s:
        s = "sha256:" + s.rsplit("@sha256:", 1)[1].split("#", 1)[0]
    elif not s.startswith("sha256:"):
        s = f"sha256:{s}"
    if not _SHA256_RE.match(s):
        raise ValueError(f"invalid image digest: {digest!r}")
    return s


def pullable_digest_ref(digest_or_ref: str, *, image: str) -> str:
    """Expand a bare digest (or keep a name@digest) into a pullable GHCR ref.

    Accepts ``sha256:<hex>`` or ``registry/name@sha256:<hex>``.
    Bare digests become ``ghcr.io/<owner>/<image>@sha256:<hex>``.
    """
    s = digest_or_ref.strip()
    lower = s.lower()
    if "@sha256:" in lower and not lower.startswith("sha256:"):
        # Already a registry ref pinned by digest.
        digest = normalize_digest(s)
        name = s.split("@", 1)[0]
        return f"{name}@{digest}"
    digest = normalize_digest(s)
    return f"ghcr.io/{config.GHCR_OWNER}/{image}@{digest}"


def baseline_build_image_ref(digest: str) -> str:
    """Campaign build-base pin → pullable pareton-baseline ref."""
    return pullable_digest_ref(digest, image=config.GHCR_BASELINE_IMAGE)


def baseline_engine_image_ref(digest: str) -> str:
    """Campaign baseline engine pin → pullable pareton-engine ref."""
    return pullable_digest_ref(digest, image=config.GHCR_IMAGE)
