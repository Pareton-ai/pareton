"""GHCR image naming helpers."""

from __future__ import annotations

import config


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
