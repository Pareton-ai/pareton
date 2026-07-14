"""Pareton-presigned S3 uploads and bounded patch fetches."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

import urllib.error
import urllib.request

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresignResult:
    upload_url: str
    retrieval_url: str
    object_key: str
    expires_in: int


def _client():
    import boto3
    from botocore.config import Config

    if not config.S3_ACCESS_KEY or not config.S3_SECRET_KEY:
        raise RuntimeError(
            "PARETON_S3_ACCESS_KEY and PARETON_S3_SECRET_KEY must be set"
        )

    kwargs: dict = {
        "aws_access_key_id": config.S3_ACCESS_KEY,
        "aws_secret_access_key": config.S3_SECRET_KEY,
        "region_name": config.S3_REGION,
        "config": Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    }
    if config.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = config.S3_ENDPOINT_URL
    return boto3.client("s3", **kwargs)


def object_key_for(campaign_id: str, hotkey: str) -> str:
    prefix = config.S3_PREFIX.strip("/")
    return f"{prefix}/campaigns/{campaign_id}/patches/{hotkey}/{uuid.uuid4()}.diff"


def public_retrieval_url(object_key: str) -> str:
    if config.S3_PUBLIC_BASE_URL:
        base = config.S3_PUBLIC_BASE_URL.rstrip("/")
        return f"{base}/{object_key}"
    if config.S3_ENDPOINT_URL:
        endpoint = config.S3_ENDPOINT_URL.rstrip("/")
        return f"{endpoint}/{config.S3_BUCKET}/{object_key}"
    return (
        f"https://{config.S3_BUCKET}.s3.{config.S3_REGION}.amazonaws.com/{object_key}"
    )


def create_presigned_patch_upload(
    *,
    campaign_id: str,
    hotkey: str,
    expires_in: int | None = None,
) -> PresignResult:
    expires = expires_in if expires_in is not None else config.PRESIGN_EXPIRES_S
    key = object_key_for(campaign_id, hotkey)
    client = _client()
    upload_url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": config.S3_BUCKET,
            "Key": key,
            "ContentType": "text/plain",
        },
        ExpiresIn=expires,
    )
    retrieval = public_retrieval_url(key)
    return PresignResult(
        upload_url=upload_url,
        retrieval_url=retrieval,
        object_key=key,
        expires_in=expires,
    )


def is_allowed_retrieval_url(url: str) -> bool:
    """Only accept URLs that point at our bucket/prefix (or configured public base)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    path = parsed.path.lstrip("/")
    prefix = config.S3_PREFIX.strip("/")
    expected_suffix = f"{prefix}/campaigns/"

    if config.S3_PUBLIC_BASE_URL:
        base = urlparse(config.S3_PUBLIC_BASE_URL)
        if parsed.netloc != base.netloc:
            return False
        base_path = base.path.lstrip("/")
        full = path
        if base_path and full.startswith(base_path + "/"):
            full = full[len(base_path) + 1 :]
        return expected_suffix in full or full.startswith(expected_suffix)

    # path-style: /bucket/prefix/campaigns/...
    if path.startswith(f"{config.S3_BUCKET}/"):
        rest = path[len(config.S3_BUCKET) + 1 :]
        return rest.startswith(expected_suffix)

    # virtual-hosted: bucket.s3.../prefix/campaigns/...
    if parsed.netloc.startswith(f"{config.S3_BUCKET}."):
        return path.startswith(expected_suffix)

    return False


def fetch_patch_bytes(url: str) -> bytes:
    """Fetch patch bytes with size/timeout/retry bounds."""
    if not is_allowed_retrieval_url(url):
        raise ValueError(f"retrieval_url not allowlisted: {url}")

    last_err: Exception | None = None
    for attempt in range(1, config.PATCH_FETCH_RETRIES + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(
                req, timeout=config.PATCH_FETCH_TIMEOUT_S
            ) as resp:
                data = resp.read(config.PATCH_MAX_BYTES + 1)
            if len(data) > config.PATCH_MAX_BYTES:
                raise ValueError(
                    f"patch exceeds max size {config.PATCH_MAX_BYTES} bytes"
                )
            return data
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_err = exc
            logger.warning("patch fetch attempt %d failed: %s", attempt, exc)
    raise RuntimeError(f"patch fetch failed after retries: {last_err}")
