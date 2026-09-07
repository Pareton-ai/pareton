"""Pareton-presigned S3 uploads and bounded patch fetches."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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
    required_headers: dict[str, str]
    already_uploaded: bool = False


def _client(*, bounded: bool = False):
    import boto3
    from botocore.config import Config

    if not config.S3_ACCESS_KEY or not config.S3_SECRET_KEY:
        raise RuntimeError(
            "PARETON_S3_ACCESS_KEY and PARETON_S3_SECRET_KEY must be set"
        )

    timeouts = (
        {
            "connect_timeout": config.PATCH_FETCH_TIMEOUT_S,
            "read_timeout": config.PATCH_FETCH_TIMEOUT_S,
            "retries": {"total_max_attempts": 1},
        }
        if bounded
        else {}
    )
    kwargs: dict = {
        "aws_access_key_id": config.S3_ACCESS_KEY,
        "aws_secret_access_key": config.S3_SECRET_KEY,
        "region_name": config.S3_REGION,
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            **timeouts,
        ),
    }
    if config.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = config.S3_ENDPOINT_URL
    return boto3.client("s3", **kwargs)


def object_key_for(campaign_id: str, hotkey: str, upload_id: str | None = None) -> str:
    prefix = config.S3_PREFIX.strip("/")
    filename = str(uuid.UUID(upload_id)) if upload_id is not None else str(uuid.uuid4())
    return f"{prefix}/private/campaigns/{campaign_id}/patches/{hotkey}/{filename}.diff"


def public_retrieval_url(object_key: str) -> str:
    if config.S3_PUBLIC_BASE_URL:
        base = config.S3_PUBLIC_BASE_URL.rstrip("/")
        return f"{base}/{object_key}"
    return _s3_retrieval_url(object_key)


def _s3_retrieval_url(object_key: str) -> str:
    """Private locators bypass any public CDN."""
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
    patch_hash: str,
    upload_id: str,
    expires_in: int | None = None,
    expires_at: int | None = None,
) -> PresignResult:
    expires = expires_in if expires_in is not None else config.PRESIGN_EXPIRES_S
    key = object_key_for(campaign_id, hotkey, upload_id)
    client = _client(bounded=True)
    checksum = _checksum(patch_hash)
    headers = {
        "Content-Type": "text/plain",
        "x-amz-checksum-sha256": checksum,
        "If-None-Match": "*",
    }
    # A retry can finish after a successful PUT whose response was lost.
    from botocore.exceptions import ClientError

    try:
        existing = client.head_object(
            Bucket=config.S3_BUCKET, Key=key, ChecksumMode="ENABLED"
        )
    except ClientError as exc:
        # S3 returns 403 for a missing key when the role cannot ListBucket.
        if exc.response["Error"]["Code"] not in (
            "403",
            "404",
            "NoSuchKey",
            "AccessDenied",
        ):
            raise
    else:
        if existing.get("ChecksumSHA256") != checksum:
            raise ValueError("upload UUID already belongs to a different patch")
        return PresignResult("", _s3_retrieval_url(key), key, expires, headers, True)
    if expires_at is not None:
        expires = min(expires, expires_at - int(time.time()))
    if expires <= 0:
        raise ValueError("upload authorization expired before signing PUT")
    upload_url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": config.S3_BUCKET,
            "Key": key,
            "ContentType": "text/plain",
            "ChecksumSHA256": checksum,
            "IfNoneMatch": "*",
        },
        ExpiresIn=expires,
    )
    retrieval = _s3_retrieval_url(key)
    return PresignResult(
        upload_url=upload_url,
        retrieval_url=retrieval,
        object_key=key,
        expires_in=expires,
        required_headers=headers,
    )


def _checksum(patch_hash: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", patch_hash):
        raise ValueError("invalid patch hash")
    return base64.b64encode(bytes.fromhex(patch_hash[7:])).decode("ascii")


def private_patch_key(url: str) -> str | None:
    """Accept only a canonical private patch locator in our configured bucket."""
    base = _s3_retrieval_url("")
    if not url.startswith(base):
        return None
    key = url[len(base) :]
    prefix = config.S3_PREFIX.strip("/") + "/private/campaigns/"
    if not key.startswith(prefix):
        return None
    parts = key[len(prefix) :].split("/")
    if len(parts) != 4 or parts[1] != "patches":
        return None
    campaign, _, hotkey, filename = parts
    if not re.fullmatch(r"[a-zA-Z0-9]+", hotkey) or not filename.endswith(".diff"):
        return None
    try:
        if str(uuid.UUID(campaign)) != campaign:
            return None
        if str(uuid.UUID(filename[:-5])) + ".diff" != filename:
            return None
    except ValueError:
        return None
    return key


def is_allowed_retrieval_url(url: str) -> bool:
    """Only accept URLs that point at our bucket/prefix (or configured public base)."""
    if private_patch_key(url) is not None:
        return True
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or parsed.query or parsed.fragment:
        return False
    path = parsed.path.lstrip("/")
    prefix = config.S3_PREFIX.strip("/")
    base = public_retrieval_url("")
    if not url.startswith(base):
        return False
    key = url[len(base) :]
    return (
        key.startswith(f"{prefix}/campaigns/")
        and all(part not in ("", ".", "..") for part in key.split("/"))
        and "%" not in path
    )


def patch_url_hotkey(url: str) -> str | None:
    """Hotkey path segment from `.../campaigns/<cid>/patches/<hotkey>/<file>`."""
    try:
        parts = urlparse(url).path.strip("/").split("/")
    except Exception:
        return None
    for i, part in enumerate(parts):
        if (
            part == "patches"
            and i >= 2
            and parts[i - 2] == "campaigns"
            and i + 1 < len(parts)
            and parts[i + 1]
        ):
            return parts[i + 1]
    return None


def fetch_patch_bytes(url: str, *, attempts: int | None = None) -> bytes:
    """Fetch patch bytes with size/timeout/retry bounds."""
    if not is_allowed_retrieval_url(url):
        raise ValueError(f"retrieval_url not allowlisted: {url}")

    attempt_limit = config.PATCH_FETCH_RETRIES if attempts is None else attempts
    if attempt_limit < 1:
        raise ValueError("patch fetch attempts must be at least 1")

    from botocore.exceptions import BotoCoreError, ClientError

    key = private_patch_key(url)
    last_err: Exception | None = None
    for attempt in range(1, attempt_limit + 1):
        try:
            if key is not None:
                body = _client(bounded=True).get_object(
                    Bucket=config.S3_BUCKET, Key=key
                )["Body"]
                try:
                    data = body.read(config.PATCH_MAX_BYTES + 1)
                finally:
                    body.close()
            else:
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
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            BotoCoreError,
            ClientError,
        ) as exc:
            last_err = exc
            logger.warning("patch fetch attempt %d failed: %s", attempt, exc)
    raise RuntimeError(
        f"patch fetch failed after {attempt_limit} attempt(s): {last_err}"
    )


@lru_cache(maxsize=1024)
def publish_patch(url: str, patch_hash: str) -> str:
    """Copy an immutable patch to public storage. Caller MUST check reveal time.

    Successful copies are cached per process. After restart, verify the public
    copy first so its availability does not depend on private-source retention.
    """
    key = private_patch_key(url)
    if key is None:
        return url  # Previously public submissions keep their original URLs.
    from botocore.exceptions import ClientError

    client = _client(bounded=True)
    prefix = config.S3_PREFIX.strip("/")
    public_key = key.replace(f"{prefix}/private/", f"{prefix}/", 1)
    public_url = public_retrieval_url(public_key)
    checksum = _checksum(patch_hash)
    try:
        published = client.head_object(
            Bucket=config.S3_BUCKET, Key=public_key, ChecksumMode="ENABLED"
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in (
            "403",
            "404",
            "NoSuchKey",
            "AccessDenied",
        ):
            raise
    else:
        # A public copy remains usable even if its private source is removed.
        if published.get("ChecksumSHA256") != checksum:
            raise ValueError("public patch checksum does not match commitment")
        return public_url
    source = client.head_object(
        Bucket=config.S3_BUCKET, Key=key, ChecksumMode="ENABLED"
    )
    if source.get("ChecksumSHA256") != checksum:
        raise ValueError("private patch checksum does not match commitment")
    if source["ContentLength"] > config.PATCH_MAX_BYTES:
        raise ValueError("private patch exceeds size limit")
    client.copy_object(
        Bucket=config.S3_BUCKET,
        Key=public_key,
        CopySource={"Bucket": config.S3_BUCKET, "Key": key},
        CopySourceIfMatch=source["ETag"],
        MetadataDirective="REPLACE",
        ChecksumAlgorithm="SHA256",
        ContentType="text/plain",
        ContentDisposition='attachment; filename="patch.diff"',
    )
    return public_url


def evidence_object_key(submission_id: str, task_id: str) -> str:
    prefix = config.S3_PREFIX.strip("/")
    return f"{prefix}/evidence/{submission_id}/{task_id}.tar.gz"


def round_evidence_object_key(round_id: str, task_id: str) -> str:
    prefix = config.S3_PREFIX.strip("/")
    return f"{prefix}/evidence/rounds/{round_id}/{task_id}.tar.gz"


def realized_trace_object_key(campaign_id: str, sha256: str) -> str:
    digest = str(sha256).lower()
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:") :]
    prefix = config.S3_PREFIX.strip("/")
    return f"{prefix}/campaigns/{campaign_id}/realized-traces/{digest}.json"


def upload_realized_trace(*, campaign_id: str, body: bytes, sha256: str) -> str:
    """Put a generated workload trace; return the public retrieval URL."""
    key = realized_trace_object_key(campaign_id, sha256)
    client = _client()
    client.put_object(
        Bucket=config.S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return public_retrieval_url(key)


def _upload_bundle(
    output_dir: Path, *, key: str, archive_name: str
) -> tuple[str, str, int]:
    """Tar.gz the bench output dir and put_object. Returns (s3_url, sha256, size)."""
    output_dir = Path(output_dir)
    required = [
        output_dir / "bench_report.json",
        output_dir / "bench_request.remote.json",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"evidence bundle missing required file: {path.name}")
    if not (output_dir / "evidence").is_dir():
        raise RuntimeError("evidence bundle missing evidence/ directory")

    with tempfile.TemporaryDirectory(prefix="pareton-evidence-") as tmp:
        archive = Path(tmp) / f"{archive_name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(output_dir, arcname=".")
        data = archive.read_bytes()
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        client = _client()
        client.put_object(
            Bucket=config.S3_BUCKET,
            Key=key,
            Body=data,
            ContentType="application/gzip",
        )
        url = f"s3://{config.S3_BUCKET}/{key}"
        return url, digest, len(data)


def upload_evidence_bundle(
    submission_id: str,
    task_id: str,
    output_dir: Path,
) -> tuple[str, str, int]:
    """Tar.gz the bench output dir and put_object (private, not public-read).

    Returns (s3_url, sha256, size_bytes).
    """
    return _upload_bundle(
        output_dir,
        key=evidence_object_key(submission_id, task_id),
        archive_name=task_id,
    )


def upload_round_evidence(
    round_id: str,
    task_id: str,
    output_dir: Path,
) -> tuple[str, str, int]:
    """One evidence bundle per round. Returns (s3_url, sha256, size_bytes)."""
    return _upload_bundle(
        output_dir,
        key=round_evidence_object_key(round_id, task_id),
        archive_name=task_id,
    )
