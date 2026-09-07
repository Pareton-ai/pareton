"""Public patch upload URLs and reveal timing, with no network or database."""

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import urlparse
from uuid import UUID

import pytest

import config
from storage import s3
from storage.visibility import patch_is_revealed, patch_reveal_at


@pytest.fixture(autouse=True)
def storage_config(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "pareton-s3")
    monkeypatch.setattr(config, "S3_REGION", "us-east-2")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "")
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "")
    monkeypatch.setattr(config, "PATCH_REVEAL_DELAY_S", 21600)


def test_release_boundary_missing_evaluation_and_retrospective_delay(monkeypatch):
    evaluated = datetime(2026, 9, 7, 10, tzinfo=timezone.utc)
    release = evaluated + timedelta(hours=6)
    assert patch_reveal_at(evaluated) == release
    assert not patch_is_revealed(evaluated, now=release - timedelta(microseconds=1))
    assert patch_is_revealed(evaluated, now=release)
    assert patch_is_revealed(evaluated, now=release + timedelta(days=20))
    assert not patch_is_revealed(None, now=release)
    monkeypatch.setattr(config, "PATCH_REVEAL_DELAY_S", 43200)
    assert not patch_is_revealed(evaluated, now=release)
    monkeypatch.setattr(config, "PATCH_REVEAL_DELAY_S", 0)
    assert patch_is_revealed(evaluated, now=evaluated)
    assert not patch_is_revealed(None, now=release)
    with pytest.raises(ValueError, match="timezone"):
        patch_reveal_at(datetime(2026, 9, 7))


def test_each_upload_uses_an_independent_uuid():
    keys = {s3.object_key_for("c1", "hk1") for _ in range(30)}
    assert len(keys) == 30
    for key in keys:
        assert UUID(key.rsplit("/", 1)[1].removesuffix(".diff")).version == 4


def test_presigning_only_expires_the_upload_not_the_public_download(monkeypatch):
    calls = []

    def sign(method, **kwargs):
        calls.append(method)
        return "https://upload.example/?signature=upload-only"

    monkeypatch.setattr(
        s3, "_client", lambda: SimpleNamespace(generate_presigned_url=sign)
    )
    first = s3.create_presigned_patch_upload(campaign_id="c1", hotkey="hk1")
    second = s3.create_presigned_patch_upload(campaign_id="c1", hotkey="hk1")
    assert calls == ["put_object", "put_object"]
    assert first.retrieval_url != second.retrieval_url
    for result in (first, second):
        assert result.retrieval_url == s3.public_retrieval_url(result.object_key)
        assert urlparse(result.retrieval_url).query == ""
        assert (
            UUID(result.object_key.rsplit("/", 1)[1].removesuffix(".diff")).version == 4
        )


def test_hash_utility_matches_exact_uploaded_bytes(tmp_path, capsys):
    from miner.hash_patch import main

    patch = tmp_path / "miner.diff"
    data = b"diff --git a/x b/x\r\n+value = 1\n"
    patch.write_bytes(data)
    assert main([str(patch)]) == 0
    assert capsys.readouterr().out == f"sha256:{hashlib.sha256(data).hexdigest()}\n"
