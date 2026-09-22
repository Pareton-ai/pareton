"""Public patch upload URLs and reveal timing, with no network or database."""

import hashlib
from types import SimpleNamespace
from urllib.parse import urlparse
from uuid import UUID

import pytest

import config
from storage import s3


@pytest.fixture(autouse=True)
def storage_config(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "pareton-s3")
    monkeypatch.setattr(config, "S3_REGION", "us-east-2")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "")
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "")


def test_each_upload_uses_an_independent_uuid():
    keys = {s3.object_key_for("c1", "hk1") for _ in range(30)}
    assert len(keys) == 30
    for key in keys:
        assert UUID(key.rsplit("/", 1)[1].removesuffix(".diff")).version == 4


def test_presigning_binds_private_upload_to_checksum_and_prevents_overwrite(
    monkeypatch,
):
    from botocore.exceptions import ClientError

    calls = []

    def sign(method, **kwargs):
        calls.append((method, kwargs))
        return "https://upload.example/?signature=upload-only"

    def missing(**kwargs):
        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    monkeypatch.setattr(
        s3,
        "_client",
        lambda **_: SimpleNamespace(generate_presigned_url=sign, head_object=missing),
    )
    cid = "11111111-1111-4111-8111-111111111111"
    from uuid import uuid4

    first, second = [
        s3.create_presigned_patch_upload(
            campaign_id=cid,
            hotkey="hk1",
            patch_hash="sha256:" + "a" * 64,
            upload_id=str(uuid4()),
            expires_in=300,
        )
        for _ in range(2)
    ]
    assert [c[0] for c in calls] == ["put_object", "put_object"]
    assert first.retrieval_url != second.retrieval_url
    for result, (_, call) in zip((first, second), calls):
        assert s3.private_patch_key(result.retrieval_url) == result.object_key
        assert "/private/campaigns/" in result.retrieval_url
        assert (
            call["Params"]["ChecksumSHA256"]
            == result.required_headers["x-amz-checksum-sha256"]
        )
        assert (
            call["Params"]["IfNoneMatch"]
            == result.required_headers["If-None-Match"]
            == "*"
        )
        assert call["ExpiresIn"] == 300
        assert urlparse(result.retrieval_url).query == ""
        assert (
            UUID(result.object_key.rsplit("/", 1)[1].removesuffix(".diff")).version == 4
        )


PRIVATE_URL = (
    "https://pareton-s3.s3.us-east-2.amazonaws.com/stage0/private/campaigns/"
    "11111111-1111-4111-8111-111111111111/patches/hk1/"
    "22222222-2222-4222-8222-222222222222.diff"
)


@pytest.mark.parametrize(
    "url",
    [
        PRIVATE_URL.replace("pareton-s3.s3.us-east-2.amazonaws.com", "evil.example"),
        PRIVATE_URL.replace("amazonaws.com", "amazonaws.com.evil.example"),
        PRIVATE_URL + "?signature=fake",
        PRIVATE_URL + "#fragment",
        PRIVATE_URL.replace("hk1/", "hk1/../"),
        PRIVATE_URL.replace("hk1/", "hk1%2f/"),
        PRIVATE_URL.replace("/private/", "/private/../"),
    ],
)
def test_private_locators_reject_untrusted_host_and_noncanonical_paths(url):
    assert s3.private_patch_key(url) is None
    assert not s3.is_allowed_retrieval_url(url)


def test_private_fetch_uses_credentials_and_closes_bounded_body(monkeypatch):
    from io import BytesIO

    body = BytesIO(b"patch")
    calls = []

    def get(**kwargs):
        calls.append(kwargs)
        return {"Body": body}

    monkeypatch.setattr(s3, "_client", lambda **_: SimpleNamespace(get_object=get))
    monkeypatch.setattr(
        s3.urllib.request, "urlopen", lambda *a, **k: pytest.fail("anonymous GET")
    )
    assert s3.fetch_patch_bytes(PRIVATE_URL) == b"patch"
    assert calls == [
        {"Bucket": config.S3_BUCKET, "Key": s3.private_patch_key(PRIVATE_URL)}
    ]
    assert body.closed

    monkeypatch.setattr(config, "PATCH_MAX_BYTES", 2)
    monkeypatch.setattr(
        s3,
        "_client",
        lambda **_: SimpleNamespace(get_object=lambda **k: {"Body": BytesIO(b"patch")}),
    )
    with pytest.raises(RuntimeError, match="exceeds max size"):
        s3.fetch_patch_bytes(PRIVATE_URL, attempts=1)


@pytest.mark.parametrize("cdn", [False, True])
def test_legacy_fetch_still_works_after_public_access_is_revoked(monkeypatch, cdn):
    from io import BytesIO

    legacy = PRIVATE_URL.replace("/private/", "/")
    key = legacy.split("amazonaws.com/", 1)[1]
    if cdn:
        monkeypatch.setattr(
            config, "S3_PUBLIC_BASE_URL", "https://cdn.example.test/assets"
        )
        legacy = "https://cdn.example.test/assets/" + key
    body = BytesIO(b"old private patch")
    calls = []

    def get(**kwargs):
        calls.append(kwargs)
        return {"Body": body}

    monkeypatch.setattr(s3, "_client", lambda **_: SimpleNamespace(get_object=get))
    monkeypatch.setattr(
        s3.urllib.request, "urlopen", lambda *a, **k: pytest.fail("anonymous GET")
    )
    assert s3.fetch_patch_bytes(legacy) == b"old private patch"
    assert calls == [{"Bucket": config.S3_BUCKET, "Key": key}]
    assert body.closed


def test_real_s3_presigner_signs_checksum_and_conditional_write_headers(monkeypatch):
    from urllib.parse import parse_qs

    import boto3
    from botocore.config import Config
    from botocore.stub import Stubber

    client = boto3.client(
        "s3",
        region_name="us-east-2",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        config=Config(signature_version="s3v4"),
    )
    monkeypatch.setattr(s3, "_client", lambda **_: client)
    with Stubber(client) as stub:
        stub.add_client_error(
            "head_object", service_error_code="404", http_status_code=404
        )
        result = s3.create_presigned_patch_upload(
            campaign_id="11111111-1111-4111-8111-111111111111",
            hotkey="hk1",
            upload_id="22222222-2222-4222-8222-222222222222",
            patch_hash="sha256:" + "a" * 64,
        )
    headers = parse_qs(urlparse(result.upload_url).query)["X-Amz-SignedHeaders"][
        0
    ].split(";")
    assert {"content-type", "host", "if-none-match", "x-amz-checksum-sha256"} <= set(
        headers
    )
    assert urlparse(result.retrieval_url).query == ""


def test_presign_does_not_extend_authorization_during_slow_storage_check(monkeypatch):
    from botocore.exceptions import ClientError

    clock = [1000]
    expires = []
    monkeypatch.setattr(s3.time, "time", lambda: clock[0])

    def head(**kwargs):
        clock[0] += 10
        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def sign(*args, **kwargs):
        expires.append(kwargs["ExpiresIn"])
        return "https://s3.example/put"

    monkeypatch.setattr(
        s3,
        "_client",
        lambda **_: SimpleNamespace(head_object=head, generate_presigned_url=sign),
    )
    fields = dict(
        campaign_id="11111111-1111-4111-8111-111111111111",
        hotkey="hk1",
        upload_id="22222222-2222-4222-8222-222222222222",
        patch_hash="sha256:" + "a" * 64,
        expires_in=300,
        expires_at=1300,
    )
    assert s3.create_presigned_patch_upload(**fields).expires_in == 290
    clock[0] = 1295
    with pytest.raises(ValueError, match="expired"):
        s3.create_presigned_patch_upload(**fields)
    assert expires == [290]


def test_presign_retry_verifies_existing_object_without_granting_another_write(
    monkeypatch,
):
    monkeypatch.setattr(
        s3,
        "_client",
        lambda **_: SimpleNamespace(
            head_object=lambda **k: {
                "ChecksumSHA256": s3._checksum("sha256:" + "a" * 64)
            },
            generate_presigned_url=lambda *a, **k: pytest.fail("must not overwrite"),
        ),
    )
    fields = dict(
        campaign_id="11111111-1111-4111-8111-111111111111",
        hotkey="hk1",
        upload_id="22222222-2222-4222-8222-222222222222",
    )
    result = s3.create_presigned_patch_upload(**fields, patch_hash="sha256:" + "a" * 64)
    assert result.already_uploaded and result.upload_url == ""
    with pytest.raises(ValueError, match="different patch"):
        s3.create_presigned_patch_upload(**fields, patch_hash="sha256:" + "b" * 64)


def test_hash_utility_matches_exact_uploaded_bytes(tmp_path, capsys):
    from miner.hash_patch import main

    patch = tmp_path / "miner.diff"
    data = b"diff --git a/x b/x\r\n+value = 1\n"
    patch.write_bytes(data)
    assert main([str(patch)]) == 0
    assert capsys.readouterr().out == f"sha256:{hashlib.sha256(data).hexdigest()}\n"
