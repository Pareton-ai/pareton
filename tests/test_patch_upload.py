"""Real hotkey signatures across the miner/API boundary, without chain or S3."""

import hashlib
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
from bittensor.sp_core import Keypair
from fastapi.testclient import TestClient

import config
from api import server
from miner import commit_patch
from storage.upload_auth import upload_message

CID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def upload_api(monkeypatch):
    key = Keypair.from_uri("//Alice")  # Public development key, never a real wallet.
    monkeypatch.setattr(config, "SUBTENSOR_NETWORK", "finney")
    monkeypatch.setattr(config, "NETUID", 10)
    monkeypatch.setattr(config, "UPLOAD_AUTH_TTL_S", 300)
    monkeypatch.setattr(config, "PRESIGN_EXPIRES_S", 3600)
    monkeypatch.setattr(time, "time", lambda: 1800000000)
    monkeypatch.setattr(
        server, "get_campaign", lambda cid: SimpleNamespace(status="open")
    )
    monkeypatch.setattr(server, "campaign_hotkey_is_disqualified", lambda *a: False)
    calls = []

    def presign(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            upload_url="https://s3.example/put?token=secret",
            retrieval_url="https://s3.example/private.diff",
            object_key="private.diff",
            expires_in=kwargs["expires_in"],
            required_headers={"If-None-Match": "*"},
            already_uploaded=False,
        )

    monkeypatch.setattr(server, "create_presigned_patch_upload", presign)
    return TestClient(server.app), key, calls


def signed_request(key):
    fields = dict(
        campaign_id=CID,
        hotkey=key.ss58_address,
        patch_hash="sha256:" + "a" * 64,
        upload_id=str(uuid4()),
        network="finney",
        netuid=10,
        expires_at=int(time.time()) + 300,
    )
    return {**fields, "signature": key.sign(upload_message(fields)).hex()}


@pytest.mark.parametrize(
    "field,value",
    [
        ("hotkey", Keypair.from_uri("//Bob").ss58_address),
        ("patch_hash", "sha256:" + "b" * 64),
        ("campaign_id", "33333333-3333-4333-8333-333333333333"),
        ("upload_id", "44444444-4444-4444-8444-444444444444"),
        ("network", "test"),
        ("netuid", 11),
        ("expires_at", 1),
    ],
)
def test_presign_rejects_tampered_authorization_before_storage(
    upload_api, field, value
):
    client, key, calls = upload_api
    request = signed_request(key)
    request[field] = value
    assert client.post("/v1/uploads/patch", json=request).status_code == 403
    assert calls == []


@pytest.mark.parametrize("expires_offset", [-1, 301])
def test_presign_rejects_expired_or_unbounded_valid_signature(
    upload_api, expires_offset
):
    client, key, calls = upload_api
    fields = signed_request(key)
    del fields["signature"]
    fields["expires_at"] = int(time.time()) + expires_offset
    fields["signature"] = key.sign(upload_message(fields)).hex()
    assert client.post("/v1/uploads/patch", json=fields).status_code == 403
    assert calls == []


def test_unsigned_upload_is_rejected(upload_api):
    client, key, calls = upload_api
    assert (
        client.post(
            "/v1/uploads/patch", json={"campaign_id": CID, "hotkey": key.ss58_address}
        ).status_code
        == 422
    )
    assert calls == []


def test_miner_signs_exact_request_and_sends_required_put_headers(
    upload_api, monkeypatch
):
    client, key, calls = upload_api
    requests, puts = [], []

    def post(method, url, body):
        requests.append(body)
        response = client.post("/v1/uploads/patch", json=body)
        assert response.status_code == 200, response.text
        assert response.headers["Cache-Control"] == "no-store"
        return response.json()

    monkeypatch.setattr(commit_patch, "_http_json", post)
    monkeypatch.setattr(commit_patch, "_put_bytes", lambda *args: puts.append(args))
    url = commit_patch._upload_patch(
        SimpleNamespace(hotkey=key),
        patch_bytes=b"patch",
        campaign_id=CID,
        network="finney",
        netuid=10,
        api_base="https://api.example",
    )
    assert url == "https://s3.example/private.diff"
    assert puts == [
        ("https://s3.example/put?token=secret", b"patch", {"If-None-Match": "*"})
    ]
    assert calls[0]["hotkey"] == key.ss58_address
    assert calls[0]["patch_hash"] == "sha256:" + hashlib.sha256(b"patch").hexdigest()
    assert 0 < calls[0]["expires_in"] <= 300
    assert "secret" not in str(requests)


def test_miner_retries_same_signed_upload_after_lost_put_response(
    upload_api, monkeypatch
):
    client, key, calls = upload_api
    bodies = []

    def post(method, url, body):
        bodies.append(body)
        response = client.post("/v1/uploads/patch", json=body).json()
        response["already_uploaded"] = len(bodies) > 1
        return response

    monkeypatch.setattr(commit_patch, "_http_json", post)

    def lost(*args):
        raise TimeoutError()

    monkeypatch.setattr(commit_patch, "_put_bytes", lost)
    assert (
        commit_patch._upload_patch(
            SimpleNamespace(hotkey=key),
            patch_bytes=b"patch",
            campaign_id=CID,
            network="finney",
            netuid=10,
            api_base="https://api.example",
        )
        == "https://s3.example/private.diff"
    )
    assert bodies[0] == bodies[1]
    assert calls[0]["upload_id"] == calls[1]["upload_id"]
