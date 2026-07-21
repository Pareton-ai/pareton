"""Offline Targon provider tests with fake transport."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu.errors import DestroyError, ProvisionError
from gpu.keys import ensure_durable_keypair
from gpu.providers.targon import TargonProvider, _normalize_gpu_type
from gpu.registry import encode_pod_name, parse_pod_name
from gpu.types import Offer, Pod, PodSpec, SshTarget


@dataclass
class FakeResp:
    status_code: int
    _json: Any = None
    text: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def content(self) -> bytes:
        if self._json is None and not self.text:
            return b""
        import json

        return json.dumps(self._json if self._json is not None else {}).encode()

    def json(self) -> Any:
        return self._json


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.routes: dict[tuple[str, str], Any] = {}
        self.deploy_codes: list[int] = [200]
        self.volume_states: list[str] = ["READY"]
        self.workload_states: list[str] = ["RUNNING"]
        self.volume_delete_codes: list[int] = [200]
        self.deleted: list[str] = []
        self.workload_items: list[dict[str, Any]] | None = None
        self.volume_items: list[dict[str, Any]] | None = None

    def route(self, method: str, path_suffix: str, handler) -> None:
        self.routes[(method.upper(), path_suffix)] = handler

    def __call__(
        self, method, url, *, headers=None, json=None, params=None, timeout=30
    ):
        method = method.upper()
        self.calls.append((method, url, {"json": json, "params": params}))
        path = url.split("targon.com/tha/v2", 1)[-1] if "targon.com" in url else url

        if method == "GET" and path.startswith("/inventory"):
            return FakeResp(200, _json=self.routes.get(("GET", "/inventory"), []))

        if method == "GET" and path == "/ssh-keys":
            return FakeResp(200, _json={"items": []})

        if method == "POST" and path == "/ssh-keys":
            return FakeResp(200, _json={"uid": "ssh-key-1"})

        if method == "POST" and path == "/volumes":
            assert json is not None
            assert parse_pod_name(json["name"]) is not None
            assert json["size_in_mb"] == 250 * 1024
            return FakeResp(200, _json={"uid": "vol-uid-1"})

        if method == "GET" and path.startswith("/volumes/") and path.endswith("/state"):
            st = self.volume_states.pop(0) if self.volume_states else "READY"
            return FakeResp(200, _json={"status": st})

        if method == "POST" and path == "/workloads":
            assert json is not None
            assert parse_pod_name(json["name"]) is not None
            assert json["volumes"][0]["uid"] == "vol-uid-1"
            assert "ssh_keys" in json
            return FakeResp(200, _json={"uid": "wl-uid-1"})

        if method == "POST" and path.endswith("/deploy"):
            code = self.deploy_codes.pop(0) if self.deploy_codes else 200
            return FakeResp(code, text="deploy" if code >= 400 else "")

        if (
            method == "GET"
            and path.startswith("/workloads/")
            and path.endswith("/state")
        ):
            st = self.workload_states.pop(0) if self.workload_states else "RUNNING"
            return FakeResp(200, _json={"status": st, "message": ""})

        if method == "DELETE" and path.startswith("/workloads/"):
            self.deleted.append(path)
            return FakeResp(200, _json={})

        if method == "POST" and "/volumes/" in path and path.endswith("/delete"):
            code = self.volume_delete_codes.pop(0) if self.volume_delete_codes else 200
            if code == 200:
                self.deleted.append(path)
            return FakeResp(code, text=f"HTTP {code}")

        if method == "GET" and path == "/workloads":
            if self.workload_items is not None:
                return FakeResp(200, _json={"items": list(self.workload_items)})
            return FakeResp(
                200,
                _json={
                    "items": [
                        {"uid": "wl-uid-1", "name": encode_pod_name(ttl_hours=1.0)}
                    ]
                },
            )

        if method == "GET" and path == "/volumes":
            if self.volume_items is not None:
                return FakeResp(200, _json={"items": list(self.volume_items)})
            return FakeResp(
                200,
                _json={
                    "items": [
                        {"uid": "vol-uid-1", "name": encode_pod_name(ttl_hours=1.0)}
                    ]
                },
            )

        return FakeResp(404, text="missing route " + path)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    d = tmp_path / "gpu-state"
    ensure_durable_keypair(d)
    return d


def test_normalize_gpu_type():
    assert _normalize_gpu_type("NVIDIA-H200") == "H200"
    assert _normalize_gpu_type("NVIDIA-GeForce-RTX-4090") == "RTX-4090"


def test_search_normalization(state_dir: Path):
    tr = FakeTransport()
    tr.routes[("GET", "/inventory")] = [
        {
            "name": "res-a",
            "display_name": "H200 x1",
            "available": 2,
            "cost_per_hour": 1.5,
            "spec": {"gpu_type": "NVIDIA-H200", "gpu_count": 1},
        },
        {
            "name": "res-b",
            "display_name": "gone",
            "available": 0,
            "cost_per_hour": 0.1,
            "spec": {"gpu_type": "NVIDIA-H100", "gpu_count": 1},
        },
    ]
    p = TargonProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    offers = p.search(PodSpec(gpu_count=1, max_hourly_cents=1000))
    assert len(offers) == 1
    assert offers[0].gpu_type == "H200"
    assert offers[0].hourly_price_cents == 150


def test_rent_payload_and_deploy_retry(state_dir: Path):
    tr = FakeTransport()
    tr.deploy_codes = [502, 200]
    p = TargonProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    pub = (state_dir / "keys" / "pareton-gpu-ed25519.pub").read_text()
    offer = Offer(
        provider="targon",
        instance_id="res-a",
        description="x",
        hourly_price_cents=100,
        gpu_count=1,
        gpu_type="H200",
    )
    name = encode_pod_name(ttl_hours=2.0)
    pod = p.provision(offer, name=name, ssh_public_key=pub)
    assert pod.pod_id == "wl-uid-1"
    assert pod.raw["volume_uid"] == "vol-uid-1"
    assert pod.ssh.user == "wl-uid-1"
    # volume + workload names in POSTs
    vol_post = next(c for c in tr.calls if c[0] == "POST" and c[1].endswith("/volumes"))
    assert vol_post[2]["json"]["name"] == name
    wl_post = next(
        c for c in tr.calls if c[0] == "POST" and c[1].endswith("/workloads")
    )
    assert wl_post[2]["json"]["name"] == name


def test_deploy_502_but_state_running(state_dir: Path):
    tr = FakeTransport()
    tr.deploy_codes = [502, 502, 502]
    tr.workload_states = ["RUNNING"]  # state-check fallback during deploy
    # provision also polls wait_ready — provide another RUNNING
    tr.workload_states.append("RUNNING")
    p = TargonProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    pub = (state_dir / "keys" / "pareton-gpu-ed25519.pub").read_text()
    offer = Offer(
        provider="targon",
        instance_id="res-a",
        description="x",
        hourly_price_cents=100,
        gpu_count=1,
        gpu_type="H200",
    )
    pod = p.provision(offer, name=encode_pod_name(ttl_hours=1), ssh_public_key=pub)
    assert pod.pod_id == "wl-uid-1"


def test_rent_failure_after_volume_cleans_up(state_dir: Path):
    tr = FakeTransport()
    tr.fail_workload_create = True  # type: ignore[attr-defined]

    def call(method, url, *, headers=None, json=None, params=None, timeout=30):
        if method.upper() == "POST" and str(url).rstrip("/").endswith("/workloads"):
            return FakeResp(500, text="workload create failed")
        return FakeTransport.__call__(
            tr, method, url, headers=headers, json=json, params=params, timeout=timeout
        )

    p = TargonProvider("k", state_dir=state_dir, transport=call, sleep=lambda _s: None)
    pub = (state_dir / "keys" / "pareton-gpu-ed25519.pub").read_text()
    offer = Offer(
        provider="targon",
        instance_id="res-a",
        description="x",
        hourly_price_cents=100,
        gpu_count=1,
        gpu_type="H200",
    )
    with pytest.raises(ProvisionError):
        p.provision(offer, name=encode_pod_name(ttl_hours=1), ssh_public_key=pub)
    assert any("/volumes/vol-uid-1/delete" in c[1] for c in tr.calls)


def test_wait_ready_failed_and_timeout(state_dir: Path):
    tr = FakeTransport()
    p = TargonProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    pod = Pod(
        provider="targon",
        pod_id="wl-uid-1",
        name="n",
        ssh=SshTarget(host="h", port=22, user="u"),
        key_path=state_dir / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )
    tr.workload_states = ["FAILED"]
    with pytest.raises(ProvisionError, match="FAILED"):
        p._wait_ready(pod, timeout_s=1)

    tr.workload_states = ["DEPLOYING", "DEPLOYING"]
    with pytest.raises(ProvisionError, match="not ready"):
        p._wait_ready(pod, timeout_s=0)


def test_destroy_volume_retry_then_error(state_dir: Path):
    tr = FakeTransport()
    tr.volume_delete_codes = [409, 200]
    p = TargonProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    pod = Pod(
        provider="targon",
        pod_id="wl-uid-1",
        name="n",
        ssh=SshTarget(host="h", port=22, user="u"),
        key_path=state_dir / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
        raw={"volume_uid": "vol-uid-1"},
    )
    p.destroy(pod)

    tr2 = FakeTransport()
    tr2.volume_delete_codes = [409, 409, 409, 409, 409, 409]
    p2 = TargonProvider("k", state_dir=state_dir, transport=tr2, sleep=lambda _s: None)
    # Force short timeout by patching monotonic? Instead call _teardown_volume with fail
    with pytest.raises(DestroyError):
        # Exhaust retries quickly: patch sleep and make deadline immediate via many 409s
        # and a tiny loop — use raise_on_fail path with immediate non-retryable error
        tr2.volume_delete_codes = [400]
        p2._teardown_volume("vol-uid-1", raise_on_fail=True)


def test_destroy_surfaces_both_workload_and_volume_errors(state_dir: Path):
    def flaky(method, url, *, headers=None, json=None, params=None, timeout=30):
        if method.upper() == "DELETE" and "/workloads/" in url:
            return FakeResp(500, text="workload stuck")
        if method.upper() == "POST" and url.endswith("/delete"):
            return FakeResp(400, text="volume stuck")
        return FakeResp(404, text="no")

    p = TargonProvider("k", state_dir=state_dir, transport=flaky, sleep=lambda _s: None)
    pod = Pod(
        provider="targon",
        pod_id="wl-uid-1",
        name="n",
        ssh=SshTarget(host="h", port=22, user="u"),
        key_path=state_dir / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
        raw={"volume_uid": "vol-uid-1"},
    )
    with pytest.raises(DestroyError, match="workload teardown failed.*volume teardown"):
        p.destroy(pod)


def test_wait_ready_null_state_is_provision_error(state_dir: Path):
    # Empty body → content b"" → _req returns None
    class EmptyResp(FakeResp):
        @property
        def content(self) -> bytes:
            return b""

    def empty_body(method, url, *, headers=None, json=None, params=None, timeout=30):
        if method.upper() == "GET" and url.endswith("/state"):
            return EmptyResp(200)
        return FakeResp(404, text="no")

    p = TargonProvider(
        "k", state_dir=state_dir, transport=empty_body, sleep=lambda _s: None
    )
    pod = Pod(
        provider="targon",
        pod_id="wl-uid-1",
        name="n",
        ssh=SshTarget(host="h", port=22, user="u"),
        key_path=state_dir / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )
    with pytest.raises(ProvisionError, match="expected JSON object"):
        p._wait_ready(pod, timeout_s=1)


def test_provision_manual_polls_until_running(state_dir: Path, capsys):
    tr = FakeTransport()
    tr.workload_items = []
    tr.volume_items = []
    name = encode_pod_name(
        ttl_hours=1.0,
        created=datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc),
        uid8="abcd1234",
    )
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        if len(sleeps) == 1:
            tr.workload_items = [
                {"uid": "wl-manual-1", "name": name, "status": "PROVISIONING"}
            ]
            tr.workload_states = ["PROVISIONING"]
        elif len(sleeps) == 2:
            tr.workload_items = [
                {"uid": "wl-manual-1", "name": name, "status": "RUNNING"}
            ]
            tr.volume_items = [{"uid": "vol-manual-1", "name": name}]
            tr.workload_states = ["RUNNING"]

    p = TargonProvider("k", state_dir=state_dir, transport=tr, sleep=sleep)
    pub = (state_dir / "keys" / "pareton-gpu-ed25519.pub").read_text().strip()
    offer = Offer(
        provider="targon",
        instance_id="h200-small",
        description="H200",
        hourly_price_cents=200,
        gpu_count=1,
        gpu_type="H200",
    )
    pod = p.provision_manual(
        offer, name=name, ssh_public_key=pub, timeout_s=120, poll_s=30
    )
    out = capsys.readouterr().out
    assert "MANUAL Targon spin-up" in out
    assert name in out
    assert "pareton-gpu" in out
    assert pod.pod_id == "wl-manual-1"
    assert pod.raw.get("volume_uid") == "vol-manual-1"
    assert pod.raw.get("manual") is True
    assert sleeps == [30, 30]


def test_provision_manual_timeout(state_dir: Path):
    tr = FakeTransport()
    tr.workload_items = []
    tr.volume_items = []
    p = TargonProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    pub = (state_dir / "keys" / "pareton-gpu-ed25519.pub").read_text().strip()
    offer = Offer(
        provider="targon",
        instance_id="h200-small",
        description="H200",
        hourly_price_cents=200,
        gpu_count=1,
        gpu_type="H200",
    )
    with pytest.raises(ProvisionError, match="not RUNNING"):
        p.provision_manual(
            offer,
            name="pt-20260720120000-1h-abcd1234",
            ssh_public_key=pub,
            timeout_s=0,
            poll_s=30,
        )
