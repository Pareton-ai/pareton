"""Offline Shadeform provider tests with fake transport."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu.errors import ProvisionError
from gpu.keys import ensure_durable_keypair
from gpu.providers.shadeform import (
    ShadeformProvider,
    _format_instance_id,
    _normalize_gpu_type,
    _parse_instance_id,
    _pick_os,
)
from gpu.registry import encode_pod_name
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
        self.types: list[dict[str, Any]] = []
        self.info_statuses: list[str] = ["active"]
        self.info_ip = "1.2.3.4"
        self.volume_delete_codes: list[int] = [200]
        self.deleted: list[str] = []
        self.create_fail = False

    def __call__(
        self, method, url, *, headers=None, json=None, params=None, timeout=60
    ):
        method = method.upper()
        self.calls.append((method, url, {"json": json, "params": params}))
        path = url.split("shadeform.ai/v1", 1)[-1] if "shadeform.ai" in url else url

        if method == "GET" and path.startswith("/instances/types"):
            return FakeResp(200, _json={"instance_types": list(self.types)})

        if method == "GET" and path == "/sshkeys":
            return FakeResp(200, _json={"ssh_keys": []})

        if method == "POST" and path == "/sshkeys/add":
            return FakeResp(200, _json={"id": "ssh-1"})

        if method == "POST" and path == "/volumes/create":
            assert json is not None
            assert json["name"]
            assert json["size_in_gb"] == 250
            return FakeResp(200, _json={"id": "vol-1"})

        if method == "POST" and path == "/instances/create":
            if self.create_fail:
                return FakeResp(400, text="create failed")
            assert json is not None
            assert json["volume_ids"] == ["vol-1"]
            assert json["ssh_key_id"] == "ssh-1"
            assert json["name"]
            return FakeResp(200, _json={"id": "inst-1"})

        if (
            method == "GET"
            and path.startswith("/instances/")
            and path.endswith("/info")
        ):
            st = self.info_statuses.pop(0) if self.info_statuses else "active"
            return FakeResp(
                200,
                _json={
                    "status": st,
                    "ip": self.info_ip if st == "active" else "",
                    "ssh_port": 22,
                    "ssh_user": "shadeform",
                },
            )

        if method == "POST" and path.endswith("/delete"):
            if "/volumes/" in path:
                code = (
                    self.volume_delete_codes.pop(0) if self.volume_delete_codes else 200
                )
                if code == 200:
                    self.deleted.append(path)
                return FakeResp(code, text=f"HTTP {code}")
            self.deleted.append(path)
            return FakeResp(200, _json={})

        if method == "GET" and path == "/instances":
            return FakeResp(
                200,
                _json={
                    "instances": [
                        {
                            "id": "inst-1",
                            "name": encode_pod_name(ttl_hours=1.0),
                            "status": "active",
                            "ip": "1.2.3.4",
                        }
                    ]
                },
            )

        if method == "GET" and path == "/volumes":
            return FakeResp(
                200,
                _json={
                    "volumes": [{"id": "vol-1", "name": encode_pod_name(ttl_hours=1.0)}]
                },
            )

        return FakeResp(404, text="missing " + path)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    d = tmp_path / "gpu-state"
    ensure_durable_keypair(d)
    return d


def test_helpers():
    assert _normalize_gpu_type("NVIDIA-H100") == "H100"
    assert _parse_instance_id("aws:us-east-1:a100_1x") == (
        "aws",
        "us-east-1",
        "a100_1x",
    )
    assert _format_instance_id("aws", "us-east-1", "a100_1x") == "aws:us-east-1:a100_1x"
    assert _pick_os(["rocky8", "ubuntu22.04_cuda"]) == "ubuntu22.04_cuda"


def test_search_filters(state_dir: Path):
    tr = FakeTransport()
    tr.types = [
        {
            "cloud": "aws",
            "shade_instance_type": "h100_1x",
            "deployment_type": "vm",
            "hourly_price": 250,
            "configuration": {
                "gpu_type": "H100",
                "num_gpus": 1,
                "os_options": ["ubuntu22.04"],
            },
            "availability": [
                {"available": True, "region": "us-east-1", "display_name": "H100"}
            ],
        },
        {
            "cloud": "aws",
            "shade_instance_type": "h100_8x",
            "deployment_type": "vm",
            "hourly_price": 2000,
            "configuration": {"gpu_type": "H100", "num_gpus": 8, "os_options": []},
            "availability": [{"available": True, "region": "us-east-1"}],
        },
        {
            "cloud": "aws",
            "shade_instance_type": "a100_1x_pricey",
            "deployment_type": "vm",
            "hourly_price": 5000,
            "configuration": {"gpu_type": "A100", "num_gpus": 1, "os_options": []},
            "availability": [{"available": True, "region": "us-east-1"}],
        },
    ]
    p = ShadeformProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    offers = p.search(PodSpec(gpu_count=1, max_hourly_cents=1000))
    assert len(offers) == 1
    assert offers[0].instance_id == "aws:us-east-1:h100_1x"
    assert offers[0].hourly_price_cents == 250


def test_provision_wait_mount_and_abort(state_dir: Path, monkeypatch):
    tr = FakeTransport()
    tr.info_statuses = ["pending", "active"]
    p = ShadeformProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    mounts: list[str] = []

    def fake_mount(pod: Pod) -> bool:
        mounts.append(pod.pod_id)
        return True

    monkeypatch.setattr(p, "_mount_workspace", fake_mount)
    pub = (state_dir / "keys" / "pareton-gpu-ed25519.pub").read_text().strip()
    name = encode_pod_name(
        ttl_hours=1.0,
        created=datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc),
        uid8="abcd1234",
    )
    offer = Offer(
        provider="shadeform",
        instance_id="aws:us-east-1:h100_1x",
        description="H100",
        hourly_price_cents=250,
        gpu_count=1,
        gpu_type="H100",
        raw={
            "cloud": "aws",
            "region": "us-east-1",
            "shade_instance_type": "h100_1x",
            "os_options": ["ubuntu22.04"],
        },
    )
    pod = p.provision(offer, name=name, ssh_public_key=pub)
    assert pod.pod_id == "inst-1"
    assert pod.ssh.host == "1.2.3.4"
    assert pod.ssh.user == "shadeform"
    assert pod.raw.get("volume_uid") == "vol-1"
    assert mounts == ["inst-1"]

    # Create failure cleans volume.
    tr2 = FakeTransport()
    tr2.create_fail = True
    p2 = ShadeformProvider(
        "k", state_dir=state_dir, transport=tr2, sleep=lambda _s: None
    )
    with pytest.raises(ProvisionError, match="failed HTTP 400"):
        p2.provision(offer, name=name, ssh_public_key=pub)
    assert any("/volumes/vol-1/delete" in d for d in tr2.deleted)


def test_destroy_and_list(state_dir: Path):
    tr = FakeTransport()
    p = ShadeformProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    pod = Pod(
        provider="shadeform",
        pod_id="inst-1",
        name="n",
        ssh=SshTarget(host="h", port=22, user="shadeform"),
        key_path=state_dir / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
        raw={"volume_uid": "vol-1"},
    )
    p.destroy(pod)
    assert any("/instances/inst-1/delete" in d for d in tr.deleted)
    assert any("/volumes/vol-1/delete" in d for d in tr.deleted)

    pods = p.list_pods()
    assert pods and pods[0].pod_id == "inst-1"
    vols = p.list_volumes()
    assert vols and vols[0]["id"] == "vol-1"


def test_wait_ready_error_status(state_dir: Path, monkeypatch):
    tr = FakeTransport()
    tr.info_statuses = ["error"]
    p = ShadeformProvider("k", state_dir=state_dir, transport=tr, sleep=lambda _s: None)
    pod = Pod(
        provider="shadeform",
        pod_id="inst-1",
        name="n",
        ssh=SshTarget(host="", port=22, user="shadeform"),
        key_path=state_dir / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
        raw={"volume_uid": "vol-1"},
    )
    with pytest.raises(ProvisionError, match="entered error"):
        p._wait_ready(pod, timeout_s=30)


def test_get_provider_shadeform(monkeypatch, state_dir: Path):
    monkeypatch.setenv("PARETON_SHADEFORM_API_KEY", "sf-key")
    from gpu.providers import get_provider

    p = get_provider("shadeform", state_dir=state_dir)
    assert p.name == "shadeform"

    monkeypatch.delenv("PARETON_SHADEFORM_API_KEY", raising=False)
    import config as cfg

    monkeypatch.setattr(cfg, "SHADEFORM_API_KEY", "", raising=False)
    with pytest.raises(ProvisionError, match="PARETON_SHADEFORM_API_KEY"):
        get_provider("shadeform")
