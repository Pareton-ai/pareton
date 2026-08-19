"""Offline tests for the Runpod REST v2 provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu.errors import ProvisionError
from gpu.keys import ensure_durable_keypair
from gpu.providers.runpod import (
    RunpodProvider,
    _as_item_list,
    _gpu_type_matches,
    _normalize_gpu_type,
    _parse_direct_ssh,
)
from gpu.types import ExecResult, Offer, PodSpec, SshTarget


class FakeResp:
    def __init__(self, payload: Any, *, status_code: int = 200, ok: bool | None = None):
        self._payload = payload
        self.status_code = status_code
        self.ok = (status_code < 400) if ok is None else ok
        self.text = "" if payload is None else str(payload)
        self.content = b"x" if payload is not None else b""

    def json(self):
        return self._payload


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.ssh_keys = {"keys": ["ssh-ed25519 AAAAother other@host"]}
        self.pod: dict[str, Any] = {}
        self.polls = 0

    def __call__(
        self,
        method: str,
        url: str,
        *,
        headers=None,
        json=None,
        params=None,
        timeout=None,
    ):
        path = url.replace("https://api.runpod.io", "")
        self.calls.append((method, path, json))
        if method == "GET" and path.startswith("/v2/catalog/gpus"):
            return FakeResp(
                {
                    "gpus": [
                        {
                            "id": "NVIDIA H100 PCIe",
                            "name": "H100 PCIe",
                            "pool": None,
                            "manufacturer": "NVIDIA",
                            "memory": 80,
                            "secure": True,
                            "community": True,
                            "price": {"secure": 2.5, "community": 1.8},
                            "maxCount": {"secure": 8, "community": 2},
                            "availability": "HIGH",
                            "dataCenters": [
                                {"id": "US-TX-3", "name": "TX", "availability": "HIGH"}
                            ],
                        },
                        {
                            "id": "NVIDIA GeForce RTX 4090",
                            "name": "RTX 4090",
                            "pool": "ADA_24",
                            "manufacturer": "NVIDIA",
                            "memory": 24,
                            "secure": True,
                            "community": True,
                            "price": {"secure": 0.44, "community": 0.31},
                            "maxCount": {"secure": 8, "community": 4},
                            "availability": "NONE",
                            "dataCenters": [],
                        },
                    ]
                }
            )
        if method == "GET" and path == "/v2/account/ssh-keys":
            return FakeResp(self.ssh_keys)
        if method == "PUT" and path == "/v2/account/ssh-keys":
            self.ssh_keys = dict(json or {})
            return FakeResp(self.ssh_keys)
        if method == "POST" and path == "/v2/pods":
            self.pod = {
                "id": "pod_abc",
                "name": (json or {}).get("name"),
                "status": "PROVISIONING",
                "ssh": {"proxy": None, "direct": None},
            }
            return FakeResp(self.pod, status_code=201)
        if method == "GET" and path == "/v2/pods/pod_abc":
            self.polls += 1
            if self.polls >= 2:
                self.pod = {
                    "id": "pod_abc",
                    "name": self.pod.get("name"),
                    "status": "RUNNING",
                    "dataCenterId": "US-TX-3",
                    "ssh": {
                        "proxy": {
                            "host": "ssh.runpod.io",
                            "port": 22,
                            "username": "token",
                            "command": "ssh token@ssh.runpod.io",
                        },
                        "direct": {
                            "host": "1.2.3.4",
                            "port": 22100,
                            "username": "root",
                            "command": "ssh root@1.2.3.4 -p 22100",
                        },
                    },
                }
            return FakeResp(self.pod)
        if method == "GET" and path == "/v2/pods":
            return FakeResp({"pods": [self.pod] if self.pod else []})
        if method == "DELETE" and path == "/v2/pods/pod_abc":
            self.pod = {}
            return FakeResp(None, status_code=204)
        if method == "GET" and path == "/v2/network-volumes":
            return FakeResp({"networkVolumes": []})
        return FakeResp(
            {"error": f"unhandled {method} {path}"}, status_code=500, ok=False
        )


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    ensure_durable_keypair(tmp_path)
    return tmp_path


@pytest.fixture
def provider(state_dir: Path) -> RunpodProvider:
    return RunpodProvider(
        "test-key",
        state_dir=state_dir,
        transport=FakeTransport(),
        sleep=lambda _s: None,
    )


def test_normalize_and_match():
    assert _normalize_gpu_type("NVIDIA H100 PCIe") == "H100 PCIe"
    assert _gpu_type_matches("H100 PCIe", "H100")
    assert not _gpu_type_matches("H1000", "H100")


def test_parse_direct_ssh_ignores_proxy_only():
    assert (
        _parse_direct_ssh(
            {
                "ssh": {
                    "proxy": {
                        "host": "ssh.runpod.io",
                        "port": 22,
                        "username": "x",
                        "command": "ssh x@ssh.runpod.io",
                    },
                    "direct": None,
                }
            }
        )
        is None
    )
    ssh = _parse_direct_ssh(
        {
            "ssh": {
                "direct": {
                    "host": "1.2.3.4",
                    "port": 22,
                    "username": "root",
                    "command": "ssh root@1.2.3.4",
                }
            }
        }
    )
    assert ssh == SshTarget(host="1.2.3.4", port=22, user="root")


def test_search_filters_availability_and_price(provider: RunpodProvider, monkeypatch):
    monkeypatch.setenv("PARETON_RUNPOD_CLOUD", "ANY")
    offers = provider.search(
        PodSpec(gpu_type="H100", gpu_count=1, max_hourly_cents=200)
    )
    assert len(offers) == 1
    assert offers[0].gpu_type == "H100 PCIe"
    assert offers[0].raw["cloud"] == "COMMUNITY"
    assert offers[0].hourly_price_cents == 180


def test_ensure_ssh_key_merges(provider: RunpodProvider):
    pub = "ssh-ed25519 AAAApareton pareton@host"
    provider._ensure_ssh_key(pub)
    transport: FakeTransport = provider._transport  # type: ignore[assignment]
    keys = transport.ssh_keys["keys"]
    assert any("AAAAother" in k for k in keys)
    assert any("AAAApareton" in k for k in keys)
    # Second call is idempotent (no duplicate).
    provider._ensure_ssh_key(pub)
    assert len(transport.ssh_keys["keys"]) == 2


def test_as_item_list_accepts_top_level_array():
    assert _as_item_list([{"id": "1"}], "pods") == [{"id": "1"}]
    assert _as_item_list({"pods": [{"id": "2"}]}, "pods") == [{"id": "2"}]
    assert _as_item_list(None, "pods") == []


def test_provision_waits_for_direct_ssh(provider: RunpodProvider, monkeypatch):
    monkeypatch.setattr(RunpodProvider, "_require_docker_host", lambda self, pod: None)
    offer = Offer(
        provider="runpod",
        instance_id="COMMUNITY:NVIDIA H100 PCIe",
        description="H100",
        hourly_price_cents=180,
        gpu_count=1,
        gpu_type="H100 PCIe",
        raw={
            "gpu_id": "NVIDIA H100 PCIe",
            "cloud": "COMMUNITY",
            "data_center_ids": ["US-TX-3"],
        },
    )
    pub = "ssh-ed25519 AAAApareton pareton@host"
    pod = provider.provision(offer, name="pt-test-1h-abcd1234", ssh_public_key=pub)
    assert pod.pod_id == "pod_abc"
    assert pod.ssh.host == "1.2.3.4"
    assert pod.ssh.port == 22100
    assert pod.ssh.user == "root"
    transport: FakeTransport = provider._transport  # type: ignore[assignment]
    create = next(c for c in transport.calls if c[0] == "POST" and c[1] == "/v2/pods")
    body = create[2] or {}
    assert body["startSsh"] is True
    assert "22/tcp" in body["ports"]
    assert body["disk"] >= 200
    assert body["mounts"]["persistent"]["path"] == "/workspace"


def test_provision_destroys_when_docker_missing(provider: RunpodProvider, monkeypatch):
    def no_docker(self, pod):
        raise ProvisionError(f"Runpod pod {pod.pod_id} cannot run Docker")

    monkeypatch.setattr(RunpodProvider, "_require_docker_host", no_docker)
    offer = Offer(
        provider="runpod",
        instance_id="COMMUNITY:NVIDIA H100 PCIe",
        description="H100",
        hourly_price_cents=180,
        gpu_count=1,
        gpu_type="H100 PCIe",
        raw={"gpu_id": "NVIDIA H100 PCIe", "cloud": "COMMUNITY"},
    )
    with pytest.raises(ProvisionError, match="cannot run Docker"):
        provider.provision(
            offer,
            name="pt-test-1h-abcd1234",
            ssh_public_key="ssh-ed25519 AAAApareton x",
        )
    transport: FakeTransport = provider._transport  # type: ignore[assignment]
    assert any(c[0] == "DELETE" and c[1] == "/v2/pods/pod_abc" for c in transport.calls)


def test_destroy_and_list(provider: RunpodProvider):
    transport: FakeTransport = provider._transport  # type: ignore[assignment]
    transport.pod = {
        "id": "pod_abc",
        "name": "pt-test",
        "status": "RUNNING",
        "ssh": {
            "direct": {
                "host": "1.2.3.4",
                "port": 22,
                "username": "root",
                "command": "ssh root@1.2.3.4",
            }
        },
    }
    pods = provider.list_pods()
    assert len(pods) == 1
    provider.destroy(pods[0])
    assert transport.pod == {}


def test_list_pods_accepts_top_level_array(provider: RunpodProvider):
    transport: FakeTransport = provider._transport  # type: ignore[assignment]

    def list_as_array(method, url, **kwargs):
        path = url.replace("https://api.runpod.io", "")
        if method == "GET" and path == "/v2/pods":
            return FakeResp(
                [
                    {
                        "id": "pod_arr",
                        "name": "from-array",
                        "status": "RUNNING",
                        "ssh": {
                            "direct": {
                                "host": "9.9.9.9",
                                "port": 22,
                                "username": "root",
                                "command": "ssh root@9.9.9.9",
                            }
                        },
                    }
                ]
            )
        return transport(method, url, **kwargs)

    provider._transport = list_as_array
    pods = provider.list_pods()
    assert len(pods) == 1
    assert pods[0].pod_id == "pod_arr"


def test_require_docker_host_ok(provider: RunpodProvider, monkeypatch):
    from datetime import datetime, timezone

    from gpu.types import Pod

    monkeypatch.setattr(
        "gpu.providers.runpod.ssh_exec",
        lambda *a, **k: ExecResult(exit_code=0, stdout="", stderr=""),
    )
    ready = Pod(
        provider="runpod",
        pod_id="pod_abc",
        name="n",
        ssh=SshTarget(host="1.2.3.4", port=22, user="root"),
        key_path=provider._key_path,
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )
    provider._require_docker_host(ready)


def test_get_provider_runpod(monkeypatch, state_dir: Path):
    monkeypatch.setenv("PARETON_RUNPOD_API_KEY", "secret")
    from gpu.providers import get_provider

    p = get_provider("runpod", state_dir=state_dir, transport=FakeTransport())
    assert p.name == "runpod"


def test_get_provider_runpod_requires_key(monkeypatch):
    monkeypatch.delenv("PARETON_RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr("gpu.providers._env_or_config", lambda env, attr: "")
    from gpu.providers import get_provider

    with pytest.raises(ProvisionError, match="PARETON_RUNPOD_API_KEY"):
        get_provider("runpod")
