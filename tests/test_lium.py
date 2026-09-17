"""Offline Lium provider tests with fake SDK client."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from gpu.errors import DestroyError, ProvisionError
from gpu.providers.lium import (
    LiumProvider,
    _gpu_type_matches,
    _normalize_gpu_type,
    _parse_ssh_target,
)
from gpu.registry import encode_pod_name
from gpu.types import Offer, Pod, PodSpec, SshTarget


@dataclass
class FakeExecutor:
    id: str
    machine_name: str
    gpu_type: str
    gpu_count: int
    price_per_hour: float
    docker_in_docker: bool = True
    effective_download_speed_mbps: float | None = None
    specs: dict[str, Any] | None = None


@dataclass
class FakeVolume:
    id: str
    name: str


@dataclass
class FakeKey:
    id: str
    name: str
    public_key: str


@dataclass
class FakePodInfo:
    id: str
    name: str
    status: str
    ssh_cmd: str | None


class FakeClient:
    def __init__(self) -> None:
        self.executors: list[FakeExecutor] = []
        self.keys: list[FakeKey] = []
        self.vols: list[FakeVolume] = []
        self.pods: list[FakePodInfo] = []
        self.up_calls: list[dict[str, Any]] = []
        self.down_ids: list[str] = []
        self.deleted_volumes: list[str] = []
        self.registered: list[tuple[str, str]] = []
        self.up_fail = False
        self.wait_none = False
        self.down_fail = False
        self.volume_delete_fail = False

    def ls(self, *, gpu_type=None, gpu_count=None):
        out = list(self.executors)
        if gpu_count is not None:
            out = [e for e in out if e.gpu_count >= int(gpu_count)]
        if gpu_type is not None:
            out = [e for e in out if gpu_type.upper() in e.gpu_type.upper()]
        return out

    def list_ssh_keys(self):
        return list(self.keys)

    def register_ssh_key(self, *, name: str, public_key: str):
        self.registered.append((name, public_key))
        key = FakeKey(id="k1", name=name, public_key=public_key)
        self.keys.append(key)
        return key

    def volume_create(self, name: str, *, description: str = ""):
        vol = FakeVolume(id=f"vol-{len(self.vols) + 1}", name=name)
        self.vols.append(vol)
        return vol

    def volume_delete(self, volume_id: str):
        if self.volume_delete_fail:
            raise RuntimeError("volume delete failed")
        self.deleted_volumes.append(volume_id)
        self.vols = [v for v in self.vols if v.id != volume_id]
        return {}

    def up(self, **kwargs):
        self.up_calls.append(dict(kwargs))
        if self.up_fail:
            raise RuntimeError("up failed")
        return {"id": "pod-1", "name": kwargs.get("name")}

    def wait_ready(self, pod, *, timeout=300, poll_interval=10):
        if self.wait_none:
            return None
        name = self.up_calls[-1].get("name", "pod") if self.up_calls else "pod"
        info = FakePodInfo(
            id=str(pod) if not hasattr(pod, "id") else str(pod),
            name=str(name),
            status="RUNNING",
            ssh_cmd="ssh root@10.0.0.5 -p 2222",
        )
        # wait_ready is called with pod_id string
        if isinstance(pod, str):
            info.id = pod
        self.pods.append(info)
        return info

    def down(self, pod):
        pid = getattr(pod, "id", str(pod))
        if self.down_fail:
            raise RuntimeError("down failed")
        self.down_ids.append(pid)
        self.pods = [p for p in self.pods if p.id != pid]
        return {}

    def ps(self):
        return list(self.pods)

    def volumes(self):
        return list(self.vols)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "gpu-state"


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def provider(state_dir: Path, client: FakeClient) -> LiumProvider:
    return LiumProvider("test-key", state_dir=state_dir, client=client)


def test_normalize_gpu_type():
    assert _normalize_gpu_type("NVIDIA-H200") == "H200"
    assert _normalize_gpu_type("H100") == "H100"


def test_gpu_type_matches_token_bound():
    assert _gpu_type_matches("H200", "H200")
    assert _gpu_type_matches("H200 NVL", "H200")
    assert _gpu_type_matches("H200-SXM", "H200")
    assert not _gpu_type_matches("H1000", "H100")
    assert not _gpu_type_matches("A100", "H100")


def test_search_normalizes_vendor_prefix_on_want(
    provider: LiumProvider, client: FakeClient
):
    client.executors = [
        FakeExecutor("e1", "node", "NVIDIA-H100", 1, 2.0, True),
        FakeExecutor("e2", "other", "H1000", 1, 1.0, True),
    ]
    offers = provider.search(PodSpec(gpu_type="NVIDIA-H100", max_hourly_cents=1000))
    assert [o.instance_id for o in offers] == ["e1"]
    assert offers[0].gpu_type == "H100"


def test_search_rejects_substring_gpu_false_positive(
    provider: LiumProvider, client: FakeClient
):
    client.executors = [
        FakeExecutor("e-h1000", "fake", "H1000", 1, 1.0, True),
        FakeExecutor("e-h100", "real", "H100", 1, 2.0, True),
        FakeExecutor("e-nvl", "nvl", "H200 NVL", 1, 3.0, True),
    ]
    assert [o.instance_id for o in provider.search(PodSpec(gpu_type="H100"))] == [
        "e-h100"
    ]
    assert [o.instance_id for o in provider.search(PodSpec(gpu_type="H200"))] == [
        "e-nvl"
    ]


def test_parse_ssh_target():
    t = _parse_ssh_target("ssh ubuntu@1.2.3.4 -p 2200")
    assert t == SshTarget(host="1.2.3.4", port=2200, user="ubuntu")
    with pytest.raises(ProvisionError):
        _parse_ssh_target("")


def test_search_filters(provider: LiumProvider, client: FakeClient):
    client.executors = [
        FakeExecutor("e1", "cheap", "H200", 1, 1.5, True),
        FakeExecutor("e2", "no-dind", "H200", 1, 1.0, False),
        FakeExecutor("e3", "pricey", "H200", 1, 50.0, True),
        FakeExecutor("e4", "a100", "A100", 1, 2.0, True),
        FakeExecutor("e5", "two", "H200", 2, 3.0, True),
    ]
    offers = provider.search(
        PodSpec(gpu_count=1, gpu_type="H200", max_hourly_cents=500)
    )
    # e5 is oversized but usable; it sorts behind the exact-size e1.
    assert [o.instance_id for o in offers] == ["e1", "e5"]
    assert offers[0].hourly_price_cents == 150


def test_search_falls_back_to_oversized_node(
    provider: LiumProvider, client: FakeClient
):
    """No 1x on the market, so the run takes a bigger node and uses one GPU."""
    client.executors = [
        FakeExecutor("e1", "eight", "H200", 8, 25.6, True),
        FakeExecutor("e2", "a100", "A100", 8, 2.0, True),
    ]
    offers = provider.search(
        PodSpec(gpu_count=1, gpu_type="H200", max_hourly_cents=7500)
    )
    assert [o.instance_id for o in offers] == ["e1"]
    assert offers[0].gpu_count == 8


def test_search_prefers_fastest_download(provider: LiumProvider, client: FakeClient):
    client.executors = [
        FakeExecutor("slow", "slow", "RTX5090", 8, 1.0, True, 100.0),
        FakeExecutor("fast", "fast", "RTX5090", 8, 3.0, True, 600.0),
        FakeExecutor("unknown", "unknown", "RTX5090", 8, 0.5, True),
        FakeExecutor(
            "specs",
            "specs",
            "RTX5090",
            8,
            4.0,
            True,
            None,
            {"network": {"download_speed": 300.0}},
        ),
    ]
    offers = provider.search(
        PodSpec(gpu_count=8, gpu_type="RTX5090", max_hourly_cents=1000)
    )
    assert [o.instance_id for o in offers] == ["fast", "specs", "slow", "unknown"]
    assert offers[0].raw["download_mbps"] == 600.0
    assert offers[-1].raw["download_mbps"] == 0.0


def test_search_price_cap_still_filters_fast_executor(
    provider: LiumProvider, client: FakeClient
):
    client.executors = [
        FakeExecutor("fast-pricey", "fast", "H200", 1, 50.0, True, 900.0),
        FakeExecutor("slow-cheap", "slow", "H200", 1, 1.0, True, 100.0),
    ]
    offers = provider.search(
        PodSpec(gpu_count=1, gpu_type="H200", max_hourly_cents=500)
    )
    assert [o.instance_id for o in offers] == ["slow-cheap"]


def test_provision_and_destroy(
    provider: LiumProvider, client: FakeClient, state_dir: Path
):
    client.executors = [FakeExecutor("e1", "node", "H200", 1, 2.0, True)]
    pub = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKey material"
    name = encode_pod_name(ttl_hours=1.0)
    offer = provider.search(PodSpec(gpu_count=1, max_hourly_cents=1000))[0]
    pod = provider.provision(offer, name=name, ssh_public_key=pub)
    assert pod.provider == "lium"
    assert pod.pod_id == "pod-1"
    assert pod.ssh.user == "root"
    assert pod.ssh.host == "10.0.0.5"
    assert pod.ssh.port == 2222
    assert pod.raw["volume_uid"] == "vol-1"
    assert pod.raw["volume_name"] == name
    assert client.up_calls[0]["executor_id"] == "e1"
    assert client.up_calls[0]["volume_id"] == "vol-1"
    assert client.registered[0][0] == "pareton-gpu"
    assert pod.key_path.is_file() or pod.key_path.exists()

    provider.destroy(pod)
    assert client.down_ids == ["pod-1"]
    assert client.deleted_volumes == ["vol-1"]


def test_provision_aborts_volume_on_up_fail(provider: LiumProvider, client: FakeClient):
    client.executors = [FakeExecutor("e1", "node", "H200", 1, 2.0, True)]
    client.up_fail = True
    offer = Offer(
        provider="lium",
        instance_id="e1",
        description="n",
        hourly_price_cents=200,
        gpu_count=1,
        gpu_type="H200",
        raw={"executor_id": "e1"},
    )
    with pytest.raises(RuntimeError, match="up failed"):
        provider.provision(offer, name="pt-test", ssh_public_key="ssh-ed25519 AAAA x")
    assert client.deleted_volumes == ["vol-1"]
    assert client.down_ids == []


def test_provision_aborts_on_wait_timeout(provider: LiumProvider, client: FakeClient):
    client.executors = [FakeExecutor("e1", "node", "H200", 1, 2.0, True)]
    client.wait_none = True
    offer = Offer(
        provider="lium",
        instance_id="e1",
        description="n",
        hourly_price_cents=200,
        gpu_count=1,
        gpu_type="H200",
        raw={"executor_id": "e1"},
    )
    with pytest.raises(ProvisionError, match="not ready"):
        provider.provision(offer, name="pt-test", ssh_public_key="ssh-ed25519 AAAA x")
    assert client.down_ids == ["pod-1"]
    assert client.deleted_volumes == ["vol-1"]


def test_destroy_reports_both_failures(provider: LiumProvider, client: FakeClient):
    from datetime import datetime, timezone

    client.down_fail = True
    client.volume_delete_fail = True
    pod = Pod(
        provider="lium",
        pod_id="pod-x",
        name="pt-x",
        ssh=SshTarget("h", 22, "root"),
        key_path=Path("/tmp"),
        hourly_price_cents=0,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1.0,
        raw={"volume_uid": "vol-x"},
    )
    with pytest.raises(DestroyError, match="volume teardown"):
        provider.destroy(pod)


def test_list_pods_and_volumes(provider: LiumProvider, client: FakeClient):
    client.pods = [
        FakePodInfo("p1", "pt-a", "RUNNING", "ssh root@9.9.9.9 -p 22"),
    ]
    client.vols = [FakeVolume("v1", "pt-a")]
    pods = provider.list_pods()
    assert len(pods) == 1
    assert pods[0].ssh.host == "9.9.9.9"
    vols = provider.list_volumes()
    assert vols[0]["id"] == "v1"


def test_get_provider_lium(monkeypatch, state_dir: Path, client: FakeClient):
    monkeypatch.setenv("PARETON_LIUM_API_KEY", "secret")
    from gpu.providers import get_provider

    p = get_provider("lium", state_dir=state_dir, client=client)
    assert p.name == "lium"


def test_resolve_provider_auto_respects_env(monkeypatch):
    monkeypatch.delenv("PARETON_GPU_PROVIDERS", raising=False)
    monkeypatch.setenv("PARETON_GPU_PROVIDER", "targon")
    monkeypatch.delenv("PARETON_GPU_PROVIDER_FALLBACKS", raising=False)
    from gpu.providers import resolve_provider_name

    assert resolve_provider_name("auto") == "targon"
    assert resolve_provider_name("lium") == "lium"


def test_get_provider_auto_uses_lium_not_targon(
    monkeypatch, state_dir: Path, client: FakeClient
):
    monkeypatch.delenv("PARETON_TARGON_API_KEY", raising=False)
    monkeypatch.setenv("PARETON_LIUM_API_KEY", "secret")
    monkeypatch.delenv("PARETON_GPU_PROVIDER", raising=False)
    monkeypatch.delenv("PARETON_GPU_PROVIDER_FALLBACKS", raising=False)
    monkeypatch.setenv("PARETON_GPU_PROVIDERS", "lium,shadeform")
    monkeypatch.setattr(
        "gpu.providers._env_or_config",
        lambda env, attr: "secret" if env == "PARETON_LIUM_API_KEY" else "",
    )
    from gpu.providers import get_provider

    p = get_provider("auto", state_dir=state_dir, client=client)
    assert p.name == "lium"


def test_get_provider_lium_requires_key(monkeypatch):
    monkeypatch.delenv("PARETON_LIUM_API_KEY", raising=False)
    monkeypatch.delenv("LIUM_API_KEY", raising=False)
    monkeypatch.setattr(
        "gpu.providers._env_or_config",
        lambda env, attr: "",
    )
    from gpu.providers import get_provider

    with pytest.raises(ProvisionError, match="PARETON_LIUM_API_KEY"):
        get_provider("lium")
