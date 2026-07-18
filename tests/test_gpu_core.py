"""Offline tests for gpu registry, naming, ssh argv, static_ssh, reap timer units."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu.errors import GpuError, ProvisionError
from gpu.registry import (
    NAME_PREFIX,
    PodRegistry,
    RegistryEntry,
    encode_pod_name,
    is_expired,
    parse_pod_name,
)
from gpu.ssh import REPO_RSYNC_EXCLUDES, SshResult, exec as ssh_exec, push
from gpu.types import Pod, SshTarget

ROOT = Path(__file__).resolve().parents[1]


def _pod(tmp_path: Path, *, key: Path | None = None) -> Pod:
    key = key or (tmp_path / "key")
    key.write_text("dummy", encoding="utf-8")
    return Pod(
        provider="targon",
        pod_id="wl-1",
        name=encode_pod_name(
            ttl_hours=2.0,
            created=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
            uid8="abcd1234",
        ),
        ssh=SshTarget(host="ssh.example.com", port=22, user="wl-1"),
        key_path=key,
        hourly_price_cents=100,
        created_utc=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
        ttl_hours=2.0,
        raw={"volume_uid": "vol-1"},
    )


def test_naming_roundtrip_and_deadline():
    created = datetime(2026, 7, 18, 15, 30, 45, tzinfo=timezone.utc)
    name = encode_pod_name(ttl_hours=1.5, created=created, uid8="deadbeef")
    assert name.startswith(NAME_PREFIX)
    parsed = parse_pod_name(name)
    assert parsed is not None
    c, ttl, deadline = parsed
    assert c == created
    assert ttl == 1.5
    assert deadline == created + timedelta(hours=1.5)


def test_naming_ignores_malformed_and_foreign():
    assert parse_pod_name("cacheon-eval") is None
    assert parse_pod_name("pareton-gpu-bad") is None
    assert is_expired("not-ours") is None


def test_is_expired_with_injectable_clock():
    created = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    name = encode_pod_name(ttl_hours=1.0, created=created, uid8="11112222")
    assert is_expired(name, clock=lambda: created + timedelta(minutes=30)) is False
    assert is_expired(name, clock=lambda: created + timedelta(hours=2)) is True


def test_registry_add_remove_corrupt(tmp_path: Path):
    reg = PodRegistry(tmp_path / "gpu-state")
    entry = RegistryEntry(
        provider="targon",
        pod_id="p1",
        name="pareton-gpu-20260718-120000-2h-abcd1234",
        deadline="2026-07-18T14:00:00Z",
        hourly_price_cents=200,
        volume_uid="v1",
        state="active",
    )
    reg.add(entry)
    assert reg.get(entry.name) is not None
    entry.state = "destroy_failed"
    reg.update(entry)
    assert reg.get(entry.name).state == "destroy_failed"
    reg.remove(entry.name)
    assert reg.get(entry.name) is None

    # Corrupt file fails closed (do not drop single-flight guard)
    from gpu.errors import GpuError

    reg.path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(GpuError, match="corrupt registry"):
        reg.list()
    assert reg.path.with_suffix(".json.corrupt").is_file()


def test_corrupt_registry_blocks_provision(tmp_path: Path):
    from gpu.errors import GpuError
    from gpu.keys import ensure_durable_keypair
    from gpu.orchestrate import provision_pod
    from gpu.types import Offer, PodSpec

    ensure_durable_keypair(tmp_path / "st")
    reg = PodRegistry(tmp_path / "st")
    reg.path.write_text("{not-json", encoding="utf-8")

    class FakeProvider:
        name = "targon"

        def search(self, spec):
            return [
                Offer(
                    provider="targon",
                    instance_id="r",
                    description="x",
                    hourly_price_cents=1,
                    gpu_count=1,
                    gpu_type="H200",
                )
            ]

        def provision(self, offer, *, name, ssh_public_key):
            raise AssertionError("must not rent when registry is corrupt")

    with pytest.raises(GpuError, match="corrupt registry"):
        provision_pod(
            PodSpec(provider="targon", force=False),
            registry=reg,
            provider=FakeProvider(),
            state_dir=tmp_path / "st",
        )


def test_single_flight_blocking(tmp_path: Path):
    reg = PodRegistry(tmp_path / "gpu-state")
    assert reg.has_blocking_managed() is None
    reg.add(
        RegistryEntry(
            provider="targon",
            pod_id="p1",
            name="pareton-gpu-20260718-120000-2h-abcd1234",
            deadline="2026-07-18T14:00:00Z",
            hourly_price_cents=1,
            state="active",
        )
    )
    assert reg.has_blocking_managed() is not None
    reg.add(
        RegistryEntry(
            provider="static_ssh",
            pod_id="s1",
            name="static-box",
            deadline="",
            hourly_price_cents=0,
            state="active",
        )
    )
    # static does not clear the targon block
    assert reg.has_blocking_managed().provider == "targon"


def test_provision_lock_exclusive(tmp_path: Path):
    import threading

    reg = PodRegistry(tmp_path / "gpu-state")
    held = threading.Event()
    release = threading.Event()
    entered_second = threading.Event()

    def holder():
        with reg.provision_lock():
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    assert held.wait(timeout=2)

    def waiter():
        with reg.provision_lock():
            entered_second.set()

    t2 = threading.Thread(target=waiter)
    t2.start()
    assert not entered_second.wait(timeout=0.3)
    release.set()
    t2.join(timeout=2)
    t.join(timeout=2)
    assert entered_second.is_set()


def test_ssh_exec_argv_and_nonzero(tmp_path: Path):
    pod = _pod(tmp_path)
    calls: list[list[str]] = []

    def runner(cmd, *, timeout, input_text=None):
        calls.append(list(cmd))
        return SshResult(1, "", "boom-stderr-tail")

    with pytest.raises(GpuError, match="boom-stderr-tail"):
        ssh_exec(pod, "true", runner=runner, state_dir=tmp_path / "st")
    argv = calls[0]
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert str(pod.key_path) in argv
    assert any("UserKnownHostsFile=" in a for a in argv)


def test_rsync_excludes_env(tmp_path: Path):
    assert ".env" in REPO_RSYNC_EXCLUDES
    pod = _pod(tmp_path)
    src = tmp_path / "repo"
    src.mkdir()
    (src / "a.txt").write_text("x", encoding="utf-8")
    seen: list[list[str]] = []

    def runner(cmd, *, timeout, input_text=None):
        seen.append(list(cmd))
        return SshResult(0, "", "")

    push(pod, src, "/opt/pareton/", runner=runner, state_dir=tmp_path / "st")
    flat = " ".join(seen[0])
    assert "--exclude .env" in flat or flat.count(".env") >= 1


def test_static_ssh_never_discovers_home_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from gpu.providers import static_ssh as mod

    monkeypatch.setenv("PARETON_GPU_STATIC_SSH", "ubuntu@1.2.3.4:22")
    monkeypatch.delenv("PARETON_GPU_SSH_KEY_PATH", raising=False)

    # Missing key path errors cleanly; no ~/.ssh discovery helper exists.
    with pytest.raises(ProvisionError, match="PARETON_GPU_SSH_KEY_PATH"):
        mod.StaticSshProvider()

    key = tmp_path / "id_ed25519"
    key.write_text("k", encoding="utf-8")
    monkeypatch.setenv("PARETON_GPU_SSH_KEY_PATH", str(key))
    p = mod.StaticSshProvider()
    assert p._key_path == key.resolve()


def test_reap_timer_unit_interval():
    timer = (ROOT / "ops" / "gpu" / "pareton-gpu-reap.timer").read_text(
        encoding="utf-8"
    )
    service = (ROOT / "ops" / "gpu" / "pareton-gpu-reap.service").read_text(
        encoding="utf-8"
    )
    assert "OnUnitActiveSec=10min" in timer
    assert "python -m gpu reap" in service
