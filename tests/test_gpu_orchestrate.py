"""Offline orchestrate / reap / CLI tests with fakes."""

from __future__ import annotations

import json
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu.errors import DestroyError, GpuError, ProvisionError
from gpu.keys import ensure_durable_keypair
from gpu.orchestrate import provision_pod, run_bench_on_pod
from gpu.providers import provider_order
from gpu.reap import reap
from gpu.registry import PodRegistry, RegistryEntry, encode_pod_name
from gpu.ssh import SshResult
from gpu.types import Offer, Pod, PodSpec, SshTarget

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "fixtures" / "bench" / "sample_request.json"
SAMPLE_TRACE = ROOT / "fixtures" / "bench" / "sample_trace.json"


class FakeProvider:
    name = "targon"

    def __init__(self) -> None:
        self.pods: list[Pod] = []
        self.volumes: list[dict[str, Any]] = []
        self.destroy_calls: list[str] = []
        self.fail_destroy = False
        self.provision_calls = 0

    def search(self, spec: PodSpec) -> list[Offer]:
        return [
            Offer(
                provider=self.name,
                instance_id="res",
                description="fake",
                hourly_price_cents=100,
                gpu_count=spec.gpu_count,
                gpu_type=spec.gpu_type or "H200",
            )
        ]

    def provision(self, offer: Offer, *, name: str, ssh_public_key: str) -> Pod:
        self.provision_calls += 1
        key = Path("/tmp/fake-key")
        pod = Pod(
            provider=self.name,
            pod_id=f"wl-{self.provision_calls}",
            name=name,
            ssh=SshTarget(host="h", port=22, user=f"wl-{self.provision_calls}"),
            key_path=key,
            hourly_price_cents=offer.hourly_price_cents,
            created_utc=datetime.now(timezone.utc),
            ttl_hours=2.0,
            raw={"volume_uid": f"vol-{self.provision_calls}", "volume_name": name},
        )
        self.pods.append(pod)
        self.volumes.append({"id": f"vol-{self.provision_calls}", "name": name})
        return pod

    def provision_manual(
        self, offer: Offer, *, name: str, ssh_public_key: str, **_kwargs
    ) -> Pod:
        return self.provision(offer, name=name, ssh_public_key=ssh_public_key)

    def destroy(self, pod: Pod) -> None:
        self.destroy_calls.append(pod.name)
        if self.fail_destroy:
            raise DestroyError("volume stuck")
        self.pods = [p for p in self.pods if p.name != pod.name]
        vid = (pod.raw or {}).get("volume_uid")
        self.volumes = [v for v in self.volumes if v.get("id") != vid]

    def list_pods(self) -> list[Pod]:
        return list(self.pods)

    def list_volumes(self) -> list[dict[str, Any]]:
        return list(self.volumes)

    def _teardown_volume(self, volume_uid: str, *, raise_on_fail: bool = True) -> None:
        before = len(self.volumes)
        self.volumes = [v for v in self.volumes if v.get("id") != volume_uid]
        if len(self.volumes) == before and raise_on_fail:
            raise DestroyError("volume missing/fail")


def test_preflight_rejects_missing_trace_before_provider(tmp_path: Path):
    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(tmp_path / "missing-trace.json")
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    provider = FakeProvider()
    with pytest.raises(ProvisionError, match="preflight|not found"):
        run_bench_on_pod(
            PodSpec(provider="targon", force=True),
            request_path=req_path,
            output_dir=tmp_path / "out",
            mock_engine=True,
            provider=provider,
            state_dir=tmp_path / "st",
        )
    assert provider.provision_calls == 0


def test_orchestrate_failing_bench_still_destroys(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    calls: list[str] = []

    def runner(cmd, *, timeout, input_text=None):
        joined = " ".join(cmd)
        calls.append(joined)
        if "python -m bench" in joined:
            return SshResult(3, "bench-failed\n", "")
        return SshResult(0, "", "")

    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "deadbeef")
    # Avoid real rsync of whole repo — stub push/pull
    monkeypatch.setattr("gpu.orchestrate.push", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate.pull", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    code = run_bench_on_pod(
        PodSpec(provider="targon", force=True, ttl_hours=1),
        request_path=req_path,
        output_dir=tmp_path / "out",
        mock_engine=True,
        provider=provider,
        runner=runner,
        state_dir=tmp_path / "st",
        repo_root=ROOT,
    )
    assert code == 3
    assert provider.destroy_calls


def test_orchestrate_destroy_failure_keeps_registry(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    provider.fail_destroy = True
    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "sha")
    monkeypatch.setattr("gpu.orchestrate.push", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate.pull", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)

    def runner(cmd, *, timeout, input_text=None):
        return SshResult(0, "ok\n", "")

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    reg = PodRegistry(tmp_path / "st")

    code = run_bench_on_pod(
        PodSpec(provider="targon", force=True),
        request_path=req_path,
        output_dir=tmp_path / "out",
        mock_engine=True,
        provider=provider,
        runner=runner,
        registry=reg,
        state_dir=tmp_path / "st",
        repo_root=ROOT,
    )
    from gpu.orchestrate import EXIT_DESTROY_FAILED

    assert code == EXIT_DESTROY_FAILED  # teardown failed; may still be billing
    assert code != 2  # must stay distinct from CLI/preflight exit 2
    entries = reg.list()
    assert len(entries) == 1
    assert entries[0].state == "destroy_failed"
    assert entries[0].volume_uid


def test_registry_add_failure_destroys_cloud_rent(tmp_path: Path):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    reg = PodRegistry(tmp_path / "st")

    def boom_add(_entry):
        raise OSError("disk full")

    reg.add = boom_add  # type: ignore[method-assign]
    with pytest.raises(GpuError, match="registry.add failed"):
        provision_pod(
            PodSpec(provider="targon", force=True),
            registry=reg,
            provider=provider,
            state_dir=tmp_path / "st",
        )
    assert provider.destroy_calls
    assert not provider.pods


def test_preflight_rejects_wrong_trace_sha_before_provider(tmp_path: Path):
    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req["workload_trace"]["sha256"] = "sha256:" + ("0" * 64)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    provider = FakeProvider()
    with pytest.raises(ProvisionError, match="preflight|sha256"):
        run_bench_on_pod(
            PodSpec(provider="targon", force=True),
            request_path=req_path,
            output_dir=tmp_path / "out",
            mock_engine=True,
            provider=provider,
            state_dir=tmp_path / "st",
        )
    assert provider.provision_calls == 0


def test_remote_env_path_under_repo():
    from gpu.bootstrap import REMOTE_REPO
    from gpu.orchestrate import REMOTE_ENV

    assert REMOTE_ENV.startswith(REMOTE_REPO + "/")
    assert not REMOTE_ENV.startswith("/root/")


def test_pull_engine_images_called_when_not_mock(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    pull_calls: list[Any] = []

    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "sha")
    monkeypatch.setattr("gpu.orchestrate.push", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate.pull", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)

    def fake_pull(pod, refs, *, env_file, **kwargs):
        pull_calls.append({"refs": list(refs), "env_file": env_file})

    monkeypatch.setattr("gpu.orchestrate.pull_engine_images", fake_pull)

    def runner(cmd, *, timeout, input_text=None):
        return SshResult(0, "ok\n", "")

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    run_bench_on_pod(
        PodSpec(provider="targon", force=True),
        request_path=req_path,
        output_dir=tmp_path / "out",
        mock_engine=False,
        provider=provider,
        runner=runner,
        state_dir=tmp_path / "st",
        repo_root=ROOT,
    )
    assert len(pull_calls) == 1
    assert pull_calls[0]["env_file"].endswith(".pareton-bench.env")
    assert any("ghcr.io" in r for r in pull_calls[0]["refs"])


def test_bootstrap_error_with_destroy_failure_returns_75(tmp_path: Path, monkeypatch):
    """Destroy-failure exit must win even when the main try raised."""
    from gpu.errors import GpuError
    from gpu.orchestrate import EXIT_DESTROY_FAILED

    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    provider.fail_destroy = True
    monkeypatch.setattr(
        "gpu.orchestrate.bootstrap_pod",
        lambda *a, **k: (_ for _ in ()).throw(GpuError("bootstrap boom")),
    )
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    code = run_bench_on_pod(
        PodSpec(provider="targon", force=True),
        request_path=req_path,
        output_dir=tmp_path / "out",
        mock_engine=True,
        provider=provider,
        state_dir=tmp_path / "st",
        repo_root=ROOT,
    )
    assert code == EXIT_DESTROY_FAILED
    assert provider.destroy_calls


def test_write_remote_env_does_not_put_secrets_in_ssh_argv(tmp_path: Path, monkeypatch):
    from gpu.orchestrate import _write_remote_env
    from gpu.types import Pod, SshTarget

    ensure_durable_keypair(tmp_path / "st")
    secret = "ghp_SECRET_TOKEN_VALUE_xyz"
    monkeypatch.setenv("PARETON_GHCR_TOKEN", secret)
    monkeypatch.setenv("PARETON_GHCR_USER", "u")
    ssh_remotes: list[str] = []

    def runner(cmd, *, timeout, input_text=None):
        if cmd and cmd[0] == "ssh":
            ssh_remotes.append(cmd[-1])
        return SshResult(0, "", "")

    monkeypatch.setattr("gpu.orchestrate.push", lambda *a, **k: None)
    pod = Pod(
        provider="targon",
        pod_id="wl",
        name="n",
        ssh=SshTarget(host="h", port=22, user="u"),
        key_path=tmp_path / "st" / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )
    _write_remote_env(pod, runner=runner, state_dir=tmp_path / "st")
    assert ssh_remotes
    assert all(secret not in r for r in ssh_remotes)


def test_write_remote_env_forwards_health_timeout(tmp_path: Path, monkeypatch):
    from gpu.orchestrate import _write_remote_env
    from gpu.types import Pod, SshTarget

    ensure_durable_keypair(tmp_path / "st")
    monkeypatch.setattr("gpu.orchestrate.config.BENCH_HEALTH_TIMEOUT_S", 1800.0)
    pushed: list[str] = []

    def capturing_push(pod, local, remote, **kwargs):
        pushed.append(Path(local).read_text(encoding="utf-8"))

    monkeypatch.setattr("gpu.orchestrate.push", capturing_push)
    monkeypatch.setattr(
        "gpu.orchestrate.ssh_exec",
        lambda *a, **k: SshResult(0, "", ""),
    )
    pod = Pod(
        provider="targon",
        pod_id="wl",
        name="n",
        ssh=SshTarget(host="h", port=22, user="u"),
        key_path=tmp_path / "st" / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )
    _write_remote_env(pod, runner=None, state_dir=tmp_path / "st")
    assert pushed
    assert "PARETON_BENCH_HEALTH_TIMEOUT_S=1800.0" in pushed[0]


def test_orchestrate_keyboardinterrupt_still_destroys(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "sha")
    monkeypatch.setattr("gpu.orchestrate.push", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)

    def runner(cmd, *, timeout, input_text=None):
        if "python -m bench" in " ".join(cmd):
            raise KeyboardInterrupt()
        return SshResult(0, "", "")

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    with pytest.raises(KeyboardInterrupt):
        run_bench_on_pod(
            PodSpec(provider="targon", force=True),
            request_path=req_path,
            output_dir=tmp_path / "out",
            mock_engine=True,
            provider=provider,
            runner=runner,
            state_dir=tmp_path / "st",
            repo_root=ROOT,
        )
    assert provider.destroy_calls


def test_trace_outside_repo_rewritten(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    pushed: list[str] = []

    def fake_push(pod, local, remote, **kwargs):
        pushed.append(str(remote))

    remote_req_holder: dict[str, Any] = {}

    def capturing_push(pod, local, remote, **kwargs):
        pushed.append(str(remote))
        p = Path(local)
        if p.name.endswith(".json") and "bench_request" in p.name:
            remote_req_holder.update(json.loads(p.read_text(encoding="utf-8")))

    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "sha")
    monkeypatch.setattr("gpu.orchestrate.push", capturing_push)
    monkeypatch.setattr("gpu.orchestrate.pull", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)
    monkeypatch.setattr(
        "gpu.orchestrate.ssh_exec",
        lambda *a, **k: type("R", (), {"exit_code": 0, "stdout": "", "stderr": ""})(),
    )

    ext_trace = tmp_path / "external_trace.json"
    ext_trace.write_bytes(SAMPLE_TRACE.read_bytes())
    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(ext_trace)
    # sha must match file
    from bench.validate import sha256_file

    req["workload_trace"]["sha256"] = sha256_file(ext_trace)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    run_bench_on_pod(
        PodSpec(provider="targon", force=True),
        request_path=req_path,
        output_dir=tmp_path / "out",
        mock_engine=True,
        provider=provider,
        state_dir=tmp_path / "st",
        repo_root=ROOT,
    )
    assert any("external_trace.json" in p for p in pushed)
    assert remote_req_holder["workload_trace"]["path"].endswith("external_trace.json")


def test_orchestrate_pushes_trace_under_excluded_out(tmp_path: Path, monkeypatch):
    """B7 prepare writes traces under out/; bootstrap rsync excludes out/."""
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    pushed: list[str] = []
    remote_req_holder: dict[str, Any] = {}

    def capturing_push(pod, local, remote, **kwargs):
        pushed.append(str(remote))
        p = Path(local)
        if p.name.endswith(".json") and "bench_request" in p.name:
            remote_req_holder.update(json.loads(p.read_text(encoding="utf-8")))

    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "sha")
    monkeypatch.setattr("gpu.orchestrate.push", capturing_push)
    monkeypatch.setattr("gpu.orchestrate.pull", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)
    monkeypatch.setattr(
        "gpu.orchestrate.ssh_exec",
        lambda *a, **k: type("R", (), {"exit_code": 0, "stdout": "", "stderr": ""})(),
    )

    fake_repo = tmp_path / "repo"
    out_b7 = fake_repo / "out" / "b7" / "run"
    out_b7.mkdir(parents=True)
    from bench.validate import sha256_file

    ext_trace = out_b7 / "workload_trace.json"
    ext_trace.write_bytes(SAMPLE_TRACE.read_bytes())
    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(ext_trace.resolve())
    req["workload_trace"]["sha256"] = sha256_file(ext_trace)
    req_path = out_b7 / "bench_request.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    run_bench_on_pod(
        PodSpec(provider="targon", force=True),
        request_path=req_path,
        output_dir=tmp_path / "gpu-out",
        mock_engine=True,
        provider=provider,
        state_dir=tmp_path / "st",
        repo_root=fake_repo,
    )
    assert any(".pareton-traces/workload_trace.json" in p for p in pushed)
    assert remote_req_holder["workload_trace"]["path"].endswith(
        ".pareton-traces/workload_trace.json"
    )
    assert "/out/" not in remote_req_holder["workload_trace"]["path"]


def test_single_flight_and_force(tmp_path: Path):
    ensure_durable_keypair(tmp_path / "st")
    reg = PodRegistry(tmp_path / "st")
    reg.add(
        RegistryEntry(
            provider="targon",
            pod_id="old",
            name=encode_pod_name(ttl_hours=2),
            deadline="2099-01-01T00:00:00Z",
            hourly_price_cents=1,
            state="active",
        )
    )
    provider = FakeProvider()
    with pytest.raises(ProvisionError, match="single-flight"):
        provision_pod(
            PodSpec(provider="targon", force=False),
            registry=reg,
            provider=provider,
            state_dir=tmp_path / "st",
        )
    pod = provision_pod(
        PodSpec(provider="targon", force=True),
        registry=reg,
        provider=provider,
        state_dir=tmp_path / "st",
    )
    assert pod.name.startswith("pt-")


def test_manual_provision_path(tmp_path: Path):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    calls: list[str] = []
    orig = provider.provision_manual

    def tracked(offer, *, name, ssh_public_key, **kwargs):
        calls.append(name)
        return orig(offer, name=name, ssh_public_key=ssh_public_key, **kwargs)

    provider.provision_manual = tracked  # type: ignore[method-assign]
    pod = provision_pod(
        PodSpec(provider="targon", force=True, manual=True, ttl_hours=1.0),
        provider=provider,
        state_dir=tmp_path / "st",
    )
    assert calls and calls[0] == pod.name
    assert provider.provision_calls == 1
    assert PodRegistry(tmp_path / "st").get(pod.name) is not None


def test_reap_expired_and_dry_run(tmp_path: Path):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    created = datetime.now(timezone.utc) - timedelta(hours=5)
    name = encode_pod_name(ttl_hours=1.0, created=created, uid8="aabbccdd")
    provider.pods.append(
        Pod(
            provider="targon",
            pod_id="wl-x",
            name=name,
            ssh=SshTarget(host="h", port=22, user="wl-x"),
            key_path=Path("/tmp/k"),
            hourly_price_cents=1,
            created_utc=created,
            ttl_hours=1.0,
            raw={"volume_uid": "vol-x"},
        )
    )
    provider.volumes.append({"id": "vol-x", "name": name})

    actions = reap(
        dry_run=True,
        state_dir=tmp_path / "st",
        provider_factory=lambda name, **k: provider,
    )
    assert actions
    assert all(a.dry_run for a in actions)
    assert provider.pods  # untouched

    actions = reap(
        dry_run=False,
        state_dir=tmp_path / "st",
        provider_factory=lambda name, **k: provider,
    )
    assert any(a.destroyed for a in actions)
    assert not provider.pods


def test_reap_enriches_volume_uid_from_registry(tmp_path: Path):
    """list_pods-style empty volume_uid must still tear down registry volume."""
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    created = datetime.now(timezone.utc) - timedelta(hours=5)
    name = encode_pod_name(ttl_hours=1.0, created=created, uid8="aabbccdd")
    provider.pods.append(
        Pod(
            provider="targon",
            pod_id="wl-x",
            name=name,
            ssh=SshTarget(host="h", port=22, user="wl-x"),
            key_path=Path("/tmp/k"),
            hourly_price_cents=1,
            created_utc=created,
            ttl_hours=1.0,
            raw={"volume_uid": ""},  # mirrors TargonProvider.list_pods
        )
    )
    provider.volumes.append({"id": "vol-from-reg", "name": name})
    reg = PodRegistry(tmp_path / "st")
    reg.add(
        RegistryEntry(
            provider="targon",
            pod_id="wl-x",
            name=name,
            deadline="2000-01-01T00:00:00Z",
            hourly_price_cents=1,
            volume_uid="vol-from-reg",
            volume_name=name,
            state="active",
        )
    )
    actions = reap(
        dry_run=False,
        registry=reg,
        state_dir=tmp_path / "st",
        provider_factory=lambda name, **k: provider,
    )
    assert any(a.kind == "workload" and a.destroyed for a in actions)
    assert not any(v.get("id") == "vol-from-reg" for v in provider.volumes)
    assert reg.get(name) is None


def test_pull_engine_images_quotes_refs(tmp_path: Path):
    import shlex

    from gpu.bootstrap import pull_engine_images

    ensure_durable_keypair(tmp_path / "st")
    remote_cmds: list[str] = []

    def runner(cmd, *, timeout, input_text=None):
        # ssh ... -- <remote command>
        remote_cmds.append(cmd[-1] if cmd else "")
        return SshResult(0, "", "")

    evil = "ghcr.io/x/y;touch /tmp/pwned@sha256:" + ("a" * 64)
    pod = Pod(
        provider="targon",
        pod_id="wl",
        name="n",
        ssh=SshTarget(host="h", port=22, user="u"),
        key_path=tmp_path / "st" / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )
    pull_engine_images(
        pod,
        [evil],
        env_file="/opt/pareton/.pareton-bench.env",
        runner=runner,
        state_dir=tmp_path / "st",
    )
    assert remote_cmds
    remote = remote_cmds[0]
    # Non-root SSH user → sudo -E docker (Shadeform-compatible).
    assert f"sudo -E docker pull {shlex.quote(evil)}" in remote
    assert "docker pull ghcr.io/x/y;touch" not in remote


def test_remote_docker_root_vs_nonroot():
    from gpu.bootstrap import remote_docker

    root = Pod(
        provider="targon",
        pod_id="wl",
        name="n",
        ssh=SshTarget(host="h", port=22, user="root"),
        key_path=Path("/tmp/k"),
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )
    user = Pod(
        provider="shadeform",
        pod_id="inst",
        name="n",
        ssh=SshTarget(host="h", port=22, user="shadeform"),
        key_path=Path("/tmp/k"),
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )
    assert remote_docker(root) == "docker"
    assert remote_docker(user) == "sudo -E docker"


def test_cli_help_and_missing_key(monkeypatch):
    from gpu.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    for sub in ("provision", "destroy", "list", "reap", "bench", "exec"):
        with pytest.raises(SystemExit) as ei:
            main([sub, "--help"])
        assert ei.value.code == 0

    monkeypatch.delenv("PARETON_TARGON_API_KEY", raising=False)
    import config as cfg

    monkeypatch.setattr(cfg, "TARGON_API_KEY", "", raising=False)
    from gpu.providers import get_provider

    with pytest.raises(ProvisionError, match="PARETON_TARGON_API_KEY"):
        get_provider("targon")


def test_bootstrap_script_verify_first_no_token():
    from gpu.bootstrap import bootstrap_script

    script = bootstrap_script()
    assert "command -v docker" in script
    assert "nvidia-smi" in script
    assert "import ensurepip" in script
    assert "python${PYVER}-venv" in script or "python${PYVER}-venv" in script
    assert "ghp_" not in script
    assert "PARETON_GHCR_TOKEN" not in script
    # gpg must not prompt on existing keyring (no /dev/tty over ssh)
    assert "gpg --batch --yes --dearmor" in script
    # verify-before-install: docker check appears before get.docker.com
    assert script.index("command -v docker") < script.index("get.docker.com")
    # sock ACL after toolkit restart so chmod hits the final socket
    assert script.index("systemctl restart docker") < script.index(
        "chmod 666 /var/run/docker.sock"
    )


def test_orchestrate_repetitions_one_pod_five_runs(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    bench_cmds: list[str] = []

    def runner(cmd, *, timeout, input_text=None):
        joined = " ".join(cmd)
        if "python -m bench" in joined:
            bench_cmds.append(joined)
            return SshResult(0, "ok\n", "")
        return SshResult(0, "", "")

    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "deadbeef")
    monkeypatch.setattr("gpu.orchestrate.push", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate.pull", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    out = tmp_path / "out"

    code = run_bench_on_pod(
        PodSpec(provider="targon", force=True, ttl_hours=1),
        request_path=req_path,
        output_dir=out,
        mock_engine=True,
        repetitions=5,
        provider=provider,
        runner=runner,
        state_dir=tmp_path / "st",
        repo_root=ROOT,
    )
    assert code == 0
    assert provider.provision_calls == 1
    assert len(provider.destroy_calls) == 1
    assert len(bench_cmds) == 5
    for i, cmd in enumerate(bench_cmds, start=1):
        assert f"/opt/pareton/out/run-{i:03d}" in cmd
        assert (out / f"run-{i:03d}").is_dir()
    assert (out / "bench_request.remote.json").is_file()


def test_orchestrate_repetitions_fail_fast_still_destroys(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    n_bench = 0

    def runner(cmd, *, timeout, input_text=None):
        nonlocal n_bench
        joined = " ".join(cmd)
        if "python -m bench" in joined:
            n_bench += 1
            if n_bench == 2:
                return SshResult(3, "fail\n", "")
            return SshResult(0, "ok\n", "")
        return SshResult(0, "", "")

    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "sha")
    monkeypatch.setattr("gpu.orchestrate.push", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate.pull", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")

    code = run_bench_on_pod(
        PodSpec(provider="targon", force=True),
        request_path=req_path,
        output_dir=tmp_path / "out",
        mock_engine=True,
        repetitions=5,
        provider=provider,
        runner=runner,
        state_dir=tmp_path / "st",
        repo_root=ROOT,
    )
    assert code == 3
    assert n_bench == 2
    assert provider.destroy_calls


def test_orchestrate_repetitions_default_one_flat_layout(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    provider = FakeProvider()
    bench_cmds: list[str] = []

    def runner(cmd, *, timeout, input_text=None):
        joined = " ".join(cmd)
        if "python -m bench" in joined:
            bench_cmds.append(joined)
            return SshResult(0, "ok\n", "")
        return SshResult(0, "", "")

    monkeypatch.setattr("gpu.orchestrate.bootstrap_pod", lambda *a, **k: "sha")
    monkeypatch.setattr("gpu.orchestrate.push", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate.pull", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._write_remote_env", lambda *a, **k: None)
    monkeypatch.setattr("gpu.orchestrate._delete_remote_env", lambda *a, **k: None)

    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    out = tmp_path / "out"

    code = run_bench_on_pod(
        PodSpec(provider="targon", force=True),
        request_path=req_path,
        output_dir=out,
        mock_engine=True,
        provider=provider,
        runner=runner,
        state_dir=tmp_path / "st",
        repo_root=ROOT,
    )
    assert code == 0
    assert len(bench_cmds) == 1
    assert "/opt/pareton/out/run-" not in bench_cmds[0]
    assert "--output-dir /opt/pareton/out" in bench_cmds[0]
    assert not (out / "run-001").exists()


def test_orchestrate_repetitions_invalid(tmp_path: Path):
    req = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    req["workload_trace"]["path"] = str(SAMPLE_TRACE)
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    with pytest.raises(ProvisionError, match="repetitions"):
        run_bench_on_pod(
            PodSpec(provider="targon", force=True),
            request_path=req_path,
            output_dir=tmp_path / "out",
            mock_engine=True,
            repetitions=0,
            provider=FakeProvider(),
            state_dir=tmp_path / "st",
        )


class CapacityMissProvider(FakeProvider):
    name = "lium"

    def search(self, spec: PodSpec) -> list[Offer]:
        return []


class ProvisionErrorProvider(FakeProvider):
    name = "lium"

    def provision(self, offer: Offer, *, name: str, ssh_public_key: str) -> Pod:
        raise ProvisionError("simulated lium rent failure")


def _shadeform_fake() -> FakeProvider:
    p = FakeProvider()
    p.name = "shadeform"
    return p


def _patch_provider_factory(monkeypatch, by_name: dict) -> None:
    monkeypatch.setattr(
        "gpu.orchestrate.get_provider", lambda name, **kw: by_name[name]
    )
    monkeypatch.setenv("PARETON_GPU_PROVIDER", "lium")
    monkeypatch.setenv("PARETON_GPU_PROVIDER_FALLBACKS", "shadeform")


def test_provision_fallback_on_capacity_miss(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    lium = CapacityMissProvider()
    shade = _shadeform_fake()
    _patch_provider_factory(monkeypatch, {"lium": lium, "shadeform": shade})
    pod = provision_pod(
        PodSpec(provider="auto", gpu_type="H200"), state_dir=tmp_path / "st"
    )
    assert pod.provider == "shadeform"
    assert shade.provision_calls == 1
    assert PodRegistry(tmp_path / "st").get(pod.name) is not None


def test_provision_fallback_on_provision_error(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    lium = ProvisionErrorProvider()
    shade = _shadeform_fake()
    _patch_provider_factory(monkeypatch, {"lium": lium, "shadeform": shade})
    pod = provision_pod(
        PodSpec(provider="auto", gpu_type="H200"), state_dir=tmp_path / "st"
    )
    assert pod.provider == "shadeform"
    assert shade.provision_calls == 1


def test_provision_fallback_all_fail_raises_last(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    lium = CapacityMissProvider()
    shade = ProvisionErrorProvider()
    shade.name = "shadeform"
    _patch_provider_factory(monkeypatch, {"lium": lium, "shadeform": shade})
    with pytest.raises(ProvisionError, match="simulated lium rent failure"):
        provision_pod(
            PodSpec(provider="auto", gpu_type="H200"), state_dir=tmp_path / "st"
        )


def test_single_flight_blocks_fallback(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    reg = PodRegistry(tmp_path / "st")
    reg.add(
        RegistryEntry(
            provider="lium",
            pod_id="old",
            name=encode_pod_name(ttl_hours=2),
            deadline="2099-01-01T00:00:00Z",
            hourly_price_cents=1,
            state="active",
        )
    )
    lium = CapacityMissProvider()
    shade = _shadeform_fake()
    _patch_provider_factory(monkeypatch, {"lium": lium, "shadeform": shade})
    with pytest.raises(ProvisionError, match="single-flight"):
        provision_pod(
            PodSpec(provider="auto", gpu_type="H200"), state_dir=tmp_path / "st"
        )
    assert shade.provision_calls == 0


def test_registry_add_failure_does_not_fall_back(tmp_path: Path, monkeypatch):
    ensure_durable_keypair(tmp_path / "st")
    lium = FakeProvider()
    shade = _shadeform_fake()
    _patch_provider_factory(monkeypatch, {"lium": lium, "shadeform": shade})

    def fail_add(entry):
        raise OSError("disk full")

    monkeypatch.setattr(PodRegistry, "add", fail_add)
    with pytest.raises(GpuError, match="registry.add failed after rent"):
        provision_pod(
            PodSpec(provider="auto", gpu_type="H200"), state_dir=tmp_path / "st"
        )
    # Lium pod was destroyed after the registry failure; Shadeform never tried.
    assert lium.provision_calls == 1
    assert lium.destroy_calls
    assert shade.provision_calls == 0


def test_provider_order_default_dedup_and_static_ssh(monkeypatch):
    monkeypatch.setenv("PARETON_GPU_PROVIDER", "lium")
    monkeypatch.setenv("PARETON_GPU_PROVIDER_FALLBACKS", "shadeform")
    assert provider_order("auto") == ["lium", "shadeform"]
    assert provider_order("shadeform") == ["shadeform"]
    assert provider_order("static_ssh") == ["static_ssh"]
    monkeypatch.setenv("PARETON_GPU_PROVIDER_FALLBACKS", "")
    assert provider_order("auto") == ["lium"]


def _pod_for_keys(tmp_path: Path) -> Pod:
    ensure_durable_keypair(tmp_path / "st")
    return Pod(
        provider="targon",
        pod_id="wl",
        name="n",
        ssh=SshTarget(host="h", port=22, user="root"),
        key_path=tmp_path / "st" / "keys" / "pareton-gpu-ed25519",
        hourly_price_cents=1,
        created_utc=datetime.now(timezone.utc),
        ttl_hours=1,
    )


def test_authorize_extra_keys_appends_idempotently(tmp_path: Path):
    from gpu.bootstrap import authorize_extra_keys

    remote_cmds: list[str] = []

    def runner(cmd, *, timeout, input_text=None):
        remote_cmds.append(cmd[-1] if cmd else "")
        return SshResult(0, "", "")

    keys = ["ssh-ed25519 AAAAC3Nz xavier", "ssh-ed25519 AAAAB4Qq bohdan"]
    authorize_extra_keys(
        _pod_for_keys(tmp_path),
        keys=keys,
        runner=runner,
        state_dir=tmp_path / "st",
    )
    assert len(remote_cmds) == 1
    remote = remote_cmds[0]
    # Append-only: never truncates the file.
    assert ">>" in remote
    assert ">>>" not in remote
    assert "> $HOME/.ssh/authorized_keys" not in remote.replace(">>", "")
    # Idempotent: each key is guarded by an exact-line grep.
    for key in keys:
        assert f"grep -qxF -- '{key}'" in remote
    assert remote.count("grep -qxF") == 2
    assert "chmod 600" in remote


def test_authorize_extra_keys_noop_when_unset(tmp_path: Path):
    from gpu.bootstrap import authorize_extra_keys

    calls: list[str] = []

    def runner(cmd, *, timeout, input_text=None):
        calls.append(cmd[-1] if cmd else "")
        return SshResult(0, "", "")

    authorize_extra_keys(
        _pod_for_keys(tmp_path), keys=[], runner=runner, state_dir=tmp_path / "st"
    )
    assert calls == []


def test_authorize_extra_keys_quotes_injection(tmp_path: Path):
    from gpu.bootstrap import authorize_extra_keys

    remote_cmds: list[str] = []

    def runner(cmd, *, timeout, input_text=None):
        remote_cmds.append(cmd[-1] if cmd else "")
        return SshResult(0, "", "")

    evil = "ssh-ed25519 AAAA';rm -rf /;'"
    authorize_extra_keys(
        _pod_for_keys(tmp_path),
        keys=[evil],
        runner=runner,
        state_dir=tmp_path / "st",
    )
    assert ";rm -rf /;" not in remote_cmds[0].replace(shlex.quote(evil), "")


def test_parse_pubkeys_accepts_and_dedupes():
    from config import _parse_pubkeys

    x = "AAAAC3NzaC1lZDI1NTE5AAAAI" + "A" * 20
    r = "AAAAB3NzaC1yc2EAAAADAQAB" + "B" * 20
    raw = f"\n{x} xavier\n\n{r}\n{x} xavier\n".replace(x, f"ssh-ed25519 {x}").replace(
        r, f"ssh-rsa {r}"
    )
    assert _parse_pubkeys(raw) == [f"ssh-ed25519 {x} xavier", f"ssh-rsa {r}"]
    assert _parse_pubkeys("") == []


def test_parse_pubkeys_rejects_malformed():
    from config import _parse_pubkeys

    for bad in (
        "not-a-key",
        "ssh-ed25519",
        "AAAAC3Nz xavier",
        "ssh-ed25519 not base64!",
    ):
        with pytest.raises(ValueError):
            _parse_pubkeys(bad)
