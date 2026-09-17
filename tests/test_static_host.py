"""Static GPU housekeeping and hardware validation, without Docker or SSH."""

import json
from types import SimpleNamespace

import pytest

from gpu.errors import ProvisionError
from gpu.providers.static_ssh import StaticSshProvider
from gpu.static_host import cleanup, host_lock
from gpu.types import PodSpec


class Docker:
    def __init__(self):
        self.calls = []
        self.fail_images = {}
        self.fail_containers = False

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "ps":
            return SimpleNamespace(
                stdout="pareton-bench-012345abcdef-baseline\nother-app\npareton-bench-not-ours\n"
            )
        if args[:2] == ("network", "ls"):
            return SimpleNamespace(
                stdout="pareton-bench-012345abcdef\nbridge\nother-network\n"
            )
        if args[0] == "rm" and self.fail_containers:
            raise RuntimeError("container removal failed")
        if args[:2] == ("image", "rm") and args[2] in self.fail_images:
            raise RuntimeError(self.fail_images[args[2]])
        return SimpleNamespace(stdout="")


def test_cleanup_reclaims_only_owned_resources(tmp_path):
    tracked = tmp_path / "images.json"
    tracked.write_text(json.dumps(["old-candidate", "current-candidate", "baseline"]))
    output = tmp_path / "out"
    stale = output / "static-pt-20260915120000-2h-0123abcd"
    stale.mkdir(parents=True)
    (stale / "report.json").write_text("old")
    uncollected = output / "static-pt-20260915110000-2h-fedcba98"
    uncollected.mkdir()
    (uncollected / "report.json").write_text("only copy")
    (output / "operator-report").mkdir()
    docker = Docker()
    cleanup(
        tracked_path=tracked,
        candidates={"current-candidate", "next-candidate"},
        keep={"baseline", "current-candidate", "next-candidate"},
        docker=docker,
        output_root=output,
    )
    assert ("rm", "-f", "-v", "pareton-bench-012345abcdef-baseline") in docker.calls
    assert ("network", "rm", "pareton-bench-012345abcdef") in docker.calls
    assert [c for c in docker.calls if c[:2] == ("image", "rm")] == [
        ("image", "rm", "old-candidate")
    ]
    assert set(json.loads(tracked.read_text())) == {
        "baseline",
        "current-candidate",
        "next-candidate",
    }
    assert (stale / "report.json").read_text() == "old"
    assert (output / "operator-report").is_dir()
    assert not any("other-app" in call or "bridge" in call for call in docker.calls)

    cleanup(
        tracked_path=tracked,
        candidates=set(),
        keep={"baseline"},
        docker=docker,
        output_root=output,
        collected_output=stale.name,
    )
    assert not stale.exists()
    assert (uncollected / "report.json").read_text() == "only copy"
    assert json.loads(tracked.read_text()) == ["baseline"]
    assert ("image", "rm", "baseline") not in docker.calls


def test_failed_image_removal_stays_tracked_without_force(tmp_path):
    tracked = tmp_path / "images.json"
    tracked.write_text(json.dumps(["busy", "already-gone"]))
    docker = Docker()
    docker.fail_images = {
        "busy": "image in use",
        "already-gone": "No such image: already-gone",
    }
    failures = cleanup(
        tracked_path=tracked, candidates={"next"}, keep=set(), docker=docker
    )
    assert failures == ["image in use"]
    assert json.loads(tracked.read_text()) == ["busy", "next"]
    assert all("-f" not in c for c in docker.calls if c[:2] == ("image", "rm"))
    # A sticky image does not prevent registering later rounds; retry succeeds
    # once the external reference is released.
    assert cleanup(
        tracked_path=tracked, candidates={"later"}, keep={"later"}, docker=docker
    ) == ["image in use"]
    assert json.loads(tracked.read_text()) == ["busy", "later"]
    docker.fail_images.clear()
    assert (
        cleanup(tracked_path=tracked, candidates=set(), keep={"later"}, docker=docker)
        == []
    )
    assert json.loads(tracked.read_text()) == ["later"]


def test_cleanup_cli_distinguishes_image_retry(tmp_path, monkeypatch, caplog):
    from gpu import static_host

    monkeypatch.setattr(static_host, "REMOTE_LOCK", str(tmp_path / "host.lock"))
    monkeypatch.setattr(static_host, "cleanup", lambda **kw: ["image in use"])
    assert static_host.main([]) == static_host.IMAGE_RETRY_EXIT
    assert "candidate image cleanup needs retry" in caplog.text


def test_container_cleanup_failure_stops_preparation(tmp_path):
    tracked = tmp_path / "images.json"
    tracked.write_text('["candidate"]')
    docker = Docker()
    docker.fail_containers = True
    with pytest.raises(RuntimeError, match="container removal"):
        cleanup(tracked_path=tracked, candidates=set(), keep=set(), docker=docker)
    assert json.loads(tracked.read_text()) == ["candidate"]
    assert not any(c[:2] == ("image", "rm") for c in docker.calls)


def test_corrupt_image_tracking_fails_before_cleanup(tmp_path):
    tracked = tmp_path / "images.json"
    tracked.write_text('{"bad": "state"}')
    docker = Docker()
    with pytest.raises(ValueError, match="list of image references"):
        cleanup(tracked_path=tracked, candidates=set(), keep=set(), docker=docker)
    assert not docker.calls


def test_host_lock_refuses_overlap_and_releases_after_failure(tmp_path):
    path = tmp_path / "host.lock"
    with pytest.raises(ValueError):
        with host_lock(path):
            with pytest.raises(RuntimeError, match="busy"):
                with host_lock(path):
                    pytest.fail("overlapping session acquired the host")
            raise ValueError("run crashed")
    with host_lock(path):
        pass


def test_cleanup_cli_reports_busy_without_touching_resources(tmp_path, monkeypatch):
    from gpu import static_host

    lock = tmp_path / "host.lock"
    monkeypatch.setattr(static_host, "REMOTE_LOCK", str(lock))
    calls = []
    monkeypatch.setattr(static_host, "cleanup", lambda **kw: calls.append(kw))
    with host_lock(lock):
        assert static_host.main([]) == static_host.HOST_BUSY_EXIT
    assert not calls
    assert static_host.main([]) == 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    "names,expected,count,ok",
    [
        (["NVIDIA GeForce RTX 5090"] * 4, "RTX5090", 4, True),
        (["NVIDIA GeForce RTX 5090"] * 8, "NVIDIA-RTX5090", 4, True),
        (["NVIDIA GeForce RTX 5090"] * 4, "NVIDIA-H200", 4, False),
        (["NVIDIA GeForce RTX 5090"] * 2, "RTX5090", 4, False),
        (["NVIDIA H200", "NVIDIA GeForce RTX 5090"], "RTX5090", 1, False),
        ([], "RTX5090", 4, False),
        (["NVIDIA H1000"], "H100", 1, False),
        (["NVIDIA H200-SXM-141GB"], "H200", 1, True),
        (["NVIDIA H200 NVL"], "NVIDIA-H200", 1, True),
        (["NVIDIA H100 80GB HBM3"], "H100", 1, True),
        (["NVIDIA H1000 80GB HBM3"], "H100", 1, False),
        (["NVIDIA GeForce RTX 50900"], "RTX5090", 1, False),
    ],
)
def test_static_provider_checks_real_hardware(
    tmp_path, monkeypatch, names, expected, count, ok, caplog
):
    key = tmp_path / "id"
    key.write_text("fake")
    monkeypatch.setenv("PARETON_GPU_SSH_KEY_PATH", str(key))
    monkeypatch.setattr(
        "gpu.providers.static_ssh.ssh_exec",
        lambda *a, **k: SimpleNamespace(stdout="\n".join(names)),
    )
    provider = StaticSshProvider("root@host:22")
    offer = provider.search(PodSpec(gpu_type=expected, gpu_count=count))[0]
    if ok:
        pod = provider.provision(offer, name="manual-host", ssh_public_key="")
        provider.destroy(pod)  # Still a no-op, with no Lium API calls.
        if len(names) > count:
            assert "4 idle" in caplog.text
    else:
        with pytest.raises(ProvisionError, match="hardware mismatch"):
            provider.provision(offer, name="manual-host", ssh_public_key="")


def test_periodic_cleanup_skips_live_harness_then_reclaims_orphans(
    tmp_path, monkeypatch
):
    from gpu.static_host import reap_idle_containers

    monkeypatch.setattr(
        "gpu.static_host.cleanup",
        lambda **kw: pytest.fail("periodic cleanup must not touch images or reports"),
    )
    docker = Docker()
    checked = []
    lock = tmp_path / "host.lock"
    with host_lock(lock):
        assert (
            reap_idle_containers(
                lock_path=lock, docker=docker, verify=lambda: checked.append(True)
            )["status"]
            == "busy"
        )
    assert not docker.calls and not checked
    result = reap_idle_containers(
        lock_path=lock, docker=docker, verify=lambda: checked.append(True)
    )
    assert result == {"status": "cleaned", "containers_removed": 1}
    assert checked == [True]
    assert not any(call[:2] == ("image", "rm") for call in docker.calls)


def test_gpu_check_waits_for_exit_and_reports_stuck_processes():
    from gpu.static_host import check_idle_gpu

    outputs = iter(["123\n", ""])
    pauses = []
    check_idle_gpu(
        runner=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=next(outputs), stderr=""
        ),
        sleep=pauses.append,
    )
    assert pauses == [1]
    with pytest.raises(RuntimeError, match="processes remain.*123"):
        check_idle_gpu(
            runner=lambda *a, **k: SimpleNamespace(
                returncode=0, stdout="123\n", stderr=""
            ),
            sleep=lambda _s: None,
        )
    with pytest.raises(RuntimeError, match="cannot verify idle GPU"):
        check_idle_gpu(
            runner=lambda *a, **k: SimpleNamespace(
                returncode=1, stdout="", stderr="driver unavailable"
            ),
        )


@pytest.mark.parametrize("status", ["cleaned", "busy", "error", "dry-run"])
def test_existing_reaper_services_static_host(tmp_path, monkeypatch, status):
    from gpu import reap as module
    from gpu.registry import PodRegistry
    from gpu.errors import GpuError

    monkeypatch.setattr(module, "configured_providers", lambda: ["static_ssh"])
    monkeypatch.setattr(module, "_configured_cloud_providers", lambda **k: [])
    calls = []
    alerts = []
    monkeypatch.setattr(
        module.obs, "static_host_cleanup_failed", lambda **k: alerts.append(k)
    )

    def ssh(*args, **kwargs):
        calls.append(args[1])
        if status == "error":
            raise GpuError("GPU compute processes remain after cleanup: 123")
        return SimpleNamespace(
            stdout=json.dumps({"status": status, "containers_removed": 1})
        )

    monkeypatch.setattr(module, "ssh_exec", ssh)
    provider = SimpleNamespace(
        maintenance_pod=lambda: SimpleNamespace(pod_id="root@host:22")
    )
    actions = module.reap(
        registry=PodRegistry(tmp_path),
        dry_run=status == "dry-run",
        provider_factory=lambda *a, **k: provider,
    )
    if status == "busy":
        assert actions == [] and not alerts
    elif status == "dry-run":
        assert actions[0].dry_run and not calls
    elif status == "error":
        assert not actions[0].destroyed and alerts
    else:
        assert actions[0].destroyed and not alerts
    if calls:
        assert "--idle-containers" in calls[0]
