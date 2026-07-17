"""Unit tests for bench.lifecycle — fake docker runner + subprocess health loop.

No Docker required. Keep this in the default `pytest tests -q` suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench.lifecycle import (
    NAME_PREFIX,
    BenchNetwork,
    DockerResult,
    EngineContainer,
    EngineError,
    _redact_cmd_for_log,
    extract_digest_from_image_ref,
    normalize_image_id,
    published_host_port,
    resolve_image_digest,
    wait_until_healthy,
)
from bench.mock_engine import MockEngine, MockEngineConfig
from bench.schemas import EngineSpec

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fake docker runner
# ---------------------------------------------------------------------------


class FakeDocker:
    """Records docker argv and returns scripted results."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self._handlers: list[Any] = []
        self.containers: dict[str, dict[str, Any]] = {}
        self.networks: set[str] = set()
        self.next_container_id = "ciddeadbeef01"
        self.image_digests: dict[str, list[str]] = {}
        self.image_ids: dict[str, str] = {}
        self.published_ports: dict[str, str] = {}
        self.running: dict[str, bool] = {}
        self.logs: dict[str, str] = {}
        self.fail_cmds: set[str] = set()  # first token after docker to fail
        self.pull_ok = True
        self.container_exits_after_n_inspects = None
        self._inspect_count = 0

    def add_handler(self, pred, result_fn) -> None:
        self._handlers.append((pred, result_fn))

    def __call__(
        self,
        cmd: list[str] | tuple[str, ...],
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> DockerResult:
        argv = list(cmd)
        self.calls.append((argv, timeout))
        assert timeout > 0, "every docker call must have a positive timeout"

        for pred, result_fn in self._handlers:
            if pred(argv):
                return result_fn(argv)

        if argv[:2] == ["docker", "network"] and argv[2] == "create":
            name = argv[-1]
            self.networks.add(name)
            return DockerResult(0, name + "\n", "")
        if argv[:3] == ["docker", "network", "rm"]:
            self.networks.discard(argv[-1])
            return DockerResult(0, "", "")
        if argv[:2] == ["docker", "pull"]:
            if not self.pull_ok:
                return DockerResult(1, "", "pull failed")
            return DockerResult(0, "pulled\n", "")
        if argv[:2] == ["docker", "inspect"]:
            return self._inspect(argv)
        if argv[:2] == ["docker", "port"]:
            cid = argv[2]
            mapping = self.published_ports.get(cid, "127.0.0.1:49153")
            return DockerResult(0, mapping + "\n", "")
        if argv[:2] == ["docker", "logs"]:
            cid = argv[-1]
            return DockerResult(0, self.logs.get(cid, "log-line\n"), "")
        if argv[:2] == ["docker", "rm"]:
            cid = argv[-1]
            self.containers.pop(cid, None)
            self.running.pop(cid, None)
            return DockerResult(0, "", "")
        if argv[:2] == ["docker", "run"]:
            return self._run(argv)
        return DockerResult(1, "", f"unhandled: {argv}")

    def _inspect(self, argv: list[str]) -> DockerResult:
        fmt = ""
        if "--format" in argv:
            fmt = argv[argv.index("--format") + 1]
        target = argv[-1]

        if "RepoDigests" in fmt:
            digests = self.image_digests.get(target, [])
            return DockerResult(0, json.dumps(digests) + "\n", "")
        if "{{.Id}}" in fmt:
            iid = self.image_ids.get(target, "sha256:" + ("a" * 64))
            return DockerResult(0, iid + "\n", "")
        if "IPAddress" in fmt:
            return DockerResult(0, "172.18.0.2\n", "")
        if "Running" in fmt:
            # Count only Running probes so pre-start image inspects don't
            # burn the fail-fast budget.
            self._inspect_count += 1
            if (
                self.container_exits_after_n_inspects is not None
                and self._inspect_count > self.container_exits_after_n_inspects
            ):
                for cid in list(self.running):
                    self.running[cid] = False
            running = self.running.get(target, True)
            for cid, is_up in self.running.items():
                if target.startswith(cid) or cid.startswith(target):
                    running = is_up
            return DockerResult(0, ("true" if running else "false") + "\n", "")
        return DockerResult(0, "{}\n", "")

    def _run(self, argv: list[str]) -> DockerResult:
        cid = self.next_container_id
        name = ""
        if "--name" in argv:
            name = argv[argv.index("--name") + 1]
        self.containers[cid] = {"name": name, "argv": argv}
        self.running[cid] = True
        self.logs[cid] = f"started {name}\n"
        return DockerResult(0, cid + "\n", "")


def _spec(
    image: str = "ghcr.io/example/engine@sha256:" + ("b" * 64),
    serve_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> EngineSpec:
    return EngineSpec(
        image=image,
        serve_args=serve_args or ["--enable-prefix-caching"],
        env=env or {},
    )


# ---------------------------------------------------------------------------
# Helpers / pure functions
# ---------------------------------------------------------------------------


def test_redact_env_values_from_logged_commands():
    cmd = [
        "docker",
        "run",
        "-e",
        "HF_TOKEN=supersecret",
        "--env",
        "FOO=bar",
        "--env-file",
        "/tmp/x.env",
        "img",
    ]
    redacted = _redact_cmd_for_log(cmd)
    assert "supersecret" not in " ".join(redacted)
    assert "bar" not in redacted  # value redacted; key kept as FOO=***
    assert "HF_TOKEN=***" in redacted
    assert "FOO=***" in redacted
    assert "<redacted>" in redacted


def test_extract_digest_from_pinned_ref():
    digest = "sha256:" + ("c" * 64)
    assert extract_digest_from_image_ref(f"ghcr.io/x/y@{digest}") == digest
    assert extract_digest_from_image_ref("pareton-mock-engine:local") is None


def test_normalize_image_id():
    assert normalize_image_id("sha256:" + ("d" * 64)) == "sha256:" + ("d" * 64)
    assert normalize_image_id("e" * 64) == "sha256:" + ("e" * 64)


def test_resolve_digest_pinned_skips_inspect():
    fake = FakeDocker()
    digest = "sha256:" + ("f" * 64)
    image = f"ghcr.io/x/y@{digest}"
    assert resolve_image_digest(image, runner=fake, cmd_timeout_s=30) == digest
    # No inspect needed when already pinned
    assert not any(c[0][:2] == ["docker", "inspect"] for c in fake.calls)


def test_resolve_digest_falls_back_to_image_id_for_local():
    fake = FakeDocker()
    image = "pareton-mock-engine:local"
    fake.image_digests[image] = []  # empty RepoDigests
    fake.image_ids[image] = "sha256:" + ("1" * 64)
    got = resolve_image_digest(image, runner=fake, cmd_timeout_s=30)
    assert got == "sha256:" + ("1" * 64)


def test_resolve_digest_uses_repo_digests_when_present():
    fake = FakeDocker()
    image = "ghcr.io/x/y:tag"
    digest = "sha256:" + ("2" * 64)
    fake.image_digests[image] = [f"ghcr.io/x/y@{digest}"]
    got = resolve_image_digest(image, runner=fake, cmd_timeout_s=30)
    assert got == digest


def test_published_host_port_parses_docker_port():
    fake = FakeDocker()
    fake.published_ports["cid1"] = "127.0.0.1:54321"
    assert published_host_port("cid1", 8000, runner=fake, cmd_timeout_s=30) == 54321


# ---------------------------------------------------------------------------
# Network + container with fake docker
# ---------------------------------------------------------------------------


def test_network_create_internal_and_unique_name():
    fake = FakeDocker()
    with BenchNetwork(run_id="abc123def456", runner=fake, cmd_timeout_s=30) as net:
        assert net.name == f"{NAME_PREFIX}abc123def456"
        create = fake.calls[0][0]
        assert create[:3] == ["docker", "network", "create"]
        assert "--internal" in create
        assert create[-1] == net.name
    # teardown removed network
    assert not fake.networks


def test_engine_container_command_construction(tmp_path: Path):
    fake = FakeDocker()
    weights = tmp_path / "weights"
    weights.mkdir()
    logs = tmp_path / "logs"
    spec = _spec(
        image="pareton-mock-engine:local",
        serve_args=["--host", "0.0.0.0"],
        env={"HF_TOKEN": "sekrit", "OTHER": "x"},
    )
    fake.image_digests[spec.image] = []
    fake.image_ids[spec.image] = "sha256:" + ("3" * 64)

    with BenchNetwork(run_id="run001run001", runner=fake, cmd_timeout_s=30) as net:
        # Skip real health HTTP — patch wait by making container "healthy" via
        # a short-circuit: we monkeypatch wait_until_healthy below.
        import bench.lifecycle as life

        original_wait = life.wait_until_healthy

        def instant_healthy(*_a, **_k):
            return None

        life.wait_until_healthy = instant_healthy  # type: ignore[assignment]
        try:
            with EngineContainer(
                spec=spec,
                network=net,
                role="baseline",
                gpu_count=1,
                weights_dir=weights,
                pull=False,
                publish_port=False,
                logs_dir=logs,
                health_timeout_s=5,
                health_poll_s=0.1,
                cmd_timeout_s=30,
                pull_timeout_s=60,
            ) as handle:
                assert handle.image_digest == "sha256:" + ("3" * 64)
                assert handle.base_url.startswith("http://172.18.0.2:")
                assert handle.container_name == f"{NAME_PREFIX}run001run001-baseline"
        finally:
            life.wait_until_healthy = original_wait  # type: ignore[assignment]

    run_calls = [c for c, _t in fake.calls if c[:2] == ["docker", "run"]]
    assert len(run_calls) == 1
    run = run_calls[0]
    assert "--network" in run
    assert net.name in run
    assert "--gpus" in run
    assert run[run.index("--gpus") + 1] == "1"
    assert "all" not in run
    vol = next(a for a in run if a.startswith(str(weights)))
    assert vol.endswith(":/model:ro")
    assert "--env-file" in run
    # secrets must not appear on argv
    assert "sekrit" not in run
    assert "HF_TOKEN=sekrit" not in run
    # serve_args after image
    img_idx = run.index(spec.image)
    assert run[img_idx + 1 :] == ["--host", "0.0.0.0"]

    # teardown: logs then rm
    ops = [c[:2] for c, _ in fake.calls]
    logs_idx = next(i for i, o in enumerate(ops) if o == ["docker", "logs"])
    rm_idx = next(i for i, o in enumerate(ops) if o == ["docker", "rm"])
    assert logs_idx < rm_idx
    log_files = list(logs.glob("*.log"))
    assert len(log_files) == 1
    assert "started" in log_files[0].read_text(encoding="utf-8")


def test_publish_port_discovers_random_host_port(tmp_path: Path):
    fake = FakeDocker()
    fake.published_ports["ciddeadbeef01"] = "127.0.0.1:45678"
    spec = _spec(image="local:img")
    fake.image_digests[spec.image] = []
    fake.image_ids[spec.image] = "sha256:" + ("4" * 64)

    import bench.lifecycle as life

    original_wait = life.wait_until_healthy
    life.wait_until_healthy = lambda *a, **k: None  # type: ignore[assignment]
    try:
        with BenchNetwork(run_id="pubport00001", runner=fake, cmd_timeout_s=30) as net:
            with EngineContainer(
                spec=spec,
                network=net,
                role="candidate",
                pull=False,
                publish_port=True,
                port=8000,
                health_timeout_s=5,
                health_poll_s=0.1,
                cmd_timeout_s=30,
            ) as handle:
                assert handle.base_url == "http://127.0.0.1:45678"
        run = next(c for c, _ in fake.calls if c[:2] == ["docker", "run"])
        assert "-p" in run
        assert "127.0.0.1::8000" in run
        assert "8000:8000" not in " ".join(run)
    finally:
        life.wait_until_healthy = original_wait  # type: ignore[assignment]


def test_teardown_on_exception_still_removes(tmp_path: Path):
    fake = FakeDocker()
    spec = _spec(image="local:img")
    fake.image_digests[spec.image] = []
    fake.image_ids[spec.image] = "sha256:" + ("5" * 64)

    import bench.lifecycle as life

    def boom(*_a, **_k):
        raise EngineError("health failed")

    original_wait = life.wait_until_healthy
    life.wait_until_healthy = boom  # type: ignore[assignment]
    try:
        with BenchNetwork(run_id="teardown0001", runner=fake, cmd_timeout_s=30) as net:
            with pytest.raises(EngineError, match="health failed"):
                with EngineContainer(
                    spec=spec,
                    network=net,
                    pull=False,
                    logs_dir=tmp_path,
                    health_timeout_s=1,
                    health_poll_s=0.05,
                    cmd_timeout_s=30,
                ):
                    pass
    finally:
        life.wait_until_healthy = original_wait  # type: ignore[assignment]

    assert any(c[:2] == ["docker", "rm"] for c, _ in fake.calls)
    assert not fake.containers


def test_teardown_on_keyboard_interrupt(tmp_path: Path):
    """BaseException (Ctrl-C) during health check must still rm the container."""
    fake = FakeDocker()
    spec = _spec(image="local:img")
    fake.image_digests[spec.image] = []
    fake.image_ids[spec.image] = "sha256:" + ("7" * 64)

    import bench.lifecycle as life

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt()

    original_wait = life.wait_until_healthy
    life.wait_until_healthy = interrupt  # type: ignore[assignment]
    try:
        with BenchNetwork(run_id="kbinterrupt01", runner=fake, cmd_timeout_s=30) as net:
            with pytest.raises(KeyboardInterrupt):
                with EngineContainer(
                    spec=spec,
                    network=net,
                    pull=False,
                    logs_dir=tmp_path,
                    health_timeout_s=1,
                    health_poll_s=0.05,
                    cmd_timeout_s=30,
                ):
                    pass
    finally:
        life.wait_until_healthy = original_wait  # type: ignore[assignment]

    assert any(c[:2] == ["docker", "rm"] for c, _ in fake.calls)
    assert not fake.containers


def test_gpu_count_passed_as_docker_count_not_all():
    fake = FakeDocker()
    spec = _spec(image="local:img")
    fake.image_digests[spec.image] = []
    fake.image_ids[spec.image] = "sha256:" + ("8" * 64)

    import bench.lifecycle as life

    original_wait = life.wait_until_healthy
    life.wait_until_healthy = lambda *a, **k: None  # type: ignore[assignment]
    try:
        with BenchNetwork(run_id="gpucount0002", runner=fake, cmd_timeout_s=30) as net:
            with EngineContainer(
                spec=spec,
                network=net,
                pull=False,
                gpu_count=2,
                health_timeout_s=1,
                health_poll_s=0.05,
                cmd_timeout_s=30,
            ):
                pass
    finally:
        life.wait_until_healthy = original_wait  # type: ignore[assignment]

    run = next(c for c, _ in fake.calls if c[:2] == ["docker", "run"])
    assert run[run.index("--gpus") + 1] == "2"


def test_fail_fast_when_container_exits_during_health():
    fake = FakeDocker()
    # Make Running flip to false after a couple of inspects
    fake.container_exits_after_n_inspects = 2
    fake.running["ciddeadbeef01"] = True
    spec = _spec(image="local:img")
    fake.image_digests[spec.image] = []
    fake.image_ids[spec.image] = "sha256:" + ("6" * 64)
    fake.logs["ciddeadbeef01"] = "traceback: boom\n"

    with BenchNetwork(run_id="diedearly000", runner=fake, cmd_timeout_s=30) as net:
        with pytest.raises(EngineError, match="died before becoming healthy"):
            with EngineContainer(
                spec=spec,
                network=net,
                pull=False,
                # Point health at a URL that will never answer
                publish_port=True,
                health_timeout_s=2.0,
                health_poll_s=0.05,
                cmd_timeout_s=30,
                port=8000,
            ):
                pass


def test_pull_timeout_budget_is_generous():
    fake = FakeDocker()
    timeouts: list[float] = []

    def tracking_runner(cmd, *, timeout, input_text=None):
        timeouts.append(timeout)
        return fake(cmd, timeout=timeout, input_text=input_text)

    import bench.lifecycle as life

    original_wait = life.wait_until_healthy
    life.wait_until_healthy = lambda *a, **k: None  # type: ignore[assignment]
    try:
        with BenchNetwork(
            run_id="pullto000000", runner=tracking_runner, cmd_timeout_s=30
        ) as net:
            with EngineContainer(
                spec=_spec(image="ghcr.io/x/y:tag"),
                network=net,
                pull=True,
                pull_timeout_s=1800,
                cmd_timeout_s=30,
                health_timeout_s=1,
                health_poll_s=0.1,
            ):
                pass
    finally:
        life.wait_until_healthy = original_wait  # type: ignore[assignment]

    pull_timeouts = [t for (c, t) in fake.calls if c[:2] == ["docker", "pull"]]
    assert any(t >= 1800 for t in pull_timeouts)
    # tracking_runner sees the same timeouts as FakeDocker.__call__
    assert any(t >= 1800 for t in timeouts)


# ---------------------------------------------------------------------------
# Real health-check loop against subprocess mock engine (no Docker)
# ---------------------------------------------------------------------------


def test_wait_until_healthy_against_subprocess_mock():
    eng = MockEngine(MockEngineConfig(host="127.0.0.1", port=0))
    eng.start()
    try:
        wait_until_healthy(
            eng.base_url,
            timeout_s=5.0,
            poll_s=0.05,
            is_alive=lambda: True,
        )
    finally:
        eng.stop()


def test_wait_until_healthy_respects_startup_delay_via_cli():
    """Spawn CLI with --startup-delay-s; wait_until_healthy must succeed after."""
    # Bind ephemeral: start briefly to learn a free port, then reuse via CLI.
    probe = MockEngine(MockEngineConfig(host="127.0.0.1", port=0))
    probe.start()
    host, port = probe._server.server_address[:2]  # type: ignore[union-attr]
    probe.stop()

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bench.mock_engine",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--startup-delay-s",
            "0.4",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_healthy(
            f"http://127.0.0.1:{port}",
            timeout_s=5.0,
            poll_s=0.1,
            is_alive=lambda: proc.poll() is None,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_wait_until_healthy_timeout():
    with pytest.raises(EngineError, match="timed out"):
        wait_until_healthy(
            "http://127.0.0.1:1",  # nothing listening
            timeout_s=0.3,
            poll_s=0.05,
            is_alive=lambda: True,
        )


def test_wait_until_healthy_fail_fast_when_alive_false():
    with pytest.raises(EngineError, match="died before becoming healthy"):
        wait_until_healthy(
            "http://127.0.0.1:1",
            timeout_s=5.0,
            poll_s=0.05,
            is_alive=lambda: False,
        )


def test_mock_engine_cli_serves_models():
    probe = MockEngine(MockEngineConfig(host="127.0.0.1", port=0))
    probe.start()
    _host, port = probe._server.server_address[:2]  # type: ignore[union-attr]
    probe.stop()

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bench.mock_engine",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_healthy(
            f"http://127.0.0.1:{port}",
            timeout_s=5.0,
            poll_s=0.05,
            is_alive=lambda: proc.poll() is None,
        )
        from urllib.request import urlopen

        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["object"] == "list"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
