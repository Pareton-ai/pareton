"""Container boundary contracts that protect jobs, builder caches and GPU runs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from builder import preflight
from gpu.bootstrap import harness_command, local_code_sha

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("driver", ["docker", "docker-container"])
def test_selected_builder_gc_checked(monkeypatch, tmp_path, driver):
    calls = []
    daemon = tmp_path / "daemon.json"
    daemon.write_text('{"builder":{"gc":{"enabled":false}}}')
    monkeypatch.setattr(preflight.config, "DOCKER_DAEMON_CONFIG_PATH", daemon)
    monkeypatch.setattr(preflight.config, "BUILDER_NAME", "selected")

    def output(*args):
        calls.append(args)
        if args[1:3] == ("buildx", "inspect"):
            return f"Name: selected\nDriver: {driver}\nNodes:\nName: selected0\n"
        return "[worker.oci]\ngc=false\n[worker.containerd]\ngc=false"

    monkeypatch.setattr(preflight, "_output", output)
    preflight.validate_builder()
    assert calls[0][3] == "selected"
    if driver == "docker-container":
        assert calls[1][2] == "buildx_buildkit_selected0"
        monkeypatch.setattr(
            preflight,
            "_output",
            lambda *args: output(*args).replace("gc=false", "gc=true"),
        )
        with pytest.raises(ValueError, match="gc must be false"):
            preflight.validate_builder()
    else:
        daemon.write_text("{}")
        with pytest.raises(ValueError, match="builder.gc.enabled=false"):
            preflight.validate_builder()


def test_source_revision_survives_gitless_image(monkeypatch, tmp_path):
    monkeypatch.setenv("PARETON_CODE_SHA", "abc12345")
    assert local_code_sha(tmp_path) == "abc12345"


@pytest.mark.parametrize("user", ["root", "ubuntu"])
def test_harness_socket_host_paths_and_failure_status(tmp_path, user):
    stub = tmp_path / "bin"
    stub.mkdir()
    log = tmp_path / "argv.jsonl"
    docker = stub / "docker"
    docker.write_text(f"""#!{sys.executable}
import json, os, sys
with open(os.environ["TEST_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(7 if sys.argv[1] == "run" else 0)
""")
    sudo = stub / "sudo"
    sudo.write_text('#!/bin/sh\n[ "$1" = -E ] && shift\nexec "$@"\n')
    docker.chmod(0o755)
    sudo.chmod(0o755)
    output = tmp_path / "evidence with spaces"
    pod = SimpleNamespace(ssh=SimpleNamespace(user=user))
    command = harness_command(
        pod,
        code_sha="deadbeef",
        env_file="/opt/pareton/secret.env",
        request="/opt/pareton/request.json",
        output=str(output),
        mock_engine=True,
    )
    result = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "PATH": f"{stub}:{os.environ['PATH']}",
            "TEST_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7  # Cleanup never hides a failed benchmark.
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    argv = calls[0]
    assert argv[argv.index("--network") + 1] == "host"
    assert "/var/run/docker.sock:/var/run/docker.sock" in argv
    assert "/workspace/hf-cache:/workspace/hf-cache" in argv
    assert "/workspace/engine-cache:/workspace/engine-cache" in argv
    assert "DOCKER_CONFIG=/opt/pareton/.docker" in argv
    assert argv[-3:] == ["--output-dir", str(output), "--mock-engine"]
    assert calls[1][:2] == ["rm", "-f"]
    assert "secret.env" in command and "PARETON_GHCR_TOKEN" not in command


def test_scheduler_drains_active_command_on_sigterm(tmp_path):
    started, finished = tmp_path / "started", tmp_path / "finished"
    child = (
        f"from pathlib import Path; import time; Path({str(started)!r}).touch(); "
        f"time.sleep(0.5); Path({str(finished)!r}).touch()"
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ops.schedule",
            "--interval",
            "60",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists()
        proc.terminate()
        proc.communicate(timeout=10)
        assert proc.returncode == 0
        assert finished.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.docker
def test_named_builder_actual_gc_configuration(monkeypatch):
    """Check real Buildx inspect/config paths using an isolated builder."""
    import shutil
    import uuid

    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode:
        pytest.skip("Docker daemon is unavailable")
    name = "pareton-gc-test-" + uuid.uuid4().hex[:8]
    try:
        subprocess.run(
            [
                "docker",
                "buildx",
                "create",
                "--name",
                name,
                "--driver",
                "docker-container",
                "--buildkitd-config",
                str(ROOT / "ops/buildkitd.toml"),
                "--bootstrap",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        monkeypatch.setattr(preflight.config, "BUILDER_NAME", name)
        preflight.validate_builder()
    finally:
        subprocess.run(
            ["docker", "buildx", "rm", name],
            check=False,
            capture_output=True,
            timeout=120,
        )


@pytest.mark.parametrize("detach,build_status", [(False, 0), (True, 0), (False, 7)])
def test_baseline_helper_runs_in_compose_and_preserves_failures(
    tmp_path, detach, build_status
):
    """Exercise the wrapper and inner build script without registry/network access."""
    stub = tmp_path / "bin"
    stub.mkdir()
    log = tmp_path / "calls.jsonl"
    for tool in ("docker", "python"):
        command = stub / tool
        command.write_text(f"""#!{sys.executable}
import json, os, subprocess, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["TEST_LOG"], "a") as f:
    f.write(json.dumps({{"tool": tool, "args": args,
                        "jobs": os.environ.get("PARETON_BUILD_MAX_JOBS")}}) + "\\n")
if args[:2] == ["compose", "run"]:
    # Simulate Compose env_file credentials; do not place secrets in argv.
    os.environ["PARETON_GHCR_USERNAME"] = "test-user"
    os.environ["PARETON_GHCR_TOKEN"] = "test-token"
    sys.exit(subprocess.run(args[args.index("cli") + 1:]).returncode)
if tool == "python" and args[:2] == ["-m", "builder"]:
    sys.exit({build_status})
""")
        command.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PARETON_", "VLLM_"))
        and key not in {"BASE", "TORCH_CUDA_ARCH_LIST", "ENGINE_REF"}
    }
    env.update(
        {
            "PATH": f"{stub}:{os.environ['PATH']}",
            "TEST_LOG": str(log),
            "PARETON_REPO_DIR": str(ROOT),
            "PARETON_WORK_DIR": str(work),
            "PARETON_BUILD_MAX_JOBS": "3",
            "BASE": "test/base@sha256:abc123",
            "TORCH_CUDA_ARCH_LIST": "9.0",
        }
    )
    result = subprocess.run(
        ["bash", "ops/a2b-build.sh", *(["--detach"] if detach else [])],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == build_status, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    wrapper = calls[0]["args"]
    assert wrapper[:2] == ["compose", "run"]
    assert ("--detach" in wrapper) is detach
    assert "PARETON_GHCR_TOKEN" not in wrapper  # Comes from Compose's env_file.
    assert calls[1]["args"] == ["-m", "builder.preflight"]
    build = next(call for call in calls if call["args"][:2] == ["-m", "builder"])
    assert build["jobs"] == "3"
    assert build["args"][build["args"].index("--work-root") + 1].startswith(str(work))
    assert build["args"][build["args"].index("--base-image") + 1] == env["BASE"]
    assert (calls[-1]["args"][0] == "inspect") is (build_status == 0)
