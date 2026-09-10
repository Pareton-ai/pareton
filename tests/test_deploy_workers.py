"""Exercise deployment lifecycle, rollback, and saved Compose configuration."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def deploy(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("")
    (repo / "compose.yaml").write_text("services: {}\n")
    for name in ("ops/vector/vector.toml", "ops/caddy/Caddyfile"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("config")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "git").write_text("""#!/bin/sh
printf 'git %s\n' "$*" >> "$TEST_LOG"
case "$1" in
    rev-parse) echo new ;;
    merge) exit "${TEST_MERGE_RC:-0}" ;;
esac
""")
    (stub / "flock").write_text("""#!/bin/sh
[ "$1" = -n ] && exit 0
shift
exec "$@"
""")
    (stub / "docker").write_text("""#!/bin/sh
printf 'docker %s\n' "$*" >> "$TEST_LOG"
case " $* " in
    *" config "*) echo 'services: {}' ;;
    *" build "*) exit "${TEST_BUILD_RC:-0}" ;;
    *" validate "*) exit "${TEST_VALIDATE_RC:-0}" ;;
    *" builder.preflight "*) exit "${TEST_PREFLIGHT_RC:-0}" ;;
    *" up "*)
        case " $* " in
            *"next.yaml"*) exit "${TEST_UP_RC:-0}" ;;
        esac ;;
esac
""")
    for p in stub.iterdir():
        p.chmod(0o755)
    log = tmp_path / "commands.log"

    def run(*args, **settings):
        log.write_text("")
        result = subprocess.run(
            ["bash", str(ROOT / "ops/deploy.sh"), *args],
            env={
                **os.environ,
                "PATH": f"{stub}:{os.environ['PATH']}",
                "PARETON_REPO_DIR": str(repo),
                "PARETON_ENV_FILE": str(repo / ".env"),
                "TEST_LOG": str(log),
                **{k: str(v) for k, v in settings.items()},
            },
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return result, log.read_text()

    return repo, run


def test_prepare_before_drain_and_down_then_up(deploy):
    repo, run = deploy
    result, log = run()
    assert result.returncode == 0, result.stderr
    assert log.index(" build ") < log.index("stop watcher")
    assert log.index("stop watcher") < log.index("stop worker round-worker weights")
    assert (
        log.index("stop worker round-worker weights")
        < log.index(" down ")
        < log.index(" up ")
    )
    assert "--builder default" in log
    assert "--wait" in log
    assert "image tag pareton-runtime:new." in log
    assert " pareton-runtime:local" in log
    assert "--volumes" not in log and " down -v" not in log
    assert (repo / ".deploy-state/done").read_text().strip() == "new"
    assert (repo / ".deploy-state/current.yaml").stat().st_mode & 0o777 == 0o600
    result, log = run()
    assert result.returncode == 0
    assert "docker" not in log  # An unchanged main does not recreate services.


@pytest.mark.parametrize(
    "failure",
    ["TEST_MERGE_RC", "TEST_BUILD_RC", "TEST_VALIDATE_RC", "TEST_PREFLIGHT_RC"],
)
def test_preparation_failure_leaves_running_services_alone(deploy, failure):
    repo, run = deploy
    result, log = run(**{failure: 1})
    assert result.returncode != 0
    assert " stop " not in log and " down " not in log
    assert not (repo / ".deploy-state/done").exists()
    result, log = run()
    assert result.returncode == 0
    assert " up " in log  # Retry even though fetch already advanced HEAD.


def test_failed_start_restores_previous_release_and_does_not_mark_done(deploy):
    repo, run = deploy
    state = repo / ".deploy-state"
    state.mkdir()
    (state / "current.yaml").write_text("old config")
    (state / "done").write_text("old")
    result, log = run(TEST_UP_RC=1)
    assert result.returncode != 0
    assert "image tag" not in log
    assert "current.yaml stop worker" in log
    assert "current.yaml up -d --no-build" in log
    assert (state / "done").read_text() == "old"
    assert (state / "current.yaml").read_text() == "old config"
    result, log = run()
    assert result.returncode == 0
    assert (state / "done").read_text().strip() == "new"


def test_local_deploy_does_not_fetch_or_merge(deploy):
    _, run = deploy
    result, log = run("--local")
    assert result.returncode == 0
    assert "git fetch" not in log and "git merge" not in log
    assert " down " in log and " up " in log


@pytest.mark.docker
@pytest.mark.parametrize("args", [(), ("--local",)])
def test_saved_release_retains_cli_without_enabling_it_at_startup(deploy, args):
    """Use real Compose parsing; stubs cannot reproduce inactive-profile filtering."""
    import json
    import shutil

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is unavailable")
    if subprocess.run(
        [docker, "compose", "version"], capture_output=True, timeout=15
    ).returncode:
        pytest.skip("Docker Compose is unavailable")

    repo, run = deploy
    (repo / "compose.yaml").write_text((ROOT / "compose.yaml").read_text())
    stub = repo.parent / "bin/docker"
    stub.write_text(
        stub.read_text()
        .replace(
            "*\" config \"*) echo 'services: {}' ;;",
            '*" config "*) exec "$TEST_REAL_DOCKER" "$@" ;;',
        )
        .replace(
            '*" builder.preflight "*) exit "${TEST_PREFLIGHT_RC:-0}" ;;',
            '*" builder.preflight "*)\n'
            '        exec "$TEST_REAL_DOCKER" compose '
            '--project-directory "$PARETON_REPO_DIR" '
            '-f "$PARETON_REPO_DIR/.deploy-state/next.yaml" '
            "config --quiet cli ;;",
        )
    )
    result, _log = run(
        *args,
        TEST_REAL_DOCKER=docker,
        PARETON_AXIOM_TOKEN="test",
        COMPOSE_PROFILES="",
    )
    assert result.returncode == 0, result.stderr

    command = [
        docker,
        "compose",
        "--project-directory",
        str(repo),
        "--env-file",
        str(repo / ".env"),
        "-f",
        str(repo / ".deploy-state/current.yaml"),
    ]
    env = {**os.environ, "COMPOSE_PROFILES": ""}
    saved = json.loads(
        subprocess.check_output(
            [*command, "--profile", "*", "config", "--format", "json"],
            env=env,
            text=True,
            timeout=15,
        )
    )
    assert saved["services"]["cli"]["profiles"] == ["tools"]
    active = subprocess.check_output(
        [*command, "config", "--services"], env=env, text=True, timeout=15
    ).splitlines()
    assert "cli" not in active
    assert set(active) == set(saved["services"]) - {"cli"}


def test_failed_same_commit_redeploy_preserves_previous_release(deploy):
    repo, run = deploy
    result, first_log = run("--local")
    assert result.returncode == 0
    releases = repo / ".deploy-state/releases"
    first_release = next(releases.iterdir())
    original_vector = (first_release / "vector.toml").read_text()
    first_tag = next(
        line for line in first_log.splitlines() if "image tag" in line
    ).split()[-2]
    (repo / "ops/vector/vector.toml").write_text("changed vector configuration")
    result, second_log = run("--local", TEST_UP_RC=1)
    assert result.returncode != 0
    assert (first_release / "vector.toml").read_text() == original_vector
    assert len(list(releases.iterdir())) == 2
    assert first_tag not in second_log  # Never build over the previous image tag.


def test_stopping_watch_drains_active_deployment(deploy):
    import signal
    import time

    repo, _run = deploy
    stub = repo.parent / "bin"
    docker = stub / "docker"
    text = docker.read_text().replace(
        '*" build "*) exit "${TEST_BUILD_RC:-0}" ;;',
        '*" build "*)\n'
        '        touch "$TEST_BUILD_STARTED"\n'
        '        while [ ! -f "$TEST_BUILD_RELEASE" ]; do sleep 0.05; done\n'
        "        exit 0 ;;",
    )
    docker.write_text(text)
    started, release = repo.parent / "started", repo.parent / "release"
    log = repo.parent / "watch.log"
    proc = subprocess.Popen(
        ["bash", str(ROOT / "ops/deploy.sh"), "--watch"],
        env={
            **os.environ,
            "PATH": f"{stub}:{os.environ['PATH']}",
            "PARETON_REPO_DIR": str(repo),
            "PARETON_ENV_FILE": str(repo / ".env"),
            "TEST_LOG": str(log),
            "TEST_BUILD_STARTED": str(started),
            "TEST_BUILD_RELEASE": str(release),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists()
        proc.terminate()
        time.sleep(0.2)
        assert proc.poll() is None, (
            "controller exited before its active deployment drained"
        )
        release.touch()
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 0, (stdout, stderr)
        assert (repo / ".deploy-state/done").read_text().strip() == "new"
        assert log.read_text().count("git fetch") == 1
    finally:
        release.touch()
        # Clean up the entire isolated test process group, including any orphans.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate(timeout=5)


@pytest.mark.docker
def test_controller_drains_when_docker_stops_it():
    """Docker init must forward SIGTERM to the watch shell, which drains first."""
    import shutil
    import time
    import uuid

    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode:
        pytest.skip("Docker daemon is unavailable")
    image = os.environ.get("PARETON_DEPLOYER_IMAGE", "pareton-deployer:local")
    name = "pareton-deploy-test-" + uuid.uuid4().hex[:10]
    setup = r"""
set -eu
mkdir -p /testrepo/ops/vector /testrepo/ops/caddy /fake
touch /testrepo/.env /testrepo/compose.yaml /testrepo/ops/vector/vector.toml /testrepo/ops/caddy/Caddyfile
cat > /fake/git <<'GIT'
#!/bin/sh
case "$1" in rev-parse) echo deadbeef ;; esac
GIT
cat > /fake/docker <<'DOCKER'
#!/bin/sh
case " $* " in
    *" config "*) echo 'services: {}' ;;
    *" build "*)
        touch /started
        while [ ! -f /release ]; do sleep 0.05; done ;;
esac
DOCKER
chmod +x /fake/*
export PATH="/fake:$PATH"
export PARETON_REPO_DIR=/testrepo
exec /usr/local/bin/pareton-deploy --watch
"""
    stopping = None
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--init",
                "--name",
                name,
                "--entrypoint",
                "bash",
                image,
                "-c",
                setup,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        deadline = time.monotonic() + 10
        while subprocess.run(
            ["docker", "exec", name, "test", "-f", "/started"],
            capture_output=True,
            timeout=5,
        ).returncode:
            assert time.monotonic() < deadline, "controller never reached the build"
            time.sleep(0.1)
        stopping = subprocess.Popen(
            ["docker", "stop", "--time", "10", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.3)
        assert stopping.poll() is None, "controller stopped before the rollout drained"
        subprocess.run(
            ["docker", "exec", name, "touch", "/release"], check=True, timeout=5
        )
        stdout, stderr = stopping.communicate(timeout=15)
        assert stopping.returncode == 0, (stdout, stderr)
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.ExitCode}}", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.stdout.strip() == "0"
        logs = subprocess.run(
            ["docker", "logs", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "deploy: deadbeef running" in logs.stdout
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            check=False,
            timeout=15,
        )
        if stopping is not None:
            stopping.communicate(timeout=15)
