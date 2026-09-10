"""Exercise fetch, prepare, drain, down/up, rollback and retry with fake Docker."""

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
    assert "image tag pareton-runtime:new pareton-runtime:local" in log
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
