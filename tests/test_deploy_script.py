"""End-to-end control-flow tests for the pull deploy shell script."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = REPO_ROOT / "ops" / "deploy.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@dataclass
class DeployHarness:
    repo: Path
    script: Path
    log: Path
    head: Path
    worker_state: Path
    env: dict[str, str]

    def run(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = {**self.env, **overrides}
        return subprocess.run(
            ["bash", str(self.script)],
            cwd=self.repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def deploy_harness(tmp_path: Path) -> DeployHarness:
    repo = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    venv_bin = repo / ".venv" / "bin"
    repo.mkdir()
    fake_bin.mkdir()
    venv_bin.mkdir(parents=True)

    activity_lock = tmp_path / "worker-activity.lock"
    deploy_lock = tmp_path / "deploy.lock"
    log = tmp_path / "calls.log"
    head = tmp_path / "head"
    worker_state = tmp_path / "worker-state"
    head.write_text("old\n", encoding="utf-8")
    worker_state.write_text("running\n", encoding="utf-8")
    (repo / ".deploy-done").write_text("old\n", encoding="utf-8")
    (repo / ".env").write_text(
        f"PARETON_WORKER_ACTIVITY_LOCK_PATH={activity_lock}\n", encoding="utf-8"
    )

    script_text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    script_text = script_text.replace(
        "REPO=/opt/pareton", f"REPO={shlex.quote(str(repo))}"
    ).replace("LOCK=/run/pareton-deploy.lock", f"LOCK={shlex.quote(str(deploy_lock))}")
    script = tmp_path / "deploy.sh"
    _write_executable(script, script_text)

    _write_executable(
        fake_bin / "flock",
        """#!/usr/bin/env bash
printf 'flock %s\n' "$*" >> "$CALL_LOG"
if [ "${ACTIVITY_LOCK_BUSY:-0}" = 1 ] && [ "${2:-}" = 8 ]; then
    exit 1
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
printf 'git %s\n' "$*" >> "$CALL_LOG"
case "${1:-}" in
    fetch)
        exit 0
        ;;
    rev-parse)
        if [ "${2:-}" = origin/main ]; then
            printf '%s\n' "$REMOTE_COMMIT"
        else
            /bin/cat "$HEAD_FILE"
        fi
        ;;
    pull)
        if [ "${FAIL_STEP:-}" = pull ]; then
            exit 20
        fi
        printf '%s\n' "$REMOTE_COMMIT" > "$HEAD_FILE"
        ;;
    diff)
        if [ "${REQUIREMENTS_CHANGED:-0}" = 1 ]; then
            printf 'requirements.txt\n'
        fi
        ;;
    *)
        exit 2
        ;;
esac
""",
    )
    _write_executable(
        fake_bin / "systemctl",
        """#!/usr/bin/env bash
printf 'systemctl %s\n' "$*" >> "$CALL_LOG"
case "${1:-}:${2:-}" in
    stop:pareton-worker)
        if [ "${FAIL_STEP:-}" = stop ]; then
            exit 21
        fi
        printf 'stopped\n' > "$WORKER_STATE_FILE"
        ;;
    start:pareton-worker)
        if [ "${FAIL_STEP:-}" = start ]; then
            exit 22
        fi
        printf 'running\n' > "$WORKER_STATE_FILE"
        ;;
    restart:pareton-api)
        if [ "${FAIL_STEP:-}" = api ]; then
            exit 23
        fi
        ;;
    cat:pareton-watcher|cat:pareton-weights)
        exit 1
        ;;
    *)
        exit 2
        ;;
esac
""",
    )
    _write_executable(
        venv_bin / "python",
        """#!/usr/bin/env bash
printf 'python %s\n' "$*" >> "$CALL_LOG"
exit "${PROBE_RC:-10}"
""",
    )
    _write_executable(
        venv_bin / "pip",
        """#!/usr/bin/env bash
printf 'pip %s\n' "$*" >> "$CALL_LOG"
if [ "${FAIL_STEP:-}" = pip ]; then
    exit 24
fi
""",
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALL_LOG": str(log),
        "HEAD_FILE": str(head),
        "WORKER_STATE_FILE": str(worker_state),
        "REMOTE_COMMIT": "new",
        "PROBE_RC": "10",
    }
    return DeployHarness(repo, script, log, head, worker_state, env)


def _index(calls: list[str], prefix: str) -> int:
    return next(i for i, call in enumerate(calls) if call.startswith(prefix))


def test_success_stops_worker_before_mutation_and_starts_it_last(
    deploy_harness: DeployHarness,
):
    result = deploy_harness.run(REQUIREMENTS_CHANGED="1")

    assert result.returncode == 0, result.stderr
    calls = deploy_harness.calls()
    assert _index(calls, "systemctl stop pareton-worker") < _index(calls, "git pull")
    assert _index(calls, "git pull") < _index(calls, "pip install")
    assert _index(calls, "systemctl restart pareton-api") < _index(
        calls, "systemctl start pareton-worker"
    )
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "running\n"
    assert deploy_harness.head.read_text(encoding="utf-8") == "new\n"
    assert (deploy_harness.repo / ".deploy-done").read_text() == "new\n"
    assert not (deploy_harness.repo / ".deploy-pending").exists()


def test_failure_after_pull_leaves_worker_stopped_and_retryable(
    deploy_harness: DeployHarness,
):
    failed = deploy_harness.run(FAIL_STEP="api")

    assert failed.returncode != 0
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "stopped\n"
    assert (deploy_harness.repo / ".deploy-pending").exists()
    assert (deploy_harness.repo / ".deploy-done").read_text() == "old\n"
    assert not any(
        call.startswith("systemctl start pareton-worker")
        for call in deploy_harness.calls()
    )

    # The checkout or environment may be broken after the failed attempt.
    # A retry must trust the pending marker instead of importing its probe.
    retried = deploy_harness.run(PROBE_RC="2")

    assert retried.returncode == 0, retried.stderr
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "running\n"
    assert (deploy_harness.repo / ".deploy-done").read_text() == "new\n"
    assert not (deploy_harness.repo / ".deploy-pending").exists()
    assert sum(call.startswith("python ") for call in deploy_harness.calls()) == 1


def test_pull_failure_records_pending_before_checkout_mutation(
    deploy_harness: DeployHarness,
):
    failed = deploy_harness.run(FAIL_STEP="pull")

    assert failed.returncode != 0
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "stopped\n"
    assert (deploy_harness.repo / ".deploy-pending").exists()
    assert (deploy_harness.repo / ".deploy-done").read_text() == "old\n"
    assert deploy_harness.head.read_text(encoding="utf-8") == "old\n"


def test_running_database_work_defers_before_worker_stop(
    deploy_harness: DeployHarness,
):
    result = deploy_harness.run(PROBE_RC="0")

    assert result.returncode == 0
    calls = deploy_harness.calls()
    assert not any(call.startswith("systemctl stop") for call in calls)
    assert not any(call.startswith("git pull") for call in calls)
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "running\n"
    assert deploy_harness.head.read_text(encoding="utf-8") == "old\n"


def test_probe_import_failure_exit_one_fails_closed(
    deploy_harness: DeployHarness,
):
    result = deploy_harness.run(PROBE_RC="1")

    assert result.returncode == 0
    assert "worker probe failed (rc=1); update deferred" in result.stdout
    calls = deploy_harness.calls()
    assert not any(call.startswith("systemctl stop") for call in calls)
    assert not any(call.startswith("git pull") for call in calls)
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "running\n"


def test_worker_stop_failure_prevents_first_mutation(
    deploy_harness: DeployHarness,
):
    result = deploy_harness.run(FAIL_STEP="stop")

    assert result.returncode != 0
    calls = deploy_harness.calls()
    assert not any(call.startswith("git pull") for call in calls)
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "running\n"
    assert deploy_harness.head.read_text(encoding="utf-8") == "old\n"
    assert not (deploy_harness.repo / ".deploy-pending").exists()


def test_worker_start_failure_keeps_pending_marker_for_retry(
    deploy_harness: DeployHarness,
):
    failed = deploy_harness.run(FAIL_STEP="start")

    assert failed.returncode != 0
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "stopped\n"
    assert (deploy_harness.repo / ".deploy-pending").exists()
    assert (deploy_harness.repo / ".deploy-done").read_text() == "new\n"

    retried = deploy_harness.run(PROBE_RC="2")

    assert retried.returncode == 0, retried.stderr
    assert deploy_harness.worker_state.read_text(encoding="utf-8") == "running\n"
    assert not (deploy_harness.repo / ".deploy-pending").exists()


def test_worker_activity_lock_defers_before_database_probe(
    deploy_harness: DeployHarness,
):
    result = deploy_harness.run(ACTIVITY_LOCK_BUSY="1")

    assert result.returncode == 0
    calls = deploy_harness.calls()
    assert not any(call.startswith("python ") for call in calls)
    assert not any(call.startswith("systemctl stop") for call in calls)
    assert not any(call.startswith("git pull") for call in calls)
