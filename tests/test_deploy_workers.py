"""Run the deploy script offline, including its actual SQL busy probes.

Covers the #149 worker deferral semantics plus the stage-1 additions: the
config-sync hook gates the deploy, a failing busy probe fails the tick
(spec 6.4), steps are recorded for the notifier, and install-ops
self-installs the ops helpers atomically (spec 6.1).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def deploy(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("")
    (repo / ".deploy-done").write_text("old")
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    db = repo / "db"
    db.mkdir()
    (db / "__init__.py").write_text("")
    (db / "connection.py").write_text("""
import os
import sqlite3
from contextlib import contextmanager

@contextmanager
def db_connection():
    if os.environ['TEST_DB_ERROR'] == '1':
        raise RuntimeError('database unavailable')
    conn = sqlite3.connect(':memory:')
    for table, setting in [('submission_jobs', 'TEST_JOB'), ('rounds', 'TEST_ROUND')]:
        conn.execute(f'CREATE TABLE {table} (status TEXT)')
        if os.environ[setting] == '1':
            conn.execute(f"INSERT INTO {table} VALUES ('running')")
    class Cursor:
        def execute(self, sql, args):
            self.result = conn.execute(sql.replace('%s', '?'), args)
        def fetchone(self):
            return self.result.fetchone()
    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()
    try:
        yield Connection()
    finally:
        conn.close()
""")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ops_dir = tmp_path / "ops-lib"
    ops_dir.mkdir()
    state_dir = tmp_path / "state"
    ops_log = tmp_path / "ops.log"
    for name, body in {
        "flock": "exit 0",
        "git": 'if [ "$1" = rev-parse ]; then echo new; fi',
        "systemctl": """if [ "$1" = cat ]; then
    [ "$2" != "$TEST_MISSING_UNIT" ]
else
    echo "$2" >> restarts.log
fi""",
        "sync-config.py": 'echo "sync $@" >> "$OPS_LOG"\nexit ${FAKE_SYNC_RC:-0}\n',
        "notify-deploy-failure.py": 'echo "notify $@" >> "$OPS_LOG"\nexit 0\n',
    }.items():
        tool = bin_dir if name in ("flock", "git", "systemctl") else ops_dir
        path = tool / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)

    def run(
        job=False,
        round=False,
        error=False,
        missing="",
        sync_rc=0,
        invocation="inv-test",
    ):
        log = repo / "restarts.log"
        log.write_text("")
        ops_log.write_text("")
        result = subprocess.run(
            ["bash", str(tmp_path / "deploy.sh")],
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "PYTHONPATH": str(repo),
                "PARETON_DATABASE_URL": "",
                "PARETON_TEST_DATABASE_URL": "",
                "PARETON_DEPLOY_REPO": str(repo),
                "PARETON_OPS_DIR": str(ops_dir),
                "PARETON_STATE_DIR": str(state_dir),
                "PARETON_DEPLOY_BIN": str(tmp_path / "bin/pareton-deploy"),
                "PARETON_DEPLOY_LOCK": str(tmp_path / "lock"),
                "OPS_LOG": str(ops_log),
                "FAKE_SYNC_RC": str(sync_rc),
                "INVOCATION_ID": invocation,
                "TEST_JOB": str(int(job)),
                "TEST_ROUND": str(int(round)),
                "TEST_DB_ERROR": str(int(error)),
                "TEST_MISSING_UNIT": missing,
            },
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        restarted = [unit for unit in log.read_text().splitlines() if "worker" in unit]
        ops_calls = ops_log.read_text().splitlines()
        run_state = {}
        run_env = state_dir / "last-run.env"
        if run_env.exists():
            run_state = dict(
                line.split("=", 1)
                for line in run_env.read_text().splitlines()
                if "=" in line
            )
        return {
            "rc": result.returncode,
            "stderr": result.stderr,
            "workers": restarted,
            "ops": ops_calls,
            "state": run_state,
        }

    source = (Path(__file__).resolve().parents[1] / "ops/deploy.sh").read_text()
    (tmp_path / "deploy.sh").write_text(source)
    return repo, ops_dir, tmp_path, run


@pytest.mark.parametrize(
    "job,round,error,expected",
    [
        (True, False, False, ["pareton-round-worker"]),
        (False, True, False, []),  # Also protects a legacy combined worker's round.
        (False, False, False, ["pareton-round-worker", "pareton-worker"]),
    ],
)
def test_workers_restart_independently_and_retry_when_idle(
    deploy, job, round, error, expected
):
    repo, _, _, run = deploy
    result = run(job=job, round=round, error=error)
    assert result["rc"] == 0, result["stderr"]
    assert result["workers"] == expected
    assert any(c.startswith("notify record-success") for c in result["ops"])
    assert result["state"]["last_step"] == "done"
    for unit, flag in [
        ("pareton-worker", ".deploy-pending"),
        ("pareton-round-worker", ".deploy-rounds-pending"),
    ]:
        assert (repo / flag).exists() == (unit not in expected)
    # No new commit: only the still-owed restarts should happen on the next tick.
    assert run()["workers"] == [
        unit
        for unit in ["pareton-round-worker", "pareton-worker"]
        if unit not in expected
    ]


def test_probe_error_fails_the_deploy_and_keeps_pending(deploy):
    repo, _, _, run = deploy
    result = run(error=True)
    assert result["rc"] == 1  # Spec 6.4: broken probe alerts instead of skipping.
    assert result["workers"] == []
    assert (repo / ".deploy-pending").exists()
    assert (repo / ".deploy-rounds-pending").exists()
    assert result["state"]["last_step"] == "probe-worker"
    assert not any(c.startswith("notify record-success") for c in result["ops"])
    # The code deploy itself finished; only the alert and pending remain.
    assert (repo / ".deploy-done").read_text().strip() == "new"


def test_sync_failure_fails_the_deploy_before_restarts(deploy):
    repo, _, _, run = deploy
    result = run(sync_rc=3)
    assert result["rc"] != 0
    assert "sync deploy-hook" in " ".join(result["ops"])
    assert (repo / ".deploy-done").read_text().strip() == "old"  # Not marked deployed.
    assert "pareton-api" not in Path(repo / "restarts.log").read_text()
    assert result["state"]["last_step"] == "install-config"


def test_no_change_tick_still_runs_config_check(deploy):
    _repo, _, _, run = deploy
    first = run()
    assert first["rc"] == 0
    assert any(c.startswith("sync deploy-hook") for c in first["ops"])
    second = run()
    assert second["rc"] == 0
    # Every locked tick re-checks config; HEAD did not change (spec 5.3).
    assert any(c.startswith("sync deploy-hook") for c in second["ops"])


def test_install_ops_self_installs_helpers_atomically(deploy):
    repo, ops_dir, tmp_path, run = deploy
    ops = repo / "ops"
    (ops / "systemd").mkdir(parents=True)
    (ops / "systemd/pareton-api.service").write_text("[Unit]\n")
    for name in ("ops_common.py", "sync-config.py", "notify-deploy-failure.py"):
        (ops / name).write_text(f"# new {name}\n")
    (ops / "deploy.sh").write_text("#!/usr/bin/env bash\necho self-installed\n")
    result = run()
    assert result["rc"] == 0
    for name in ("ops_common.py", "sync-config.py", "notify-deploy-failure.py"):
        installed = ops_dir / name
        assert installed.exists()
        assert installed.read_text() == f"# new {name}\n"
    deployed_bin = tmp_path / "bin/pareton-deploy"
    assert deployed_bin.read_text().startswith("#!/usr/bin/env bash")
    # Main entry replaced last; no temp leftovers anywhere (spec 6.1).
    leftovers = [f.name for f in ops_dir.iterdir() if ".new." in f.name]
    assert leftovers == []
    leftovers_bin = [f.name for f in (tmp_path / "bin").iterdir() if ".new." in f.name]
    assert leftovers_bin == []


def test_run_state_records_invocation_and_commits(deploy):
    _repo, _, _, run = deploy
    result = run(invocation="inv-abc")
    state = result["state"]
    assert state["invocation_id"] == "inv-abc"
    assert state["target_commit"] == "new"
    assert state["last_step"] == "done"
    assert "started_at" in state
