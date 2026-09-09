"""Run the deploy script offline, including its actual SQL busy probes."""

import os
import shlex
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
    for name, body in {
        "flock": "exit 0",
        "git": 'if [ "$1" = rev-parse ]; then echo new; fi',
        "systemctl": """if [ "$1" = cat ]; then
    [ "$2" != "$TEST_MISSING_UNIT" ]
else
    echo "$2" >> restarts.log
fi""",
    }.items():
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\n" + body + "\n")
        tool.chmod(0o755)
    script = tmp_path / "deploy.sh"
    source = (Path(__file__).resolve().parents[1] / "ops/deploy.sh").read_text()
    script.write_text(
        source.replace("REPO=/opt/pareton", f"REPO={shlex.quote(str(repo))}").replace(
            "LOCK=/run/pareton-deploy.lock",
            f"LOCK={shlex.quote(str(tmp_path / 'lock'))}",
        )
    )

    def run(job=False, round=False, error=False, missing=""):
        log = repo / "restarts.log"
        log.write_text("")
        result = subprocess.run(
            ["bash", str(script)],
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "PYTHONPATH": str(repo),
                "PARETON_DATABASE_URL": "",
                "PARETON_TEST_DATABASE_URL": "",
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
        assert result.returncode == 0, result.stderr
        return [unit for unit in log.read_text().splitlines() if "worker" in unit]

    return repo, run


@pytest.mark.parametrize(
    "job,round,error,expected",
    [
        (True, False, False, ["pareton-round-worker"]),
        (False, True, False, []),  # Also protects a legacy combined worker's round.
        (False, False, True, []),
        (False, False, False, ["pareton-round-worker", "pareton-worker"]),
    ],
)
def test_workers_restart_independently_and_retry_when_idle(
    deploy, job, round, error, expected
):
    repo, run = deploy
    assert run(job=job, round=round, error=error) == expected
    for unit, flag in [
        ("pareton-worker", ".deploy-pending"),
        ("pareton-round-worker", ".deploy-rounds-pending"),
    ]:
        assert (repo / flag).exists() == (unit not in expected)
    # No new commit: only the still-owed restarts should happen on the next tick.
    assert run() == [
        unit
        for unit in ["pareton-round-worker", "pareton-worker"]
        if unit not in expected
    ]


def test_round_restart_remains_pending_until_unit_installed(deploy):
    repo, run = deploy
    assert run(missing="pareton-round-worker") == ["pareton-worker"]
    assert (repo / ".deploy-rounds-pending").exists()
    assert run() == ["pareton-round-worker"]
