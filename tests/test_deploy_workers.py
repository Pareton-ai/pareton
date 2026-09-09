"""Exercise deploy restart decisions with fake services and an offline database."""

import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def deploy(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("")
    (repo / ".deploy-done").write_text("old\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tool = textwrap.dedent(
        f"""\
        #!{sys.executable}
        import json
        from pathlib import Path
        import sys
        name = Path(sys.argv[0]).name
        if name == 'flock':
            sys.exit(0)
        state = json.loads(Path('state.json').read_text())
        if name == 'git':
            if sys.argv[1] == 'rev-parse':
                print('new')
            if sys.argv[1] == 'pull' and state.get('pull_error'):
                sys.exit(1)
        elif name == 'systemctl':
            if sys.argv[1] == 'cat':
                sys.exit(1 if sys.argv[2] in state.get('missing_units', []) else 0)
            if sys.argv[1] == 'restart':
                with Path('restarts.log').open('a') as log:
                    log.write(sys.argv[2] + '\\n')
        """
    )
    for name in ("git", "systemctl", "flock"):
        path = bin_dir / name
        path.write_text(fake_tool)
        path.chmod(0o755)
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    db_dir = repo / "db"
    db_dir.mkdir()
    (db_dir / "__init__.py").write_text("")
    (db_dir / "connection.py").write_text(
        textwrap.dedent(
            """\
            import json
            from pathlib import Path

            class Connection:
                def __enter__(self):
                    self.state = json.loads(Path('state.json').read_text())
                    if self.state.get('db_error'):
                        raise RuntimeError('offline database')
                    return self

                def __exit__(self, *args):
                    pass

                def cursor(self):
                    return self

                def execute(self, sql):
                    table = sql.split('FROM ')[1].split()[0]
                    self.busy = self.state.get(table, False)

                def fetchone(self):
                    return (1,) if self.busy else None

            db_connection = Connection
            """
        )
    )
    source = Path(__file__).resolve().parents[1] / "ops/deploy.sh"
    script = tmp_path / "deploy.sh"
    script.write_text(
        source.read_text()
        .replace("REPO=/opt/pareton", f"REPO={shlex.quote(str(repo))}")
        .replace("LOCK=/run/pareton-deploy.lock", f"LOCK={tmp_path}/deploy.lock")
    )

    def run(**state):
        (repo / "state.json").write_text(json.dumps(state))
        (repo / "restarts.log").write_text("")
        result = subprocess.run(
            ["bash", str(script)],
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "PYTHONPATH": str(repo),
                "PARETON_DATABASE_URL": "",
                "PARETON_TEST_DATABASE_URL": "",
            },
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return result, (repo / "restarts.log").read_text().splitlines()

    return repo, run


def test_blocked_build_does_not_defer_round_worker_restart(deploy):
    repo, run = deploy
    result, restarts = run(submission_jobs=True)
    assert result.returncode == 0, result.stderr
    assert "pareton-round-worker" in restarts
    assert "pareton-worker" not in restarts
    assert (repo / ".deploy-pending").exists()
    assert not (repo / ".deploy-rounds-pending").exists()


def test_running_round_protects_round_and_legacy_combined_workers(deploy):
    repo, run = deploy
    result, restarts = run(rounds=True)
    assert result.returncode == 0, result.stderr
    assert "pareton-round-worker" not in restarts
    assert "pareton-worker" not in restarts
    assert (repo / ".deploy-pending").exists()
    assert (repo / ".deploy-rounds-pending").exists()

    # A later idle tick owes both restarts even though main has not changed.
    result, restarts = run()
    assert result.returncode == 0, result.stderr
    assert restarts == ["pareton-round-worker", "pareton-worker"]
    assert not (repo / ".deploy-pending").exists()
    assert not (repo / ".deploy-rounds-pending").exists()


def test_database_failure_defers_both_restarts(deploy):
    repo, run = deploy
    result, restarts = run(db_error=True)
    assert result.returncode == 0, result.stderr
    assert "pareton-worker" not in restarts
    assert "pareton-round-worker" not in restarts
    assert result.stdout.count("probe failed") == 2
    assert (repo / ".deploy-pending").exists()
    assert (repo / ".deploy-rounds-pending").exists()


def test_missing_round_unit_does_not_abort_deploy(deploy):
    repo, run = deploy
    result, restarts = run(missing_units=["pareton-round-worker"])
    assert result.returncode == 0, result.stderr
    assert "pareton-worker" in restarts
    assert "pareton-round-worker" not in restarts
    assert (repo / ".deploy-rounds-pending").exists()


def test_failed_deploy_never_restarts_either_worker(deploy):
    repo, run = deploy
    result, restarts = run(pull_error=True)
    assert result.returncode != 0
    assert restarts == []
    assert (repo / ".deploy-pending").exists()
    assert (repo / ".deploy-rounds-pending").exists()
