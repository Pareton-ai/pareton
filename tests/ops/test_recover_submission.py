"""Tests for ops/recover_submission.py decision rules (spec 5.1, B22).

These unit tests pin the evidence rules that gate the database-backed
paths and the CLI entry itself. The marked integration test exercises the
transactional recover path against the isolated PostgreSQL test database.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest
from e2e_db import require_e2e_database_url

REPO_ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "recover_submission", REPO_ROOT / "ops" / "recover_submission.py"
)


def test_module_loads_with_venv_imports():
    # db.connection imports psycopg2 lazily; module import alone must work.
    assert spec is not None
    assert load() is not None


def load():
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "states,outcome,expected",
    [
        # No durable evidence: requeue allowed, settle refused.
        ([], "requeue", "pending"),
        ([], "settle", None),
        # Terminal reject: settle to failed, requeue refused.
        (["rejected"], "settle", "failed"),
        (["rejected_duplicate"], "settle", "failed"),
        (["disqualified"], "settle", "failed"),
        (["rejected"], "requeue", None),
        # Progressed past gates: settle to done, requeue refused.
        (["bench_queued"], "settle", "done"),
        (["scored"], "settle", "done"),
        (["bench_queued"], "requeue", None),
        # Contradictory evidence: everything refused.
        (["rejected", "bench_queued"], "settle", None),
        (["rejected", "bench_queued"], "requeue", None),
    ],
)
def test_decide_rules(states, outcome, expected):
    module = load()
    klass = module.evidence_class(states)
    assert module._decide(outcome, klass, states) == expected


def test_evidence_class_ordering():
    module = load()
    assert module.evidence_class([]) == "none"
    assert module.evidence_class(["ingested", "gates_passed"]) == "none"
    assert module.evidence_class(["built"]) == "none"
    assert module.evidence_class(["bench_queued"]) == "progressed"
    assert module.evidence_class(["rejected"]) == "terminal-reject"
    assert module.evidence_class(["scored", "rejected"]) == "contradictory"


@pytest.mark.parametrize(
    "campaign_bench,expected",
    [(None, "done"), ({"baseline_engine_image_digest": "sha256:x"}, None)],
)
def test_built_settles_only_when_campaign_has_no_bench(campaign_bench, expected):
    module = load()
    states = ["built"]
    assert (
        module._decide(
            "settle",
            module.evidence_class(states),
            states,
            campaign_bench=campaign_bench,
        )
        == expected
    )


def test_no_submission_event_writes():
    # The recovery audit must never touch submission_events: its state
    # vocabulary is miner-visible (latest_state / by_latest_state), and
    # 'recovered' is not a SubmissionState member. Source-level guard for
    # the fix that once failed to land while its commit claimed otherwise.
    source = (REPO_ROOT / "ops" / "recover_submission.py").read_text()
    assert "INSERT INTO submission_events" not in source
    assert "submission_recovered" in source  # audit goes to stdout/journald


def test_bootstrap_anchors_target_after_shared_users_are_stopped():
    runbook = (REPO_ROOT / "ops" / "runbook.md").read_text()
    bootstrap = runbook.split("## S4b.", 1)[1].split("## S5.", 1)[0]
    assert "bootstrap-timers.env" in bootstrap
    assert "checkout.sha" in bootstrap
    assert (
        "systemctl disable --now pareton-deploy.timer pareton-gpu-reap.timer pareton-builder-cleanup.timer"
        in bootstrap
    )
    assert "restore_bootstrap_timers()" in bootstrap
    assert "TARGET_SHA=<approved stage-2 commit>" in bootstrap
    assert "git -C /opt/pareton fetch origin main" in bootstrap
    assert 'git -C /opt/pareton checkout --detach "$TARGET_SHA"' in bootstrap
    assert 'test "$(git -C /opt/pareton rev-parse HEAD)" = "$TARGET_SHA"' in bootstrap
    assert bootstrap.index(
        'git -C /opt/pareton checkout --detach "$TARGET_SHA"'
    ) > bootstrap.index("3. Quiesce the remaining shared-environment users")
    assert bootstrap.index(
        'git -C /opt/pareton checkout --detach "$TARGET_SHA"'
    ) < bootstrap.index("for f in release.py ops_common.py")


def test_cli_help_runs_from_clean_env():
    # The runbook invocation is a plain file path with the venv python;
    # a clean environment (no pytest-injected paths) must reach --help
    # .
    import os
    import subprocess
    import sys

    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "PARETON"))
    }
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "ops" / "recover_submission.py"), "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "inspect" in result.stdout


def test_inspect_loads_env_file_database_config(tmp_path, monkeypatch):
    # R2-4: a manual shell has no unit EnvironmentFile; the CLI must load
    # /opt/pareton/.env itself or inspect dies with DatabaseNotConfigured
    # before ever reaching the database.
    import os
    import subprocess
    import sys

    env_file = tmp_path / ".env"
    env_file.write_text("PARETON_DATABASE_URL=postgres://invalid.invalid/db\n")
    clean = {
        k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "PARETON"))
    }
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "ops" / "recover_submission.py"),
            "inspect",
            "--job",
            "1",
        ],
        capture_output=True,
        text=True,
        env={**clean, "PARETON_ENV_FILE": str(env_file)},
        check=False,
        timeout=30,
    )
    # The URL is now seen: the failure moved past configuration into the
    # (unreachable here) connection.
    assert "DatabaseNotConfigured" not in result.stderr


@pytest.mark.e2e
def test_pg_recover_built_requires_null_campaign_bench(tmp_path, monkeypatch, capsys):
    url = require_e2e_database_url()
    monkeypatch.setenv("PARETON_DATABASE_URL", url)
    from db import connection

    monkeypatch.setattr(connection, "DATABASE_URL", url)
    monkeypatch.setattr(connection, "_pool", None)

    module = load()
    state_path = tmp_path / "release-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "op_id": "recover-test",
                "phase": "draining",
                "scope": "full",
                "direction": "forward",
                "from_commit": "a",
                "target_commit": "b",
                "verified_commit": "a",
            }
        )
    )
    monkeypatch.setenv("PARETON_RELEASE_STATE", str(state_path))
    monkeypatch.setenv("PARETON_DEPLOY_LOCK", str(tmp_path / "deploy.lock"))
    monkeypatch.setenv("PARETON_ACTIVITY_LOCK", str(tmp_path / "activity.lock"))

    campaign_ids = []
    submission_ids = []
    try:
        with connection.db_connection() as conn, conn.cursor() as cur:
            jobs = []
            for bench, attempt in ((None, 3), ({"mode": "bench"}, 4)):
                campaign_id = uuid4()
                submission_id = uuid4()
                campaign_ids.append(campaign_id)
                submission_ids.append(submission_id)
                cur.execute(
                    """
                    INSERT INTO campaigns (
                        id, baseline_repo, baseline_commit, base_image_digest,
                        priority_metric, success_threshold, manifest_hash, status,
                        bench, scoring_rule
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, 'draft', %s, %s
                    )
                    """,
                    (
                        str(campaign_id),
                        "https://example.invalid/baseline.git",
                        "deadbeef",
                        "sha256:" + "a" * 64,
                        "throughput",
                        "test",
                        "recover-" + uuid4().hex,
                        json.dumps(bench) if bench is not None else None,
                        json.dumps({}),
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO submissions (
                        id, campaign_id, patch_hash, hotkey, baseline_commit,
                        retrieval_url
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(submission_id),
                        str(campaign_id),
                        "sha256:" + uuid4().hex * 2,
                        "recover-test-hotkey",
                        "deadbeef",
                        "https://example.invalid/patch.diff",
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO submission_jobs (submission_id, status, attempts, phase)
                    VALUES (%s, 'running', %s, 'provisioning')
                    RETURNING id
                    """,
                    (str(submission_id), attempt),
                )
                job_id = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO submission_events (submission_id, state)
                    VALUES (%s, 'built')
                    """,
                    (str(submission_id),),
                )
                jobs.append((job_id, attempt))

        args = argparse.Namespace(
            job=jobs[0][0],
            attempt=jobs[0][1],
            outcome="settle",
            operator="e2e-test",
            reason="worker interrupted after build",
        )
        assert module.cmd_recover(args) == 0
        capsys.readouterr()
        with connection.db_connection(readonly=True) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status, phase, heartbeat_at FROM submission_jobs WHERE id = %s",
                (jobs[0][0],),
            )
            assert cur.fetchone() == ("done", None, None)

        args.job, args.attempt = jobs[1]
        assert module.cmd_recover(args) == 1
        capsys.readouterr()
        with connection.db_connection(readonly=True) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM submission_jobs WHERE id = %s",
                (jobs[1][0],),
            )
            assert cur.fetchone() == ("running",)
    finally:
        with connection.db_connection() as conn, conn.cursor() as cur:
            submission_id_values = [
                str(submission_id) for submission_id in submission_ids
            ]
            cur.execute(
                "DELETE FROM submission_jobs WHERE submission_id = ANY(%s::uuid[])",
                (submission_id_values,),
            )
            cur.execute(
                "DELETE FROM submission_events WHERE submission_id = ANY(%s::uuid[])",
                (submission_id_values,),
            )
            cur.execute(
                "DELETE FROM submissions WHERE id = ANY(%s::uuid[])",
                (submission_id_values,),
            )
            cur.execute(
                "DELETE FROM campaigns WHERE id = ANY(%s::uuid[])",
                ([str(campaign_id) for campaign_id in campaign_ids],),
            )
