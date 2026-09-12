"""Offline tests for ops/recover_submission.py decision rules (spec 5.1, B22).

These unit tests pin the evidence rules that gate the database-backed
paths and the CLI entry itself. The transactional recover/inspect paths
against a real Postgres (spec B22) remain an open gap: no e2e-marked test
covers them yet — they need PARETON_TEST_DATABASE_URL infrastructure.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "recover_submission", REPO_ROOT / "ops" / "recover_submission.py"
)


def test_module_loads_with_venv_imports():
    # db.connection imports psycopg2 lazily; module import alone must work.
    assert spec is not None


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
    assert module.evidence_class(["bench_queued"]) == "progressed"
    assert module.evidence_class(["rejected"]) == "terminal-reject"
    assert module.evidence_class(["scored", "rejected"]) == "contradictory"


def test_no_submission_event_writes():
    # The recovery audit must never touch submission_events: its state
    # vocabulary is miner-visible (latest_state / by_latest_state), and
    # 'recovered' is not a SubmissionState member. Source-level guard for
    # the fix that once failed to land while its commit claimed otherwise.
    source = (REPO_ROOT / "ops" / "recover_submission.py").read_text()
    assert "INSERT INTO submission_events" not in source
    assert "submission_recovered" in source  # audit goes to stdout/journald


def test_cli_help_runs_from_clean_env():
    # The runbook invocation is a plain file path with the venv python;
    # a clean environment (no pytest-injected paths) must reach --help
    # (PR-review P2-2).
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
    )
    assert result.returncode == 0, result.stderr
    assert "inspect" in result.stdout
