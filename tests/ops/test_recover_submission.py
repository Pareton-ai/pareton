"""Offline tests for ops/recover_submission.py decision rules (spec 5.1, B22).

The database-backed paths run against the dedicated e2e database (marked
e2e); these unit tests pin the evidence rules that gate them.
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
