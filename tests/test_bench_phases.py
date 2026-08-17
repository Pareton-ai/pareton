"""Unit tests for live bench phases, attempt-scoped writes, and reporting."""

from __future__ import annotations

import contextlib
import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from bench.output import OutputLayout
from bench.phases import (
    BENCH_PHASES,
    POD_REPORTABLE_PHASES,
    BenchPhase,
    coerce_phase,
    coerce_progress,
)
from worker.phase_reporter import (
    NullPhaseReporter,
    PhaseReporter,
    reporter_for_job,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


# -- Vocabulary --------------------------------------------------------------


def test_phases_are_disjoint_from_pipeline_states():
    """Phases and pipeline states must not share names."""
    from gate.types import SUBMISSION_STATES

    assert set(BENCH_PHASES).isdisjoint(SUBMISSION_STATES)


def test_pod_cannot_claim_worker_owned_phases():
    """Pod lifecycle belongs to the worker; a pod claiming it is bogus."""
    assert BenchPhase.PROVISIONING.value not in POD_REPORTABLE_PHASES
    assert BenchPhase.TEARDOWN.value not in POD_REPORTABLE_PHASES
    assert POD_REPORTABLE_PHASES < set(BENCH_PHASES)


@pytest.mark.parametrize("phase", BENCH_PHASES)
def test_coerce_phase_accepts_every_member(phase: str):
    assert coerce_phase(phase) == phase


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "",
        "benched",
        "downloading_model; DROP TABLE submission_jobs",
        "<script>alert(1)</script>",
        "DOWNLOADING_MODEL",
        ["downloading_model"],
    ],
)
def test_coerce_phase_drops_untrusted_input(value: Any):
    assert coerce_phase(value) is None


def test_coerce_phase_tolerates_surrounding_whitespace():
    """Marker text arrives via `head`, so a trailing newline is normal."""
    assert coerce_phase(" sla_bench\n") == "sla_bench"


def test_coerce_phase_narrowed_to_pod_reportable():
    assert coerce_phase("provisioning", allowed=POD_REPORTABLE_PHASES) is None, (
        "a pod must not be able to claim it is renting itself"
    )
    assert coerce_phase("correctness", allowed=POD_REPORTABLE_PHASES) == "correctness"


def test_coerce_progress_clamps_pod_supplied_detail():
    long_value = "x" * 500
    out = coerce_progress(
        {
            "percent": 42,
            "note": long_value,
            "ok": True,
            "nested": {"no": "objects"},
            "bad key": "dropped",
            "": "dropped",
        }
    )
    assert out is not None
    assert out["percent"] == 42
    assert out["ok"] is True
    assert len(out["note"]) == 64
    assert "nested" not in out
    assert "bad key" not in out


def test_coerce_progress_rejects_empty_and_non_objects():
    assert coerce_progress({}) is None
    assert coerce_progress("downloading") is None
    assert coerce_progress({"nested": {"a": 1}}) is None


def test_coerce_progress_caps_key_count():
    out = coerce_progress({f"k{i}": i for i in range(50)})
    assert out is not None and len(out) == 8


def test_schema_phase_check_matches_the_enum():
    """submission_jobs.phase CHECK must match BENCH_PHASES."""
    schema = (ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    block = schema.split("CREATE TABLE IF NOT EXISTS submission_jobs")[1].split(");")[0]
    match = re.search(r"phase IN \(([^)]*)\)", block, re.DOTALL)
    assert match, "submission_jobs.phase must keep a CHECK constraint"
    assert tuple(re.findall(r"'([a-z_]+)'", match.group(1))) == BENCH_PHASES


# -- Store: attempt-scoped writes -------------------------------------------


class _RecordingCursor:
    """Records SQL and answers just enough for the job-row queries."""

    def __init__(self, sqls: list[str], args: list[Any], rowcount: int) -> None:
        self._sqls = sqls
        self._args = args
        self.rowcount = rowcount

    def execute(self, sql: str, args: Any = None) -> None:
        self._sqls.append(" ".join(sql.split()))
        self._args.append(args)

    def fetchone(self) -> Any:
        sql = self._sqls[-1] if self._sqls else ""
        if "SELECT j.id AS job_id" in sql:
            return {
                "job_id": 7,
                "submission_id": "00000000-0000-0000-0000-000000000001",
                "kind": "bench",
            }
        if "RETURNING attempts" in sql:
            return {"attempts": 3}
        if "SELECT s.*" in sql:
            return {"id": "00000000-0000-0000-0000-000000000001", "job_attempt": 3}
        return None

    def fetchall(self) -> list[Any]:
        return []

    def __enter__(self) -> _RecordingCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@contextlib.contextmanager
def _fake_db(sqls: list[str], args: list[Any], *, rowcount: int = 1):
    class _Conn:
        def cursor(self, cursor_factory=None):
            return _RecordingCursor(sqls, args, rowcount)

    yield _Conn()


def _patch_db(monkeypatch, *, rowcount: int = 1) -> tuple[list[str], list[Any]]:
    from campaign import store

    sqls: list[str] = []
    args: list[Any] = []
    monkeypatch.setattr(
        store,
        "db_connection",
        lambda **_kw: _fake_db(sqls, args, rowcount=rowcount),
    )
    return sqls, args


def test_set_job_phase_is_scoped_to_the_running_attempt(monkeypatch):
    from campaign import store

    sqls, args = _patch_db(monkeypatch)
    assert (
        store.set_job_phase(
            job_id=7, attempt=2, phase="downloading_model", progress={"percent": 10}
        )
        is True
    )
    sql = sqls[-1]
    assert "WHERE id = %s AND attempts = %s AND status = 'running'" in sql
    assert "phase IS DISTINCT FROM %s" in sql
    assert args[-1][3:] == (7, 2)


def test_set_job_phase_reports_a_superseded_attempt(monkeypatch):
    """Zero rows updated means another attempt owns the row now."""
    from campaign import store

    _patch_db(monkeypatch, rowcount=0)
    assert store.set_job_phase(job_id=7, attempt=1, phase="correctness") is False, (
        "a stale attempt must not be able to move the phase backwards"
    )


def test_touch_job_heartbeat_is_scoped_and_only_touches_liveness(monkeypatch):
    from campaign import store

    sqls, args = _patch_db(monkeypatch)
    assert store.touch_job_heartbeat(job_id=9, attempt=4) is True
    sql = sqls[-1]
    assert "SET heartbeat_at = now()" in sql
    assert "phase =" not in sql
    assert "WHERE id = %s AND attempts = %s AND status = 'running'" in sql
    assert args[-1] == (9, 4)


def test_claim_resets_live_activity_and_returns_the_attempt(monkeypatch):
    """A fresh attempt starts clean, and the caller learns which attempt it is."""
    from campaign import store

    sqls, _args = _patch_db(monkeypatch)
    row = store.claim_next_job(kind="bench")
    assert row is not None and row["job_attempt"] == 3
    claim = next(s for s in sqls if "SET status = 'running'" in s)
    assert "attempts = attempts + 1" in claim
    assert "phase = NULL" in claim
    assert "heartbeat_at = now()" in claim
    assert "RETURNING attempts" in claim


def test_settling_a_job_clears_live_activity(monkeypatch):
    from campaign import store

    sqls, _args = _patch_db(monkeypatch)
    store.set_job_status("00000000-0000-0000-0000-000000000001", "failed", job_id=5)
    assert "phase = NULL" in sqls[-1]
    assert "heartbeat_at = NULL" in sqls[-1]

    sqls.clear()
    store.finalize_bench_job(
        submission_id="00000000-0000-0000-0000-000000000001",
        job_id=5,
        task_id="t",
        report_rows=[],
        events=[],
        job_status="done",
    )
    final = next(s for s in sqls if "UPDATE submission_jobs" in s)
    assert "phase = NULL" in final
    assert "progress = NULL" in final


def test_set_job_status_still_bumps_attempts_on_request(monkeypatch):
    from campaign import store

    sqls, _args = _patch_db(monkeypatch)
    store.set_job_status(
        "00000000-0000-0000-0000-000000000001",
        "pending",
        kind="gates",
        bump_attempts=True,
    )
    assert "attempts = attempts + 1" in sqls[-1]

    sqls.clear()
    store.set_job_status("00000000-0000-0000-0000-000000000001", "pending", job_id=1)
    assert "attempts = attempts + 1" not in sqls[-1]


def test_job_listing_exposes_live_activity(monkeypatch):
    from campaign import store

    sqls, _args = _patch_db(monkeypatch)
    store.list_submission_jobs("00000000-0000-0000-0000-000000000001")
    sql = sqls[-1]
    for column in ("phase", "phase_started_at", "heartbeat_at", "progress"):
        assert column in sql


# -- Reporter ----------------------------------------------------------------


class _Writer:
    """Phase/heartbeat writer that records calls and can reject them."""

    def __init__(self, *, landed: bool = True, raises: bool = False) -> None:
        self.phases: list[tuple[str, dict[str, Any] | None]] = []
        self.beats = 0
        self._landed = landed
        self._raises = raises

    def write_phase(self, *, job_id, attempt, phase, progress=None) -> bool:
        if self._raises:
            raise RuntimeError("neon is down")
        self.phases.append((phase, progress))
        return self._landed

    def beat(self, *, job_id, attempt) -> bool:
        if self._raises:
            raise RuntimeError("neon is down")
        self.beats += 1
        return self._landed


def _reporter(writer: _Writer, **kwargs: Any) -> PhaseReporter:
    return PhaseReporter(
        job_id=1,
        attempt=2,
        phase_writer=writer.write_phase,
        heartbeat_writer=writer.beat,
        **kwargs,
    )


def test_reporter_writes_once_per_phase_change():
    writer = _Writer()
    reporter = _reporter(writer)
    reporter.set(BenchPhase.PROVISIONING)
    reporter.set(BenchPhase.PROVISIONING)
    reporter.set("bootstrapping")
    # A multi-SKU run legitimately returns to an earlier phase.
    reporter.set(BenchPhase.PROVISIONING)
    assert [p for p, _ in writer.phases] == [
        "provisioning",
        "bootstrapping",
        "provisioning",
    ]


def test_reporter_forwards_and_clamps_progress():
    writer = _Writer()
    _reporter(writer).set(BenchPhase.DOWNLOADING_MODEL, percent=12, junk={"a": 1})
    assert writer.phases == [("downloading_model", {"percent": 12})]


def test_reporter_drops_unknown_phase_names():
    writer = _Writer()
    reporter = _reporter(writer)
    reporter.set("benched")
    reporter.set(None)
    assert writer.phases == []


def test_reporter_goes_quiet_once_its_attempt_is_superseded():
    """Rejected writes stop further reporting."""
    writer = _Writer(landed=False)
    reporter = _reporter(writer)
    reporter.set(BenchPhase.CORRECTNESS)
    reporter.set(BenchPhase.SLA_BENCH)
    reporter._beat()
    assert [p for p, _ in writer.phases] == ["correctness"]
    assert writer.beats == 0


def test_reporter_swallows_write_failures():
    """Progress reporting must never be able to fail a bench run."""
    writer = _Writer(raises=True)
    reporter = _reporter(writer)
    reporter.set(BenchPhase.SLA_BENCH)
    reporter._beat()
    reporter.stop()


def test_reporter_heartbeats_in_the_background():
    writer = _Writer()
    beat = threading.Event()
    original = writer.beat

    def counting_beat(**kwargs: Any) -> bool:
        result = original(**kwargs)
        beat.set()
        return result

    reporter = PhaseReporter(
        job_id=1,
        attempt=1,
        phase_writer=writer.write_phase,
        heartbeat_writer=counting_beat,
        interval_s=0.01,
    )
    with reporter:
        assert beat.wait(5.0), "a blocked bench run must still prove liveness"
    assert reporter._thread is None


def test_reporter_for_unclaimed_row_writes_nothing():
    """Direct `process_bench_job` calls have no attempt to scope writes to."""
    assert isinstance(reporter_for_job({"job_id": 1}), NullPhaseReporter)
    assert isinstance(reporter_for_job({"job_attempt": 1}), NullPhaseReporter)
    claimed = reporter_for_job({"job_id": 4, "job_attempt": 2})
    assert isinstance(claimed, PhaseReporter)
    assert (claimed.job_id, claimed.attempt) == (4, 2)


# -- Harness marker ----------------------------------------------------------


def test_layout_publishes_a_phase_marker(tmp_path: Path):
    layout = OutputLayout(tmp_path / "out")
    layout.prepare()
    layout.write_phase(BenchPhase.DOWNLOADING_MODEL.value, percent=3)
    record = json.loads(layout.phase_path.read_text(encoding="utf-8"))
    assert record["phase"] == "downloading_model"
    assert record["progress"] == {"percent": 3}
    assert record["at"]

    layout.write_phase(BenchPhase.SLA_BENCH.value)
    record = json.loads(layout.phase_path.read_text(encoding="utf-8"))
    assert record["phase"] == "sla_bench"
    assert "progress" not in record
    assert not list(layout.root.glob("*.tmp"))


def test_layout_phase_write_never_raises(tmp_path: Path):
    layout = OutputLayout(tmp_path / "missing")
    layout.write_phase(BenchPhase.TEARDOWN.value)


def test_harness_reports_engine_start_then_hands_back_to_the_module(tmp_path: Path):
    """starting_engine during load, then back to the module."""
    from bench.main import _EngineProvider
    from bench.schemas import BenchRequest
    from bench.validate import load_bench_request

    req: BenchRequest = load_bench_request(
        ROOT / "fixtures" / "bench" / "sample_request.json"
    )[0]
    seen: list[str] = []
    provider = _EngineProvider(req=req, mock=False, phase_sink=seen.append)
    provider._enter_module(BenchPhase.CORRECTNESS)

    class _Net:
        run_id = "run"
        runner = None

    container = provider._docker_phase(_Net(), "baseline")
    assert seen == ["correctness", "starting_engine"]
    container.on_ready()
    assert seen[-1] == "correctness"
