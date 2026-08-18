"""Hermetic tests for observability.events lifecycle emitter.

No DB, GPU, or network required.  Validates:
- Every event function produces valid single-line JSON.
- Required fields are present.
- Job-scoped events carry non-empty submission_id.
- Forbidden keys (secrets) are stripped.
- Timer context manager tracks elapsed time.
"""

from __future__ import annotations

import json
import logging
import time

import pytest

from observability import events as obs
from observability.events import Timer

# ---- Helpers ---------------------------------------------------------------

_ALL_EVENTS = [
    "heartbeat",
    "chain_scanned",
    "submission_ingested",
    "gate_passed",
    "gate_failed",
    "build_started",
    "build_succeeded",
    "build_failed",
    "pod_provisioned",
    "pod_provision_failed",
    "pod_destroyed",
    "destroy_failed",
    "pod_ttl_exceeded",
    "bench_started",
    "bench_completed",
    "bench_failed",
    "provider_balance_low",
    "job_failed",
]

_JOB_SCOPED_EVENTS = [
    "submission_ingested",
    "gate_passed",
    "gate_failed",
    "build_started",
    "build_succeeded",
    "build_failed",
    "bench_started",
    "bench_completed",
    "bench_failed",
    "job_failed",
]


def _call_event(name: str) -> dict:
    """Call the named event function with minimal valid kwargs."""
    fn = getattr(obs, name)
    kwargs: dict = {}
    if name == "heartbeat":
        kwargs = {}
    elif name == "chain_scanned":
        kwargs = {"block": 1234, "commitments_seen": 3, "ingested": 1}
    elif name == "submission_ingested":
        kwargs = {
            "submission_id": "sub-1",
            "campaign_id": "camp-1",
            "patch_sha256": "abc123",
            "hotkey": "hk" * 16,
        }
    elif name == "gate_passed":
        kwargs = {"submission_id": "sub-1", "gate": "identity", "patch_sha256": "abc"}
    elif name == "gate_failed":
        kwargs = {
            "submission_id": "sub-1",
            "gate": "integrity",
            "error": "hash mismatch",
            "patch_sha256": "abc",
        }
    elif name == "build_started":
        kwargs = {"submission_id": "sub-1", "patch_sha256": "abc"}
    elif name == "build_succeeded":
        kwargs = {
            "submission_id": "sub-1",
            "patch_sha256": "abc",
            "image_digest": "sha256:dead",
            "duration_s": 120.5,
        }
    elif name == "build_failed":
        kwargs = {
            "submission_id": "sub-1",
            "patch_sha256": "abc",
            "error": "oom",
            "duration_s": 60.0,
        }
    elif name == "pod_provisioned":
        kwargs = {"pod": "pt-xyz", "provider": "lium"}
    elif name == "pod_provision_failed":
        kwargs = {"provider": "lium", "error": "no capacity"}
    elif name == "pod_destroyed":
        kwargs = {"pod": "pt-xyz", "provider": "lium"}
    elif name == "destroy_failed":
        kwargs = {"pod": "pt-xyz", "provider": "lium", "error": "api timeout"}
    elif name == "pod_ttl_exceeded":
        kwargs = {"pod": "pt-xyz", "provider": "lium"}
    elif name == "bench_started":
        kwargs = {"submission_id": "sub-1", "job_id": "42", "stage": "bench"}
    elif name == "bench_completed":
        kwargs = {
            "submission_id": "sub-1",
            "job_id": "42",
            "stage": "bench",
            "duration_s": 300.0,
            "verdict": "pass",
            "evidence_s3_url": "s3://pareton-s3/stage0/evidence/sub-1/bundle.tgz",
            "evidence_sha256": "sha256:abc123",
        }
    elif name == "bench_failed":
        kwargs = {
            "submission_id": "sub-1",
            "job_id": "42",
            "stage": "bench",
            "error": "exit_3",
        }
    elif name == "provider_balance_low":
        kwargs = {"provider": "lium", "balance": 5.0, "threshold": 10.0}
    elif name == "job_failed":
        kwargs = {
            "submission_id": "sub-1",
            "job_id": "42",
            "stage": "bench",
            "error": "oom",
        }
    return fn(**kwargs)


# ---- Tests -----------------------------------------------------------------


class TestEventEmitter:
    """Core emitter behavior."""

    @pytest.mark.parametrize("event_name", _ALL_EVENTS)
    def test_produces_valid_json(self, event_name: str) -> None:
        payload = _call_event(event_name)
        raw = json.dumps(payload, default=str)
        parsed = json.loads(raw)
        assert parsed["event"] == event_name

    @pytest.mark.parametrize("event_name", _ALL_EVENTS)
    def test_event_field_present(self, event_name: str) -> None:
        payload = _call_event(event_name)
        assert "event" in payload
        assert payload["event"] == event_name

    @pytest.mark.parametrize("event_name", _JOB_SCOPED_EVENTS)
    def test_job_scoped_has_submission_id(self, event_name: str) -> None:
        payload = _call_event(event_name)
        assert "submission_id" in payload
        assert payload["submission_id"] != ""

    @pytest.mark.parametrize("event_name", _ALL_EVENTS)
    def test_no_forbidden_keys(self, event_name: str) -> None:
        payload = _call_event(event_name)
        for key in obs._FORBIDDEN_KEYS:
            assert key not in payload

    def test_forbidden_keys_stripped(self) -> None:
        payload = obs._emit("test_event", token="secret123", submission_id="s1")
        assert "token" not in payload
        assert payload["submission_id"] == "s1"

    def test_none_and_empty_values_omitted(self) -> None:
        payload = obs._emit("test_event", a=None, b="", c="present")
        assert "a" not in payload
        assert "b" not in payload
        assert payload["c"] == "present"

    def test_hotkey_truncated(self) -> None:
        long_hotkey = "a" * 64
        payload = obs.submission_ingested(
            submission_id="sub-1",
            campaign_id="camp-1",
            patch_sha256="abc",
            hotkey=long_hotkey,
        )
        assert len(payload["hotkey"]) == 16

    def test_duration_rounded(self) -> None:
        payload = obs.build_succeeded(
            submission_id="sub-1",
            patch_sha256="abc",
            duration_s=123.456789,
        )
        assert payload["duration_s"] == 123.5


class TestChainScanned:
    """The scan event that makes a stalled chain read alertable."""

    def test_zero_counts_survive_the_empty_value_filter(self) -> None:
        # A quiet scan is the case the alert depends on, so 0 must be emitted
        # rather than dropped as an empty value.
        payload = obs.chain_scanned(block=88, commitments_seen=0, ingested=0)
        assert payload["commitments_seen"] == 0
        assert payload["ingested"] == 0
        assert payload["block"] == 88


class TestHeartbeatQueueDepth:
    """queue_depth on the heartbeat: real number, and no beat lost on DB error."""

    def test_reports_pending_job_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from worker import main as worker_main

        monkeypatch.setattr(worker_main, "count_pending_jobs", lambda: 4)
        assert worker_main._queue_depth() == 4

    def test_zero_pending_is_reported_not_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from worker import main as worker_main

        monkeypatch.setattr(worker_main, "count_pending_jobs", lambda: 0)
        payload = obs.heartbeat(queue_depth=worker_main._queue_depth())
        assert payload["queue_depth"] == 0

    def test_database_error_still_beats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from worker import main as worker_main

        def _boom() -> int:
            raise RuntimeError("no database")

        monkeypatch.setattr(worker_main, "count_pending_jobs", _boom)
        assert worker_main._queue_depth() is None
        assert "queue_depth" not in obs.heartbeat(queue_depth=None)


class TestLogger:
    """Verify events go through the logging system."""

    def test_emits_to_lifecycle_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="pareton.lifecycle"):
            obs.heartbeat()
        assert len(caplog.records) == 1
        record = caplog.records[0]
        parsed = json.loads(record.message)
        assert parsed["event"] == "heartbeat"

    def test_single_line_json(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="pareton.lifecycle"):
            obs.submission_ingested(
                submission_id="s1",
                campaign_id="c1",
                patch_sha256="abc",
                hotkey="hk" * 16,
            )
        assert "\n" not in caplog.records[0].message


class TestTimer:
    """Timer context manager."""

    def test_tracks_elapsed(self) -> None:
        t = Timer()
        with t:
            time.sleep(0.05)
        assert t.elapsed_s >= 0.04

    def test_zero_before_use(self) -> None:
        t = Timer()
        assert t.elapsed_s == 0.0


class TestHeartbeatLoop:
    """Background heartbeat thread: keeps emitting while jobs block the loop."""

    def test_emits_repeatedly_until_stopped(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading

        from worker import main as worker_main

        monkeypatch.setattr(worker_main, "count_pending_jobs", lambda: 2)
        stop = threading.Event()
        thread = threading.Thread(
            target=worker_main._heartbeat_loop,
            args=(stop,),
            kwargs={"interval_s": 0.05},
            daemon=True,
        )
        with caplog.at_level(logging.INFO, logger="pareton.lifecycle"):
            thread.start()
            time.sleep(0.18)
            stop.set()
            thread.join(timeout=2)
        beats = [r for r in caplog.records if '"heartbeat"' in r.message]
        assert len(beats) >= 2
        assert all(json.loads(r.message)["queue_depth"] == 2 for r in beats)
