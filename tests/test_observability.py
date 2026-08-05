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
