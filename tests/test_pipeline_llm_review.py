"""Wiring of the optional d2 LLM review gate into the submission pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gate.types import GateResult, SubmissionState
from worker import pipeline

pytestmark = pytest.mark.unit

PATCH = b"diff --git a/x.py b/x.py\n"


@pytest.fixture
def run(monkeypatch):
    def _run(bench, review_result=None, job=(None, None)):
        calls = {"review": [], "build": 0, "events": []}
        campaign = SimpleNamespace(
            campaign_id="c1",
            bench=bench,
            baseline_commit="abc",
            allowed_paths=["**"],
            denied_paths=[],
            engine="sglang",
            base_image_digest="sha256:" + "0" * 64,
        )
        ok = GateResult.success(SubmissionState.VERIFIED)
        monkeypatch.setattr(pipeline, "get_campaign", lambda _cid: campaign)
        monkeypatch.setattr(
            pipeline,
            "append_event",
            lambda _sid, state, detail=None: calls["events"].append((state, detail)),
        )
        monkeypatch.setattr(pipeline, "set_job_status", lambda *a, **k: None)
        monkeypatch.setattr(pipeline, "check_identity", lambda **_: ok)
        monkeypatch.setattr(
            pipeline,
            "check_integrity",
            lambda **_: GateResult.success(
                SubmissionState.VERIFIED, patch_bytes=PATCH, patch_hash="ab" * 32
            ),
        )
        monkeypatch.setattr(pipeline, "check_base_apply_local", lambda **_: ok)
        monkeypatch.setattr(
            pipeline,
            "check_surface",
            lambda **_: GateResult.success(SubmissionState.SURFACE_OK, files=[]),
        )

        def _review(**kwargs):
            calls["review"].append(kwargs)
            return review_result

        monkeypatch.setattr(pipeline, "check_llm_review", _review)

        def _build(**_):
            calls["build"] += 1
            return GateResult.reject("stop_after_gates")

        monkeypatch.setattr(pipeline, "build_engine_image_local_mock", _build)
        res = pipeline.process_submission(
            {
                "id": "s1",
                "campaign_id": "c1",
                "hotkey": "hk",
                "patch_hash": "ab" * 32,
                "retrieval_url": "https://x",
                "baseline_commit": "abc",
                "job_id": job[0],
                "job_attempt": job[1],
            },
            registered_hotkeys=["hk"],
            mock_build=True,
            local_repo=pipeline.Path("/nonexistent"),
        )
        return res, calls

    return _run


def test_review_off_by_default_skips_gate(run):
    res, calls = run({})
    assert calls["review"] == []
    assert calls["build"] == 1
    assert res.reason == "stop_after_gates"


def test_review_reject_stops_before_build_and_keeps_patch(run):
    spec = {"policy": "dtype_precision_v1", "model": "kimi-k3"}
    rejected = GateResult.reject("llm_review_rejected", llm_reason="KV fp4")
    res, calls = run({"llm_review": spec}, rejected)
    assert res.reason == "llm_review_rejected"
    assert calls["build"] == 0
    (call,) = calls["review"]
    assert call["patch_bytes"] is PATCH
    assert call["spec"] == spec
    state, detail = calls["events"][-1]
    assert state == SubmissionState.REJECTED
    assert detail["llm_reason"] == "KV fp4"


def test_review_pass_continues_to_build(run):
    spec = {"policy": "dtype_precision_v1", "model": "kimi-k3"}
    passed = GateResult.success(SubmissionState.SURFACE_OK)
    res, calls = run({"llm_review": spec}, passed)
    assert len(calls["review"]) == 1
    assert calls["build"] == 1


UNAVAILABLE = GateResult.reject(
    "llm_review_unavailable", retryable=True, detail="ReadTimeout"
)
SPEC = {"policy": "dtype_precision_v1", "model": "kimi-k3"}


def test_outage_defers_job_with_backoff_and_never_builds(run, monkeypatch):
    import config

    monkeypatch.setattr(config, "LLM_REVIEW_MAX_ATTEMPTS", 6)
    monkeypatch.setattr(config, "LLM_REVIEW_RETRY_BASE_S", 300.0)
    monkeypatch.setattr(config, "LLM_REVIEW_RETRY_MAX_S", 3600.0)
    deferred = []
    monkeypatch.setattr(pipeline, "defer_job", lambda **kw: deferred.append(kw) or True)
    res, calls = run({"llm_review": SPEC}, UNAVAILABLE, job=(7, 2))
    assert res.reason == "llm_review_deferred"
    assert calls["build"] == 0
    assert deferred == [
        {
            "job_id": 7,
            "attempt": 2,
            "delay_s": 600.0,
            "last_error": "llm_review_unavailable: ReadTimeout",
        }
    ]
    # No terminal event: the submission is not rejected while deferred.
    assert all(state != SubmissionState.REJECTED for state, _ in calls["events"])


def test_backoff_is_capped(run, monkeypatch):
    import config

    monkeypatch.setattr(config, "LLM_REVIEW_MAX_ATTEMPTS", 20)
    deferred = []
    monkeypatch.setattr(pipeline, "defer_job", lambda **kw: deferred.append(kw) or True)
    run({"llm_review": SPEC}, UNAVAILABLE, job=(7, 10))
    assert deferred[0]["delay_s"] == config.LLM_REVIEW_RETRY_MAX_S


def test_outage_rejects_once_attempts_run_out(run, monkeypatch):
    import config

    monkeypatch.setattr(config, "LLM_REVIEW_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(
        pipeline, "defer_job", lambda **kw: pytest.fail("must not defer")
    )
    res, calls = run({"llm_review": SPEC}, UNAVAILABLE, job=(7, 3))
    assert res.reason == "llm_review_unavailable"
    assert calls["build"] == 0
    assert calls["events"][-1][0] == SubmissionState.REJECTED


def test_obfuscation_gate_runs_before_llm_review(run, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "check_obfuscation",
        lambda **_: GateResult.reject("obfuscation_detected", rules=["dynamic_code"]),
    )
    res, calls = run(
        {"patch_limits": {"reject_obfuscation": True}, "llm_review": SPEC},
        GateResult.success(SubmissionState.SURFACE_OK),
    )
    assert res.reason == "obfuscation_detected"
    assert calls["review"] == []
    assert calls["build"] == 0


def test_obfuscation_gate_off_by_default(run, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "check_obfuscation",
        lambda **_: pytest.fail("must not run when the flag is unset"),
    )
    _res, calls = run({})
    assert calls["build"] == 1
