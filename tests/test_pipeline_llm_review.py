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
    def _run(bench, review_result=None):
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


def test_review_skipped_on_outage_continues_to_build(run):
    spec = {"policy": "dtype_precision_v1", "model": "kimi-k3"}
    skipped = GateResult.success(
        SubmissionState.SURFACE_OK, skipped=True, detail="HTTP 503"
    )
    res, calls = run({"llm_review": spec}, skipped)
    assert calls["build"] == 1
    assert res.reason == "stop_after_gates"
