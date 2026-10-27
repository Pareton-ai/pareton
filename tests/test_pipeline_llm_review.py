"""Wiring of the d1 obfuscation and d2 LLM review gates into the pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gate.types import GateResult, SubmissionState
from worker import pipeline

pytestmark = pytest.mark.unit

PATCH = b"diff --git a/x.py b/x.py\n"
VIEW = SimpleNamespace(
    review_diff="diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
    post_texts={"x.py": "b\n"},
    strip_summary=lambda: {"status_counts": {"no_comments": 2}, "comments_kept": []},
)


@pytest.fixture
def run(monkeypatch):
    def _run(bench, review_result=None, job=(None, None), obfuscation=None, view=None):
        calls = {"review": [], "obfuscation": [], "view": [], "build": 0, "events": []}
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
        monkeypatch.setattr(
            pipeline,
            "check_base_apply_local",
            lambda **_: GateResult.success(SubmissionState.APPLIED, repo_dir="/repo"),
        )

        def _view(repo_dir, patch_bytes):
            calls["view"].append((repo_dir, patch_bytes))
            return view() if view else VIEW

        monkeypatch.setattr(pipeline, "build_source_view", _view)

        def _obfuscation(**kwargs):
            calls["obfuscation"].append(kwargs)
            return obfuscation or GateResult.success(SubmissionState.SURFACE_OK)

        monkeypatch.setattr(pipeline, "check_obfuscation", _obfuscation)
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
    assert call["review_text"] == VIEW.review_diff
    assert call["strip_summary"] == VIEW.strip_summary()
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


def test_outage_defers_job_and_never_builds(run, monkeypatch):
    import config

    monkeypatch.setattr(config, "LLM_REVIEW_MAX_ATTEMPTS", 6)
    deferred = []
    monkeypatch.setattr(pipeline, "defer_job", lambda **kw: deferred.append(kw) or True)
    res, calls = run({"llm_review": SPEC}, UNAVAILABLE, job=(7, 2))
    assert res.reason == "llm_review_deferred"
    assert res.evidence["retry_in_s"] == config.LLM_REVIEW_RETRY_AFTER_S
    assert calls["build"] == 0
    assert deferred == [
        {
            "job_id": 7,
            "attempt": 2,
            "last_error": "llm_review_unavailable: ReadTimeout",
        }
    ]
    # No terminal event: the submission is not rejected while deferred.
    assert all(state != SubmissionState.REJECTED for state, _ in calls["events"])


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


def test_obfuscation_gate_runs_before_llm_review(run):
    res, calls = run(
        {"llm_review": SPEC},
        GateResult.success(SubmissionState.SURFACE_OK),
        obfuscation=GateResult.reject("obfuscation_detected", rules=["dynamic_code"]),
    )
    assert res.reason == "obfuscation_detected"
    assert calls["review"] == []
    assert calls["build"] == 0
    assert calls["events"][-1][0] == SubmissionState.REJECTED


def test_obfuscation_gate_runs_for_every_campaign(run):
    _res, calls = run({})
    (call,) = calls["obfuscation"]
    assert call["patch_bytes"] is PATCH
    assert call["post_texts"] == VIEW.post_texts
    assert calls["view"] == [(pipeline.Path("/repo"), PATCH)]
    assert calls["build"] == 1


def test_source_view_failure_rejects(run):
    def _boom():
        raise pipeline.SourceViewError("git apply failed")

    res, calls = run(
        {"llm_review": SPEC}, GateResult.success(SubmissionState.SURFACE_OK), view=_boom
    )
    assert res.reason == "source_view_failed"
    assert calls["review"] == []
    assert calls["build"] == 0


def test_obfuscation_event_omits_patch_snippets(run):
    finding = {"file": "x.py", "line": 3, "rule": "dynamic_code", "snippet": "eval(s)"}
    res, calls = run(
        {},
        obfuscation=GateResult.reject(
            "obfuscation_detected", rules=["dynamic_code"], findings=[finding]
        ),
    )
    state, detail = calls["events"][-1]
    assert state == SubmissionState.REJECTED
    assert detail["findings"] == [{"file": "x.py", "line": 3, "rule": "dynamic_code"}]
    assert "eval(s)" not in str(detail)
