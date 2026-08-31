"""Unit tests for the scoring rule seam: dispatch, the metric, and the pin."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

pytestmark = pytest.mark.unit


from bench.score import (
    DEFAULT_SPEED_TOLERANCE,
    SCORING_RULES,
    PromptTiming,
    aligned_e2e_s,
    prompt_speedup,
    score_candidate,
)
from campaign.models import SCORING_RULE_NAMES

RULE: dict[str, Any] = {"name": "median_e2e_speedup"}


def timing(ttft_s: float, gap_s: float, tokens: int) -> PromptTiming:
    """A run that emits `tokens` tokens with a constant gap between them."""
    return PromptTiming(
        ttft_s=ttft_s, itl_s=[gap_s] * (tokens - 1), completion_tokens=tokens
    )


# -- Dispatch ----------------------------------------------------------------


def test_every_validated_rule_name_has_an_implementation():
    assert set(SCORING_RULES) == set(SCORING_RULE_NAMES)


def test_an_unknown_rule_name_is_refused():
    with pytest.raises(ValueError, match="scoring_rule.name must be one of"):
        score_candidate({"name": "vibes"}, baseline={}, candidate={})


# -- median_e2e_speedup ------------------------------------------------------


def test_aligned_e2e_stops_at_the_kth_token():
    t = timing(0.1, 0.01, 5)
    assert aligned_e2e_s(t, 1) == pytest.approx(0.1)
    assert aligned_e2e_s(t, 3) == pytest.approx(0.12)
    assert aligned_e2e_s(t, 6) is None


def test_a_twice_as_fast_candidate_scores_one_half():
    score = prompt_speedup("p1", timing(0.2, 0.02, 10), timing(0.1, 0.01, 10))
    assert score.speedup == pytest.approx(0.5)
    assert score.baseline_e2e_s == pytest.approx(0.38)
    assert score.candidate_e2e_s == pytest.approx(0.19)
    assert score.aligned_tokens == 10


def test_a_slower_candidate_scores_below_zero():
    score = prompt_speedup("p1", timing(0.1, 0.01, 10), timing(0.2, 0.02, 10))
    assert score.speedup == pytest.approx(-1.0)


def test_speed_is_compared_at_the_same_token_count():
    # The candidate stops one token early: it is not credited for the token
    # the baseline spent time on.
    base = timing(0.1, 0.01, 10)
    cand = timing(0.1, 0.01, 9)
    score = prompt_speedup("p1", base, cand)
    assert score.aligned_tokens == 9
    assert score.speedup == pytest.approx(0.0)


def test_output_below_the_tolerance_gate_earns_no_credit():
    base = timing(0.2, 0.02, 10)
    truncated = timing(0.01, 0.001, 5)
    score = prompt_speedup("p1", base, truncated, tolerance=DEFAULT_SPEED_TOLERANCE)
    assert score.speedup == 0.0
    assert score.reason == "candidate output below tolerance"


def test_a_prompt_the_candidate_never_answered_earns_no_credit():
    score = prompt_speedup("p1", timing(0.2, 0.02, 10), None)
    assert score.speedup == 0.0
    assert score.reason == "no candidate timing"


def test_the_round_score_is_the_median_across_prompts():
    base = {p: timing(0.2, 0.02, 10) for p in ("p1", "p2", "p3")}
    # 50 percent, 0 percent, 50 percent faster -> median 0.5.
    candidate = {
        "p1": timing(0.1, 0.01, 10),
        "p2": timing(0.2, 0.02, 10),
        "p3": timing(0.1, 0.01, 10),
    }
    result = score_candidate(RULE, baseline=base, candidate=candidate)
    assert result.score == pytest.approx(0.5)
    assert result.rule == "median_e2e_speedup"
    assert [p.request_id for p in result.per_prompt] == ["p1", "p2", "p3"]


def test_the_report_keeps_absolute_seconds_not_only_the_ratio():
    result = score_candidate(
        RULE,
        baseline={"p1": timing(0.2, 0.02, 10)},
        candidate={"p1": timing(0.1, 0.01, 10)},
    )
    report = result.to_report()
    assert report["rule"] == "median_e2e_speedup"
    assert report["prompts"][0]["baseline_e2e_s"] == pytest.approx(0.38)
    assert report["prompts"][0]["candidate_e2e_s"] == pytest.approx(0.19)


def test_the_tolerance_is_read_from_the_rule():
    base = timing(0.2, 0.02, 10)
    cand = timing(0.05, 0.005, 6)
    strict = score_candidate(RULE, baseline={"p1": base}, candidate={"p1": cand})
    loose = score_candidate(
        {**RULE, "tolerance": 0.5}, baseline={"p1": base}, candidate={"p1": cand}
    )
    assert strict.score == 0.0
    assert loose.score > 0.5


# -- scoring_rule immutability ----------------------------------------------


class _Cursor:
    def __init__(self, row: dict[str, Any] | None, sqls: list[str], args: list[Any]):
        self._row = row
        self._sqls = sqls
        self._args = args
        self.rowcount = 1

    def execute(self, sql: str, args: Any = None) -> None:
        self._sqls.append(" ".join(sql.split()))
        self._args.append(args)

    def fetchone(self) -> Any:
        return self._row if "SELECT" in self._sqls[-1] else None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _patch_db(monkeypatch, row: dict[str, Any] | None):
    from campaign import store

    sqls: list[str] = []
    args: list[Any] = []

    @contextlib.contextmanager
    def fake(**_kw):
        class _Conn:
            def cursor(self, cursor_factory=None):
                return _Cursor(row, sqls, args)

        yield _Conn()

    monkeypatch.setattr(store, "db_connection", fake)
    return sqls, args


def _campaign_row(status: str) -> dict[str, Any]:
    return {
        "id": "0f9a1b2c-0000-4000-8000-000000000001",
        "profile_id": None,
        "baseline_repo": "https://github.com/vllm-project/vllm",
        "baseline_commit": "a" * 40,
        "base_image_digest": "sha256:" + "b" * 64,
        "gpu_skus": ["H100_80GB"],
        "workload_trace_sha256": "c" * 64,
        "workload_trace_url": "s3://traces/t.json",
        "sla": {"p99_ttft_ms": 200, "p99_itl_ms": 20, "min_throughput_rps": 1},
        "scoring_config_sha256": None,
        "scoring_config_url": None,
        "allowed_paths": ["vllm/**"],
        "denied_paths": [],
        "status": status,
        "priority_metric": "latency",
        "success_threshold": "20 percent faster",
        "customer_signoff": None,
        "manifest_hash": "sha256:" + "d" * 64,
        "bench": None,
        "engine": None,
        "workload_pool": None,
        "sampling_rule": None,
        "scoring_rule": {"name": "median_e2e_speedup"},
        "submission_fee": {"amount_tao": "0.0005", "recipient": "5Recipient"},
        "created_at": None,
    }


def test_writing_scoring_rule_on_a_non_draft_campaign_raises(monkeypatch):
    from campaign import store

    for status in ("open", "closed"):
        sqls, _ = _patch_db(monkeypatch, _campaign_row(status))
        with pytest.raises(ValueError, match="fixed once a campaign leaves draft"):
            store.set_campaign_scoring_rule(
                _campaign_row(status)["id"],
                {"name": "median_e2e_speedup", "tolerance": 0.5},
            )
        assert not any("UPDATE campaigns" in s for s in sqls)


def test_writing_scoring_rule_on_a_draft_campaign_rehashes_the_manifest(monkeypatch):
    from campaign import store

    sqls, args = _patch_db(monkeypatch, _campaign_row("draft"))
    manifest = store.set_campaign_scoring_rule(
        _campaign_row("draft")["id"], {"name": "median_e2e_speedup", "tolerance": 0.5}
    )
    assert manifest.scoring_rule == {"name": "median_e2e_speedup", "tolerance": 0.5}
    # The stored hash described the old rule and must not survive the write.
    assert manifest.manifest_hash != _campaign_row("draft")["manifest_hash"]
    update = [s for s in sqls if "UPDATE campaigns" in s]
    assert len(update) == 1
    assert "SET scoring_rule = %s, manifest_hash = %s" in update[0]
    assert args[-1][1] == manifest.manifest_hash


def test_the_guard_is_a_plain_check_on_status():
    from campaign import store

    store.assert_scoring_rule_mutable("draft")
    with pytest.raises(ValueError, match="fixed once a campaign leaves draft"):
        store.assert_scoring_rule_mutable("open")
