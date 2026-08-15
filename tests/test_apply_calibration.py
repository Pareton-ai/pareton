"""Unit tests for apply_campaign_correctness_calibration (mocked DB, no network)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from campaign.store import apply_campaign_correctness_calibration

pytestmark = pytest.mark.unit

CID = uuid4()


def _campaign_row(
    *, status: str = "draft", bench: dict[str, Any] | None = None
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": CID,
        "created_at": now,
        "profile_id": None,
        "baseline_repo": "https://github.com/vllm-project/vllm.git",
        "baseline_commit": "ee0da84ab9e04ac7610e28580af62c365e898389",
        "base_image_digest": "sha256:" + ("a" * 64),
        "gpu_skus": ["H200-SXM-141GB"],
        "workload_trace_sha256": "sha256:" + ("b" * 64),
        "workload_trace_url": "file:///tmp/trace.json",
        "sla": {"p99_ttft_ms": 2000.0, "p99_itl_ms": 50.0},
        "scoring_config_sha256": None,
        "scoring_config_url": None,
        "allowed_paths": ["vllm/"],
        "denied_paths": [],
        "priority_metric": "gpu_hours",
        "success_threshold": ">=10%",
        "status": status,
        "bench": bench
        or {
            "model": {
                "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
                "hf_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
                "dtype": "bfloat16",
                "quantization": None,
                "max_model_len": 8192,
            },
            "baseline_engine_image_digest": "sha256:" + ("c" * 64),
            "gpu_count": 1,
            "correctness": None,
        },
        "engine": None,
        "manifest_hash": "sha256:" + ("d" * 64),
        "customer_signoff": None,
    }


class _FakeCursor:
    def __init__(self, row: dict, n_subs: int) -> None:
        self._row = row
        self._n_subs = n_subs
        self._last_sql = ""
        self.updates: list[tuple] = []

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self._last_sql = " ".join(sql.split())
        self._params = params or ()
        if "UPDATE campaigns" in self._last_sql:
            self.updates.append(self._params)

    def fetchone(self) -> dict | None:
        if "FROM campaigns" in self._last_sql:
            return self._row
        if "COUNT(*)" in self._last_sql:
            return {"n": self._n_subs}
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeConn:
    def __init__(self, cur: _FakeCursor) -> None:
        self._cur = cur

    def cursor(self, cursor_factory=None):  # noqa: ANN001
        return self._cur


def _patch_db(
    monkeypatch: pytest.MonkeyPatch, row: dict, n_subs: int = 0
) -> _FakeCursor:
    cur = _FakeCursor(row, n_subs)

    @contextmanager
    def fake_db():
        yield _FakeConn(cur)

    monkeypatch.setattr("campaign.store.db_connection", fake_db)
    return cur


def test_apply_updates_bench_and_signoff(monkeypatch: pytest.MonkeyPatch):
    cur = _patch_db(monkeypatch, _campaign_row())
    correctness = {
        "num_prompts": 16,
        "max_new_tokens": 64,
        "min_positions_compared": 1024,
        "thresholds": {
            "mean_abs_logprob_diff": 0.05,
            "max_abs_logprob_diff": 0.1,
            "argmax_mismatch_rate": 0.01,
        },
        "calibration": {
            "thresholds": {
                "mean_abs_logprob_diff": 0.05,
                "max_abs_logprob_diff": 0.1,
                "argmax_mismatch_rate": 0.01,
            },
            "fingerprint": {"model_repo": "Qwen/Qwen2.5-7B-Instruct"},
        },
    }
    new_hash = apply_campaign_correctness_calibration(CID, correctness)
    assert new_hash.startswith("sha256:")
    assert len(cur.updates) == 1
    bench_json, mh, signoff, cid = cur.updates[0]
    assert str(cid) == str(CID)
    assert mh == new_hash
    assert (
        bench_json.adapted["correctness"]["calibration"]["thresholds"][
            "mean_abs_logprob_diff"
        ]
        == 0.05
    )
    assert signoff.adapted["approved_manifest_hash"] == new_hash


def test_apply_rejects_non_draft(monkeypatch: pytest.MonkeyPatch):
    _patch_db(monkeypatch, _campaign_row(status="open"))
    with pytest.raises(ValueError, match="must be draft"):
        apply_campaign_correctness_calibration(CID, {"thresholds": {}})


def test_apply_rejects_with_submissions(monkeypatch: pytest.MonkeyPatch):
    _patch_db(monkeypatch, _campaign_row(), n_subs=1)
    with pytest.raises(ValueError, match="submissions"):
        apply_campaign_correctness_calibration(CID, {"thresholds": {}})
