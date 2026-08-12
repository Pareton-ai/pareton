"""Unit tests for bench-claim sampling + finality infra failure."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from worker.bench_job import (
    BenchInfraError,
    campaign_uses_dynamic_sampling,
    realize_submission_sample,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRACE = ROOT / "fixtures" / "campaigns" / "synthetic_v0" / "workload_trace.json"


def _sha_file(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_campaign_uses_dynamic_sampling_flags():
    assert campaign_uses_dynamic_sampling({}) is False
    assert campaign_uses_dynamic_sampling({"sampling_rule": {"type": "uniform_index"}})
    assert campaign_uses_dynamic_sampling(
        {
            "workload_pool": [
                {"sha256": "sha256:" + ("a" * 64), "url": "u1"},
                {"sha256": "sha256:" + ("b" * 64), "url": "u2"},
            ]
        }
    )


def test_realize_sample_writes_receipt(tmp_path: Path):
    sha = _sha_file(SAMPLE_TRACE)
    pool = [
        {"sha256": sha, "url": f"file://{SAMPLE_TRACE.resolve()}"},
        {"sha256": "sha256:" + ("b" * 64), "url": "file:///other.json"},
    ]
    recorded: dict = {}

    def _record(**kwargs):
        recorded.update(kwargs)

    row = {
        "id": str(uuid4()),
        "campaign_id": str(uuid4()),
        "patch_hash": "aa" * 32,
        "commit_block": 1000,
        "workload_trace_sha256": sha,
        "workload_trace_url": f"file://{SAMPLE_TRACE.resolve()}",
        "workload_pool": pool,
        "sampling_rule": {"type": "uniform_index", "seed_block_offset": 10},
    }
    out = realize_submission_sample(
        row,
        block_hash_fn=lambda _b: "cc" * 32,
        record_sample_fn=_record,
    )
    assert out["sha256"] in {pool[0]["sha256"], pool[1]["sha256"]}
    assert recorded["sample_seed_block"] == 1010
    assert recorded["sampling_receipt"]["index"] in (0, 1)
    assert row["sampled_trace_sha256"] == out["sha256"]


def test_realize_sample_infra_on_missing_block_hash():
    row = {
        "id": str(uuid4()),
        "campaign_id": str(uuid4()),
        "patch_hash": "aa" * 32,
        "commit_block": 1000,
        "workload_trace_sha256": "sha256:" + ("a" * 64),
        "workload_trace_url": "file:///t.json",
        "workload_pool": [
            {"sha256": "sha256:" + ("a" * 64), "url": "file:///t.json"},
            {"sha256": "sha256:" + ("b" * 64), "url": "file:///u.json"},
        ],
        "sampling_rule": {"type": "uniform_index", "seed_block_offset": 10},
    }

    def _boom(_block: int) -> str:
        raise BenchInfraError("sample_seed_block_unavailable", "pruned")

    with pytest.raises(BenchInfraError) as ei:
        realize_submission_sample(row, block_hash_fn=_boom, record_sample_fn=lambda **_: None)
    assert ei.value.code == "sample_seed_block_unavailable"


def test_realize_sample_infra_on_empty_hash():
    row = {
        "id": str(uuid4()),
        "campaign_id": str(uuid4()),
        "patch_hash": "aa" * 32,
        "commit_block": 1000,
        "workload_trace_sha256": "sha256:" + ("a" * 64),
        "workload_trace_url": "file:///t.json",
        "sampling_rule": {"type": "uniform_index", "seed_block_offset": 5},
    }
    with pytest.raises(BenchInfraError) as ei:
        realize_submission_sample(
            row,
            block_hash_fn=lambda _b: "",
            record_sample_fn=lambda **_: None,
        )
    assert ei.value.code == "sample_seed_block_unavailable"
