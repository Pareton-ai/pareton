"""Unit tests for bench-claim sampling + finality infra failure."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from worker.bench_job import (
    BenchInfraError,
    campaign_calibration_fingerprint,
    campaign_uses_dynamic_sampling,
    realize_submission_sample,
)

pytestmark = pytest.mark.unit

HF_RULE = {
    "type": "hf_rows",
    "seed_block_offset": 1,
    "dataset": "nebius/SWE-agent-trajectories",
    "revision": "deadbeef" * 5,
    "config": "default",
    "split": "train",
    "n_rows": 8,
    "n_prompts": 3,
    "max_tokens": 128,
    "algo_version": 1,
}


def _user_row(text: str) -> dict:
    return {"trajectory": [{"role": "user", "text": text}]}


def _fetcher():
    rows = [_user_row(f"prompt-{i}") for i in range(8)]

    def fetch(idx: int) -> dict:
        return rows[idx]

    return fetch


def test_campaign_uses_dynamic_sampling_flags():
    assert campaign_uses_dynamic_sampling({}) is False
    assert (
        campaign_uses_dynamic_sampling({"sampling_rule": {"type": "uniform_index"}})
        is False
    )
    assert campaign_uses_dynamic_sampling({"sampling_rule": HF_RULE})
    assert (
        campaign_uses_dynamic_sampling(
            {
                "workload_pool": [
                    {"sha256": "sha256:" + ("a" * 64), "url": "u1"},
                    {"sha256": "sha256:" + ("b" * 64), "url": "u2"},
                ]
            }
        )
        is False
    )


def test_realize_sample_writes_receipt(tmp_path: Path):
    recorded: dict = {}
    uploaded: dict = {}

    def _record(**kwargs):
        recorded.update(kwargs)

    def _upload(*, campaign_id: str, body: bytes, sha256: str) -> str:
        dest = tmp_path / f"{sha256.split(':')[1][:12]}.json"
        dest.write_bytes(body)
        uploaded["sha256"] = sha256
        uploaded["bytes"] = body
        return f"file://{dest.resolve()}"

    row = {
        "id": str(uuid4()),
        "campaign_id": str(uuid4()),
        "patch_hash": "aa" * 32,
        "commit_block": 1000,
        "workload_trace_sha256": "sha256:" + ("a" * 64),
        "workload_trace_url": "file:///placeholder.json",
        "sampling_rule": HF_RULE,
    }
    out = realize_submission_sample(
        row,
        block_hash_fn=lambda _b: "cc" * 32,
        record_sample_fn=_record,
        row_fetcher=_fetcher(),
        upload_trace_fn=_upload,
    )
    assert out["sha256"] == uploaded["sha256"]
    assert recorded["sample_seed_block"] == 1001
    assert recorded["sampling_receipt"]["type"] == "hf_rows"
    assert recorded["sampling_receipt"]["row_indices"]
    assert row["sampled_trace_sha256"] == out["sha256"]
    assert row["workload_trace_url"].startswith("file://")


def test_realize_sample_infra_on_missing_block_hash():
    row = {
        "id": str(uuid4()),
        "campaign_id": str(uuid4()),
        "patch_hash": "aa" * 32,
        "commit_block": 1000,
        "workload_trace_sha256": "sha256:" + ("a" * 64),
        "workload_trace_url": "file:///t.json",
        "sampling_rule": HF_RULE,
    }

    def _boom(_block: int) -> str:
        raise BenchInfraError("sample_seed_block_unavailable", "pruned")

    with pytest.raises(BenchInfraError) as ei:
        realize_submission_sample(
            row,
            block_hash_fn=_boom,
            record_sample_fn=lambda **_: None,
            row_fetcher=_fetcher(),
            upload_trace_fn=lambda **_: "file:///x",
        )
    assert ei.value.code == "sample_seed_block_unavailable"


def test_realize_sample_infra_on_empty_hash():
    row = {
        "id": str(uuid4()),
        "campaign_id": str(uuid4()),
        "patch_hash": "aa" * 32,
        "commit_block": 1000,
        "workload_trace_sha256": "sha256:" + ("a" * 64),
        "workload_trace_url": "file:///t.json",
        "sampling_rule": {**HF_RULE, "seed_block_offset": 5},
    }
    with pytest.raises(BenchInfraError) as ei:
        realize_submission_sample(
            row,
            block_hash_fn=lambda _b: "",
            record_sample_fn=lambda **_: None,
            row_fetcher=_fetcher(),
            upload_trace_fn=lambda **_: "file:///x",
        )
    assert ei.value.code == "sample_seed_block_unavailable"


def test_fingerprint_omits_trace_sha256_and_binds_sampling_pin():
    bench = {
        "model": {
            "hf_repo": "zai-org/GLM-5.2-FP8",
            "hf_revision": "abc",
            "dtype": "auto",
            "quantization": "fp8",
            "max_model_len": 131072,
        },
        "baseline_engine_image_digest": "sha256:" + ("a" * 64),
        "gpu_count": 8,
        "serve_args": [],
    }
    row = {
        "workload_trace_sha256": "sha256:" + ("f" * 64),
        "sampling_rule": HF_RULE,
    }
    fp = campaign_calibration_fingerprint(bench, row)
    assert "trace_sha256" not in fp
    assert fp["sampling_dataset"] == HF_RULE["dataset"]
    assert fp["sampling_revision"] == HF_RULE["revision"]
    assert fp["sampling_n_prompts"] == 3
    assert fp["sampling_algo_version"] == 1
