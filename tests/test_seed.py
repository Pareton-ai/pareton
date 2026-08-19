"""Offline unit tests for campaign seed CLI helpers."""

from __future__ import annotations

from uuid import uuid4

import pytest

from campaign import seed
from campaign.seed import (
    DEFAULT_BASE_IMAGE_DIGEST,
    DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST,
    FIXTURE_TRACE,
    _sha256_file,
    main,
    seed_synthetic_campaign,
)

pytestmark = pytest.mark.unit

REAL_BASE = "sha256:" + ("a" * 64)
REAL_ENGINE = "sha256:" + ("d" * 64)
HTTPS_TRACE = "https://example.test/stage0/workload_trace.json"


def _fixture_sha() -> str:
    return _sha256_file(FIXTURE_TRACE)


def _patch_store(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {"manifest": None, "inserts": 0, "profile_data": None}

    monkeypatch.setattr(seed, "list_campaigns", lambda status="open": [])

    def _insert_profile(**kwargs):
        captured["profile_data"] = kwargs.get("data")
        return uuid4()

    monkeypatch.setattr(seed, "insert_profile", _insert_profile)

    def _insert(manifest):
        captured["manifest"] = manifest
        captured["inserts"] += 1
        return str(manifest.campaign_id)

    monkeypatch.setattr(seed, "insert_campaign", _insert)
    return captured


def test_default_file_url_with_placeholders(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    assert captured["inserts"] == 1
    m = captured["manifest"]
    assert m.workload_trace_url == f"file://{FIXTURE_TRACE.resolve()}"
    assert m.workload_trace_sha256 == _fixture_sha()


def test_bench_flags_shape_correctness(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(
        allow_placeholders=True,
        bench_quantization="fp8",
        bench_correctness_num_prompts=16,
        bench_correctness_max_new_tokens=64,
    )
    bench = captured["manifest"].bench
    assert bench["model"]["quantization"] == "fp8"
    assert bench["correctness"] == {"num_prompts": 16, "max_new_tokens": 64}
    # Seed pins sample sizes only, never thresholds.
    assert "thresholds" not in bench["correctness"]


def test_bench_flags_default_to_none(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    bench = captured["manifest"].bench
    assert bench["model"]["quantization"] is None
    assert bench["correctness"] is None


def test_https_url_with_matching_sha(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(
        base_image_digest=REAL_BASE,
        baseline_engine_image_digest=REAL_ENGINE,
        workload_trace_url=HTTPS_TRACE,
        workload_trace_sha256=_fixture_sha(),
    )
    m = captured["manifest"]
    assert m.workload_trace_url == HTTPS_TRACE
    assert m.workload_trace_sha256 == _fixture_sha()


def test_https_without_sha_raises_before_insert(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="requires --workload-trace-sha256"):
        seed_synthetic_campaign(
            base_image_digest=REAL_BASE,
            baseline_engine_image_digest=REAL_ENGINE,
            workload_trace_url=HTTPS_TRACE,
        )
    assert captured["inserts"] == 0


def test_sha_mismatch_raises_before_insert(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="does not match local fixture"):
        seed_synthetic_campaign(
            base_image_digest=REAL_BASE,
            baseline_engine_image_digest=REAL_ENGINE,
            workload_trace_url=HTTPS_TRACE,
            workload_trace_sha256="sha256:" + ("0" * 64),
        )
    assert captured["inserts"] == 0


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://example.test/t.json",
        "/abs/path/t.json",
        "  ",
    ],
)
def test_invalid_scheme_raises_before_insert(
    monkeypatch: pytest.MonkeyPatch, bad_url: str
):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError):
        seed_synthetic_campaign(
            base_image_digest=REAL_BASE,
            baseline_engine_image_digest=REAL_ENGINE,
            workload_trace_url=bad_url,
            workload_trace_sha256=_fixture_sha(),
            allow_placeholders=True,
        )
    assert captured["inserts"] == 0


@pytest.mark.parametrize(
    "base,engine",
    [
        (DEFAULT_BASE_IMAGE_DIGEST, REAL_ENGINE),
        (REAL_BASE, DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST),
        ("SHA256:" + ("B" * 64), REAL_ENGINE),
        (REAL_BASE, "SHA256:" + ("C" * 64)),
    ],
)
def test_placeholder_digest_refused(
    monkeypatch: pytest.MonkeyPatch, base: str, engine: str
):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="placeholder digests refused"):
        seed_synthetic_campaign(
            base_image_digest=base,
            baseline_engine_image_digest=engine,
        )
    assert captured["inserts"] == 0


def test_main_allow_placeholders_smoke(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    rc = main(["--allow-placeholders"])
    assert rc == 0
    assert captured["inserts"] == 1
    assert captured["manifest"].workload_trace_url.startswith("file://")


def test_main_https_flags_wired(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    rc = main(
        [
            "--base-image-digest",
            REAL_BASE,
            "--baseline-engine-image-digest",
            REAL_ENGINE,
            "--workload-trace-url",
            HTTPS_TRACE,
            "--workload-trace-sha256",
            _fixture_sha(),
        ]
    )
    assert rc == 0
    assert captured["manifest"].workload_trace_url == HTTPS_TRACE


def test_seed_pins_the_priority_metric_and_threshold_text(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    m = captured["manifest"]
    assert m.priority_metric == "gpu_hours"
    assert "10%" in m.success_threshold


def test_seed_defaults_to_the_median_e2e_speedup_rule(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    assert captured["manifest"].scoring_rule == {"name": "median_e2e_speedup"}


def test_seed_rejects_an_unknown_scoring_rule(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="scoring_rule.name must be one of"):
        seed_synthetic_campaign(allow_placeholders=True, scoring_rule={"name": "vibes"})
    assert captured["inserts"] == 0


def test_seed_profile_uses_cli_metrics(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(
        allow_placeholders=True,
        priority_metric="latency",
        success_threshold=">=5% p99 ITL at SLA",
    )
    assert captured["profile_data"]["priority_metric"] == "latency"
    assert captured["profile_data"]["success_threshold"] == ">=5% p99 ITL at SLA"
    assert captured["manifest"].priority_metric == "latency"
    assert captured["manifest"].success_threshold == ">=5% p99 ITL at SLA"


def test_seed_defaults_to_draft_single_hopper_sku(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    m = captured["manifest"]
    assert m.status == "draft"
    assert m.gpu_skus == ["H200-SXM-141GB"]
    assert captured["profile_data"]["hardware"] == ["H200-SXM-141GB"]


def test_seed_gpu_skus_and_status_override(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(
        allow_placeholders=True,
        gpu_skus=["H200-SXM-141GB", "B200"],
        status="draft",
    )
    m = captured["manifest"]
    assert m.status == "draft"
    assert m.gpu_skus == ["H200-SXM-141GB", "B200"]
    assert captured["profile_data"]["hardware"] == ["H200-SXM-141GB", "B200"]


def test_seed_rejects_empty_gpu_skus(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="gpu_skus must contain"):
        seed_synthetic_campaign(allow_placeholders=True, gpu_skus=["  ", ""])
    assert captured["inserts"] == 0


def test_seed_rejects_invalid_status(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="status must be one of"):
        seed_synthetic_campaign(allow_placeholders=True, status="live")
    assert captured["inserts"] == 0


def test_draft_seed_ignores_existing_open(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)

    class _Existing:
        campaign_id = "already-open"

    monkeypatch.setattr(seed, "list_campaigns", lambda status="open": [_Existing()])
    seed_synthetic_campaign(allow_placeholders=True, status="draft")
    assert captured["inserts"] == 1
    assert captured["manifest"].status == "draft"


def test_open_seed_short_circuits_without_force(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)

    class _Existing:
        campaign_id = "already-open"

    monkeypatch.setattr(seed, "list_campaigns", lambda status="open": [_Existing()])
    cid = seed_synthetic_campaign(allow_placeholders=True, status="open")
    assert cid == "already-open"
    assert captured["inserts"] == 0


def test_main_gpu_skus_status_wired(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    rc = main(
        [
            "--allow-placeholders",
            "--gpu-skus",
            "H200-SXM-141GB",
            "--status",
            "draft",
        ]
    )
    assert rc == 0
    assert captured["manifest"].status == "draft"
    assert captured["manifest"].gpu_skus == ["H200-SXM-141GB"]


def test_no_bench_seed_omits_bench(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True, no_bench=True)
    m = captured["manifest"]
    assert m.bench is None
    assert m.to_public_dict()["bench"] is None


def test_main_no_bench_wired(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    rc = main(["--allow-placeholders", "--no-bench"])
    assert rc == 0
    assert captured["manifest"].bench is None
