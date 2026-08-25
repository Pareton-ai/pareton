"""Offline unit tests for campaign seed CLI helpers."""

from __future__ import annotations

from uuid import uuid4

import pytest

import config
from campaign import seed
from campaign.seed import (
    DEFAULT_BASE_IMAGE_DIGEST,
    DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST,
    FIXTURE_SAMPLING_RULE,
    main,
    seed_synthetic_campaign,
)

pytestmark = pytest.mark.unit

REAL_BASE = "sha256:" + ("a" * 64)
REAL_ENGINE = "sha256:" + ("d" * 64)


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


def test_default_seed_pins_hf_rows_and_stores_no_trace(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    assert captured["inserts"] == 1
    m = captured["manifest"]
    assert m.workload_trace_url is None
    assert m.workload_trace_sha256 is None
    assert m.sampling_rule is not None
    assert m.sampling_rule["type"] == "hf_rows"
    assert m.sampling_rule["dataset"] == "nebius/SWE-agent-trajectories"
    public = m.to_public_dict()
    assert "workload_trace_url" not in public
    assert public["sampling_rule"]["type"] == "hf_rows"


def test_bench_flags_shape_correctness(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(
        allow_placeholders=True,
        bench_quantization="fp8",
        bench_correctness_num_prompts=16,
        bench_correctness_thresholds={"min_mean_logprob": -3.5},
    )
    bench = captured["manifest"].bench
    assert bench["model"]["quantization"] == "fp8"
    assert bench["correctness"]["num_prompts"] == 16
    assert bench["correctness"]["thresholds"]["min_mean_logprob"] == -3.5
    # An override fills one bar; the rest still come from the seed defaults.
    assert bench["correctness"]["thresholds"]["min_token_logprob"] == -12.0


def test_correctness_thresholds_are_always_pinned(monkeypatch: pytest.MonkeyPatch):
    """The shared scorer's bars live in the manifest, never in the pod's env."""
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    bench = captured["manifest"].bench
    assert bench["model"]["quantization"] is None
    assert set(bench["correctness"]["thresholds"]) == {
        "min_mean_logprob",
        "min_token_logprob",
        "min_token_quantile",
        "min_coverage_ratio",
    }
    assert "num_prompts" not in bench["correctness"]


def test_open_requires_correctness_thresholds():
    """A bar that is not in the manifest can move under a live campaign
    without the manifest hash changing, so opening requires all three."""
    from campaign.seed import require_correctness_thresholds

    require_correctness_thresholds(
        {
            "correctness": {
                "thresholds": {
                    "min_mean_logprob": -4.0,
                    "min_token_logprob": -12.0,
                    "min_token_quantile": 0.001,
                    "min_coverage_ratio": 0.5,
                }
            }
        }
    )
    with pytest.raises(ValueError, match="min_token_logprob"):
        require_correctness_thresholds(
            {"correctness": {"thresholds": {"min_mean_logprob": -4.0}}}
        )
    with pytest.raises(ValueError, match="status=open requires"):
        require_correctness_thresholds({})


def test_coverage_ratio_must_be_a_fraction():
    from campaign.seed import build_seed_bench_spec

    with pytest.raises(ValueError, match="min_coverage_ratio"):
        build_seed_bench_spec(correctness_thresholds={"min_coverage_ratio": 0.0})


def test_token_quantile_must_be_a_fraction():
    """0 is allowed and means the plain minimum; 1 would gate on the maximum."""
    from campaign.seed import build_seed_bench_spec

    assert (
        build_seed_bench_spec(correctness_thresholds={"min_token_quantile": 0.0})[
            "correctness"
        ]["thresholds"]["min_token_quantile"]
        == 0.0
    )
    with pytest.raises(ValueError, match="min_token_quantile"):
        build_seed_bench_spec(correctness_thresholds={"min_token_quantile": 1.0})


def test_bad_sampling_rule_raises_before_insert(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="unsupported sampling_rule.type"):
        seed_synthetic_campaign(
            allow_placeholders=True,
            sampling_rule={"type": "fixed_trace"},
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
    assert captured["manifest"].sampling_rule["type"] == "hf_rows"
    assert captured["manifest"].workload_trace_url is None


def test_main_sampling_rule_flag_wired(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    rc = main(
        [
            "--base-image-digest",
            REAL_BASE,
            "--baseline-engine-image-digest",
            REAL_ENGINE,
            "--sampling-rule-json",
            str(FIXTURE_SAMPLING_RULE),
        ]
    )
    assert rc == 0
    assert captured["manifest"].sampling_rule["dataset"] == (
        "nebius/SWE-agent-trajectories"
    )
    assert captured["manifest"].workload_trace_url is None


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


def test_seed_pins_the_emission_rule_from_config(monkeypatch: pytest.MonkeyPatch):
    """Every seeded campaign carries a pay schedule, signed into the hash."""
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    m = captured["manifest"]
    assert m.emission_rule == {
        "name": "linear_decay",
        "start_weight": config.EMISSION_START_WEIGHT,
        "floor_weight": config.EMISSION_FLOOR_WEIGHT,
        "decay_blocks": config.EMISSION_DECAY_BLOCKS,
    }
    assert m.to_public_dict()["emission_rule"] == m.emission_rule


def test_seed_rejects_an_emission_rule_that_over_commits_the_subnet(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="emission_rule.start_weight must be in"):
        seed_synthetic_campaign(
            allow_placeholders=True, emission_rule={"start_weight": 1.5}
        )
    assert captured["inserts"] == 0


def test_open_requires_an_emission_rule():
    """No rule means the campaign is left out of the vector and pays nobody."""
    from campaign.seed import require_emission_rule

    require_emission_rule({"name": "linear_decay"})
    with pytest.raises(ValueError, match="status=open requires"):
        require_emission_rule(None)


def test_main_emission_flags_wired(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    rc = main(
        [
            "--allow-placeholders",
            "--emission-start-weight",
            "0.25",
            "--emission-decay-blocks",
            "100800",
        ]
    )
    assert rc == 0
    rule = captured["manifest"].emission_rule
    assert rule["start_weight"] == 0.25
    assert rule["decay_blocks"] == 100800
    # An override fills one term; the rest still come from the seed defaults.
    assert rule["floor_weight"] == config.EMISSION_FLOOR_WEIGHT
