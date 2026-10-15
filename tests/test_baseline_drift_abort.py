import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_concurrency import timings
from test_natural_stop_reproducibility import workload

from bench import main as bm
from bench.concurrency import tier_completion_metrics
from bench.lifecycle import BaselineDriftError
from bench.output import OutputLayout
from bench.schemas import BenchRequest
from bench.sla_bench import EngineReplay

pytestmark = pytest.mark.unit


def run(tmp_path, monkeypatch, *, ceiling, drift_2k_s):
    """Run a round whose drift run's 2k tier takes ``drift_2k_s`` against 80s."""
    req, trace = workload(ignore_eos=True)
    req.baseline_drift_ceiling = ceiling
    layout = OutputLayout(tmp_path)
    layout.prepare()
    calls = []

    class Provider:
        @contextmanager
        def start(self, start, phase):
            yield start.role

    def replay(url, *, requests, **kwargs):
        calls.append(url)
        rows = []
        for rep in range(1, 4):
            rep_rows = [
                {
                    "request_id": r.id,
                    "rep": rep,
                    "input_length_group": r.input_length_group,
                    "group_start_offset_ms": 0,
                    "admission_offset_ms": 0,
                    "protocol_completion_offset_ms": 1000
                    * (
                        drift_2k_s
                        if url == "baseline-drift" and r.input_length_group == "2k"
                        else 80
                    ),
                    "completion_tokens": 2,
                    "finish_reason": "length",
                }
                for r in requests
            ]
            path = layout.sla_bench_dir / url / f"rep_{rep}" / "requests.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(row) for row in rep_rows))
            rows.extend(rep_rows)
        tiers = tier_completion_metrics(rows, 3)
        result = SimpleNamespace(
            role=url,
            timings=timings(),
            tier_completion=tiers,
            cross_rep_variance={"tier_completion_max_rel_range": 0.0},
        )
        return EngineReplay(result, {}, {})

    monkeypatch.setattr(bm, "run_sla_engine", replay)
    monkeypatch.setattr(
        bm, "baseline_prompt_drops", lambda trace, replay, *, dropped: dropped
    )
    monkeypatch.setattr(
        "bench.workload_preflight.validate_engine_workload", lambda *a, **kw: None
    )
    return calls, lambda: bm.run_round(
        req=req, provider=Provider(), prompts=[], trace=trace, layout=layout
    )


def test_drift_over_the_ceiling_stops_before_any_candidate(tmp_path, monkeypatch):
    calls, go = run(tmp_path, monkeypatch, ceiling=0.1, drift_2k_s=96)
    with pytest.raises(BaselineDriftError) as exc:
        go()
    # 2k ran 20% slower on the second baseline run.
    assert exc.value.drift == pytest.approx(0.2)
    assert exc.value.error_role == "baseline_drift"
    assert calls == ["baseline", "baseline-drift"]


@pytest.mark.parametrize("ceiling", [None, 0.25])
def test_drift_within_the_ceiling_or_unset_runs_the_candidates(
    tmp_path, monkeypatch, ceiling
):
    calls, go = run(tmp_path, monkeypatch, ceiling=ceiling, drift_2k_s=96)
    baseline, drift, runs, _ = go()
    assert calls == ["baseline", "baseline-drift", "candidate-0"]
    # The early check is the comparison the report carries to settlement.
    req, _ = workload(ignore_eos=True)
    assert bm.baseline_drift(req, baseline, drift) == pytest.approx(0.2)
    assert len(runs) == 1


def test_engine_error_report_carries_the_drift(tmp_path):
    req, _ = workload(ignore_eos=True)
    layout = OutputLayout(tmp_path)
    layout.prepare()
    env = SimpleNamespace(
        to_dict=lambda: {
            "gpu": [],
            "driver_version": "",
            "cuda_version": "",
            "docker_version": "",
            "harness_version": "",
            "hostname_hash": "",
        }
    )
    bm._write_engine_error_report(
        layout=layout,
        req=req,
        request_raw=b"{}",
        env=env,
        started_at="t0",
        exc=BaselineDriftError(0.18, 0.1),
        model_weights_sha256="sha256:" + "0" * 64,
    )
    report = json.loads(layout.report_path.read_text())
    assert report["error_role"] == "baseline_drift"
    assert report["baseline_drift"] == 0.18
    assert report["error"] == "baseline drift 0.1800 exceeds ceiling 0.1"


@pytest.mark.parametrize("bad", [True, "0.1", 0, -0.1, float("inf")])
def test_request_rejects_a_bad_drift_ceiling(bad):
    req, _ = workload()
    raw = json.loads(json.dumps(req.to_dict(), default=str))
    raw["baseline_drift_ceiling"] = bad
    with pytest.raises(ValueError, match="baseline_drift_ceiling"):
        BenchRequest.from_dict(raw)
