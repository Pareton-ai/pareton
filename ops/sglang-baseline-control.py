"""Run the pinned Qwen baseline as its own candidate on an idle 4xRTX5090 host.

No campaign/DB/chain writes or candidate build. Without --sampling-rule this
uses unqualified source rows as a diagnostic, not campaign qualification.
Run from the repository root with PYTHONPATH=. and requirements.txt installed.
"""

import argparse
import hashlib
import json
import logging
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import config
from bench.correctness import BASELINE_INDEX, grade_all
from bench.longform import require_qualification
from bench.main import main as run_bench
from bench.preview_longform import preview
from bench.sampler import parse_sampling_rule
from bench.sla_bench import run_sla_engine
from bench.validate import validate_bench_request_dict
from campaign.models import SLA
from gpu.static_host import REMOTE_LOCK, check_idle_gpu, host_lock
from worker.round_job import build_round_request

logger = logging.getLogger(__name__)

FIELDS = Path(__file__).resolve().parents[1] / (
    "fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json"
)


def likelihood_summary(reports, thresholds):
    """Report both proposed relative bars from the same trusted-scoring results."""
    baseline = reports.get(BASELINE_INDEX)
    candidate = reports.get(0)
    valid_scores = all(
        report is not None and report.num_positions_scored > 0
        for report in (baseline, candidate)
    )
    drop = baseline.mean_logprob - candidate.mean_logprob if valid_scores else None
    engines = {}
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if report is None:
            engines[name] = {"scored": False}
            continue
        scored = report.num_positions_scored > 0
        engines[name] = {
            "scored": scored,
            "verdict": report.verdict,
            "reason": report.reason,
            "num_prompts": report.num_prompts,
            "num_positions_scored": report.num_positions_scored,
            "mean_logprob": report.mean_logprob if scored else None,
            "min_logprob": report.min_logprob if scored else None,
            "quantile_logprob": report.quantile_logprob if scored else None,
            "coverage_ratio": report.coverage_ratio if scored else None,
            "absolute_checks": {
                "mean_logprob_pass": report.mean_logprob
                >= thresholds["min_mean_logprob"]
                if scored
                else None,
                # Production applies min_token_logprob to the configured quantile.
                "token_quantile_pass": report.quantile_logprob
                >= thresholds["min_token_logprob"]
                if scored
                else None,
                "coverage_pass": report.coverage_ratio
                >= thresholds["min_coverage_ratio"]
                if scored
                else None,
            },
        }
    return {
        "thresholds": thresholds,
        "engines": engines,
        "baseline_reference_passed": baseline is not None
        and baseline.verdict == "pass",
        "mean_logprob_drop": drop,
        "relative_drop_checks": {
            str(limit): None if drop is None else drop <= limit
            for limit in sorted({1.5, 2.5, thresholds["max_mean_logprob_drop"]})
        },
    }


@contextmanager
def endpoint_stress(root, temperatures, thresholds):
    """Override only this diagnostic process; production trace/API stay unchanged.

    Keep the sampled receipt as the input source. Actual role-specific generation
    settings are recorded separately and in every warmup/measured SLA evidence row.
    """
    low, high = temperatures
    reports = {}
    (root / "temperature_overrides.json").write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "baseline": low,
                "baseline-drift": low,
                "candidate-0": high,
                "seed": 0,
                "applies_to": "all warmups and measured repetitions",
                "source_trace": "inputs/workload_trace.json",
                "performance_comparison_valid": False,
            },
            indent=2,
        )
        + "\n"
    )

    def replay(base_url, *, role, requests, **kwargs):
        if role not in ("baseline", "baseline-drift", "candidate-0"):
            raise ValueError(f"unexpected diagnostic engine role: {role}")
        temperature = high if role == "candidate-0" else low
        overridden = [
            replace(
                request, sampling=replace(request.sampling, temperature=temperature)
            )
            for request in requests
        ]
        logger.info(
            "Endpoint stress: %s temperature=%s prompts=%s",
            role,
            temperature,
            len(overridden),
        )
        return run_sla_engine(base_url, role=role, requests=overridden, **kwargs)

    def grade(*args, **kwargs):
        result = grade_all(*args, **kwargs)
        reports.update(result)
        (root / "endpoint_correctness.json").write_text(
            json.dumps(
                {
                    "baseline"
                    if key == BASELINE_INDEX
                    else f"candidate-{key}": value.to_dict()
                    for key, value in result.items()
                },
                indent=2,
            )
            + "\n"
        )
        return result

    try:
        with (
            patch("bench.main.run_sla_engine", replay),
            patch("bench.main.grade_all", grade),
        ):
            yield
    finally:
        summary = likelihood_summary(reports, thresholds)
        (root / "likelihood_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)


def prepare_request(fields, trace_path):
    image = fields["bench"]["baseline_engine_image_digest"]
    request = build_round_request(
        {
            "gpu_sku": "RTX5090",
            "sampled_trace_sha256": "sha256:"
            + hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "scoring_rule": fields["scoring_rule"],
        },
        SimpleNamespace(
            bench=fields["bench"], engine=fields["engine"], sla=SLA(**fields["sla"])
        ),
        [
            {"role": role, "engine_image_ref": image}
            for role in ("baseline", "challenger")
        ],
        task_id=str(uuid4()),
        trace_path=str(trace_path),
    )
    request["sla_bench"]["repetitions"] = 3
    validate_bench_request_dict(request)
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling-rule", type=Path)
    parser.add_argument("--block-hash", default="c" * 64)
    parser.add_argument(
        "--temperature-endpoints",
        action="store_true",
        help="Stress test: both baseline runs at the lower endpoint, candidate at the upper endpoint. Not a performance comparison.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fields = json.loads(FIELDS.read_text())
    if args.sampling_rule:
        rule = parse_sampling_rule(json.loads(args.sampling_rule.read_text()))
        source_rule = {
            key: value
            for key, value in rule.items()
            if key not in ("qualification", "eligible_row_indices")
        }
        if source_rule != parse_sampling_rule(fields["sampling_rule"]):
            raise ValueError("qualified rule must match the current campaign fixture")
        fields["sampling_rule"] = rule
        require_qualification(
            fields["sampling_rule"], fields["bench"], fields["engine"]
        )
    else:
        print(
            "Unqualified source pool: diagnostic only, not launch qualification.",
            flush=True,
        )

    lock = Path(REMOTE_LOCK)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with host_lock(lock):
        check_idle_gpu()  # Never stop other workloads or clean their containers.
        preview(
            fields=fields,
            output_dir=root / "inputs",
            campaign_id=UUID("00000000-0000-0000-0000-000000000001"),
            seed_block=1,
            block_hash=args.block_hash,
        )
        request = prepare_request(fields, root / "inputs/workload_trace.json")
        request_path = root / "bench_request.json"
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        print(
            "Running baseline, repeatability baseline, baseline-as-candidate, scorer.",
            flush=True,
        )
        bench_args = [
            "--request",
            str(request_path),
            "--output-dir",
            str(root / "output"),
        ]
        if args.temperature_endpoints:
            with endpoint_stress(
                root,
                fields["sampling_rule"]["temperature_range"],
                request["correctness"]["thresholds"],
            ):
                code = run_bench(bench_args)
        else:
            code = run_bench(bench_args)
        if code:
            return code

    report = json.loads((root / "output/bench_report.json").read_text())
    comparison = report.get("baseline_drift")
    repeatable = (
        comparison is not None and abs(comparison) <= config.BASELINE_DRIFT_CEILING
    )
    passed = (
        report["verdict"] == "pass"
        and bool(report["entries"])
        and all(entry["status"] == "scored" for entry in report["entries"])
        and repeatable
    )
    summary = {
        "control_passed": passed,
        "temperature_endpoint_stress": args.temperature_endpoints,
        "performance_comparison_valid": not args.temperature_endpoints,
        "qualified_source_pool": args.sampling_rule is not None,
        "initial_baseline_repeatability": comparison,
        "repeatability_ceiling": config.BASELINE_DRIFT_CEILING,
        "entries": [
            {
                key: entry.get(key)
                for key in ("status", "score", "reason", "correctness")
            }
            for entry in report["entries"]
        ],
    }
    (root / "control_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(
        f"Evidence: {root}; one passing control does not establish a false-positive rate."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
