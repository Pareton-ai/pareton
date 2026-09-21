"""Run the pinned Qwen baseline as its own candidate on an idle 4xRTX5090 host.

No campaign/DB/chain writes or candidate build. Without --sampling-rule this
uses unqualified source rows as a diagnostic, not campaign qualification.
Run from the repository root with PYTHONPATH=. and requirements.txt installed.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import config
from bench.longform import require_qualification
from bench.main import main as run_bench
from bench.preview_longform import preview
from bench.sampler import parse_sampling_rule
from bench.validate import validate_bench_request_dict
from campaign.models import SLA
from gpu.static_host import REMOTE_LOCK, check_idle_gpu, host_lock
from worker.round_job import build_round_request

FIELDS = Path(__file__).resolve().parents[1] / (
    "fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json"
)


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
        code = run_bench(
            ["--request", str(request_path), "--output-dir", str(root / "output")]
        )
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
