"""Run one full shadow round for a new long-form campaign, on its GPU host.

Samples the round's workload from the qualified rule the campaign preflight
wrote, builds the bench request with the worker's own builder (the baseline
image doubles as the only candidate), and runs the production harness on it.
The round passes when the harness exits 0, the report's verdict is ``pass``,
the unchanged candidate is scored rather than disqualified, and the baseline
drift stays under the campaign's ceiling.

Nothing here writes to a campaign or database. Exit 0: the round passed.
1: it ran and failed (see summary.json). 2: it could not complete.
3: an engine failed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import bench.main as harness
from bench.lifecycle import EngineError
from bench.sampler import (
    SamplerError,
    build_prompt_formatter,
    compute_sample_seed,
    fetch_hf_row,
    generate_trace,
    sampling_context_for_rule,
)
from bench.validate import sha256_file
from campaign.models import SLA
from worker.round_job import build_round_request

# Load the sibling by path: tests/ops shadows the ops namespace under pytest.
_spec = importlib.util.spec_from_file_location(
    "campaign_preflight", Path(__file__).with_name("campaign_preflight.py")
)
_preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preflight)
check_gpus = _preflight.check_gpus
load_qualified_rule = _preflight.load_qualified_rule
save = _preflight.save
apply_health_timeout = _preflight.apply_health_timeout


CAMPAIGN_ID = "00000000-0000-0000-0000-000000000001"
BLOCK_HASH = "c" * 64


def sample_round(fields, output_dir):
    """Sample the round workload as the controller does, and replay its receipt."""
    rule = fields["sampling_rule"]
    model = fields["bench"]["model"]
    kwargs = {
        "rule": rule,
        "seed_hex": compute_sample_seed(block_hash=BLOCK_HASH, campaign_id=CAMPAIGN_ID),
        "row_fetcher": lambda i: fetch_hf_row(rule, i),
        "prompt_formatter": build_prompt_formatter(
            rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
        ),
        "sample_seed_block": 1,
        "sample_seed_block_hash": BLOCK_HASH,
        "sampling_context": sampling_context_for_rule(
            rule, fields["bench"], fields["engine"]
        ),
    }
    sampled = generate_trace(**kwargs)
    if generate_trace(**kwargs, sampling_receipt=sampled.receipt).body != sampled.body:
        raise SamplerError("receipt replay changed the round trace")
    output_dir.mkdir(parents=True)
    (output_dir / "workload_trace.json").write_bytes(sampled.body)
    save(output_dir / "sampling_receipt.json", sampled.receipt)
    return output_dir / "workload_trace.json"


def round_request(fields, trace_path):
    image = fields["bench"]["baseline_engine_image_digest"]
    return build_round_request(
        {
            "gpu_sku": fields["gpu_skus"][0],
            "sampled_trace_sha256": sha256_file(trace_path),
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


def review(report, drift_ceiling=None):
    """Failures in a finished round report; empty when the shadow round passed."""
    failures = []
    drift = report.get("baseline_drift")
    # Settlement voids a round whose baseline drifts past the campaign ceiling.
    if drift_ceiling is not None and drift is not None and drift > drift_ceiling:
        failures.append(f"baseline drift {drift:.3f} > ceiling {drift_ceiling}")
    if report.get("verdict") != "pass":
        failures.append(f"verdict {report.get('verdict')!r}")
    entries = report.get("entries") or []
    if not entries:
        failures.append("no candidate entry")
    for entry in entries:
        if entry.get("status") != "scored":
            failures.append(
                f"candidate {entry.get('index')} {entry.get('status')}: "
                f"{entry.get('reason')}"
            )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-fields", type=Path, required=True)
    parser.add_argument(
        "--qualified-rule",
        type=Path,
        required=True,
        help="qualification/sampling_rule.json from the campaign preflight",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=_preflight.DEFAULT_HEALTH_TIMEOUT_S,
        help="Seconds each engine may take to become healthy",
    )
    args = parser.parse_args(argv)
    apply_health_timeout(args.health_timeout)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "failed"}
    started = time.monotonic()
    try:
        fields = json.loads(args.campaign_fields.read_text())
        fields["sampling_rule"] = load_qualified_rule(args.qualified_rule, fields)
        state["gpus"] = check_gpus(fields["gpu_skus"][0], fields["bench"]["gpu_count"])
        print("Sampling the round workload from the qualified rule...", flush=True)
        trace_path = sample_round(fields, root / "preview")
        request_path = root / "bench_request.json"
        save(request_path, round_request(fields, trace_path))
        state["trace_sha256"] = sha256_file(trace_path)
        print("Running the shadow round...", flush=True)
        code = harness.run_bench(request_path, root / "round")
        state["harness_exit_code"] = code
        report_path = root / "round" / "bench_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        state["report"] = str(report_path)
        state["verdict"] = report.get("verdict")
        state["baseline_drift"] = report.get("baseline_drift")
        tiers = (report.get("baseline") or {}).get("tier_completion") or {}
        state["baseline_tier_completion_s"] = {
            tier: value.get("completion_s") for tier, value in tiers.items()
        }
        state["entries"] = [
            {k: e.get(k) for k in ("index", "status", "score", "reason")}
            for e in report.get("entries") or []
        ]
        if code not in (0, 1) and not report:
            raise EngineError(f"harness exited {code} without a report")
        failures = ([f"harness exit {code}"] if code else []) + review(
            report, fields["scoring_rule"].get("baseline_drift_ceiling")
        )
        state["failures"] = failures
        state["status"] = "passed" if not failures else "failed_checks"
        for failure in failures:
            print(f"FAILED {failure}", flush=True)
        return 0 if not failures else 1
    except Exception as exc:  # noqa: BLE001 -- keep the failure in the evidence
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(state["error"], file=sys.stderr)
        return 3 if isinstance(exc, EngineError) else 2
    finally:
        state["elapsed_s"] = time.monotonic() - started
        save(root / "summary.json", state)
        print(
            f"Shadow status: {state['status']}; summary: {root / 'summary.json'}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
