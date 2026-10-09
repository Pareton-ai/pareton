"""Run one full shadow round for a new long-form campaign, on its GPU host.

Samples the round's workload from the qualified rule the campaign preflight
wrote, builds the bench request with the worker's own builder (the baseline
image doubles as the only candidate), and runs the production harness on it.
The round passes when the harness exits 0, the report's verdict is ``pass``
and the unchanged candidate is scored rather than disqualified.

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
from bench.preview_longform import preview
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


def review(report):
    """Failures in a finished round report; empty when the shadow round passed."""
    failures = []
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
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "failed"}
    started = time.monotonic()
    try:
        fields = json.loads(args.campaign_fields.read_text())
        fields["sampling_rule"] = load_qualified_rule(args.qualified_rule, fields)
        state["gpus"] = check_gpus(fields["gpu_skus"][0], fields["bench"]["gpu_count"])
        print("Sampling the round workload from the qualified rule...", flush=True)
        preview(
            fields=fields,
            output_dir=root / "preview",
            campaign_id="00000000-0000-0000-0000-000000000001",
            seed_block=1,
            block_hash="c" * 64,
        )
        trace_path = root / "preview" / "workload_trace.json"
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
        state["entries"] = [
            {k: e.get(k) for k in ("index", "status", "score", "reason")}
            for e in report.get("entries") or []
        ]
        if code not in (0, 1) and not report:
            raise EngineError(f"harness exited {code} without a report")
        failures = ([f"harness exit {code}"] if code else []) + review(report)
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
