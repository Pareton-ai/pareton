"""Qualify the Kimi K3 8xB300 campaign's long-form source pool on its GPU host.

Starts the trusted baseline exactly as a round would (the worker's request
builder and the harness's planned baseline start, with /model and /draft
mounted read-only and the port published on 127.0.0.1). Then it runs
``bench.qualify_longform`` against it. Writes local evidence and a qualified
sampling rule for ``ops/seed-sglang-kimi-k3-b300.sh``; never touches a
database or campaign.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from bench.lifecycle import BenchNetwork, EngineContainer, EngineError, new_run_id
from bench.main import plan_round_starts
from bench.qualify_longform import qualify
from bench.sampler import SamplerError
from bench.validate import sha256_file, validate_bench_request_dict
from bench.weights import stage_weights
from campaign.models import SLA
from worker.round_job import build_round_request

FIXTURE = Path("fixtures/campaigns/sglang_kimi_k3_b300/campaign-fields.json")
KIMI_REPO = "moonshotai/Kimi-K3"

logger = logging.getLogger(__name__)


def check_fields(fields):
    """Refuse anything but an unqualified Kimi K3 B300 fixture."""
    bench = fields["bench"]
    rule = fields["sampling_rule"]
    if bench["model"]["hf_repo"] != KIMI_REPO or not bench.get("draft_model"):
        raise ValueError("expected the Kimi K3 fixture with its pinned DSPARK draft")
    if fields["engine"]["name"] != "sglang":
        raise ValueError("Kimi K3 qualification runs on the SGLang baseline")
    if fields["base_image_digest"] != bench["baseline_engine_image_digest"]:
        raise ValueError("base and baseline image digests must match")
    if "qualification" in rule or "eligible_row_indices" in rule:
        raise ValueError("qualify from the unqualified fixture, not a qualified rule")


def check_gpus(sku, count, runner=subprocess.run):
    result = runner(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    matching = [name for name in names if sku.upper() in name.upper().replace(" ", "")]
    if len(matching) < count or len(matching) != len(names):
        raise EngineError(f"need {count} {sku} GPUs and no others, found {names}")
    return names


def baseline_request(fields, work_dir):
    """The round's bench request; qualification needs only its baseline start."""
    # build_round_request reads a trace; the baseline start does not depend on it.
    trace = work_dir / "placeholder_trace.json"
    trace.write_text(json.dumps({"schema_version": 1, "requests": [{"id": "-"}]}))
    request = build_round_request(
        {
            "gpu_sku": fields["gpu_skus"][0],
            "sampled_trace_sha256": sha256_file(trace),
            "scoring_rule": fields["scoring_rule"],
        },
        SimpleNamespace(
            bench=fields["bench"], engine=fields["engine"], sla=SLA(**fields["sla"])
        ),
        [
            {
                "role": role,
                "engine_image_ref": fields["bench"]["baseline_engine_image_digest"],
            }
            for role in ("baseline", "challenger")
        ],
        task_id=str(uuid4()),
        trace_path=str(trace),
    )
    req = validate_bench_request_dict(request)
    starts = plan_round_starts(
        req.engines, correctness_serve_args=req.correctness.serve_args
    )
    return req, next(start for start in starts if start.kind == "baseline")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-fields", type=Path, default=FIXTURE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=3600,
        help="Seconds to wait for the baseline to load (default 3600)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        fields = json.loads(args.campaign_fields.read_text())
        check_fields(fields)
        print("GPUs:", check_gpus(fields["gpu_skus"][0], fields["bench"]["gpu_count"]))
        req, start = baseline_request(fields, root)
        weights = stage_weights(req.model, token_env=req.hf_token_env).path
        draft = stage_weights(
            req.draft_model, token_env=req.hf_token_env, require_tokenizer=False
        ).path
        engine_ref = fields["bench"]["baseline_engine_image_digest"]
        # qualify_longform binds its endpoint to a published loopback port,
        # which an internal network cannot publish.
        with BenchNetwork(run_id=new_run_id(), internal=False) as network:
            container = EngineContainer(
                spec=start.spec,
                network=network,
                role="qualification-baseline",
                gpu_count=req.hardware.gpu_count,
                weights_dir=weights,
                draft_dir=draft,
                publish_port=True,
                health_timeout_s=args.health_timeout,
                logs_dir=root / "engine_logs",
            )
            with container as handle:
                print(f"Baseline healthy at {handle.base_url}; qualifying.", flush=True)
                qualify(
                    fields=fields,
                    base_url=handle.base_url,
                    container=handle.container_id,
                    engine_ref=engine_ref,
                    output_dir=root / "qualification",
                    pool_size=args.pool_size,
                    repetitions=args.repetitions,
                    concurrency=args.concurrency,
                    timeout=args.timeout,
                )
    except (SamplerError, EngineError, ValueError) as exc:
        print(f"qualification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(root / "qualification" / "sampling_rule.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
