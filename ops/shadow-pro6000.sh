#!/usr/bin/env bash
# Run the full preview/request/shadow sequence under one nohup controller.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${PRO6000_RUN_DIR:?Restore the GPU run environment first}"
: "${PRO6000_FIELDS:?}"
if [[ ! -f "$PRO6000_RUN_DIR/step2.exit-code" ]] || [[ "$(cat "$PRO6000_RUN_DIR/step2.exit-code")" != 0 ]]; then
  echo 'Step 2 must finish successfully before starting the shadow sequence.' >&2
  exit 1
fi
mkdir "$PRO6000_RUN_DIR/step3.lock" || {
  echo 'Step 3 already claimed this run directory; inspect its PID, logs and exit code.' >&2
  exit 1
}
printf '%s\n' "$$" > "$PRO6000_RUN_DIR/step3.pid"
finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$PRO6000_RUN_DIR/step3.exit-code.tmp"
  mv "$PRO6000_RUN_DIR/step3.exit-code.tmp" "$PRO6000_RUN_DIR/step3.exit-code"
  if (( rc != 0 )); then
    printf 'Step 3 stopped (exit %s); inspect step3.log and preserve all evidence.\n' "$rc" >&2
  else
    echo 'Step 3 completed; review shadow/bench_report.json and evidence before seeding.'
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u -m ops.pro6000_preflight
echo 'Generating the qualified C4 preview...'
python -u -m bench.preview_longform \
  --campaign-fields "$PRO6000_FIELDS" \
  --sampling-rule "$PRO6000_RUN_DIR/qualification/sampling_rule.json" \
  --output-dir "$PRO6000_RUN_DIR/preview" 2>&1 | tee "$PRO6000_RUN_DIR/preview.log"
echo 'Building the worker-derived shadow request...'
python -u - <<'PYTHON' 2>&1 | tee "$PRO6000_RUN_DIR/prepare-request.log"
import json, os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from bench.longform import require_qualification
from bench.sampler import parse_sampling_rule
from bench.validate import load_workload_trace, sha256_file
from campaign.models import SLA
from worker.round_job import build_round_request

root = Path(os.environ["PRO6000_RUN_DIR"]).resolve()
fields = json.loads(Path(os.environ["PRO6000_FIELDS"]).read_text())
rule = parse_sampling_rule(json.loads((root / "qualification/sampling_rule.json").read_text()))
require_qualification(rule, fields["bench"], fields["engine"])
assert rule["algo_version"] == 5 and rule["request_concurrency"] == 4
assert rule["n_prompts"] == 32 and rule["request_timeout_s"] == 600
assert fields["patch_visibility"] == {"mode": "private"}
trace = root / "preview/workload_trace.json"
trace_hash = sha256_file(trace)
parsed_trace = load_workload_trace(trace, expected_sha256=trace_hash)
assert len(parsed_trace.requests) == 32
assert parsed_trace.meta.sampling["request_concurrency"] == 4
campaign = SimpleNamespace(bench=fields["bench"], engine=fields["engine"],
                           sla=SLA.from_dict(fields["sla"]))
engine_ref = fields["bench"]["baseline_engine_image_digest"]
request = build_round_request(
    {"gpu_sku": fields["gpu_skus"][0], "sampled_trace_sha256": trace_hash,
     "scoring_rule": fields["scoring_rule"]}, campaign,
    [{"role": "baseline", "engine_image_ref": engine_ref},
     {"role": "challenger", "engine_image_ref": engine_ref}],
    task_id=str(uuid4()), trace_path=str(trace),
)
(root / "bench_request.json").write_text(json.dumps(request, indent=2) + "\n")
PYTHON
unset PARETON_BENCH_ENGINE_CACHE_DIR
echo 'Running the v5 C4 shadow round...'
python -u -m ops.pro6000_model_volume --request "$PRO6000_RUN_DIR/bench_request.json" \
  --output-dir "$PRO6000_RUN_DIR/shadow" 2>&1 | tee "$PRO6000_RUN_DIR/shadow.log"
jq '{verdict, entries, error}' "$PRO6000_RUN_DIR/shadow/bench_report.json"
