"""Scorer-only smoke for a long-context campaign, run on its GPU host first.

Starts the campaign's scorer exactly as a round would (the worker's request
builder and the harness's planned scorer start, with ``/model`` and any
``/draft`` mounted read-only) and teacher-forces the longest prompts the
campaign can sample, each followed by a ``max_tokens`` continuation. That is
the largest sequence a round can ever send the scorer, so a memory setting
that survives it cannot fail on length in a round.

1. **Prepare (CPU).** Scan the prepared campaign's rows for its highest input
   tier, keep one prompt per session (a session's later calls only extend its
   earlier ones), and take the longest. Each continuation is the pinned
   follow-up text repeated to exactly ``max_tokens`` tokens: memory and
   coverage depend on length, not on what the text says. ``--prepare-only``
   stops here and needs no GPU.
2. **Capacity.** Before sending anything, the scorer's ``max_req_input_len``
   must exceed the longest prompt plus continuation. SGLang rejects a longer
   input instead of truncating it, so this is the check a round would fail
   only after every candidate had run.
3. **Scoring.** Every sequence is scored ``--repetitions`` times through the
   harness's own scoring path, within the rounds' 300 s request timeout, and
   must come back with every continuation position scored.

The scorer's serving arguments are part of the qualification contract, so run
this before qualifying. Nothing here writes to a campaign or database.
Exit 0: passed. 1: a check failed (see summary.json). 2: the run could not
complete. 3: the scorer failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import config
from bench.correctness import (
    CapturedOutput,
    probe_logprob_capability,
    score_captured_output,
)
from bench.http import get_json
from bench.lifecycle import EngineError
from bench.longform import (
    candidate_for_row,
    digest,
    length_groups,
    sampling_context_for_campaign,
)
from bench.main import _EngineProvider, plan_round_starts
from bench.phases import BenchPhase
from bench.sampler import (
    PromptRenderError,
    build_prompt_formatter,
    fetch_hf_row,
    parse_sampling_rule,
)
from bench.validate import sha256_file, validate_bench_request_dict
from bench.weights import stage_weights
from campaign.models import SLA
from worker.round_job import build_round_request

DEFAULT_HEALTH_TIMEOUT_S = float(os.environ.get("PARETON_BENCH_HEALTH_TIMEOUT_S", 3600))
# bench.main grades with grade_all's default request timeout.
ROUND_SCORER_TIMEOUT_S = 300.0
# The scorer re-tokenizes prompt + continuation, which can merge a few tokens.
MIN_SPAN_RATIO = 0.85
FATAL_LOG = re.compile(r"out of memory|CUDA error", re.IGNORECASE)


class CheckFailed(Exception):
    """A smoke check ran and failed (exit 1, not an infrastructure error)."""


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


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


def inputs_digest(fields):
    """What the sequences depend on, so a scorer-args change keeps them valid."""
    return digest(
        {
            "sampling_rule": fields["sampling_rule"],
            "model": fields["bench"]["model"],
            "engine": fields["engine"],
        }
    )


def top_tier(rule):
    return length_groups(rule["n_prompts"], rule.get("input_tiers"))[-1]


def continuation(formatter, text, tokens):
    """The shortest prefix of ``text`` repeated that encodes to ``tokens`` tokens."""
    source = text.strip() + "\n\n"
    while len(formatter.encode(source)) < tokens + 1:
        source += source
    lo, hi = 1, len(source)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(formatter.encode(source[:mid])) >= tokens:
            hi = mid
        else:
            lo = mid + 1
    result = source[:lo]
    if len(formatter.encode(result)) != tokens:
        raise ValueError(f"cannot build a {tokens}-token continuation")
    return result


def select_sequences(fields, count, *, formatter=None, row_fetcher=None):
    """The ``count`` longest highest-tier prompts, one per session, with continuations."""
    rule = parse_sampling_rule(fields["sampling_rule"])
    model = fields["bench"]["model"]
    formatter = formatter or build_prompt_formatter(
        rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
    )
    row_fetcher = row_fetcher or (lambda i: fetch_hf_row(rule, i))
    context = sampling_context_for_campaign(fields["bench"], fields["engine"])
    tier = top_tier(rule)
    best = {}
    for index in range(rule["n_rows"]):
        if index and index % 500 == 0:
            print(f"Scanned {index}/{rule['n_rows']} rows", flush=True)
        try:
            item = candidate_for_row(
                index, row_fetcher(index), formatter, rule, context
            )
        except PromptRenderError:
            continue
        if item is None or item["input_length_group"] != tier["name"]:
            continue
        key = item.get("session_id") or item["input_ids_sha256"]
        if key not in best or item["input_tokens"] > best[key]["input_tokens"]:
            best[key] = item
    chosen = sorted(best.values(), key=lambda i: (-i["input_tokens"], i["row_index"]))
    if len(chosen) < count:
        raise CheckFailed(
            f"only {len(chosen)} distinct {tier['name']} prompts, need {count}"
        )
    output = continuation(formatter, rule["followup_prompt"], rule["max_tokens"])
    return tier["name"], [
        {
            "id": f"smoke-{i}",
            "row_index": item["row_index"],
            "session_id": item.get("session_id"),
            "input_tokens": item["input_tokens"],
            "continuation_tokens": rule["max_tokens"],
            "total_tokens": item["input_tokens"] + rule["max_tokens"],
            "prompt": item["prompt"],
            "output_text": output,
        }
        for i, item in enumerate(chosen[:count])
    ]


def scorer_request(fields, work_dir):
    """The round's bench request and its planned scorer start."""
    # build_round_request reads a trace; the engine starts do not depend on it.
    trace = work_dir / "placeholder_trace.json"
    trace.write_text(json.dumps({"schema_version": 1, "requests": [{"id": "-"}]}))
    image = fields["bench"]["baseline_engine_image_digest"]
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
            {"role": role, "engine_image_ref": image}
            for role in ("baseline", "challenger")
        ],
        task_id=str(uuid4()),
        trace_path=str(trace),
    )
    req = validate_bench_request_dict(request)
    starts = plan_round_starts(
        req.engines, correctness_serve_args=req.correctness.serve_args
    )
    return req, next(s for s in starts if s.kind == "scorer")


def capacity(info, sequences):
    """The scorer's resolved limits; fails when the longest sequence cannot fit."""
    flat = {**info, **(info.get("server_args") or {})}
    limit = flat.get("max_req_input_len")
    if type(limit) is not int or limit < 1:
        raise EngineError("scorer server info lacks max_req_input_len")
    longest = max(s["total_tokens"] for s in sequences)
    result = {
        "max_req_input_len": limit,
        "max_total_num_tokens": flat.get("max_total_num_tokens"),
        "attn_dcp_size": flat.get("attn_dcp_size"),
        "max_running_requests": flat.get("max_running_requests"),
        "mem_fraction_static": flat.get("mem_fraction_static"),
        "chunked_prefill_size": flat.get("chunked_prefill_size"),
        "longest_sequence_tokens": longest,
        "headroom_tokens": limit - longest,
    }
    # SGLang rejects inputs at or above max_req_input_len.
    if longest >= limit:
        raise CheckFailed(
            f"scorer accepts inputs below {limit} tokens, longest sequence is {longest}"
        )
    return result


def score_all(url, sequences, *, repetitions, timeout):
    rows, failures = [], []
    for rep in range(repetitions):
        for seq in sequences:
            started = time.monotonic()
            scores, span, _ = score_captured_output(
                url,
                CapturedOutput(
                    request_id=seq["id"],
                    prompt=seq["prompt"],
                    output_text=seq["output_text"],
                    completion_tokens=seq["continuation_tokens"],
                    ignore_eos=True,
                ),
                request_timeout_s=timeout,
                engine_name="sglang",
            )
            logprobs = [s.logprob for s in scores]
            row = {
                "request_id": seq["id"],
                "repetition": rep + 1,
                "input_tokens": seq["input_tokens"],
                "span_positions": span,
                "scored_positions": len(logprobs),
                "elapsed_s": time.monotonic() - started,
                "finite": all(math.isfinite(v) for v in logprobs),
            }
            rows.append(row)
            print(
                f"{seq['id']} rep {rep + 1}: {seq['input_tokens']} + {span} tokens "
                f"scored in {row['elapsed_s']:.1f} s",
                flush=True,
            )
            if (
                len(logprobs) != span
                or span < MIN_SPAN_RATIO * seq["continuation_tokens"]
                or not row["finite"]
            ):
                failures.append(
                    f"{seq['id']} rep {rep + 1}: scored {len(logprobs)} of {span} "
                    f"positions for a {seq['continuation_tokens']}-token continuation"
                )
    return rows, failures


@contextmanager
def memory_samples(path, runner=subprocess.run):
    """Sampled per-GPU memory while the scorer runs (not allocator peaks)."""
    stop = threading.Event()
    peaks, errors = {}, []

    def sample():
        while not stop.is_set():
            try:
                result = runner(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,memory.total,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                for uuid, total, used in csv.reader(result.stdout.splitlines()):
                    entry = peaks.setdefault(
                        uuid.strip(), {"total_mib": float(total), "peak_used_mib": 0.0}
                    )
                    entry["peak_used_mib"] = max(entry["peak_used_mib"], float(used))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append(str(exc))
                return
            stop.wait(1.0)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield peaks
    finally:
        stop.set()
        thread.join(timeout=15)
        save(path, {"gpus": peaks, "errors": errors, "sample_period_s": 1.0})


def fatal_log_lines(logs_dir):
    return [
        f"{path.name}: {line.strip()}"
        for path in sorted(Path(logs_dir).glob("**/*.log"))
        for line in path.read_text(errors="replace").splitlines()
        if FATAL_LOG.search(line)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--campaign-fields",
        type=Path,
        required=True,
        help="Prepared campaign fields (the CPU audit's campaign-fields.json)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--count",
        type=int,
        help="Prompts to score (default: the highest tier's per-round quota)",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=ROUND_SCORER_TIMEOUT_S)
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Reuse sequences.json from an earlier --prepare-only run",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--health-timeout", type=float, default=DEFAULT_HEALTH_TIMEOUT_S
    )
    args = parser.parse_args(argv)
    if args.repetitions < 1 or not args.request_timeout > 0:
        parser.error("repetitions and request timeout must be positive")
    config.BENCH_HEALTH_TIMEOUT_S = args.health_timeout
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "failed", "repetitions": args.repetitions}
    started = time.monotonic()
    try:
        fields = json.loads(args.campaign_fields.read_text())
        if fields["engine"]["name"] != "sglang":
            raise ValueError("the smoke scores through SGLang's native endpoints")
        state["inputs_sha256"] = inputs_digest(fields)
        state["scorer_serve_args"] = fields["bench"]["correctness"]["serve_args"]
        if args.prepared:
            prepared = json.loads((args.prepared / "sequences.json").read_text())
            if prepared["inputs_sha256"] != state["inputs_sha256"]:
                raise ValueError("prepared sequences belong to another workload")
        else:
            rule = parse_sampling_rule(fields["sampling_rule"])
            count = args.count or top_tier(rule)["count"]
            tier, sequences = select_sequences(fields, count)
            prepared = {
                "inputs_sha256": state["inputs_sha256"],
                "tier": tier,
                "sequences": sequences,
            }
        save(root / "sequences.json", prepared)
        sequences = prepared["sequences"]
        state.update(
            tier=prepared["tier"],
            input_tokens=[s["input_tokens"] for s in sequences],
            longest_sequence_tokens=max(s["total_tokens"] for s in sequences),
        )
        if args.prepare_only:
            state["status"] = "prepared_only"
            return 0

        state["gpus"] = check_gpus(fields["gpu_skus"][0], fields["bench"]["gpu_count"])
        req, scorer = scorer_request(fields, root)
        weights = stage_weights(req.model, token_env=req.hf_token_env).path
        provider = _EngineProvider(
            req=req, mock=False, logs_dir=root / "engine_logs", weights_dir=weights
        )
        if req.draft_model is not None:
            provider.draft_dir = stage_weights(
                req.draft_model, token_env=req.hf_token_env, require_tokenizer=False
            ).path
        state["scorer_args"] = scorer.spec.serve_args
        with memory_samples(root / "telemetry.json") as peaks:
            with provider.start(scorer, phase=BenchPhase.CORRECTNESS) as url:
                info = get_json(url, "/server_info", timeout=600)
                save(root / "server_info.json", info)
                state["capacity"] = capacity(info, sequences)
                print(f"Scorer capacity: {state['capacity']}", flush=True)
                probe_logprob_capability(url, timeout=args.request_timeout)
                rows, failures = score_all(
                    url,
                    sequences,
                    repetitions=args.repetitions,
                    timeout=args.request_timeout,
                )
        state["peak_used_mib"] = max(
            (gpu["peak_used_mib"] for gpu in peaks.values()), default=None
        )
        save(root / "results.json", rows)
        state["slowest_s"] = max(row["elapsed_s"] for row in rows)
        state["log_errors"] = fatal_log_lines(root / "engine_logs")
        failures += [f"engine log: {line}" for line in state["log_errors"]]
        state["failures"] = failures
        state["status"] = "passed" if not failures else "failed_checks"
        for failure in failures:
            print(f"FAILED {failure}", flush=True)
        return 0 if not failures else 1
    except CheckFailed as exc:
        state["status"] = "failed_checks"
        state["failures"] = [str(exc)]
        print(f"FAILED {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- keep the failure in the evidence
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(state["error"], file=sys.stderr)
        return 3 if isinstance(exc, EngineError) else 2
    finally:
        state["elapsed_s"] = time.monotonic() - started
        save(root / "summary.json", state)
        print(
            f"Scorer smoke status: {state['status']}; summary: {root / 'summary.json'}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
