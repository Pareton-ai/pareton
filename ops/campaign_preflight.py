"""Pre-launch checks for a new long-form campaign, run on its GPU host.

One trusted baseline start, exactly as a round would start it (the worker's
request builder and the harness's planned baseline, with ``/model`` and any
``/draft`` mounted read-only), serves every baseline stage:

1. **Qualification.** ``bench.qualify_longform`` qualifies the source pool and
   writes the sampling rule that seeding requires. ``--qualified-rule`` reuses
   an earlier result instead, after checking it against the campaign fields.
2. **Highest-tier prompt.** The qualified rule samples a fixed-seed trace, and
   one request from its highest input tier (16k for LongWriter) is used for
   the rest of the run.
3. **Temperature extremes.** That prompt is generated ``--repetitions`` times
   at each end of ``temperature_range`` with natural EOS, in waves of the
   campaign's ``request_concurrency``.
4. **Greedy repeatability.** One greedy reference, then a full concurrent wave
   of the same greedy request. Each must match the reference's tokens at the
   SLA quality floor (0.99).
5. **SLA.** Every streamed request above counts toward p99 TTFT and p99
   inter-chunk latency against the campaign's ``sla``.

Then the campaign's own scorer teacher-forces every temperature-extreme output.
Each output must clear the absolute correctness bars on its own: mean logprob,
the token logprob at ``min_token_quantile``, and coverage. The raw minimum
token logprob is recorded against ``min_token_logprob`` as well.

Nothing here writes to a campaign or database. Exit 0: every check passed.
1: a check failed (see summary.json). 2: the run could not complete. 3: an
engine failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import config
from bench.correctness import CapturedOutput, quantile_low, score_captured_output
from bench.http import post_completion_stream
from bench.lifecycle import BenchNetwork, EngineContainer, EngineError, new_run_id
from bench.longform import (
    length_groups,
    require_qualification,
    sampling_context_for_campaign,
)
from bench.main import _EngineProvider, plan_round_starts
from bench.phases import BenchPhase
from bench.qualify_longform import qualify
from bench.sampler import (
    SamplerError,
    build_prompt_formatter,
    fetch_hf_row,
    generate_trace,
    parse_sampling_rule,
)
from bench.sla_bench import percentile
from bench.validate import sha256_file, validate_bench_request_dict
from bench.weights import stage_weights
from campaign.models import SLA
from worker.round_job import build_round_request

# A fixed seed makes the sampled prompt reproducible across runs of this check.
TRACE_SEED = "0" * 64
# The campaign SLA's quality floor: "greedy token-match >= 0.99 vs baseline".
GREEDY_MATCH_FLOOR = 0.99


# Kimi K3 needs far longer than config's 600 s default to load and capture graphs.
DEFAULT_HEALTH_TIMEOUT_S = float(os.environ.get("PARETON_BENCH_HEALTH_TIMEOUT_S", 3600))


def apply_health_timeout(seconds):
    """Engine starts this module doesn't build (the scorer, rounds) read config."""
    config.BENCH_HEALTH_TIMEOUT_S = seconds


class CheckFailed(Exception):
    """A pre-launch check ran and failed (exit 1, not an infrastructure error)."""


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


def baseline_request(fields, work_dir):
    """The round's bench request and its planned engine starts."""
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
        # A round requires a candidate; only the baseline and scorer start here.
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
    return req, starts


def load_qualified_rule(path, fields):
    """Accept a saved rule only if it qualifies exactly this campaign's pins."""
    rule = parse_sampling_rule(json.loads(Path(path).read_text()))
    base = {
        k: v
        for k, v in rule.items()
        if k not in ("qualification", "eligible_row_indices")
    }
    if base != parse_sampling_rule(fields["sampling_rule"]):
        raise SamplerError("qualified rule differs from the campaign's sampling rule")
    require_qualification(rule, fields["bench"], fields["engine"])
    return rule


def highest_tier_request(fields, rule, request_index):
    """One request from the highest input tier of a fixed-seed trace."""
    model = fields["bench"]["model"]
    formatter = build_prompt_formatter(
        rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
    )
    sampled = generate_trace(
        rule=rule,
        seed_hex=TRACE_SEED,
        row_fetcher=lambda i: fetch_hf_row(rule, i),
        prompt_formatter=formatter,
        sampling_context=sampling_context_for_campaign(
            fields["bench"], fields["engine"]
        ),
    )
    trace = json.loads(sampled.body)
    tier = length_groups(rule["n_prompts"], rule.get("input_tiers"))[-1]["name"]
    requests = [r for r in trace["requests"] if r.get("input_length_group") == tier]
    if not 0 <= request_index < len(requests):
        raise ValueError(f"request index must be below {len(requests)} ({tier} tier)")
    return trace, tier, requests[request_index], formatter


def temperature_extremes(rule):
    bounds = rule.get("temperature_range")
    if bounds is None:
        raise ValueError("sampling rule has no temperature_range")
    return [float(bounds[0]), float(bounds[1])]


def stream_one(url, request, *, label, temperature, seed, timeout):
    result = post_completion_stream(
        url,
        prompt=request["prompt"],
        max_tokens=request["max_tokens"],
        temperature=temperature,
        top_p=request["sampling"].get("top_p"),
        seed=seed,
        timeout=timeout,
        # Speculative decoding may stream several tokens per chunk.
        require_token_timing=False,
    )
    if result.prompt_tokens != request["input_tokens"]:
        raise EngineError(
            f"{label}: engine counted {result.prompt_tokens} prompt tokens, "
            f"trace has {request['input_tokens']}"
        )
    tokens = result.completion_tokens or 0
    return {
        "request_id": label,
        "temperature": temperature,
        "seed": seed,
        "finish_reason": result.finish_reason,
        "completion_tokens": tokens,
        "ttft_ms": result.ttft_s * 1000,
        "itl_ms": [gap * 1000 for gap in result.itl_s],
        "tpot_ms": (
            (result.e2e_s - result.ttft_s) * 1000 / (tokens - 1) if tokens > 1 else None
        ),
        "token_timing_available": len(result.itl_s) >= tokens - 1,
        "output_text": result.text,
    }


def run_waves(url, request, jobs, *, concurrency, timeout):
    """Run (label, temperature, seed) jobs, ``concurrency`` at a time, in order."""
    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(jobs), concurrency):
            wave = jobs[start : start + concurrency]
            futures = [
                pool.submit(
                    stream_one,
                    url,
                    request,
                    label=label,
                    temperature=temperature,
                    seed=seed,
                    timeout=timeout,
                )
                for label, temperature, seed in wave
            ]
            for future in futures:
                row = future.result()
                rows.append(row)
                print(
                    f"{row['request_id']}: {row['completion_tokens']} tokens, "
                    f"TTFT {row['ttft_ms']:.0f} ms",
                    flush=True,
                )
    return rows


def token_match(reference, other):
    """Position-wise token agreement over the longer of the two sequences."""
    if not reference and not other:
        return 1.0
    same = sum(a == b for a, b in zip(reference, other, strict=False))
    return same / max(len(reference), len(other))


def sla_summary(rows, sla):
    ttft = [row["ttft_ms"] for row in rows]
    gaps = [gap for row in rows for gap in row["itl_ms"]]
    tpot = [row["tpot_ms"] for row in rows if row["tpot_ms"] is not None]
    summary = {
        "requests": len(rows),
        "p99_ttft_ms": percentile(ttft, 99),
        "p99_itl_ms": percentile(gaps, 99) if gaps else None,
        "p99_tpot_ms": percentile(tpot, 99) if tpot else None,
        "per_token_timing": all(row["token_timing_available"] for row in rows),
        "limits": {"p99_ttft_ms": sla["p99_ttft_ms"], "p99_itl_ms": sla["p99_itl_ms"]},
    }
    failures = []
    if summary["p99_ttft_ms"] > sla["p99_ttft_ms"]:
        failures.append(
            f"p99 TTFT {summary['p99_ttft_ms']:.0f} ms > {sla['p99_ttft_ms']:.0f} ms"
        )
    if summary["p99_itl_ms"] is None or summary["p99_itl_ms"] > sla["p99_itl_ms"]:
        failures.append(
            f"p99 inter-chunk latency {summary['p99_itl_ms']} ms > {sla['p99_itl_ms']} ms"
        )
    summary["failures"] = failures
    return summary


def grade_output(scorer_url, captured, thresholds, *, timeout):
    """Apply the campaign's absolute bars to a single output."""
    scores, span, _ = score_captured_output(
        scorer_url, captured, request_timeout_s=timeout, engine_name="sglang"
    )
    logprobs = [s.logprob for s in scores]
    result = {
        "request_id": captured.request_id,
        "completion_tokens": captured.completion_tokens,
        "span_positions": span,
        "scored_positions": len(logprobs),
    }
    if not logprobs or span < 1:
        return {**result, "verdict": "fail", "reason": "scorer returned no logprobs"}
    mean = statistics.fmean(logprobs)
    quantile = quantile_low(logprobs, thresholds.min_token_quantile)
    coverage = len(logprobs) / span
    failures = []
    if mean < thresholds.min_mean_logprob:
        failures.append(f"mean logprob {mean:.3f} < {thresholds.min_mean_logprob}")
    if quantile < thresholds.min_token_logprob:
        failures.append(
            f"token logprob at quantile {thresholds.min_token_quantile} "
            f"{quantile:.3f} < {thresholds.min_token_logprob}"
        )
    if coverage < thresholds.min_coverage_ratio:
        failures.append(f"coverage {coverage:.3f} < {thresholds.min_coverage_ratio}")
    return {
        **result,
        "mean_logprob": mean,
        "quantile_logprob": quantile,
        "min_logprob": min(logprobs),
        "raw_min_below_min_token_logprob": min(logprobs) < thresholds.min_token_logprob,
        "coverage_ratio": coverage,
        "verdict": "fail" if failures else "pass",
        "reason": "; ".join(failures) or None,
    }


def summarize_logprobs(results, temperatures):
    by_temperature = {}
    for temperature in temperatures:
        rows = [r for r in results if r["temperature"] == temperature]
        by_temperature[str(temperature)] = {
            "samples": len(rows),
            "passed": sum(r["verdict"] == "pass" for r in rows),
            "worst_mean_logprob": min(
                (r["mean_logprob"] for r in rows if "mean_logprob" in r), default=None
            ),
            "worst_quantile_logprob": min(
                (r["quantile_logprob"] for r in rows if "quantile_logprob" in r),
                default=None,
            ),
            "worst_min_logprob": min(
                (r["min_logprob"] for r in rows if "min_logprob" in r), default=None
            ),
        }
    return by_temperature


def request_timeout(args, rule):
    return args.request_timeout or float(rule.get("request_timeout_s") or 600)


def run_baseline_stages(args, fields, req, start, root, state):
    """Qualify, then generate every baseline sample, on one baseline start."""
    engine_ref = fields["bench"]["baseline_engine_image_digest"]
    rule = fields["sampling_rule"]
    concurrency = int(rule.get("request_concurrency") or 1)
    timeout = request_timeout(args, rule)
    # qualify_longform binds its endpoint to a published loopback port, which
    # an internal network cannot publish.
    with BenchNetwork(run_id=new_run_id(), internal=False) as network:
        container = EngineContainer(
            spec=start.spec,
            network=network,
            role="preflight-baseline",
            gpu_count=req.hardware.gpu_count,
            weights_dir=state["weights_dir"],
            draft_dir=state["draft_dir"],
            publish_port=True,
            health_timeout_s=args.health_timeout,
            logs_dir=root / "engine_logs",
        )
        with container as handle:
            print(f"Baseline healthy at {handle.base_url}.", flush=True)
            if args.qualified_rule:
                qualified = load_qualified_rule(args.qualified_rule, fields)
                save(root / "qualification" / "sampling_rule.json", qualified)
                state["qualification"] = {"reused": str(args.qualified_rule)}
            else:
                try:
                    qualified = qualify(
                        fields=fields,
                        base_url=handle.base_url,
                        container=handle.container_id,
                        engine_ref=engine_ref,
                        output_dir=root / "qualification",
                        pool_size=args.pool_size,
                        repetitions=args.qualification_repetitions,
                        concurrency=args.qualification_concurrency,
                        timeout=timeout,
                    )
                except SamplerError as exc:
                    if "cannot fill" in str(exc):
                        raise CheckFailed(f"qualification: {exc}") from exc
                    raise
                state["qualification"] = json.loads(
                    (root / "qualification" / "summary.json").read_text()
                )
            print("Qualified rule ready; sampling the highest-tier prompt.", flush=True)
            trace, tier, request, formatter = highest_tier_request(
                fields, qualified, args.request_index
            )
            save(root / "workload_trace.json", trace)
            temperatures = temperature_extremes(qualified)
            state.update(
                tier=tier,
                source_request_id=request["id"],
                input_tokens=request["input_tokens"],
                max_tokens=request["max_tokens"],
                temperatures=temperatures,
                concurrency=concurrency,
            )
            jobs = [
                (f"t{t}-r{rep}", t, rep)
                for t in temperatures
                for rep in range(args.repetitions)
            ]
            samples = run_waves(
                handle.base_url, request, jobs, concurrency=concurrency, timeout=timeout
            )
            # Alone first, then a full concurrent wave, as rounds run.
            reference = run_waves(
                handle.base_url,
                request,
                [("greedy-ref", 0.0, 0)],
                concurrency=1,
                timeout=timeout,
            )
            greedy = run_waves(
                handle.base_url,
                request,
                [(f"greedy-{i}", 0.0, 0) for i in range(concurrency)],
                concurrency=concurrency,
                timeout=timeout,
            )
    return request, formatter, samples, reference, greedy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-fields", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qualified-rule",
        type=Path,
        help="Reuse a sampling rule qualified for these exact pins",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        help="Qualification pool (default: twice n_prompts, split across tiers)",
    )
    parser.add_argument("--qualification-repetitions", type=int, default=2)
    parser.add_argument("--qualification-concurrency", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--request-index",
        type=int,
        default=0,
        help="Which highest-tier request of the fixed-seed trace to use",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        help="Per-request timeout (default: the rule's request_timeout_s)",
    )
    parser.add_argument(
        "--health-timeout", type=float, default=DEFAULT_HEALTH_TIMEOUT_S
    )
    args = parser.parse_args(argv)
    apply_health_timeout(args.health_timeout)
    if args.repetitions < 1 or (
        args.request_timeout is not None
        and (not math.isfinite(args.request_timeout) or args.request_timeout <= 0)
    ):
        parser.error("repetitions and request timeout must be positive")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "failed", "repetitions": args.repetitions}
    started = time.monotonic()
    try:
        fields = json.loads(args.campaign_fields.read_text())
        if fields["engine"]["name"] != "sglang":
            raise ValueError("these checks score through SGLang's native endpoints")
        if "eligible_row_indices" in fields["sampling_rule"]:
            raise ValueError("pass the unqualified campaign fields")
        save(root / "campaign_fields.json", fields)
        state["gpus"] = check_gpus(fields["gpu_skus"][0], fields["bench"]["gpu_count"])
        req, starts = baseline_request(fields, root)
        thresholds = req.correctness.thresholds
        state["thresholds"] = asdict(thresholds)
        state["weights_dir"] = stage_weights(req.model, token_env=req.hf_token_env).path
        state["draft_dir"] = None
        if req.draft_model is not None:
            state["draft_dir"] = stage_weights(
                req.draft_model, token_env=req.hf_token_env, require_tokenizer=False
            ).path
        baseline = next(s for s in starts if s.kind == "baseline")
        scorer = next(s for s in starts if s.kind == "scorer")
        request, formatter, samples, reference, greedy = run_baseline_stages(
            args, fields, req, baseline, root, state
        )
        save(root / "outputs.json", [*samples, *reference, *greedy])

        reference_ids = formatter.encode(reference[0]["output_text"])
        state["greedy"] = {
            "floor": GREEDY_MATCH_FLOOR,
            "reference_tokens": len(reference_ids),
            "matches": {
                row["request_id"]: token_match(
                    reference_ids, formatter.encode(row["output_text"])
                )
                for row in greedy
            },
        }
        state["sla"] = sla_summary([*samples, *reference, *greedy], fields["sla"])

        provider = _EngineProvider(
            req=req,
            mock=False,
            logs_dir=root / "engine_logs",
            weights_dir=state["weights_dir"],
        )
        provider.draft_dir = state["draft_dir"]
        results = []
        with provider.start(scorer, phase=BenchPhase.CORRECTNESS) as url:
            for row in samples:
                captured = CapturedOutput(
                    request_id=row["request_id"],
                    prompt=request["prompt"],
                    output_text=row["output_text"],
                    completion_tokens=row["completion_tokens"],
                )
                graded = (
                    grade_output(
                        url,
                        captured,
                        thresholds,
                        timeout=request_timeout(args, fields["sampling_rule"]),
                    )
                    if row["output_text"]
                    else {
                        "request_id": row["request_id"],
                        "verdict": "fail",
                        "reason": "empty output",
                    }
                )
                results.append(
                    {
                        **graded,
                        "temperature": row["temperature"],
                        "finish_reason": row["finish_reason"],
                    }
                )
                print(
                    f"{row['request_id']}: {graded['verdict']}"
                    + (f" ({graded['reason']})" if graded.get("reason") else ""),
                    flush=True,
                )
        save(root / "results.json", results)
        state["by_temperature"] = summarize_logprobs(results, state["temperatures"])
        state["failed_samples"] = [
            r["request_id"] for r in results if r["verdict"] != "pass"
        ]
        failures = [
            *(f"logprob: {r}" for r in state["failed_samples"]),
            *(f"sla: {f}" for f in state["sla"]["failures"]),
            *(
                f"greedy: {label} matched {ratio:.4f} < {GREEDY_MATCH_FLOOR}"
                for label, ratio in state["greedy"]["matches"].items()
                if ratio < GREEDY_MATCH_FLOOR
            ),
        ]
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
        for key in ("weights_dir", "draft_dir"):
            if state.get(key) is not None:
                state[key] = str(state[key])
        save(root / "summary.json", state)
        print(
            f"Preflight status: {state['status']}; summary: {root / 'summary.json'}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
