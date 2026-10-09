"""Check one campaign prompt's logprobs at the lowest and highest sampling temperatures.

Run on the campaign's GPU host before opening a campaign. The trusted baseline
engine generates the same LongWriter prompt ``--repetitions`` times at each end
of the sampling rule's ``temperature_range``, with natural EOS and a different
seed per repetition. The campaign's own scorer then teacher-forces every output,
and each one must clear the campaign's absolute correctness bars on its own:
mean logprob, the token logprob at ``min_token_quantile``, and coverage. The raw
minimum token logprob is also recorded against ``min_token_logprob``.

Nothing here writes to a campaign or database.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from bench.correctness import CapturedOutput, quantile_low, score_captured_output
from bench.http import post_completion
from bench.lifecycle import EngineError
from bench.longform import sampling_context_for_campaign
from bench.main import _EngineProvider, plan_round_starts
from bench.phases import BenchPhase
from bench.sampler import build_prompt_formatter, fetch_hf_row, generate_trace
from bench.schemas import WorkloadTrace
from bench.validate import sha256_file, validate_bench_request_dict
from bench.weights import stage_weights
from campaign.models import SLA
from worker.round_job import build_round_request

# A fixed seed makes the sampled prompt reproducible across runs of this check.
TRACE_SEED = "0" * 64


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sample_prompt(fields, request_index):
    """One request from the campaign's own sampler, rendered with its template."""
    rule = fields["sampling_rule"]
    model = fields["bench"]["model"]
    sampled = generate_trace(
        rule=rule,
        seed_hex=TRACE_SEED,
        row_fetcher=lambda i: fetch_hf_row(rule, i),
        prompt_formatter=build_prompt_formatter(
            rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
        ),
        sampling_context=sampling_context_for_campaign(
            fields["bench"], fields["engine"]
        ),
    )
    trace = json.loads(sampled.body)
    if not 0 <= request_index < len(trace["requests"]):
        raise ValueError(f"request index must be below {len(trace['requests'])}")
    return trace, trace["requests"][request_index]


def temperature_extremes(rule):
    bounds = rule.get("temperature_range")
    if bounds is None:
        raise ValueError("sampling rule has no temperature_range")
    return [float(bounds[0]), float(bounds[1])]


def build_request(fields, trace_path):
    request = build_round_request(
        {
            "gpu_sku": fields["gpu_skus"][0],
            "sampled_trace_sha256": sha256_file(trace_path),
            "scoring_rule": fields["scoring_rule"],
        },
        SimpleNamespace(
            bench=fields["bench"], engine=fields["engine"], sla=SLA(**fields["sla"])
        ),
        # A round requires a candidate; this check starts only baseline and scorer.
        [
            {"role": role, "engine_image_ref": image}
            for role in ("baseline", "challenger")
            for image in [fields["bench"]["baseline_engine_image_digest"]]
        ],
        task_id=str(uuid4()),
        trace_path=str(trace_path),
    )
    return validate_bench_request_dict(request)


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


def summarize(results, temperatures):
    by_temperature = {}
    for temperature in temperatures:
        rows = [r for r in results if r["temperature"] == temperature]
        means = [r["mean_logprob"] for r in rows if "mean_logprob" in r]
        by_temperature[str(temperature)] = {
            "samples": len(rows),
            "passed": sum(r["verdict"] == "pass" for r in rows),
            "worst_mean_logprob": min(means) if means else None,
            "worst_quantile_logprob": min(
                (r["quantile_logprob"] for r in rows if "quantile_logprob" in r),
                default=None,
            ),
            "worst_min_logprob": min(
                (r["min_logprob"] for r in rows if "min_logprob" in r),
                default=None,
            ),
        }
    return by_temperature


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-fields", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--request-index",
        type=int,
        default=0,
        help="Which request of the sampled campaign trace to use",
    )
    parser.add_argument("--request-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if (
        args.repetitions < 1
        or not math.isfinite(args.request_timeout)
        or args.request_timeout <= 0
    ):
        parser.error("repetitions and request timeout must be positive")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "failed", "repetitions": args.repetitions}
    started = time.monotonic()
    try:
        fields = json.loads(args.campaign_fields.read_text())
        if fields["engine"]["name"] != "sglang":
            raise ValueError("this check scores through SGLang's native endpoints")
        save(root / "campaign_fields.json", fields)
        temperatures = temperature_extremes(fields["sampling_rule"])
        trace, request = sample_prompt(fields, args.request_index)
        trace_path = root / "workload_trace.json"
        save(trace_path, trace)
        WorkloadTrace.from_dict(trace)
        req = build_request(fields, trace_path)
        thresholds = req.correctness.thresholds
        state.update(
            temperatures=temperatures,
            source_request_id=request["id"],
            input_tokens=request.get("input_tokens"),
            max_tokens=request["max_tokens"],
            thresholds=asdict(thresholds),
        )
        staged = stage_weights(req.model, token_env=req.hf_token_env)
        provider = _EngineProvider(
            req=req, mock=False, logs_dir=root / "engine_logs", weights_dir=staged.path
        )
        if req.draft_model is not None:
            provider.draft_dir = stage_weights(
                req.draft_model, token_env=req.hf_token_env, require_tokenizer=False
            ).path
        starts = plan_round_starts(
            req.engines, correctness_serve_args=req.correctness.serve_args
        )
        baseline = next(s for s in starts if s.kind == "baseline")
        scorer = next(s for s in starts if s.kind == "scorer")
        outputs = []
        with provider.start(baseline, phase=BenchPhase.SLA_BENCH) as url:
            for temperature in temperatures:
                for rep in range(args.repetitions):
                    request_id = f"t{temperature}-r{rep}"
                    response = post_completion(
                        url,
                        prompt=request["prompt"],
                        max_tokens=request["max_tokens"],
                        logprobs=None,
                        temperature=temperature,
                        top_p=request["sampling"].get("top_p"),
                        seed=rep,
                        timeout=args.request_timeout,
                    )
                    choice = response["choices"][0]
                    usage = response.get("usage") or {}
                    outputs.append(
                        (
                            temperature,
                            choice.get("finish_reason"),
                            CapturedOutput(
                                request_id=request_id,
                                prompt=request["prompt"],
                                output_text=choice.get("text") or "",
                                completion_tokens=int(
                                    usage.get("completion_tokens") or 0
                                ),
                            ),
                        )
                    )
                    print(
                        f"{request_id}: {outputs[-1][2].completion_tokens} tokens",
                        flush=True,
                    )
        save(
            root / "outputs.json",
            [
                {
                    "temperature": t,
                    "finish_reason": finish,
                    "request_id": o.request_id,
                    "completion_tokens": o.completion_tokens,
                    "output_text": o.output_text,
                }
                for t, finish, o in outputs
            ],
        )
        results = []
        with provider.start(scorer, phase=BenchPhase.CORRECTNESS) as url:
            for temperature, finish, captured in outputs:
                if not captured.output_text:
                    graded = {
                        "request_id": captured.request_id,
                        "verdict": "fail",
                        "reason": "empty output",
                    }
                else:
                    graded = grade_output(
                        url, captured, thresholds, timeout=args.request_timeout
                    )
                results.append(
                    {**graded, "temperature": temperature, "finish_reason": finish}
                )
                print(
                    f"{captured.request_id}: {graded['verdict']}"
                    + (f" ({graded['reason']})" if graded.get("reason") else ""),
                    flush=True,
                )
        save(root / "results.json", results)
        state["by_temperature"] = summarize(results, temperatures)
        failed = [r["request_id"] for r in results if r["verdict"] != "pass"]
        state["failed_samples"] = failed
        state["status"] = "passed" if not failed else "failed_thresholds"
        return 0 if not failed else 1
    except Exception as exc:  # noqa: BLE001 -- keep the failure in the evidence
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(state["error"], file=sys.stderr)
        if isinstance(exc, EngineError):
            return 3
        return 2
    finally:
        state["elapsed_s"] = time.monotonic() - started
        save(root / "summary.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
