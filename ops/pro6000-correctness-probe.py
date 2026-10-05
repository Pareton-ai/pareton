"""Opt-in PAR-144 LongWriter scorer diagnostic; no campaign or production writes.

Run from the repository root with PYTHONPATH=. This intentionally transforms
LongWriter inputs into a longest-tier diagnostic, not a qualified campaign trace.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import bench.main as harness
from bench.correctness import BASELINE_INDEX, grade_all
from bench.http import get_json
from bench.lifecycle import EngineContainer, EngineError
from bench.longform import (
    input_group,
    request_for_candidate,
    sampling_context_for_campaign,
    source_messages,
)
from bench.sampler import (
    PromptRenderError,
    build_prompt_formatter,
    fetch_hf_row,
    parse_sampling_rule,
)
from bench.schemas import TraceMeta, WorkloadTrace
from bench.sla_bench import capture_baseline_natural_stops, run_sla_engine
from bench.trajectory import token_ids_sha256
from bench.validate import (
    sha256_file,
    validate_bench_request_dict,
    validate_workload_trace_dict,
)
from bench.workload_preflight import validate_engine_workload
from campaign.models import SLA
from gpu.static_host import REMOTE_LOCK, check_idle_gpu, host_lock
from worker.round_job import build_round_request

MAX_INPUT = 16384
MAX_OUTPUT = 5120
SCALARS = {
    "--context-length": "context_length",
    "--mem-fraction-static": "mem_fraction_static",
    "--tp": "tp_size",
    "--tp-size": "tp_size",
    "--tensor-parallel-size": "tp_size",
    "--pp-size": "pp_size",
    "--pipeline-parallel-size": "pp_size",
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def scalar_args(args):
    """Read only known scalar aliases; never deduplicate repeatable engine flags."""
    found = {}
    i = 0
    while i < len(args):
        flag, sep, value = args[i].partition("=")
        if flag in SCALARS:
            if not sep:
                i += 1
                if i == len(args) or args[i].startswith("--"):
                    raise ValueError(f"missing value for {flag}")
                value = args[i]
            found[SCALARS[flag]] = value
        i += 1
    return found


def normalize_scalars(args):
    """Preserve argparse last-value-wins semantics for the diagnostic launch."""
    effective = scalar_args(args)
    result = []
    i = 0
    while i < len(args):
        flag, sep, _ = args[i].partition("=")
        if flag in SCALARS:
            i += 1 if sep else 2
        else:
            result.append(args[i])
            i += 1
    canonical = {value: key for key, value in reversed(list(SCALARS.items()))}
    for key, value in effective.items():
        result.extend([canonical[key], value])
    return result


def prepare_fields(fields, generation_fraction, scorer_fraction):
    fields = copy.deepcopy(fields)
    bench = fields["bench"]
    model = bench["model"]
    if (
        fields["engine"]["name"] != "sglang"
        or fields["gpu_skus"] != ["RTXPRO6000"]
        or bench["gpu_count"] != 1
        or model["hf_repo"] != "Qwen/Qwen3.8-27B-FP8"
        or model["quantization"] != "fp8"
        or fields["sampling_rule"]["dataset"] != "zai-org/LongWriter-6k"
        or fields["sampling_rule"]["algo_version"] != 4
    ):
        raise ValueError("requires a TP1 RTXPRO6000 SGLang FP8 LongWriter v4 fixture")
    for value in (generation_fraction, scorer_fraction):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("memory fractions must be finite and between 0 and 1")
    bench["serve_args"] = normalize_scalars(
        list(bench.get("serve_args") or [])
        + ["--mem-fraction-static", str(generation_fraction)]
    )
    corr = bench["correctness"]
    corr["serve_args"] = normalize_scalars(
        list(corr.get("serve_args") or [])
        + ["--mem-fraction-static", str(scorer_fraction)]
    )
    for args in (bench["serve_args"], bench["serve_args"] + corr["serve_args"]):
        effective = scalar_args(args)
        if (
            int(effective.get("tp_size", 1)) != 1
            or int(effective.get("pp_size", 1)) != 1
        ):
            raise ValueError("diagnostic requires TP1/PP1")
    # Context overrides would invalidate admission and the scorer's +7 contract.
    for args in (bench["serve_args"], corr["serve_args"]):
        context = scalar_args(args).get("context_length")
        if context is not None and int(context) != model["max_model_len"]:
            raise ValueError("serving context must equal the model context pin")
    if "context_length" in scalar_args(corr["serve_args"]):
        raise ValueError("correctness context override would undo scorer headroom")
    if corr["thresholds"].get("max_mean_logprob_drop") is None:
        raise ValueError("baseline reference scoring requires max_mean_logprob_drop")
    return fields


def build_source_preview(fields, output_dir, count):
    """Scan the pinned corpus without balanced-tier quotas; retain the longest N.

    Shorter fallback uses complete rendered conversations, never padding,
    truncation, or duplicated prompts. Only this diagnostic changes selection.
    """
    rule = parse_sampling_rule(fields["sampling_rule"])
    model = fields["bench"]["model"]
    context = sampling_context_for_campaign(fields["bench"], fields["engine"])
    formatter = build_prompt_formatter(
        rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
    )
    retained = {}
    seen = set()
    eligible = 0
    eligible_16k = 0
    for index in range(rule["n_rows"]):
        messages = source_messages(fetch_hf_row(rule, index), rule)
        if messages is None:
            continue
        try:
            prompt = formatter.render(messages)
            ids = formatter.encode(prompt)
        except PromptRenderError:
            continue
        size = len(ids)
        if (
            not 1 <= size <= min(MAX_INPUT, context["max_input_tokens"])
            or size + MAX_OUTPUT + context["engine_reserve"] > context["max_model_len"]
        ):
            continue
        digest = token_ids_sha256(ids)
        if digest in seen:
            continue
        seen.add(digest)
        eligible += 1
        eligible_16k += input_group(size) == "16k"
        retained[digest] = {
            "row_index": index,
            "prompt": prompt,
            "input_tokens": size,
            "input_ids_sha256": digest,
            "input_length_group": input_group(size) or "shorter",
        }
        if len(retained) > count:
            shortest = min(
                retained,
                key=lambda key: (
                    retained[key]["input_tokens"],
                    -retained[key]["row_index"],
                ),
            )
            del retained[shortest]
    if len(retained) < count:
        raise ValueError(
            f"not enough distinct eligible prompts even with shorter fallback: {len(retained)}/{count}"
        )
    selected = sorted(
        retained.values(), key=lambda r: (-r["input_tokens"], r["row_index"])
    )
    generation_rule = {**rule, "max_tokens": MAX_OUTPUT}
    source = {
        "schema_version": 1,
        "meta": {
            "name": "PAR-144-longest-available-source",
            "pro6000_source_version": 1,
            "context": context,
        },
        "requests": [
            request_for_candidate(item, generation_rule, i, generation_seed="c" * 64)
            for i, item in enumerate(selected)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "workload_trace.json"
    save(path, source)
    receipt = {
        "dataset": rule["dataset"],
        "revision": rule["revision"],
        "config": rule["config"],
        "split": rule["split"],
        **formatter.receipt,
        "sampled_trace_sha256": sha256_file(path),
        "context": context,
        "diagnostic_only": True,
        "selection": "longest_available_with_shorter_fallback",
        "rows_scanned": rule["n_rows"],
        "eligible_distinct_prompts": eligible,
        "eligible_16k_prompts": eligible_16k,
        "requests": [
            {k: v for k, v in item.items() if k != "prompt"} for item in selected
        ],
    }
    save(output_dir / "sampling_receipt.json", receipt)
    print(
        f"Selected {count} longest available prompts; {eligible_16k} distinct 16k-tier prompts in corpus",
        flush=True,
    )


def longest_trace(source, fields, *, count, prefixes, capacity):
    """Prefer longest inputs and fill with shorter distinct inputs as necessary."""
    if fields["bench"]["model"]["max_model_len"] < MAX_INPUT + MAX_OUTPUT + 2:
        raise ValueError("input plus output and engine reservation exceed context")
    original = validate_workload_trace_dict(source)
    context = sampling_context_for_campaign(fields["bench"], fields["engine"])
    if original.meta.sampling and original.meta.sampling.get("algo_version") == 4:
        source_context = original.meta.sampling["context"]
    elif source.get("meta", {}).get("pro6000_source_version") == 1:
        source_context = source["meta"].get("context")
    else:
        raise ValueError(
            "source must be a LongWriter v4 preview or diagnostic source preview"
        )
    if source_context != context:
        raise ValueError("source context differs from campaign")
    for request in source["requests"]:
        size = request.get("input_tokens")
        if (
            type(size) is not int
            or not 1 <= size <= MAX_INPUT
            or not isinstance(request.get("prompt"), str)
            or not request["prompt"].strip()
            or not request.get("input_ids_sha256")
        ):
            raise ValueError("invalid token-counted source input")
    candidates = sorted(source["requests"], key=lambda r: (-r["input_tokens"], r["id"]))
    unique = {r["input_ids_sha256"]: r for r in reversed(candidates)}
    candidates = sorted(unique.values(), key=lambda r: (-r["input_tokens"], r["id"]))
    if (
        count < 1
        or not candidates
        or (prefixes == "distinct" and len(candidates) < count)
    ):
        raise ValueError(
            "not enough distinct eligible prompts even with shorter fallback; generate a larger source preview"
        )
    selected = candidates[:count] if prefixes == "distinct" else [candidates[0]] * count
    requests = []
    for i, source_request in enumerate(selected):
        req = copy.deepcopy(source_request)
        if (
            req["input_tokens"] + MAX_OUTPUT + context["engine_reserve"]
            > context["max_model_len"]
        ):
            raise ValueError("input plus output and engine reservation exceed context")
        req.update(id=f"probe-{i:03d}", max_tokens=MAX_OUTPUT, arrival_offset_ms=i * 2)
        req["sampling"]["ignore_eos"] = capacity
        requests.append(req)
    # This is deliberately not presented as the sampler's balanced four-tier trace.
    # The hook below still performs the real engine/tokenizer capacity preflight.
    diagnostic = {
        "schema_version": 1,
        "meta": {
            "name": "PAR-144-longest-tier-diagnostic",
            "description": "Not a campaign qualification trace",
            "selection": "longest_available_with_shorter_fallback",
            "shorter_fallbacks": [
                {
                    "request_id": req["id"],
                    "source_request_id": item["id"],
                    "input_tokens": req["input_tokens"],
                }
                for req, item in zip(requests, selected, strict=True)
                if input_group(req["input_tokens"]) != "16k"
            ],
        },
        "requests": requests,
    }
    validate_workload_trace_dict(diagnostic)
    return diagnostic


def prepare_request(fields, trace_path):
    image = fields["bench"]["baseline_engine_image_digest"]
    request = build_round_request(
        {
            "gpu_sku": "RTXPRO6000",
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
    request["sla_bench"]["repetitions"] = 3
    request["correctness"]["num_prompts"] = len(
        json.loads(trace_path.read_text())["requests"]
    )
    validate_bench_request_dict(request)
    return request


def validate_runtime(info, args):
    actual = {**info, **(info.get("server_args") or {})}
    for key, raw in scalar_args(args).items():
        if key not in actual or not math.isclose(
            float(actual[key]), float(raw), abs_tol=1e-8
        ):
            raise EngineError(f"runtime {key} does not confirm requested value {raw}")


def require_reports(reports, count):
    if set(reports) != {BASELINE_INDEX, 0}:
        raise EngineError(
            "baseline or identical-image candidate correctness report missing"
        )
    for report in reports.values():
        if (
            report.verdict != "pass"
            or report.num_prompts != count
            or report.num_positions_scored < count
            or report.coverage_ratio != 1.0
        ):
            raise EngineError("incomplete or failed correctness; see saved reports")


@contextmanager
def diagnostic_hooks(root, fields, trace, *, capacity, timeout, scorer_repetitions):
    """Hooks are confined to this CLI process; shared harness files stay untouched."""
    state = {"starts": [], "scorer_repetitions": 0}
    context = sampling_context_for_campaign(fields["bench"], fields["engine"])
    preflight_trace = WorkloadTrace.from_dict(trace)
    preflight_trace.meta = TraceMeta("diagnostic", sampling={"context": context})
    count = len(trace["requests"])

    class RecordedContainer(EngineContainer):
        def __enter__(self):
            raw = list(self.spec.serve_args)
            self.spec = replace(self.spec, serve_args=normalize_scalars(raw))
            dest = root / "runtime" / self.role
            save(
                dest / "launch.json",
                {
                    "image": self.spec.image,
                    "raw_args": raw,
                    "effective_args": self.spec.serve_args,
                    "env": self.spec.env,
                    "mount_engine_cache": self.mount_engine_cache,
                    "gpu_count": self.gpu_count,
                },
            )
            if self.gpu_count != 1:
                raise EngineError("harness changed the requested GPU count")
            handle = super().__enter__()
            try:
                info = get_json(handle.base_url, "/server_info", timeout=timeout)
                save(dest / "server_info.json", info)
                validate_runtime(info, self.spec.serve_args)
                save(dest / "handle.json", asdict(handle))
                if self.role != "scorer":
                    validate_engine_workload(
                        handle.base_url,
                        preflight_trace,
                        engine_name="sglang",
                        max_model_len=context["max_model_len"],
                        evidence_dir=dest,
                        verify_tokenizer=True,
                    )
                state["starts"].append(self.role)
                return handle
            except BaseException:
                super().__exit__(*sys.exc_info())
                raise

        def __exit__(self, *args):
            try:
                if self._handle:
                    result = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            '{"state":{{json .State}},"restart_count":{{.RestartCount}}}',
                            self._handle.container_id,
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=30,
                    )
                    status = json.loads(result.stdout)
                    save(root / "runtime" / self.role / "exit-state.json", status)
                    save(
                        root / "runtime" / self.role / "server_info-after.json",
                        get_json(
                            self._handle.base_url, "/server_info", timeout=timeout
                        ),
                    )
                    if (
                        status["state"].get("OOMKilled")
                        or not status["state"].get("Running")
                        or status["restart_count"] != 0
                    ):
                        raise EngineError(
                            "container stopped or was OOM-killed during diagnostic"
                        )
            finally:
                super().__exit__(*args)

    def replay(url, **kwargs):
        return run_sla_engine(url, **{**kwargs, "request_timeout_s": timeout})

    def natural_stops(url, **kwargs):
        return capture_baseline_natural_stops(
            url, **{**kwargs, "request_timeout_s": timeout}
        )

    def grade(url, pending, **kwargs):
        save(root / "captured_outputs.json", [asdict(item) for item in pending])
        if len(pending) != 2 or any(len(item.outputs) != count for item in pending):
            raise EngineError(
                "missing captured outputs; cannot qualify scorer prompt count"
            )
        if capacity and any(
            out.completion_tokens != MAX_OUTPUT
            for item in pending
            for out in item.outputs
        ):
            raise EngineError("forced capacity output did not reach 5120 tokens")
        for rep in range(scorer_repetitions):
            dest = root / "scorer_replays" / f"rep-{rep + 1}"
            dest.mkdir(parents=True)
            started = time.monotonic()
            reports = grade_all(
                url,
                pending,
                **{**kwargs, "evidence_dir": dest, "request_timeout_s": timeout},
            )
            save(
                dest / "reports.json",
                {
                    str(k): {**v.to_dict(), "evidence": Path(v.evidence).name}
                    for k, v in reports.items()
                },
            )
            save(dest / "timing.json", {"elapsed_s": time.monotonic() - started})
            require_reports(reports, count)
            # Verify each continuation, not just aggregate coverage. Tokenized text
            # can differ from generation token counts; record any such mismatch.
            for path in (dest / "baseline.jsonl", dest / "candidate_0.jsonl"):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if len(rows) != count or any(
                    r.get("dropped")
                    or r.get("span_positions", 0) < 1
                    or r.get("scored_positions") != r.get("span_positions")
                    for r in rows
                ):
                    raise EngineError(
                        "missing/truncated scorer positions or excluded prompts"
                    )
                if capacity and any(r["span_positions"] != MAX_OUTPUT for r in rows):
                    raise EngineError(
                        "scorer did not consume a full 5120-token continuation"
                    )
            state["scorer_repetitions"] += 1
        # Preserve the standard report's evidence links as well as every replay.
        standard = kwargs["evidence_dir"]
        standard.mkdir(parents=True, exist_ok=True)
        for artifact in dest.iterdir():
            if artifact.name not in ("reports.json", "timing.json"):
                shutil.copy2(artifact, standard / artifact.name)
        return reports

    try:
        with (
            patch("bench.main.EngineContainer", RecordedContainer),
            patch("bench.main.run_sla_engine", replay),
            patch("bench.main.grade_all", grade),
            patch("bench.main.capture_baseline_natural_stops", natural_stops),
        ):
            yield state
    finally:
        save(root / "lifecycle.json", state)


@contextmanager
def memory_samples(root):
    stop = threading.Event()
    errors = []
    peaks = {}

    def sample():
        with (root / "gpu-memory.jsonl").open("w") as stream:
            while not stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10,
                    )
                    for row in csv.reader(result.stdout.splitlines()):
                        _index, uuid, name, total, used, free, _utilization = [
                            v.strip() for v in row
                        ]
                        entry = peaks.setdefault(
                            uuid,
                            {
                                "name": name,
                                "total_mib": float(total),
                                "sampled_peak_used_mib": 0,
                                "sampled_min_free_mib": float(total),
                            },
                        )
                        entry["sampled_peak_used_mib"] = max(
                            entry["sampled_peak_used_mib"], float(used)
                        )
                        entry["sampled_min_free_mib"] = min(
                            entry["sampled_min_free_mib"], float(free)
                        )
                    phase_path = root / "round" / "phase.json"
                    phase = (
                        json.loads(phase_path.read_text())
                        if phase_path.exists()
                        else None
                    )
                    stream.write(
                        json.dumps(
                            {
                                "monotonic_s": time.monotonic(),
                                "phase": phase,
                                "csv": result.stdout,
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    errors.append(str(exc))
                    stop.set()
                stop.wait(0.5)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=15)
        save(
            root / "telemetry.json",
            {
                "sample_period_s": 0.5,
                "errors": errors,
                "gpus": peaks,
                "scope": "sampled device memory, not allocator peak; sub-sample peaks may be missed",
            },
        )
        if errors or not peaks or thread.is_alive():
            raise EngineError("GPU memory sampling failed; see telemetry.json")


def verify_hardware(root):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    save(root / "hardware.json", {"query": result.stdout})
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1 or "RTX PRO 6000" not in rows[0]:
        raise ValueError("requires a dedicated host exposing exactly one RTX PRO 6000")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-fields", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation-memory-fraction", type=float, required=True)
    parser.add_argument("--scorer-memory-fraction", type=float, required=True)
    parser.add_argument("--case", choices=("natural", "capacity"), default="natural")
    parser.add_argument(
        "--prefixes", choices=("distinct", "repeated"), default="distinct"
    )
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument(
        "--source-preview",
        type=Path,
        help="Existing preview directory (trace and receipt)",
    )
    parser.add_argument("--scorer-repetitions", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate inputs without Docker/GPU work",
    )
    args = parser.parse_args(argv)
    if (
        args.prompt_count < 1
        or args.scorer_repetitions < 3
        or not math.isfinite(args.request_timeout)
        or args.request_timeout <= 0
    ):
        parser.error(
            "positive prompt count/timeout and at least three scorer repetitions required"
        )
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "failed",
        "diagnostic_only": True,
        "case": args.case,
        "advertised_262144_context_qualified": False,
        "exact_21504_boundary_exercised": False,
    }
    started = time.monotonic()
    try:
        fields = prepare_fields(
            json.loads(args.campaign_fields.read_text()),
            args.generation_memory_fraction,
            args.scorer_memory_fraction,
        )
        save(
            root / "source_campaign_fields.json",
            json.loads(args.campaign_fields.read_text()),
        )
        save(root / "effective_campaign_fields.json", fields)
        save(root / "invocation.json", {"argv": sys.argv if argv is None else argv})
        if args.source_preview:
            source_dir = args.source_preview
        else:
            source_dir = root / "source_preview"
            build_source_preview(
                fields,
                source_dir,
                args.prompt_count if args.prefixes == "distinct" else 1,
            )
        source_path = source_dir / "workload_trace.json"
        receipt = json.loads((source_dir / "sampling_receipt.json").read_text())
        model = fields["bench"]["model"]
        if (
            receipt["sampled_trace_sha256"] != sha256_file(source_path)
            or receipt["dataset"] != fields["sampling_rule"]["dataset"]
            or receipt["revision"] != fields["sampling_rule"]["revision"]
            or receipt["chat_template"]["model_repo"] != model["hf_repo"]
            or receipt["chat_template"]["model_revision"] != model["hf_revision"]
        ):
            raise ValueError(
                "preview receipt does not match trace, dataset, or model pins"
            )
        save(root / "source_sampling_receipt.json", receipt)
        source = json.loads(source_path.read_text())
        save(root / "source_workload_trace.json", source)
        trace = longest_trace(
            source,
            fields,
            count=args.prompt_count,
            prefixes=args.prefixes,
            capacity=args.case == "capacity",
        )
        state["shorter_fallbacks"] = trace["meta"]["shorter_fallbacks"]
        state["shorter_fallback_count"] = len(state["shorter_fallbacks"])
        state["selection"] = trace["meta"]["selection"]
        print(
            f"Shorter input fallbacks: {state['shorter_fallback_count']}/{len(trace['requests'])}; output target remains {MAX_OUTPUT} tokens",
            flush=True,
        )
        trace_path = root / "workload_trace.json"
        save(trace_path, trace)
        formatter = build_prompt_formatter(
            fields["sampling_rule"],
            model_repo=model["hf_repo"],
            model_revision=model["hf_revision"],
        )
        for key, value in formatter.receipt.items():
            if receipt.get(key) != value:
                raise ValueError(
                    "source tokenizer/template receipt differs from pinned assets"
                )
        for request_input in trace["requests"]:
            ids = formatter.encode(request_input["prompt"])
            if (
                len(ids) != request_input["input_tokens"]
                or token_ids_sha256(ids) != request_input["input_ids_sha256"]
            ):
                raise ValueError(
                    "source input token counts/hashes failed CPU verification"
                )
        request = prepare_request(fields, trace_path)
        request_path = root / "bench_request.json"
        save(request_path, request)
        state.update(
            input_tokens=[r["input_tokens"] for r in trace["requests"]],
            requested_output_tokens=MAX_OUTPUT,
            trace_sha256=sha256_file(trace_path),
            request_sha256=sha256_file(request_path),
            generation_load="2 ms interval burst; not closed-loop concurrency",
            scorer_load="production sequential grading",
        )
        if args.prepare_only:
            state["status"] = "prepared_only"
            return 0
        lock = Path(REMOTE_LOCK)
        lock.parent.mkdir(parents=True, exist_ok=True)
        with host_lock(lock):
            verify_hardware(root)
            check_idle_gpu()
            with (
                memory_samples(root),
                diagnostic_hooks(
                    root,
                    fields,
                    trace,
                    capacity=args.case == "capacity",
                    timeout=args.request_timeout,
                    scorer_repetitions=args.scorer_repetitions,
                ) as lifecycle,
            ):
                code = harness.run_bench(request_path, root / "round")
            if (
                code
                or lifecycle["starts"]
                != ["baseline", "baseline-drift", "candidate-0", "scorer"]
                or lifecycle["scorer_repetitions"] != args.scorer_repetitions
            ):
                raise EngineError(f"round/scorer incomplete (harness exit {code})")
        for log in (root / "round" / "evidence" / "correctness" / "engine_logs").glob(
            "*.log"
        ):
            if re.search(
                r"out of memory|CUDA error|Traceback \(most recent call last\)",
                log.read_text(),
                re.IGNORECASE,
            ):
                raise EngineError(
                    "engine logs contain a memory or runtime error; review retained logs"
                )
        state["status"] = "diagnostic_completed"
        # Even a successful 16k-band test is not an exact-boundary capacity proof.
        state["exact_21504_boundary_exercised"] = args.case == "capacity" and all(
            n == MAX_INPUT for n in state["input_tokens"]
        )
        return 0
    except Exception as exc:  # noqa: BLE001 -- preserve diagnostic failures in evidence
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(state["error"], file=sys.stderr)
        return 1
    finally:
        state["elapsed_s"] = time.monotonic() - started
        save(root / "summary.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
