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
from bench.http import get_json, post_completion_stream
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
from bench.schemas import ModelSpec, TraceMeta, WorkloadTrace
from bench.sla_bench import (
    aggregate_rep_metrics,
    capture_baseline_natural_stops,
    run_sla_engine,
)
from bench.trajectory import token_ids_sha256
from bench.validate import (
    sha256_file,
    validate_bench_request_dict,
    validate_workload_trace_dict,
)
from bench.weights import stage_weights
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


# Reviewed hardware/model pairs. Any v4 or v5 LongWriter fixture replays the
# diagnostic's own 2 ms burst trace; v5's closed-loop schedule is not exercised.
PROFILES = {
    ("RTXPRO6000", "Qwen/Qwen3.8-27B-FP8"): {
        "gpu_count": 1,
        "gpu_name": "RTX PRO 6000",
        "tp_size": 1,
        "quantization": "fp8",
    },
    # MXFP4 is detected from the checkpoint, so the fixture pins no quantization.
    # The harness stages only /model; this diagnostic stages the DSPARK draft.
    ("B300", "moonshotai/Kimi-K3"): {
        "gpu_count": 8,
        "gpu_name": "B300",
        "tp_size": 8,
        "quantization": None,
        "draft": {
            "hf_repo": "RadixArk/Kimi-K3-DSpark",
            "hf_revision": "3c5bac301d9cf392706189d82ed947feca6c2f0f",
            "mount": "/root/models/kimi-k3-dspark",
        },
    },
}


def profile_for(fields):
    skus = fields["gpu_skus"]
    key = (skus[0] if len(skus) == 1 else None, fields["bench"]["model"]["hf_repo"])
    if key not in PROFILES:
        raise ValueError(
            "requires a fixture matching a reviewed profile: "
            + ", ".join(f"{sku} {repo}" for sku, repo in PROFILES)
        )
    return PROFILES[key]


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
    profile = profile_for(fields)
    if (
        fields["engine"]["name"] != "sglang"
        or bench["gpu_count"] != profile["gpu_count"]
        or model.get("quantization") != profile["quantization"]
        or fields["sampling_rule"]["dataset"] != "zai-org/LongWriter-6k"
        or fields["sampling_rule"]["algo_version"] not in (4, 5)
    ):
        raise ValueError(
            f"requires an SGLang LongWriter v4/v5 fixture with {profile['gpu_count']} "
            f"GPU(s) and quantization {profile['quantization']}"
        )
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
            int(effective.get("tp_size", 1)) != profile["tp_size"]
            or int(effective.get("pp_size", 1)) != 1
        ):
            raise ValueError(f"diagnostic requires TP{profile['tp_size']}/PP1")
    draft = profile.get("draft")
    if draft is not None:
        for args in (bench["serve_args"], bench["serve_args"] + corr["serve_args"]):
            paths = [
                args[i + 1]
                for i, arg in enumerate(args[:-1])
                if arg
                in ("--speculative-draft-model-path", "--speculative-draft-model")
            ]
            if not paths or paths[-1] != draft["mount"]:
                raise ValueError(f"draft model path must be {draft['mount']}")
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


def review_engine_logs(root):
    """Review saved logs without changing the run's original summary or verdict."""
    paths = sorted((root / "round/evidence/correctness/engine_logs").glob("*.log"))
    if not paths:
        raise EngineError("no retained engine logs found")
    warnings, failures = [], []
    # Only a complete, explicitly ignored optional audio import block is exempt.
    optional = re.compile(
        r"Ignore import error when loading sglang\.srt\.multimodal\.processors\.[A-Za-z_][A-Za-z_0-9]*: "
        r"Could not load libtorchcodec\.(?:(?!\n\[\d{4}-).)*?"
        r"\[start of libtorchcodec loading traceback\]"
        r"(?:(?!\n\[\d{4}-).)*?\[end of libtorchcodec loading traceback\]",
        re.DOTALL,
    )
    context_warning = re.compile(
        r"Warning: (?:User-specified|Target model's) context_length \(262151\) is greater than the derived "
        r"context_length \(262144\)\. This may lead to incorrect model outputs or CUDA errors\. "
        r"Note that the derived context_length may differ from max_position_embeddings in the model's config\."
    )
    fatal = re.compile(
        r"out of memory|CUDA error|Traceback \(most recent call last\)", re.IGNORECASE
    )
    for path in paths:
        content = path.read_text()
        lines = content.splitlines()
        ignored_spans = []
        for match in optional.finditer(content):
            # Never exempt OOM/CUDA failures embedded in an optional import block.
            if not re.search(r"out of memory|CUDA error", match.group(), re.IGNORECASE):
                ignored_spans.append((match.start(), match.end()))
                warnings.append(
                    {
                        "file": path.name,
                        "line": content.count("\n", 0, match.start()) + 1,
                        "kind": "ignored_optional_audio_import",
                    }
                )
        for match in fatal.finditer(content):
            line_number = content.count("\n", 0, match.start()) + 1
            line = lines[line_number - 1]
            if any(start <= match.start() < end for start, end in ignored_spans):
                continue
            if context_warning.search(line) and len(list(fatal.finditer(line))) == 1:
                warnings.append(
                    {
                        "file": path.name,
                        "line": line_number,
                        "kind": "scorer_context_headroom_warning",
                        "text": line,
                    }
                )
                continue
            failures.append({"file": path.name, "line": line_number, "text": line})
    result = {
        "status": "failed" if failures else "passed",
        "warnings": warnings,
        "failures": failures,
        "files_reviewed": [p.name for p in paths],
        "scope": "log review only; does not change original run summary",
    }
    save(root / "engine_log_review.json", result)
    if failures:
        first = failures[0]
        raise EngineError(
            f"engine log error at {first['file']}:{first['line']}; see engine_log_review.json"
        )
    print(
        f"Log review passed; {len(warnings)} known warnings retained in engine_log_review.json",
        flush=True,
    )
    return result


class DockerModelVolume:
    """Copy locally verified weights through Docker's API, avoiding host binds."""

    def __init__(self, root):
        self.root = root
        self.name = "pareton-pro6000-model-" + uuid4().hex
        self.source = None
        self.created = False

    def command(self, *args):
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=7200, check=False
        )
        if result.returncode:
            raise EngineError(f"model volume {args[0]} failed: {result.stderr}")
        return result.stdout

    def prepare(self, source, image):
        source = source.resolve()
        if self.source is not None:
            if source != self.source:
                raise EngineError("model path changed during diagnostic")
            return
        print(
            "Copying staged weights into a Docker-managed model volume...", flush=True
        )
        expected = {
            str(path.relative_to(source)): sha256_file(path)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        }
        if not expected or any(path.is_symlink() for path in source.rglob("*")):
            raise EngineError(
                "model volume requires nonempty, symlink-free staged weights"
            )
        self.command("pull", image)
        self.command("volume", "create", self.name)
        self.created = True
        helper = self.name + "-copy"
        try:
            self.command(
                "create",
                "--name",
                helper,
                "--network",
                "none",
                "--mount",
                f"type=volume,src={self.name},dst=/model,volume-nocopy",
                "--entrypoint",
                "/bin/true",
                image,
            )
            self.command("cp", str(source) + "/.", helper + ":/model")
        finally:
            self.command("rm", "-f", helper)
        # Read back every copied file inside the daemon's mount namespace.
        verify = """import hashlib, json, pathlib
root = pathlib.Path('/model')
result = {}
for path in sorted(root.rglob('*')):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        result[str(path.relative_to(root))] = "sha256:" + digest.hexdigest()
print(json.dumps(result))
"""
        actual = json.loads(
            self.command(
                "run",
                "--rm",
                "--network",
                "none",
                "--mount",
                f"type=volume,src={self.name},dst=/model,readonly,volume-nocopy",
                "--entrypoint",
                "python3",
                image,
                "-c",
                verify,
            )
        )
        if actual != expected:
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            mismatched = sorted(
                key
                for key in expected.keys() & actual.keys()
                if expected[key] != actual[key]
            )
            save(
                self.root / "model_volume.json",
                {
                    "name": self.name,
                    "source": str(source),
                    "verified": False,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "missing": missing,
                    "extra": extra,
                    "mismatched": mismatched,
                },
            )
            raise EngineError(
                "Docker model volume file hashes differ from staged weights: "
                f"{len(missing)} missing, {len(extra)} extra, "
                f"{len(mismatched)} mismatched; see model_volume.json"
            )
        self.source = source
        save(
            self.root / "model_volume.json",
            {
                "name": self.name,
                "source": str(source),
                "sha256": expected,
                "verified": True,
            },
        )
        print("Docker model volume verified; starting engine lifecycle.", flush=True)

    def wrap_runner(self, runner):
        def run(cmd, **kwargs):
            cmd = list(cmd)
            if cmd[:2] == ["docker", "run"]:
                old = f"{self.source}:/model:ro"
                matches = [
                    i
                    for i in range(1, len(cmd))
                    if cmd[i - 1] == "-v" and cmd[i] == old
                ]
                if len(matches) != 1:
                    raise EngineError("expected exactly one staged model bind mount")
                i = matches[0]
                cmd[i - 1 : i + 1] = [
                    "--mount",
                    f"type=volume,src={self.name},dst=/model,readonly,volume-nocopy",
                ]
            return runner(cmd, **kwargs)

        return run

    def close(self):
        if self.created:
            self.command("volume", "rm", self.name)


@contextmanager
def diagnostic_hooks(
    root,
    fields,
    trace,
    *,
    capacity,
    timeout,
    scorer_repetitions,
    docker_model_volume=False,
):
    """Hooks are confined to this CLI process; shared harness files stay untouched."""
    state = {"starts": [], "scorer_repetitions": 0}
    context = sampling_context_for_campaign(fields["bench"], fields["engine"])
    preflight_trace = WorkloadTrace.from_dict(trace)
    preflight_trace.meta = TraceMeta("diagnostic", sampling={"context": context})
    count = len(trace["requests"])
    volume = DockerModelVolume(root) if docker_model_volume else None
    gpu_count = fields["bench"]["gpu_count"]
    draft = profile_for(fields).get("draft")
    if draft is not None and docker_model_volume:
        raise ValueError("--docker-model-volume does not stage the draft model")
    draft_dir = None
    if draft is not None:
        staged = stage_weights(
            ModelSpec(
                hf_repo=draft["hf_repo"],
                hf_revision=draft["hf_revision"],
                dtype=fields["bench"]["model"]["dtype"],
                quantization=None,
                max_model_len=fields["bench"]["model"]["max_model_len"],
            )
        )
        draft_dir = staged.path.resolve()
        save(
            root / "draft_model.json",
            {
                **draft,
                "path": str(draft_dir),
                "weights_sha256": staged.weights_sha256,
            },
        )

    def mount_draft(runner):
        def run(cmd, **kwargs):
            cmd = list(cmd)
            if cmd[:2] == ["docker", "run"]:
                cmd[2:2] = ["-v", f"{draft_dir}:{draft['mount']}:ro"]
            return runner(cmd, **kwargs)

        return run

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
                    "docker_model_volume": volume.name if volume else None,
                },
            )
            if self.gpu_count != gpu_count:
                raise EngineError("harness changed the requested GPU count")
            if volume is not None:
                volume.prepare(self.weights_dir, self.spec.image)
                self.runner = volume.wrap_runner(self.runner)
            if draft_dir is not None:
                self.runner = mount_draft(self.runner)
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

    def diagnostic_stream(url, **kwargs):
        return post_completion_stream(url, **{**kwargs, "require_token_timing": False})

    def diagnostic_metrics(rows, **kwargs):
        return aggregate_rep_metrics(rows, **{**kwargs, "require_token_timing": False})

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
            patch("bench.sla_bench.post_completion_stream", diagnostic_stream),
            patch("bench.sla_bench.aggregate_rep_metrics", diagnostic_metrics),
            patch("bench.main.run_sla_engine", replay),
            patch("bench.main.grade_all", grade),
            patch("bench.main.capture_baseline_natural_stops", natural_stops),
        ):
            yield state
    finally:
        save(root / "lifecycle.json", state)
        if volume is not None:
            volume.close()


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


def verify_hardware(root, profile):
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
    if len(rows) != profile["gpu_count"] or any(
        profile["gpu_name"] not in row for row in rows
    ):
        raise ValueError(
            f"requires a dedicated host exposing exactly {profile['gpu_count']} "
            f"{profile['gpu_name']} GPU(s)"
        )


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
    parser.add_argument(
        "--docker-model-volume",
        action="store_true",
        help="Copy and verify weights in a temporary Docker volume; avoid host bind mounts",
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
        "performance_score_valid": False,
        "stream_timing": "SSE chunk gaps, not per-token ITL; SLA and speedup scores are not qualification evidence",
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
            verify_hardware(root, profile_for(fields))
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
                    docker_model_volume=args.docker_model_volume,
                ) as lifecycle,
            ):
                print(
                    "Correctness-only diagnostic: accepting multi-token SSE chunks; performance scores are invalid.",
                    flush=True,
                )
                code = harness.run_bench(request_path, root / "round")
                report_path = root / "round" / "bench_report.json"
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                    report["diagnostic_only"] = True
                    report["performance_score_valid"] = False
                    report["stream_timing"] = state["stream_timing"]
                    save(report_path, report)
            if (
                code
                or lifecycle["starts"]
                != ["baseline", "baseline-drift", "candidate-0", "scorer"]
                or lifecycle["scorer_repetitions"] != args.scorer_repetitions
            ):
                raise EngineError(f"round/scorer incomplete (harness exit {code})")
        review_engine_logs(root)
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
