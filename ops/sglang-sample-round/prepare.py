"""Prepare a standalone sample with the campaign's sampler; no DB or chain."""

import hashlib
import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path

import config
from bench.longform import require_qualification
from bench.sampler import (
    LONGFORM_ALGO_VERSION,
    build_prompt_formatter,
    fetch_hf_row,
    generate_trace,
    parse_sampling_rule,
    sampling_context_for_rule,
)
from bench.validate import load_workload_trace, validate_bench_request_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s sample-round: %(message)s",
)
logger = logging.getLogger(__name__)

repo_root = Path(__file__).resolve().parents[2]
root = Path(sys.argv[1]).resolve()
fields = json.loads(
    (
        repo_root / "fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json"
    ).read_text()
)
bench = fields["bench"]
model = bench["model"]
cache = (
    config.BENCH_HF_CACHE_DIR
    / model["hf_repo"].replace("/", "--")
    / model["hf_revision"]
)
rule = parse_sampling_rule(
    json.loads(Path(sys.argv[2]).read_text())
    if len(sys.argv) > 2
    else fields["sampling_rule"]
)
if rule["algo_version"] == LONGFORM_ALGO_VERSION:
    require_qualification(rule, bench, fields["engine"])


def load_cached_tokenizer_config(**_):
    with (cache / "tokenizer_config.json").open(encoding="utf-8") as fh:
        tokenizer_config = json.load(fh)
    if not isinstance(tokenizer_config, dict):
        raise TypeError("tokenizer_config.json must contain an object")
    if not tokenizer_config.get("chat_template"):
        with (cache / "chat_template.jinja").open(encoding="utf-8", newline="") as fh:
            tokenizer_config["chat_template"] = fh.read()
    return tokenizer_config


logger.info("Loading pinned tokenizer from %s", cache)
formatter = build_prompt_formatter(
    rule,
    model_repo=model["hf_repo"],
    model_revision=model["hf_revision"],
    config_loader=load_cached_tokenizer_config,
    tokenizer_loader=lambda **_: (cache / "tokenizer.json").read_text(),
)
trace_path = root / "workload_trace.json"
if not trace_path.exists():
    logger.info(
        "Sampling %s prompts from %s@%s",
        rule["n_prompts"],
        rule["dataset"],
        rule["revision"],
    )
    fetched_rows = 0

    def fetch_row(index):
        global fetched_rows
        started = time.monotonic()
        logger.info("Fetching dataset row %s (fetch #%s)", index, fetched_rows + 1)
        row = fetch_hf_row(rule, index)
        fetched_rows += 1
        logger.info("Fetched row %s in %.1fs", index, time.monotonic() - started)
        return row

    sampled = generate_trace(
        rule=rule,
        seed_hex=hashlib.sha256(b"pareton-standalone-sglang-sample-v1").hexdigest(),
        row_fetcher=fetch_row,
        prompt_formatter=formatter,
        sampling_context=sampling_context_for_rule(rule, bench, fields["engine"]),
    )
    trace_path.write_bytes(sampled.body)
    (root / "sampling_receipt.json").write_text(json.dumps(sampled.receipt, indent=2))
    logger.info("Saved trace and sampling receipt after %s row fetches", fetched_rows)
else:
    receipt = json.loads((root / "sampling_receipt.json").read_text())
    replayed = generate_trace(
        rule=rule,
        seed_hex=receipt["seed_hex"],
        row_fetcher=lambda i: fetch_hf_row(rule, i),
        prompt_formatter=formatter,
        sampling_context=sampling_context_for_rule(rule, bench, fields["engine"]),
        sampling_receipt=receipt,
        sample_seed_block=receipt["sample_seed_block"],
        sample_seed_block_hash=receipt["sample_seed_block_hash"],
    )
    if trace_path.read_bytes() != replayed.body:
        raise ValueError(
            "existing workload differs from its receipt; use a fresh sample directory"
        )
    logger.info("Verified existing workload trace and receipt: %s", trace_path)


def image_id(ref):
    logger.info("Resolving local Docker image: %s", ref)
    result = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", ref], text=True
    ).strip()
    logger.info("Resolved image ID: %s", result)
    return result


def engine(ref):
    # Match worker.round_job.build_round_request: fixture args are extras only.
    serve_args = [
        "--model-path",
        "/model",
        "--context-length",
        str(model["max_model_len"]),
        "--dtype",
        str(model.get("dtype") or "bfloat16"),
    ]
    quantization = model.get("quantization")
    if quantization is not None and str(quantization).strip():
        serve_args.extend(["--quantization", str(quantization)])
    serve_args.extend(str(arg) for arg in bench.get("serve_args", []))
    return {
        "name": "sglang",
        "image": image_id(ref),
        "serve_args": serve_args,
        "env": {},
        "cache_dir": fields["engine"]["cache_dir"],
    }


request = {
    "schema_version": 1,
    "task_id": str(uuid.uuid4()),
    "mode": "all",
    "model": model,
    "hardware": {"gpu_count": bench["gpu_count"], "gpu_sku_expected": "RTX5090"},
    "engines": {
        "baseline": engine(bench["baseline_engine_image_digest"]),
        "candidates": [engine("pareton-sample:sglang-minimal")],
    },
    "workload_trace": {
        "path": str(trace_path),
        "sha256": "sha256:" + hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    },
    "correctness": bench["correctness"],
    "sla_bench": {
        "repetitions": config.BENCH_SLA_REPETITIONS,
        "thresholds": {
            key: fields["sla"][key] for key in ("p99_ttft_ms", "p99_itl_ms")
        },
    },
    "scoring_rule": fields["scoring_rule"],
}
validate_bench_request_dict(request)
trace = load_workload_trace(
    trace_path, expected_sha256=request["workload_trace"]["sha256"]
)
logger.info(
    "Validated request: GPUs=%s prompts=%s repetitions=%s context=%s trace=%s",
    bench["gpu_count"],
    len(trace.requests),
    config.BENCH_SLA_REPETITIONS,
    model["max_model_len"],
    request["workload_trace"]["sha256"],
)
(root / "bench_request.json").write_text(json.dumps(request, indent=2) + "\n")
logger.info("Request ready: %s", root / "bench_request.json")
