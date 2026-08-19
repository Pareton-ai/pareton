"""Load and validate bench_request.json / workload_trace.json."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from bench.schemas import BenchRequest, WorkloadTrace

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# Bare digest, or registry ref pinned with @sha256:<64 hex> (optional #fragment).
_IMAGE_DIGEST_RE = re.compile(
    r"^(?:sha256:[0-9a-f]{64}|.+@sha256:[0-9a-f]{64})(?:#.*)?$",
    re.IGNORECASE,
)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
# HuggingFace org/name — blocks path traversal and cache-slug ambiguity.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RequestValidationError(ValueError):
    """Raised when bench_request.json fails schema / semantic checks (exit 1)."""


def extract_image_digest(image_ref: str) -> str:
    """Return canonical sha256:<64 hex> from a digest-pinned image ref.

    Accepts bare ``sha256:...`` or ``registry/name@sha256:...``.
    Raises RequestValidationError if the ref is not digest-pinned.
    """
    s = str(image_ref).strip()
    if not _IMAGE_DIGEST_RE.match(s):
        raise RequestValidationError(
            f"engine image must be digest-referenced "
            f"(sha256:<64 hex> or name@sha256:<64 hex>), got {image_ref!r}"
        )
    # Strip optional URL fragment, then take the digest suffix (case-insensitive).
    s = s.split("#", 1)[0]
    lower = s.lower()
    if lower.startswith("sha256:"):
        hex_part = s.split(":", 1)[1]
    else:
        idx = lower.rfind("@sha256:")
        hex_part = s[idx + len("@sha256:") :]
    digest = f"sha256:{hex_part.lower()}"
    if not _SHA256_RE.match(digest):
        raise RequestValidationError(
            f"engine image digest malformed after parse: {digest!r} from {image_ref!r}"
        )
    return digest


def is_digest_pinned_image(image_ref: str) -> bool:
    try:
        extract_image_digest(image_ref)
        return True
    except RequestValidationError:
        return False


def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RequestValidationError(f"cannot read {path}: {exc}") from exc
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestValidationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RequestValidationError(f"{path}: top-level JSON must be an object")
    return obj


def _require_keys(d: dict[str, Any], keys: list[str], *, ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise RequestValidationError(f"{ctx}: missing required fields: {missing}")


def validate_bench_request_dict(d: dict[str, Any]) -> BenchRequest:
    """Validate a parsed request dict; raise RequestValidationError on failure."""
    try:
        return _validate_bench_request_dict(d)
    except RequestValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RequestValidationError(str(exc)) from exc


def _validate_bench_request_dict(d: dict[str, Any]) -> BenchRequest:
    _require_keys(
        d,
        [
            "schema_version",
            "task_id",
            "mode",
            "model",
            "hardware",
            "engines",
            "workload_trace",
            "correctness",
            "sla_bench",
        ],
        ctx="bench_request",
    )
    if int(d["schema_version"]) != 1:
        raise RequestValidationError(
            f"unsupported schema_version {d['schema_version']!r} (expected 1)"
        )
    if not _UUID_RE.match(str(d["task_id"])):
        raise RequestValidationError(f"task_id must be a UUID, got {d['task_id']!r}")

    model = d["model"]
    if not isinstance(model, dict):
        raise RequestValidationError("model must be an object")
    _require_keys(
        model,
        ["hf_repo", "hf_revision", "dtype", "max_model_len"],
        ctx="model",
    )
    if "quantization" not in model:
        raise RequestValidationError("model: missing required fields: ['quantization']")
    if not _GIT_SHA_RE.match(str(model["hf_revision"])):
        raise RequestValidationError(
            f"model.hf_revision must be a git commit sha, got {model['hf_revision']!r}"
        )
    hf_repo = str(model["hf_repo"])
    if ".." in hf_repo or not _HF_REPO_RE.match(hf_repo):
        raise RequestValidationError(
            f"model.hf_repo must be a HuggingFace org/name id, got {hf_repo!r}"
        )

    hardware = d["hardware"]
    if not isinstance(hardware, dict):
        raise RequestValidationError("hardware must be an object")
    _require_keys(hardware, ["gpu_count", "gpu_sku_expected"], ctx="hardware")
    if int(hardware["gpu_count"]) < 1:
        raise RequestValidationError("hardware.gpu_count must be >= 1")

    engines = d["engines"]
    if not isinstance(engines, dict):
        raise RequestValidationError("engines must be an object")
    _require_keys(engines, ["baseline", "candidate"], ctx="engines")
    for role in ("baseline", "candidate"):
        eng = engines[role]
        if not isinstance(eng, dict):
            raise RequestValidationError(f"engines.{role} must be an object")
        _require_keys(eng, ["image"], ctx=f"engines.{role}")
        try:
            extract_image_digest(str(eng["image"]))
        except RequestValidationError as exc:
            raise RequestValidationError(f"engines.{role}.image: {exc}") from exc

    wt = d["workload_trace"]
    if not isinstance(wt, dict):
        raise RequestValidationError("workload_trace must be an object")
    _require_keys(wt, ["path", "sha256"], ctx="workload_trace")
    if not _SHA256_RE.match(str(wt["sha256"]).lower()):
        raise RequestValidationError(
            f"workload_trace.sha256 must match sha256:<64 hex>, got {wt['sha256']!r}"
        )

    corr = d["correctness"]
    if not isinstance(corr, dict):
        raise RequestValidationError("correctness must be an object")
    _require_keys(
        corr, ["num_prompts", "max_new_tokens", "thresholds"], ctx="correctness"
    )
    if int(corr["num_prompts"]) < 1:
        raise RequestValidationError("correctness.num_prompts must be >= 1")
    if int(corr["max_new_tokens"]) < 1:
        raise RequestValidationError("correctness.max_new_tokens must be >= 1")
    if "min_positions_compared" in corr and int(corr["min_positions_compared"]) < 1:
        raise RequestValidationError("correctness.min_positions_compared must be >= 1")
    thr = corr["thresholds"]
    if not isinstance(thr, dict):
        raise RequestValidationError("correctness.thresholds must be an object")
    _require_keys(
        thr,
        ["mean_abs_logprob_diff", "max_abs_logprob_diff", "argmax_mismatch_rate"],
        ctx="correctness.thresholds",
    )

    sla = d["sla_bench"]
    if not isinstance(sla, dict):
        raise RequestValidationError("sla_bench must be an object")
    _require_keys(
        sla, ["repetitions", "warmup_requests", "thresholds"], ctx="sla_bench"
    )
    sla_thr = sla["thresholds"]
    if not isinstance(sla_thr, dict):
        raise RequestValidationError("sla_bench.thresholds must be an object")
    _require_keys(sla_thr, ["p99_ttft_ms", "p99_itl_ms"], ctx="sla_bench.thresholds")

    return BenchRequest.from_dict(d)


def load_bench_request(path: Path) -> tuple[BenchRequest, bytes]:
    """Load + validate request file. Returns (request, raw_bytes) for fingerprinting."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RequestValidationError(f"cannot read {path}: {exc}") from exc
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestValidationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RequestValidationError(f"{path}: top-level JSON must be an object")
    return validate_bench_request_dict(obj), raw


def validate_workload_trace_dict(d: dict[str, Any]) -> WorkloadTrace:
    try:
        return _validate_workload_trace_dict(d)
    except RequestValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RequestValidationError(f"workload_trace: {exc}") from exc


def _validate_workload_trace_dict(d: dict[str, Any]) -> WorkloadTrace:
    _require_keys(d, ["schema_version", "requests"], ctx="workload_trace")
    if int(d["schema_version"]) != 1:
        raise RequestValidationError(
            f"unsupported trace schema_version {d['schema_version']!r} (expected 1)"
        )
    if not isinstance(d["requests"], list) or not d["requests"]:
        raise RequestValidationError("workload_trace.requests must be a non-empty list")
    seen_ids: set[str] = set()
    for i, req in enumerate(d["requests"]):
        if not isinstance(req, dict):
            raise RequestValidationError(
                f"workload_trace.requests[{i}] must be an object"
            )
        rid = req.get("id")
        if rid is None or str(rid).strip() == "":
            raise RequestValidationError(
                f"workload_trace.requests[{i}]: id must be a non-empty string"
            )
        rid_s = str(rid)
        if rid_s in seen_ids:
            raise RequestValidationError(
                f"workload_trace.requests: duplicate id {rid_s!r}"
            )
        seen_ids.add(rid_s)
    return WorkloadTrace.from_dict(d)


def load_workload_trace(
    path: Path, *, expected_sha256: str | None = None
) -> WorkloadTrace:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RequestValidationError(f"cannot read trace {path}: {exc}") from exc
    if expected_sha256 is not None:
        actual = sha256_bytes(raw)
        if actual.lower() != expected_sha256.lower():
            raise RequestValidationError(
                f"trace sha256 mismatch: expected {expected_sha256}, got {actual}"
            )
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestValidationError(f"invalid JSON in trace {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RequestValidationError(f"{path}: top-level JSON must be an object")
    return validate_workload_trace_dict(obj)


def validate_report_dict(d: dict[str, Any]) -> None:
    """Lightweight structural check that a report dict matches the expected skeleton."""
    _require_keys(
        d,
        [
            "schema_version",
            "task_id",
            "verdict",
            "started_at",
            "finished_at",
            "environment",
            "inputs_fingerprint",
        ],
        ctx="bench_report",
    )
    if d["verdict"] not in (
        "pass",
        "fail_correctness",
        "fail_sla",
        "error",
    ):
        raise RequestValidationError(f"invalid verdict {d['verdict']!r}")
    env = d["environment"]
    _require_keys(
        env,
        [
            "gpu",
            "driver_version",
            "cuda_version",
            "docker_version",
            "harness_version",
            "hostname_hash",
        ],
        ctx="environment",
    )
    fp = d["inputs_fingerprint"]
    _require_keys(
        fp,
        [
            "baseline_image_digest",
            "candidate_image_digest",
            "model_repo",
            "model_revision",
            "model_weights_sha256",
            "trace_sha256",
            "request_sha256",
        ],
        ctx="inputs_fingerprint",
    )
