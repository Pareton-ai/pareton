"""CLI entrypoint: python -m bench --request ... --output-dir ...

Exit codes (from outsource spec §4.1; CLI name overridden by roadmap):
  0 = harness completed (pass/fail is in the report)
  1 = bad request / schema validation failure
  2 = environment error
  3 = engine failure
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from bench import __version__
from bench.env import (
    collect_env_raw_dumps,
    collect_environment,
    warn_gpu_sku_mismatch,
)
from bench.output import JsonlFileHandler, OutputLayout
from bench.schemas import BenchReport, InputsFingerprint
from bench.validate import (
    RequestValidationError,
    load_bench_request,
    sha256_bytes,
    validate_report_dict,
)

EXIT_OK = 0
EXIT_BAD_REQUEST = 1
EXIT_ENV = 2
EXIT_ENGINE = 3

logger = logging.getLogger("bench")


def _extract_digest(image_ref: str) -> str:
    """Pull sha256:... out of an image reference for inputs_fingerprint."""
    if "@sha256:" in image_ref:
        return "sha256:" + image_ref.split("@sha256:", 1)[1]
    if image_ref.startswith("sha256:"):
        return image_ref
    # Spec requires digest refs; validation already enforces this. Fallback:
    return image_ref


def build_stub_report(
    *,
    request_raw: bytes,
    task_id: str,
    model_repo: str,
    model_revision: str,
    baseline_image: str,
    candidate_image: str,
    trace_sha256: str,
    env,
) -> BenchReport:
    now = datetime.now(timezone.utc)
    iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return BenchReport(
        schema_version=1,
        task_id=task_id,
        verdict="error",
        started_at=iso,
        finished_at=iso,
        environment=env,
        inputs_fingerprint=InputsFingerprint(
            baseline_image_digest=_extract_digest(baseline_image),
            candidate_image_digest=_extract_digest(candidate_image),
            model_repo=model_repo,
            model_revision=model_revision,
            model_weights_sha256="sha256:" + ("0" * 64),  # not downloaded yet (WS-B5)
            trace_sha256=trace_sha256,
            request_sha256=sha256_bytes(request_raw),
        ),
        stub_note=(
            "skeleton/stub run: Modules A/B/C not implemented yet "
            f"(bench {__version__}). Request validated; environment fingerprinted."
        ),
    )


def run_stub(request_path: Path, output_dir: Path) -> int:
    try:
        req, raw = load_bench_request(request_path)
    except RequestValidationError as exc:
        print(f"error: invalid request: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST

    layout = OutputLayout(output_dir)
    layout.prepare()

    # Structured log into harness.log
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = JsonlFileHandler(layout)
    handler.setFormatter(logging.Formatter("%(asctime)s"))
    root_logger.addHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        root_logger.addHandler(sh)

    env = collect_environment()
    warn = warn_gpu_sku_mismatch(env, req.hardware.gpu_sku_expected)
    if warn:
        logger.warning(warn)

    layout.write_env_dumps(collect_env_raw_dumps())
    layout.append_log(
        {
            "event": "stub_start",
            "task_id": req.task_id,
            "mode": req.mode,
            "harness_version": __version__,
        }
    )

    report = build_stub_report(
        request_raw=raw,
        task_id=req.task_id,
        model_repo=req.model.hf_repo,
        model_revision=req.model.hf_revision,
        baseline_image=req.engines.baseline.image,
        candidate_image=req.engines.candidate.image,
        trace_sha256=req.workload_trace.sha256,
        env=env,
    )
    report_dict = report.to_dict()
    try:
        validate_report_dict(report_dict)
    except RequestValidationError as exc:
        print(f"error: stub report failed self-validation: {exc}", file=sys.stderr)
        return EXIT_ENV

    layout.write_report(report)
    layout.append_log({"event": "stub_done", "report": str(layout.report_path)})
    print(f"wrote {layout.report_path}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description="Pareton bench harness (Stages 1–3). Stub mode until modules land.",
    )
    p.add_argument("--request", required=True, type=Path, help="Path to bench_request.json")
    p.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    p.add_argument(
        "--mock-engine",
        action="store_true",
        help="Reserved: use in-process mock engines (Modules A–C; not yet wired).",
    )
    p.add_argument("--version", action="version", version=f"bench {__version__}")
    args = p.parse_args(argv)

    # --mock-engine is accepted so future Module A wiring doesn't break the CLI;
    # stub path does not start engines yet.
    if args.mock_engine:
        logger.info("--mock-engine set (no-op in stub; engines start in WS-B3/B4)")

    return run_stub(args.request, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
