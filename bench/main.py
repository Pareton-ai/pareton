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
from bench.correctness import (
    PromptCase,
    collect_baseline_correctness,
    finish_correctness_with_candidate,
    load_correctness_prompts,
    run_correctness,
)
from bench.env import (
    collect_env_raw_dumps,
    collect_environment,
    warn_gpu_sku_mismatch,
)
from bench.lifecycle import (
    BenchNetwork,
    EngineContainer,
    EngineError,
    new_run_id,
)
from bench.output import JsonlFileHandler, OutputLayout
from bench.schemas import (
    BenchReport,
    BenchRequest,
    CorrectnessReport,
    InputsFingerprint,
)
from bench.validate import (
    RequestValidationError,
    extract_image_digest,
    load_bench_request,
    sha256_bytes,
    validate_report_dict,
)

EXIT_OK = 0
EXIT_BAD_REQUEST = 1
EXIT_ENV = 2
EXIT_ENGINE = 3

logger = logging.getLogger("bench")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_inputs_fingerprint(
    *,
    request_raw: bytes,
    req: BenchRequest,
    baseline_digest: str,
    candidate_digest: str,
) -> InputsFingerprint:
    return InputsFingerprint(
        baseline_image_digest=baseline_digest,
        candidate_image_digest=candidate_digest,
        model_repo=req.model.hf_repo,
        model_revision=req.model.hf_revision,
        model_weights_sha256="sha256:" + ("0" * 64),  # WS-B5
        trace_sha256=req.workload_trace.sha256,
        request_sha256=sha256_bytes(request_raw),
    )


def _attach_logging(layout: OutputLayout) -> list[logging.Handler]:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    added: list[logging.Handler] = []
    handler = JsonlFileHandler(layout)
    handler.setFormatter(logging.Formatter("%(asctime)s"))
    root_logger.addHandler(handler)
    added.append(handler)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        root_logger.addHandler(sh)
        added.append(sh)
    return added


def _detach_logging(handlers: list[logging.Handler]) -> None:
    root_logger = logging.getLogger()
    for h in handlers:
        root_logger.removeHandler(h)
        h.close()


def _write_validated_report(layout: OutputLayout, report: BenchReport) -> int:
    report_dict = report.to_dict()
    try:
        validate_report_dict(report_dict)
    except RequestValidationError as exc:
        print(
            f"error: report failed self-validation: {exc}",
            file=sys.stderr,
        )
        return EXIT_BAD_REQUEST
    layout.write_report(report)
    layout.append_log({"event": "done", "report": str(layout.report_path)})
    print(f"wrote {layout.report_path}")
    return EXIT_OK


def run_correctness_against_urls(
    *,
    req: BenchRequest,
    layout: OutputLayout,
    prompts: list[PromptCase],
    baseline_url: str,
    candidate_url: str,
) -> CorrectnessReport:
    return run_correctness(
        baseline_url,
        candidate_url,
        prompts=prompts,
        cfg=req.correctness,
        task_id=req.task_id,
        evidence_dir=layout.correctness_dir,
    )


def run_with_mock_engines(
    *,
    req: BenchRequest,
    layout: OutputLayout,
    prompts: list[PromptCase],
    tampered_candidate: bool,
) -> tuple[CorrectnessReport, str, str]:
    """In-process mock engines. Digests come from the request image pins."""
    from bench.mock_engine import MockEngine, MockEngineConfig

    baseline_digest = extract_image_digest(req.engines.baseline.image)
    candidate_digest = extract_image_digest(req.engines.candidate.image)
    with (
        MockEngine(
            MockEngineConfig(model="mock-baseline", host="127.0.0.1", port=0)
        ) as base,
        MockEngine(
            MockEngineConfig(
                model="mock-candidate",
                host="127.0.0.1",
                port=0,
                tampered=tampered_candidate,
            )
        ) as cand,
    ):
        report = run_correctness_against_urls(
            req=req,
            layout=layout,
            prompts=prompts,
            baseline_url=base.base_url,
            candidate_url=cand.base_url,
        )
    return report, baseline_digest, candidate_digest


def _should_pull_image(image: str) -> bool:
    """Bare ``sha256:<id>`` is a local image id — do not docker pull."""
    return not image.strip().lower().startswith("sha256:")


def _effective_gpu_count(requested: int) -> int:
    """Honor hardware.gpu_count only when NVIDIA runtime is present.

    CPU/mock-engine CI hosts reject ``--gpus``; GPU pods have nvidia-smi.
    """
    if requested <= 0:
        return 0
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return requested if r.returncode == 0 else 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0


def run_with_docker_engines(
    *,
    req: BenchRequest,
    layout: OutputLayout,
    prompts: list[PromptCase],
) -> tuple[CorrectnessReport, str, str]:
    """Containerized engines via B3 lifecycle, sequential baseline then candidate.

    Spec §3 allows Module A coexistence, but two full-size vLLM loads on one GPU
    commonly OOM. Generate+score on baseline, tear it down, then score candidate.
    Engines use an ``--internal`` network with no published ports.
    """
    run_id = new_run_id()
    logs_dir = layout.correctness_dir / "engine_logs"
    gpu_count = _effective_gpu_count(req.hardware.gpu_count)
    with BenchNetwork(run_id=run_id, internal=True) as net:
        with EngineContainer(
            spec=req.engines.baseline,
            network=net,
            role="baseline",
            gpu_count=gpu_count,
            publish_port=False,
            pull=_should_pull_image(req.engines.baseline.image),
            logs_dir=logs_dir,
        ) as base:
            baseline_phase = collect_baseline_correctness(
                base.base_url,
                prompts=prompts,
                cfg=req.correctness,
                task_id=req.task_id,
            )
            baseline_digest = base.image_digest
        with EngineContainer(
            spec=req.engines.candidate,
            network=net,
            role="candidate",
            gpu_count=gpu_count,
            publish_port=False,
            pull=_should_pull_image(req.engines.candidate.image),
            logs_dir=logs_dir,
        ) as cand:
            report = finish_correctness_with_candidate(
                cand.base_url,
                baseline_phase,
                cfg=req.correctness,
                evidence_dir=layout.correctness_dir,
            )
            return report, baseline_digest, cand.image_digest


def build_stub_remainder_report(
    *,
    request_raw: bytes,
    req: BenchRequest,
    env,
    baseline_digest: str,
    candidate_digest: str,
    correctness: CorrectnessReport | None,
    started_at: str,
) -> BenchReport:
    """After Module A passes under mode=all, B/C are still stubbed."""
    return BenchReport(
        schema_version=1,
        task_id=req.task_id,
        verdict="error",
        started_at=started_at,
        finished_at=_utc_now_iso(),
        environment=env,
        inputs_fingerprint=build_inputs_fingerprint(
            request_raw=request_raw,
            req=req,
            baseline_digest=baseline_digest,
            candidate_digest=candidate_digest,
        ),
        correctness=correctness,
        stub_note=(
            "Modules B/C (perf_screen / sla_bench) not implemented yet "
            f"(bench {__version__}). Correctness gate completed."
        ),
    )


def build_correctness_only_report(
    *,
    request_raw: bytes,
    req: BenchRequest,
    env,
    baseline_digest: str,
    candidate_digest: str,
    correctness: CorrectnessReport,
    started_at: str,
) -> BenchReport:
    overall: str
    if correctness.verdict == "pass":
        overall = "pass"
    else:
        overall = "fail_correctness"
    return BenchReport(
        schema_version=1,
        task_id=req.task_id,
        verdict=overall,  # type: ignore[arg-type]
        started_at=started_at,
        finished_at=_utc_now_iso(),
        environment=env,
        inputs_fingerprint=build_inputs_fingerprint(
            request_raw=request_raw,
            req=req,
            baseline_digest=baseline_digest,
            candidate_digest=candidate_digest,
        ),
        correctness=correctness,
    )


def build_legacy_stub_report(
    *,
    request_raw: bytes,
    req: BenchRequest,
    env,
) -> BenchReport:
    now = _utc_now_iso()
    return BenchReport(
        schema_version=1,
        task_id=req.task_id,
        verdict="error",
        started_at=now,
        finished_at=now,
        environment=env,
        inputs_fingerprint=build_inputs_fingerprint(
            request_raw=request_raw,
            req=req,
            baseline_digest=extract_image_digest(req.engines.baseline.image),
            candidate_digest=extract_image_digest(req.engines.candidate.image),
        ),
        stub_note=(
            f"skeleton/stub run for mode={req.mode!r}: module not implemented yet "
            f"(bench {__version__}). Request validated; environment fingerprinted."
        ),
    )


def run_bench(
    request_path: Path,
    output_dir: Path,
    *,
    mock_engine: bool = False,
    mock_tampered_candidate: bool = False,
) -> int:
    try:
        req, raw = load_bench_request(request_path)
    except RequestValidationError as exc:
        print(f"error: invalid request: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST

    if mock_tampered_candidate and not mock_engine:
        print(
            "error: --mock-tampered-candidate requires --mock-engine",
            file=sys.stderr,
        )
        return EXIT_BAD_REQUEST

    layout = OutputLayout(output_dir)
    layout.prepare()
    added = _attach_logging(layout)
    started_at = _utc_now_iso()

    try:
        env = collect_environment()
        warn = warn_gpu_sku_mismatch(env, req.hardware.gpu_sku_expected)
        if warn:
            logger.warning(warn)

        layout.write_env_dumps(collect_env_raw_dumps())
        layout.append_log(
            {
                "event": "start",
                "task_id": req.task_id,
                "mode": req.mode,
                "harness_version": __version__,
                "mock_engine": mock_engine,
            }
        )

        run_correctness_modes = req.mode in ("all", "correctness")
        if not run_correctness_modes:
            report = build_legacy_stub_report(request_raw=raw, req=req, env=env)
            return _write_validated_report(layout, report)

        prompts = load_correctness_prompts(
            trace_ref_path=req.workload_trace.path,
            expected_sha256=req.workload_trace.sha256,
            num_prompts=req.correctness.num_prompts,
            request_path=request_path,
        )

        try:
            if mock_engine:
                corr, base_d, cand_d = run_with_mock_engines(
                    req=req,
                    layout=layout,
                    prompts=prompts,
                    tampered_candidate=mock_tampered_candidate,
                )
            else:
                corr, base_d, cand_d = run_with_docker_engines(
                    req=req,
                    layout=layout,
                    prompts=prompts,
                )
        except EngineError as exc:
            logger.error("engine failure: %s", exc)
            print(f"error: engine failure: {exc}", file=sys.stderr)
            return EXIT_ENGINE

        if corr.verdict != "pass":
            report = build_correctness_only_report(
                request_raw=raw,
                req=req,
                env=env,
                baseline_digest=base_d,
                candidate_digest=cand_d,
                correctness=corr,
                started_at=started_at,
            )
            return _write_validated_report(layout, report)

        if req.mode == "correctness":
            report = build_correctness_only_report(
                request_raw=raw,
                req=req,
                env=env,
                baseline_digest=base_d,
                candidate_digest=cand_d,
                correctness=corr,
                started_at=started_at,
            )
            return _write_validated_report(layout, report)

        # mode=all and correctness passed — B/C still stubbed.
        report = build_stub_remainder_report(
            request_raw=raw,
            req=req,
            env=env,
            baseline_digest=base_d,
            candidate_digest=cand_d,
            correctness=corr,
            started_at=started_at,
        )
        return _write_validated_report(layout, report)
    except RequestValidationError as exc:
        print(f"error: invalid request: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST
    finally:
        _detach_logging(added)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description="Pareton bench harness (Stages 1–3).",
    )
    p.add_argument(
        "--request", required=True, type=Path, help="Path to bench_request.json"
    )
    p.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    p.add_argument(
        "--mock-engine",
        action="store_true",
        help="Use in-process mock engines (no Docker/GPU).",
    )
    p.add_argument(
        "--mock-tampered-candidate",
        action="store_true",
        help="With --mock-engine, offset candidate logprobs (adversarial fail).",
    )
    p.add_argument("--version", action="version", version=f"bench {__version__}")
    args = p.parse_args(argv)

    return run_bench(
        args.request,
        args.output_dir,
        mock_engine=args.mock_engine,
        mock_tampered_candidate=args.mock_tampered_candidate,
    )


# Re-export for tests that imported the old stub helpers.
def run_stub(request_path: Path, output_dir: Path) -> int:
    return run_bench(request_path, output_dir, mock_engine=False)


if __name__ == "__main__":
    raise SystemExit(main())
