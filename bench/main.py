"""CLI entrypoint: python -m bench --request ... --output-dir ...

Exit codes:
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
    resolve_trace_path,
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
    HostEnvironmentError,
    new_run_id,
)
from bench.mock_engine import MockEngine, MockEngineConfig
from bench.output import JsonlFileHandler, OutputLayout
from bench.perf_screen import (
    finish_perf_screen,
    run_perf_screen,
    run_perf_screen_engine,
)
from bench.schemas import (
    BenchReport,
    BenchRequest,
    CorrectnessReport,
    InputsFingerprint,
    PerfScreenReport,
    SlaBenchReport,
    WorkloadTrace,
)
from bench.sla_bench import finish_sla_bench, run_sla_bench, run_sla_engine
from bench.validate import (
    RequestValidationError,
    extract_image_digest,
    load_bench_request,
    load_workload_trace,
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


class _EngineProvider:
    """Supplies healthy engine base URLs for each module run.

    - mock: both in-process mocks start on first use and stay up for the run
      (cheap, no GPU); digests come from the request image pins.
    - docker: each module phase starts the needed container(s) on an internal
      network, runs, and tears them down; baseline and candidate never coexist.
    """

    def __init__(
        self,
        *,
        req: BenchRequest,
        mock: bool,
        tampered_candidate: bool = False,
        baseline_token_latency_s: float = 0.0,
        candidate_token_latency_s: float = 0.0,
        logs_dir: Path | None = None,
    ) -> None:
        self._req = req
        self._mock = mock
        self._tampered = tampered_candidate
        self._base_latency = baseline_token_latency_s
        self._cand_latency = candidate_token_latency_s
        self._logs_dir = logs_dir or Path(".")
        self.baseline_digest = extract_image_digest(req.engines.baseline.image)
        self.candidate_digest = extract_image_digest(req.engines.candidate.image)
        self._mocks: tuple[MockEngine, MockEngine] | None = None

    def _ensure_mocks(self) -> tuple[MockEngine, MockEngine]:
        if self._mocks is None:
            base = MockEngine(
                MockEngineConfig(
                    model="mock-baseline",
                    host="127.0.0.1",
                    port=0,
                    token_latency_s=self._base_latency,
                )
            )
            cand = MockEngine(
                MockEngineConfig(
                    model="mock-candidate",
                    host="127.0.0.1",
                    port=0,
                    tampered=self._tampered,
                    token_latency_s=self._cand_latency,
                )
            )
            base.__enter__()
            cand.__enter__()
            self._mocks = (base, cand)
        return self._mocks

    def shutdown(self) -> None:
        if self._mocks is not None:
            base, cand = self._mocks
            cand.__exit__(None, None, None)
            base.__exit__(None, None, None)
            self._mocks = None

    def _docker_phase(self, net: BenchNetwork, role: str) -> EngineContainer:
        spec = (
            self._req.engines.baseline
            if role == "baseline"
            else self._req.engines.candidate
        )
        return EngineContainer(
            spec=spec,
            network=net,
            role=role,
            gpu_count=_effective_gpu_count(self._req.hardware.gpu_count),
            publish_port=False,
            pull=_should_pull_image(spec.image),
            logs_dir=self._logs_dir,
        )

    def run_correctness(
        self, prompts: list[PromptCase], cfg, task_id: str, evidence_dir: Path
    ) -> CorrectnessReport:
        if self._mock:
            base, cand = self._ensure_mocks()
            return run_correctness(
                base.base_url,
                cand.base_url,
                prompts=prompts,
                cfg=cfg,
                task_id=task_id,
                evidence_dir=evidence_dir,
            )
        with BenchNetwork(run_id=new_run_id(), internal=True) as net:
            with self._docker_phase(net, "baseline") as base:
                phase = collect_baseline_correctness(
                    base.base_url, prompts=prompts, cfg=cfg, task_id=task_id
                )
                self.baseline_digest = base.image_digest
            with self._docker_phase(net, "candidate") as cand:
                report = finish_correctness_with_candidate(
                    cand.base_url, phase, cfg=cfg, evidence_dir=evidence_dir
                )
                self.candidate_digest = cand.image_digest
                return report

    def run_perf_screen(self, requests, cfg, evidence_dir: Path) -> PerfScreenReport:
        if self._mock:
            base, cand = self._ensure_mocks()
            return run_perf_screen(
                base.base_url,
                cand.base_url,
                requests=requests,
                cfg=cfg,
                evidence_dir=evidence_dir,
            )
        with BenchNetwork(run_id=new_run_id(), internal=True) as net:
            with self._docker_phase(net, "baseline") as base:
                base_metrics = run_perf_screen_engine(
                    base.base_url,
                    role="baseline",
                    requests=requests,
                    cfg=cfg,
                    evidence_dir=evidence_dir,
                )
            with self._docker_phase(net, "candidate") as cand:
                return finish_perf_screen(
                    cand.base_url,
                    baseline=base_metrics,
                    requests=requests,
                    cfg=cfg,
                    evidence_dir=evidence_dir,
                )

    def run_sla_bench(self, trace, cfg, evidence_dir: Path) -> SlaBenchReport:
        if self._mock:
            base, cand = self._ensure_mocks()
            return run_sla_bench(
                base.base_url,
                cand.base_url,
                trace=trace,
                cfg=cfg,
                evidence_dir=evidence_dir,
            )
        requests = list(trace.requests)
        with BenchNetwork(run_id=new_run_id(), internal=True) as net:
            with self._docker_phase(net, "baseline") as base:
                base_reps = run_sla_engine(
                    base.base_url,
                    role="baseline",
                    requests=requests,
                    cfg=cfg,
                    evidence_dir=evidence_dir,
                )
            with self._docker_phase(net, "candidate") as cand:
                return finish_sla_bench(
                    cand.base_url,
                    baseline_reps=base_reps,
                    requests=requests,
                    cfg=cfg,
                    evidence_dir=evidence_dir,
                )


def run_all_modules(
    *,
    req: BenchRequest,
    provider: _EngineProvider,
    prompts: list[PromptCase],
    trace: WorkloadTrace,
    layout: OutputLayout,
) -> tuple[
    CorrectnessReport | None, PerfScreenReport | None, SlaBenchReport | None, str | None
]:
    corr: CorrectnessReport | None = None
    perf: PerfScreenReport | None = None
    sla: SlaBenchReport | None = None
    skipped_note: str | None = None

    if req.mode in ("all", "correctness"):
        corr = provider.run_correctness(
            prompts, req.correctness, req.task_id, layout.correctness_dir
        )
        if corr.verdict != "pass":
            skipped_note = (
                "correctness gate failed; perf_screen/sla_bench skipped"
                if req.mode == "all"
                else None
            )
            return corr, None, None, skipped_note

    if req.mode in ("all", "perf_screen"):
        perf = provider.run_perf_screen(
            trace.requests, req.perf_screen, layout.perf_screen_dir
        )
        if perf.verdict != "pass":
            skipped_note = (
                "perf_screen failed; sla_bench skipped" if req.mode == "all" else None
            )
            return corr, perf, None, skipped_note

    if req.mode in ("all", "sla_bench"):
        sla = provider.run_sla_bench(trace, req.sla_bench, layout.sla_bench_dir)

    return corr, perf, sla, skipped_note


def _verdict(corr, perf, sla) -> str:
    if corr is not None and corr.verdict != "pass":
        return "fail_correctness"
    if perf is not None and perf.verdict != "pass":
        return "fail_perf_screen"
    if sla is not None and sla.verdict != "pass":
        return "fail_sla" if sla.verdict == "fail_sla" else "error"
    return "pass"


def build_bench_report(
    *,
    request_raw: bytes,
    req: BenchRequest,
    env,
    baseline_digest: str,
    candidate_digest: str,
    corr: CorrectnessReport | None,
    perf: PerfScreenReport | None,
    sla: SlaBenchReport | None,
    skipped_note: str | None,
    started_at: str,
) -> BenchReport:
    return BenchReport(
        schema_version=1,
        task_id=req.task_id,
        verdict=_verdict(corr, perf, sla),  # type: ignore[arg-type]
        started_at=started_at,
        finished_at=_utc_now_iso(),
        environment=env,
        inputs_fingerprint=build_inputs_fingerprint(
            request_raw=request_raw,
            req=req,
            baseline_digest=baseline_digest,
            candidate_digest=candidate_digest,
        ),
        correctness=corr,
        perf_screen=perf,
        sla_bench=sla,
        skipped_note=skipped_note,
    )


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


def run_bench(
    request_path: Path,
    output_dir: Path,
    *,
    mock_engine: bool = False,
    mock_tampered_candidate: bool = False,
    mock_baseline_token_latency_s: float = 0.0,
    mock_candidate_token_latency_s: float = 0.0,
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

        # Load + validate the trace (and Module A prompts) before engines start.
        trace_path = resolve_trace_path(
            req.workload_trace.path, request_path=request_path
        )
        trace = load_workload_trace(
            trace_path, expected_sha256=req.workload_trace.sha256
        )
        prompts: list[PromptCase] = []
        if req.mode in ("all", "correctness"):
            prompts = load_correctness_prompts(
                trace_ref_path=req.workload_trace.path,
                expected_sha256=req.workload_trace.sha256,
                num_prompts=req.correctness.num_prompts,
                request_path=request_path,
            )

        provider = _EngineProvider(
            req=req,
            mock=mock_engine,
            tampered_candidate=mock_tampered_candidate,
            baseline_token_latency_s=mock_baseline_token_latency_s,
            candidate_token_latency_s=mock_candidate_token_latency_s,
            logs_dir=layout.correctness_dir / "engine_logs",
        )

        try:
            corr, perf, sla, skipped_note = run_all_modules(
                req=req,
                provider=provider,
                prompts=prompts,
                trace=trace,
                layout=layout,
            )
        except HostEnvironmentError as exc:
            logger.error("environment error: %s", exc)
            print(f"error: environment: {exc}", file=sys.stderr)
            return EXIT_ENV
        except EngineError as exc:
            logger.error("engine failure: %s", exc)
            print(f"error: engine failure: {exc}", file=sys.stderr)
            return EXIT_ENGINE
        finally:
            provider.shutdown()

        report = build_bench_report(
            request_raw=raw,
            req=req,
            env=env,
            baseline_digest=provider.baseline_digest,
            candidate_digest=provider.candidate_digest,
            corr=corr,
            perf=perf,
            sla=sla,
            skipped_note=skipped_note,
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
    p.add_argument(
        "--mock-baseline-token-latency-s",
        type=float,
        default=0.0,
        help="With --mock-engine, per-token latency (s) for the baseline engine.",
    )
    p.add_argument(
        "--mock-candidate-token-latency-s",
        type=float,
        default=0.0,
        help="With --mock-engine, per-token latency (s) for the candidate engine.",
    )
    p.add_argument("--version", action="version", version=f"bench {__version__}")
    args = p.parse_args(argv)

    return run_bench(
        args.request,
        args.output_dir,
        mock_engine=args.mock_engine,
        mock_tampered_candidate=args.mock_tampered_candidate,
        mock_baseline_token_latency_s=args.mock_baseline_token_latency_s,
        mock_candidate_token_latency_s=args.mock_candidate_token_latency_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
