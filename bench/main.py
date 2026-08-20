"""CLI entrypoint: python -m bench --request ... --output-dir ...

One ``bench_request.json`` describes one whole round: a baseline and every
candidate. The harness starts one engine container at a time, in this order:

1. The baseline, with the engine compile cache mounted read-write. Its SLA
   replay is the fixed reference every candidate is scored against.
2. Each candidate, with no cache mount and in production configuration, so
   they all meet the same cold cache and are timed on the same footing.
3. One scorer, after the last candidate has stopped, teacher-forcing every
   output the candidates produced. Then stopped.
4. The baseline again, SLA only, to measure how far the pod drifted while
   the round ran.

That is ``3 + len(candidates)`` engine starts. Only the scorer runs with
correctness-specific serve args, so the count does not depend on which engine
the campaign pins: a vLLM round and an SGLang round are the same size.

Exit codes:
  0 = harness completed (per-entry outcomes are in the report)
  1 = bad request / schema validation failure
  2 = environment error
  3 = engine failure the round cannot continue past
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from bench import __version__
from bench.correctness import (
    PendingCorrectness,
    PromptCase,
    capture_outputs,
    grade_all,
    load_correctness_prompts,
    resolve_trace_path,
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
from bench.phases import BenchPhase
from bench.schemas import (
    BenchReport,
    BenchRequest,
    CorrectnessReport,
    EngineSpec,
    EnginesSpec,
    InputsFingerprint,
    RoundEntryReport,
    WorkloadTrace,
)
from bench.score import score_candidate
from bench.sla_bench import REPRO_BAR_MAX_REL_RANGE, EngineReplay, run_sla_engine
from bench.validate import (
    RequestValidationError,
    extract_image_digest,
    load_bench_request,
    load_workload_trace,
    sha256_bytes,
    validate_report_dict,
)
from bench.weights import stage_weights

MOCK_WEIGHTS_SHA256 = "sha256:" + ("0" * 64)

# Serve args for the scorer, which needs reproducible logprobs rather than
# speed. Exactly one container in a round is started with them. The autotune
# flag is conditional on the pinned vLLM accepting
# --no-enable-flashinfer-autotune (GPU smoke later may drop it).
CORRECTNESS_EXTRA_SERVE_ARGS = [
    "--no-enable-prefix-caching",
    "--no-enable-flashinfer-autotune",
]


def correctness_extra_serve_args(serve_args: list[str]) -> list[str]:
    """vLLM-only scorer flags. SGLang uses --tp-size and rejects these."""
    if "--tp-size" in serve_args:
        return []
    return list(CORRECTNESS_EXTRA_SERVE_ARGS)


EXIT_OK = 0
EXIT_BAD_REQUEST = 1
EXIT_ENV = 2
EXIT_ENGINE = 3

logger = logging.getLogger("bench")


def scorer_engine_spec(spec: EngineSpec) -> EngineSpec:
    """The scorer: the campaign's own baseline image plus the scorer flags.

    The scorer is per campaign rather than per candidate, and it is derived
    from the pinned baseline rather than named separately, so a campaign
    manifest carries no scorer field of its own.
    """
    extra = correctness_extra_serve_args(spec.serve_args)
    return EngineSpec(
        image=spec.image,
        serve_args=list(spec.serve_args) + extra,
        env=dict(spec.env),
    )


# ---------------------------------------------------------------------------
# Round plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineStart:
    """One engine container start in a round.

    ``role`` names the container and its evidence directory, so every start in
    a round is distinguishable after the fact.
    """

    role: str
    kind: str  # "baseline" | "candidate" | "scorer" | "drift"
    spec: EngineSpec
    mount_engine_cache: bool
    candidate_index: int | None = None


def plan_round_starts(engines: EnginesSpec, *, mode: str = "all") -> list[EngineStart]:
    """Every container this round will start, in order.

    The runner consumes this list, so the plan is the only place a start can
    be added. Two invariants live here:

    * only a baseline start mounts the engine compile cache, so every
      candidate begins in the same cache state and whatever it compiles dies
      with the container;
    * the closing drift baseline mounts it too, because the drift number is a
      comparison against the first baseline run and the two must differ only
      in when they ran.
    """
    starts = [
        EngineStart(
            role="baseline",
            kind="baseline",
            spec=engines.baseline,
            mount_engine_cache=True,
        )
    ]
    for i, cand in enumerate(engines.candidates):
        starts.append(
            EngineStart(
                role=f"candidate-{i}",
                kind="candidate",
                spec=cand,
                mount_engine_cache=False,
                candidate_index=i,
            )
        )
    if mode == "all":
        starts.append(
            EngineStart(
                role="scorer",
                kind="scorer",
                spec=scorer_engine_spec(engines.baseline),
                mount_engine_cache=False,
            )
        )
    starts.append(
        EngineStart(
            role="baseline-drift",
            kind="drift",
            spec=engines.baseline,
            mount_engine_cache=True,
        )
    )
    return starts


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_inputs_fingerprint(
    *,
    request_raw: bytes,
    req: BenchRequest,
    baseline_digest: str,
    candidate_digests: list[str],
    model_weights_sha256: str,
) -> InputsFingerprint:
    return InputsFingerprint(
        baseline_image_digest=baseline_digest,
        candidate_image_digest=list(candidate_digests),
        model_repo=req.model.hf_repo,
        model_revision=req.model.hf_revision,
        model_weights_sha256=model_weights_sha256,
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


def _write_engine_error_report(
    *,
    layout: OutputLayout,
    req: BenchRequest,
    request_raw: bytes,
    env,
    started_at: str,
    exc: EngineError,
    model_weights_sha256: str,
) -> None:
    """Minimal schema-valid verdict=error report with optional error_role."""
    role = getattr(exc, "error_role", None)
    report: dict = {
        "schema_version": 1,
        "task_id": req.task_id,
        "verdict": "error",
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "environment": env.to_dict(),
        "inputs_fingerprint": {
            "baseline_image_digest": extract_image_digest(req.engines.baseline.image),
            "candidate_image_digest": [
                extract_image_digest(c.image) for c in req.engines.candidates
            ],
            "model_repo": req.model.hf_repo,
            "model_revision": req.model.hf_revision,
            "model_weights_sha256": model_weights_sha256,
            "trace_sha256": req.workload_trace.sha256,
            "request_sha256": sha256_bytes(request_raw),
        },
        "error": str(exc),
    }
    if role:
        report["error_role"] = role
    validate_report_dict(report)
    layout.write_report(report)
    layout.append_log(
        {
            "event": "engine_error",
            "error_role": role,
            "report": str(layout.report_path),
        }
    )
    print(f"wrote {layout.report_path}")


# ---------------------------------------------------------------------------
# Mock round scripting
# ---------------------------------------------------------------------------


@dataclass
class MockCandidatePlan:
    """One mock candidate's behaviour.

    Together these cover every outcome a real candidate can reach: faster than
    the baseline, barely different from it, wrong, or unable to start.
    """

    speed_factor: float = 1.0
    garbage: bool = False
    infra_fail: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> MockCandidatePlan:
        return cls(
            speed_factor=float(d.get("speed_factor", 1.0)),
            garbage=bool(d.get("garbage", False)),
            infra_fail=bool(d.get("infra_fail", False)),
        )


@dataclass
class MockPlan:
    baseline_token_latency_s: float = 0.0
    candidates: list[MockCandidatePlan] = field(default_factory=list)

    def candidate(self, index: int) -> MockCandidatePlan:
        if index < len(self.candidates):
            return self.candidates[index]
        return MockCandidatePlan()


class _EngineProvider:
    """Yields one healthy engine base URL per planned start, then tears it down.

    Mock and Docker modes both honour the plan one start at a time, so the
    start count a test observes is the start count a round performs.
    """

    def __init__(
        self,
        *,
        req: BenchRequest,
        mock: bool,
        mock_plan: MockPlan | None = None,
        logs_dir: Path | None = None,
        weights_dir: Path | None = None,
        phase_sink: Callable[[str], None] | None = None,
        docker_runner=None,
    ) -> None:
        self._req = req
        self._mock = mock
        self._mock_plan = mock_plan or MockPlan()
        self._logs_dir = logs_dir or Path(".")
        # Test seam: a fake runner exercises the real docker argv without a daemon.
        self._docker_runner = docker_runner
        self.weights_dir = weights_dir
        self._phase_sink = phase_sink
        self.baseline_digest = extract_image_digest(req.engines.baseline.image)
        self.candidate_digests = [
            extract_image_digest(c.image) for c in req.engines.candidates
        ]
        self.starts: list[str] = []

    def _write_phase(self, phase: BenchPhase) -> None:
        if self._phase_sink is not None:
            self._phase_sink(phase.value)

    def _mock_config(self, start: EngineStart) -> MockEngineConfig:
        latency = self._mock_plan.baseline_token_latency_s
        if start.candidate_index is None:
            return MockEngineConfig(
                model=f"mock-{start.role}",
                host="127.0.0.1",
                port=0,
                token_latency_s=latency,
            )
        plan = self._mock_plan.candidate(start.candidate_index)
        return MockEngineConfig(
            model=f"mock-{start.role}",
            host="127.0.0.1",
            port=0,
            token_latency_s=latency,
            speed_factor=plan.speed_factor,
            garbage=plan.garbage,
        )

    @contextmanager
    def start(self, start: EngineStart, *, phase: BenchPhase) -> Iterator[str]:
        """Bring one planned engine up, yield its base URL, always tear it down."""
        self.starts.append(start.role)
        self._write_phase(BenchPhase.STARTING_ENGINE)
        if self._mock:
            plan = (
                self._mock_plan.candidate(start.candidate_index)
                if start.candidate_index is not None
                else None
            )
            if plan is not None and plan.infra_fail:
                raise EngineError(
                    f"mock engine {start.role} failed to start",
                    error_role=start.role,
                )
            engine = MockEngine(self._mock_config(start))
            engine.__enter__()
            try:
                self._write_phase(phase)
                yield engine.base_url
            finally:
                engine.__exit__(None, None, None)
            return

        net_kwargs = (
            {} if self._docker_runner is None else {"runner": self._docker_runner}
        )
        with BenchNetwork(run_id=new_run_id(), internal=True, **net_kwargs) as net:
            container = EngineContainer(
                spec=start.spec,
                network=net,
                role=start.role,
                gpu_count=_effective_gpu_count(self._req.hardware.gpu_count),
                weights_dir=self.weights_dir,
                publish_port=False,
                pull=_should_pull_image(start.spec.image),
                logs_dir=self._logs_dir,
                mount_engine_cache=start.mount_engine_cache,
                on_ready=lambda: self._write_phase(phase),
            )
            with container as handle:
                if start.kind == "baseline":
                    self.baseline_digest = handle.image_digest
                elif start.candidate_index is not None:
                    self.candidate_digests[start.candidate_index] = handle.image_digest
                yield handle.base_url


# ---------------------------------------------------------------------------
# Round execution
# ---------------------------------------------------------------------------


@dataclass
class _CandidateRun:
    index: int
    status: str
    replay: EngineReplay | None = None
    reason: str | None = None


def run_round(
    *,
    req: BenchRequest,
    provider: _EngineProvider,
    prompts: list[PromptCase],
    trace: WorkloadTrace,
    layout: OutputLayout,
) -> tuple[
    EngineReplay,
    EngineReplay,
    list[_CandidateRun],
    dict[int, CorrectnessReport],
]:
    """Execute the whole round against one pod. Returns the raw material."""
    requests = list(trace.requests)
    plan = plan_round_starts(req.engines, mode=req.mode)
    layout.append_log(
        {"event": "round_plan", "starts": [s.role for s in plan], "count": len(plan)}
    )

    baseline: EngineReplay | None = None
    drift: EngineReplay | None = None
    runs: list[_CandidateRun] = []
    pending: list[PendingCorrectness] = []
    correctness: dict[int, CorrectnessReport] = {}

    for start in plan:
        if start.kind in ("baseline", "drift"):
            phase = BenchPhase.SLA_BENCH
            try:
                with provider.start(start, phase=phase) as url:
                    replay = run_sla_engine(
                        url,
                        role=start.role,
                        requests=requests,
                        cfg=req.sla_bench,
                        evidence_dir=layout.sla_bench_dir,
                    )
            except EngineError as exc:
                # The baseline is the fixed reference every candidate is
                # scored against, so the round cannot continue without it.
                raise EngineError(str(exc), error_role="baseline") from exc
            if start.kind == "baseline":
                baseline = replay
            else:
                drift = replay

        elif start.kind == "candidate":
            index = start.candidate_index
            assert index is not None
            try:
                with provider.start(start, phase=BenchPhase.SLA_BENCH) as url:
                    replay = run_sla_engine(
                        url,
                        role=start.role,
                        requests=requests,
                        cfg=req.sla_bench,
                        evidence_dir=layout.sla_bench_dir,
                    )
            except EngineError as exc:
                # One candidate failing to run is that entry's problem, not
                # the round's. The round continues with the rest.
                logger.warning("candidate %d infra failure: %s", index, exc)
                runs.append(
                    _CandidateRun(index=index, status="infra_failed", reason=str(exc))
                )
                continue
            runs.append(_CandidateRun(index=index, status="scored", replay=replay))
            pending.append(
                PendingCorrectness(
                    candidate_index=index,
                    outputs=capture_outputs(
                        prompts, timings=replay.result.timings, outputs=replay.outputs
                    ),
                )
            )

        elif start.kind == "scorer":
            if not pending:
                continue
            try:
                with provider.start(start, phase=BenchPhase.CORRECTNESS) as url:
                    correctness = grade_all(
                        url,
                        pending,
                        cfg=req.correctness,
                        evidence_dir=layout.correctness_dir,
                    )
            except EngineError as exc:
                # Correctness is a hard gate, so an unusable scorer means no
                # entry in this round can be judged.
                raise EngineError(str(exc), error_role="scorer") from exc

    if baseline is None or drift is None:
        raise EngineError("round plan did not produce both baseline runs")
    return baseline, drift, runs, correctness


def _build_entries(
    *,
    req: BenchRequest,
    baseline: EngineReplay,
    runs: list[_CandidateRun],
    correctness: dict[int, CorrectnessReport],
    digests: list[str],
) -> list[RoundEntryReport]:
    entries: list[RoundEntryReport] = []
    for run in runs:
        digest = digests[run.index] if run.index < len(digests) else ""
        if run.status == "infra_failed" or run.replay is None:
            entries.append(
                RoundEntryReport(
                    index=run.index,
                    image_digest=digest,
                    status="infra_failed",
                    reason=run.reason,
                )
            )
            continue

        corr = correctness.get(run.index)
        if corr is not None and corr.verdict == "infra_failed":
            entries.append(
                RoundEntryReport(
                    index=run.index,
                    image_digest=digest,
                    status="infra_failed",
                    sla=run.replay.result,
                    correctness=corr,
                    reason=corr.reason,
                )
            )
            continue
        if corr is not None and corr.verdict != "pass":
            # Correctness is a hard gate: a wrong image gets no score at
            # all, so it cannot take the crown on speed.
            entries.append(
                RoundEntryReport(
                    index=run.index,
                    image_digest=digest,
                    status="disqualified",
                    sla=run.replay.result,
                    correctness=corr,
                    reason=corr.reason,
                )
            )
            continue

        variance = run.replay.result.cross_rep_variance or {}
        rel_range = float(variance.get("p99_e2e_ms_rel_range") or 0.0)
        if rel_range > REPRO_BAR_MAX_REL_RANGE:
            entries.append(
                RoundEntryReport(
                    index=run.index,
                    image_digest=digest,
                    status="infra_failed",
                    sla=run.replay.result,
                    correctness=corr,
                    reason=(
                        f"p99_e2e_ms_rel_range {rel_range:.4f} exceeds "
                        f"reproducibility bar {REPRO_BAR_MAX_REL_RANGE}"
                    ),
                )
            )
            continue

        scored = score_candidate(
            req.scoring_rule,
            baseline=baseline.result.timings,
            candidate=run.replay.result.timings,
        )
        entries.append(
            RoundEntryReport(
                index=run.index,
                image_digest=digest,
                status="scored",
                score=scored.score,
                score_report=scored.to_report(),
                sla=run.replay.result,
                correctness=corr,
            )
        )
    return entries


def baseline_drift(
    req: BenchRequest, baseline: EngineReplay, drift: EngineReplay
) -> float:
    """How far the pod moved between the round's two baseline runs.

    Drift is ``last_baseline_score - first_baseline_score``. The opening
    baseline scores 0.0 against itself under any speedup rule, so the
    difference is exactly the closing run's score. Positive means the pod got
    faster while the round ran; negative, slower. A round whose drift is too
    large was not measuring the candidates.
    """
    return score_candidate(
        req.scoring_rule,
        baseline=baseline.result.timings,
        candidate=drift.result.timings,
    ).score


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
    mock_plan: MockPlan | None = None,
) -> int:
    try:
        req, raw = load_bench_request(request_path)
    except RequestValidationError as exc:
        print(f"error: invalid request: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST

    if mock_plan is not None and not mock_engine:
        print("error: --mock-candidates requires --mock-engine", file=sys.stderr)
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
                "candidates": len(req.engines.candidates),
                "harness_version": __version__,
                "mock_engine": mock_engine,
            }
        )

        # Load + validate the trace and the correctness prompts before any
        # engine starts: a bad trace must not cost a pod-hour.
        trace_path = resolve_trace_path(
            req.workload_trace.path, request_path=request_path
        )
        trace = load_workload_trace(
            trace_path, expected_sha256=req.workload_trace.sha256
        )
        prompts: list[PromptCase] = []
        if req.mode == "all":
            prompts = load_correctness_prompts(
                trace_ref_path=req.workload_trace.path,
                expected_sha256=req.workload_trace.sha256,
                num_prompts=req.correctness.num_prompts,
                request_path=request_path,
            )

        # Mock mode has no weights; docker mode stages before engines start.
        model_weights_sha256 = MOCK_WEIGHTS_SHA256
        provider = _EngineProvider(
            req=req,
            mock=mock_engine,
            mock_plan=mock_plan,
            logs_dir=layout.correctness_dir / "engine_logs",
            weights_dir=None,
            phase_sink=layout.write_phase,
        )

        try:
            if not mock_engine:
                # First real wait of a cold run: hundreds of GB of weights.
                layout.write_phase(BenchPhase.DOWNLOADING_MODEL.value)
                staged = stage_weights(req.model, token_env=req.hf_token_env)
                provider.weights_dir = staged.path
                model_weights_sha256 = staged.weights_sha256
                layout.write_weights_manifest(
                    staged.manifest, aggregate=staged.weights_sha256
                )
                layout.append_log(
                    {
                        "event": "weights_staged",
                        "repo": req.model.hf_repo,
                        "revision": req.model.hf_revision,
                        "path": str(staged.path),
                        "weights_sha256": staged.weights_sha256,
                        "num_files": staged.num_files,
                        "total_bytes": staged.total_bytes,
                    }
                )
            baseline, drift, runs, correctness = run_round(
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
            try:
                _write_engine_error_report(
                    layout=layout,
                    req=req,
                    request_raw=raw,
                    env=env,
                    started_at=started_at,
                    exc=exc,
                    model_weights_sha256=model_weights_sha256,
                )
            except Exception as write_exc:
                logger.error("failed to write engine error report: %s", write_exc)
            return EXIT_ENGINE

        entries = _build_entries(
            req=req,
            baseline=baseline,
            runs=runs,
            correctness=correctness,
            digests=provider.candidate_digests,
        )
        report = BenchReport(
            schema_version=1,
            task_id=req.task_id,
            verdict="pass",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            environment=env,
            inputs_fingerprint=build_inputs_fingerprint(
                request_raw=raw,
                req=req,
                baseline_digest=provider.baseline_digest,
                candidate_digests=provider.candidate_digests,
                model_weights_sha256=model_weights_sha256,
            ),
            scoring_rule=dict(req.scoring_rule),
            baseline=baseline.result,
            drift_baseline=drift.result,
            baseline_drift=baseline_drift(req, baseline, drift),
            entries=entries,
        )
        return _write_validated_report(layout, report)
    except RequestValidationError as exc:
        print(f"error: invalid request: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST
    finally:
        _detach_logging(added)


def _parse_mock_candidates(raw: str) -> MockPlan:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequestValidationError(f"--mock-candidates is not valid JSON: {exc}")
    if not isinstance(parsed, list):
        raise RequestValidationError("--mock-candidates must be a JSON list")
    return MockPlan(candidates=[MockCandidatePlan.from_dict(dict(c)) for c in parsed])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description="Pareton round bench harness.",
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
        "--mock-baseline-token-latency-s",
        type=float,
        default=0.0,
        help="With --mock-engine, per-token latency (s) every mock engine starts from.",
    )
    p.add_argument(
        "--mock-candidates",
        default=None,
        help=(
            "With --mock-engine, a JSON list scripting each candidate, e.g. "
            '\'[{"speed_factor": 1.5}, {"garbage": true}, {"infra_fail": true}]\''
        ),
    )
    p.add_argument("--version", action="version", version=f"bench {__version__}")
    args = p.parse_args(argv)

    plan: MockPlan | None = None
    if args.mock_candidates is not None:
        try:
            plan = _parse_mock_candidates(args.mock_candidates)
        except RequestValidationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_BAD_REQUEST
    if args.mock_baseline_token_latency_s:
        plan = plan or MockPlan()
        plan.baseline_token_latency_s = args.mock_baseline_token_latency_s

    return run_bench(
        args.request,
        args.output_dir,
        mock_engine=args.mock_engine,
        mock_plan=plan,
    )


if __name__ == "__main__":
    raise SystemExit(main())
