"""Dataclasses for bench_request / bench_report / workload_trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from bench.score import PromptTiming

# ---------------------------------------------------------------------------
# Workload trace
# ---------------------------------------------------------------------------


@dataclass
class TraceSampling:
    temperature: float
    top_p: float
    ignore_eos: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceSampling:
        ignore_eos = d.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise ValueError("sampling.ignore_eos must be a boolean")
        return cls(
            temperature=float(d["temperature"]),
            top_p=float(d["top_p"]),
            ignore_eos=ignore_eos,
        )


@dataclass
class TraceRequest:
    id: str
    arrival_offset_ms: int
    max_tokens: int
    sampling: TraceSampling
    prompt: str | None = None
    prompt_token_ids: list[int] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceRequest:
        prompt = d.get("prompt")
        prompt_token_ids = d.get("prompt_token_ids")
        if prompt is None and prompt_token_ids is None:
            raise ValueError(
                f"trace request {d.get('id')!r}: need prompt or prompt_token_ids"
            )
        return cls(
            id=str(d["id"]),
            arrival_offset_ms=int(d["arrival_offset_ms"]),
            max_tokens=int(d["max_tokens"]),
            sampling=TraceSampling.from_dict(d["sampling"]),
            prompt=None if prompt is None else str(prompt),
            prompt_token_ids=None
            if prompt_token_ids is None
            else list(prompt_token_ids),
        )


@dataclass
class TraceMeta:
    name: str
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceMeta:
        return cls(
            name=str(d.get("name", "")), description=str(d.get("description", ""))
        )


@dataclass
class WorkloadTrace:
    schema_version: int
    meta: TraceMeta
    requests: list[TraceRequest]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkloadTrace:
        return cls(
            schema_version=int(d["schema_version"]),
            meta=TraceMeta.from_dict(d.get("meta") or {}),
            requests=[TraceRequest.from_dict(r) for r in d["requests"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# bench_request.json
# ---------------------------------------------------------------------------

# "all" runs the whole round, scorer included. "sla_bench" stops after the
# timing runs and is a debugging mode. Correctness has no mode of its own,
# because the text it grades is what the SLA replay captured.
BenchMode = Literal["all", "sla_bench"]


@dataclass
class ModelSpec:
    hf_repo: str
    hf_revision: str
    dtype: str
    quantization: str | None
    max_model_len: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(
            hf_repo=str(d["hf_repo"]),
            hf_revision=str(d["hf_revision"]),
            dtype=str(d["dtype"]),
            quantization=d.get("quantization"),
            max_model_len=int(d["max_model_len"]),
        )


@dataclass
class HardwareSpec:
    gpu_count: int
    gpu_sku_expected: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareSpec:
        return cls(
            gpu_count=int(d["gpu_count"]), gpu_sku_expected=str(d["gpu_sku_expected"])
        )


@dataclass
class EngineSpec:
    """One engine container: its image, how to serve, and its compile cache.

    ``cache_dir`` is the container-side path the campaign's engine profile
    pins (``campaign/engine.py``); the harness mounts the host cache there for
    the starts that ask for it. A request that omits it gets the vLLM path,
    the same default ``resolve_engine(None)`` applies campaign-side.
    """

    image: str
    serve_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cache_dir: str = "/root/.cache/vllm"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineSpec:
        return cls(
            image=str(d["image"]),
            serve_args=[str(x) for x in (d.get("serve_args") or [])],
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            cache_dir=str(d.get("cache_dir") or "/root/.cache/vllm"),
        )


@dataclass
class EnginesSpec:
    """One baseline plus every candidate in the round.

    A round is one baseline and N candidates (the leader is just another
    candidate to the harness), so one ``bench_request.json`` describes the
    whole round.
    """

    baseline: EngineSpec
    candidates: list[EngineSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnginesSpec:
        return cls(
            baseline=EngineSpec.from_dict(d["baseline"]),
            candidates=[EngineSpec.from_dict(c) for c in (d.get("candidates") or [])],
        )


@dataclass
class WorkloadTraceRef:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkloadTraceRef:
        return cls(path=str(d["path"]), sha256=str(d["sha256"]))


@dataclass
class CorrectnessThresholds:
    """The bars the shared scorer grades a captured output against.

    The first four are absolute logprob bars: a greedy continuation of the
    pinned model scores around -0.5 to -2.0 per token under that same model,
    and garbage scores below -15. ``min_token_logprob`` is applied to the
    k-th lowest scored position, with k = ceil(``min_token_quantile`` *
    positions), not to the outright minimum (PAR-94). A quantile of 0 is the
    plain minimum.

    ``max_mean_logprob_drop`` measures the candidate against the baseline's
      own mean logprob, same scorer and same prompts, catching the opposite
      move: a candidate that degrades the model and clears the floor anyway.

    Repeat-loop detection is deliberately absent here. It is a mandatory
    harness exploit check rather than miner-visible competition policy.
    ``max_mean_logprob_drop`` remains ``None`` when a legacy campaign does not
    carry it, so enabling the relative quality bar still means re-seeding.
    """

    min_mean_logprob: float
    min_token_logprob: float
    min_token_quantile: float
    min_coverage_ratio: float
    max_mean_logprob_drop: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CorrectnessThresholds:
        def optional(key: str) -> float | None:
            raw = d.get(key)
            return None if raw is None else float(raw)

        return cls(
            min_mean_logprob=float(d["min_mean_logprob"]),
            min_token_logprob=float(d["min_token_logprob"]),
            min_token_quantile=float(d["min_token_quantile"]),
            min_coverage_ratio=float(d["min_coverage_ratio"]),
            max_mean_logprob_drop=optional("max_mean_logprob_drop"),
        )


@dataclass
class CorrectnessConfig:
    """How many captured outputs to grade, and the bars to grade them against.

    ``num_prompts`` counts trace requests in trace order; the scorer grades
    that many of each candidate's captured outputs.
    """

    num_prompts: int
    thresholds: CorrectnessThresholds

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CorrectnessConfig:
        return cls(
            num_prompts=int(d["num_prompts"]),
            thresholds=CorrectnessThresholds.from_dict(d["thresholds"]),
        )


@dataclass
class SlaThresholds:
    p99_ttft_ms: float
    p99_itl_ms: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SlaThresholds:
        return cls(
            p99_ttft_ms=float(d["p99_ttft_ms"]), p99_itl_ms=float(d["p99_itl_ms"])
        )


@dataclass
class SlaBenchConfig:
    repetitions: int
    thresholds: SlaThresholds

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SlaBenchConfig:
        return cls(
            repetitions=int(d["repetitions"]),
            thresholds=SlaThresholds.from_dict(d["thresholds"]),
        )


@dataclass
class BenchRequest:
    """One round: a baseline, every candidate, and the rule that ranks them.

    ``scoring_rule`` is the round's snapshot of ``campaigns.scoring_rule``.
    The harness carries it so the pod can score its own entries and the
    baseline drift with the same formula the round was created under.
    """

    schema_version: int
    task_id: str
    mode: BenchMode
    model: ModelSpec
    hardware: HardwareSpec
    engines: EnginesSpec
    workload_trace: WorkloadTraceRef
    correctness: CorrectnessConfig
    sla_bench: SlaBenchConfig
    scoring_rule: dict[str, Any] = field(default_factory=dict)
    hf_token_env: str = "HF_TOKEN"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchRequest:
        mode = d.get("mode", "all")
        if mode not in ("all", "sla_bench"):
            raise ValueError(f"mode must be one of all|sla_bench, got {mode!r}")
        return cls(
            schema_version=int(d["schema_version"]),
            task_id=str(d["task_id"]),
            mode=mode,  # type: ignore[arg-type]
            model=ModelSpec.from_dict(d["model"]),
            hardware=HardwareSpec.from_dict(d["hardware"]),
            engines=EnginesSpec.from_dict(d["engines"]),
            workload_trace=WorkloadTraceRef.from_dict(d["workload_trace"]),
            correctness=CorrectnessConfig.from_dict(d["correctness"]),
            sla_bench=SlaBenchConfig.from_dict(d["sla_bench"]),
            scoring_rule=dict(d.get("scoring_rule") or {}),
            hf_token_env=str(d.get("hf_token_env") or "HF_TOKEN"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# bench_report.json
# ---------------------------------------------------------------------------

# Whether the round itself ran end to end. Each candidate's own outcome lives
# in ``entries``, so one candidate failing correctness is that entry's
# verdict, not the round's.
Verdict = Literal["pass", "error"]


@dataclass
class GpuInfo:
    index: int
    name: str
    vbios: str
    memory_mb: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentInfo:
    gpu: list[GpuInfo]
    driver_version: str
    cuda_version: str
    docker_version: str
    harness_version: str
    hostname_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": [g.to_dict() for g in self.gpu],
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "docker_version": self.docker_version,
            "harness_version": self.harness_version,
            "hostname_hash": self.hostname_hash,
        }


@dataclass
class InputsFingerprint:
    baseline_image_digest: str
    candidate_image_digest: list[str]
    model_repo: str
    model_revision: str
    model_weights_sha256: str
    trace_sha256: str
    request_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectnessReport:
    """One candidate graded by the shared scorer.

    ``infra_failed`` means the scorer produced too few logprobs to judge on.
    That is a harness problem rather than the candidate being wrong, so the
    entry is requeued instead of disqualified.
    """

    verdict: str  # "pass" | "fail_correctness" | "infra_failed"
    num_prompts: int
    num_positions_scored: int
    mean_logprob: float
    min_logprob: float
    # The k-th lowest scored position, which is what the min-token bar is
    # actually applied to. ``min_logprob`` stays the outright minimum, kept
    # because it is how PAR-94 was diagnosed in the first place.
    quantile_logprob: float
    coverage_ratio: float
    evidence: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LatencyPercentiles:
    p50: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineSlaMetrics:
    ttft_ms: LatencyPercentiles
    itl_ms: LatencyPercentiles
    e2e_ms: LatencyPercentiles
    output_tokens_per_s: float
    requests_per_s: float
    sla_goodput_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms.to_dict(),
            "itl_ms": self.itl_ms.to_dict(),
            "e2e_ms": self.e2e_ms.to_dict(),
            "output_tokens_per_s": self.output_tokens_per_s,
            "requests_per_s": self.requests_per_s,
            "sla_goodput_ratio": self.sla_goodput_ratio,
        }


@dataclass
class EngineSlaResult:
    """One engine's SLA replay: aggregate metrics plus the per-request timings.

    ``timings`` is the shape ``bench.score.score_candidate`` consumes: request
    id -> ``PromptTiming``. The aggregate metrics sit alongside it in absolute
    units, not only as a ratio against the baseline, so a stored report still
    answers questions about a campaign it was never compared within.
    """

    role: str
    metrics: EngineSlaMetrics
    cross_rep_variance: dict[str, float]
    timings: dict[str, PromptTiming]
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "metrics": self.metrics.to_dict(),
            "cross_rep_variance": self.cross_rep_variance,
            "timings": {rid: asdict(t) for rid, t in self.timings.items()},
            "evidence": self.evidence,
        }


@dataclass
class RoundEntryReport:
    """One candidate's outcome. ``index`` is its position in ``engines.candidates``."""

    index: int
    image_digest: str
    status: str  # "scored" | "disqualified" | "infra_failed"
    score: float | None = None
    score_report: dict[str, Any] | None = None
    sla: EngineSlaResult | None = None
    correctness: CorrectnessReport | None = None
    reason: str | None = None
    # True only when the engine's own process died before becoming healthy.
    # The worker keys the incumbent's infra remap on this flag; inferring the
    # crash from a missing correctness block would conflate fixtures and any
    # future disqualified payload that omits it.
    engine_crashed: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "image_digest": self.image_digest,
            "status": self.status,
            "score": self.score,
            "reason": self.reason,
        }
        if self.engine_crashed:
            out["engine_crashed"] = True
        if self.score_report is not None:
            out["score_report"] = self.score_report
        if self.sla is not None:
            out["sla"] = self.sla.to_dict()
        if self.correctness is not None:
            out["correctness"] = self.correctness.to_dict()
        return out


@dataclass
class BenchReport:
    """One round's result: the baseline reference, every entry, and the drift."""

    schema_version: int
    task_id: str
    verdict: Verdict
    started_at: str
    finished_at: str
    environment: EnvironmentInfo
    inputs_fingerprint: InputsFingerprint
    scoring_rule: dict[str, Any] = field(default_factory=dict)
    baseline: EngineSlaResult | None = None
    drift_baseline: EngineSlaResult | None = None
    baseline_drift: float | None = None
    entries: list[RoundEntryReport] = field(default_factory=list)
    # Stub/skeleton note (omitted from to_dict when empty).
    stub_note: str | None = None
    # Fail-fast skip note (omitted from to_dict when empty).
    skipped_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "verdict": self.verdict,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "environment": self.environment.to_dict(),
            "inputs_fingerprint": self.inputs_fingerprint.to_dict(),
            "scoring_rule": self.scoring_rule,
            "entries": [e.to_dict() for e in self.entries],
            "baseline_drift": self.baseline_drift,
        }
        if self.baseline is not None:
            out["baseline"] = self.baseline.to_dict()
        if self.drift_baseline is not None:
            out["drift_baseline"] = self.drift_baseline.to_dict()
        if self.stub_note:
            out["stub_note"] = self.stub_note
        if self.skipped_note:
            out["skipped_note"] = self.skipped_note
        return out
