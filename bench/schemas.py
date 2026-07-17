"""Dataclasses for bench_request / bench_report / workload_trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Workload trace
# ---------------------------------------------------------------------------


@dataclass
class TraceSampling:
    temperature: float
    top_p: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceSampling:
        return cls(temperature=float(d["temperature"]), top_p=float(d["top_p"]))


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

BenchMode = Literal["all", "correctness", "perf_screen", "sla_bench"]


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
    image: str
    serve_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineSpec:
        return cls(
            image=str(d["image"]),
            serve_args=[str(x) for x in (d.get("serve_args") or [])],
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
        )


@dataclass
class EnginesSpec:
    baseline: EngineSpec
    candidate: EngineSpec

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnginesSpec:
        return cls(
            baseline=EngineSpec.from_dict(d["baseline"]),
            candidate=EngineSpec.from_dict(d["candidate"]),
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
    mean_abs_logprob_diff: float
    max_abs_logprob_diff: float
    argmax_mismatch_rate: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CorrectnessThresholds:
        return cls(
            mean_abs_logprob_diff=float(d["mean_abs_logprob_diff"]),
            max_abs_logprob_diff=float(d["max_abs_logprob_diff"]),
            argmax_mismatch_rate=float(d["argmax_mismatch_rate"]),
        )


@dataclass
class CorrectnessConfig:
    num_prompts: int
    max_new_tokens: int
    thresholds: CorrectnessThresholds

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CorrectnessConfig:
        return cls(
            num_prompts=int(d["num_prompts"]),
            max_new_tokens=int(d["max_new_tokens"]),
            thresholds=CorrectnessThresholds.from_dict(d["thresholds"]),
        )


@dataclass
class PerfScreenConfig:
    num_requests: int
    concurrency: int
    min_throughput_ratio: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PerfScreenConfig:
        return cls(
            num_requests=int(d["num_requests"]),
            concurrency=int(d["concurrency"]),
            min_throughput_ratio=float(d["min_throughput_ratio"]),
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
    warmup_requests: int
    thresholds: SlaThresholds

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SlaBenchConfig:
        return cls(
            repetitions=int(d["repetitions"]),
            warmup_requests=int(d["warmup_requests"]),
            thresholds=SlaThresholds.from_dict(d["thresholds"]),
        )


@dataclass
class BenchRequest:
    schema_version: int
    task_id: str
    mode: BenchMode
    model: ModelSpec
    hardware: HardwareSpec
    engines: EnginesSpec
    workload_trace: WorkloadTraceRef
    correctness: CorrectnessConfig
    perf_screen: PerfScreenConfig
    sla_bench: SlaBenchConfig
    hf_token_env: str = "HF_TOKEN"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchRequest:
        mode = d.get("mode", "all")
        if mode not in ("all", "correctness", "perf_screen", "sla_bench"):
            raise ValueError(
                f"mode must be one of all|correctness|perf_screen|sla_bench, got {mode!r}"
            )
        return cls(
            schema_version=int(d["schema_version"]),
            task_id=str(d["task_id"]),
            mode=mode,  # type: ignore[arg-type]
            model=ModelSpec.from_dict(d["model"]),
            hardware=HardwareSpec.from_dict(d["hardware"]),
            engines=EnginesSpec.from_dict(d["engines"]),
            workload_trace=WorkloadTraceRef.from_dict(d["workload_trace"]),
            correctness=CorrectnessConfig.from_dict(d["correctness"]),
            perf_screen=PerfScreenConfig.from_dict(d["perf_screen"]),
            sla_bench=SlaBenchConfig.from_dict(d["sla_bench"]),
            hf_token_env=str(d.get("hf_token_env") or "HF_TOKEN"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# bench_report.json
# ---------------------------------------------------------------------------

Verdict = Literal["pass", "fail_correctness", "fail_perf_screen", "fail_sla", "error"]


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
    candidate_image_digest: str
    model_repo: str
    model_revision: str
    model_weights_sha256: str
    trace_sha256: str
    request_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectnessReport:
    verdict: str
    num_prompts: int
    num_positions_compared: int
    mean_abs_logprob_diff: float
    max_abs_logprob_diff: float
    argmax_mismatch_rate: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerfScreenReport:
    verdict: str
    baseline_output_tokens_per_s: float
    candidate_output_tokens_per_s: float
    throughput_ratio: float
    evidence: str

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
class SlaBenchReport:
    verdict: str
    repetitions: int
    candidate: EngineSlaMetrics
    baseline: EngineSlaMetrics
    speedup: dict[str, float]
    cross_rep_variance: dict[str, float]
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "repetitions": self.repetitions,
            "candidate": self.candidate.to_dict(),
            "baseline": self.baseline.to_dict(),
            "speedup": self.speedup,
            "cross_rep_variance": self.cross_rep_variance,
            "evidence": self.evidence,
        }


@dataclass
class BenchReport:
    schema_version: int
    task_id: str
    verdict: Verdict
    started_at: str
    finished_at: str
    environment: EnvironmentInfo
    inputs_fingerprint: InputsFingerprint
    correctness: CorrectnessReport | None = None
    perf_screen: PerfScreenReport | None = None
    sla_bench: SlaBenchReport | None = None
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
        }
        if self.correctness is not None:
            out["correctness"] = self.correctness.to_dict()
        if self.perf_screen is not None:
            out["perf_screen"] = self.perf_screen.to_dict()
        if self.sla_bench is not None:
            out["sla_bench"] = self.sla_bench.to_dict()
        if self.stub_note:
            out["stub_note"] = self.stub_note
        if self.skipped_note:
            out["skipped_note"] = self.skipped_note
        return out
