"""Correctness gate: one shared scorer grades every candidate's own output.

Each candidate engine runs once, in production configuration, and its SLA
replay captures the text it produced. Once every candidate has stopped, a
single scorer engine (the campaign's pinned baseline image, started with the
correctness serve args) teacher-forces each captured output and reports the
logprob it assigns to the candidate's own tokens. An engine serving the
pinned model scores near that model's own greedy path; nonsense scores far
below it.

The bars in ``CorrectnessThresholds`` are absolute logprobs rather than a
comparison against a reference run, because one scorer grades everything and
there is no second set of logprobs to compare against. The gate therefore
asks whether an output is plausible under the pinned model, not whether it
matches the baseline engine token for token: a candidate that answers
differently but sensibly passes.

Pure URL-based logic. The scorer container's lifecycle lives in bench/main.py,
which starts it once, grades every candidate through it, and stops it.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.http import post_completion
from bench.lifecycle import EngineError
from bench.mock_engine import response_shape_fingerprint
from bench.schemas import CorrectnessConfig, CorrectnessReport, TraceRequest
from bench.validate import RequestValidationError, load_workload_trace

logger = logging.getLogger(__name__)

SHAPE_EVIDENCE_FILENAME = "completion_response_shape.json"


@dataclass(frozen=True)
class PromptCase:
    id: str
    prompt: str


def resolve_trace_path(trace_path: str, *, request_path: Path | None = None) -> Path:
    """Resolve workload_trace.path: absolute as-is; relative = request dir, then CWD."""
    p = Path(trace_path)
    if p.is_absolute():
        if p.is_file():
            return p.resolve()
        raise RequestValidationError(f"workload trace not found: {trace_path}")
    candidates: list[Path] = []
    if request_path is not None:
        candidates.append((request_path.parent / p).resolve())
    candidates.append((Path.cwd() / p).resolve())
    for c in candidates:
        if c.is_file():
            return c
    raise RequestValidationError(f"workload trace not found: {trace_path}")


def load_correctness_prompts(
    *,
    trace_ref_path: str,
    expected_sha256: str,
    num_prompts: int,
    request_path: Path,
) -> list[PromptCase]:
    """Resolve, verify, and select prompts before engines start."""
    trace_path = resolve_trace_path(trace_ref_path, request_path=request_path)
    return select_correctness_prompts(
        trace_path=trace_path,
        expected_sha256=expected_sha256,
        num_prompts=num_prompts,
    )


def select_correctness_prompts(
    *,
    trace_path: Path,
    expected_sha256: str,
    num_prompts: int,
) -> list[PromptCase]:
    """First N prompts from the trace head; require text ``prompt`` fields.

    These are the trace requests whose captured output the scorer grades. The
    ids match the SLA replay's request ids, so a captured output is always
    paired with the prompt that produced it.
    """
    if num_prompts < 1:
        raise RequestValidationError("correctness.num_prompts must be >= 1")
    trace = load_workload_trace(trace_path, expected_sha256=expected_sha256)
    if len(trace.requests) < num_prompts:
        raise RequestValidationError(
            f"trace has {len(trace.requests)} requests but "
            f"correctness.num_prompts={num_prompts}"
        )
    out: list[PromptCase] = []
    for req in trace.requests[:num_prompts]:
        out.append(_prompt_case_from_trace_request(req))
    return out


def _prompt_case_from_trace_request(req: TraceRequest) -> PromptCase:
    if req.prompt is None or req.prompt == "":
        raise RequestValidationError(
            f"trace request {req.id!r}: correctness requires a text prompt "
            f"(prompt_token_ids-only entries are not supported yet)"
        )
    return PromptCase(id=req.id, prompt=req.prompt)


def probe_logprob_capability(base_url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Require echo+logprobs support; return the probe response for shape capture."""
    resp = post_completion(
        base_url,
        prompt="probe",
        max_tokens=0,
        echo=True,
        logprobs=1,
        temperature=0.0,
        timeout=timeout,
    )
    try:
        choice = resp["choices"][0]
        lp = choice.get("logprobs")
    except (KeyError, IndexError, TypeError) as exc:
        raise EngineError(
            f"engine at {base_url} missing logprob capability: "
            f"malformed /v1/completions response ({exc})"
        ) from exc
    if not isinstance(lp, dict):
        raise EngineError(
            f"engine at {base_url} missing logprob capability: "
            f"choices[0].logprobs is not an object"
        )
    for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset"):
        if key not in lp:
            raise EngineError(
                f"engine at {base_url} missing logprob capability: "
                f"logprobs.{key} absent"
            )
    return resp


def write_completion_response_shape(evidence_dir: Path, resp: dict[str, Any]) -> Path:
    """Persist structural fingerprint of a real /v1/completions response."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / SHAPE_EVIDENCE_FILENAME
    payload = {
        "fingerprint": response_shape_fingerprint(resp),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _top1_token(top: dict[str, float] | None) -> str | None:
    if not top:
        return None
    # Highest logprob wins; ties broken by insertion order via max + key.
    return max(top.items(), key=lambda kv: kv[1])[0]


@dataclass
class _PositionScore:
    position: int
    token: str
    text_offset: int
    logprob: float
    top1: str | None


def _score_at(
    i: int,
    *,
    tokens: list[Any],
    token_logprobs: list[Any],
    top_logprobs: list[Any],
    text_offset: int,
) -> _PositionScore | None:
    raw_lp = token_logprobs[i]
    if raw_lp is None:
        return None
    try:
        lp_val = float(raw_lp)
    except (TypeError, ValueError) as exc:
        raise EngineError(
            f"malformed logprobs.token_logprobs[{i}]: {raw_lp!r}"
        ) from exc
    top = top_logprobs[i]
    top_dict = top if isinstance(top, dict) else None
    return _PositionScore(
        position=i,
        token=str(tokens[i]),
        text_offset=text_offset,
        logprob=lp_val,
        top1=_top1_token(top_dict),
    )


def extract_output_logprobs(
    resp: dict[str, Any],
    *,
    original_prompt: str,
    continuation: str,
) -> list[_PositionScore]:
    """Positions in the forced continuation with non-null logprobs.

    Alignment rules:
    - Skip null first-prompt-token logprob (vLLM quirk).
    - Forced span is ``len(P) <= text_offset < len(P+C)``.
    - Extra tokens from the HTTP max_tokens=0→1 clamp are dropped.
    """
    try:
        lp = resp["choices"][0]["logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EngineError(f"malformed completions response: {exc}") from exc
    if not isinstance(lp, dict):
        raise EngineError("completions response missing logprobs object")

    tokens = lp.get("tokens") or []
    token_logprobs = lp.get("token_logprobs") or []
    top_logprobs = lp.get("top_logprobs") or []
    text_offset = lp.get("text_offset") or []
    n = len(tokens)
    if not (
        len(token_logprobs) == n and len(top_logprobs) == n and len(text_offset) == n
    ):
        raise EngineError(
            "logprobs arrays misaligned "
            f"(tokens={n}, logprobs={len(token_logprobs)}, "
            f"top={len(top_logprobs)}, offset={len(text_offset)})"
        )

    cut_lo = len(original_prompt)
    cut_hi = cut_lo + len(continuation)
    out: list[_PositionScore] = []
    for i in range(n):
        try:
            off = int(text_offset[i])
        except (TypeError, ValueError) as exc:
            raise EngineError(
                f"malformed logprobs.text_offset[{i}]: {text_offset[i]!r}"
            ) from exc
        if off < cut_lo or off >= cut_hi:
            continue
        scored = _score_at(
            i,
            tokens=tokens,
            token_logprobs=token_logprobs,
            top_logprobs=top_logprobs,
            text_offset=off,
        )
        if scored is not None:
            out.append(scored)
    if out:
        return out
    # SGLang: offsets are often -1 or 0-based into choices[0].text, so the
    # char-offset cut yields nothing. Echo arrays are P+C plus the clamp
    # extra (usage.completion_tokens), or C only. Never treat
    # completion_tokens as the scored window.
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    try:
        n_comp = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        n_comp = 0
    n_extra = n_comp if n > n_comp > 0 else 0
    end = n - n_extra
    body = [str(tokens[i]) for i in range(end)]
    joined = "".join(body)
    if joined == continuation:
        start = 0
    elif joined == original_prompt + continuation:
        acc = 0
        start = None
        for i, piece in enumerate(body):
            if acc == cut_lo:
                start = i
                break
            acc += len(piece)
        if start is None or "".join(body[start:]) != continuation:
            raise EngineError(
                "echo logprobs did not split on the forced continuation "
                f"(n={n} n_comp={n_comp} prompt_len={cut_lo} "
                f"cont_len={len(continuation)})"
            )
    else:
        raise EngineError(
            "echo logprobs did not reconstruct the forced sequence "
            f"(n={n} n_comp={n_comp} joined_len={len(joined)} "
            f"prompt_len={cut_lo} cont_len={len(continuation)})"
        )
    logger.warning(
        "logprobs text_offset missed forced span "
        "(n=%d n_comp=%d n_extra=%d start=%d end=%d prompt_len=%d cont_len=%d)",
        n,
        n_comp,
        n_extra,
        start,
        end,
        cut_lo,
        len(continuation),
    )
    for i in range(start, end):
        try:
            off = int(text_offset[i])
        except (TypeError, ValueError):
            off = i
        scored = _score_at(
            i,
            tokens=tokens,
            token_logprobs=token_logprobs,
            top_logprobs=top_logprobs,
            text_offset=off,
        )
        if scored is not None:
            out.append(scored)
    return out


@dataclass(frozen=True)
class CapturedOutput:
    """One request's output as the candidate actually produced it."""

    request_id: str
    prompt: str
    output_text: str
    completion_tokens: int


@dataclass
class PendingCorrectness:
    """A candidate waiting on the shared scorer.

    Queued while the candidate's own container is still running, graded in one
    batch once every candidate has stopped and the single scorer is up.
    """

    candidate_index: int
    outputs: list[CapturedOutput]


def capture_outputs(
    prompts: list[PromptCase],
    *,
    timings,
    outputs: dict[str, str],
) -> list[CapturedOutput]:
    """Pair the correctness prompts with what the candidate emitted for them.

    A request the engine never answered is dropped here rather than graded as
    a wrong answer: the SLA replay already fails the run on a request error.
    """
    captured: list[CapturedOutput] = []
    for case in prompts:
        text = outputs.get(case.id)
        if text is None:
            continue
        timing = timings.get(case.id)
        captured.append(
            CapturedOutput(
                request_id=case.id,
                prompt=case.prompt,
                output_text=text,
                completion_tokens=(timing.completion_tokens if timing else 0),
            )
        )
    return captured


def score_captured_output(
    scorer_url: str,
    captured: CapturedOutput,
    *,
    request_timeout_s: float = 300.0,
) -> list[float]:
    """Logprobs the scorer assigns to the candidate's own forced tokens."""
    full = captured.prompt + captured.output_text
    resp = post_completion(
        scorer_url,
        prompt=full,
        max_tokens=0,
        echo=True,
        logprobs=1,
        temperature=0.0,
        seed=0,
        timeout=request_timeout_s,
    )
    scores = extract_output_logprobs(
        resp,
        original_prompt=captured.prompt,
        continuation=captured.output_text,
    )
    return [s.logprob for s in scores]


def grade_candidate(
    scorer_url: str,
    outputs: list[CapturedOutput],
    *,
    cfg: CorrectnessConfig,
    evidence_path: Path,
    request_timeout_s: float = 300.0,
) -> CorrectnessReport:
    """Teacher-force one candidate's captured outputs through the scorer.

    Three outcomes, and the difference between the last two matters:

    * ``pass``: the scorer finds the output plausible.
    * ``fail_correctness``: the output is wrong. The entry is disqualified.
    * ``infra_failed``: the scorer produced too few logprobs to judge on.
      That is a harness problem, and the entry is requeued, not disqualified.
    """
    thr = cfg.thresholds
    logprobs: list[float] = []
    streamed_tokens = 0
    empty: list[str] = []

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    partial = evidence_path.with_suffix(evidence_path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as ef:
        for captured in outputs:
            if not captured.output_text:
                empty.append(captured.request_id)
                continue
            scored = score_captured_output(
                scorer_url, captured, request_timeout_s=request_timeout_s
            )
            logprobs.extend(scored)
            streamed_tokens += captured.completion_tokens
            ef.write(
                json.dumps(
                    {
                        "request_id": captured.request_id,
                        "streamed_tokens": captured.completion_tokens,
                        "scored_positions": len(scored),
                        "mean_logprob": (sum(scored) / len(scored) if scored else None),
                        "min_logprob": min(scored) if scored else None,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    partial.replace(evidence_path)

    rel_evidence = f"evidence/correctness/{evidence_path.name}"
    if empty:
        return CorrectnessReport(
            verdict="fail_correctness",
            num_prompts=len(outputs),
            num_positions_scored=len(logprobs),
            mean_logprob=0.0,
            min_logprob=0.0,
            coverage_ratio=0.0,
            evidence=rel_evidence,
            reason=f"engine returned no output for {len(empty)} prompt(s): {empty[0]}",
        )

    # Coverage is measured against what the candidate streamed. A near-empty
    # extraction means the scorer never saw the output, which is infrastructure,
    # not a wrong answer. Tokenizers differ between the stream and the scorer's
    # echo, so the floor is loose on purpose.
    coverage = (len(logprobs) / streamed_tokens) if streamed_tokens > 0 else 0.0
    if not logprobs:
        return CorrectnessReport(
            verdict="infra_failed",
            num_prompts=len(outputs),
            num_positions_scored=0,
            mean_logprob=0.0,
            min_logprob=0.0,
            coverage_ratio=0.0,
            evidence=rel_evidence,
            reason="scorer produced no logprobs for any captured output",
        )
    if coverage < thr.min_coverage_ratio:
        return CorrectnessReport(
            verdict="infra_failed",
            num_prompts=len(outputs),
            num_positions_scored=len(logprobs),
            mean_logprob=statistics.fmean(logprobs),
            min_logprob=min(logprobs),
            coverage_ratio=coverage,
            evidence=rel_evidence,
            reason=(
                f"scorer covered {len(logprobs)}/{streamed_tokens} streamed tokens, "
                f"below the {thr.min_coverage_ratio:.0%} floor"
            ),
        )

    mean_lp = statistics.fmean(logprobs)
    min_lp = min(logprobs)
    reason: str | None = None
    if mean_lp < thr.min_mean_logprob:
        reason = f"mean logprob {mean_lp:.3f} below {thr.min_mean_logprob}"
    elif min_lp < thr.min_token_logprob:
        reason = f"min token logprob {min_lp:.3f} below {thr.min_token_logprob}"

    verdict = "fail_correctness" if reason else "pass"
    logger.info(
        "correctness %s positions=%d mean=%.4f min=%.4f coverage=%.3f",
        verdict,
        len(logprobs),
        mean_lp,
        min_lp,
        coverage,
    )
    return CorrectnessReport(
        verdict=verdict,
        num_prompts=len(outputs),
        num_positions_scored=len(logprobs),
        mean_logprob=mean_lp,
        min_logprob=min_lp,
        coverage_ratio=coverage,
        evidence=rel_evidence,
        reason=reason,
    )


def grade_all(
    scorer_url: str,
    pending: list[PendingCorrectness],
    *,
    cfg: CorrectnessConfig,
    evidence_dir: Path,
    request_timeout_s: float = 300.0,
) -> dict[int, CorrectnessReport]:
    """Grade every queued candidate against one already-running scorer.

    The caller starts the scorer, calls this, and stops it, so correctness
    costs one engine start per round however many candidates the round holds.
    """
    probe_resp = probe_logprob_capability(scorer_url, timeout=request_timeout_s)
    write_completion_response_shape(evidence_dir, probe_resp)
    reports: dict[int, CorrectnessReport] = {}
    for item in pending:
        reports[item.candidate_index] = grade_candidate(
            scorer_url,
            item.outputs,
            cfg=cfg,
            evidence_path=evidence_dir / f"candidate_{item.candidate_index}.jsonl",
            request_timeout_s=request_timeout_s,
        )
    return reports
