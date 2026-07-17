"""Module A — greedy teacher-forced logprob correctness gate (spec §5).

Baseline generates greedy continuations; both engines score the identical
forced sequence via echo+logprobs; per-position logprobs are compared against
CorrectnessThresholds. Pure URL-based logic — unit tests use in-process mocks.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bench.lifecycle import EngineError
from bench.schemas import CorrectnessConfig, CorrectnessReport, TraceRequest
from bench.validate import RequestValidationError, load_workload_trace

logger = logging.getLogger(__name__)

EVIDENCE_FILENAME = "logprob_diffs.jsonl"


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
    """First N prompts from the trace head; require text ``prompt`` fields."""
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
            f"trace request {req.id!r}: Module A requires a text prompt "
            f"(prompt_token_ids-only entries are not supported yet)"
        )
    return PromptCase(id=req.id, prompt=req.prompt)


def post_completion(
    base_url: str,
    *,
    prompt: str,
    max_tokens: int = 16,
    echo: bool = False,
    logprobs: int | None = 1,
    temperature: float = 0.0,
    seed: int | None = 0,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Serial OpenAI-compatible /v1/completions client (stdlib only)."""
    url = base_url.rstrip("/") + "/v1/completions"
    body: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "echo": echo,
        "temperature": temperature,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    if seed is not None:
        body["seed"] = seed
    data = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            payload = json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise EngineError(f"completions HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise EngineError(f"completions request failed for {url}: {exc}") from exc
    except TimeoutError as exc:
        raise EngineError(f"completions timed out for {url}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineError(
            f"invalid JSON from completions endpoint {url}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EngineError(
            f"completions response from {url} must be a JSON object, "
            f"got {type(payload).__name__}"
        )
    return payload


def probe_logprob_capability(base_url: str, *, timeout: float = 30.0) -> None:
    """Require echo+logprobs support; raise EngineError naming the gap."""
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


def extract_output_logprobs(
    resp: dict[str, Any],
    *,
    original_prompt: str,
) -> list[_PositionScore]:
    """Positions in the output portion with non-null logprobs.

    Alignment rules:
    - Skip null first-prompt-token logprob (vLLM quirk).
    - Output portion = ``text_offset >= len(original_prompt)``.
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

    prompt_len = len(original_prompt)
    out: list[_PositionScore] = []
    for i in range(n):
        try:
            off = int(text_offset[i])
        except (TypeError, ValueError) as exc:
            raise EngineError(
                f"malformed logprobs.text_offset[{i}]: {text_offset[i]!r}"
            ) from exc
        if off < prompt_len:
            continue
        raw_lp = token_logprobs[i]
        if raw_lp is None:
            continue
        try:
            lp_val = float(raw_lp)
        except (TypeError, ValueError) as exc:
            raise EngineError(
                f"malformed logprobs.token_logprobs[{i}]: {raw_lp!r}"
            ) from exc
        top = top_logprobs[i]
        top_dict = top if isinstance(top, dict) else None
        out.append(
            _PositionScore(
                position=i,
                token=str(tokens[i]),
                text_offset=off,
                logprob=lp_val,
                top1=_top1_token(top_dict),
            )
        )
    return out


def scoring_order(prompt_ids: list[str], task_id: str) -> list[str]:
    """Deterministic shuffle of scoring order seeded by task_id (spec §5.3)."""
    order = list(prompt_ids)
    rng = random.Random(task_id)
    rng.shuffle(order)
    return order


@dataclass
class BaselineCorrectnessPhase:
    """Baseline generate+score results; candidate can run after baseline teardown."""

    prompts: list[PromptCase]
    scored: list[tuple[PromptCase, str, list[_PositionScore]]]


def collect_baseline_correctness(
    baseline_url: str,
    *,
    prompts: list[PromptCase],
    cfg: CorrectnessConfig,
    task_id: str,
    request_timeout_s: float = 60.0,
) -> BaselineCorrectnessPhase:
    """Generate forced continuations and score them on the baseline engine."""
    if not prompts:
        raise RequestValidationError("correctness: no prompts to evaluate")

    probe_logprob_capability(baseline_url, timeout=request_timeout_s)

    forced: dict[str, str] = {}
    for case in prompts:
        gen = post_completion(
            baseline_url,
            prompt=case.prompt,
            max_tokens=cfg.max_new_tokens,
            echo=False,
            logprobs=None,
            temperature=0.0,
            seed=0,
            timeout=request_timeout_s,
        )
        try:
            continuation = str(gen["choices"][0]["text"])
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError(
                f"baseline generation malformed for {case.id}: {exc}"
            ) from exc
        if continuation == "":
            raise EngineError(f"baseline generated empty continuation for {case.id}")
        forced[case.id] = continuation
        logger.info(
            "correctness generate id=%s prompt_len=%d cont_len=%d",
            case.id,
            len(case.prompt),
            len(continuation),
        )

    by_id = {c.id: c for c in prompts}
    order = scoring_order([c.id for c in prompts], task_id)
    scored: list[tuple[PromptCase, str, list[_PositionScore]]] = []
    for pid in order:
        case = by_id[pid]
        continuation = forced[pid]
        full = case.prompt + continuation
        base_resp = post_completion(
            baseline_url,
            prompt=full,
            max_tokens=0,
            echo=True,
            logprobs=1,
            temperature=0.0,
            seed=0,
            timeout=request_timeout_s,
        )
        base_scores = extract_output_logprobs(base_resp, original_prompt=case.prompt)
        if not base_scores:
            raise EngineError(
                f"baseline echo scoring produced 0 output positions for {case.id}"
            )
        scored.append((case, continuation, base_scores))
    return BaselineCorrectnessPhase(prompts=prompts, scored=scored)


def finish_correctness_with_candidate(
    candidate_url: str,
    baseline: BaselineCorrectnessPhase,
    *,
    cfg: CorrectnessConfig,
    evidence_dir: Path,
    request_timeout_s: float = 60.0,
) -> CorrectnessReport:
    """Score forced sequences on the candidate and compare to baseline."""
    probe_logprob_capability(candidate_url, timeout=request_timeout_s)

    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / EVIDENCE_FILENAME
    partial_path = evidence_dir / f"{EVIDENCE_FILENAME}.partial"

    abs_diffs: list[float] = []
    argmax_mismatches = 0
    positions_compared = 0

    with partial_path.open("w", encoding="utf-8") as ef:
        for case, continuation, base_scores in baseline.scored:
            full = case.prompt + continuation
            cand_resp = post_completion(
                candidate_url,
                prompt=full,
                max_tokens=0,
                echo=True,
                logprobs=1,
                temperature=0.0,
                seed=0,
                timeout=request_timeout_s,
            )
            cand_scores = extract_output_logprobs(
                cand_resp, original_prompt=case.prompt
            )
            if len(base_scores) != len(cand_scores):
                raise EngineError(
                    f"output logprob length mismatch for {case.id}: "
                    f"baseline={len(base_scores)} candidate={len(cand_scores)}"
                )
            for b, c in zip(base_scores, cand_scores, strict=True):
                if b.text_offset != c.text_offset:
                    raise EngineError(
                        f"text_offset mismatch for {case.id} at position "
                        f"{b.position}: baseline={b.text_offset} "
                        f"candidate={c.text_offset}"
                    )
                # Same forced token is required before any logprob comparison.
                if b.token != c.token:
                    raise EngineError(
                        f"forced token mismatch for {case.id} at position "
                        f"{b.position} (offset={b.text_offset}): "
                        f"baseline={b.token!r} candidate={c.token!r}"
                    )
                diff = abs(b.logprob - c.logprob)
                abs_diffs.append(diff)
                mismatch = b.top1 != c.top1
                if mismatch:
                    argmax_mismatches += 1
                positions_compared += 1
                ef.write(
                    json.dumps(
                        {
                            "prompt_id": case.id,
                            "position": b.position,
                            "text_offset": b.text_offset,
                            "token_baseline": b.token,
                            "token_candidate": c.token,
                            "logprob_baseline": b.logprob,
                            "logprob_candidate": c.logprob,
                            "abs_diff": diff,
                            "top1_baseline": b.top1,
                            "top1_candidate": c.top1,
                            "argmax_mismatch": mismatch,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    if positions_compared == 0:
        raise EngineError(
            "correctness compared 0 output positions "
            "(empty continuations or alignment produced no scores)"
        )

    partial_path.replace(evidence_path)

    mean_diff = sum(abs_diffs) / positions_compared
    max_diff = max(abs_diffs)
    argmax_rate = argmax_mismatches / positions_compared
    thr = cfg.thresholds
    # Spec §5.1: all three metrics must be strictly below their thresholds.
    passed = (
        mean_diff < thr.mean_abs_logprob_diff
        and max_diff < thr.max_abs_logprob_diff
        and argmax_rate < thr.argmax_mismatch_rate
    )
    verdict = "pass" if passed else "fail_correctness"
    rel_evidence = f"evidence/correctness/{EVIDENCE_FILENAME}"
    logger.info(
        "correctness done verdict=%s positions=%d mean=%.6g max=%.6g argmax=%.6g",
        verdict,
        positions_compared,
        mean_diff,
        max_diff,
        argmax_rate,
    )
    return CorrectnessReport(
        verdict=verdict,
        num_prompts=len(baseline.prompts),
        num_positions_compared=positions_compared,
        mean_abs_logprob_diff=mean_diff,
        max_abs_logprob_diff=max_diff,
        argmax_mismatch_rate=argmax_rate,
        evidence=rel_evidence,
    )


def run_correctness(
    baseline_url: str,
    candidate_url: str,
    *,
    prompts: list[PromptCase],
    cfg: CorrectnessConfig,
    task_id: str,
    evidence_dir: Path,
    request_timeout_s: float = 60.0,
) -> CorrectnessReport:
    """Run Module A when both engines are reachable (e.g. in-process mocks)."""
    baseline = collect_baseline_correctness(
        baseline_url,
        prompts=prompts,
        cfg=cfg,
        task_id=task_id,
        request_timeout_s=request_timeout_s,
    )
    return finish_correctness_with_candidate(
        candidate_url,
        baseline,
        cfg=cfg,
        evidence_dir=evidence_dir,
        request_timeout_s=request_timeout_s,
    )
