"""Correctness gate: one shared scorer grades every candidate's own output.

Each candidate engine runs once, in production configuration, and its SLA
replay captures the text it produced. Once every candidate has stopped, a
single scorer engine (the campaign's pinned baseline image, started with the
correctness serve args) teacher-forces each captured output and reports the
logprob it assigns to the candidate's own tokens. An engine serving the
pinned model scores near that model's own greedy path; nonsense scores far
below it.

Most bars in ``CorrectnessThresholds`` are absolute logprobs. The gate asks
whether an output is plausible under the pinned model, not whether it matches
the baseline engine token for token: a candidate that answers differently but
sensibly passes.

Absolute plausibility is not monotone in output quality (PAR-108): a repeat
loop is the most predictable text there is, so it scores *better* than a real
answer on every absolute bar. Two mandatory harness checks identify it:

* distinct character n-grams separate short-period loops from real answers;
* the longest repeated span catches a miner that scales the loop period with
  the output budget.

They are exploit checks, not competition parameters, and therefore do not
live in the campaign manifest or bench request. For an ordinary completion,
the checks grade the whole output. For a forced-length completion, the pinned
baseline is replayed once with EOS handling restored. Absolute checks grade
the candidate only through that trusted, prompt-specific stop boundary, while
the complete forced output must not be more degenerate than the baseline's
complete output. Candidate-reported stop positions are never used.

The manifest-pinned ``max_mean_logprob_drop`` separately catches a candidate
that degrades the model and still clears the absolute floor.

When the relative bar is enabled, the baseline is queued under
``BASELINE_INDEX`` and graded first so that reference exists. A baseline that
cannot be graded fails the round, never a candidate. Campaigns without that
bar retain the pre-PAR-108 grading path.

The min-token bar is applied to the k-th lowest scored position rather than
the outright minimum (PAR-94). Scorer and candidate are separate instances of
the same image, so numerical divergence can flip the argmax at a single
high-entropy position and have the scorer rate the candidate's own token near
zero probability. Round 7 of campaign de622a8d disqualified a behaviourally
identical noop on exactly one such token out of 4097.

Pure URL-based logic. The scorer container's lifecycle lives in bench/main.py,
which starts it once, grades every candidate through it, and stops it.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bench.http import post_completion
from bench.lifecycle import EngineError
from bench.mock_engine import response_shape_fingerprint
from bench.schemas import CorrectnessConfig, CorrectnessReport, TraceRequest
from bench.validate import RequestValidationError, load_workload_trace

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.sla_bench import NaturalStopReference

logger = logging.getLogger(__name__)

SHAPE_EVIDENCE_FILENAME = "completion_response_shape.json"

# The baseline queues its outputs under this index so the relative bar has a
# reference. No candidate index is negative.
BASELINE_INDEX = -1

# N-gram width for the repetition bar, in characters, and the small length
# below which neither text bar applies. These are harness policy, not campaign
# parameters. The baseline stop boundary is prompt-specific, so none of them
# encode or assume a particular output-token budget.
DEGENERACY_NGRAM = 16
DEGENERACY_MIN_CHARS = 64
DEGENERACY_MIN_DISTINCT_NGRAM_RATIO = 0.15
DEGENERACY_MAX_REPEATED_SPAN_RATIO = 0.25


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


def distinct_ngram_ratio(text: str, n: int = DEGENERACY_NGRAM) -> float:
    """Share of the output's character n-grams that are distinct.

    Characters, not words, because splitting on whitespace hands the miner a
    one-character evasion: a loop emitted without spaces is a single word and
    no word-level statistic sees it.

    Counting distinct n-grams rather than the share taken by the most common
    one gives a statable property: text repeating a unit of p characters over
    a length L lands near p/L, so the bar catches any repeated unit shorter
    than ``floor * L``. A tuned loop with a long enough period still clears
    it; this bar is the cheap stop, not a proof.

    Prose and code sit near 1.0 because almost every n-gram occurs once.
    """
    if len(text) < n:
        return 1.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams)


def longest_repeated_substring_ratio(text: str) -> float:
    """Length of the longest repeated character span, divided by text length.

    A suffix automaton finds the longest substring occurring at least twice in
    linear time. Overlapping occurrences count: a periodic output has a ratio
    near 1.0 however long its repeated unit is, while honest repeated
    boilerplate occupies only the boilerplate's share of the answer.
    """
    if not text:
        return 0.0

    transitions: list[dict[str, int]] = [{}]
    links = [-1]
    lengths = [0]
    occurrences = [0]
    last = 0

    for char in text:
        current = len(transitions)
        transitions.append({})
        links.append(0)
        lengths.append(lengths[last] + 1)
        occurrences.append(1)

        state = last
        while state >= 0 and char not in transitions[state]:
            transitions[state][char] = current
            state = links[state]

        if state < 0:
            links[current] = 0
        else:
            target = transitions[state][char]
            if lengths[state] + 1 == lengths[target]:
                links[current] = target
            else:
                clone = len(transitions)
                transitions.append(dict(transitions[target]))
                links.append(links[target])
                lengths.append(lengths[state] + 1)
                occurrences.append(0)
                while state >= 0 and transitions[state].get(char) == target:
                    transitions[state][char] = clone
                    state = links[state]
                links[target] = clone
                links[current] = clone
        last = current

    length_counts = [0] * (len(text) + 1)
    for length in lengths:
        length_counts[length] += 1
    for length in range(1, len(length_counts)):
        length_counts[length] += length_counts[length - 1]
    states_by_length = [0] * len(transitions)
    for state in range(len(transitions) - 1, -1, -1):
        length = lengths[state]
        length_counts[length] -= 1
        states_by_length[length_counts[length]] = state
    for state in reversed(states_by_length):
        if state == 0:
            continue
        occurrences[links[state]] += occurrences[state]
    longest = max(
        (
            lengths[state]
            for state in range(1, len(transitions))
            if occurrences[state] >= 2
        ),
        default=0,
    )
    return longest / len(text)


def degeneracy_reason(
    text: str,
    *,
    distinct_ratio: float | None = None,
    repeated_span_ratio: float | None = None,
) -> str | None:
    """Why this output reads as a repeat loop, or None if it does not.

    The thresholds are harness invariants rather than campaign policy. A miner
    cannot opt out through a legacy manifest or tune a competition around the
    exact exploit boundary.
    """
    if len(text) < DEGENERACY_MIN_CHARS:
        return None
    ratio = distinct_ratio if distinct_ratio is not None else distinct_ngram_ratio(text)
    if ratio < DEGENERACY_MIN_DISTINCT_NGRAM_RATIO:
        return (
            f"distinct {DEGENERACY_NGRAM}-gram ratio {ratio:.3f} below harness floor "
            f"over {len(text)} chars"
        )
    ratio = (
        repeated_span_ratio
        if repeated_span_ratio is not None
        else longest_repeated_substring_ratio(text)
    )
    if ratio >= DEGENERACY_MAX_REPEATED_SPAN_RATIO:
        return (
            f"longest repeated span ratio {ratio:.3f} at or above harness ceiling "
            f"over {len(text)} chars"
        )
    return None


def relative_degeneracy_reason(
    text: str,
    *,
    baseline_distinct_ratio: float,
    baseline_repeated_span_ratio: float,
    distinct_ratio: float | None = None,
    repeated_span_ratio: float | None = None,
) -> str | None:
    """Why a forced output is more degenerate than the same baseline output.

    The absolute bars still define the suspicious region. The baseline only
    prevents expected forced-length padding from becoming a false positive;
    it does not make ordinary non-degenerate text compete on tiny metric
    differences.
    """
    if len(text) < DEGENERACY_MIN_CHARS:
        return None
    distinct = (
        distinct_ratio if distinct_ratio is not None else distinct_ngram_ratio(text)
    )
    if (
        distinct < DEGENERACY_MIN_DISTINCT_NGRAM_RATIO
        and distinct < baseline_distinct_ratio
    ):
        return (
            f"distinct {DEGENERACY_NGRAM}-gram ratio {distinct:.3f} below "
            f"baseline {baseline_distinct_ratio:.3f} over {len(text)} chars"
        )
    repeated = (
        repeated_span_ratio
        if repeated_span_ratio is not None
        else longest_repeated_substring_ratio(text)
    )
    if (
        repeated >= DEGENERACY_MAX_REPEATED_SPAN_RATIO
        and repeated > baseline_repeated_span_ratio
    ):
        return (
            f"longest repeated span ratio {repeated:.3f} above baseline "
            f"{baseline_repeated_span_ratio:.3f} over {len(text)} chars"
        )
    return None


def quantile_low(values: list[float], quantile: float) -> float:
    """The k-th lowest of ``values``, k = ceil(``quantile`` * len(values)).

    k is clamped to at least 1, so a quantile of 0 and any sample too small
    for the quantile to reach a second position both give the plain minimum.
    """
    if not values:
        raise ValueError("quantile_low requires at least one value")
    k = max(1, math.ceil(quantile * len(values)))
    k = min(k, len(values))
    return sorted(values)[k - 1]


@dataclass(frozen=True)
class CapturedOutput:
    """One request's output as the candidate actually produced it."""

    request_id: str
    prompt: str
    output_text: str
    completion_tokens: int


@dataclass(frozen=True)
class BaselineDegeneracyReference:
    """Trusted prompt-specific bounds for forced-length loop checks."""

    natural_stop_tokens: int
    full_distinct_ngram_ratio: float
    full_repeated_span_ratio: float


@dataclass
class PendingCorrectness:
    """An engine's outputs waiting on the shared scorer.

    Queued while the engine's own container is still running, graded in one
    batch once every engine has stopped and the single scorer is up. The
    baseline is queued the same way, under ``BASELINE_INDEX``.
    """

    candidate_index: int
    outputs: list[CapturedOutput]

    @property
    def evidence_name(self) -> str:
        if self.candidate_index == BASELINE_INDEX:
            return "baseline"
        return f"candidate_{self.candidate_index}"


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


def build_baseline_degeneracy_references(
    outputs: list[CapturedOutput],
    natural_stops: Mapping[str, NaturalStopReference],
    output_samples: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, BaselineDegeneracyReference]:
    """Build prompt-specific bounds from every measured baseline repetition."""
    references: dict[str, BaselineDegeneracyReference] = {}
    for captured in outputs:
        stop = natural_stops.get(captured.request_id)
        if stop is None:
            raise EngineError(
                "baseline natural-stop reference missing correctness request "
                f"{captured.request_id!r}"
            )
        if stop.completion_tokens < 1 or not stop.text:
            raise EngineError(
                "baseline natural-stop reference is empty for correctness request "
                f"{captured.request_id!r}"
            )
        natural_reason = degeneracy_reason(stop.text)
        if natural_reason is not None:
            raise EngineError(
                f"baseline natural output {captured.request_id!r} is degenerate: "
                f"{natural_reason}"
            )
        samples = (
            (captured.output_text,)
            if output_samples is None
            else output_samples.get(captured.request_id, ())
        )
        if not samples:
            raise EngineError(
                "baseline output samples missing correctness request "
                f"{captured.request_id!r}"
            )
        references[captured.request_id] = BaselineDegeneracyReference(
            natural_stop_tokens=stop.completion_tokens,
            full_distinct_ngram_ratio=min(
                distinct_ngram_ratio(text) for text in samples
            ),
            full_repeated_span_ratio=max(
                longest_repeated_substring_ratio(text) for text in samples
            ),
        )
    return references


def score_captured_output(
    scorer_url: str,
    captured: CapturedOutput,
    *,
    request_timeout_s: float = 300.0,
) -> tuple[list[_PositionScore], int]:
    """Scored token positions in the candidate's forced output.

    Returns the scorer's token-aligned positions and the size of the forced
    span. Both come from the scorer, so the candidate cannot choose where the
    trusted baseline stop boundary lands in its own text.
    """
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
    if not scores:
        return [], 0
    # Positions in the forced span are contiguous, and a position is only
    # dropped when the scorer returned no logprob for it, so first..last spans
    # everything the scorer saw of this output.
    span = scores[-1].position - scores[0].position + 1
    return scores, span


def grade_candidate(
    scorer_url: str,
    outputs: list[CapturedOutput],
    *,
    cfg: CorrectnessConfig,
    evidence_path: Path,
    baseline_outputs: Mapping[str, str] | None = None,
    request_timeout_s: float = 300.0,
    baseline_mean_logprob: float | None = None,
    baseline_degeneracy: Mapping[str, BaselineDegeneracyReference] | None = None,
) -> CorrectnessReport:
    """Teacher-force one engine's captured outputs through the scorer.

    Three outcomes, and the difference between the last two matters:

    * ``pass``: the scorer finds the output plausible.
      An empty output also passes for a request where the opening baseline's
      captured output was empty; there are no continuation tokens to score,
      and matching the pinned reference must not disqualify a candidate.
    * ``fail_correctness``: the output is wrong. The entry is disqualified.
    * ``infra_failed``: the scorer produced too few logprobs to judge on.
      That is a harness problem, and the entry is requeued, not disqualified.

    ``baseline_mean_logprob`` is the baseline's own mean through this same
    scorer. It is None when grading the baseline itself, and when the
    campaign carries no ``max_mean_logprob_drop``; either way the relative
    bar is skipped.
    """
    thr = cfg.thresholds
    logprobs: list[float] = []
    span_positions = 0
    empty: list[str] = []
    baseline_empty_matches: list[str] = []
    baseline_outputs = baseline_outputs or {}
    degenerate: str | None = None

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    partial = evidence_path.with_suffix(evidence_path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as ef:
        for captured in outputs:
            if not captured.output_text:
                if baseline_outputs.get(captured.request_id) == "":
                    baseline_empty_matches.append(captured.request_id)
                    ef.write(
                        json.dumps(
                            {
                                "request_id": captured.request_id,
                                "streamed_tokens": captured.completion_tokens,
                                "span_positions": 0,
                                "scored_positions": 0,
                                "mean_logprob": None,
                                "min_logprob": None,
                                "baseline_empty_match": True,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue
                empty.append(captured.request_id)
                continue
            positions, span = score_captured_output(
                scorer_url, captured, request_timeout_s=request_timeout_s
            )
            scored = [position.logprob for position in positions]
            logprobs.extend(scored)
            span_positions += span
            distinct_ratio = distinct_ngram_ratio(captured.output_text)
            repeated_span_ratio = longest_repeated_substring_ratio(captured.output_text)
            reference = (
                None
                if baseline_degeneracy is None
                else baseline_degeneracy.get(captured.request_id)
            )
            if baseline_degeneracy is not None and reference is None:
                raise EngineError(
                    "baseline degeneracy reference missing correctness request "
                    f"{captured.request_id!r}"
                )
            if reference is None:
                prefix_text = captured.output_text
                prefix_distinct_ratio = distinct_ratio
                prefix_repeated_span_ratio = repeated_span_ratio
                this_degenerate = degeneracy_reason(
                    prefix_text,
                    distinct_ratio=prefix_distinct_ratio,
                    repeated_span_ratio=prefix_repeated_span_ratio,
                )
                relative_degenerate = None
            else:
                prefix_text = "".join(
                    position.token
                    for position in positions[: reference.natural_stop_tokens]
                )
                prefix_distinct_ratio = distinct_ngram_ratio(prefix_text)
                prefix_repeated_span_ratio = longest_repeated_substring_ratio(
                    prefix_text
                )
                this_degenerate = degeneracy_reason(
                    prefix_text,
                    distinct_ratio=prefix_distinct_ratio,
                    repeated_span_ratio=prefix_repeated_span_ratio,
                )
                relative_degenerate = relative_degeneracy_reason(
                    captured.output_text,
                    baseline_distinct_ratio=reference.full_distinct_ngram_ratio,
                    baseline_repeated_span_ratio=reference.full_repeated_span_ratio,
                    distinct_ratio=distinct_ratio,
                    repeated_span_ratio=repeated_span_ratio,
                )
                if this_degenerate is None:
                    this_degenerate = relative_degenerate
            if this_degenerate and degenerate is None:
                degenerate = f"{captured.request_id}: {this_degenerate}"
            ef.write(
                json.dumps(
                    {
                        "request_id": captured.request_id,
                        "streamed_tokens": captured.completion_tokens,
                        "span_positions": span,
                        "scored_positions": len(scored),
                        "mean_logprob": (sum(scored) / len(scored) if scored else None),
                        "min_logprob": min(scored) if scored else None,
                        "natural_stop_tokens": (
                            None if reference is None else reference.natural_stop_tokens
                        ),
                        "prefix_chars": len(prefix_text),
                        "prefix_distinct_ngram_ratio": prefix_distinct_ratio,
                        "prefix_longest_repeated_substring_ratio": (
                            prefix_repeated_span_ratio
                        ),
                        "distinct_ngram_ratio": distinct_ratio,
                        "longest_repeated_substring_ratio": repeated_span_ratio,
                        "baseline_distinct_ngram_ratio": (
                            None
                            if reference is None
                            else reference.full_distinct_ngram_ratio
                        ),
                        "baseline_longest_repeated_substring_ratio": (
                            None
                            if reference is None
                            else reference.full_repeated_span_ratio
                        ),
                        "relative_degenerate": relative_degenerate,
                        "degenerate": this_degenerate,
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
            quantile_logprob=0.0,
            coverage_ratio=0.0,
            evidence=rel_evidence,
            reason=f"engine returned no output for {len(empty)} prompt(s): {empty[0]}",
        )

    if not logprobs:
        if baseline_empty_matches and len(baseline_empty_matches) == len(outputs):
            return CorrectnessReport(
                verdict="pass",
                num_prompts=len(outputs),
                num_positions_scored=0,
                mean_logprob=0.0,
                min_logprob=0.0,
                quantile_logprob=0.0,
                coverage_ratio=1.0,
                evidence=rel_evidence,
            )
        return CorrectnessReport(
            verdict="infra_failed",
            num_prompts=len(outputs),
            num_positions_scored=0,
            mean_logprob=0.0,
            min_logprob=0.0,
            quantile_logprob=0.0,
            coverage_ratio=0.0,
            evidence=rel_evidence,
            reason="scorer produced no logprobs for any captured output",
        )

    # Coverage is how much of what the scorer saw it managed to score. Both
    # sides of the ratio come from the scorer, never from the candidate's own
    # report of itself, so a candidate cannot move this number.
    coverage = (len(logprobs) / span_positions) if span_positions > 0 else 0.0

    mean_lp = statistics.fmean(logprobs)
    min_lp = min(logprobs)
    # The bar is applied to the k-th lowest position, so one high-entropy
    # position the scorer and the candidate disagreed on cannot fail an
    # otherwise faithful output. See the module docstring.
    q_lp = quantile_low(logprobs, thr.min_token_quantile)
    reason: str | None = None
    verdict = "pass"
    if mean_lp < thr.min_mean_logprob:
        reason = f"mean logprob {mean_lp:.3f} below {thr.min_mean_logprob}"
    elif q_lp < thr.min_token_logprob:
        reason = (
            f"token logprob at quantile {thr.min_token_quantile} is "
            f"{q_lp:.3f}, below {thr.min_token_logprob}"
        )
    elif degenerate is not None:
        # A loop clears both bars above by scoring near zero.
        reason = f"degenerate output ({degenerate})"
    elif (
        thr.max_mean_logprob_drop is not None
        and baseline_mean_logprob is not None
        and mean_lp < baseline_mean_logprob - thr.max_mean_logprob_drop
    ):
        reason = (
            f"mean logprob {mean_lp:.3f} is more than "
            f"{thr.max_mean_logprob_drop} below the baseline's "
            f"{baseline_mean_logprob:.3f}"
        )

    if reason:
        # Whatever the scorer did manage to read was bad. Thin coverage is no
        # defence: judging it on the evidence there is beats handing back a
        # requeue for output already known to be wrong.
        verdict = "fail_correctness"
    elif coverage < thr.min_coverage_ratio:
        # It read too little to call this good. Nothing here says the
        # candidate is wrong, so the entry is requeued rather than failed.
        verdict = "infra_failed"
        reason = (
            f"scorer scored {len(logprobs)}/{span_positions} positions it saw, "
            f"below the {thr.min_coverage_ratio:.0%} floor"
        )
    logger.info(
        "correctness %s positions=%d mean=%.4f min=%.4f q=%.4f coverage=%.3f",
        verdict,
        len(logprobs),
        mean_lp,
        min_lp,
        q_lp,
        coverage,
    )
    return CorrectnessReport(
        verdict=verdict,
        num_prompts=len(outputs),
        num_positions_scored=len(logprobs),
        mean_logprob=mean_lp,
        min_logprob=min_lp,
        quantile_logprob=q_lp,
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
    baseline_outputs: Mapping[str, str] | None = None,
    request_timeout_s: float = 300.0,
    baseline_degeneracy: Mapping[str, BaselineDegeneracyReference] | None = None,
) -> dict[int, CorrectnessReport]:
    """Grade everything queued against one already-running scorer.

    The caller starts the scorer, calls this, and stops it, so correctness
    costs one engine start per round however many candidates the round holds.

    ``baseline_outputs`` comes from the opening baseline replay and supplies
    the parity exception for empty decoded continuations.

    The baseline is graded first when it is queued, and its mean logprob
    becomes the reference every candidate's relative bar is measured against.
    A baseline that does not grade cleanly leaves that reference unset, and
    candidates are then graded on the absolute bars alone; the caller decides
    what a failed baseline means for the round.
    """
    probe_resp = probe_logprob_capability(scorer_url, timeout=request_timeout_s)
    write_completion_response_shape(evidence_dir, probe_resp)
    reports: dict[int, CorrectnessReport] = {}
    ordered = [p for p in pending if p.candidate_index == BASELINE_INDEX] + [
        p for p in pending if p.candidate_index != BASELINE_INDEX
    ]
    baseline_mean: float | None = None
    for item in ordered:
        is_baseline = item.candidate_index == BASELINE_INDEX
        evidence_rel = f"evidence/correctness/{item.evidence_name}.jsonl"
        try:
            report = grade_candidate(
                scorer_url,
                item.outputs,
                cfg=cfg,
                evidence_path=evidence_dir / f"{item.evidence_name}.jsonl",
                baseline_outputs=baseline_outputs,
                request_timeout_s=request_timeout_s,
                baseline_mean_logprob=None if is_baseline else baseline_mean,
                baseline_degeneracy=baseline_degeneracy,
            )
        except EngineError as exc:
            # The text being forced through the scorer is whatever an engine
            # chose to emit, so grading one can fail on input nobody else in
            # the round sent. That is this entry's outcome, not the round's:
            # keep going and grade the others.
            logger.warning("scoring %s failed: %s", item.evidence_name, exc)
            report = CorrectnessReport(
                verdict="infra_failed",
                num_prompts=len(item.outputs),
                num_positions_scored=0,
                mean_logprob=0.0,
                min_logprob=0.0,
                quantile_logprob=0.0,
                coverage_ratio=0.0,
                evidence=evidence_rel,
                reason=f"scorer could not grade this output: {exc}",
            )
        if is_baseline and report.verdict == "pass":
            baseline_mean = report.mean_logprob
        reports[item.candidate_index] = report
    return reports
