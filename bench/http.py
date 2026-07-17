"""Stdlib-only OpenAI-compatible /v1/completions clients.

Shared by Modules A/B/C. Non-streaming ``post_completion`` returns the full
decoded response; ``post_completion_stream`` parses server-sent events and
captures TTFT/ITL timings for SLA measurement. All failures raise EngineError
so the CLI maps them to exit code 3.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bench.lifecycle import EngineError


def post_completion(
    base_url: str,
    *,
    prompt: str,
    max_tokens: int = 16,
    echo: bool = False,
    logprobs: int | None = 1,
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int | None = 0,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Serial non-streaming /v1/completions client (stdlib only)."""
    url = base_url.rstrip("/") + "/v1/completions"
    body: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "echo": echo,
        "temperature": temperature,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    if top_p is not None:
        body["top_p"] = top_p
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


@dataclass
class StreamResult:
    """Aggregated result of one streaming completion."""

    text: str
    finish_reason: str | None
    completion_tokens: int | None
    ttft_s: float
    itl_s: list[float] = field(default_factory=list)
    e2e_s: float = 0.0


def post_completion_stream(
    base_url: str,
    *,
    prompt: str,
    max_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float | None = None,
    seed: int | None = 0,
    timeout: float = 120.0,
) -> StreamResult:
    """Streaming /v1/completions client; parses SSE and times TTFT/ITL."""
    url = base_url.rstrip("/") + "/v1/completions"
    body: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # OpenAI/vLLM omit usage on stream chunks unless asked.
        "stream_options": {"include_usage": True},
    }
    if top_p is not None:
        body["top_p"] = top_p
    if seed is not None:
        body["seed"] = seed
    data = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    send = time.monotonic()
    try:
        resp = urlopen(req, timeout=timeout)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise EngineError(f"completions HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise EngineError(f"completions request failed for {url}: {exc}") from exc
    except TimeoutError as exc:
        raise EngineError(f"completions timed out for {url}") from exc

    text_parts: list[str] = []
    itl_s: list[float] = []
    finish_reason: str | None = None
    completion_tokens: int | None = None
    ttft_s = 0.0
    last_chunk: float | None = None
    saw_done = False

    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError as exc:
                raise EngineError(
                    f"malformed SSE chunk from {url}: {data_str[:120]!r}"
                ) from exc
            if not isinstance(chunk, dict):
                raise EngineError(
                    f"malformed SSE chunk from {url}: expected object, "
                    f"got {type(chunk).__name__}"
                )
            usage = chunk.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = chunk.get("choices")
            # vLLM/OpenAI may emit a final usage-only chunk with choices=[].
            if not choices:
                continue
            try:
                choice = choices[0]
                delta = str(choice.get("text", ""))
                fr = choice.get("finish_reason")
            except (IndexError, TypeError, AttributeError) as exc:
                raise EngineError(f"malformed SSE chunk from {url}: {exc}") from exc
            now = time.monotonic()
            if last_chunk is None:
                ttft_s = now - send
            else:
                itl_s.append(now - last_chunk)
            last_chunk = now
            text_parts.append(delta)
            if fr is not None:
                finish_reason = str(fr)
    except TimeoutError as exc:
        raise EngineError(f"completions stream timed out for {url}") from exc
    finally:
        resp.close()

    if not saw_done:
        raise EngineError(f"completions stream from {url} ended without [DONE]")
    if last_chunk is None:
        raise EngineError(
            f"completions stream from {url} produced no choice chunks "
            f"(cannot measure TTFT/ITL)"
        )
    # include_usage was requested; without a count the coalesced-stream and
    # Module C multi-token ITL guards cannot run and fail open.
    if completion_tokens is None:
        raise EngineError(
            f"completions stream from {url} omitted usage.completion_tokens"
        )
    if completion_tokens > 1 and not itl_s:
        raise EngineError(
            f"completions stream from {url}: completion_tokens={completion_tokens} "
            f"but no inter-token gaps (coalesced stream)"
        )
    e2e_s = last_chunk - send
    return StreamResult(
        text="".join(text_parts),
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
        ttft_s=ttft_s,
        itl_s=itl_s,
        e2e_s=e2e_s,
    )
