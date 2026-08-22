"""In-process OpenAI-compatible /v1/completions mock (vLLM-shaped).

Configurable per-token logprobs and per-token latencies so the harness can
test TTFT/ITL deterministically.

Two knobs let one mock round stand in for a real one, by giving each image in
the round its own config:

* ``speed_factor`` divides every per-token latency, so a round can hold a
  baseline at 1.0, challengers a few percent above it, and a clear leader.
* ``garbage`` makes an engine emit ``GARBAGE_TEXT``. Any engine asked to score
  that text assigns ``garbage_logprob`` to those tokens, so the shared scorer
  fails it on correctness the same way it would fail a real engine emitting
  nonsense. The emitting engine's own logprobs never come into it: the scorer
  judges the text, and a candidate is only ever asked for text.

The checked-in fixture at fixtures/bench/vllm_completion_response_shape.json
documents the expected response shape.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization (deterministic, mock-only — not real BPE)
# ---------------------------------------------------------------------------


def mock_tokenize(text: str) -> list[str]:
    """Split into whitespace-delimited tokens, keeping trailing spaces on prior tokens.

    Empty string → []. Leading whitespace becomes its own token so round-trips
    preserve length for echo scoring.
    """
    if not text:
        return []
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            if tokens:
                tokens[-1] = tokens[-1] + ch
            else:
                buf.append(ch)
        else:
            if buf and buf[0].isspace():
                tokens.append("".join(buf))
                buf = []
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def mock_detokenize(tokens: list[str]) -> str:
    return "".join(tokens)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# A token containing this marker is nonsense no engine would greedily emit.
GARBAGE_MARKER = "zqxj"
GARBAGE_TEXT = " zqxj zqxj zqxj"


@dataclass
class MockEngineConfig:
    model: str = "mock-model"
    """Base logprob for token i: base + i * slope."""
    logprob_base: float = -0.5
    logprob_slope: float = -0.01
    """Per-output-token latency in seconds (TTFT ≈ first, ITL ≈ rest)."""
    token_latency_s: float = 0.0
    """Divides every per-token latency: 2.0 serves twice as fast as 1.0."""
    speed_factor: float = 1.0
    """Emit GARBAGE_TEXT instead of greedy_text (scores below the correctness bar)."""
    garbage: bool = False
    """Logprob any engine assigns to a garbage token when scoring it."""
    garbage_logprob: float = -30.0
    """Optional explicit logprob list (cycled); overrides base/slope when set."""
    logprobs: list[float] | None = None
    """Optional per-token latency list in seconds (cycled)."""
    latencies_s: list[float] | None = None
    """Fixed greedy continuation when generating (not echoing)."""
    greedy_text: str = " OK"
    top_logprobs_k: int = 1
    host: str = "127.0.0.1"
    port: int = 0  # 0 = ephemeral

    def continuation_text(self) -> str:
        return GARBAGE_TEXT if self.garbage else self.greedy_text


# ---------------------------------------------------------------------------
# Logprob helpers
# ---------------------------------------------------------------------------


def _logprob_at(cfg: MockEngineConfig, index: int, token: str = "") -> float:
    """Logprob this engine assigns to ``token`` at ``index`` while scoring.

    Garbage is judged by its text, not by who produced it, which is what lets
    one shared scorer grade output captured from any number of candidates.
    """
    if GARBAGE_MARKER in token:
        return float(cfg.garbage_logprob)
    if cfg.logprobs is not None and cfg.logprobs:
        lp = cfg.logprobs[index % len(cfg.logprobs)]
    else:
        lp = cfg.logprob_base + index * cfg.logprob_slope
    return float(lp)


def _latency_at(cfg: MockEngineConfig, index: int) -> float:
    if cfg.latencies_s is not None and cfg.latencies_s:
        base = float(cfg.latencies_s[index % len(cfg.latencies_s)])
    else:
        base = float(cfg.token_latency_s)
    factor = float(cfg.speed_factor)
    return base / factor if factor > 0 else base


def build_logprobs_payload(
    tokens: list[str],
    *,
    cfg: MockEngineConfig,
    prompt_token_count: int,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Build vLLM-shaped logprobs object.

    Quirks mirrored:
    - first prompt token logprob is null
    - subsequent tokens have float logprobs
    - top_logprobs entries align 1:1 (null for first prompt token)
    - each non-null top_logprobs dict has exactly ``top_k`` entries
    """
    # Request count wins when provided; otherwise MockEngineConfig.top_logprobs_k.
    k = max(0, int(top_k) if top_k is not None else cfg.top_logprobs_k)
    token_logprobs: list[float | None] = []
    top_logprobs: list[dict[str, float] | None] = []
    text_offset: list[int] = []
    offset = 0
    for i, tok in enumerate(tokens):
        text_offset.append(offset)
        offset += len(tok)
        if i < prompt_token_count and i == 0:
            token_logprobs.append(None)
            top_logprobs.append(None)
            continue
        # For prompt tokens after the first, and all completion tokens, assign logprobs.
        # Index used for deterministic series is absolute position in the echoed sequence.
        lp = _logprob_at(cfg, i, tok)
        token_logprobs.append(lp)
        top: dict[str, float] = {}
        if k >= 1:
            top[tok] = lp
        for j in range(1, k):
            top[f"{tok}_{j}"] = lp - (2.0 * j)
        top_logprobs.append(top)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


def build_completion_response(
    *,
    cfg: MockEngineConfig,
    prompt: str,
    max_tokens: int,
    echo: bool,
    temperature: float,
    logprobs_requested: int | None,
) -> dict[str, Any]:
    prompt_tokens = mock_tokenize(prompt)

    if max_tokens <= 0:
        # Score-only / teacher-force path: no new tokens; echo prompt logprobs.
        completion_tokens: list[str] = []
    else:
        # Greedy continuation, cycled to fill max_tokens (long completions let
        # SLA tests measure throughput); teacher forcing uses max_tokens<=0 and
        # is unaffected. Truncate when max_tokens is smaller than the cycle.
        cont = mock_tokenize(cfg.continuation_text())
        if not cont:
            cont = [" OK"]
        completion_tokens = [cont[i % len(cont)] for i in range(max_tokens)]
        # Apply per-token latency (sleep) for completion tokens only.
        for i in range(len(completion_tokens)):
            delay = _latency_at(cfg, i)
            if delay > 0:
                time.sleep(delay)

    if echo:
        all_tokens = prompt_tokens + completion_tokens
        text = mock_detokenize(all_tokens)
        prompt_count = len(prompt_tokens)
    else:
        all_tokens = completion_tokens
        text = mock_detokenize(completion_tokens)
        prompt_count = 0  # no prompt tokens in logprobs when echo=false

    logprobs_obj = None
    if logprobs_requested is not None and logprobs_requested != 0:
        # Client `logprobs` count overrides config default.
        top_k = int(logprobs_requested)
        tokens_for_lp = all_tokens if echo else completion_tokens
        prompt_count_for_lp = prompt_count if echo else 0
        logprobs_obj = build_logprobs_payload(
            tokens_for_lp,
            cfg=cfg,
            prompt_token_count=prompt_count_for_lp,
            top_k=top_k,
        )

    if max_tokens <= 0:
        finish_reason = "length"
    elif len(completion_tokens) >= max_tokens:
        finish_reason = "length"
    else:
        finish_reason = "stop"

    created = int(time.time())
    resp_id = f"cmpl-mock-{uuid.uuid4().hex[:12]}"
    choice: dict[str, Any] = {
        "index": 0,
        "text": text,
        "logprobs": logprobs_obj,
        "finish_reason": finish_reason,
    }
    return {
        "id": resp_id,
        "object": "text_completion",
        "created": created,
        "model": cfg.model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(completion_tokens),
            "total_tokens": len(prompt_tokens) + len(completion_tokens),
        },
    }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server: "MockEngineServer"  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("mock_engine: " + fmt, *args)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8") or "{}")

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _write_sse(self, payload: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_completion(
        self, *, prompt: str, max_tokens: int, temperature: float
    ) -> None:
        """vLLM-shaped SSE: one chunk per completion token with per-token delay.

        First-chunk delay ⇒ TTFT; inter-chunk gaps ⇒ ITL. Final chunk carries
        finish_reason + usage; stream ends with ``data: [DONE]``.
        """
        cfg = self.server.cfg
        cont = mock_tokenize(cfg.continuation_text()) or [" OK"]
        n = max(1, max_tokens)
        completion_tokens = [cont[i % len(cont)] for i in range(n)]
        prompt_count = len(mock_tokenize(prompt))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        resp_id = f"cmpl-mock-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        try:
            for i, tok in enumerate(completion_tokens):
                delay = _latency_at(cfg, i)
                if delay > 0:
                    time.sleep(delay)
                last = i == len(completion_tokens) - 1
                chunk: dict[str, Any] = {
                    "id": resp_id,
                    "object": "text_completion",
                    "created": created,
                    "model": cfg.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": tok,
                            "logprobs": None,
                            "finish_reason": "length" if last else None,
                        }
                    ],
                }
                if last:
                    chunk["usage"] = {
                        "prompt_tokens": prompt_count,
                        "completion_tokens": len(completion_tokens),
                        "total_tokens": prompt_count + len(completion_tokens),
                    }
                self._write_sse(chunk)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("mock_engine: client disconnected mid-stream")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/v1/models"):
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.server.cfg.model, "object": "model"}],
                },
            )
            return
        self._write_json(404, {"error": {"message": f"not found: {path}"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/v1/completions":
            self._write_json(404, {"error": {"message": f"not found: {path}"}})
            return
        try:
            req = self._read_json()
        except json.JSONDecodeError as exc:
            self._write_json(400, {"error": {"message": f"invalid json: {exc}"}})
            return

        prompt = req.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        prompt = str(prompt)
        max_tokens = int(req.get("max_tokens", 16))
        echo = bool(req.get("echo", False))
        temperature = float(req.get("temperature", 1.0))
        stream = bool(req.get("stream", False))
        logprobs_raw = req.get("logprobs", None)
        logprobs_requested: int | None
        if logprobs_raw is None:
            logprobs_requested = None
        else:
            logprobs_requested = int(logprobs_raw)

        # Ignore unused fields (model, seed, top_p, ...) — accepted for compatibility.
        _ = req.get("model"), req.get("seed"), req.get("top_p"), temperature

        if stream:
            self._stream_completion(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature
            )
            return

        resp = build_completion_response(
            cfg=self.server.cfg,
            prompt=prompt,
            max_tokens=max_tokens,
            echo=echo,
            temperature=temperature,
            logprobs_requested=logprobs_requested,
        )
        self._write_json(200, resp)


class MockEngineServer(ThreadingHTTPServer):
    def __init__(self, cfg: MockEngineConfig) -> None:
        self.cfg = cfg
        super().__init__((cfg.host, cfg.port), _Handler)


@dataclass
class MockEngine:
    """Context-manager wrapper around the in-process mock server."""

    cfg: MockEngineConfig = field(default_factory=MockEngineConfig)
    _server: MockEngineServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("mock engine not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    def start(self) -> "MockEngine":
        if self._server is not None:
            return self
        self._server = MockEngineServer(self.cfg)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(
            "mock engine listening on %s (speed_factor=%s garbage=%s)",
            self.base_url,
            self.cfg.speed_factor,
            self.cfg.garbage,
        )
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None

    def __enter__(self) -> "MockEngine":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def post_completion(
    url: str,
    *,
    prompt: str,
    max_tokens: int = 16,
    echo: bool = False,
    logprobs: int = 1,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Tiny urllib client for tests (no third-party deps)."""
    import urllib.request

    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "echo": echo,
            "logprobs": logprobs,
            "temperature": temperature,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def response_shape_fingerprint(resp: dict[str, Any]) -> dict[str, Any]:
    """Structural fingerprint used to compare against the checked-in fixture."""

    def _type_of(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int) and not isinstance(v, bool):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
        if isinstance(v, list):
            if not v:
                return "empty_list"
            # List of dicts (no nulls) → fingerprint first element deeply.
            if all(isinstance(x, dict) for x in v):
                return [_type_of(v[0])]
            # Nullable dict slots (top_logprobs first-prompt null quirk).
            if any(x is None for x in v) and all(
                x is None or isinstance(x, dict) for x in v
            ):
                return ["dict|null"]
            # Nullable numeric slots (token_logprobs first-prompt null quirk).
            if any(x is None for x in v) and all(
                x is None or isinstance(x, (int, float)) for x in v
            ):
                return ["float|null"]
            kinds = {
                (
                    "dict"
                    if isinstance(x, dict)
                    else ("float" if isinstance(x, float) else _type_of(x))
                )
                for x in v
            }
            if len(kinds) == 1:
                return [kinds.pop()]
            return sorted(str(k) for k in kinds)
        if isinstance(v, dict):
            return {k: _type_of(val) for k, val in sorted(v.items())}
        return type(v).__name__

    return _type_of(resp)


# ---------------------------------------------------------------------------
# CLI (containerized / subprocess mock engine)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Serve the mock engine until SIGTERM/SIGINT. Exit 0 on clean shutdown."""
    import argparse
    import signal

    p = argparse.ArgumentParser(
        prog="python -m bench.mock_engine",
        description="In-process OpenAI-compatible mock engine for bench CI.",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    p.add_argument("--model", default="mock-model", help="Reported model id")
    p.add_argument(
        "--garbage",
        action="store_true",
        help="Emit nonsense output so the shared scorer fails it on correctness",
    )
    p.add_argument(
        "--token-latency-s",
        type=float,
        default=0.0,
        help="Per-output-token sleep in seconds",
    )
    p.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="Divide per-token latency by this (2.0 = twice the baseline speed)",
    )
    p.add_argument(
        "--startup-delay-s",
        type=float,
        default=0.0,
        help="Sleep before listening (tests health-check wait)",
    )
    # A real engine accepts many serve args this mock does not model, and the
    # harness appends the scorer flags to whatever the campaign pinned, so an
    # unknown flag is expected. Ignore it rather than refusing to start.
    args, unknown = p.parse_known_args(argv)
    if unknown:
        logger.info("mock engine ignoring unmodelled serve args: %s", unknown)

    if args.startup_delay_s > 0:
        time.sleep(args.startup_delay_s)

    cfg = MockEngineConfig(
        model=args.model,
        garbage=args.garbage,
        token_latency_s=args.token_latency_s,
        speed_factor=args.speed_factor,
        host=args.host,
        port=args.port,
    )
    engine = MockEngine(cfg)
    engine.start()

    stop = threading.Event()

    def _shutdown(*_args: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        stop.wait()
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
