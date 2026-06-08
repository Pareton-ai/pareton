"""Docker container lifecycle, HTTP client, and evaluation orchestration.

Shells out to the ``docker`` CLI for container management and uses
stdlib ``urllib`` / ``http`` for talking to miner and baseline servers.
No ``docker`` Python SDK, no ``requests`` -- keeps the dependency surface
at zero beyond the system Python.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .baseline import BaselineCache, BaselinePromptResult, derive_cache_key
from .chain import CommitmentRecord
from .eval_schema import PerPromptResult, Prompt
from .scoring import (
    aligned_e2e_improvement,
    aligned_e2e_seconds,
    compute_pass1_aggregate_match,
    compute_speed_improvement,
    compute_teacher_forcing_verdict,
    compute_token_match_rate,
    pass1_match_passes,
)
from . import config as validator_config
from .state import EvaluationRecord

logger = logging.getLogger(__name__)

EVAL_N_WARMUP: int = 2
"""Warmup prompts discarded before scored eval (baseline + challengers)."""

EVAL_N_SCORED: int = 10
"""Scored eval prompts per miner after warmup discards."""


def _stream_decode_tps(
    target_token_count: int, decode_elapsed_secs: list[float]
) -> float:
    """Decode-window TPS from SSE timing (telemetry only; not used for scoring)."""
    if target_token_count < 2:
        return 0.0
    if len(decode_elapsed_secs) < target_token_count:
        return 0.0
    elapsed = decode_elapsed_secs[target_token_count - 1] - decode_elapsed_secs[0]
    if elapsed <= 0:
        return 0.0
    return target_token_count / elapsed


_BUFFERED_STREAM_TTFT_RATIO = 2.0
"""Log when miner TTFT exceeds baseline by this factor and decode window ~0."""


def _miner_e2e_s(
    baseline_n: int,
    ttft_s: float,
    decode_elapsed_secs: list[float],
    output_tokens: int,
) -> float:
    k = min(baseline_n, output_tokens)
    e2e = aligned_e2e_seconds(ttft_s, decode_elapsed_secs, k)
    return e2e if e2e is not None else 0.0


def _log_buffered_stream_if_suspected(
    uid: int,
    prompt_index: int,
    *,
    bl_ttft: float,
    mn_ttft: float,
    mn_decode: list[float],
    aligned_k: int,
) -> None:
    if aligned_k < 2 or len(mn_decode) < aligned_k:
        return
    if bl_ttft <= 0 or mn_ttft <= bl_ttft * _BUFFERED_STREAM_TTFT_RATIO:
        return
    decode_window = mn_decode[aligned_k - 1] - mn_decode[0]
    if decode_window > 0.001:
        return
    logger.info(
        "UID %d prompt %d: buffered stream suspected "
        "(ttft=%.3fs vs baseline %.3fs, decode_window=%.6fs, k=%d)",
        uid,
        prompt_index,
        mn_ttft,
        bl_ttft,
        decode_window,
        aligned_k,
    )


# --------------------------------------------------------------------------- #
# Internal data type
# --------------------------------------------------------------------------- #


@dataclass
class RawPromptResult:
    """Pre-scoring data from one prompt against one server."""

    prompt_index: int
    output_text: str
    tokens: list[str]
    top_logprobs: list[list[dict[str, Any]]] | None
    ttft_s: float
    throughput_tps: float
    output_tokens: int
    decode_elapsed_secs: list[float] | None = None
    error: str | None = None


@dataclass(frozen=True)
class TeacherForcingScore:
    """Pass 2 teacher-forcing result from the scoring vLLM."""

    logprobs: list[float]
    scoring_canonical_token_count: int


# --------------------------------------------------------------------------- #
# Docker lifecycle
# --------------------------------------------------------------------------- #


INTERNAL_NETWORK = "cacheon-internal"


def _ensure_network(name: str, *, internal: bool = False) -> None:
    """Create a Docker network if it doesn't already exist."""
    result = subprocess.run(
        ["docker", "network", "inspect", name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    cmd = ["docker", "network", "create"]
    if internal:
        cmd.append("--internal")
    cmd.append(name)
    logger.info("Creating Docker network: %s (internal=%s)", name, internal)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create Docker network {name}: {result.stderr.strip()}"
        )


def ensure_eval_network() -> None:
    """Create the internal eval network (no internet, no egress)."""
    _ensure_network(INTERNAL_NETWORK, internal=True)


def pull_image(image: str, digest: str, timeout_s: float = 600) -> None:
    """Pull a Docker image by digest. Raises on failure."""
    ref = f"{image}@{digest}" if digest else image
    logger.info("Pulling image %s", ref)
    result = subprocess.run(
        ["docker", "pull", ref],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker pull failed (rc={result.returncode}): {result.stderr.strip()}"
        )


def remove_image(image: str, digest: str, timeout_s: float = 60) -> None:
    """Remove a pulled miner image. Best-effort; never raises."""
    ref = f"{image}@{digest}" if digest else image
    try:
        result = subprocess.run(
            ["docker", "rmi", ref],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0:
            logger.info("Removed miner image %s", ref)
        else:
            logger.warning(
                "docker rmi %s failed (rc=%d): %s",
                ref,
                result.returncode,
                result.stderr.strip(),
            )
    except Exception as exc:
        logger.warning("docker rmi %s failed: %s", ref, exc)


def start_container(
    image: str,
    digest: str,
    *,
    model_volume: str,
    container_port: int = 8000,
    memory: str = "200g",
    cpus: int = 32,
    shm_size: str = "16g",
    cmd_args: list[str] | None = None,
    container_name: str | None = None,
    extra_env: dict[str, str] | None = None,
    compile_cache_volume: str | None = None,
) -> tuple[str, str]:
    """Start an isolated container and return ``(container_id, base_url)``.

    The container is placed on the internal Docker network.  The
    validator (also on the same network) reaches it via its container
    IP; no host port publishing is needed.

    ``cmd_args`` are appended after the image reference and become the
    container CMD (e.g. ``["--model", "/models"]`` for vLLM).  Miner
    images define their own entrypoint so this is typically only used
    for the baseline.
    """
    ensure_eval_network()
    ref = f"{image}@{digest}" if digest else image

    if container_name:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=15,
        )

    cmd = [
        "docker",
        "run",
        "-d",
        "--init",
        "--network",
        INTERNAL_NETWORK,
        "-v",
        f"{model_volume}:/models:ro",
        "--shm-size",
        shm_size,
        "--pids-limit",
        "4096",
        "--memory",
        memory,
        "--cpus",
        str(cpus),
        "--gpus",
        "all",
        "--device",
        "nvidia.com/gpu=all",
    ]
    if compile_cache_volume:
        cmd.extend(["-v", f"{compile_cache_volume}:/root/.cache/vllm"])
    if extra_env:
        for k, v in extra_env.items():
            cmd.extend(["-e", f"{k}={v}"])
    if container_name:
        cmd.extend(["--name", container_name])
    cmd.append(ref)
    if cmd_args:
        cmd.extend(cmd_args)
    logger.info("Starting container: image=%s", ref)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker run failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    container_id = result.stdout.strip()
    try:
        ip = _get_container_ip(container_id)
    except Exception:
        stop_and_remove(container_id)
        reset_gpu_state()
        raise
    base_url = f"http://{ip}:{container_port}"
    logger.info("🐳 Container started: %s url=%s", container_id[:12], base_url)
    return container_id, base_url


def _get_container_ip(container_id: str) -> str:
    """Return the container's IP address on the internal eval network."""
    template = (
        '{{index .NetworkSettings.Networks "' + INTERNAL_NETWORK + '" "IPAddress"}}'
    )
    result = subprocess.run(
        ["docker", "inspect", "-f", template, container_id],
        capture_output=True,
        text=True,
        timeout=10,
    )
    ip = result.stdout.strip()
    if result.returncode != 0 or not ip:
        raise RuntimeError(
            f"Could not get IP for container {container_id[:12]} "
            f"on network {INTERNAL_NETWORK}: {result.stderr.strip()}"
        )
    return ip


def capture_container_logs(
    container_name_or_id: str,
    state_dir: str | Path,
    label: str,
) -> None:
    """Save ``docker logs`` output to ``state_dir/container_logs/{label}.log``.

    Best-effort: never raises. Called before ``stop_and_remove`` so the
    container still exists.
    """
    try:
        result = subprocess.run(
            ["docker", "logs", container_name_or_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        if not output.strip():
            return
        log_dir = Path(state_dir) / "container_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{label}.log"
        log_path.write_text(output, encoding="utf-8")
        logger.info("Container logs saved: %s (%d chars)", log_path, len(output))
    except Exception as exc:
        logger.warning("Failed to capture container logs for %s: %s", label, exc)


def stop_and_remove(container_id: str) -> None:
    """Stop and remove a container. Best-effort, never raises."""
    for action in ("stop", "rm"):
        try:
            subprocess.run(
                ["docker", action, container_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("docker %s %s failed: %s", action, container_id[:12], exc)


def reset_gpu_state() -> None:
    """Attempt to reset GPU state between evaluations. Best-effort."""
    try:
        subprocess.run(
            ["nvidia-smi", "--gpu-reset"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        logger.debug("nvidia-smi --gpu-reset failed (non-fatal): %s", exc)


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #


def _container_status(container_id: str) -> str | None:
    """Return the container's status string ('running', 'exited', etc.).

    Returns None if the inspect call fails (container removed, Docker
    unavailable, etc.).
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _container_exit_code(container_id: str) -> int | None:
    """Return the container's exit code, or None if unavailable."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_id],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def wait_for_health(
    base_url: str,
    timeout_s: float = 600,
    poll_interval_s: float = 5,
    container_id: str | None = None,
) -> None:
    """Poll GET /health until 200. Raises TimeoutError on expiry.

    If *container_id* is provided, checks whether the container is still
    running on every iteration and bails immediately when it has exited.
    """
    url = f"{base_url}/health"
    deadline = time.monotonic() + timeout_s
    last_err: str = ""
    while time.monotonic() < deadline:
        if container_id:
            status = _container_status(container_id)
            if status and status != "running":
                exit_code = _container_exit_code(container_id)
                raise RuntimeError(
                    f"Container {container_id[:12]} exited "
                    f"(status={status}, exit_code={exit_code}) "
                    f"before /health became ready"
                )
        try:
            resp = urlopen(url, timeout=5)
            if resp.status == 200:
                logger.info("✅ Container healthy at %s", base_url)
                return
            last_err = f"status={resp.status}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"/health at {base_url} not ready after {timeout_s}s: {last_err}"
    )


def send_prompt(
    base_url: str,
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0,
    stream: bool = True,
    logprobs: bool = False,
    top_logprobs: int = 5,
    timeout_s: float = 120,
    prompt_index: int = 0,
) -> RawPromptResult:
    """Send a chat completion request and parse the response.

    Single-pass design: ``stream=True, logprobs=True`` measures TTFT and
    throughput while simultaneously collecting tokens + logprobs for
    correctness checking. The ``stream=False`` path is kept for testing
    but is not used in production scoring.
    """
    url = f"{base_url}/v1/chat/completions"
    body: dict[str, Any] = {
        "model": "Qwen2.5-72B-Instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs

    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})

    t_start = time.monotonic()
    try:
        resp = urlopen(req, timeout=timeout_s)
    except Exception as exc:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=0.0,
            throughput_tps=0.0,
            output_tokens=0,
            decode_elapsed_secs=[],
            error=f"request_failed: {exc}",
        )

    if stream:
        return _parse_sse_response(resp, t_start, prompt_index, max_tokens)
    else:
        return _parse_json_response(resp, t_start, prompt_index)


def _parse_sse_response(
    resp: Any, t_start: float, prompt_index: int, max_tokens: int
) -> RawPromptResult:
    """Parse a streaming SSE response, extracting speed metrics and
    correctness data in a single pass.

    Returns a ``RawPromptResult`` with TTFT, throughput, output tokens,
    and (when the server includes them) per-token logprobs.

    Token source: when logprobs are present in the SSE chunks, tokens
    are taken from ``logprobs.content[].token`` so each token and its
    logprobs stay paired by construction.

    ``challenger_output_text`` reconstruction (Pass 2 source of truth):
    primary = concatenate ``delta.content``; if logprobs were streamed
    but content deltas are empty, fall back to ``"".join(tokens)``.
    See docs/evaluation/scoring.
    """
    tokens: list[str] = []
    output_parts: list[str] = []
    all_top_logprobs: list[list[dict[str, Any]]] = []
    decode_elapsed_secs: list[float] = []
    t_first: float | None = None
    saw_logprobs = False
    capped = False

    def _record_token(token: str, top_lp: list[dict[str, Any]] | None = None) -> bool:
        """Append one completion token; return True if max_tokens reached."""
        nonlocal t_first
        now = time.monotonic()
        if t_first is None:
            t_first = now
        decode_elapsed_secs.append(now - t_first)
        tokens.append(token)
        if top_lp is not None:
            all_top_logprobs.append(top_lp)
        return len(tokens) >= max_tokens

    try:
        for raw_line in resp:
            if capped:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")

            lp_data = choice.get("logprobs") or {}
            lp_content = lp_data.get("content") or []
            if lp_content:
                saw_logprobs = True
                for entry in lp_content:
                    if not isinstance(entry, dict):
                        continue
                    if _record_token(
                        entry.get("token", ""),
                        entry.get("top_logprobs", []),
                    ):
                        capped = True
                        break

            if content:
                output_parts.append(content)
                if not saw_logprobs:
                    if _record_token(content):
                        capped = True

            if capped:
                break
    except Exception as exc:
        output_text = "".join(output_parts)
        if saw_logprobs and not output_text:
            output_text = "".join(tokens)
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text=output_text,
            tokens=tokens,
            top_logprobs=all_top_logprobs if saw_logprobs else None,
            ttft_s=(t_first - t_start) if t_first else 0.0,
            throughput_tps=0.0,
            output_tokens=len(tokens),
            decode_elapsed_secs=decode_elapsed_secs,
            error=f"stream_error: {exc}",
        )

    if t_first is None:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=0.0,
            throughput_tps=0.0,
            output_tokens=0,
            decode_elapsed_secs=[],
            error="no_tokens_in_stream",
        )

    ttft = t_first - t_start
    n_tokens = len(tokens)
    tps = _stream_decode_tps(n_tokens, decode_elapsed_secs)

    logprobs_out = all_top_logprobs if saw_logprobs else None

    output_text = "".join(output_parts)
    if saw_logprobs and not output_text:
        output_text = "".join(tokens)

    if logprobs_out is not None and len(logprobs_out) != n_tokens:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text=output_text,
            tokens=tokens,
            top_logprobs=logprobs_out,
            ttft_s=ttft,
            throughput_tps=tps,
            output_tokens=n_tokens,
            decode_elapsed_secs=decode_elapsed_secs,
            error=(
                f"logprob_token_mismatch: {len(logprobs_out)} logprob "
                f"entries vs {n_tokens} tokens"
            ),
        )

    return RawPromptResult(
        prompt_index=prompt_index,
        output_text=output_text,
        tokens=tokens,
        top_logprobs=logprobs_out,
        ttft_s=ttft,
        throughput_tps=tps,
        output_tokens=n_tokens,
        decode_elapsed_secs=decode_elapsed_secs,
    )


def _parse_json_response(
    resp: Any, t_start: float, prompt_index: int
) -> RawPromptResult:
    """Parse a non-streaming JSON response for correctness checking."""
    t_received = time.monotonic()
    try:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=0.0,
            throughput_tps=0.0,
            output_tokens=0,
            decode_elapsed_secs=[],
            error=f"json_parse_failed: {exc}",
        )

    choices = body.get("choices", [])
    if not choices:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=t_received - t_start,
            throughput_tps=0.0,
            output_tokens=0,
            decode_elapsed_secs=[],
            error="no_choices_in_response",
        )

    choice = choices[0]
    message = choice.get("message", {})
    output_text = message.get("content", "")

    try:
        lp_data = choice.get("logprobs") or {}
        lp_content = lp_data.get("content") or []
    except AttributeError:
        lp_data = {}
        lp_content = []

    tokens: list[str] = []
    all_top_logprobs: list[list[dict[str, Any]]] = []
    for entry in lp_content:
        if not isinstance(entry, dict):
            continue
        tokens.append(entry.get("token", ""))
        all_top_logprobs.append(entry.get("top_logprobs", []))

    ttft = t_received - t_start
    n_tokens = len(tokens)

    return RawPromptResult(
        prompt_index=prompt_index,
        output_text=output_text,
        tokens=tokens,
        top_logprobs=all_top_logprobs if lp_content else None,
        ttft_s=ttft,
        throughput_tps=0.0,
        output_tokens=n_tokens,
        decode_elapsed_secs=[],
    )


# --------------------------------------------------------------------------- #
# Teacher-forcing scoring pass
# --------------------------------------------------------------------------- #


def _tokenize(
    base_url: str, messages: list[dict[str, str]], timeout_s: float
) -> list[int]:
    """Return token IDs for a list of messages via vLLM's /tokenize endpoint."""
    body = {"model": "Qwen2.5-72B-Instruct", "messages": messages}
    req = Request(
        f"{base_url}/tokenize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urlopen(req, timeout=timeout_s)
    return json.loads(resp.read().decode("utf-8", errors="replace")).get("tokens", [])


def _trim_messages_to_fit(
    base_url: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    max_model_len: int,
    timeout_s: float,
    prompt_index: int = 0,
) -> list[dict[str, str]]:
    """Trim user content so tokenized input plus max_tokens fits max_model_len."""
    max_input = max_model_len - max_tokens
    if max_input <= 0:
        return messages

    try:
        input_tokens = len(_tokenize(base_url, messages, timeout_s))
    except Exception as exc:
        logger.warning("Prompt %d: /tokenize preflight failed: %s", prompt_index, exc)
        return messages

    if input_tokens <= max_input:
        return messages

    user_idx = max(
        (i for i, m in enumerate(messages) if m.get("role") == "user"),
        default=-1,
    )
    if user_idx < 0:
        return messages

    trimmed = [{**m} for m in messages]
    content = trimmed[user_idx]["content"]
    orig_len = len(content)
    lo, hi = 0, len(content)
    best = ""

    while lo <= hi:
        mid = (lo + hi) // 2
        trial = content[:mid]
        if mid < len(content):
            sp = trial.rfind(" ")
            if sp > mid // 2:
                trial = trial[:sp]
        trimmed[user_idx]["content"] = trial
        try:
            trial_tokens = len(_tokenize(base_url, trimmed, timeout_s))
        except Exception:
            hi = mid - 1
            continue
        if trial_tokens <= max_input:
            best = trial
            lo = mid + 1
        else:
            hi = mid - 1

    trimmed[user_idx]["content"] = best
    try:
        final_tokens = len(_tokenize(base_url, trimmed, timeout_s))
    except Exception:
        final_tokens = -1
    logger.warning(
        "Prompt %d: trimmed for max_model_len=%d (content %d->%d chars, "
        "input tokens %d->%d, max_output=%d)",
        prompt_index,
        max_model_len,
        orig_len,
        len(best),
        input_tokens,
        final_tokens,
        max_tokens,
    )
    return trimmed


_EOS_TOKENS = {"<|im_end|>", "</s>", "<|eot_id|>", "<|endoftext|>", "<eos>"}


def score_miner_output(
    base_url: str,
    messages: list[dict[str, str]],
    miner_output_text: str,
    timeout_s: float = 60,
) -> TeacherForcingScore:
    """Score challenger_output_text via scoring vLLM teacher-forcing.

    Uses exact ``miner_output_text`` bytes (no strip). Returns
    ``scoring_logprobs`` and ``scoring_canonical_token_count`` from the
    scoring vLLM /tokenize path only.
    """
    empty = TeacherForcingScore(logprobs=[], scoring_canonical_token_count=0)

    if not miner_output_text or not miner_output_text.strip():
        return empty

    scoring_messages = list(messages) + [
        {"role": "assistant", "content": miner_output_text}
    ]

    try:
        prefix_token_ids = _tokenize(base_url, messages, timeout_s)
    except Exception as exc:
        logger.warning("Scoring pass: /tokenize failed: %s", exc)
        return empty

    prefix_len = len(prefix_token_ids)
    if prefix_len == 0:
        logger.warning("Scoring pass: /tokenize returned empty prefix")
        return empty

    try:
        full_token_ids = _tokenize(base_url, scoring_messages, timeout_s)
    except Exception as exc:
        logger.warning("Scoring pass: /tokenize full messages failed: %s", exc)
        return empty

    scoring_canonical_token_count = len(full_token_ids) - prefix_len
    if scoring_canonical_token_count <= 0:
        logger.warning("Scoring pass: no assistant tokens in canonical tokenization")
        return empty

    logger.info(
        "Scoring pass: prefix_len=%d scoring_canonical_token_count=%d total_len=%d",
        prefix_len,
        scoring_canonical_token_count,
        len(full_token_ids),
    )

    url = f"{base_url}/v1/chat/completions"
    body: dict[str, Any] = {
        "model": "Qwen2.5-72B-Instruct",
        "messages": scoring_messages,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "prompt_logprobs": 1,
        "add_generation_prompt": False,
    }

    try:
        req = Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urlopen(req, timeout=timeout_s)
        result = json.loads(resp.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.warning("Scoring pass HTTP %d: %s", exc.code, err_body)
        return TeacherForcingScore([], scoring_canonical_token_count)
    except Exception as exc:
        logger.warning("Scoring pass failed: %s", exc)
        return TeacherForcingScore([], scoring_canonical_token_count)

    prompt_logprobs = result.get("prompt_logprobs") or []

    if not prompt_logprobs or not isinstance(prompt_logprobs, list):
        logger.warning("Scoring pass: prompt_logprobs missing from response")
        return TeacherForcingScore([], scoring_canonical_token_count)

    assistant_logprobs: list[float] = []
    for i in range(prefix_len, len(prompt_logprobs)):
        lp_entry = prompt_logprobs[i]
        if not lp_entry or not isinstance(lp_entry, dict):
            continue

        if len(lp_entry) == 1:
            token_data = next(iter(lp_entry.values()))
        else:
            non_top1 = [v for v in lp_entry.values() if v["rank"] != 1]
            token_data = non_top1[0] if non_top1 else next(iter(lp_entry.values()))

        if token_data["decoded_token"] in _EOS_TOKENS:
            break

        assistant_logprobs.append(float(token_data["logprob"]))

    if not assistant_logprobs:
        logger.warning("Scoring pass: no assistant token logprobs extracted")

    return TeacherForcingScore(
        logprobs=assistant_logprobs,
        scoring_canonical_token_count=scoring_canonical_token_count,
    )


def score_challenger_teacher_forcing(
    scoring_url: str,
    scored_prompts: list[Prompt],
    output_texts: list[str],
    miner_tokens_list: list[list[str]],
    *,
    log_prefix: str = "",
) -> tuple[bool, list[str], list[float]]:
    """Teacher-forcing correctness for Pass 2 prompts.

    Returns ``(passed, fail_reasons, per_prompt_mean_logprobs)``.
    Stops at the first failed prompt (same as prod gpu_eval).
    """
    n_prompts = min(
        len(output_texts),
        len(miner_tokens_list),
        len(scored_prompts),
    )
    fail_reasons: list[str] = []
    mean_logprobs: list[float] = []

    for i in range(n_prompts):
        msgs = [
            {"role": m.role, "content": m.content} for m in scored_prompts[i].messages
        ]
        challenger_stream_token_count = len(miner_tokens_list[i])
        label = f"{log_prefix} " if log_prefix else ""
        logger.info(
            "%sprompt %d: Pass 2 scoring start (challenger_stream_token_count=%d)",
            label,
            i,
            challenger_stream_token_count,
        )
        tf_score = score_miner_output(scoring_url, msgs, output_texts[i])
        scoring_canonical_token_count = tf_score.scoring_canonical_token_count
        scoring_logprob_count = len(tf_score.logprobs)
        scoring_scored_token_count = (
            min(scoring_canonical_token_count, scoring_logprob_count)
            if scoring_canonical_token_count > 0
            else scoring_logprob_count
        )

        if (
            scoring_canonical_token_count > 0
            and challenger_stream_token_count > 1.5 * scoring_canonical_token_count
        ):
            logger.warning(
                "%sprompt %d: stream_canonical_drift: "
                "challenger_stream_token_count=%d scoring_canonical_token_count=%d",
                log_prefix + " " if log_prefix else "",
                i,
                challenger_stream_token_count,
                scoring_canonical_token_count,
            )
        elif challenger_stream_token_count > scoring_canonical_token_count + 10:
            logger.warning(
                "%sprompt %d: possible_stream_spam: "
                "challenger_stream_token_count=%d scoring_canonical_token_count=%d",
                log_prefix + " " if log_prefix else "",
                i,
                challenger_stream_token_count,
                scoring_canonical_token_count,
            )

        verdict = compute_teacher_forcing_verdict(
            tf_score.logprobs,
            scoring_canonical_token_count=scoring_canonical_token_count,
            mean_logprob_threshold=validator_config.MEAN_LOGPROB_THRESHOLD,
            min_logprob_threshold=validator_config.MIN_LOGPROB_THRESHOLD,
        )
        mean_logprobs.append(verdict.mean_logprob)
        logger.info(
            "%sprompt %d: correctness=%s mean_lp=%.3f min_lp=%.3f "
            "challenger_stream_token_count=%d scoring_canonical_token_count=%d "
            "scoring_logprob_count=%d scoring_scored_token_count=%d",
            label,
            i,
            "PASS" if verdict.passed else "FAIL",
            verdict.mean_logprob,
            verdict.min_logprob,
            challenger_stream_token_count,
            scoring_canonical_token_count,
            scoring_logprob_count,
            scoring_scored_token_count,
        )
        if not verdict.passed:
            reason = verdict.reason or "unknown"
            if reason.startswith("scoring_infra_fail:"):
                fail_reasons.append(f"prompt {i}: {reason}")
            else:
                fail_reasons.append(f"prompt {i}: {reason}")
            return False, fail_reasons, mean_logprobs

    return True, fail_reasons, mean_logprobs


# --------------------------------------------------------------------------- #
# Orchestration helpers
# --------------------------------------------------------------------------- #


def _run_prompts_on_server(
    base_url: str,
    prompts: list[Prompt],
    *,
    stream: bool,
    logprobs: bool,
    per_prompt_timeout_s: int,
    n_warmup: int = 0,
    max_model_len: int | None = None,
) -> list[RawPromptResult]:
    """Send prompts to a running server, optionally discarding warmup results."""
    results: list[RawPromptResult] = []
    for i, prompt in enumerate(prompts):
        msgs = [{"role": m.role, "content": m.content} for m in prompt.messages]
        if max_model_len is not None:
            msgs = _trim_messages_to_fit(
                base_url,
                msgs,
                max_tokens=prompt.max_tokens,
                max_model_len=max_model_len,
                timeout_s=per_prompt_timeout_s,
                prompt_index=i,
            )
        r = send_prompt(
            base_url,
            msgs,
            max_tokens=prompt.max_tokens,
            stream=stream,
            logprobs=logprobs,
            timeout_s=per_prompt_timeout_s,
            prompt_index=i,
        )
        if i < n_warmup:
            logger.debug("Warmup prompt %d/%d discarded", i + 1, n_warmup)
            continue
        results.append(r)
    return results


def _detect_gpu_count() -> int:
    """Count GPUs via nvidia-smi. Returns 0 on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except Exception:
        pass
    return 0


def _read_max_position_embeddings(model_path: str) -> int | None:
    """Read effective max_position_embeddings from a local model's config.json.

    For YARN-scaled models (like Qwen2.5-72B), vLLM uses
    ``rope_scaling.original_max_position_embeddings`` as the validation floor
    rather than the full ``max_position_embeddings``. We return the minimum of
    both to match vLLM's behavior.
    """
    import json

    cfg_path = Path(model_path) / "config.json"
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        val = cfg.get("max_position_embeddings")
        if not isinstance(val, (int, float)) or val <= 0:
            logger.warning(
                "config.json at %s missing valid max_position_embeddings", cfg_path
            )
            return None

        effective = int(val)

        rope = cfg.get("rope_scaling") or {}
        original = rope.get("original_max_position_embeddings")
        if isinstance(original, (int, float)) and original > 0:
            effective = min(effective, int(original))
            logger.info(
                "YARN rope_scaling detected: original_max_position_embeddings=%d, "
                "effective max=%d",
                int(original),
                effective,
            )

        return effective
    except Exception as exc:
        logger.warning("Could not read %s: %s", cfg_path, exc)
    return None


def _max_model_len(gpu_count: int, model_path: str = "") -> int:
    """Choose vLLM max_model_len based on GPU count, capped by the model.

    More GPUs = more memory after TP-sharding the 72B model = room for
    longer KV caches.  The heuristic is then capped by the model's
    effective ``max_position_embeddings`` (from config.json) so vLLM never
    refuses to start. For YARN-scaled models, we also consider
    ``rope_scaling.original_max_position_embeddings`` which vLLM uses as
    its validation floor.
    """
    if gpu_count >= 8:
        heuristic = 131_072
    elif gpu_count >= 4:
        heuristic = 65_536
    else:
        heuristic = 32_768

    if model_path:
        model_max = _read_max_position_embeddings(model_path)
        if model_max is not None and model_max < heuristic:
            heuristic_k = int(round(heuristic / 1000))
            model_max_k = int(round(model_max / 1000))
            logger.info(
                "Capping max_model_len from %dk to %dk (model max_position_embeddings)",
                heuristic_k,
                model_max_k,
            )

            return model_max
        if model_max is None:
            logger.warning(
                "Could not read max_position_embeddings from %s; falling back to 32768",
                model_path,
            )
            return 32_768

    return heuristic


def _baseline_cmd_args(gpu_count: int) -> list[str]:
    """Build the vLLM server command args for the baseline container."""
    mml = _max_model_len(gpu_count, model_path=validator_config.MODEL_PATH)
    args = [
        "--model",
        "/models",
        "--served-model-name",
        "Qwen2.5-72B-Instruct",
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(mml),
    ]
    if gpu_count > 1:
        args.extend(["--tensor-parallel-size", str(gpu_count)])
    return args


def _scoring_baseline_cmd_args(gpu_count: int) -> list[str]:
    """vLLM args for the teacher-forcing scoring baseline."""
    from .prompts import SCORING_MAX_MODEL_LEN

    mml = SCORING_MAX_MODEL_LEN
    args = [
        "--model",
        "/models",
        "--served-model-name",
        "Qwen2.5-72B-Instruct",
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(mml),
        "--gpu-memory-utilization",
        "0.8",
        "--enforce-eager",
        "--disable-custom-all-reduce",
    ]
    if gpu_count > 1:
        args.extend(["--tensor-parallel-size", str(gpu_count)])
    return args


def run_baseline(
    prompts: list[Prompt],
    *,
    baseline_image: str,
    baseline_digest: str,
    model_volume: str,
    gpu_count: int,
    block_hash: str,
    evaluation_block: int,
    state_dir: str | Path = "",
    startup_timeout_s: int = 1200,
    per_prompt_timeout_s: int = 120,
    n_warmup: int = 2,
) -> BaselineCache:
    """Run the vLLM baseline container once and return measurements."""
    cache_key = derive_cache_key(block_hash, baseline_digest)
    logger.info("Running eval baseline (key=%s)", cache_key)

    baseline_args = _baseline_cmd_args(gpu_count)
    logger.info("Baseline cmd args: %s", baseline_args)

    container_name = "cacheon-baseline"
    cid: str | None = None
    pull_image(baseline_image, baseline_digest)
    try:
        cid, base_url = start_container(
            baseline_image,
            baseline_digest,
            model_volume=model_volume,
            cmd_args=baseline_args,
            container_name=container_name,
            compile_cache_volume=validator_config.baseline_compile_cache_dir(),
        )
        wait_for_health(base_url, timeout_s=startup_timeout_s, container_id=cid)

        mml = _max_model_len(gpu_count, model_path=validator_config.MODEL_PATH)
        results = _run_prompts_on_server(
            base_url,
            prompts,
            stream=True,
            logprobs=True,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=n_warmup,
            max_model_len=mml,
        )
    finally:
        if state_dir:
            cid_short = (cid or container_name)[:12]
            log_label = f"baseline_{evaluation_block}_{cid_short}"
            capture_container_logs(cid or container_name, state_dir, log_label)
        stop_and_remove(cid or container_name)
        reset_gpu_state()

    errors = [r for r in results if r.error]
    if errors:
        err_msg = "; ".join(f"prompt {r.prompt_index}: {r.error}" for r in errors)
        raise RuntimeError(f"Baseline had prompt errors: {err_msg}")

    baseline_results: list[BaselinePromptResult] = []
    for r in results:
        baseline_results.append(
            BaselinePromptResult(
                tokens=r.tokens,
                top_logprobs=r.top_logprobs or [],
                ttft_s=r.ttft_s,
                throughput_tps=r.throughput_tps,
                output_tokens=r.output_tokens,
                decode_elapsed_secs=r.decode_elapsed_secs or [],
            )
        )

    logger.info(
        "✅ Baseline complete: key=%s, %d prompts", cache_key, len(baseline_results)
    )
    return BaselineCache(cache_key=cache_key, results=baseline_results)


def start_baseline_for_scoring(
    *,
    model_volume: str,
    gpu_count: int,
    state_dir: str | Path = "",
    startup_timeout_s: int = 900,
) -> tuple[str, str]:
    """Start the baseline container for the teacher-forcing scoring pass.

    Uses the dedicated SCORING_IMAGE (default v0.9.2) for prompt_logprobs
    on Pass 2 correctness prompts (~4k context).

    Returns ``(container_id, base_url)``. Caller is responsible for
    calling ``stop_and_remove`` when done.
    """
    scoring_image = validator_config.SCORING_IMAGE
    scoring_args = _scoring_baseline_cmd_args(gpu_count)
    logger.info("Scoring baseline cmd args: %s", scoring_args)
    container_name = "cacheon-baseline-scoring"
    pull_image(scoring_image, "")
    cid, base_url = start_container(
        scoring_image,
        "",
        model_volume=model_volume,
        cmd_args=scoring_args,
        container_name=container_name,
    )
    try:
        wait_for_health(base_url, timeout_s=startup_timeout_s, container_id=cid)
    except Exception:
        stop_scoring_baseline(cid, state_dir, log_label="baseline_scoring_startup_fail")
        raise
    return cid, base_url


def pause_scoring_baseline(container_id: str) -> None:
    """Stop scoring container without removing it (GPU handoff to miner eval)."""
    try:
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            text=True,
            timeout=60,
        )
        logger.info("Scoring baseline paused: %s", container_id[:12])
    except Exception as exc:
        logger.warning("docker stop %s failed (non-fatal): %s", container_id[:12], exc)


def resume_scoring_baseline(
    container_id: str,
    *,
    startup_timeout_s: int = 600,
) -> str:
    """Start a stopped scoring container and return its base URL.

    Deprecated: gpu_eval batches teacher-forcing at the end with a fresh container
    instead of pause/resume. Kept for tests.
    """
    result = subprocess.run(
        ["docker", "start", container_id],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker start failed for scoring container {container_id[:12]}: "
            f"{result.stderr.strip()}"
        )
    ip = _get_container_ip(container_id)
    base_url = f"http://{ip}:8000"
    wait_for_health(base_url, timeout_s=startup_timeout_s, container_id=container_id)
    logger.info("Scoring baseline resumed at %s", base_url)
    return base_url


def stop_scoring_baseline(
    container_id: str | None,
    state_dir: str | Path = "",
    *,
    log_label: str = "baseline_scoring",
) -> None:
    """Save docker logs, stop the scoring baseline container, reset GPUs."""
    name = container_id or "cacheon-baseline-scoring"
    if state_dir:
        capture_container_logs(name, state_dir, log_label)
    stop_and_remove(name)
    reset_gpu_state()


def evaluate_challenger(
    com: CommitmentRecord,
    eval_prompts: list[Prompt],
    baseline: BaselineCache,
    *,
    model_volume: str,
    startup_timeout_s: int,
    per_prompt_timeout_s: int,
    n_warmup: int,
    current_block: int,
    state_dir: str | Path = "",
    log_label: str = "",
    collected_tf_output_texts: list | None = None,
    collected_tf_miner_tokens: list | None = None,
    max_model_len: int | None = None,
    on_pulled: Callable[[], None] | None = None,
    fingerprint_registry: dict | None = None,
    fingerprint_result: list | None = None,
) -> EvaluationRecord:
    """Full lifecycle for one challenger. Returns an EvaluationRecord.

    Runs eval prompts in one miner container session. Speed score
    uses the same outputs that are teacher-forced for correctness. Aggregate
    token match vs baseline is a cheap pre-filter only; teacher-forcing is
    the authoritative correctness gate (deferred to the scoring container).

    If ``collected_tf_output_texts`` / ``collected_tf_miner_tokens`` are
    provided, scored (post-warmup) outputs are appended for deferred scoring.
    """
    container_name = f"cacheon-uid{com.uid}-{com.hotkey[:8]}"
    cid: str | None = None
    eval_results: list[RawPromptResult] = []
    eval_error: Exception | None = None

    try:
        pull_image(com.image, com.digest)
        if on_pulled is not None:
            on_pulled()
        cid, base_url = start_container(
            com.image,
            com.digest,
            model_volume=model_volume,
            container_name=container_name,
        )
        wait_for_health(base_url, timeout_s=startup_timeout_s, container_id=cid)

        from .content_fingerprint import check_and_register_fingerprint

        fp_result = check_and_register_fingerprint(
            fingerprint_registry,
            container_id=cid,
            com=com,
            state_dir=state_dir,
        )
        if fingerprint_result is not None:
            fingerprint_result.append(fp_result)
        if fp_result.dq_reason is not None:
            return _dq_record(com, current_block, fp_result.dq_reason)

        eval_results = _run_prompts_on_server(
            base_url,
            eval_prompts,
            stream=True,
            logprobs=True,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=n_warmup,
            max_model_len=max_model_len,
        )
    except Exception as exc:
        logger.error("❌ Challenger UID %d failed: %s", com.uid, exc)
        eval_error = exc
    finally:
        if state_dir:
            _label = log_label or f"uid{com.uid}_{com.hotkey[:8]}_{current_block}"
            capture_container_logs(cid or container_name, state_dir, _label)
        stop_and_remove(cid or container_name)
        reset_gpu_state()
        remove_image(com.image, com.digest)

    if eval_error is not None:
        return _dq_record(com, current_block, str(eval_error))

    errors = [r for r in eval_results if r.error]
    if errors:
        err_msg = "; ".join(f"prompt {r.prompt_index}: {r.error}" for r in errors)
        logger.warning("Challenger UID %d had prompt errors: %s", com.uid, err_msg)
        return _dq_record(com, current_block, f"prompt_errors: {err_msg}")

    if collected_tf_output_texts is not None:
        for r in eval_results:
            collected_tf_output_texts.append(r.output_text)
    if collected_tf_miner_tokens is not None:
        for r in eval_results:
            collected_tf_miner_tokens.append(r.tokens)

    baseline_tokens = [bl.tokens for bl in baseline.results]
    miner_stress_tokens = [r.tokens for r in eval_results]
    agg_match_rate = compute_pass1_aggregate_match(
        baseline_tokens,
        miner_stress_tokens,
    )

    if not pass1_match_passes(
        agg_match_rate, validator_config.PASS1_MATCH_DQ_THRESHOLD
    ):
        reason = (
            f"pass1_match_fail: aggregate match {agg_match_rate:.4f} "
            f"below threshold {validator_config.PASS1_MATCH_DQ_THRESHOLD}"
        )
        logger.warning("Challenger UID %d Pass 1 match DQ: %s", com.uid, reason)

        n_scored = min(len(eval_results), len(baseline.results))
        per_prompt: list[PerPromptResult] = []
        for i in range(n_scored):
            bl = baseline.results[i]
            r = eval_results[i]
            match_rate = compute_token_match_rate(bl.tokens, r.tokens)
            miner_decode = r.decode_elapsed_secs or []
            bl_decode = bl.decode_elapsed_secs or []
            _, bl_e2e, mn_e2e_opt = aligned_e2e_improvement(
                bl.ttft_s,
                bl_decode,
                bl.output_tokens,
                r.ttft_s,
                miner_decode,
                r.output_tokens,
            )
            mn_e2e = (
                mn_e2e_opt
                if mn_e2e_opt is not None
                else _miner_e2e_s(
                    bl.output_tokens, r.ttft_s, miner_decode, r.output_tokens
                )
            )
            per_prompt.append(
                PerPromptResult(
                    ttft_s=r.ttft_s,
                    e2e_s=mn_e2e,
                    output_tokens=r.output_tokens,
                    token_match_rate=match_rate,
                    baseline_e2e_s=bl_e2e if bl_e2e is not None else 0.0,
                )
            )
            logger.info(
                "  prompt %d: match=%.4f ttft=%.4fs e2e=%.4fs tokens=%d",
                i,
                match_rate,
                r.ttft_s,
                mn_e2e,
                r.output_tokens,
            )

        per_prompt_dicts = [pp.to_dict() for pp in per_prompt] if per_prompt else None
        return EvaluationRecord(
            uid=com.uid,
            hotkey=com.hotkey,
            commit_block=com.commit_block,
            image=com.image,
            digest=com.digest,
            score=0.0,
            speed_improvement=0.0,
            token_match_rate=agg_match_rate,
            disqualified=True,
            disqualify_reason=reason,
            evaluated_at=time.time(),
            evaluation_block=current_block,
            per_prompt=per_prompt_dicts,
        )

    per_prompt = []
    per_prompt_improvements: list[float] = []

    n_scored = min(len(eval_results), len(baseline.results))
    for i in range(n_scored):
        bl = baseline.results[i]
        r = eval_results[i]
        match_rate = compute_token_match_rate(bl.tokens, r.tokens)

        miner_decode = r.decode_elapsed_secs or []
        bl_decode = bl.decode_elapsed_secs or []
        improvement, bl_e2e, mn_e2e_opt = aligned_e2e_improvement(
            bl.ttft_s,
            bl_decode,
            bl.output_tokens,
            r.ttft_s,
            miner_decode,
            r.output_tokens,
        )
        aligned_k = min(bl.output_tokens, r.output_tokens)
        _log_buffered_stream_if_suspected(
            com.uid,
            i,
            bl_ttft=bl.ttft_s,
            mn_ttft=r.ttft_s,
            mn_decode=miner_decode,
            aligned_k=aligned_k,
        )
        if len(miner_decode) < bl.output_tokens:
            logger.debug(
                "Challenger UID %d prompt %d: no speed credit "
                "(miner emitted %d tokens, baseline %d)",
                com.uid,
                i,
                len(miner_decode),
                bl.output_tokens,
            )

        per_prompt_improvements.append(improvement)
        mn_e2e = (
            mn_e2e_opt
            if mn_e2e_opt is not None
            else _miner_e2e_s(bl.output_tokens, r.ttft_s, miner_decode, r.output_tokens)
        )

        per_prompt.append(
            PerPromptResult(
                ttft_s=r.ttft_s,
                e2e_s=mn_e2e,
                output_tokens=r.output_tokens,
                token_match_rate=match_rate,
                baseline_e2e_s=bl_e2e if bl_e2e is not None else 0.0,
            )
        )

    per_prompt_dicts = [pp.to_dict() for pp in per_prompt] if per_prompt else None

    speed_improvement = compute_speed_improvement(per_prompt_improvements)
    score = speed_improvement

    logger.info(
        "Challenger UID %d scored: score=%.4f speed_imp=%.4f "
        "match_rate=%.4f (%d eval prompts)",
        com.uid,
        score,
        speed_improvement,
        agg_match_rate,
        len(per_prompt),
    )
    for pp in per_prompt:
        logger.debug(
            "  prompt: ttft=%.4fs e2e=%.4fs tokens=%d match=%.4f",
            pp.ttft_s,
            pp.e2e_s,
            pp.output_tokens,
            pp.token_match_rate,
        )

    return EvaluationRecord(
        uid=com.uid,
        hotkey=com.hotkey,
        commit_block=com.commit_block,
        image=com.image,
        digest=com.digest,
        score=score,
        speed_improvement=speed_improvement,
        token_match_rate=agg_match_rate,
        disqualified=False,
        disqualify_reason=None,
        evaluated_at=time.time(),
        evaluation_block=current_block,
        per_prompt=per_prompt_dicts,
    )


def _dq_record(
    com: CommitmentRecord, current_block: int, reason: str
) -> EvaluationRecord:
    return EvaluationRecord(
        uid=com.uid,
        hotkey=com.hotkey,
        commit_block=com.commit_block,
        image=com.image,
        digest=com.digest,
        score=0.0,
        speed_improvement=0.0,
        token_match_rate=0.0,
        disqualified=True,
        disqualify_reason=reason,
        evaluated_at=time.time(),
        evaluation_block=current_block,
    )


# --------------------------------------------------------------------------- #
# EvalFn factory
# --------------------------------------------------------------------------- #


def make_eval_fn(
    *,
    model_volume: str,
    baseline_image: str,
    baseline_digest: str,
    gpu_count: int = 0,
    state_dir: str = "",
    startup_timeout_s: int = validator_config.CHALLENGER_STARTUP_TIMEOUT_S,
    per_prompt_timeout_s: int = 120,
    n_warmup: int = 2,
) -> Callable:
    """Return an ``EvalFn`` that runs Docker eval per challenger.

    For each challenger, runs the full Docker lifecycle sequentially.
    Baseline is run once per eval invocation and reused in memory.

    If ``gpu_count`` is positive it is used directly; otherwise
    ``nvidia-smi`` auto-detection is attempted on first eval.
    """
    resolved_gpu_count = gpu_count

    def eval_fn(
        challengers: list[CommitmentRecord],
        *,
        current_block: int,
        block_hash: str | None,
    ) -> list[EvaluationRecord]:
        nonlocal resolved_gpu_count
        if not block_hash:
            logger.error("No block_hash available -- cannot derive prompt seed")
            return [_dq_record(c, current_block, "no_block_hash") for c in challengers]

        if resolved_gpu_count <= 0:
            resolved_gpu_count = _detect_gpu_count()
            if resolved_gpu_count <= 0:
                logger.error("Could not detect GPU count via nvidia-smi")
                return [
                    _dq_record(c, current_block, "no_gpu_count") for c in challengers
                ]
            logger.info("Auto-detected %d GPU(s)", resolved_gpu_count)

        from .prompts import EVAL_CONTEXT_TOKENS, sample_eval_prompts

        eval_prompts = sample_eval_prompts(
            block_hash,
            n=EVAL_N_SCORED + n_warmup,
            max_context_tokens=EVAL_CONTEXT_TOKENS,
        )

        baseline = run_baseline(
            eval_prompts,
            baseline_image=baseline_image,
            baseline_digest=baseline_digest,
            model_volume=model_volume,
            gpu_count=resolved_gpu_count,
            block_hash=block_hash,
            evaluation_block=current_block,
            state_dir=state_dir,
            startup_timeout_s=startup_timeout_s,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=n_warmup,
        )

        results: list[EvaluationRecord] = []
        for com in challengers:
            logger.info(
                "⚔️  Evaluating challenger UID %d (%s) image=%s",
                com.uid,
                com.hotkey[:16],
                com.image,
            )
            record = evaluate_challenger(
                com,
                eval_prompts,
                baseline,
                model_volume=model_volume,
                startup_timeout_s=startup_timeout_s,
                per_prompt_timeout_s=per_prompt_timeout_s,
                n_warmup=n_warmup,
                current_block=current_block,
                state_dir=state_dir,
            )
            results.append(record)

        return results

    return eval_fn
