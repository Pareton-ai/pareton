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

from .baseline import (
    BaselineCache,
    BaselinePromptResult,
    derive_cache_key,
    load_cached_baseline,
    save_baseline_cache,
)
from .chain import CommitmentRecord
from .eval_schema import PerPromptResult, Prompt
from .scoring import (
    compute_aligned_throughput_tps,
    compute_improvements,
    compute_pass1_aggregate_match,
    compute_teacher_forcing_verdict,
    compute_token_match_rate,
    pass1_match_passes,
)
from . import config as validator_config
from .state import EvaluationRecord

logger = logging.getLogger(__name__)

EVAL_N_WARMUP: int = 2
"""Warmup prompts discarded before scored eval (baseline + challengers)."""

EVAL_N_STRESS_SCORED: int = 10
"""Scored stress prompts per miner after warmup discards."""


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


def pull_image(image: str, digest: str, timeout_s: float = 300) -> None:
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
    logprobs stay paired by construction.  ``delta.content`` is used
    for timing and display text only.
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
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="".join(output_parts),
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
    tps = compute_aligned_throughput_tps(n_tokens, decode_elapsed_secs)

    logprobs_out = all_top_logprobs if saw_logprobs else None

    if logprobs_out is not None and len(logprobs_out) != n_tokens:
        return RawPromptResult(
            prompt_index=prompt_index,
            output_text="".join(output_parts),
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
        output_text="".join(output_parts),
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


def score_miner_output(
    base_url: str,
    messages: list[dict[str, str]],
    miner_output_text: str,
    timeout_s: float = 60,
) -> list[float]:
    """Score miner's output tokens using the baseline model (teacher-forcing).

    Uses vLLM's prompt_logprobs extension on /v1/chat/completions. The
    baseline processes the full prompt + assistant reply in one forward pass
    and returns per-token logprobs for every position. We slice to just the
    assistant output tokens by first tokenizing the prompt-only prefix to
    find where the assistant content starts.

    For each position in the assistant output, the actual token's logprob is
    identified by rank: with prompt_logprobs=1, each position contains the
    actual token plus the top-1 alternative (when different). The actual token
    is the one with rank != 1 when two entries exist, or the sole entry when
    it matches top-1. We stop at the first end-of-turn marker.

    Returns one logprob per assistant output token. Empty list on any error.
    """
    _EOS_TOKENS = {"<|im_end|>", "</s>", "<|eot_id|>", "<|endoftext|>", "<eos>"}

    if not miner_output_text or not miner_output_text.strip():
        return []

    stripped = miner_output_text.strip()
    scoring_messages = list(messages) + [{"role": "assistant", "content": stripped}]

    try:
        prefix_token_ids = _tokenize(base_url, messages, timeout_s)
    except Exception as exc:
        logger.warning("Scoring pass: /tokenize failed: %s", exc)
        return []

    prefix_len = len(prefix_token_ids)
    if prefix_len == 0:
        logger.warning("Scoring pass: /tokenize returned empty prefix")
        return []

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
        return []
    except Exception as exc:
        logger.warning("Scoring pass failed: %s", exc)
        return []

    prompt_logprobs = result.get("prompt_logprobs") or []

    if not prompt_logprobs or not isinstance(prompt_logprobs, list):
        logger.warning("Scoring pass: prompt_logprobs missing from response")
        return []

    # Extract the logprob of the actual token at each assistant output position.
    # prompt_logprobs[i] is None | dict{token_id_str: {logprob, rank, decoded_token}}.
    # With prompt_logprobs=1, each non-None entry contains:
    #   - always: the actual token at that position
    #   - additionally: the top-1 predicted token if it differs from actual
    # The actual token is identified by rank > 1 when 2 entries exist (the
    # top-1 alternative always has rank=1). When only 1 entry, actual == top-1.
    # We stop at the first end-of-turn marker to exclude chat template suffixes.
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

    return assistant_logprobs


def score_challenger_teacher_forcing(
    scoring_url: str,
    scored_prompts: list[Prompt],
    output_texts: list[str],
    miner_tokens_list: list[list[str]],
    *,
    log_prefix: str = "",
) -> tuple[bool, list[str], list[float]]:
    """Teacher-forcing correctness for Pass 2 audit prompts.

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
        scoring_logprobs = score_miner_output(scoring_url, msgs, output_texts[i])
        verdict = compute_teacher_forcing_verdict(
            miner_tokens_list[i],
            scoring_logprobs,
            mean_logprob_threshold=validator_config.MEAN_LOGPROB_THRESHOLD,
            min_logprob_threshold=validator_config.MIN_LOGPROB_THRESHOLD,
        )
        mean_logprobs.append(verdict.mean_logprob)
        label = f"{log_prefix} " if log_prefix else ""
        logger.info(
            "%sprompt %d: correctness=%s mean_lp=%.3f min_lp=%.3f",
            label,
            i,
            "PASS" if verdict.passed else "FAIL",
            verdict.mean_logprob,
            verdict.min_logprob,
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
) -> list[RawPromptResult]:
    """Send prompts to a running server, optionally discarding warmup results."""
    results: list[RawPromptResult] = []
    for i, prompt in enumerate(prompts):
        msgs = [{"role": m.role, "content": m.content} for m in prompt.messages]
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

    return heuristic


def _baseline_cmd_args(gpu_count: int, model_volume: str = "") -> list[str]:
    """Build the vLLM server command args for the baseline container."""
    mml = _max_model_len(gpu_count, model_path=model_volume)
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


def _scoring_baseline_cmd_args(gpu_count: int, model_volume: str = "") -> list[str]:
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
        "--disable-custom-all-reduce",
    ]
    if gpu_count > 1:
        args.extend(["--tensor-parallel-size", str(gpu_count)])
    return args


def run_baseline_if_needed(
    prompts: list[Prompt],
    *,
    baseline_image: str,
    baseline_digest: str,
    model_volume: str,
    gpu_count: int,
    cache_dir: Path,
    block_hash: str,
    state_dir: str | Path = "",
    startup_timeout_s: int = 600,
    per_prompt_timeout_s: int = 120,
    n_warmup: int = 2,
) -> BaselineCache:
    """Load cached baseline or run the vLLM baseline container, measure, and cache."""
    cache_key = derive_cache_key(block_hash, baseline_digest, regime="stress")
    cached = load_cached_baseline(cache_dir, cache_key)
    if cached is not None:
        logger.info(
            "Baseline cache hit for key=%s (%d prompts)", cache_key, len(cached.results)
        )
        return cached

    logger.info("Baseline cache miss for key=%s -- running baseline", cache_key)

    baseline_args = _baseline_cmd_args(gpu_count, model_volume=model_volume)
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
            extra_env={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
            compile_cache_volume=validator_config.VLLM_COMPILE_CACHE_DIR,
        )
        wait_for_health(base_url, timeout_s=startup_timeout_s, container_id=cid)

        results = _run_prompts_on_server(
            base_url,
            prompts,
            stream=True,
            logprobs=True,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=n_warmup,
        )
    finally:
        if state_dir:
            log_label = f"baseline_{cache_key}"
            capture_container_logs(cid or container_name, state_dir, log_label)
        stop_and_remove(cid or container_name)
        reset_gpu_state()

    errors = [r for r in results if r.error]
    if errors:
        err_msg = "; ".join(f"prompt {r.prompt_index}: {r.error}" for r in errors)
        raise RuntimeError(f"Baseline had prompt errors (not caching): {err_msg}")

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

    cache = BaselineCache(cache_key=cache_key, results=baseline_results)
    save_baseline_cache(cache_dir, cache_key, cache)
    logger.info(
        "✅ Baseline cached: key=%s, %d prompts", cache_key, len(baseline_results)
    )
    return cache


def start_baseline_for_scoring(
    *,
    model_volume: str,
    gpu_count: int,
    state_dir: str | Path = "",
    startup_timeout_s: int = 900,
) -> tuple[str, str]:
    """Start the baseline container for the teacher-forcing scoring pass.

    Uses the dedicated SCORING_IMAGE (default v0.9.2) for prompt_logprobs
    on Pass 2 audit prompts (~4k context).

    Returns ``(container_id, base_url)``. Caller is responsible for
    calling ``stop_and_remove`` when done.
    """
    scoring_image = validator_config.SCORING_IMAGE
    scoring_args = _scoring_baseline_cmd_args(gpu_count, model_volume=model_volume)
    logger.info("Scoring baseline cmd args: %s", scoring_args)
    container_name = "cacheon-baseline-scoring"
    pull_image(scoring_image, "")
    cid, base_url = start_container(
        scoring_image,
        "",
        model_volume=model_volume,
        cmd_args=scoring_args,
        container_name=container_name,
        extra_env={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        compile_cache_volume=validator_config.VLLM_COMPILE_CACHE_DIR,
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
    startup_timeout_s: int = 120,
) -> str:
    """Start a stopped scoring container and return its base URL."""
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
    stress_prompts: list[Prompt],
    audit_prompts: list[Prompt],
    stress_baseline: BaselineCache,
    *,
    model_volume: str,
    startup_timeout_s: int,
    per_prompt_timeout_s: int,
    n_warmup: int,
    current_block: int,
    state_dir: str | Path = "",
    log_label: str = "",
    collected_audit_output_texts: list | None = None,
    collected_audit_miner_tokens: list | None = None,
) -> EvaluationRecord:
    """Full lifecycle for one challenger. Returns an EvaluationRecord.

    Runs Pass 1 (stress) and Pass 2 (audit) in one miner container session.
    Speed score uses stress prompts only. Pass 1 aggregate token match vs
    stress baseline is a hard DQ gate. Pass 2 teacher-forcing is done by
    the caller after GPU handoff to the scoring container.

    If ``collected_audit_output_texts`` / ``collected_audit_miner_tokens``
    are provided, audit outputs are appended for deferred Pass 2 scoring.
    """
    container_name = f"cacheon-uid{com.uid}-{com.hotkey[:8]}"
    cid: str | None = None
    stress_results: list[RawPromptResult] = []
    audit_results: list[RawPromptResult] = []
    eval_error: Exception | None = None

    try:
        pull_image(com.image, com.digest)
        cid, base_url = start_container(
            com.image,
            com.digest,
            model_volume=model_volume,
            container_name=container_name,
        )
        wait_for_health(base_url, timeout_s=startup_timeout_s, container_id=cid)

        stress_results = _run_prompts_on_server(
            base_url,
            stress_prompts,
            stream=True,
            logprobs=True,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=n_warmup,
        )
        audit_results = _run_prompts_on_server(
            base_url,
            audit_prompts,
            stream=True,
            logprobs=True,
            per_prompt_timeout_s=per_prompt_timeout_s,
            n_warmup=0,
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

    all_results = stress_results + audit_results
    errors = [r for r in all_results if r.error]
    if errors:
        err_msg = "; ".join(f"prompt {r.prompt_index}: {r.error}" for r in errors)
        logger.warning("Challenger UID %d had prompt errors: %s", com.uid, err_msg)
        return _dq_record(com, current_block, f"prompt_errors: {err_msg}")

    if collected_audit_output_texts is not None:
        for r in audit_results:
            collected_audit_output_texts.append(r.output_text)
    if collected_audit_miner_tokens is not None:
        for r in audit_results:
            collected_audit_miner_tokens.append(r.tokens)

    baseline_tokens = [bl.tokens for bl in stress_baseline.results]
    miner_stress_tokens = [r.tokens for r in stress_results]
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
        return EvaluationRecord(
            uid=com.uid,
            hotkey=com.hotkey,
            commit_block=com.commit_block,
            image=com.image,
            digest=com.digest,
            score=0.0,
            ttft_improvement=0.0,
            throughput_improvement=0.0,
            token_match_rate=agg_match_rate,
            disqualified=True,
            disqualify_reason=reason,
            evaluated_at=time.time(),
            evaluation_block=current_block,
        )

    per_prompt: list[PerPromptResult] = []
    miner_ttfts: list[float] = []
    miner_tps_list: list[float] = []
    baseline_ttfts: list[float] = []
    baseline_tps_list: list[float] = []

    n_scored = min(len(stress_results), len(stress_baseline.results))
    for i in range(n_scored):
        bl = stress_baseline.results[i]
        r = stress_results[i]
        match_rate = compute_token_match_rate(bl.tokens, r.tokens)

        miner_decode = r.decode_elapsed_secs or []
        bl_decode = bl.decode_elapsed_secs or []
        miner_tps = compute_aligned_throughput_tps(bl.output_tokens, miner_decode)
        baseline_tps = compute_aligned_throughput_tps(bl.output_tokens, bl_decode)
        if len(miner_decode) < bl.output_tokens:
            logger.debug(
                "Challenger UID %d prompt %d: no throughput credit "
                "(miner emitted %d tokens, baseline %d)",
                com.uid,
                i,
                len(miner_decode),
                bl.output_tokens,
            )

        miner_ttfts.append(r.ttft_s)
        miner_tps_list.append(miner_tps)
        baseline_ttfts.append(bl.ttft_s)
        baseline_tps_list.append(baseline_tps)

        per_prompt.append(
            PerPromptResult(
                ttft_s=r.ttft_s,
                throughput_tps=miner_tps,
                output_tokens=r.output_tokens,
                token_match_rate=match_rate,
            )
        )

    per_prompt_dicts = [pp.to_dict() for pp in per_prompt] if per_prompt else None

    score, ttft_imp, tps_imp = compute_improvements(
        baseline_ttfts,
        miner_ttfts,
        baseline_tps_list,
        miner_tps_list,
    )

    logger.info(
        "Challenger UID %d scored: score=%.4f ttft_imp=%.4f tps_imp=%.4f "
        "match_rate=%.4f (%d stress prompts)",
        com.uid,
        score,
        ttft_imp,
        tps_imp,
        agg_match_rate,
        len(per_prompt),
    )
    for pp in per_prompt:
        logger.debug(
            "  prompt: ttft=%.4fs tps=%.1f tokens=%d match=%.4f",
            pp.ttft_s,
            pp.throughput_tps,
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
        ttft_improvement=ttft_imp,
        throughput_improvement=tps_imp,
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
        ttft_improvement=0.0,
        throughput_improvement=0.0,
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
    baseline_cache_dir: str,
    baseline_image: str,
    baseline_digest: str,
    gpu_count: int = 0,
    state_dir: str = "",
    startup_timeout_s: int = 600,
    per_prompt_timeout_s: int = 120,
    n_warmup: int = 2,
) -> Callable:
    """Return an ``EvalFn`` that runs Docker eval per challenger.

    For each challenger, runs the full Docker lifecycle sequentially.
    Baseline is run once (or loaded from cache) per block hash.

    If ``gpu_count`` is positive it is used directly; otherwise
    ``nvidia-smi`` auto-detection is attempted on first eval.
    """
    cache_dir = Path(baseline_cache_dir)
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

        from .prompts import sample_audit_prompts, sample_stress_prompts

        mml = _max_model_len(resolved_gpu_count, model_path=model_volume)
        stress_prompts = sample_stress_prompts(
            block_hash,
            n=EVAL_N_STRESS_SCORED + n_warmup,
            max_context_tokens=mml,
        )
        audit_prompts = sample_audit_prompts(block_hash)

        baseline = run_baseline_if_needed(
            stress_prompts,
            baseline_image=baseline_image,
            baseline_digest=baseline_digest,
            model_volume=model_volume,
            gpu_count=resolved_gpu_count,
            cache_dir=cache_dir,
            block_hash=block_hash,
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
                stress_prompts,
                audit_prompts,
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
