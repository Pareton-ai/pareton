"""Version 3 SWE-agent history sampling, with token-counted context coverage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bench.sampler import (
    PromptFormatter,
    PromptRenderError,
    SampledTrace,
    SamplerError,
    encode_trace,
)

GROUPS = ("4k", "8k", "16k", "32k")


def sampling_context_for_campaign(
    bench: dict[str, Any] | None, engine: dict[str, Any] | None = None
) -> dict[str, Any]:
    model = (bench or {}).get("model")
    context = model.get("max_model_len") if isinstance(model, dict) else None
    if type(context) is not int or context < 32:
        raise SamplerError(
            "trajectory sampling requires bench.model.max_model_len >= 32"
        )
    name = (engine or {}).get("name", "vllm")
    if name not in ("vllm", "sglang"):
        raise SamplerError(f"unsupported trajectory engine: {name!r}")
    # SGLang: max_req_len <= context - 1; output <= max_req_len - input - 1;
    # input < max_req_len - 5. Actual capacity is checked before baseline replay.
    limits = {
        "engine_name": name,
        "max_model_len": context,
        "engine_reserve": 2 if name == "sglang" else 0,
        "max_input_tokens": context - (7 if name == "sglang" else 1),
    }
    if limits["max_input_tokens"] < length_groups(4)[-1]["min_tokens"]:
        raise SamplerError("campaign context is too small for the fixed 32K input tier")
    return limits


def _validate_context(context: dict[str, Any] | None) -> None:
    if not isinstance(context, dict):
        raise SamplerError("algo_version 3 requires campaign context limits")
    expected_context = sampling_context_for_campaign(
        {"model": {"max_model_len": context.get("max_model_len")}},
        {"name": context.get("engine_name")},
    )
    if context != expected_context:
        raise SamplerError(
            "sampling context does not match the engine's length contract"
        )


def length_groups(n_prompts: int) -> list[dict[str, Any]]:
    targets = (4096, 8192, 16384, 32768)
    return [
        {
            "name": name,
            "min_tokens": (target * 9 + 9) // 10,
            "max_tokens": target,
            "count": n_prompts // len(GROUPS) + (i < n_prompts % len(GROUPS)),
        }
        for i, (name, target) in enumerate(zip(GROUPS, targets))
    ]


def validate_trajectory_trace(
    requests: list[dict[str, Any]], sampling: dict[str, Any]
) -> None:
    """One workload contract for trace construction and replay validation."""
    if not isinstance(sampling, dict) or sampling.get("algo_version") != 3:
        raise SamplerError("unsupported trace sampling version")
    context = sampling.get("context")
    _validate_context(context)
    interval = sampling.get("request_interval_ms")
    if (
        type(interval) is not int
        or interval < 0
        or not isinstance(sampling.get("enable_thinking"), bool)
    ):
        raise SamplerError(
            "trace needs a nonnegative interval and boolean thinking mode"
        )
    if len(requests) < len(GROUPS):
        raise SamplerError(
            "trajectory trace requires at least 4 requests for context coverage"
        )
    groups = {g["name"]: g for g in length_groups(len(requests))}
    counts = dict.fromkeys(GROUPS, 0)
    for i, request in enumerate(requests):
        size = request.get("input_tokens")
        group = groups.get(request.get("input_length_group"))
        for key, minimum in (("max_tokens", 1), ("arrival_offset_ms", 0)):
            if type(request.get(key)) is not int or request[key] < minimum:
                raise SamplerError(
                    f"trace request {i}: {key} must be an integer >= {minimum}"
                )
        if type(size) is not int or not 1 <= size <= context["max_input_tokens"]:
            raise SamplerError("trace request has invalid input token count")
        if (
            size + request["max_tokens"] + context["engine_reserve"]
            > context["max_model_len"]
        ):
            raise SamplerError("trace request exceeds its context window")
        if (
            not isinstance(request.get("prompt"), str)
            or not request["prompt"]
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(request.get("input_ids_sha256", ""))
            )
        ):
            raise SamplerError(
                "trajectory trace requires text and a pinned input token hash"
            )
        if request["arrival_offset_ms"] != i * interval:
            raise SamplerError("trace arrivals do not match request_interval_ms")
        if group is None or not group["min_tokens"] <= size <= group["max_tokens"]:
            raise SamplerError("trace input does not match its context length group")
        counts[group["name"]] += 1
    if any(counts[name] != group["count"] for name, group in groups.items()):
        raise SamplerError("trace does not fill every input-length group")


def normalize_trajectory(row: dict[str, Any]) -> list[tuple[int, dict[str, str]]]:
    """Keep source indices for receipts; malformed histories are ineligible."""
    raw = row.get("trajectory")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    messages = []
    expected_role = "user"
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            return []
        role = message.get("role")
        if role == "system":
            continue
        role = "assistant" if role == "ai" else role
        if role != expected_role:
            return []
        content = message.get("text")
        if not isinstance(content, str) or not content.strip():
            content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return []
        messages.append((index, {"role": role, "content": content}))
        expected_role = "assistant" if role == "user" else "user"
    return messages


def token_ids_sha256(ids: list[int]) -> str:
    raw = json.dumps(ids, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Prefix:
    row_index: int
    end_message_index: int
    prompt: str
    input_tokens: int
    input_ids_sha256: str
    max_tokens: int


def _prefixes(
    row_index: int,
    row: dict[str, Any],
    formatter: PromptFormatter,
    context: dict[str, Any],
    max_tokens: int,
    *,
    end_message_index: int | None = None,
):
    messages = normalize_trajectory(row)
    assert formatter.encode is not None
    for i, (source_index, message) in enumerate(messages[:-1]):
        if message["role"] != "user":
            continue
        if end_message_index is not None and source_index != end_message_index:
            continue
        prompt = formatter.render([m for _, m in messages[: i + 1]])
        ids = formatter.encode(prompt)
        size = len(ids)
        allowance = min(
            max_tokens, context["max_model_len"] - size - context["engine_reserve"]
        )
        if not size or size > context["max_input_tokens"] or allowance < 1:
            continue
        yield Prefix(
            row_index, source_index, prompt, size, token_ids_sha256(ids), allowance
        )


def _fetch(row_fetcher: Callable[[int], dict[str, Any]], index: int) -> dict[str, Any]:
    try:
        row = row_fetcher(index)
    except Exception as exc:
        raise SamplerError(
            f"row fetch failed at {index}: {type(exc).__name__}"
        ) from exc
    return row if isinstance(row, dict) else {}


def _select(
    rule: dict[str, Any],
    seed: str,
    row_fetcher: Callable[[int], dict[str, Any]],
    formatter: PromptFormatter,
    context: dict[str, Any],
    groups: list[dict[str, Any]],
) -> list[tuple[str, Prefix]]:
    slots = {g["name"]: [None] * g["count"] for g in groups}
    options: dict[int, dict[str, Prefix]] = {}

    # A row can fit several groups. Reassign selected rows when needed so an
    # early choice cannot steal the only row that can fill another group.
    def place(row: int, visited: set[int]) -> bool:
        if row in visited:
            return False
        visited.add(row)
        for group in reversed(GROUPS):
            if group not in options[row]:
                continue
            for i, previous in enumerate(slots[group]):
                if previous is None or place(previous, visited):
                    slots[group][i] = row
                    return True
        return False

    # Every row is reachable, including rare long histories. Hash sorting is
    # stable across Python versions and needs no persisted random generator.
    order = sorted(
        range(rule["n_rows"]),
        key=lambda i: hashlib.sha256(f"{seed}:row:{i}".encode()).digest(),
    )
    for row_index in order:
        eligible = {}
        row = _fetch(row_fetcher, row_index)
        try:
            for prefix in _prefixes(
                row_index, row, formatter, context, rule["max_tokens"]
            ):
                for group in groups:
                    if (
                        group["min_tokens"]
                        <= prefix.input_tokens
                        <= group["max_tokens"]
                    ):
                        eligible[group["name"]] = prefix
        except PromptRenderError:
            # Discard the whole row, including prefixes rendered before the
            # failure. Receipt replay bypasses selection and must still fail.
            continue
        if not eligible:
            continue
        options[row_index] = eligible
        if not place(row_index, set()):
            del options[row_index]
        if all(all(row is not None for row in rows) for rows in slots.values()):
            break
    missing = {g: rows.count(None) for g, rows in slots.items() if None in rows}
    if missing:
        raise SamplerError(
            f"trajectory input-length coverage unavailable: {missing}; no shorter fallback"
        )
    selected = [(g, options[row][g]) for g, rows in slots.items() for row in rows]
    return sorted(
        selected,
        key=lambda item: hashlib.sha256(
            f"{seed}:arrival:{item[1].row_index}:{item[1].end_message_index}".encode()
        ).digest(),
    )


def generate_trajectory_trace(
    *,
    rule: dict[str, Any],
    seed_hex: str,
    row_fetcher: Callable[[int], dict[str, Any]],
    formatter: PromptFormatter | None,
    context: dict[str, Any] | None,
    receipt: dict[str, Any] | None,
    sample_seed_block: int,
    sample_seed_block_hash: str,
) -> SampledTrace:
    if formatter is None or formatter.encode is None:
        raise SamplerError(
            "algo_version 3 requires a pinned chat formatter and tokenizer"
        )
    _validate_context(context)
    seed = seed_hex.removeprefix("sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", seed):
        raise SamplerError("trajectory sampling requires a 64-character hex seed")
    template = formatter.receipt.get("chat_template", {})
    if template.get("enable_thinking") is not rule["enable_thinking"]:
        raise SamplerError("formatter thinking mode does not match sampling_rule")
    if not formatter.receipt.get("tokenizer"):
        raise SamplerError("algo_version 3 requires tokenizer receipt metadata")
    groups = length_groups(rule["n_prompts"])
    if receipt is None:
        selected = _select(rule, seed, row_fetcher, formatter, context, groups)
    else:
        selections = receipt.get("requests")
        if not isinstance(selections, list) or len(selections) != rule["n_prompts"]:
            raise SamplerError("sampling receipt is missing request selections")
        selected = []
        seen = set()
        for selection in selections:
            if not isinstance(selection, dict):
                raise SamplerError("sampling receipt selection must be an object")
            row_index = selection.get("row_index")
            end = selection.get("end_message_index")
            if (
                type(row_index) is not int
                or row_index in seen
                or not 0 <= row_index < rule["n_rows"]
                or type(end) is not int
            ):
                raise SamplerError("sampling receipt has an invalid row or cut point")
            seen.add(row_index)
            choices = list(
                _prefixes(
                    row_index,
                    _fetch(row_fetcher, row_index),
                    formatter,
                    context,
                    rule["max_tokens"],
                    end_message_index=end,
                )
            )
            if len(choices) != 1:
                raise SamplerError(
                    "sampling receipt cut point is not an eligible conversation prefix"
                )
            selected.append((selection.get("input_length_group"), choices[0]))

    requests, selections = [], []
    for i, (group, prefix) in enumerate(selected):
        settings = {"temperature": 0.0, "top_p": 1.0}
        if rule.get("ignore_eos"):
            settings["ignore_eos"] = True
        metadata = {
            "request_id": f"hf-{i:03d}",
            "input_tokens": prefix.input_tokens,
            "input_ids_sha256": prefix.input_ids_sha256,
            "input_length_group": group,
            "max_tokens": prefix.max_tokens,
            "arrival_offset_ms": i * rule["request_interval_ms"],
        }
        requests.append(
            {
                "id": metadata["request_id"],
                "arrival_offset_ms": metadata["arrival_offset_ms"],
                "prompt": prefix.prompt,
                "max_tokens": prefix.max_tokens,
                "sampling": settings,
                "input_tokens": prefix.input_tokens,
                "input_ids_sha256": prefix.input_ids_sha256,
                "input_length_group": group,
            }
        )
        selections.append(
            {
                **metadata,
                "row_index": prefix.row_index,
                "end_message_index": prefix.end_message_index,
            }
        )
    workload = {
        "algo_version": 3,
        "request_interval_ms": rule["request_interval_ms"],
        "enable_thinking": rule["enable_thinking"],
        "context": context,
    }
    trace = {
        "schema_version": 1,
        "meta": {
            "name": f"hf-trajectories-{seed[:12]}",
            "description": f"{len(requests)} conversation prefixes across context lengths",
            "sampling": workload,
        },
        "requests": requests,
    }
    validate_trajectory_trace(requests, workload)
    body = encode_trace(trace)
    sha = "sha256:" + hashlib.sha256(body).hexdigest()
    result_receipt = {
        **rule,
        **formatter.receipt,
        "context": context,
        "length_groups": groups,
        "sample_seed_block": sample_seed_block,
        "sample_seed_block_hash": sample_seed_block_hash.strip().lower(),
        "seed_hex": seed,
        "row_indices": [p.row_index for _, p in selected],
        "requests": selections,
        "sampled_trace_sha256": sha,
    }
    if receipt is not None and result_receipt != receipt:
        raise SamplerError("sampling receipt does not reproduce the selected trace")
    return SampledTrace(
        sha,
        body,
        seed,
        sample_seed_block,
        sample_seed_block_hash.strip().lower(),
        tuple(result_receipt["row_indices"]),
        result_receipt,
    )


def preflight_trajectory_campaign(
    rule: dict[str, Any], bench: dict[str, Any], engine: dict[str, Any] | None
) -> SampledTrace:
    """Verify all length groups before seeding an open campaign; no inference."""
    from bench.sampler import build_prompt_formatter, fetch_hf_row, generate_trace

    model = bench["model"]
    return generate_trace(
        rule=rule,
        seed_hex="0" * 64,
        row_fetcher=lambda i: fetch_hf_row(rule, i),
        prompt_formatter=build_prompt_formatter(
            rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
        ),
        sampling_context=sampling_context_for_campaign(bench, engine),
    )
