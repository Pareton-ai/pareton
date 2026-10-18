"""Pinned Exgentic v2 request snapshots for the version 6 tiered workload.

The source unit is a session; the sampling unit is a successful LLM call.
Normalize to Arrow-backed JSON rows so nested message variants cannot change
the inferred Arrow schema. Source outputs and live tool execution are unused.
"""

from __future__ import annotations

import json
import logging

from bench.sampler import SamplerError

DATASET = "Exgentic/agent-llm-traces-v2"
REVISION = "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814"
TIERS = ("16k", "32k", "64k", "128k")
logger = logging.getLogger(__name__)


def json_list(value, field):
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise ValueError(f"invalid {field} JSON") from exc
    if not isinstance(result, list) or any(not isinstance(v, dict) for v in result):
        raise ValueError(f"{field} must be a list of objects")
    return result


def text_value(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def normalize_messages(value):
    """Translate OTel parts into chat messages without flattening tool structure."""
    result = []
    for message in json_list(value, "messages"):
        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unsupported message role")
        parts = json_list(message.get("parts"), "message parts")
        current = {"role": role, "content": ""}

        def flush(role=role):
            nonlocal current
            if (
                current["content"]
                or "tool_calls" in current
                or "reasoning_content" in current
            ):
                result.append(current)
            current = {"role": role, "content": ""}

        for part in parts:
            kind = part.get("type")
            if kind == "text":
                if not isinstance(part.get("content"), str):
                    raise ValueError("text part lacks string content")
                current["content"] += part["content"]
            elif kind == "thinking" and role == "assistant":
                if not isinstance(part.get("thinking"), str):
                    raise ValueError("thinking part lacks text")
                current["reasoning_content"] = (
                    current.get("reasoning_content", "") + part["thinking"]
                )
            elif kind == "tool_call" and role == "assistant":
                if not all(
                    isinstance(part.get(k), str) and part[k] for k in ("id", "name")
                ):
                    raise ValueError("tool call lacks name or id")
                current.setdefault("tool_calls", []).append(
                    {
                        "id": part["id"],
                        "type": "function",
                        "function": {
                            "name": part["name"],
                            "arguments": text_value(part.get("arguments")),
                        },
                    }
                )
            elif kind == "tool_call_response":
                if not isinstance(part.get("id"), str) or not part["id"]:
                    raise ValueError("tool response lacks id")
                flush()
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": part["id"],
                        "content": text_value(part.get("result")),
                    }
                )
            else:
                # Reject unsupported modalities instead of silently deleting input.
                raise ValueError(f"unsupported message part: {kind!r}")
        flush()
    if not result:
        raise ValueError("empty messages")
    return result


def normalize_snapshot(row, source_row, span_index):
    """Return one replayable request, or None for a failed/unsupported call."""
    span = row["spans"][span_index]
    if not isinstance(span, dict):
        return None
    attrs = span.get("attributes") or {}
    if not isinstance(attrs, dict):
        return None
    if (span.get("status") or {}).get("code") == 2 or attrs.get("error.type"):
        return None
    if attrs.get("gen_ai.operation.name") != "chat" and span.get("type") != "llm_call":
        return None
    counts = [attrs.get(f"gen_ai.usage.{kind}_tokens") for kind in ("input", "output")]
    if any(type(count) is not int or count <= 0 for count in counts):
        return None
    session = row.get("session_id")
    if not isinstance(session, str) or not session:
        raise SamplerError(f"Exgentic source row {source_row} has no session_id")
    try:
        messages = normalize_messages(attrs.get("gen_ai.input.messages"))
        system = attrs.get("gen_ai.system_instructions")
        if system:
            if not isinstance(system, str):
                raise ValueError("system instructions must be a string")
            try:
                parts = json.loads(system)
            except ValueError:
                parts = None
            prefix = (
                normalize_messages([{"role": "system", "parts": parts}])
                if isinstance(parts, list)
                else [{"role": "system", "content": system}]
            )
            messages = prefix + messages
        tools = []
        if attrs.get("gen_ai.tool.definitions"):
            for tool in json_list(attrs["gen_ai.tool.definitions"], "tool definitions"):
                if tool.get("type") != "function" or not isinstance(
                    tool.get("name"), str
                ):
                    raise ValueError("unsupported tool definition")
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description") or "",
                            "parameters": tool.get("parameters") or {},
                        },
                    }
                )
    except (ValueError, TypeError) as exc:
        logger.debug(
            "Skipping Exgentic row=%s span=%s: %s", source_row, span_index, exc
        )
        return None
    return {
        "session_id": session,
        "source_row": source_row,
        "span_index": span_index,
        "benchmark": row.get("benchmark"),
        "harness": row.get("harness"),
        "source_model": attrs.get("gen_ai.request.model"),
        "recorded_input_tokens": counts[0],
        "recorded_output_tokens": counts[1],
        "messages": messages,
        "tools": tools,
    }


def _flatten_batch(batch, indices):
    snapshots = []
    for offset, index in enumerate(indices):
        row = {key: values[offset] for key, values in batch.items()}
        spans = row.get("spans")
        if not isinstance(spans, list):
            raise SamplerError(f"Exgentic source row {index} has invalid spans")
        for span_index in range(len(spans)):
            snapshot = normalize_snapshot(row, index, span_index)
            if snapshot is not None:
                snapshots.append(
                    json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
                )
    return {"snapshot_json": snapshots}


def flatten_dataset(dataset):
    """Deterministic session-order/span-order expansion, cached by datasets.map."""
    from datasets import Features, Value

    return dataset.map(
        _flatten_batch,
        batched=True,
        with_indices=True,
        batch_size=1,
        remove_columns=dataset.column_names,
        features=Features({"snapshot_json": Value("string")}),
        desc="Normalize Exgentic v2 request snapshots",
    )


def snapshot_from_row(row):
    try:
        snapshot = json.loads(row["snapshot_json"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SamplerError("invalid normalized Exgentic snapshot") from exc
    if not isinstance(snapshot, dict) or not snapshot.get("session_id"):
        raise SamplerError("invalid normalized Exgentic snapshot identity")
    return snapshot


def candidate_for_snapshot(row_index, row, formatter, rule, context):
    from bench.longform import digest, input_group
    from bench.trajectory import token_ids_sha256

    snapshot = snapshot_from_row(row)
    # Preserve the source snapshot; append the pinned task only for rendering.
    # Tier selection and capacity checks include its tokens and chat framing.
    messages = [
        *snapshot["messages"],
        {"role": "user", "content": rule["followup_prompt"]},
    ]
    prompt = formatter.render(messages, tools=snapshot["tools"] or None)
    ids = formatter.encode(prompt)
    group = input_group(len(ids), rule["input_tiers"])
    if (
        group is None
        or len(ids) > context["max_input_tokens"]
        or len(ids) + rule["max_tokens"] + context["engine_reserve"]
        > context["max_model_len"]
    ):
        return None
    return {
        "row_index": row_index,
        "prompt": prompt,
        "input_tokens": len(ids),
        "input_ids_sha256": token_ids_sha256(ids),
        "input_length_group": group,
        "session_id": snapshot["session_id"],
        "source_row": snapshot["source_row"],
        "span_index": snapshot["span_index"],
        "snapshot_sha256": digest(snapshot),
    }
