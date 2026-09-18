"""Version 4: tiered LongWriter follow-ups with natural EOS stopping."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from bench.sampler import PromptRenderError, SampledTrace, SamplerError, encode_trace
from bench.trajectory import token_ids_sha256

DEFAULT_FOLLOWUP_PROMPT = (
    "Write a new, original, self-contained long-form work. Use the previous "
    "response as an example of depth and detail, but choose a different topic "
    "from coding, fiction, non-fiction, science, history, astronomy, or internet "
    "culture. Surprise me. Produce the complete work, not an outline, summary, "
    "or discussion of what you would write. Aim for approximately 4,000-6,000 words."
)


def length_groups(n_prompts):
    """Keep 90-100% input bands, with equal quotas at 2K, 4K, 8K and 16K."""
    return [
        {
            "name": f"{target // 1024}k",
            "min_tokens": (target * 9 + 9) // 10,
            "max_tokens": target,
            "count": n_prompts // 4,
        }
        for target in (2048, 4096, 8192, 16384)
    ]


def input_group(input_tokens):
    return next(
        (
            g["name"]
            for g in length_groups(4)
            if g["min_tokens"] <= input_tokens <= g["max_tokens"]
        ),
        None,
    )


def digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )


def parse_longform_fields(rule, parsed):
    if parsed.get("ignore_eos"):
        raise SamplerError("long-form sampling requires normal EOS stopping")
    if parsed["enable_thinking"]:
        raise SamplerError("long-form sampling requires enable_thinking=false")
    if parsed["n_prompts"] < 4 or parsed["n_prompts"] % 4:
        raise SamplerError("algo_version 4 requires n_prompts to be a multiple of 4")
    prompt = rule.get("followup_prompt", DEFAULT_FOLLOWUP_PROMPT)
    if not isinstance(prompt, str) or not prompt.strip():
        raise SamplerError("followup_prompt must be nonempty text")
    minimum = rule.get("min_output_tokens", 3000)
    if type(minimum) is not int or minimum < 1:
        raise SamplerError("min_output_tokens must be a positive integer")
    if minimum > parsed["max_tokens"]:
        raise SamplerError("min_output_tokens exceeds max_tokens")
    result = {"followup_prompt": prompt, "min_output_tokens": minimum}
    if "eligible_row_indices" in rule:
        rows = rule["eligible_row_indices"]
        if (
            not isinstance(rows, list)
            or len(rows) < parsed["n_prompts"]
            or any(type(i) is not int or not 0 <= i < parsed["n_rows"] for i in rows)
            or len(set(rows)) != len(rows)
            or rows != sorted(rows)
        ):
            raise SamplerError(
                "eligible_row_indices must be sorted, unique, in range and fill n_prompts"
            )
        result["eligible_row_indices"] = list(rows)
    if "qualification" in rule:
        q = rule["qualification"]
        if (
            not isinstance(q, dict)
            or set(q) != {"contract_sha256", "evidence_sha256", "repetitions"}
            or type(q["repetitions"]) is not int
            or q["repetitions"] < 2
            or any(
                not re.fullmatch(r"sha256:[0-9a-f]{64}", str(q[k]))
                for k in ("contract_sha256", "evidence_sha256")
            )
            or "eligible_row_indices" not in result
        ):
            raise SamplerError("invalid long-form qualification metadata")
        result["qualification"] = dict(q)
    return result


def sampling_context_for_campaign(bench, engine=None):
    model = (bench or {}).get("model") or {}
    limit = model.get("max_model_len")
    name = (engine or {}).get("name", "vllm")
    if type(limit) is not int or limit < 32 or name not in ("vllm", "sglang"):
        raise SamplerError("long-form sampling requires valid model context and engine")
    return {
        "engine_name": name,
        "max_model_len": limit,
        "engine_reserve": 2 if name == "sglang" else 0,
        "max_input_tokens": limit - (7 if name == "sglang" else 1),
    }


def validate_context(context):
    if not isinstance(context, dict) or context != sampling_context_for_campaign(
        {"model": {"max_model_len": context.get("max_model_len")}},
        {"name": context.get("engine_name")},
    ):
        raise SamplerError("invalid long-form context contract")


def qualification_contract(rule, bench, engine):
    return digest(
        {
            "sampling_rule": {k: v for k, v in rule.items() if k != "qualification"},
            "bench": bench,
            "engine": engine,
        }
    )


def require_qualification(rule, bench, engine):
    q = rule.get("qualification")
    if not q or q["contract_sha256"] != qualification_contract(rule, bench, engine):
        raise SamplerError(
            "long-form campaign requires baseline qualification for these exact pins; "
            "run python -m bench.qualify_longform and pass its sampling rule"
        )


def source_messages(row, rule):
    """Use the source exchange as history and append the pinned user follow-up."""
    messages = row.get("messages") if isinstance(row, dict) else None
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or any(not isinstance(m, dict) for m in messages)
        or [m.get("role") for m in messages] != ["user", "assistant"]
        or any(
            not isinstance(m.get("content"), str) or not m["content"].strip()
            for m in messages
        )
    ):
        return None
    return [{"role": m["role"], "content": m["content"]} for m in messages] + [
        {"role": "user", "content": rule["followup_prompt"]}
    ]


def candidate_for_row(row_index, row, formatter, rule, context):
    messages = source_messages(row, rule)
    if messages is None:
        return None
    prompt = formatter.render(messages)
    ids = formatter.encode(prompt)
    group = input_group(len(ids))
    if (
        group is None
        or len(ids) > context["max_input_tokens"]
        or len(ids) + rule["max_tokens"] + context["engine_reserve"]
        > context["max_model_len"]
    ):
        return None
    history_answer = messages[1]["content"]
    return {
        "row_index": row_index,
        "prompt": prompt,
        "input_tokens": len(ids),
        "input_ids_sha256": token_ids_sha256(ids),
        "history_answer_tokens": len(formatter.encode(history_answer)),
        "history_answer_sha256": "sha256:"
        + hashlib.sha256(history_answer.encode()).hexdigest(),
        "input_length_group": group,
    }


def request_for_candidate(candidate, rule, index):
    return {
        "id": f"hf-{index:03d}",
        "arrival_offset_ms": index * rule["request_interval_ms"],
        "prompt": candidate["prompt"],
        "max_tokens": rule["max_tokens"],
        "sampling": {"temperature": 0.0, "top_p": 1.0},
        "input_tokens": candidate["input_tokens"],
        "input_ids_sha256": candidate["input_ids_sha256"],
        "input_length_group": candidate["input_length_group"],
    }


def ordered_rows(rule, seed):
    pool = rule.get("eligible_row_indices", range(rule["n_rows"]))
    return sorted(pool, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).digest())


def generate_longform_trace(
    *,
    rule,
    seed_hex,
    row_fetcher,
    formatter,
    context,
    receipt,
    sample_seed_block,
    sample_seed_block_hash,
):
    if (
        formatter is None
        or formatter.encode is None
        or not formatter.receipt.get("tokenizer")
    ):
        raise SamplerError(
            "algo_version 4 requires a pinned chat formatter and tokenizer"
        )
    if formatter.receipt.get("chat_template", {}).get("enable_thinking") is not False:
        raise SamplerError("formatter thinking mode does not match sampling_rule")
    validate_context(context)
    seed = seed_hex.removeprefix("sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", seed):
        raise SamplerError("long-form sampling requires a 64-character hex seed")
    if receipt is None:
        indices = ordered_rows(rule, seed)
    else:
        indices = receipt.get("row_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != rule["n_prompts"]
            or any(type(i) is not int or not 0 <= i < rule["n_rows"] for i in indices)
            or len(set(indices)) != len(indices)
            or (
                "eligible_row_indices" in rule
                and not set(indices) <= set(rule["eligible_row_indices"])
            )
        ):
            raise SamplerError("invalid long-form receipt row selections")
    groups = length_groups(rule["n_prompts"])
    remaining = {g["name"]: g["count"] for g in groups}
    selected, seen_prompts = [], set()
    for index in indices:
        try:
            row = row_fetcher(index)
        except Exception as exc:
            raise SamplerError(f"row fetch failed at {index}") from exc
        try:
            candidate = candidate_for_row(index, row, formatter, rule, context)
        except PromptRenderError:
            if receipt is not None:
                raise
            continue
        if candidate is None or candidate["input_ids_sha256"] in seen_prompts:
            if receipt is not None:
                raise SamplerError("long-form receipt selected an ineligible row")
            continue
        group = candidate["input_length_group"]
        if remaining[group] == 0:
            if receipt is not None:
                raise SamplerError("long-form receipt exceeds its input tier quota")
            continue
        remaining[group] -= 1
        selected.append(candidate)
        seen_prompts.add(candidate["input_ids_sha256"])
        if len(selected) == rule["n_prompts"]:
            break
    if len(selected) != rule["n_prompts"]:
        raise SamplerError(
            f"insufficient distinct long-form prompts; missing by input tier: {remaining}; "
            "no shorter fallback"
        )
    requests = [request_for_candidate(item, rule, i) for i, item in enumerate(selected)]
    workload = {
        "algo_version": 4,
        "enable_thinking": False,
        "context": context,
        "request_interval_ms": rule["request_interval_ms"],
        "max_tokens": rule["max_tokens"],
        "min_output_tokens": rule["min_output_tokens"],
        "length_groups": groups,
    }
    validate_longform_trace(requests, workload)
    body = encode_trace(
        {
            "schema_version": 1,
            "meta": {"name": f"hf-longform-{seed[:12]}", "sampling": workload},
            "requests": requests,
        }
    )
    sha = "sha256:" + hashlib.sha256(body).hexdigest()
    result = {
        **rule,
        **formatter.receipt,
        "context": context,
        "sample_seed_block": sample_seed_block,
        "sample_seed_block_hash": sample_seed_block_hash.strip().lower(),
        "seed_hex": seed,
        "row_indices": [p["row_index"] for p in selected],
        "requests": [
            {k: v for k, v in item.items() if k != "prompt"} for item in selected
        ],
        "sampled_trace_sha256": sha,
    }
    if receipt is not None and result != receipt:
        raise SamplerError("sampling receipt does not reproduce the selected trace")
    return SampledTrace(
        sha,
        body,
        seed,
        sample_seed_block,
        sample_seed_block_hash.strip().lower(),
        tuple(result["row_indices"]),
        result,
    )


def validate_longform_trace(requests, sampling):
    if (
        sampling.get("algo_version") != 4
        or sampling.get("enable_thinking") is not False
    ):
        raise SamplerError("invalid long-form trace mode")
    context = sampling.get("context")
    validate_context(context)
    for key in ("request_interval_ms", "max_tokens", "min_output_tokens"):
        value = sampling.get(key)
        if type(value) is not int or value < (0 if key == "request_interval_ms" else 1):
            raise SamplerError(f"invalid long-form {key}")
    if sampling["min_output_tokens"] > sampling["max_tokens"]:
        raise SamplerError("long-form output threshold exceeds allowance")
    groups = length_groups(len(requests))
    if (
        len(requests) < 4
        or len(requests) % 4
        or sampling.get("length_groups") != groups
    ):
        raise SamplerError("invalid long-form input tier contract")
    counts = {g["name"]: 0 for g in groups}
    for i, request in enumerate(requests):
        size = request.get("input_tokens")
        if (
            type(size) is not int
            or size < 1
            or size > context["max_input_tokens"]
            or type(request.get("max_tokens")) is not int
            or request["max_tokens"] != sampling["max_tokens"]
            or size + request["max_tokens"] + context["engine_reserve"]
            > context["max_model_len"]
            or not isinstance(request.get("prompt"), str)
            or not request["prompt"].strip()
            or request.get("prompt_token_ids") is not None
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(request.get("input_ids_sha256"))
            )
            or request.get("sampling")
            not in (
                {"temperature": 0.0, "top_p": 1.0},
                {"temperature": 0.0, "top_p": 1.0, "ignore_eos": False},
            )
            or type(request.get("arrival_offset_ms")) is not int
            or request["arrival_offset_ms"] != i * sampling["request_interval_ms"]
        ):
            raise SamplerError(
                "invalid long-form request or forced generation settings"
            )
        group = input_group(size)
        if group is None or request.get("input_length_group") != group:
            raise SamplerError("long-form request is outside its input tier")
        counts[group] += 1
    if counts != {g["name"]: g["count"] for g in groups}:
        raise SamplerError("long-form trace does not fill every input tier")


def preflight_longform_campaign(rule, bench, engine):
    from bench.sampler import build_prompt_formatter, fetch_hf_row, generate_trace

    require_qualification(rule, bench, engine)
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


def validate_natural_baseline(trace, replay):
    """A changed baseline workload must void the round, never reward a miner."""
    from bench.lifecycle import EngineError

    sampling = trace.meta.sampling or {}
    if sampling.get("algo_version") != 4:
        return
    for request in trace.requests:
        samples = replay.completion_token_samples.get(request.id, [])
        if not samples or any(n < sampling["min_output_tokens"] for n in samples):
            raise EngineError(
                f"long-form baseline request {request.id} fell below the qualified "
                f"{sampling['min_output_tokens']}-token output threshold; requalify the workload"
            )
