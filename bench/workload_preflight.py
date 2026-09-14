"""Check version 3 tokenization and capacity before warming an engine."""

from __future__ import annotations

import json
from pathlib import Path

from bench.http import get_json, post_json
from bench.lifecycle import EngineError
from bench.schemas import WorkloadTrace
from bench.trajectory import token_ids_sha256


def validate_engine_workload(
    base_url: str,
    trace: WorkloadTrace,
    *,
    engine_name: str,
    max_model_len: int,
    evidence_dir: Path,
    verify_tokenizer: bool,
) -> None:
    sampling = trace.meta.sampling
    if sampling is None:
        return
    context = sampling["context"]
    if (
        context["max_model_len"] != max_model_len
        or context["engine_name"] != engine_name
    ):
        raise EngineError("trace context contract does not match bench request")
    evidence = {"context": context, "tokenizer_verified": verify_tokenizer}
    if engine_name == "sglang":
        info = get_json(base_url, "/server_info")
        fields = (
            "context_length",
            "max_req_input_len",
            "max_total_num_tokens",
            "page_size",
        )
        if any(type(info.get(key)) is not int or info[key] < 1 for key in fields):
            raise EngineError(
                "SGLang server info lacks resolved context/capacity limits"
            )
        if info["context_length"] != max_model_len:
            raise EngineError("SGLang context length differs from campaign")
        page = info["page_size"]
        dcp = info.get("attn_dcp_size", 1)
        if type(dcp) is not int or dcp < 1:
            raise EngineError("SGLang server info has invalid attention parallelism")
        capacity = info["max_total_num_tokens"] * dcp
        reserved = 0
        if str(info.get("speculative_algorithm") or "").upper() in ("EAGLE", "EAGLE3"):
            keys = (
                "speculative_eagle_topk",
                "speculative_num_steps",
                "speculative_num_draft_tokens",
            )
            if any(type(info.get(k)) is not int or info[k] < 1 for k in keys):
                raise EngineError(
                    "SGLang speculation requires resolved token reservations"
                )
            reserved = max(info[keys[0]] * info[keys[1]], info[keys[2]])
        for request in trace.requests:
            size = request.input_tokens
            assert size is not None
            paged = ((size + page - 1) // page) * page
            if (
                size >= info["max_req_input_len"]
                or size + request.max_tokens > info["max_req_input_len"] + 4
                or paged + request.max_tokens + page >= capacity
                or size + request.max_tokens + reserved > max_model_len
            ):
                raise EngineError(
                    f"request {request.id}: SGLang would truncate the pinned workload"
                )
        evidence["engine_limits"] = {key: info[key] for key in fields}
    else:
        info = get_json(base_url, "/v1/models")
        models = info.get("data") or []
        if not models or any(
            model.get("max_model_len") != max_model_len for model in models
        ):
            raise EngineError(
                "vLLM model metadata does not match campaign context length"
            )
        evidence["engine_limits"] = {"max_model_len": max_model_len}
    if verify_tokenizer:
        for request in trace.requests:
            response = post_json(
                base_url,
                "/tokenize",
                {"prompt": request.prompt, "add_special_tokens": True},
            )
            ids = response.get("tokens")
            if (
                not isinstance(ids, list)
                or any(type(i) is not int or i < 0 for i in ids)
                or len(ids) != request.input_tokens
                or token_ids_sha256(ids) != request.input_ids_sha256
            ):
                raise EngineError(
                    f"request {request.id}: trusted engine tokenization differs from sampled input"
                )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "workload_preflight.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
