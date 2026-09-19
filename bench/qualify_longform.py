"""Qualify natural long responses on a trusted baseline before opening a campaign.

Run with --help. This writes local evidence and a sampling rule, never DB rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass

from bench.correctness import degeneracy_reason
from bench.http import post_json
from bench.lifecycle import EngineError
from bench.longform import (
    candidate_for_row,
    digest,
    generation_fields,
    length_groups,
    ordered_rows,
    qualification_contract,
    request_for_candidate,
    sampling_context_for_campaign,
)
from bench.sampler import (
    LONGFORM_ALGO_VERSION,
    PromptRenderError,
    SamplerError,
    build_prompt_formatter,
    fetch_hf_row,
    parse_sampling_rule,
)
from bench.schemas import WorkloadTrace
from bench.workload_preflight import validate_engine_workload

logger = logging.getLogger(__name__)


def verify_baseline_image(*, engine_ref, base_url, container):
    """Bind a loopback endpoint to an image inspected on the local Docker daemon."""
    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", engine_ref):
        raise EngineError(
            "qualification requires a published image reference by digest"
        )
    try:
        url = urlsplit(base_url)
        port = url.port or 80
    except ValueError as exc:
        raise EngineError("invalid qualification endpoint") from exc
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or url.path not in ("", "/")
        or url.query
        or url.fragment
        or url.username is not None
        or url.password is not None
    ):
        raise EngineError(
            "qualification requires a direct http://127.0.0.1:PORT endpoint on the Docker host"
        )
    if getproxies().get("http") and not proxy_bypass("127.0.0.1"):
        raise EngineError("qualification endpoint must bypass HTTP proxies")

    def inspect(kind, identifier):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "--host",
                    "unix:///var/run/docker.sock",
                    kind,
                    "inspect",
                    "--",
                    identifier,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            items = json.loads(result.stdout)
            if (
                not isinstance(items, list)
                or len(items) != 1
                or not isinstance(items[0], dict)
            ):
                raise ValueError("invalid inspect output")
            return items[0]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise EngineError(
                f"cannot verify baseline {kind} through local Docker inspection"
            ) from exc

    running = inspect("container", container)
    state = running.get("State") or {}
    if (
        state.get("Running") is not True
        or state.get("Paused")
        or state.get("Restarting")
    ):
        raise EngineError("qualification baseline container is not running normally")
    if (running.get("HostConfig") or {}).get("NetworkMode") == "host":
        raise EngineError(
            "host-network endpoints cannot be bound to a unique baseline container"
        )
    ports = (running.get("NetworkSettings") or {}).get("Ports") or {}
    bindings = [
        key
        for key, entries in ports.items()
        if key.endswith("/tcp")
        for entry in entries or []
        if entry.get("HostIp") in ("127.0.0.1", "0.0.0.0")
        and entry.get("HostPort") == str(port)
    ]
    if len(bindings) != 1:
        raise EngineError(
            "base-url does not match the baseline container's published port"
        )
    image_id = running.get("Image")
    if not isinstance(image_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_id
    ):
        raise EngineError("baseline container has no verifiable image ID")
    image = inspect("image", image_id)
    digests = image.get("RepoDigests") or []
    if image.get("Id") != image_id or engine_ref not in digests:
        raise EngineError("serving baseline image digest does not match --engine-ref")
    if not running.get("Id") or not state.get("StartedAt"):
        raise EngineError("baseline container lacks stable identity metadata")
    return {
        "engine_ref": next(ref for ref in digests if ref == engine_ref),
        "image_id": image_id,
        "container_id": running["Id"],
        "started_at": state["StartedAt"],
        "restart_count": running.get("RestartCount"),
        "base_url": base_url.rstrip("/"),
        "container_port": bindings[0],
    }


def evaluate_response(response, rule, input_tokens):
    """Length is observed, never requested through min_tokens or ignore_eos."""
    choices = response.get("choices")
    usage = response.get("usage")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or not isinstance(usage, dict)
    ):
        raise EngineError("qualification response lacks one choice and token usage")
    choice = choices[0]
    count = usage.get("completion_tokens")
    if (
        type(count) is not int
        or count < 0
        or usage.get("prompt_tokens") != input_tokens
    ):
        raise EngineError("qualification response has invalid token usage")
    if count > rule["max_tokens"]:
        raise EngineError("qualification response exceeds the output allowance")
    text = choice.get("text")
    if not isinstance(text, str):
        raise EngineError("qualification response lacks output text")
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length") or (
        finish == "length" and count != rule["max_tokens"]
    ):
        raise EngineError(
            "qualification response has an invalid finish reason or truncated allowance"
        )
    reason = None
    if count < rule["min_output_tokens"]:
        reason = "short_output"
    elif not text.strip():
        reason = "empty_output"
    else:
        reason = degeneracy_reason(text)
    return {
        "completion_tokens": count,
        "finish_reason": finish,
        "text": text,
        "rejection": reason,
    }


def qualify(
    *,
    fields,
    base_url,
    container,
    engine_ref,
    output_dir,
    pool_size=None,
    max_rows=6000,
    repetitions=2,
    concurrency=4,
    timeout=600,
    row_fetcher=None,
    formatter=None,
):
    if type(concurrency) is not int or concurrency < 1:
        raise SamplerError("concurrency must be a positive integer")
    rule = parse_sampling_rule(fields["sampling_rule"])
    if rule["algo_version"] != LONGFORM_ALGO_VERSION:
        raise SamplerError("qualification requires algo_version 4")
    if pool_size is None:
        pool_size = 2 * rule["n_prompts"]
    if repetitions < 2 or pool_size < rule["n_prompts"] or max_rows < pool_size:
        raise SamplerError(
            "qualification needs >=2 repetitions and max_rows >= pool_size >= n_prompts"
        )
    if pool_size % 4:
        raise SamplerError("version 4 pool_size must be a multiple of 4")
    quotas = {group["name"]: group["count"] for group in length_groups(pool_size)}
    qualified_counts = dict.fromkeys(quotas, 0)
    # Requalification starts from source, not a previous winning subset.
    rule.pop("qualification", None)
    rule.pop("eligible_row_indices", None)
    identity = verify_baseline_image(
        engine_ref=engine_ref, base_url=base_url, container=container
    )
    bench = {**fields["bench"], "baseline_engine_image_digest": identity["engine_ref"]}
    engine = fields["engine"]
    model = bench["model"]
    context = sampling_context_for_campaign(bench, engine)
    formatter = formatter or build_prompt_formatter(
        rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
    )
    fetcher = row_fetcher or (lambda i: fetch_hf_row(rule, i))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "qualification.jsonl"
    rule_path = output_dir / "sampling_rule.json"
    if evidence_path.exists() or rule_path.exists():
        raise SamplerError("use a fresh qualification output directory")
    qualified, seen = [], set()
    seed = hashlib.sha256(b"pareton-longform-qualification-v1").hexdigest()
    with evidence_path.open("x", encoding="utf-8") as evidence:
        evidence.write(
            json.dumps(
                {
                    "type": "contract",
                    "bench": bench,
                    "engine": engine,
                    "sampling_rule": rule,
                    "formatter": formatter.receipt,
                    "baseline_identity": identity,
                    "concurrency": concurrency,
                }
            )
            + "\n"
        )
        evidence.flush()
        candidates = {group: [] for group in quotas}
        indices = ordered_rows(rule, seed)[:max_rows]
        logger.info("Indexing %s source rows before GPU qualification", len(indices))
        for scanned, row_index in enumerate(indices, 1):
            row = fetcher(row_index)
            try:
                candidate = candidate_for_row(row_index, row, formatter, rule, context)
            except PromptRenderError:
                candidate = None
            if candidate is not None and candidate["input_ids_sha256"] not in seen:
                seen.add(candidate["input_ids_sha256"])
                candidates[candidate["input_length_group"]].append(candidate)
            if scanned % 500 == 0:
                logger.info("Indexed %s/%s source rows", scanned, len(indices))
        logger.info(
            "Eligible inputs by tier: %s",
            {g: len(rows) for g, rows in candidates.items()},
        )
        evidence_lock = threading.Lock()

        def qualify_candidate(candidate):
            row_index = candidate["row_index"]
            group = candidate["input_length_group"]
            # Preflight one already-rendered candidate, using the same request
            # builder as rounds. Round-wide tier quotas apply to the saved pool,
            # not this internal single-request capacity/tokenization check.
            trace = WorkloadTrace.from_dict(
                {
                    "schema_version": 1,
                    "meta": {"sampling": {"context": context}},
                    "requests": [
                        request_for_candidate(
                            candidate, rule, 0, generation_seed=f"{seed}:{row_index}"
                        )
                    ],
                }
            )
            validate_engine_workload(
                base_url,
                trace,
                engine_name=engine["name"],
                max_model_len=model["max_model_len"],
                evidence_dir=output_dir / "preflight" / str(row_index),
                verify_tokenizer=True,
            )
            accepted = True
            for rep in range(repetitions):
                sampling = trace.requests[0].sampling
                temperature = sampling.temperature
                # Test both ends of a range before accepting a source row.
                if "temperature_range" in rule and rep < 2:
                    temperature = rule["temperature_range"][rep]
                settings = {
                    "temperature": temperature,
                    "top_p": sampling.top_p,
                    "seed": 0,
                }
                response = post_json(
                    base_url,
                    "/v1/completions",
                    {
                        "prompt": candidate["prompt"],
                        "max_tokens": rule["max_tokens"],
                        **settings,
                        "ignore_eos": False,
                        "stream": False,
                    },
                    timeout=timeout,
                )
                result = evaluate_response(response, rule, candidate["input_tokens"])
                with evidence_lock:
                    evidence.write(
                        json.dumps(
                            {
                                "type": "response",
                                "row_index": row_index,
                                "repetition": rep,
                                "sampling": settings,
                                "input_ids_sha256": candidate["input_ids_sha256"],
                                "input_tokens": candidate["input_tokens"],
                                "input_length_group": group,
                                **result,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    evidence.flush()
                logger.info(
                    "Row %s tier=%s repetition=%s/%s tokens=%s result=%s",
                    row_index,
                    group,
                    rep + 1,
                    repetitions,
                    result["completion_tokens"],
                    result["rejection"] or "pass",
                )
                if result["rejection"]:
                    accepted = False
                    break
            return accepted

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Finish the scarce longest tier before spending GPU time on others.
            for group in reversed(quotas):
                pool = candidates[group]
                offset = 0
                while qualified_counts[group] < quotas[group]:
                    remaining = len(pool) - offset
                    if qualified_counts[group] + remaining < quotas[group]:
                        raise SamplerError(
                            f"input tiers: {group} cannot fill {quotas[group]} slots "
                            f"({qualified_counts[group]} qualified, {remaining} untried); "
                            "evidence saved, no launch rule written"
                        )
                    size = min(concurrency, quotas[group] - qualified_counts[group])
                    batch = pool[offset : offset + size]
                    offset += len(batch)
                    logger.info(
                        "Qualifying tier=%s batch=%s accepted=%s/%s",
                        group,
                        len(batch),
                        qualified_counts[group],
                        quotas[group],
                    )
                    # Consume results in source order, independent of completion timing.
                    for candidate, accepted in zip(
                        batch, executor.map(qualify_candidate, batch), strict=True
                    ):
                        if accepted:
                            qualified.append(candidate["row_index"])
                            qualified_counts[group] += 1
                            logger.info(
                                "Qualified row %s (%s/%s)",
                                candidate["row_index"],
                                len(qualified),
                                pool_size,
                            )
    if (
        verify_baseline_image(
            engine_ref=engine_ref, base_url=base_url, container=identity["container_id"]
        )
        != identity
    ):
        raise EngineError(
            "baseline container changed during qualification; no launch rule written"
        )
    rule["eligible_row_indices"] = sorted(qualified)
    rule["qualification"] = {
        "contract_sha256": qualification_contract(rule, bench, engine),
        "repetitions": repetitions,
    }
    rule = parse_sampling_rule(rule)
    rule_path.write_text(json.dumps(rule, indent=2) + "\n")
    summary = {
        "evidence_sha256": "sha256:"
        + hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "qualified_rows": len(qualified),
        "repetitions": repetitions,
        "min_output_tokens": rule["min_output_tokens"],
        "max_tokens": rule["max_tokens"],
        "ignore_eos": False,
        "enable_thinking": False,
        "sampling": {**generation_fields(rule), "top_p": 1.0},
        "sampling_rule_sha256": digest(rule),
        "qualified_rows_by_input_tier": qualified_counts,
        "concurrency": concurrency,
        "scope": "bounded-concurrency natural-output qualification; full concurrent GPU round still required",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rule


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Direct http://127.0.0.1:PORT endpoint on the local Linux Docker host",
    )
    parser.add_argument(
        "--container",
        required=True,
        help="Running baseline Docker container name or ID",
    )
    parser.add_argument(
        "--engine-ref",
        required=True,
        help="Published image digest running at that endpoint",
    )
    parser.add_argument(
        "--campaign-fields",
        type=Path,
        default=Path("fixtures/campaigns/sglang_qwen38_27b/campaign-fields.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pool-size",
        type=int,
        help="Default: twice n_prompts, split equally across four input tiers",
    )
    parser.add_argument("--max-rows", type=int, default=6000)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum simultaneous candidate requests (default: 4)",
    )
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fields = json.loads(args.campaign_fields.read_text())
    try:
        qualify(
            fields=fields,
            base_url=args.base_url,
            container=args.container,
            engine_ref=args.engine_ref,
            output_dir=args.output_dir,
            pool_size=args.pool_size,
            max_rows=args.max_rows,
            repetitions=args.repetitions,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )
    except (SamplerError, EngineError) as exc:
        parser.exit(1, f"qualification failed: {exc}\n")
    print(args.output_dir / "sampling_rule.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
