"""Qualify natural long responses on a trusted baseline before opening a campaign.

Run with --help. This writes local evidence and a sampling rule, never DB rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from bench.correctness import degeneracy_reason
from bench.http import post_json
from bench.lifecycle import EngineError
from bench.longform import (
    candidate_for_row,
    digest,
    ordered_rows,
    qualification_contract,
    sampling_context_for_campaign,
)
from bench.sampler import (
    LONGFORM_ALGO_VERSION,
    PromptRenderError,
    SamplerError,
    build_prompt_formatter,
    fetch_hf_row,
    generate_trace,
    parse_sampling_rule,
)
from bench.validate import validate_workload_trace_dict
from bench.workload_preflight import validate_engine_workload

logger = logging.getLogger(__name__)


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
    output_dir,
    pool_size=128,
    max_rows=6000,
    repetitions=2,
    timeout=600,
    row_fetcher=None,
    formatter=None,
):
    rule = parse_sampling_rule(fields["sampling_rule"])
    if rule["algo_version"] != LONGFORM_ALGO_VERSION:
        raise SamplerError("qualification requires algo_version 4")
    if repetitions < 2 or pool_size < rule["n_prompts"] or max_rows < pool_size:
        raise SamplerError(
            "qualification needs >=2 repetitions and max_rows >= pool_size >= n_prompts"
        )
    # Requalification starts from source, not a previous winning subset.
    rule.pop("qualification", None)
    rule.pop("eligible_row_indices", None)
    bench, engine = fields["bench"], fields["engine"]
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
                }
            )
            + "\n"
        )
        for row_index in ordered_rows(rule, seed)[:max_rows]:
            row = fetcher(row_index)
            try:
                candidate = candidate_for_row(row_index, row, formatter, rule, context)
            except PromptRenderError:
                continue
            if candidate is None or candidate["input_ids_sha256"] in seen:
                continue
            seen.add(candidate["input_ids_sha256"])
            # Reuse the real trace and engine preflight contracts, including
            # server tokenizer verification. No historical answer is sent.
            one_rule = {**rule, "n_prompts": 1, "eligible_row_indices": [row_index]}
            sampled = generate_trace(
                rule=one_rule,
                seed_hex=seed,
                row_fetcher=lambda _, row=row: row,
                prompt_formatter=formatter,
                sampling_context=context,
            )
            trace = validate_workload_trace_dict(json.loads(sampled.body))
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
                response = post_json(
                    base_url,
                    "/v1/completions",
                    {
                        "prompt": candidate["prompt"],
                        "max_tokens": rule["max_tokens"],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": 0,
                        "ignore_eos": False,
                        "stream": False,
                    },
                    timeout=timeout,
                )
                result = evaluate_response(response, rule, candidate["input_tokens"])
                evidence.write(
                    json.dumps(
                        {
                            "type": "response",
                            "row_index": row_index,
                            "repetition": rep,
                            "input_ids_sha256": candidate["input_ids_sha256"],
                            "input_tokens": candidate["input_tokens"],
                            **result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                evidence.flush()
                if result["rejection"]:
                    accepted = False
                    break
            if accepted:
                qualified.append(row_index)
                logger.info(
                    "Qualified row %s (%s/%s)", row_index, len(qualified), pool_size
                )
            if len(qualified) == pool_size:
                break
    if len(qualified) < pool_size:
        raise SamplerError(
            f"only {len(qualified)}/{pool_size} prompts qualified; evidence saved, no launch rule written"
        )
    rule["eligible_row_indices"] = sorted(qualified)
    rule["qualification"] = {
        "contract_sha256": qualification_contract(rule, bench, engine),
        "evidence_sha256": "sha256:"
        + hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "repetitions": repetitions,
    }
    rule = parse_sampling_rule(rule)
    rule_path.write_text(json.dumps(rule, indent=2) + "\n")
    summary = {
        "qualified_rows": len(qualified),
        "repetitions": repetitions,
        "min_output_tokens": rule["min_output_tokens"],
        "max_tokens": rule["max_tokens"],
        "ignore_eos": False,
        "enable_thinking": False,
        "sampling_rule_sha256": digest(rule),
        "scope": "sequential natural-output qualification; full concurrent GPU round still required",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rule


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Trusted running baseline endpoint; match the campaign's model and serve args",
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
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=6000)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fields = json.loads(args.campaign_fields.read_text())
    fields["bench"]["baseline_engine_image_digest"] = args.engine_ref
    try:
        qualify(
            fields=fields,
            base_url=args.base_url,
            output_dir=args.output_dir,
            pool_size=args.pool_size,
            max_rows=args.max_rows,
            repetitions=args.repetitions,
            timeout=args.timeout,
        )
    except (SamplerError, EngineError) as exc:
        parser.exit(1, f"qualification failed: {exc}\n")
    print(args.output_dir / "sampling_rule.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
