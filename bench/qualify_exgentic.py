"""Prepare and qualify pinned Exgentic snapshots. Never writes campaign DB rows."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bench.exgentic import DATASET, REVISION, prompt_kv_tokens, snapshot_from_row
from bench.lifecycle import EngineError
from bench.longform import candidate_for_row, digest, sampling_context_for_campaign
from bench.qualify_longform import qualify
from bench.sampler import (
    PromptRenderError,
    SamplerError,
    _cached_hf_split,
    build_prompt_formatter,
    generate_trace,
    parse_sampling_rule,
)

logger = logging.getLogger(__name__)

DEFAULT_FIELDS = Path(
    "fixtures/campaigns/sglang_kimi_k3_b300_exgentic/campaign-fields.json"
)


def prepare(fields, output_dir, *, dataset=None, formatter=None):
    """Materialize snapshots, pin their count, and audit actual target-token tiers."""
    fields = json.loads(json.dumps(fields))
    rule = fields["sampling_rule"]
    if rule.get("algo_version") != 6 or (rule.get("dataset"), rule.get("revision")) != (
        DATASET,
        REVISION,
    ):
        raise SamplerError("prepare requires the pinned Exgentic v6 campaign template")
    # n_rows=0 marks an unprepared template and is never a valid launch rule.
    if rule.get("config") != "default" or rule.get("split") != "train":
        raise SamplerError("Exgentic v6 requires default config and train split")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    dataset = _cached_hf_split(rule) if dataset is None else dataset
    rule.pop("qualification", None)
    rule.pop("eligible_row_indices", None)
    rule["n_rows"] = len(dataset)
    rule = parse_sampling_rule(rule)
    fields["sampling_rule"] = rule
    model = fields["bench"]["model"]
    formatter = formatter or build_prompt_formatter(
        rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
    )
    context = sampling_context_for_campaign(fields["bench"], fields["engine"])
    tiers = rule["input_tiers"]
    counts = dict.fromkeys(tiers, 0)
    sessions = {tier: set() for tier in tiers}
    hashes = set()
    render_failures = 0
    for index in range(len(dataset)):
        if index and index % 100 == 0:
            logger.info(
                "Indexed %s/%s snapshots; eligible=%s", index, len(dataset), counts
            )
        try:
            candidate = candidate_for_row(
                index, dataset[index], formatter, rule, context
            )
        except PromptRenderError:
            render_failures += 1
            continue
        if candidate is None or candidate["input_ids_sha256"] in hashes:
            continue
        hashes.add(candidate["input_ids_sha256"])
        tier = candidate["input_length_group"]
        counts[tier] += 1
        sessions[tier].add(candidate["session_id"])
    audit = {
        "dataset": DATASET,
        "revision": REVISION,
        "normalized_snapshots": len(dataset),
        "formatter": formatter.receipt,
        "followup_prompt_sha256": digest(rule["followup_prompt"]),
        "input_length_basis": "complete rendered snapshot plus follow-up and tool definitions",
        "render_failures": render_failures,
        "eligible_snapshots_by_tier": counts,
        "distinct_sessions_by_tier": {
            tier: len(value) for tier, value in sessions.items()
        },
        "scope": "CPU input coverage only; no GPU qualification",
    }
    # Written before the preview so a sampler failure still leaves the counts.
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (root / "campaign-fields.json").write_text(json.dumps(fields, indent=2) + "\n")
    # Preview the actual round sampler and prove receipt reconstruction.
    kwargs = dict(
        rule=rule,
        seed_hex="0" * 64,
        row_fetcher=lambda i: dataset[i],
        prompt_formatter=formatter,
        sampling_context=context,
    )
    sample = generate_trace(**kwargs)
    if generate_trace(**kwargs, sampling_receipt=sample.receipt).body != sample.body:
        raise SamplerError("Exgentic CPU receipt replay changed trace bytes")
    prompts = [r["prompt"] for r in json.loads(sample.body)["requests"]]
    audit["preview_round_kv"] = prompt_kv_tokens(formatter.encode(p) for p in prompts)
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (root / "workload_trace.json").write_bytes(sample.body)
    (root / "sampling_receipt.json").write_text(
        json.dumps(sample.receipt, indent=2) + "\n"
    )
    for index, row_index in enumerate(sample.row_indices):
        (root / f"hf-{index:03d}.snapshot.json").write_text(
            json.dumps(
                snapshot_from_row(dataset[row_index]), ensure_ascii=False, indent=2
            )
            + "\n"
        )
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="CPU coverage audit and exact replay preview",
    )
    parser.add_argument(
        "--base-url", help="Direct http://127.0.0.1:PORT endpoint on the Docker host"
    )
    parser.add_argument("--container")
    parser.add_argument(
        "--engine-ref", help="Published baseline image reference by digest"
    )
    parser.add_argument(
        "--pool-size", type=int, help="Default twice n_prompts, equally divided by tier"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Snapshot scan bound; default all normalized snapshots",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Maximum simultaneous candidates; defaults to campaign request_concurrency",
    )
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args(argv)
    if not args.prepare_only and not all(
        (args.base_url, args.container, args.engine_ref)
    ):
        parser.error("qualification requires --base-url, --container and --engine-ref")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        fields = json.loads(args.campaign_fields.read_text())
        if fields["sampling_rule"].get("algo_version") != 6:
            raise SamplerError("qualify_exgentic requires sampler v6")
        if args.engine_ref:
            fields["base_image_digest"] = args.engine_ref
            fields["bench"]["baseline_engine_image_digest"] = args.engine_ref
        if args.prepare_only:
            prepare(fields, args.output_dir)
            print(
                f"CPU coverage and replay evidence: {args.output_dir}; GPU qualification still required"
            )
            return 0
        if fields["sampling_rule"].get("n_rows", 0) < 1:
            raise SamplerError(
                "run --prepare-only first, then pass its campaign-fields.json"
            )
        result = qualify(
            fields=fields,
            base_url=args.base_url,
            container=args.container,
            engine_ref=args.engine_ref,
            output_dir=args.output_dir,
            pool_size=args.pool_size,
            max_rows=(
                args.max_rows
                if args.max_rows is not None
                else fields["sampling_rule"]["n_rows"]
            ),
            repetitions=args.repetitions,
            concurrency=(
                args.concurrency
                if args.concurrency is not None
                else fields["sampling_rule"]["request_concurrency"]
            ),
            timeout=args.timeout,
        )
        fields["sampling_rule"] = result
        (args.output_dir / "campaign-fields.json").write_text(
            json.dumps(fields, indent=2) + "\n"
        )
    except (SamplerError, EngineError) as exc:
        parser.exit(1, f"Exgentic qualification failed: {exc}\n")
    print(args.output_dir / "sampling_rule.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
