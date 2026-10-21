"""CPU audit for the pinned Exgentic v6 workload. Never writes campaign DB rows.

Writes a prepared campaign-fields.json for GPU qualification with
``python -m bench.qualify_longform --campaign-fields PREPARED --timeout 1200``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bench.exgentic import DATASET, REVISION, prompt_kv_tokens
from bench.longform import (
    RENDER_FAILED,
    digest,
    sampling_context_for_campaign,
    scan_candidates,
)
from bench.sampler import (
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
    scan = scan_candidates(
        range(len(dataset)), dataset.__getitem__, formatter, rule, context
    )
    for index, candidate in scan:
        if index and index % 5000 == 0:
            logger.info(
                "Indexed %s/%s snapshots; eligible=%s", index, len(dataset), counts
            )
        if candidate == RENDER_FAILED:
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
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        prepare(json.loads(args.campaign_fields.read_text()), args.output_dir)
    except SamplerError as exc:
        parser.exit(1, f"Exgentic CPU audit failed: {exc}\n")
    print(args.output_dir / "campaign-fields.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
