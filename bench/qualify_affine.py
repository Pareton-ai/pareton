"""Preview and qualify Affine corpus workloads on CPU.

`preview` draws one round with the campaign sampler and verifies exact receipt
replay. `qualify` scans a deterministic slice of the pinned corpus with the
campaign tokenizer, records token distributions and rejection reasons, and
writes a sampling rule bound to the exact campaign pins.

Neither command runs inference, measures output length or performance, or
writes DB rows. GPU qualification of the H100 campaign is a separate step.
Run with --help.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from uuid import UUID

from bench.affine_corpus import (
    AffineCorpus,
    CorpusIntegrityError,
    candidate_for_row,
    input_group,
    length_groups,
    ordered_turn_ids,
    qualification_contract,
    sampling_context_for_campaign,
    turn_messages,
)
from bench.longform import digest
from bench.sampler import (
    AFFINE_RULE_TYPE,
    PromptRenderError,
    SamplerError,
    build_prompt_formatter,
    compute_sample_seed,
    generate_trace,
    parse_sampling_rule,
)

logger = logging.getLogger(__name__)

DEFAULT_RULE = Path("fixtures/workloads/affine_corpus_e103/sampling_rule.json")
QUALIFICATION_SEED = hashlib.sha256(b"pareton-affine-qualification-v1").hexdigest()
DEMO_BLOCK_HASH = "c" * 64
DEMO_CAMPAIGN_ID = "00000000-0000-0000-0000-000000000001"


def _setup(fields, *, corpus=None, formatter=None):
    rule = parse_sampling_rule(fields["sampling_rule"])
    if rule["type"] != AFFINE_RULE_TYPE:
        raise SamplerError("this command requires an affine_corpus sampling rule")
    bench, engine = fields["bench"], fields["engine"]
    model = bench["model"]
    context = sampling_context_for_campaign(bench, engine)
    corpus = corpus or AffineCorpus(rule)
    formatter = formatter or build_prompt_formatter(
        rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
    )
    return rule, bench, engine, context, corpus, formatter


def _draw(rule, corpus, formatter, context, *, campaign_id, seed_block, block_hash):
    kwargs = {
        "rule": rule,
        "seed_hex": compute_sample_seed(block_hash=block_hash, campaign_id=campaign_id),
        "row_fetcher": corpus,
        "prompt_formatter": formatter,
        "sample_seed_block": seed_block,
        "sample_seed_block_hash": block_hash,
        "sampling_context": context,
    }
    sampled = generate_trace(**kwargs)
    rebuilt = generate_trace(**kwargs, sampling_receipt=sampled.receipt)
    if rebuilt.body != sampled.body or rebuilt.receipt != sampled.receipt:
        raise SamplerError("receipt replay changed the trace")
    return sampled


def preview(
    *,
    fields,
    output_dir,
    campaign_id=DEMO_CAMPAIGN_ID,
    seed_block=1,
    block_hash=DEMO_BLOCK_HASH,
    corpus=None,
    formatter=None,
):
    rule, _, _, context, corpus, formatter = _setup(
        fields, corpus=corpus, formatter=formatter
    )
    sampled = _draw(
        rule,
        corpus,
        formatter,
        context,
        campaign_id=str(campaign_id),
        seed_block=seed_block,
        block_hash=block_hash,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "workload_trace.json").write_bytes(sampled.body)
    (root / "sampling_receipt.json").write_text(
        json.dumps(sampled.receipt, indent=2, ensure_ascii=False) + "\n"
    )
    trace = json.loads(sampled.body)
    pool = corpus.eligible_rows()
    with (root / "index.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            (
                "request_id",
                "turn_id",
                "source",
                "action_kind",
                "phase",
                "input_tier",
                "input_tokens",
                "history_messages",
                "max_tokens",
            )
        )
        for request, source in zip(
            trace["requests"], sampled.receipt["requests"], strict=True
        ):
            name = request["id"]
            writer.writerow(
                (
                    name,
                    source["turn_id"],
                    source["source"],
                    source["action_kind"],
                    source["phase"],
                    request["input_length_group"],
                    request["input_tokens"],
                    source["history_messages"],
                    request["max_tokens"],
                )
            )
            (root / f"{name}.prompt.txt").write_text(request["prompt"])
            row = pool[source["turn_id"]]
            messages, _ = turn_messages(corpus.record(row), row)
            (root / f"{name}.messages.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2) + "\n"
            )
    return sampled


def _percentiles(values):
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p10": pick(0.1),
        "p50": pick(0.5),
        "p90": pick(0.9),
        "max": ordered[-1],
    }


def scan_turn(row, corpus, formatter, rule, context):
    """Evidence for one turn: its rendered size and why it is or is not usable."""
    entry = {
        "turn_id": row["turn_id"],
        "source": row["source"],
        "action_kind": row["action_kind"],
        "phase": row["phase"],
        "chunk_key": row["chunk_key"],
        "n_prefix_chars": row["n_prefix_chars"],
    }
    try:
        messages, _ = turn_messages(corpus.record(row), row)
        ids = formatter.encode(formatter.render(messages))
    except CorpusIntegrityError as exc:
        return {**entry, "rejection": "corpus_integrity", "detail": str(exc)}
    except PromptRenderError as exc:
        return {**entry, "rejection": "render_error", "detail": str(exc)}
    entry.update(input_tokens=len(ids), history_messages=len(messages))
    if input_group(len(ids)) is None:
        return {**entry, "rejection": "outside_input_tiers"}
    if candidate_for_row(row, corpus, formatter, rule, context) is None:
        return {**entry, "rejection": "exceeds_model_context"}
    return {**entry, "input_length_group": input_group(len(ids)), "rejection": None}


def qualify(*, fields, output_dir, scan_turns=2048, corpus=None, formatter=None):
    fields = {**fields, "sampling_rule": dict(fields["sampling_rule"])}
    # Requalification always starts from the source rule.
    fields["sampling_rule"].pop("qualification", None)
    rule, bench, engine, context, corpus, formatter = _setup(
        fields, corpus=corpus, formatter=formatter
    )
    if type(scan_turns) is not int or scan_turns < rule["n_prompts"]:
        raise SamplerError("scan_turns must be an integer >= n_prompts")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "qualification.jsonl"
    rule_path = output_dir / "sampling_rule.json"
    if evidence_path.exists() or rule_path.exists():
        raise SamplerError("use a fresh qualification output directory")

    manifest = corpus.manifest()
    pool = corpus.eligible_rows()
    corpus_summary = {
        "base_url": rule["base_url"],
        "manifest_sha256": rule["manifest_sha256"],
        "manifest_published_at": manifest.get("published_at"),
        "corpus_epoch": manifest.get("corpus_epoch"),
        "schema_version": manifest.get("schema_version"),
        "view_spec": manifest.get("view_spec"),
        "index_key": manifest["index"]["key"],
        "index_sha256": rule["index_sha256"],
        **corpus.index_stats,
        "eligible_before_tokenization": len(pool),
        "eligible_by_source": dict(
            sorted(Counter(r["source"] for r in pool.values()).items())
        ),
        "eligible_by_action_kind": dict(
            sorted(Counter(r["action_kind"] for r in pool.values()).items())
        ),
    }
    # The slice is chosen in seed order; it is scanned chunk by chunk so each
    # chunk is inflated once. Evidence lines follow the scan order.
    turn_ids = sorted(
        ordered_turn_ids(pool, QUALIFICATION_SEED)[:scan_turns],
        key=lambda t: (pool[t]["chunk_key"], pool[t]["traj_line"], t),
    )
    logger.info(
        "Scanning %s of %s eligible turns from manifest %s",
        len(turn_ids),
        len(pool),
        rule["manifest_sha256"],
    )
    results = []
    with evidence_path.open("x", encoding="utf-8") as evidence:
        evidence.write(
            json.dumps(
                {
                    "type": "contract",
                    "bench": bench,
                    "engine": engine,
                    "sampling_rule": rule,
                    "formatter": formatter.receipt,
                    "context": context,
                    "corpus": corpus_summary,
                    "scan_seed": QUALIFICATION_SEED,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for scanned, turn_id in enumerate(turn_ids, 1):
            result = scan_turn(pool[turn_id], corpus, formatter, rule, context)
            results.append(result)
            evidence.write(json.dumps({"type": "turn", **result}) + "\n")
            if scanned % 256 == 0:
                logger.info("Scanned %s/%s turns", scanned, len(turn_ids))

    tiers = [g["name"] for g in length_groups(4)]
    accepted = [r for r in results if r["rejection"] is None]
    eligible_by_tier = {
        t: sum(r["input_length_group"] == t for r in accepted) for t in tiers
    }
    tokens_by_tier = defaultdict(list)
    for r in accepted:
        tokens_by_tier[r["input_length_group"]].append(r["input_tokens"])
    summary = {
        "scope": "CPU input qualification of the pinned corpus; no inference, "
        "output length, correctness or performance is measured",
        "corpus": corpus_summary,
        "scanned_turns": len(results),
        "accepted_turns": len(accepted),
        "rejections": dict(Counter(r["rejection"] for r in results if r["rejection"])),
        "eligible_by_tier": eligible_by_tier,
        "input_tokens": {
            "all_rendered": _percentiles(
                [r["input_tokens"] for r in results if "input_tokens" in r]
            ),
            **{t: _percentiles(tokens_by_tier[t]) for t in tiers},
        },
        "accepted_by_source": dict(
            sorted(Counter(r["source"] for r in accepted).items())
        ),
        "accepted_by_action_kind": dict(
            sorted(Counter(r["action_kind"] for r in accepted).items())
        ),
        "evidence_sha256": "sha256:"
        + hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    }
    # Require headroom: four times each tier's per-round quota inside the scan.
    short = {t: n for t, n in eligible_by_tier.items() if n < rule["n_prompts"]}
    if short:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise SamplerError(
            f"insufficient eligible turns by tier within the scan: {short}; "
            "evidence saved, no launch rule written"
        )
    rule["qualification"] = {
        "contract_sha256": qualification_contract(rule, bench, engine),
        "scanned_turns": len(results),
        "eligible_by_tier": eligible_by_tier,
    }
    rule = parse_sampling_rule(rule)
    sampled = _draw(
        rule,
        corpus,
        formatter,
        context,
        campaign_id=DEMO_CAMPAIGN_ID,
        seed_block=1,
        block_hash=DEMO_BLOCK_HASH,
    )
    summary["demo_draw"] = {
        "campaign_id": DEMO_CAMPAIGN_ID,
        "block_hash": DEMO_BLOCK_HASH,
        "sampled_trace_sha256": sampled.sha256,
        "turn_ids": sampled.receipt["turn_ids"],
        "receipt_replay": "exact",
    }
    summary["sampling_rule_sha256"] = digest(rule)
    rule_path.write_text(json.dumps(rule, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rule


def _load_fields(args):
    fields = json.loads(args.campaign_fields.read_text())
    if args.sampling_rule:
        fields["sampling_rule"] = json.loads(args.sampling_rule.read_text())
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "qualify"):
        command = commands.add_parser(name)
        command.add_argument(
            "--campaign-fields",
            type=Path,
            required=True,
            help="Campaign fields JSON with bench, engine and sampling_rule",
        )
        command.add_argument(
            "--sampling-rule", type=Path, help="Override the fields' sampling_rule"
        )
        command.add_argument("--output-dir", type=Path, required=True)
    commands.choices["preview"].add_argument(
        "--campaign-id", type=UUID, default=UUID(DEMO_CAMPAIGN_ID)
    )
    commands.choices["preview"].add_argument("--seed-block", type=int, default=1)
    commands.choices["preview"].add_argument("--block-hash", default=DEMO_BLOCK_HASH)
    commands.choices["qualify"].add_argument("--scan-turns", type=int, default=2048)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fields = _load_fields(args)
    try:
        if args.command == "preview":
            preview(
                fields=fields,
                output_dir=args.output_dir,
                campaign_id=args.campaign_id,
                seed_block=args.seed_block,
                block_hash=args.block_hash,
            )
            print(
                f"CPU preview and exact receipt replay: {args.output_dir / 'index.tsv'}"
            )
        else:
            qualify(
                fields=fields, output_dir=args.output_dir, scan_turns=args.scan_turns
            )
            print(args.output_dir / "sampling_rule.json")
    except SamplerError as exc:
        parser.exit(1, f"{args.command} failed: {exc}\n")
    print("No inference ran; output length and GPU capacity are not established.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
