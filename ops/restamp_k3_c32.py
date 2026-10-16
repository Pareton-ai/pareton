"""Move a qualified Kimi K3 C16 rule to the 32-prompt C32 contract without GPU work.

Qualification is a per-row property: each pooled row produced enough clean output
at both temperature endpoints. It does not depend on how many rows a round draws
or how many run at once. This keeps the qualified pool and repetitions unchanged
and only rewrites n_prompts, request_concurrency, max_baseline_prompt_drops,
request_timeout_s (600 to 1200 s) and the correctness prompt count, then
re-hashes the contract against the fixture.

It first proves the source rule was qualified against exactly the current K3
fixture apart from those fields. A pool of 32 rows (8 per tier) means every
round replays all of them in a seed-dependent order with new temperatures.

  PYTHONPATH=. python ops/restamp_k3_c32.py OLD_RULE.json NEW_RULE.json

``--apply-to CAMPAIGN_ID`` instead switches the open K3 campaign in place, so its
submissions, leader and fees carry over. It restamps the campaign's stored rule,
raises its correctness prompt count and recomputes manifest_hash in one
transaction, and refuses while the campaign has a pending or running round.
Without ``--yes`` it prints the change and writes nothing.

  PYTHONPATH=. python ops/restamp_k3_c32.py --apply-to CAMPAIGN_ID [--yes]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from bench.longform import qualification_contract, require_qualification
from bench.sampler import SamplerError, parse_sampling_rule

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "fixtures/campaigns/sglang_kimi_k3_b300/campaign-fields.json"
SOURCE_PINS = {
    "n_prompts": 16,
    "request_concurrency": 16,
    "max_baseline_prompt_drops": 4,
    "request_timeout_s": 600.0,
}
SOURCE_CORRECTNESS_PROMPTS = 16


def require(condition, message):
    if not condition:
        raise SamplerError(f"restamp refused: {message}")


def restamp(old_rule, fields):
    old = parse_sampling_rule(old_rule)
    require("qualification" in old, "source rule is not qualified")
    new = parse_sampling_rule(fields["sampling_rule"])
    require(
        "qualification" not in new and "eligible_row_indices" not in new,
        "target must be the unqualified campaign fixture",
    )
    bench, engine = fields["bench"], fields["engine"]
    source_bench = copy.deepcopy(bench)
    source_bench["correctness"]["num_prompts"] = SOURCE_CORRECTNESS_PROMPTS
    source_rule = {**new, **SOURCE_PINS}
    unqualified = {
        k: v
        for k, v in old.items()
        if k not in ("qualification", "eligible_row_indices")
    }
    require(
        unqualified == source_rule,
        "source rule differs from the C16 K3 contract beyond the C32 fields",
    )
    # The stored hash must match the fixture as it was, so nothing else drifted.
    require_qualification(old, source_bench, engine)
    pool = old["eligible_row_indices"]
    tiers = len(new["input_tiers"])
    require(
        len(pool) >= new["n_prompts"] and len(pool) % tiers == 0,
        f"pool of {len(pool)} rows cannot fill {new['n_prompts']} prompts",
    )
    new["eligible_row_indices"] = list(pool)
    new["qualification"] = {
        **old["qualification"],
        "contract_sha256": qualification_contract(new, bench, engine),
    }
    require_qualification(new, bench, engine)
    return new


def restamp_campaign_row(cur, campaign_id, fields):
    """Lock an open K3 campaign row and return (row, updated row) for C32."""
    from campaign.store import _parse_json_obj

    cur.execute("SELECT * FROM campaigns WHERE id = %s FOR UPDATE", (campaign_id,))
    row = cur.fetchone()
    require(row is not None, f"campaign {campaign_id} not found")
    row = dict(row)
    require(row["status"] == "open", f"campaign is {row['status']!r}, not open")
    cur.execute(
        "SELECT COUNT(*) AS n FROM rounds "
        "WHERE campaign_id = %s AND status IN ('pending', 'running')",
        (campaign_id,),
    )
    require(
        cur.fetchone()["n"] == 0,
        "a round is pending or running; stop pareton-round-worker and let it settle",
    )
    source_bench = copy.deepcopy(fields["bench"])
    source_bench["correctness"]["num_prompts"] = SOURCE_CORRECTNESS_PROMPTS
    require(
        _parse_json_obj(row["bench"]) == source_bench
        and _parse_json_obj(row["engine"]) == fields["engine"],
        "campaign bench or engine differs from the C16 K3 fixture",
    )
    new = restamp(_parse_json_obj(row["sampling_rule"]), fields)
    # A None hash makes the manifest recompute it, as set_campaign_scoring_rule does.
    return row, {
        **row,
        "sampling_rule": new,
        "bench": fields["bench"],
        "manifest_hash": None,
    }


def apply_to_campaign(campaign_id, fields, *, write):
    from psycopg2.extras import Json, RealDictCursor

    from campaign.store import _row_to_manifest
    from db.connection import db_connection

    with db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        old, row = restamp_campaign_row(cur, str(campaign_id), fields)
        manifest_hash = _row_to_manifest(row).manifest_hash
        print(f"manifest_hash {old['manifest_hash']} -> {manifest_hash}")
        if not write:
            conn.rollback()
            return row["sampling_rule"]
        cur.execute(
            "UPDATE campaigns SET sampling_rule = %s, bench = %s, "
            "manifest_hash = %s WHERE id = %s AND status = 'open'",
            (
                Json(row["sampling_rule"]),
                Json(row["bench"]),
                manifest_hash,
                str(campaign_id),
            ),
        )
        require(cur.rowcount == 1, "campaign row was not updated")
    return row["sampling_rule"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--apply-to", metavar="CAMPAIGN_ID")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    fields = json.loads(FIELDS.read_text())
    if args.apply_to:
        require(args.source is None, "--apply-to takes no rule files")
        new = apply_to_campaign(args.apply_to, fields, write=args.yes)
        print(
            f"{'Updated' if args.yes else 'Dry run, nothing written to'} campaign "
            f"{args.apply_to}: {new['n_prompts']} prompts at "
            f"C{new['request_concurrency']}, "
            f"{len(new['eligible_row_indices'])} qualified rows, contract "
            f"{new['qualification']['contract_sha256']}.",
            flush=True,
        )
        return
    require(
        args.source is not None and args.output is not None,
        "pass OLD_RULE NEW_RULE, or --apply-to CAMPAIGN_ID",
    )
    require(not args.output.exists(), "output file already exists")
    new = restamp(json.loads(args.source.read_text()), fields)
    args.output.write_text(json.dumps(new, indent=2) + "\n")
    print(
        f"Restamped {len(new['eligible_row_indices'])} qualified rows for "
        f"{new['n_prompts']} prompts at C{new['request_concurrency']}; "
        f"contract {new['qualification']['contract_sha256']}. "
        "No GPU requests made.",
        flush=True,
    )


if __name__ == "__main__":
    main()
