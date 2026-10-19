"""Enter the latest submissions of a closed campaign into its replacement.

Usage:
    PARETON_DATABASE_URL=... python -m campaign.carryover \
      --from-campaign OLD --to-campaign NEW --last 10 \
      --operator NAME --reason TEXT --evidence-ref URL
    # Review the plan, then repeat with --apply --plan-sha256 HASH.

Without --apply nothing is written. Each carried patch is copied under the new
campaign's private prefix and goes through the normal gate, build and rounds
there. Images, scores and payment references stay with the old campaign. The
source commit block is kept, so carried patches keep their on-chain order in
round cohorts and ties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Callable
from uuid import UUID, uuid5

from psycopg2.extras import RealDictCursor

from campaign.store import (
    CampaignHotkeyDisqualified,
    _campaign_hotkey_is_disqualified,
    insert_submission,
)
from db.connection import db_connection
from gate.integrity import check_integrity, patch_fingerprint_bytes
from gate.types import SubmissionState
from storage.s3 import copy_private_patch, fetch_patch_bytes, private_patch_key

# A patch is only valid against the tree and path policy it was written for.
MATCHING_FIELDS = ("baseline_repo", "baseline_commit", "allowed_paths", "denied_paths")
# Never ran, and the same patch would fail the same gate again. Round verdicts
# are not here: they were reached on the old workload.
SKIPPED_SOURCE_STATES = (SubmissionState.REJECTED, SubmissionState.REJECTED_DUPLICATE)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def plan_sha256(plan: list[dict[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(plan).encode()).hexdigest()


def _campaigns(cur, source_id: str, target_id: str) -> None:
    if source_id == target_id:
        raise ValueError("source and target campaigns must differ")
    cur.execute(
        "SELECT * FROM campaigns WHERE id = ANY(%s::uuid[])", ([source_id, target_id],)
    )
    rows = {str(row["id"]): row for row in cur.fetchall()}
    if set(rows) != {source_id, target_id}:
        raise ValueError("unknown campaign")
    if rows[source_id]["status"] not in ("closed", "archived"):
        raise ValueError("close the source campaign before carrying submissions")
    if rows[target_id]["status"] != "open":
        raise ValueError("target campaign must be open")
    for key in MATCHING_FIELDS:
        if _canonical(rows[source_id][key]) != _canonical(rows[target_id][key]):
            raise ValueError(f"campaigns differ in {key}")


def build_plan(
    source_id: str,
    target_id: str,
    last: int,
    *,
    fetcher: Callable[[str], bytes] = fetch_patch_bytes,
) -> list[dict[str, Any]]:
    """The latest ``last`` source submissions, oldest first, each kept or skipped."""
    source_id, target_id = str(UUID(source_id)), str(UUID(target_id))
    if type(last) is not int or last < 1:
        raise ValueError("--last must be a positive integer")
    with db_connection(readonly=True) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            _campaigns(cur, source_id, target_id)
            cur.execute(
                """
                SELECT s.id, s.hotkey, s.patch_hash, s.baseline_commit,
                       s.retrieval_url, s.commit_block, latest.state AS source_state
                FROM submissions s
                LEFT JOIN LATERAL (
                    SELECT e.state FROM submission_events e
                    WHERE e.submission_id = s.id
                    ORDER BY e.created_at DESC, e.id DESC
                    LIMIT 1
                ) latest ON true
                WHERE s.campaign_id = %s
                ORDER BY s.created_at DESC, s.id DESC
                LIMIT %s
                """,
                (source_id, last),
            )
            rows = list(reversed(cur.fetchall()))
            plan, fingerprints = [], {}
            for row in rows:
                item = {
                    "source_submission_id": str(row["id"]),
                    "hotkey": row["hotkey"],
                    "patch_hash": row["patch_hash"],
                    "baseline_commit": row["baseline_commit"],
                    "retrieval_url": row["retrieval_url"],
                    "commit_block": row["commit_block"],
                    "source_state": row["source_state"],
                    "patch_fingerprint": None,
                    "skip": None,
                }
                plan.append(item)
                if row["source_state"] in SKIPPED_SOURCE_STATES:
                    item["skip"] = f"source state {row['source_state']}"
                    continue
                for label, cid in (("source", source_id), ("target", target_id)):
                    if _campaign_hotkey_is_disqualified(cur, cid, row["hotkey"]):
                        item["skip"] = f"hotkey disqualified in {label} campaign"
                        break
                if item["skip"]:
                    continue
                cur.execute(
                    "SELECT 1 FROM submissions WHERE campaign_id = %s AND patch_hash = %s",
                    (target_id, row["patch_hash"]),
                )
                if cur.fetchone() is not None:
                    item["skip"] = "already in target campaign"
                    continue
                # Only private objects can be copied under the new campaign.
                if private_patch_key(row["retrieval_url"]) is None:
                    item["skip"] = "not a private patch locator"
                    continue
                result = check_integrity(
                    retrieval_url=row["retrieval_url"],
                    expected_patch_hash=row["patch_hash"],
                    hotkey=row["hotkey"],
                    campaign_id=source_id,
                    fetcher=fetcher,
                )
                if not result.ok:
                    item["skip"] = f"integrity: {result.reason}"
                    continue
                fingerprint = patch_fingerprint_bytes(result.evidence["patch_bytes"])
                item["patch_fingerprint"] = fingerprint
                if fingerprint in fingerprints:
                    item["skip"] = f"duplicate of {fingerprints[fingerprint]}"
                    continue
                fingerprints[fingerprint] = item["source_submission_id"]
                cur.execute(
                    """
                    SELECT 1 FROM submissions s
                    JOIN submission_events e ON e.submission_id = s.id
                    WHERE s.campaign_id = %s AND e.state = 'committed'
                      AND e.detail ->> 'patch_fingerprint' = %s
                    LIMIT 1
                    """,
                    (target_id, fingerprint),
                )
                if cur.fetchone() is not None:
                    item["skip"] = "duplicate already in target campaign"
    return plan


def apply_plan(
    plan: list[dict[str, Any]],
    source_id: str,
    target_id: str,
    *,
    operator: str,
    reason: str,
    evidence_ref: str,
    copier: Callable[..., str] | None = None,
) -> list[dict[str, Any]]:
    source_id, target_id = str(UUID(source_id)), str(UUID(target_id))
    copier = copier or copy_private_patch
    results = []
    for item in plan:
        if item["skip"]:
            continue
        source = item["source_submission_id"]
        url = copier(
            item["retrieval_url"],
            campaign_id=target_id,
            hotkey=item["hotkey"],
            # Deterministic, so a retried run reuses the same object.
            upload_id=str(uuid5(UUID(target_id), f"carryover:{source}")),
            patch_hash=item["patch_hash"],
        )
        try:
            sid = insert_submission(
                campaign_id=target_id,
                patch_hash=item["patch_hash"],
                hotkey=item["hotkey"],
                baseline_commit=item["baseline_commit"],
                retrieval_url=url,
                commit_block=item["commit_block"],
                patch_fingerprint=item["patch_fingerprint"],
                event_detail={
                    "source": "operator_carryover",
                    "source_campaign_id": source_id,
                    "source_submission_id": source,
                    "operator": operator,
                    "reason": reason,
                    "evidence_ref": evidence_ref,
                },
            )
        except CampaignHotkeyDisqualified:
            outcome = "skipped: hotkey disqualified in target campaign"
        else:
            outcome = f"entered {sid}" if sid else "skipped: already present"
        results.append({"source_submission_id": source, "outcome": outcome})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Carry the latest submissions of a closed campaign into a new one"
    )
    parser.add_argument("--from-campaign", required=True)
    parser.add_argument("--to-campaign", required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence-ref", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--plan-sha256", help="Required with --apply: the hash the dry run printed"
    )
    args = parser.parse_args(argv)
    if args.apply and not args.plan_sha256:
        parser.error("--apply requires --plan-sha256 from a reviewed dry run")
    try:
        plan = build_plan(args.from_campaign, args.to_campaign, args.last)
        digest = plan_sha256(plan)
        for item in plan:
            print(
                json.dumps(
                    {
                        k: item[k]
                        for k in (
                            "source_submission_id",
                            "hotkey",
                            "patch_hash",
                            "commit_block",
                            "source_state",
                            "skip",
                        )
                    }
                )
            )
        carried = sum(not item["skip"] for item in plan)
        print(f"plan: {carried} to carry, {len(plan) - carried} skipped")
        print(f"plan_sha256: {digest}")
        if not args.apply:
            print("dry run; nothing written")
            return 0
        if args.plan_sha256 != digest:
            raise ValueError("plan changed since the reviewed dry run")
        for result in apply_plan(
            plan,
            args.from_campaign,
            args.to_campaign,
            operator=args.operator,
            reason=args.reason,
            evidence_ref=args.evidence_ref,
        ):
            print(json.dumps(result))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
