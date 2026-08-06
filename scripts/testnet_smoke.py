"""Helpers for the nightly Bittensor testnet smoke workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from campaign.manifest import build_manifest
from campaign.models import SLA
from campaign.store import (
    get_submission,
    insert_campaign,
    insert_profile,
    list_events,
)
from db.connection import db_connection

BASELINE_REPO = "https://github.com/vllm-project/vllm.git"
BASELINE_COMMIT = "ee0da84ab9e04ac7610e28580af62c365e898389"
PROFILE_NAME = "e2e-testnet-smoke"
PLACEHOLDER_IMAGE_DIGEST = "sha256:" + ("d" * 64)
EXPECTED_EVENT_SEQUENCE = [
    "committed",
    "fetched",
    "verified",
    "applied",
    "surface_ok",
    "built",
]
_SAFE_RUN_ID = re.compile(r"[^a-zA-Z0-9._-]+")


def _require_test_runtime() -> None:
    """Refuse to run the smoke helper against an unmarked database."""
    database_url = (os.environ.get("PARETON_DATABASE_URL") or "").strip()
    test_database_url = (os.environ.get("PARETON_TEST_DATABASE_URL") or "").strip()
    if not database_url or not test_database_url:
        raise RuntimeError(
            "PARETON_DATABASE_URL and PARETON_TEST_DATABASE_URL are required"
        )
    if database_url != test_database_url:
        raise RuntimeError(
            "PARETON_DATABASE_URL must equal PARETON_TEST_DATABASE_URL "
            "for the live smoke"
        )
    if os.environ.get("PARETON_NETWORK") != "test":
        raise RuntimeError("PARETON_NETWORK must be test")
    if os.environ.get("PARETON_NETUID") != "543":
        raise RuntimeError("PARETON_NETUID must be 543")


def check_database() -> None:
    """Check that the test branch has the tables used by the smoke."""
    _require_test_runtime()
    required = (
        "public.profiles",
        "public.campaigns",
        "public.submissions",
        "public.submission_events",
        "public.submission_jobs",
    )
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT " + ", ".join(["to_regclass(%s)"] * len(required)),
            required,
        )
        present = cur.fetchone()
    missing = [name for name, value in zip(required, present) if value is None]
    if missing:
        raise RuntimeError(f"test database schema missing: {', '.join(missing)}")
    print("test database schema ready")


def _find_open_smoke_campaign() -> str | None:
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id
            FROM campaigns c
            JOIN profiles p ON p.id = c.profile_id
            WHERE p.name = %s
              AND c.status = 'open'
              AND c.bench IS NULL
              AND c.baseline_commit = %s
              AND now() BETWEEN c.window_opens_at AND c.window_closes_at
            ORDER BY c.created_at DESC
            LIMIT 1
            """,
            (PROFILE_NAME, BASELINE_COMMIT),
        )
        row = cur.fetchone()
    return str(row[0]) if row else None


def ensure_campaign() -> str:
    """Return the reusable no-bench campaign, creating it when absent."""
    _require_test_runtime()
    existing = _find_open_smoke_campaign()
    if existing:
        return existing

    profile_id = insert_profile(
        PROFILE_NAME,
        {
            "fixture": True,
            "network": "test",
            "netuid": 543,
            "purpose": "nightly commitment and watcher smoke",
        },
    )
    trace_path = (
        REPO_ROOT / "fixtures" / "campaigns" / "synthetic_v0" / "workload_trace.json"
    )
    trace_hash = "sha256:" + hashlib.sha256(trace_path.read_bytes()).hexdigest()
    manifest = build_manifest(
        campaign_id=None,
        profile_id=profile_id,
        baseline_repo=BASELINE_REPO,
        baseline_commit=BASELINE_COMMIT,
        base_image_digest=PLACEHOLDER_IMAGE_DIGEST,
        gpu_skus=["TEST-MOCK"],
        workload_trace_sha256=trace_hash,
        workload_trace_url="http://127.0.0.1:9000/trace.json",
        sla=SLA(quality_floor_spec="testnet smoke only"),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**", ".github/**", "**/Dockerfile*"],
        window_opens_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_closes_at=datetime(2035, 1, 1, tzinfo=timezone.utc),
        priority_metric="throughput",
        success_threshold="mock build reaches built",
        status="open",
        customer_signoff=None,
        bench=None,
    )
    return str(insert_campaign(manifest))


def _safe_run_id(value: str) -> str:
    cleaned = _SAFE_RUN_ID.sub("-", value.strip()).strip("-")
    if not cleaned:
        raise ValueError("run id has no safe characters")
    return cleaned[:80]


def build_unique_patch(run_id: str) -> bytes:
    """Create an allowed, run-unique patch that adds one inert text file."""
    slug = _safe_run_id(run_id)
    path = f"vllm/pareton_testnet_smoke/{slug}.txt"
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1 @@\n"
        f"+pareton testnet smoke run {slug}\n"
    ).encode()


def prepare_patch(
    *,
    campaign_id: str,
    hotkey: str,
    run_id: str,
    http_root: Path,
    public_base_url: str,
    s3_prefix: str,
) -> dict[str, str]:
    """Write a locally served patch and return its commitment metadata."""
    patch = build_unique_patch(run_id)
    patch_hash = "sha256:" + hashlib.sha256(patch).hexdigest()
    relative = (
        Path(s3_prefix.strip("/"))
        / "campaigns"
        / campaign_id
        / "patches"
        / hotkey
        / f"{_safe_run_id(run_id)}.diff"
    )
    patch_path = http_root / relative
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(patch)
    retrieval_url = f"{public_base_url.rstrip('/')}/{relative.as_posix()}"
    return {
        "campaign_id": campaign_id,
        "patch_hash": patch_hash,
        "patch_path": str(patch_path),
        "retrieval_url": retrieval_url,
    }


def _write_github_outputs(values: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as output:
        output.writelines(f"{key}={value}\n" for key, value in values.items())


def restore_wallet(
    *,
    wallet_name: str,
    wallet_hotkey: str,
    expected_hotkey: str,
    wallet_path: Path | None,
) -> None:
    """Restore the CI hotkey without persisting or printing its secret."""
    material = (os.environ.get("CI_TESTNET_WALLET_SEED") or "").strip()
    if not material:
        raise RuntimeError("CI_TESTNET_WALLET_SEED is required")

    import bittensor as bt

    path = wallet_path or Path.home() / ".bittensor" / "wallets"
    wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey, path=str(path))
    restore: dict[str, str]
    if " " in material:
        restore = {"mnemonic": material}
    else:
        restore = {"seed": material}
    wallet.regenerate_hotkey(
        **restore,
        use_password=False,
        overwrite=True,
        suppress=True,
    )
    actual = wallet.hotkey.ss58_address
    if actual != expected_hotkey:
        raise RuntimeError(
            f"restored hotkey {actual} does not match expected {expected_hotkey}"
        )
    print(f"restored CI hotkey {actual}")


def _ordered_states(submission_id: str) -> list[str]:
    return [str(event["state"]) for event in list_events(submission_id)]


def _print_events(submission_id: str) -> None:
    rows = list_events(submission_id)
    print("ordered submission_events:")
    for row in rows:
        detail = row.get("detail") or {}
        print(f"- {row['state']}: {json.dumps(detail, sort_keys=True, default=str)}")


def _assert_built(submission: dict[str, Any]) -> None:
    submission_id = str(submission["id"])
    states = _ordered_states(submission_id)
    if states != EXPECTED_EVENT_SEQUENCE:
        raise RuntimeError(
            f"unexpected event sequence: {states!r}; "
            f"expected {EXPECTED_EVENT_SEQUENCE!r}"
        )
    if not submission.get("engine_image_ref"):
        raise RuntimeError("built submission has no engine_image_ref")
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT kind, status
            FROM submission_jobs
            WHERE submission_id = %s
            ORDER BY kind
            """,
            (submission_id,),
        )
        jobs = cur.fetchall()
    if jobs != [("gates", "done")]:
        raise RuntimeError(f"unexpected jobs after mock build: {jobs!r}")


def poll_for_built(patch_hash: str, *, timeout_s: float, interval_s: float) -> int:
    """Poll the test DB. Return 78 when the chain never yields a submission."""
    _require_test_runtime()
    deadline = time.monotonic() + timeout_s
    seen_submission: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        seen_submission = get_submission(patch_hash)
        if seen_submission is not None:
            states = _ordered_states(str(seen_submission["id"]))
            if "rejected" in states:
                _print_events(str(seen_submission["id"]))
                return 1
            if states and states[-1] == "built":
                _assert_built(seen_submission)
                _print_events(str(seen_submission["id"]))
                print(f"testnet smoke built patch {patch_hash}")
                return 0
        time.sleep(interval_s)

    if seen_submission is None:
        print(
            f"no submission observed for {patch_hash} before timeout; "
            "testnet reveal or RPC may be unavailable",
            file=sys.stderr,
        )
        return 78
    _print_events(str(seen_submission["id"]))
    print(
        f"submission did not reach built before timeout: {patch_hash}", file=sys.stderr
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-db")

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--hotkey", required=True)
    prepare.add_argument("--http-root", required=True, type=Path)
    prepare.add_argument("--public-base-url", default="http://127.0.0.1:9000")
    prepare.add_argument("--s3-prefix", default="stage0")

    restore = sub.add_parser("restore-wallet")
    restore.add_argument("--wallet-name", required=True)
    restore.add_argument("--wallet-hotkey", required=True)
    restore.add_argument("--expected-hotkey", required=True)
    restore.add_argument("--wallet-path", type=Path)

    poll = sub.add_parser("poll")
    poll.add_argument("--patch-hash", required=True)
    poll.add_argument("--timeout-s", type=float, default=900)
    poll.add_argument("--interval-s", type=float, default=10)

    args = parser.parse_args(argv)
    if args.command == "check-db":
        check_database()
        return 0
    if args.command == "prepare":
        campaign_id = ensure_campaign()
        values = prepare_patch(
            campaign_id=campaign_id,
            hotkey=args.hotkey,
            run_id=args.run_id,
            http_root=args.http_root,
            public_base_url=args.public_base_url,
            s3_prefix=args.s3_prefix,
        )
        _write_github_outputs(values)
        print(json.dumps(values, sort_keys=True))
        return 0
    if args.command == "restore-wallet":
        restore_wallet(
            wallet_name=args.wallet_name,
            wallet_hotkey=args.wallet_hotkey,
            expected_hotkey=args.expected_hotkey,
            wallet_path=args.wallet_path,
        )
        return 0
    if args.command == "poll":
        return poll_for_built(
            args.patch_hash,
            timeout_s=args.timeout_s,
            interval_s=args.interval_s,
        )
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
