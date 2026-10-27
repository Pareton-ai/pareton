"""Manually run the review gates (d1 obfuscation, d2 LLM) on one saved patch.

Runs what the worker runs after the surface check: base apply against the
campaign baseline, the comment-stripped source view, the obfuscation scan and
the LLM review, and shows each step. Reads PARETON_LLM_REVIEW_* like the
worker does. It uses a throwaway cache directory unless --cache-dir is given,
so it never touches the worker's verdict cache.

The baseline comes from --campaign-id (read from the database), from
--baseline-repo and --baseline-commit, or from --repo-dir (a checkout whose
HEAD is the baseline).

    python scripts/llm_review_manual.py --patch-file PATCH.diff \\
        --patch-hash sha256:... --campaign-id UUID --model kimi-k3
    python scripts/llm_review_manual.py --probe-response-format --model kimi-k3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from gate import llm_review  # noqa: E402
from gate.base_apply import check_base_apply, check_base_apply_local  # noqa: E402
from gate.comment_strip import NO_COMMENTS, STRIPPED, strip_comments  # noqa: E402
from gate.integrity import hash_patch_bytes  # noqa: E402
from gate.obfuscation import check_obfuscation  # noqa: E402
from gate.source_view import SourceViewError, build_source_view  # noqa: E402

PREVIEW_CHARS = 2000


def _step(n: int, title: str) -> None:
    print(f"\n== {n}. {title}")


def _ok(cond: bool, label: str) -> bool:
    print(f"  [{'OK' if cond else 'FAIL'}] {label}")
    return cond


def review(args: argparse.Namespace) -> int:
    ok = True
    spec = llm_review.validate_llm_review({"policy": args.policy, "model": args.model})

    _step(1, "Config")
    print(f"  base_url        = {config.LLM_REVIEW_BASE_URL or '(unset)'}")
    print(f"  api_key         = {'set' if config.LLM_REVIEW_API_KEY else '(unset)'}")
    print(f"  model / policy  = {spec['model']} / {spec['policy']}")
    print(f"  response_format = {config.LLM_REVIEW_RESPONSE_FORMAT}")
    print(f"  timeout_s       = {config.LLM_REVIEW_TIMEOUT_S}")
    print(f"  max_output_tok  = {config.LLM_REVIEW_MAX_OUTPUT_TOKENS}")
    ok &= _ok(
        bool(config.LLM_REVIEW_BASE_URL and config.LLM_REVIEW_API_KEY),
        "endpoint and key are set",
    )

    _step(2, "Patch integrity")
    patch_bytes = Path(args.patch_file).read_bytes()
    actual = hash_patch_bytes(patch_bytes)
    patch_hash = args.patch_hash or actual
    print(f"  file  = {args.patch_file} ({len(patch_bytes)} bytes)")
    print(f"  hash  = {actual}")
    ok &= _ok(actual == patch_hash, "file hash matches --patch-hash")
    try:
        patch_text = patch_bytes.decode("utf-8")
    except UnicodeDecodeError:
        _ok(False, "patch is UTF-8 (the gate would reject it)")
        return 1
    _ok(True, "patch is UTF-8")

    _step(3, "Baseline checkout and apply (gate c)")
    work = args.work_dir
    if args.repo_dir:
        apply_res = check_base_apply_local(
            repo_dir=Path(args.repo_dir), patch_bytes=patch_bytes
        )
    else:
        repo, commit = _baseline(args)
        print(f"  baseline = {repo} @ {commit}")
        apply_res = check_base_apply(
            baseline_repo=repo,
            baseline_commit=commit,
            patch_bytes=patch_bytes,
            work_root=work,
        )
    if not _ok(apply_res.ok, f"patch applies to the baseline ({apply_res.reason})"):
        print(json.dumps(apply_res.evidence, indent=2, default=str)[:PREVIEW_CHARS])
        return 1

    _step(4, "Comment stripping (whole files, verified)")
    try:
        view = build_source_view(Path(apply_res.evidence["repo_dir"]), patch_bytes)
    except SourceViewError as exc:
        _ok(False, f"source view failed: {exc} (the worker would reject)")
        return 1
    print(f"  {'file':60} {'baseline':>16} {'patched':>16}")
    for f in view.files:
        cells = []
        for res in (f.pre_strip, f.post_strip):
            cells.append("-" if res is None else f"{res.status}:{res.comments}")
        print(f"  {f.path[-60:]:60} {cells[0]:>16} {cells[1]:>16}")
    print(f"  summary = {json.dumps(view.strip_summary())}")
    stripped_ok = True
    for f in view.files:
        for text, res in ((f.pre, f.pre_strip), (f.post, f.post_strip)):
            if res is None or text is None:
                continue
            stripped_ok &= res.text.count("\n") == text.count("\n")
            if res.status == STRIPPED:
                # Stripping the result again must find nothing left.
                stripped_ok &= strip_comments(f.path, res.text).status == NO_COMMENTS
    ok &= _ok(stripped_ok, "line counts kept and no comments left in stripped files")
    kept = view.strip_summary()["comments_kept"]
    _ok(
        not kept,
        "every supported file was stripped"
        if not kept
        else f"{len(kept)} file(s) shown with comments, marked with '# note:'",
    )

    _step(5, "Obfuscation scan (gate d1, every campaign)")
    obf = check_obfuscation(patch_bytes=patch_bytes, post_texts=view.post_texts)
    if obf.ok:
        print("  [OK] no obfuscation findings")
    else:
        print(f"  [INFO] d1 rejects: {obf.reason}, rules={obf.evidence.get('rules')}")
        for f in obf.evidence.get("findings", [])[:20]:
            print(f"    {f['file']}:{f['line']} {f['rule']}: {f['snippet']}")
        print("  (the LLM review below still runs, for comparison)")

    _step(6, "What the reviewer sees")
    review_text = view.review_diff
    hunks = llm_review._diff_hunks(review_text)
    n_hunks = sum(len(h) for h in hunks.values())
    print(f"  files = {len(hunks)}, hunks = {n_hunks}")
    for path, headers in sorted(hunks.items())[:20]:
        print(f"    {path}: {len(headers)} hunk(s)")
    if len(hunks) > 20:
        print(f"    ... {len(hunks) - 20} more files")
    print(f"  review diff ({len(review_text)} chars, raw patch {len(patch_text)}):")
    print("    " + review_text[:PREVIEW_CHARS].replace("\n", "\n    "))
    messages = llm_review.build_messages(spec["policy"], patch_hash, review_text)
    user = messages[1]["content"]
    tag = user.split("<", 1)[1].split(">", 1)[0]
    print(f"  fence tag = {tag}")
    print(
        f"  prompt chars = {sum(len(m['content']) for m in messages)}"
        f" (~{sum(len(m['content']) for m in messages) // 4} tokens)"
    )
    ok &= _ok(
        [m["role"] for m in messages] == ["system", "user"], "two messages, no tools"
    )
    ok &= _ok(review_text in user, "stripped review diff is in the prompt")
    ok &= _ok(tag not in review_text, "review diff does not contain the fence tag")
    ok &= _ok(
        len(review_text) <= config.LLM_REVIEW_MAX_INPUT_CHARS,
        f"within MAX_INPUT_CHARS ({config.LLM_REVIEW_MAX_INPUT_CHARS})",
    )

    _step(7, "Model call through the gate")
    cache_dir = Path(args.cache_dir or tempfile.mkdtemp(prefix="llm-review-manual-"))
    print(f"  cache_dir = {cache_dir}")
    raw: dict = {}
    try:
        real = llm_review.openai_chat_client(spec["model"])
    except llm_review.LLMReviewError as exc:
        real = None
        print(f"  client error: {exc}")

    def capture(msgs):
        start = time.monotonic()
        try:
            raw["content"] = real(msgs)
        except llm_review.LLMReviewError as exc:
            raw["error"] = str(exc)
            raise
        finally:
            raw["seconds"] = round(time.monotonic() - start, 1)
        return raw["content"]

    result = llm_review.check_llm_review(
        review_text=review_text,
        patch_hash=patch_hash,
        spec=spec,
        strip_summary=view.strip_summary(),
        client=capture if real else None,
        cache_dir=cache_dir,
    )
    if "seconds" in raw:
        print(f"  latency = {raw['seconds']}s")
    if "error" in raw:
        print(f"  call error = {raw['error']}")
    if "content" in raw:
        content = raw["content"]
        print(f"  raw response ({len(content)} chars):")
        print("    " + content[:PREVIEW_CHARS].replace("\n", "\n    "))
        try:
            json.loads(content)
            print("  [INFO] raw response is bare JSON")
        except ValueError:
            print(
                "  [INFO] raw response is not bare JSON; the gate unwraps a "
                "leading <think> block or ``` fence, see step 8"
            )

    _step(8, "Verdict validation")
    if result.evidence.get("retryable"):
        _ok(False, "provider unavailable; the worker would defer and retry")
    elif result.ok or result.reason == "llm_review_rejected":
        ok &= _ok(True, "verdict passed schema and hunk checks")
    else:
        ok &= _ok(False, f"gate failed closed: {result.reason}")

    _step(9, "Cache")
    again = llm_review.check_llm_review(
        review_text=review_text,
        patch_hash=patch_hash,
        spec=spec,
        client=_no_call,
        cache_dir=cache_dir,
    )
    if result.evidence.get("retryable") or not (
        result.ok or result.reason == "llm_review_rejected"
    ):
        _ok(True, "nothing cached for an outage or failure (expected)")
    else:
        ok &= _ok(
            again.evidence.get("cached") is True and again.ok == result.ok,
            "second run is served from cache with the same outcome",
        )

    _step(10, "Gate result (d1 then d2, as the worker runs them)")
    final = obf if not obf.ok else result
    outcome = (
        "DEFER" if final.evidence.get("retryable") else "PASS" if final.ok else "REJECT"
    )
    print(f"  d1 obfuscation: {'pass' if obf.ok else obf.reason}")
    print(f"  d2 llm review:  {'pass' if result.ok else result.reason}")
    print(f"  OUTCOME: {outcome}")
    print(
        json.dumps(
            {"ok": final.ok, "reason": final.reason, **final.evidence},
            indent=2,
            default=str,
        )
    )
    print(f"\nchecks {'all passed' if ok else 'had failures'}")
    return 0 if ok else 1


def _baseline(args: argparse.Namespace) -> tuple[str, str]:
    if args.baseline_repo and args.baseline_commit:
        return args.baseline_repo, args.baseline_commit
    if not args.campaign_id:
        raise SystemExit(
            "give --campaign-id, --baseline-repo with --baseline-commit, or --repo-dir"
        )
    from campaign.store import get_campaign

    campaign = get_campaign(args.campaign_id)
    if campaign is None:
        raise SystemExit(f"campaign {args.campaign_id} not found")
    return campaign.baseline_repo, campaign.baseline_commit


def _no_call(_messages):
    raise llm_review.LLMReviewError("llm_review_unavailable", "cache miss")


def probe(args: argparse.Namespace) -> int:
    """Ask for non-JSON and see whether each response_format forces JSON."""
    messages = [{"role": "user", "content": "Reply with the single word hello."}]
    for mode in ("off", "json_object", "json_schema"):
        config.LLM_REVIEW_RESPONSE_FORMAT = mode
        try:
            content = llm_review.openai_chat_client(args.model)(messages)
        except llm_review.LLMReviewError as exc:
            print(f"{mode:12} error: {exc}")
            continue
        try:
            data = json.loads(content)
            shape = (
                "matches verdict schema"
                if isinstance(data, dict)
                and set(data) == {"verdict", "reason", "offending_hunks"}
                else "JSON, other shape"
            )
        except ValueError:
            shape = "not JSON"
        print(f"{mode:12} {shape}: {content[:200]!r}")
    print(
        "\njson_schema is enforced only if its row says 'matches verdict schema'."
        "\nIf it says 'not JSON', the provider ignored response_format."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--patch-file")
    p.add_argument("--patch-hash", help="expected sha256:...; defaults to the file's")
    p.add_argument("--model", required=True)
    p.add_argument("--policy", default=llm_review.POLICY_DTYPE_PRECISION_V1)
    p.add_argument("--campaign-id", help="read the baseline from this campaign")
    p.add_argument("--baseline-repo")
    p.add_argument("--baseline-commit")
    p.add_argument("--repo-dir", help="a checkout whose HEAD is the baseline")
    p.add_argument("--cache-dir")
    p.add_argument("--probe-response-format", action="store_true")
    args = p.parse_args(argv)
    if args.probe_response_format:
        return probe(args)
    if not args.patch_file:
        p.error("--patch-file is required unless --probe-response-format")
    # The baseline clone holds a copy of the (private) patch; always remove it.
    args.work_dir = Path(tempfile.mkdtemp(prefix="llm-review-manual-apply-"))
    try:
        return review(args)
    finally:
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
