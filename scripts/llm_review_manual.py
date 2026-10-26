"""Manually run the LLM review gate (d2) on one saved patch and show each step.

Reads the endpoint and key from PARETON_LLM_REVIEW_* like the worker does. It
uses a throwaway cache directory unless --cache-dir is given, so it never
touches the worker's verdict cache.

    python scripts/llm_review_manual.py --patch-file PATCH.diff \\
        --patch-hash sha256:... --model kimi-k3
    python scripts/llm_review_manual.py --probe-response-format --model kimi-k3
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from gate import llm_review  # noqa: E402
from gate.integrity import hash_patch_bytes  # noqa: E402
from gate.obfuscation import check_obfuscation  # noqa: E402

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

    _step(3, "Obfuscation scan (d1, runs before the LLM)")
    obf = check_obfuscation(patch_bytes=patch_bytes)
    if obf.ok:
        print("  [OK] no obfuscation findings")
    else:
        print(
            f"  [INFO] d1 would reject: {obf.reason}, rules={obf.evidence.get('rules')}"
        )
        for f in obf.evidence.get("findings", [])[:20]:
            print(f"    {f['file']}:{f['line']} {f['rule']}: {f['snippet']}")
        print("  (the LLM review below still runs, for comparison)")

    _step(4, "What the reviewer sees")
    hunks = llm_review._diff_hunks(patch_text)
    n_hunks = sum(len(h) for h in hunks.values())
    print(f"  files = {len(hunks)}, hunks = {n_hunks}")
    for path, headers in sorted(hunks.items())[:20]:
        print(f"    {path}: {len(headers)} hunk(s)")
    if len(hunks) > 20:
        print(f"    ... {len(hunks) - 20} more files")
    messages = llm_review.build_messages(spec["policy"], patch_hash, patch_text)
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
    ok &= _ok(patch_text in user, "diff is sent verbatim (nothing stripped)")
    ok &= _ok(tag not in patch_text, "patch does not contain the fence tag")
    ok &= _ok(
        len(patch_text) <= config.LLM_REVIEW_MAX_INPUT_CHARS,
        f"within MAX_INPUT_CHARS ({config.LLM_REVIEW_MAX_INPUT_CHARS})",
    )

    _step(5, "Model call through the gate")
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
        patch_bytes=patch_bytes,
        patch_hash=patch_hash,
        spec=spec,
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
                "leading <think> block or ``` fence, see step 6"
            )

    _step(6, "Verdict validation")
    if result.evidence.get("retryable"):
        _ok(False, "provider unavailable; the worker would defer and retry")
    elif result.ok or result.reason == "llm_review_rejected":
        ok &= _ok(True, "verdict passed schema and hunk checks")
    else:
        ok &= _ok(False, f"gate failed closed: {result.reason}")

    _step(7, "Cache")
    again = llm_review.check_llm_review(
        patch_bytes=patch_bytes,
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

    _step(8, "Gate result")
    outcome = (
        "DEFER"
        if result.evidence.get("retryable")
        else "PASS"
        if result.ok
        else "REJECT"
    )
    print(f"  OUTCOME: {outcome}")
    print(
        json.dumps(
            {"ok": result.ok, "reason": result.reason, **result.evidence},
            indent=2,
            default=str,
        )
    )
    print(f"\nchecks {'all passed' if ok else 'had failures'}")
    return 0 if ok else 1


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
    p.add_argument("--cache-dir")
    p.add_argument("--probe-response-format", action="store_true")
    args = p.parse_args(argv)
    if args.probe_response_format:
        return probe(args)
    if not args.patch_file:
        p.error("--patch-file is required unless --probe-response-format")
    return review(args)


if __name__ == "__main__":
    sys.exit(main())
