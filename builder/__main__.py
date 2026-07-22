"""Ops CLI for hermetic engine builds (A2b empty-patch baseline).

Usage:
  python -m builder \\
    --baseline-repo https://github.com/vllm-project/vllm.git \\
    --baseline-commit ee0da84ab9e04ac7610e28580af62c365e898389 \\
    --base-image ghcr.io/pareton-ai/pareton-baseline@sha256:... \\
    --image-ref ghcr.io/pareton-ai/pareton-engine:baseline \\
    --empty-patch \\
    --push
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from builder.hermetic import build_engine_image


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hermetic Pareton engine image build")
    p.add_argument("--baseline-repo", required=True)
    p.add_argument("--baseline-commit", required=True)
    p.add_argument("--base-image", required=True, help="Digest-pinned build base")
    p.add_argument(
        "--image-ref",
        required=True,
        help="Tag to build/push (e.g. ghcr.io/pareton-ai/pareton-engine:baseline)",
    )
    p.add_argument(
        "--patch-file",
        type=Path,
        default=None,
        help="Git patch file (default: empty when --empty-patch)",
    )
    p.add_argument(
        "--empty-patch",
        action="store_true",
        help="Ops-only: allow empty patch (A2b baseline engine)",
    )
    p.add_argument("--push", action="store_true", default=False)
    p.add_argument("--no-push", action="store_true", default=False)
    p.add_argument("--work-root", type=Path, default=None)
    args = p.parse_args(argv)

    if args.empty_patch and args.patch_file is not None:
        p.error("use either --empty-patch or --patch-file, not both")
    if not args.empty_patch and args.patch_file is None:
        p.error("provide --patch-file or --empty-patch")

    patch_bytes = b"" if args.empty_patch else args.patch_file.read_bytes()
    patch_hash = "sha256:" + hashlib.sha256(patch_bytes).hexdigest()
    push = bool(args.push) and not args.no_push

    result = build_engine_image(
        baseline_repo=args.baseline_repo,
        baseline_commit=args.baseline_commit,
        base_image=args.base_image,
        patch_bytes=patch_bytes,
        patch_hash=patch_hash,
        work_root=args.work_root,
        push=push,
        allow_empty_patch=bool(args.empty_patch),
        image_ref_override=args.image_ref,
    )
    if not result.ok:
        print(f"FAIL {result.reason}: {result.evidence}", file=sys.stderr)
        return 1
    image = result.evidence.get("image_ref") or result.evidence.get("image_tag")
    log = result.evidence.get("build_log")
    if log:
        print(f"build_log={log}", file=sys.stderr)
    print(image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
