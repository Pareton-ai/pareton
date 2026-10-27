"""Baseline and patched versions of every file a patch touches.

Built from the baseline checkout left by gate c, without touching its working
tree: the patch is applied to a throwaway index (``git apply --cached``) and
both versions are read from git objects. The review diff that the LLM reads
(d2) is the diff between the comment-stripped versions, so a comment cannot
reach the reviewer and a comment-only change disappears. Line numbers are those
of the real files, because stripping keeps every line.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from gate.comment_strip import STRIPPED, StripResult, strip_comments

_GIT_TIMEOUT_S = 120
_HUNK_NEW_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class SourceViewError(Exception):
    pass


@dataclass(frozen=True)
class FileView:
    path: str
    pre: str | None  # None: the patch creates the file
    post: str | None  # None: the patch deletes the file
    pre_strip: StripResult | None
    post_strip: StripResult | None
    utf8: bool = True
    # Lines the patch adds to this file, by line number in the patched file,
    # as git computes them from the two versions (comments included).
    added: dict[int, str] | None = None


@dataclass(frozen=True)
class SourceView:
    files: tuple[FileView, ...]
    review_diff: str

    @property
    def post_texts(self) -> dict[str, str]:
        return {f.path: f.post for f in self.files if f.post is not None}

    @property
    def added_lines(self) -> dict[str, dict[int, str]]:
        return {f.path: f.added for f in self.files if f.added}

    def strip_summary(self) -> dict[str, object]:
        """Per-status file counts and the files whose comments were kept."""
        counts: dict[str, int] = {}
        kept: list[dict[str, str]] = []
        for f in self.files:
            for side, res in (("pre", f.pre_strip), ("post", f.post_strip)):
                if res is None:
                    continue
                counts[res.status] = counts.get(res.status, 0) + 1
                if res.status in ("unparseable", "check_failed"):
                    kept.append({"file": f.path, "side": side, "status": res.status})
        return {"status_counts": counts, "comments_kept": kept[:50]}


def _git(
    repo: Path, *args: str, env: dict[str, str] | None = None, stdin: bytes = b""
) -> bytes:
    proc = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=repo,
        env=env,
        input=stdin,
        capture_output=True,
        timeout=_GIT_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise SourceViewError(
            f"git {args[0]} failed: {proc.stderr.decode(errors='replace')[-500:]}"
        )
    return proc.stdout


def _cat_blobs(repo: Path, specs: list[str], env: dict[str, str]) -> list[bytes]:
    if not specs:
        return []
    out = _git(
        repo,
        "cat-file",
        "--batch",
        env=env,
        stdin="".join(s + "\n" for s in specs).encode(),
    )
    blobs: list[bytes] = []
    pos = 0
    for spec in specs:
        eol = out.index(b"\n", pos)
        header = out[pos:eol].split()
        if len(header) != 3 or header[1] != b"blob":
            raise SourceViewError(f"cannot read {spec}")
        size = int(header[2])
        start = eol + 1
        blobs.append(out[start : start + size])
        pos = start + size + 1
    return blobs


def _decode(data: bytes) -> tuple[str, bool]:
    try:
        return data.decode("utf-8"), True
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), False


def build_source_view(
    repo_dir: Path, patch_bytes: bytes, *, base_ref: str = "HEAD"
) -> SourceView:
    """Apply the patch to a scratch index over ``base_ref`` and collect files."""
    with tempfile.TemporaryDirectory(prefix="pareton-view-") as td:
        tmp = Path(td)
        env = {**os.environ, "GIT_INDEX_FILE": str(tmp / "index")}
        patch_path = tmp / "submission.diff"
        patch_path.write_bytes(patch_bytes)
        _git(repo_dir, "read-tree", base_ref, env=env)
        _git(
            repo_dir,
            "apply",
            "--cached",
            "--whitespace=nowarn",
            str(patch_path),
            env=env,
        )
        raw = _git(
            repo_dir,
            "diff",
            "--cached",
            "--no-renames",
            "--name-status",
            "-z",
            base_ref,
            env=env,
        )
        fields = raw.decode("utf-8", errors="surrogateescape").split("\0")
        entries: list[tuple[str, str]] = []
        for i in range(0, len(fields) - 1, 2):
            status, path = fields[i], fields[i + 1]
            if "\n" in path:
                raise SourceViewError("path with a newline")
            entries.append((status[:1], path))
        pre_specs = [f"{base_ref}:{p}" for s, p in entries if s != "A"]
        post_specs = [f":{p}" for s, p in entries if s != "D"]
        pre_blobs = iter(_cat_blobs(repo_dir, pre_specs, env))
        post_blobs = iter(_cat_blobs(repo_dir, post_specs, env))

        files: list[FileView] = []
        for status, path in entries:
            pre = post = None
            utf8 = True
            if status != "A":
                pre, ok = _decode(next(pre_blobs))
                utf8 &= ok
            if status != "D":
                post, ok = _decode(next(post_blobs))
                utf8 &= ok
            files.append(
                FileView(
                    path=path,
                    pre=pre,
                    post=post,
                    added=_added_lines(tmp, pre, post),
                    # A file that isn't valid UTF-8 is shown as is.
                    pre_strip=strip_comments(path, pre)
                    if pre is not None and utf8
                    else None,
                    post_strip=strip_comments(path, post)
                    if post is not None and utf8
                    else None,
                    utf8=utf8,
                )
            )
        review = "".join(_file_diff(tmp, f) for f in files)
    return SourceView(tuple(files), review)


def _added_lines(tmp: Path, pre: str | None, post: str | None) -> dict[int, str]:
    """Lines of ``post`` that are not in ``pre``, by git's own diff."""
    if post is None:
        return {}
    sides = []
    for name, text in (("pre-raw", pre), ("post-raw", post)):
        if text is None:
            sides.append("/dev/null")
            continue
        p = tmp / name
        p.write_text(text, encoding="utf-8", errors="surrogatepass")
        sides.append(str(p))
    proc = subprocess.run(
        ["git", "diff", "--no-index", "--no-color", "--no-ext-diff", "--text", "-U0"]
        + sides,
        capture_output=True,
        timeout=_GIT_TIMEOUT_S,
    )
    if proc.returncode not in (0, 1):
        raise SourceViewError("git diff failed")
    added: dict[int, str] = {}
    new_line = 0
    in_hunk = False
    for raw in proc.stdout.decode("utf-8", errors="replace").split("\n"):
        h = _HUNK_NEW_RE.match(raw)
        if h:
            new_line, in_hunk = int(h.group(1)), True
            continue
        # With -U0 a hunk has only "-", "+" and "\ No newline" lines.
        if in_hunk and raw.startswith("+"):
            added[new_line] = raw[1:]
            new_line += 1
    return added


def _view_text(text: str | None, res: StripResult | None) -> str | None:
    if text is None:
        return None
    return res.text if res is not None else text


def _file_diff(tmp: Path, f: FileView) -> str:
    """Unified diff of the stripped versions, with our own header."""
    pre = _view_text(f.pre, f.pre_strip)
    post = _view_text(f.post, f.post_strip)
    if pre == post:
        return ""
    sides = []
    for name, text in (("pre", pre), ("post", post)):
        if text is None:
            sides.append("/dev/null")
            continue
        p = tmp / name
        p.write_text(text, encoding="utf-8", errors="surrogatepass")
        sides.append(str(p))
    proc = subprocess.run(
        [
            "git",
            "diff",
            "--no-index",
            "--no-color",
            "--no-ext-diff",
            "--text",
            "-U3",
            *sides,
        ],
        capture_output=True,
        timeout=_GIT_TIMEOUT_S,
    )
    if proc.returncode not in (0, 1):
        raise SourceViewError(f"git diff failed for {f.path}")
    body = proc.stdout.decode("utf-8", errors="replace")
    at = body.find("\n@@ ")
    if at < 0:
        return ""
    old = "/dev/null" if pre is None else f"a/{f.path}"
    new = "/dev/null" if post is None else f"b/{f.path}"
    notes = []
    for side, res in (("baseline", f.pre_strip), ("patched", f.post_strip)):
        if res is not None and res.status not in (STRIPPED, "no_comments"):
            notes.append(f"# note: {side} file shown with comments ({res.status})\n")
    if not f.utf8:
        notes.append(
            "# note: file is not valid UTF-8; shown with replacement characters\n"
        )
    return (
        f"diff --git a/{f.path} b/{f.path}\n"
        + "".join(notes)
        + f"--- {old}\n+++ {new}\n"
        + body[at + 1 :]
    )
