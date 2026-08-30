"""Gate d: surface check — allowed/denied paths and adversarial diff shapes."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass

from gate.types import GateResult, SubmissionState

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_RENAME_FROM_RE = re.compile(r"^rename from (.+)$")
_RENAME_TO_RE = re.compile(r"^rename to (.+)$")
_COPY_FROM_RE = re.compile(r"^copy from (.+)$")
_COPY_TO_RE = re.compile(r"^copy to (.+)$")
_NEW_FILE_RE = re.compile(r"^new file mode (\d+)$")
_DELETED_FILE_RE = re.compile(r"^deleted file mode (\d+)$")
_SIMILARITY_RE = re.compile(r"^similarity index ")
_INDEX_RE = re.compile(r"^index ")
_BINARY_RE = re.compile(r"^GIT binary patch$|^Binary files .* differ$")


@dataclass(frozen=True)
class DiffFileChange:
    path: str
    old_path: str | None = None
    is_symlink: bool = False
    is_binary: bool = False
    is_new: bool = False
    is_deleted: bool = False


def _normalize_path(path: str) -> str:
    p = path.strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def _is_unsafe_path(path: str) -> bool:
    if not path:
        return True
    # Absolute or drive-like
    if path.startswith(("/", "\\")):
        return True
    if len(path) >= 2 and path[1] == ":":
        return True
    parts = path.replace("\\", "/").split("/")
    return ".." in parts


def _match_globs(path: str, globs: list[str]) -> bool:
    norm = path.replace("\\", "/")
    for pattern in globs:
        pat = pattern.replace("\\", "/")
        if fnmatch.fnmatch(norm, pat):
            return True
        # also allow matching basename-style **/x
        if pat.startswith("**/") and fnmatch.fnmatch(norm, pat[3:]):
            return True
        if "/" not in pat and fnmatch.fnmatch(norm.split("/")[-1], pat):
            return True
    return False


def _git_numstat_paths(patch_bytes: bytes) -> tuple[list[str] | None, GateResult | None]:
    """Return the paths Git resolves from the patch's authoritative headers."""
    try:
        result = subprocess.run(
            ["git", "apply", "--numstat", "-z"],
            input=patch_bytes,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None, GateResult.reject("git_apply_numstat_timeout")
    except OSError as exc:
        return None, GateResult.reject("git_apply_numstat_error", error=str(exc))

    if result.returncode != 0:
        return None, GateResult.reject(
            "git_apply_numstat_failed",
            stderr=result.stderr.decode("utf-8", errors="replace")[-2000:],
        )

    paths: list[str] = []
    try:
        records = result.stdout.split(b"\0")
        for record in records:
            if not record:
                continue
            fields = record.split(b"\t", 2)
            if len(fields) != 3 or not fields[2]:
                return None, GateResult.reject("git_apply_numstat_invalid_output")
            paths.append(fields[2].decode("utf-8"))
    except UnicodeDecodeError:
        return None, GateResult.reject("git_apply_numstat_path_not_utf8")
    return paths, None


def parse_diff_paths(patch_text: str) -> list[DiffFileChange]:
    """Parse unified/git diff headers into changed file records."""
    changes: list[DiffFileChange] = []
    lines = patch_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _DIFF_GIT_RE.match(line)
        if not m:
            i += 1
            continue
        old_p = _normalize_path(m.group(1))
        new_p = _normalize_path(m.group(2))
        is_symlink = False
        is_binary = False
        is_new = False
        is_deleted = False
        rename_from: str | None = None
        rename_to: str | None = None
        j = i + 1
        while j < len(lines) and not lines[j].startswith("diff --git "):
            l2 = lines[j]
            if _BINARY_RE.match(l2):
                is_binary = True
            nm = _NEW_FILE_RE.match(l2)
            if nm:
                is_new = True
                mode = nm.group(1)
                if mode.startswith("12"):  # symlink modes 120000
                    is_symlink = True
            dm = _DELETED_FILE_RE.match(l2)
            if dm:
                is_deleted = True
                if dm.group(1).startswith("12"):
                    is_symlink = True
            rm = _RENAME_TO_RE.match(l2)
            if rm:
                rename_to = _normalize_path(rm.group(1))
            rfm = _RENAME_FROM_RE.match(l2)
            if rfm:
                rename_from = _normalize_path(rfm.group(1))
            cm = _COPY_TO_RE.match(l2)
            if cm:
                rename_to = _normalize_path(cm.group(1))
            cfm = _COPY_FROM_RE.match(l2)
            if cfm:
                rename_from = _normalize_path(cfm.group(1))
            if l2.startswith("new mode ") and "120000" in l2:
                is_symlink = True
            j += 1
        final_path = rename_to or (new_p if new_p != "/dev/null" else old_p)
        old_path = rename_from or (old_p if old_p != "/dev/null" else None)
        changes.append(
            DiffFileChange(
                path=final_path,
                old_path=old_path,
                is_symlink=is_symlink,
                is_binary=is_binary,
                is_new=is_new,
                is_deleted=is_deleted,
            )
        )
        i = j
    return changes


def check_surface(
    *,
    patch_bytes: bytes,
    allowed_paths: list[str],
    denied_paths: list[str],
) -> GateResult:
    try:
        text = patch_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return GateResult.reject("patch_not_utf8")

    if not text.strip():
        return GateResult.reject("empty_patch")

    if ".gitmodules" in text or "Subproject commit" in text:
        return GateResult.reject("submodule_change_forbidden")

    changes = parse_diff_paths(text)
    if not changes:
        # fallback: still reject binary markers even without parseable headers
        if "GIT binary patch" in text or "Binary files" in text:
            return GateResult.reject("binary_patch_forbidden")
        return GateResult.reject("empty_patch_no_files")

    touched: list[str] = []
    for ch in changes:
        if ch.is_binary:
            return GateResult.reject("binary_patch_forbidden", path=ch.path)
        if ch.is_symlink:
            return GateResult.reject("symlink_forbidden", path=ch.path)
        for candidate in filter(None, [ch.path, ch.old_path]):
            if _is_unsafe_path(candidate):
                return GateResult.reject("path_traversal", path=candidate)
            if candidate == ".gitmodules" or candidate.endswith("/.gitmodules"):
                return GateResult.reject("submodule_change_forbidden", path=candidate)

        final = ch.path
        if _match_globs(final, denied_paths):
            return GateResult.reject("path_denied", path=final)
        if not _match_globs(final, allowed_paths):
            return GateResult.reject("path_not_allowed", path=final)
        if ch.old_path and ch.old_path != final:
            # rename/copy: both ends must be in-bounds
            if _match_globs(ch.old_path, denied_paths):
                return GateResult.reject("rename_from_denied", path=ch.old_path)
            if not _match_globs(ch.old_path, allowed_paths):
                return GateResult.reject("rename_from_not_allowed", path=ch.old_path)

    git_paths, git_error = _git_numstat_paths(patch_bytes)
    if git_error is not None:
        return git_error
    if not git_paths:
        return GateResult.reject("empty_patch_no_git_files")

    for path in git_paths:
        if _is_unsafe_path(path):
            return GateResult.reject("path_traversal", path=path)
        if path == ".gitmodules" or path.endswith("/.gitmodules"):
            return GateResult.reject("submodule_change_forbidden", path=path)
        if _match_globs(path, denied_paths):
            return GateResult.reject("path_denied", path=path)
        if not _match_globs(path, allowed_paths):
            return GateResult.reject("path_not_allowed", path=path)
        touched.append(path)

    return GateResult.success(
        SubmissionState.SURFACE_OK,
        files=touched,
        file_count=len(touched),
    )
