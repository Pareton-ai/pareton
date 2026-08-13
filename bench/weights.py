"""HuggingFace weights staging for egress-free engine containers.

Host downloads a pinned revision into a local cache, validates the tree, then
lifecycle mounts it read-only at ``/model``. Engines never see HF tokens and
never need network egress.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from bench.lifecycle import HostEnvironmentError
from bench.schemas import ModelSpec
from bench.validate import sha256_file

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path.home() / ".cache" / "pareton" / "hf"

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
_TOKENIZER_NAMES = frozenset({"tokenizer.json", "tokenizer.model"})
_STAGING_ATTEMPTS = 5
_TRANSIENT_STAGING = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionError,
    TimeoutError,
)
# EPIPE, ECONNRESET, ETIMEDOUT (Linux numbers; macOS EPIPE is also 32).
_TRANSIENT_ERRNOS = frozenset({32, 104, 110})
_TRANSIENT_EXC_NAMES = frozenset(
    {"RemoteProtocolError", "ReadTimeout", "ConnectTimeout", "ReadError"}
)


class WeightsError(HostEnvironmentError):
    """Weights staging failure (maps to CLI exit code 2)."""


@dataclass
class StagedWeights:
    path: Path
    weights_sha256: str
    num_files: int
    total_bytes: int
    manifest: dict[str, Any]


def _cache_root(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir.expanduser().resolve()
    try:
        import config as _cfg

        return (
            Path(getattr(_cfg, "BENCH_HF_CACHE_DIR", _DEFAULT_CACHE))
            .expanduser()
            .resolve()
        )
    except Exception:  # noqa: BLE001 — standalone pod may lack config
        return _DEFAULT_CACHE.resolve()


def repo_slug(hf_repo: str) -> str:
    return hf_repo.replace("/", "--")


def _under_cache_meta(path: Path, root: Path) -> bool:
    try:
        return ".cache" in path.relative_to(root).parts
    except ValueError:
        return False


def _iter_staged_paths(root: Path):
    for path in root.rglob("*"):
        if _under_cache_meta(path, root):
            continue
        yield path


def assert_no_symlinks(root: Path) -> None:
    for path in _iter_staged_paths(root):
        if path.is_symlink():
            rel = path.relative_to(root).as_posix()
            raise WeightsError(f"symlink found under staged weights: {rel}")


def assert_complete(root: Path) -> None:
    has_config = False
    has_weights = False
    has_tokenizer = False
    for path in _iter_staged_paths(root):
        if path.is_symlink() or not path.is_file():
            continue
        name = path.name
        if name == "config.json":
            has_config = True
        if name.endswith(_WEIGHT_SUFFIXES):
            has_weights = True
        if name in _TOKENIZER_NAMES or name.startswith("vocab."):
            has_tokenizer = True
    missing: list[str] = []
    if not has_config:
        missing.append("config.json")
    if not has_weights:
        missing.append("weights (*.safetensors|*.bin|*.pt)")
    if not has_tokenizer:
        missing.append("tokenizer (tokenizer.json|tokenizer.model|vocab.*)")
    if missing:
        raise WeightsError(
            f"incomplete weights snapshot under {root}: missing {', '.join(missing)}"
        )


def validate_staged_tree(root: Path) -> None:
    """No-symlink + completeness (identical for cache hits and post-download)."""
    assert_no_symlinks(root)
    assert_complete(root)


def build_weights_manifest(
    root: Path, *, repo: str, revision: str
) -> tuple[dict[str, Any], str, int, int]:
    """Return (manifest, aggregate sha256:<hex>, num_files, total_bytes)."""
    entries: list[dict[str, Any]] = []
    for path in _iter_staged_paths(root):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        entries.append(
            {
                "path": rel,
                "sha256": sha256_file(path),
                "size": size,
            }
        )
    entries.sort(key=lambda e: e["path"])
    manifest: dict[str, Any] = {
        "repo": repo,
        "revision": revision,
        "files": entries,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    aggregate = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    total_bytes = sum(int(e["size"]) for e in entries)
    return manifest, aggregate, len(entries), total_bytes


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _is_transient_staging_error(exc: BaseException) -> bool:
    """True for dropped connections that huggingface_hub can resume."""
    if isinstance(exc, _TRANSIENT_STAGING):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in _TRANSIENT_ERRNOS:
        return True
    if type(exc).__name__ in _TRANSIENT_EXC_NAMES:
        return True
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return _is_transient_staging_error(cause)
    return False


def _finalize(root: Path, model: ModelSpec) -> StagedWeights:
    validate_staged_tree(root)
    manifest, aggregate, n_files, total = build_weights_manifest(
        root, repo=model.hf_repo, revision=model.hf_revision
    )
    return StagedWeights(
        path=root,
        weights_sha256=aggregate,
        num_files=n_files,
        total_bytes=total,
        manifest=manifest,
    )


def _download_to_partial(
    model: ModelSpec,
    *,
    partial: Path,
    token_env: str,
) -> None:
    token = os.environ.get(token_env)
    for attempt in range(1, _STAGING_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=model.hf_repo,
                revision=model.hf_revision,
                token=token,
                local_dir=str(partial),
            )
            return
        except Exception as exc:  # noqa: BLE001 — wrap all hub/network failures
            transient = _is_transient_staging_error(exc)
            if transient and attempt < _STAGING_ATTEMPTS:
                logger.warning(
                    "transient %s staging %s@%s (attempt %d/%d); retrying",
                    type(exc).__name__,
                    model.hf_repo,
                    model.hf_revision,
                    attempt,
                    _STAGING_ATTEMPTS,
                )
                time.sleep(min(2**attempt, 30))
                continue
            if not transient:
                _rm_tree(partial)
            raise WeightsError(
                f"{type(exc).__name__} staging {model.hf_repo}@{model.hf_revision}"
            ) from None


def stage_weights(
    model: ModelSpec,
    *,
    token_env: str = "HF_TOKEN",
    cache_dir: Path | None = None,
) -> StagedWeights:
    """Download (or reuse) pinned HF weights; return staged path + aggregate hash."""
    cache_root = _cache_root(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    slug = repo_slug(model.hf_repo)
    final = cache_root / slug / model.hf_revision

    if final.is_dir():
        try:
            return _finalize(final, model)
        except WeightsError as exc:
            logger.warning(
                "invalid cache at %s (%s); deleting and redownloading",
                final,
                exc,
            )
            _rm_tree(final)

    # Deterministic partial dir so a later bench on the same pod resumes.
    partial_root = cache_root / ".partial"
    partial_root.mkdir(parents=True, exist_ok=True)
    partial = partial_root / f"{slug}-{model.hf_revision}"
    partial.mkdir(parents=True, exist_ok=True)

    _download_to_partial(model, partial=partial, token_env=token_env)

    try:
        validate_staged_tree(partial)
    except WeightsError:
        _rm_tree(partial)
        raise

    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        _rm_tree(final)
    try:
        os.rename(partial, final)
    except OSError:
        _rm_tree(partial)
        raise WeightsError(
            f"OSError publishing staged weights for {model.hf_repo}@{model.hf_revision}"
        ) from None

    return _finalize(final, model)
