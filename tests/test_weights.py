"""Offline unit tests for bench.weights (WS-B5). No network, no Docker daemon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bench.lifecycle import BenchNetwork
from bench.main import (
    MOCK_WEIGHTS_SHA256,
    _EngineProvider,
    build_bench_report,
    build_inputs_fingerprint,
)
from bench.output import OutputLayout
from bench.schemas import ModelSpec
from bench.validate import (
    RequestValidationError,
    load_json,
    validate_bench_request_dict,
)
from bench.weights import (
    WeightsError,
    build_weights_manifest,
    repo_slug,
    stage_weights,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST = ROOT / "fixtures" / "bench" / "sample_request.json"


def _model(
    repo: str = "org/model",
    revision: str = "abcdef1",
) -> ModelSpec:
    return ModelSpec(
        hf_repo=repo,
        hf_revision=revision,
        dtype="bfloat16",
        quantization=None,
        max_model_len=2048,
    )


def _write_complete_snapshot(root: Path, *, weight_byte: bytes = b"WEIGHTS") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"architectures":["Fake"]}\n', encoding="utf-8")
    (root / "model.safetensors").write_bytes(weight_byte)
    (root / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")


def _fake_snapshot_download(
    *,
    repo_id: str,
    revision: str,
    token: str | None,
    local_dir: str,
    weight_byte: bytes = b"WEIGHTS",
    **kwargs: Any,
) -> str:
    assert "local_dir_use_symlinks" not in kwargs
    path = Path(local_dir)
    _write_complete_snapshot(path, weight_byte=weight_byte)
    # Record last call for assertions via closure attributes.
    _fake_snapshot_download.last_call = {  # type: ignore[attr-defined]
        "repo_id": repo_id,
        "revision": revision,
        "token": token,
        "local_dir": local_dir,
        "kwargs": kwargs,
    }
    return str(path)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "hf-cache"


# ---------------------------------------------------------------------------
# stage_weights
# ---------------------------------------------------------------------------


def test_happy_path_atomic_publish(monkeypatch: pytest.MonkeyPatch, cache_dir: Path):
    monkeypatch.setattr("bench.weights.snapshot_download", _fake_snapshot_download)
    model = _model()
    staged = stage_weights(model, cache_dir=cache_dir)

    expected = cache_dir / repo_slug(model.hf_repo) / model.hf_revision
    assert staged.path == expected.resolve()
    assert staged.path.is_dir()
    assert not (cache_dir / ".partial").exists() or not any(
        (cache_dir / ".partial").iterdir()
    )
    assert staged.num_files == 3
    assert staged.total_bytes > 0
    assert staged.manifest["repo"] == model.hf_repo
    assert staged.manifest["revision"] == model.hf_revision
    assert len(staged.manifest["files"]) == 3
    assert staged.weights_sha256.startswith("sha256:")
    assert len(staged.weights_sha256) == len("sha256:") + 64

    # Deterministic aggregate
    again = stage_weights(model, cache_dir=cache_dir)
    assert again.weights_sha256 == staged.weights_sha256

    # Changed byte -> different sha (force redownload into fresh cache)
    cache2 = cache_dir.parent / "hf-cache-2"
    calls: list[bytes] = []

    def download_variant(**kwargs: Any) -> str:
        byte = b"WEIGHTS" if not calls else b"CHANGED"
        calls.append(byte)
        return _fake_snapshot_download(weight_byte=byte, **kwargs)

    monkeypatch.setattr("bench.weights.snapshot_download", download_variant)
    a = stage_weights(model, cache_dir=cache2)
    # Corrupt final to force miss with different bytes
    import shutil

    shutil.rmtree(a.path)
    b = stage_weights(model, cache_dir=cache2)
    assert a.weights_sha256 != b.weights_sha256


def test_download_call_shape_and_token(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
):
    monkeypatch.setattr("bench.weights.snapshot_download", _fake_snapshot_download)
    monkeypatch.setenv("MY_HF_TOKEN", "tok-secret-value")
    model = _model()
    stage_weights(model, token_env="MY_HF_TOKEN", cache_dir=cache_dir)

    call = _fake_snapshot_download.last_call  # type: ignore[attr-defined]
    assert call["repo_id"] == model.hf_repo
    assert call["revision"] == model.hf_revision
    assert call["token"] == "tok-secret-value"
    assert "/.partial/" in call["local_dir"].replace("\\", "/")
    assert Path(call["local_dir"]).name == f"{repo_slug(model.hf_repo)}-{model.hf_revision}"
    assert "local_dir_use_symlinks" not in call["kwargs"]

    monkeypatch.delenv("MY_HF_TOKEN", raising=False)
    cache2 = cache_dir.parent / "cache-no-token"
    stage_weights(model, token_env="MY_HF_TOKEN", cache_dir=cache2)
    call2 = _fake_snapshot_download.last_call  # type: ignore[attr-defined]
    assert call2["token"] is None


def test_cache_hit_skips_download(monkeypatch: pytest.MonkeyPatch, cache_dir: Path):
    model = _model()
    final = cache_dir / repo_slug(model.hf_repo) / model.hf_revision
    _write_complete_snapshot(final)

    mock_dl = MagicMock(side_effect=AssertionError("download should not be called"))
    monkeypatch.setattr("bench.weights.snapshot_download", mock_dl)

    staged = stage_weights(model, cache_dir=cache_dir)
    assert staged.path == final.resolve()
    mock_dl.assert_not_called()
    assert staged.num_files == 3


def test_corrupt_cache_is_miss_and_redownloads(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
):
    model = _model()
    final = cache_dir / repo_slug(model.hf_repo) / model.hf_revision
    final.mkdir(parents=True)
    (final / "model.safetensors").write_bytes(b"x")
    # missing config.json + tokenizer

    monkeypatch.setattr("bench.weights.snapshot_download", _fake_snapshot_download)
    staged = stage_weights(model, cache_dir=cache_dir)
    assert staged.path == final.resolve()
    assert (final / "config.json").is_file()
    call = _fake_snapshot_download.last_call  # type: ignore[attr-defined]
    assert call["repo_id"] == model.hf_repo


@pytest.mark.parametrize(
    "drop",
    ["config.json", "model.safetensors", "tokenizer.json"],
)
def test_incomplete_after_download_raises(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path, drop: str
):
    def incomplete(**kwargs: Any) -> str:
        path = Path(kwargs["local_dir"])
        _write_complete_snapshot(path)
        (path / drop).unlink()
        return str(path)

    monkeypatch.setattr("bench.weights.snapshot_download", incomplete)
    model = _model()
    with pytest.raises(WeightsError, match="incomplete"):
        stage_weights(model, cache_dir=cache_dir)
    final = cache_dir / repo_slug(model.hf_repo) / model.hf_revision
    assert not final.exists()


def test_symlink_post_download_raises(monkeypatch: pytest.MonkeyPatch, cache_dir: Path):
    def with_symlink(**kwargs: Any) -> str:
        path = Path(kwargs["local_dir"])
        _write_complete_snapshot(path)
        (path / "link.bin").symlink_to(path / "model.safetensors")
        return str(path)

    monkeypatch.setattr("bench.weights.snapshot_download", with_symlink)
    model = _model()
    with pytest.raises(WeightsError, match="symlink"):
        stage_weights(model, cache_dir=cache_dir)
    final = cache_dir / repo_slug(model.hf_repo) / model.hf_revision
    assert not final.exists()


def test_symlink_cache_hit_redownloads(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
):
    model = _model()
    final = cache_dir / repo_slug(model.hf_repo) / model.hf_revision
    _write_complete_snapshot(final)
    (final / "evil.bin").symlink_to(final / "model.safetensors")

    monkeypatch.setattr("bench.weights.snapshot_download", _fake_snapshot_download)
    staged = stage_weights(model, cache_dir=cache_dir)
    assert staged.path == final.resolve()
    assert not any(p.is_symlink() for p in final.rglob("*"))
    call = _fake_snapshot_download.last_call  # type: ignore[attr-defined]
    assert call["repo_id"] == model.hf_repo


def test_download_failure_message_hygiene(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
):
    secret = "hf_super_secret_token_xyz"
    monkeypatch.setenv("HF_TOKEN", secret)

    class IncompleteSnapshotError(Exception):
        pass

    def boom(**_kwargs: Any) -> str:
        raise IncompleteSnapshotError(
            f"gated repo auth failed with token={secret} detail=leak-me"
        )

    monkeypatch.setattr("bench.weights.snapshot_download", boom)
    model = _model(repo="org/gated", revision="abc1234")
    with pytest.raises(WeightsError) as ei:
        stage_weights(model, token_env="HF_TOKEN", cache_dir=cache_dir)
    msg = str(ei.value)
    assert "IncompleteSnapshotError" in msg
    assert "org/gated" in msg
    assert "abc1234" in msg
    assert secret not in msg
    assert "leak-me" not in msg
    assert "token=" not in msg
    partial = cache_dir / ".partial" / f"{repo_slug(model.hf_repo)}-{model.hf_revision}"
    assert not partial.exists()


def test_transient_broken_pipe_retries_then_publishes(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
):
    calls = {"n": 0}

    def flaky(**kwargs: Any) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise BrokenPipeError("hub download")
        return _fake_snapshot_download(**kwargs)

    monkeypatch.setattr("bench.weights.snapshot_download", flaky)
    monkeypatch.setattr("bench.weights.time.sleep", lambda _s: None)
    model = _model()
    staged = stage_weights(model, cache_dir=cache_dir)
    assert calls["n"] == 3
    assert staged.path.is_dir()
    assert (staged.path / "model.safetensors").is_file()


def test_exhausted_transient_keeps_partial_for_resume(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
):
    def boom(**kwargs: Any) -> str:
        path = Path(kwargs["local_dir"])
        path.mkdir(parents=True, exist_ok=True)
        (path / "shard.bin").write_bytes(b"partial-bytes")
        raise BrokenPipeError("hub download")

    monkeypatch.setattr("bench.weights.snapshot_download", boom)
    monkeypatch.setattr("bench.weights.time.sleep", lambda _s: None)
    model = _model()
    with pytest.raises(WeightsError, match="BrokenPipeError"):
        stage_weights(model, cache_dir=cache_dir)
    partial = cache_dir / ".partial" / f"{repo_slug(model.hf_repo)}-{model.hf_revision}"
    assert (partial / "shard.bin").read_bytes() == b"partial-bytes"

    monkeypatch.setattr("bench.weights.snapshot_download", _fake_snapshot_download)
    staged = stage_weights(model, cache_dir=cache_dir)
    assert staged.path.is_dir()
    assert not partial.exists()


# ---------------------------------------------------------------------------
# hf_repo validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_repo",
    [
        "../../etc",
        "org",
        "org/name/extra",
        "org/foo..bar",
        "../org/name",
        "org/../name",
    ],
)
def test_hf_repo_validation_rejects(bad_repo: str):
    raw = load_json(SAMPLE_REQUEST)
    raw["model"]["hf_repo"] = bad_repo
    with pytest.raises(RequestValidationError, match="hf_repo"):
        validate_bench_request_dict(raw)


def test_hf_repo_validation_accepts_sample():
    raw = load_json(SAMPLE_REQUEST)
    req = validate_bench_request_dict(raw)
    assert req.model.hf_repo == "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# OutputLayout + fingerprint + provider wiring
# ---------------------------------------------------------------------------


def test_write_weights_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("bench.weights.snapshot_download", _fake_snapshot_download)
    cache = tmp_path / "cache"
    staged = stage_weights(_model(), cache_dir=cache)
    layout = OutputLayout(tmp_path / "out")
    layout.prepare()
    path = layout.write_weights_manifest(
        staged.manifest, aggregate=staged.weights_sha256
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["aggregate"] == staged.weights_sha256
    assert data["repo"] == staged.manifest["repo"]
    assert data["files"] == staged.manifest["files"]


def test_engine_provider_passes_weights_dir(tmp_path: Path):
    from tests.test_lifecycle import FakeDocker

    raw = load_json(SAMPLE_REQUEST)
    req = validate_bench_request_dict(raw)
    weights = tmp_path / "staged"
    weights.mkdir()
    provider = _EngineProvider(
        req=req, mock=False, logs_dir=tmp_path / "logs", weights_dir=weights
    )

    fake = FakeDocker()
    fake.image_digests[req.engines.baseline.image] = [
        f"{req.engines.baseline.image}@sha256:" + ("a" * 64)
    ]
    import bench.lifecycle as life

    original_wait = life.wait_until_healthy
    life.wait_until_healthy = lambda *_a, **_k: None  # type: ignore[assignment]
    try:
        with BenchNetwork(run_id="w5run0000001", runner=fake, cmd_timeout_s=30) as net:
            with provider._docker_phase(net, "baseline") as handle:
                assert handle.container_name.startswith("pareton-bench-")
    finally:
        life.wait_until_healthy = original_wait  # type: ignore[assignment]
        provider.shutdown()

    run = next(c for c, _ in fake.calls if c[:2] == ["docker", "run"])
    vol = next(a for a in run if a.endswith(":/model:ro"))
    assert str(weights.resolve()) in vol


def test_fingerprint_uses_real_aggregate_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("bench.weights.snapshot_download", _fake_snapshot_download)
    staged = stage_weights(_model(), cache_dir=tmp_path / "cache")
    assert staged.weights_sha256 != MOCK_WEIGHTS_SHA256

    raw = SAMPLE_REQUEST.read_bytes()
    req = validate_bench_request_dict(json.loads(raw))
    fp = build_inputs_fingerprint(
        request_raw=raw,
        req=req,
        baseline_digest="sha256:" + ("a" * 64),
        candidate_digest="sha256:" + ("b" * 64),
        model_weights_sha256=staged.weights_sha256,
    )
    assert fp.model_weights_sha256 == staged.weights_sha256
    assert fp.model_weights_sha256 != MOCK_WEIGHTS_SHA256

    from bench.env import collect_environment

    report = build_bench_report(
        request_raw=raw,
        req=req,
        env=collect_environment(),
        baseline_digest=fp.baseline_image_digest,
        candidate_digest=fp.candidate_image_digest,
        model_weights_sha256=staged.weights_sha256,
        corr=None,
        perf=None,
        sla=None,
        skipped_note=None,
        started_at="2026-07-18T00:00:00Z",
    )
    d = report.to_dict()
    assert d["inputs_fingerprint"]["model_weights_sha256"] == staged.weights_sha256

    # Independent recompute from manifest alone
    recomputed, agg, _, _ = build_weights_manifest(
        staged.path, repo=staged.manifest["repo"], revision=staged.manifest["revision"]
    )
    assert agg == staged.weights_sha256
    assert recomputed == staged.manifest


def test_validate_report_dict_with_real_weights_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Docker-mode fingerprint with real aggregate passes report validation."""
    from bench.main import run_bench

    monkeypatch.setattr("bench.weights.snapshot_download", _fake_snapshot_download)
    # Use mock engine so we don't need Docker; fingerprint still gets zero
    # placeholder in mock mode. For real aggregate + validate_report_dict,
    # construct a report from a successful mock run and patch the field.
    out = tmp_path / "out"
    rc = run_bench(SAMPLE_REQUEST, out, mock_engine=True)
    assert rc == 0
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    assert report["inputs_fingerprint"]["model_weights_sha256"] == MOCK_WEIGHTS_SHA256

    # Simulate docker-mode digest substitution
    staged = stage_weights(_model(), cache_dir=tmp_path / "cache")
    report["inputs_fingerprint"]["model_weights_sha256"] = staged.weights_sha256
    from bench.validate import validate_report_dict

    validate_report_dict(report)
