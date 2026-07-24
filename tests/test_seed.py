"""Offline unit tests for campaign seed CLI helpers."""

from __future__ import annotations

from uuid import uuid4

import pytest

from campaign import seed
from campaign.seed import (
    DEFAULT_BASE_IMAGE_DIGEST,
    DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST,
    FIXTURE_TRACE,
    _sha256_file,
    main,
    seed_synthetic_campaign,
)

pytestmark = pytest.mark.unit

REAL_BASE = "sha256:" + ("a" * 64)
REAL_ENGINE = "sha256:" + ("d" * 64)
HTTPS_TRACE = "https://example.test/stage0/workload_trace.json"


def _fixture_sha() -> str:
    return _sha256_file(FIXTURE_TRACE)


def _patch_store(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {"manifest": None, "inserts": 0}

    monkeypatch.setattr(seed, "list_campaigns", lambda status="open": [])
    monkeypatch.setattr(seed, "insert_profile", lambda **kwargs: uuid4())

    def _insert(manifest):
        captured["manifest"] = manifest
        captured["inserts"] += 1
        return str(manifest.campaign_id)

    monkeypatch.setattr(seed, "insert_campaign", _insert)
    return captured


def test_default_file_url_with_placeholders(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(allow_placeholders=True)
    assert captured["inserts"] == 1
    m = captured["manifest"]
    assert m.workload_trace_url == f"file://{FIXTURE_TRACE.resolve()}"
    assert m.workload_trace_sha256 == _fixture_sha()


def test_https_url_with_matching_sha(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    seed_synthetic_campaign(
        base_image_digest=REAL_BASE,
        baseline_engine_image_digest=REAL_ENGINE,
        workload_trace_url=HTTPS_TRACE,
        workload_trace_sha256=_fixture_sha(),
    )
    m = captured["manifest"]
    assert m.workload_trace_url == HTTPS_TRACE
    assert m.workload_trace_sha256 == _fixture_sha()


def test_https_without_sha_raises_before_insert(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="requires --workload-trace-sha256"):
        seed_synthetic_campaign(
            base_image_digest=REAL_BASE,
            baseline_engine_image_digest=REAL_ENGINE,
            workload_trace_url=HTTPS_TRACE,
        )
    assert captured["inserts"] == 0


def test_sha_mismatch_raises_before_insert(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="does not match local fixture"):
        seed_synthetic_campaign(
            base_image_digest=REAL_BASE,
            baseline_engine_image_digest=REAL_ENGINE,
            workload_trace_url=HTTPS_TRACE,
            workload_trace_sha256="sha256:" + ("0" * 64),
        )
    assert captured["inserts"] == 0


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://example.test/t.json",
        "/abs/path/t.json",
        "  ",
    ],
)
def test_invalid_scheme_raises_before_insert(
    monkeypatch: pytest.MonkeyPatch, bad_url: str
):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError):
        seed_synthetic_campaign(
            base_image_digest=REAL_BASE,
            baseline_engine_image_digest=REAL_ENGINE,
            workload_trace_url=bad_url,
            workload_trace_sha256=_fixture_sha(),
            allow_placeholders=True,
        )
    assert captured["inserts"] == 0


@pytest.mark.parametrize(
    "base,engine",
    [
        (DEFAULT_BASE_IMAGE_DIGEST, REAL_ENGINE),
        (REAL_BASE, DEFAULT_BASELINE_ENGINE_IMAGE_DIGEST),
        ("SHA256:" + ("B" * 64), REAL_ENGINE),
        (REAL_BASE, "SHA256:" + ("C" * 64)),
    ],
)
def test_placeholder_digest_refused(
    monkeypatch: pytest.MonkeyPatch, base: str, engine: str
):
    captured = _patch_store(monkeypatch)
    with pytest.raises(ValueError, match="placeholder digests refused"):
        seed_synthetic_campaign(
            base_image_digest=base,
            baseline_engine_image_digest=engine,
        )
    assert captured["inserts"] == 0


def test_main_allow_placeholders_smoke(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    rc = main(["--allow-placeholders"])
    assert rc == 0
    assert captured["inserts"] == 1
    assert captured["manifest"].workload_trace_url.startswith("file://")


def test_main_https_flags_wired(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_store(monkeypatch)
    rc = main(
        [
            "--base-image-digest",
            REAL_BASE,
            "--baseline-engine-image-digest",
            REAL_ENGINE,
            "--workload-trace-url",
            HTTPS_TRACE,
            "--workload-trace-sha256",
            _fixture_sha(),
        ]
    )
    assert rc == 0
    assert captured["manifest"].workload_trace_url == HTTPS_TRACE
