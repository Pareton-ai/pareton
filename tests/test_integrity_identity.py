"""Unit tests for identity and integrity gates."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


from uuid import uuid4

from campaign.manifest import build_manifest
from campaign.models import SLA
from gate.identity import check_identity
from gate.integrity import (
    PATCH_HASH_MISMATCH,
    check_integrity,
    hash_patch_bytes,
    patch_fingerprint_bytes,
)
from storage.s3 import is_allowed_retrieval_url, patch_url_hotkey
import config


def _campaign(**overrides):
    kwargs = dict(
        campaign_id=uuid4(),
        profile_id=uuid4(),
        baseline_repo="https://github.com/vllm-project/vllm.git",
        baseline_commit="a" * 40,
        base_image_digest="sha256:" + "b" * 64,
        gpu_skus=["H200"],
        workload_trace_sha256="sha256:" + "c" * 64,
        workload_trace_url="https://example.com/t",
        sla=SLA(),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["vllm/**"],
        denied_paths=["tests/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        submission_fee={"amount_tao": "0", "recipient": "5Test"},
        status="open",
    )
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_identity_accepts_registered_on_open_campaign():
    c = _campaign()
    res = check_identity(
        hotkey="hk1",
        registered_hotkeys=["hk1", "hk2"],
        campaign=c,
        baseline_commit="a" * 40,
    )
    assert res.ok


@pytest.mark.parametrize("status", ["draft", "closed"])
def test_identity_rejects_campaign_not_open(status):
    """Status is the only intake switch now that the window is gone."""
    c = _campaign(status=status)
    res = check_identity(
        hotkey="hk1",
        registered_hotkeys=["hk1"],
        campaign=c,
        baseline_commit="a" * 40,
    )
    assert not res.ok
    assert "expected open" in res.reason


def test_identity_rejects_unregistered():
    c = _campaign()
    res = check_identity(
        hotkey="hkX",
        registered_hotkeys=["hk1"],
        campaign=c,
        baseline_commit="a" * 40,
    )
    assert not res.ok
    assert "registered" in res.reason


def test_identity_rejects_baseline_mismatch():
    c = _campaign()
    res = check_identity(
        hotkey="hk1",
        registered_hotkeys=["hk1"],
        campaign=c,
        baseline_commit="f" * 40,
    )
    assert not res.ok
    assert "baseline" in res.reason


def test_integrity_hash_match():
    data = b"hello patch"
    expected = hash_patch_bytes(data)
    res = check_integrity(
        retrieval_url="https://example.com/stage0/campaigns/x/patches/h/1.diff",
        expected_patch_hash=expected,
        fetcher=lambda _u: data,
    )
    # URL allowlist may reject example.com — override by patching config in test
    assert res.ok or res.reason == "retrieval_url not allowlisted"


def test_integrity_with_allowlisted_url(monkeypatch):
    monkeypatch.setattr(config, "S3_BUCKET", "pareton-patches")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com/pareton")
    url = "https://cdn.example.com/pareton/stage0/campaigns/cid/patches/hk/1.diff"
    assert is_allowed_retrieval_url(url)
    data = b"abc"
    expected = hash_patch_bytes(data)
    res = check_integrity(
        retrieval_url=url,
        expected_patch_hash=expected,
        fetcher=lambda _u: data,
    )
    assert res.ok
    assert res.evidence["patch_bytes"] == data


def test_integrity_mismatch(monkeypatch):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    url = "https://cdn.example.com/stage0/campaigns/cid/patches/hk/1.diff"
    res = check_integrity(
        retrieval_url=url,
        expected_patch_hash="sha256:" + "0" * 64,
        fetcher=lambda _u: b"abc",
    )
    assert not res.ok
    assert res.reason == PATCH_HASH_MISMATCH


def test_patch_url_hotkey_extracts_segment():
    url = "https://cdn.example.com/stage0/campaigns/cid/patches/5Abc/1.diff"
    assert patch_url_hotkey(url) == "5Abc"
    assert patch_url_hotkey("https://cdn.example.com/nope") is None


def test_integrity_rejects_hotkey_mismatch(monkeypatch):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    url = "https://cdn.example.com/stage0/campaigns/cid/patches/victim/1.diff"
    res = check_integrity(
        retrieval_url=url,
        expected_patch_hash="sha256:" + "0" * 64,
        hotkey="attacker",
        fetcher=lambda _u: b"abc",
    )
    assert not res.ok
    assert res.reason == "retrieval_url hotkey mismatch"


def test_integrity_accepts_matching_hotkey(monkeypatch):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://cdn.example.com")
    monkeypatch.setattr(config, "S3_PREFIX", "stage0")
    url = "https://cdn.example.com/stage0/campaigns/cid/patches/hk1/1.diff"
    data = b"abc"
    res = check_integrity(
        retrieval_url=url,
        expected_patch_hash=hash_patch_bytes(data),
        hotkey="hk1",
        fetcher=lambda _u: data,
    )
    assert res.ok


def test_patch_fingerprint_strips_blank_lines_without_git_diff():
    assert patch_fingerprint_bytes(b"alpha\n\n\nomega\n") == patch_fingerprint_bytes(
        b"alpha\nomega\n"
    )


def test_patch_fingerprint_strips_trailing_whitespace():
    assert patch_fingerprint_bytes(b"alpha   \nomega\t\n") == patch_fingerprint_bytes(
        b"alpha\nomega\n"
    )


def test_patch_fingerprint_strips_comments_from_git_hunk_content():
    first = b"""diff --git a/vllm/x.py b/vllm/x.py
--- a/vllm/x.py
+++ b/vllm/x.py
@@ -1 +1 @@
-x = 1  # old note
+x = 1  # first note
"""
    second = first.replace(b"first note", b"changed note")
    assert hash_patch_bytes(first) != hash_patch_bytes(second)
    assert patch_fingerprint_bytes(first) == patch_fingerprint_bytes(second)


def test_patch_fingerprint_ignores_volatile_git_metadata():
    first = b"""diff --git a/vllm/x.py b/vllm/x.py
index 1111111..2222222 100644
--- a/vllm/x.py
+++ b/vllm/x.py
@@ -1 +1,2 @@ def run():
 x = 1
+# first note
"""
    second = b"""diff --git a/vllm/x.py b/vllm/x.py
index 1111111..3333333 100644
--- a/vllm/x.py
+++ b/vllm/x.py
@@ -1 +1,3 @@ def run():
 x = 1
+# changed note
+# another note
"""
    assert hash_patch_bytes(first) != hash_patch_bytes(second)
    assert patch_fingerprint_bytes(first) == patch_fingerprint_bytes(second)


def test_patch_fingerprint_tracks_block_comments_across_hunk_sides():
    first = b"""diff --git a/vllm/x.cpp b/vllm/x.cpp
--- a/vllm/x.cpp
+++ b/vllm/x.cpp
@@ -1,5 +1,5 @@
 /*
-old note
+first note
 */
-x = 1; // old note
+x = 1; // first note
"""
    second = first.replace(b"first note", b"changed note")
    assert hash_patch_bytes(first) != hash_patch_bytes(second)
    assert patch_fingerprint_bytes(first) == patch_fingerprint_bytes(second)


def test_patch_fingerprint_keeps_python_floor_division():
    half = b"""diff --git a/vllm/x.py b/vllm/x.py
--- a/vllm/x.py
+++ b/vllm/x.py
@@ -1 +1 @@
-result = n // 1
+result = n // 2
"""
    quarter = half.replace(b"n // 2", b"n // 4")
    assert patch_fingerprint_bytes(half) != patch_fingerprint_bytes(quarter)


def test_patch_fingerprint_keeps_c_preprocessor_directives():
    small = b"""diff --git a/vllm/kernel.cu b/vllm/kernel.cu
--- a/vllm/kernel.cu
+++ b/vllm/kernel.cu
@@ -1 +1 @@
-#define BLOCK_SIZE 64
+#define BLOCK_SIZE 128
"""
    large = small.replace(b"BLOCK_SIZE 128", b"BLOCK_SIZE 256")
    assert patch_fingerprint_bytes(small) != patch_fingerprint_bytes(large)


def test_patch_fingerprint_strips_comments_in_any_directory():
    first = b"""diff --git a/sglang/x.py b/sglang/x.py
--- a/sglang/x.py
+++ b/sglang/x.py
@@ -1 +1 @@
-# old note
+# first note
"""
    second = first.replace(b"first note", b"changed note")
    assert patch_fingerprint_bytes(first) == patch_fingerprint_bytes(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (b'+value = "# first"\n', b'+value = "# changed"\n'),
        (b'+url = "https://one.example"\n', b'+url = "https://two.example"\n'),
        (b"+value = '/* first */'\n", b"+value = '/* changed */'\n"),
        (b"+value = `// first`\n", b"+value = `// changed`\n"),
    ],
)
def test_patch_fingerprint_keeps_comment_markers_in_strings(first, second):
    assert patch_fingerprint_bytes(first) != patch_fingerprint_bytes(second)


def test_patch_fingerprint_keeps_code_changes():
    first = b"+result = original\n"
    second = b"+renamed = original\n"
    assert patch_fingerprint_bytes(first) != patch_fingerprint_bytes(second)


def test_patch_fingerprint_keeps_code_after_block_comment():
    first = b"/* note */ result = original\n"
    second = b"/* note */ renamed = original\n"
    assert patch_fingerprint_bytes(first) != patch_fingerprint_bytes(second)


def test_patch_fingerprint_preserves_comments_without_git_diff():
    assert patch_fingerprint_bytes(b"# first\n") != patch_fingerprint_bytes(
        b"# changed\n"
    )
