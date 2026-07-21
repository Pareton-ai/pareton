"""Unit tests for hermetic Dockerfile generation and registry digest refs."""

from __future__ import annotations

import pytest

from builder.hermetic import build_engine_image, dockerfile_for_patch
from builder.registry import (
    baseline_build_image_ref,
    baseline_engine_image_ref,
    pullable_digest_ref,
)


@pytest.mark.unit
def test_dockerfile_nonempty_has_apply_and_entrypoint():
    text = dockerfile_for_patch(
        allow_empty_patch=False,
        patch_bytes=b"diff --git a/x b/x\n",
    )
    assert "git apply --whitespace=nowarn /tmp/submission.diff" in text
    assert "pip install --no-deps --no-build-isolation -e ." in text
    assert "|| true" not in text
    assert "rm -rf /src/.git" in text
    assert 'ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]' in text


@pytest.mark.unit
def test_dockerfile_empty_allowed_skips_apply():
    text = dockerfile_for_patch(allow_empty_patch=True, patch_bytes=b"")
    assert "git apply" not in text
    assert "pip install --no-deps --no-build-isolation -e ." in text
    assert "|| true" not in text
    assert 'ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]' in text


@pytest.mark.unit
def test_dockerfile_empty_not_allowed_still_has_apply():
    text = dockerfile_for_patch(allow_empty_patch=False, patch_bytes=b"   \n")
    assert "git apply --whitespace=nowarn /tmp/submission.diff" in text


@pytest.mark.unit
def test_dockerfile_whitespace_empty_with_allow_skips_apply():
    text = dockerfile_for_patch(allow_empty_patch=True, patch_bytes=b"  \n\t")
    assert "git apply" not in text


@pytest.mark.unit
def test_build_rejects_empty_patch_without_allow(tmp_path):
    result = build_engine_image(
        baseline_repo="https://example.com/vllm.git",
        baseline_commit="a" * 40,
        base_image="ghcr.io/pareton-ai/pareton-baseline@sha256:" + ("b" * 64),
        patch_bytes=b"",
        patch_hash="sha256:" + ("c" * 64),
        work_root=tmp_path,
        push=False,
        allow_empty_patch=False,
    )
    assert not result.ok
    assert result.reason == "empty_patch_not_allowed"


@pytest.mark.unit
def test_pullable_digest_ref_bare_and_pinned():
    digest = "sha256:" + ("a" * 64)
    assert (
        pullable_digest_ref(digest, image="pareton-engine")
        == f"ghcr.io/pareton-ai/pareton-engine@{digest}"
    )
    pinned = f"ghcr.io/other/engine@{digest.upper()}"
    assert pullable_digest_ref(pinned, image="pareton-engine") == (
        f"ghcr.io/other/engine@{digest}"
    )
    assert baseline_build_image_ref(digest) == (
        f"ghcr.io/pareton-ai/pareton-baseline@{digest}"
    )
    assert baseline_engine_image_ref(digest) == (
        f"ghcr.io/pareton-ai/pareton-engine@{digest}"
    )


@pytest.mark.unit
def test_pullable_digest_ref_rejects_bad():
    with pytest.raises(ValueError):
        pullable_digest_ref("not-a-digest", image="pareton-engine")
