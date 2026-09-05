"""Unit tests for scrubbing a void detail before it becomes public."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


from round.void_detail import MAX_VOID_DETAIL, REDACTED, sanitize_void_detail


def test_a_plain_detail_survives_untouched():
    assert sanitize_void_detail(">1048576 bytes") == ">1048576 bytes"


def test_empty_detail_is_empty_so_the_column_can_be_null():
    assert sanitize_void_detail("") == ""
    assert sanitize_void_detail(None) == ""
    assert sanitize_void_detail("   \n  ") == ""


def test_a_presigned_url_loses_its_signature():
    """round_job raises RoundInfraError(VOID_TRACE_UNAVAILABLE, url) with a
    presigned trace URL. The column is public, so the query must not survive."""
    detail = (
        "https://bucket.s3.amazonaws.com/traces/abc.json"
        "?X-Amz-Signature=deadbeefcafe&X-Amz-Credential=AKIAEXAMPLE"
    )
    out = sanitize_void_detail(detail)
    assert out == f"https://bucket.s3.amazonaws.com/traces/abc.json?{REDACTED}"
    assert "deadbeefcafe" not in out
    assert "AKIAEXAMPLE" not in out


def test_the_host_and_path_stay_so_the_detail_is_still_useful():
    out = sanitize_void_detail("https://api.provider.io/v1/pods?token=abc")
    assert "api.provider.io/v1/pods" in out


def test_a_url_without_a_query_is_left_alone():
    detail = "https://ghcr.io/pareton-ai/engine"
    assert sanitize_void_detail(detail) == detail


@pytest.mark.parametrize(
    "secret",
    [
        "token=sk-live-abc123",
        "api_key=abc123",
        "Authorization: Bearer abc123",
        "secret=hunter2",
        "AWS_ACCESS_KEY=AKIAEXAMPLE",
    ],
)
def test_loose_credential_pairs_are_redacted(secret: str):
    out = sanitize_void_detail(f"provision failed: {secret} while retrying")
    assert REDACTED in out
    for leaked in ("sk-live-abc123", "abc123", "hunter2", "AKIAEXAMPLE"):
        assert leaked not in out
    # The surrounding prose is the part an operator reads.
    assert "provision failed" in out
    assert "while retrying" in out


@pytest.mark.parametrize(
    "blob",
    [
        '{"api_key": "secret"}',
        "{'Authorization': 'Bearer secret'}",
        '{"api_key":"secret"}',
        '{"Authorization": "Bearer secret"}',
    ],
)
def test_quoted_json_and_python_dict_credentials_are_redacted(blob: str):
    """Provider adapters include raw response bodies in ProvisionError."""
    out = sanitize_void_detail(f"provision failed: {blob} while retrying")
    assert REDACTED in out
    assert "secret" not in out
    assert "Bearer" not in out
    assert "provision failed" in out
    assert "while retrying" in out


def test_escaped_quotes_inside_a_json_value_do_not_leave_the_secret():
    """`\"` is a character in the value, not the end of the quoted value."""
    detail = json.dumps({"password": 'prefix"SYNTHETIC_SECRET'})
    out = sanitize_void_detail(detail)
    assert "SYNTHETIC_SECRET" not in out
    assert REDACTED in out


def test_json_encoded_credential_fields_are_redacted():
    """A JSON body nested inside another JSON string still carries the key."""
    detail = json.dumps({"body": json.dumps({"api_key": "SYNTHETIC_SECRET"})})
    out = sanitize_void_detail(detail)
    assert "SYNTHETIC_SECRET" not in out
    assert REDACTED in out


def test_provider_http_body_is_scrubbed_on_the_void_write_path():
    """Shadeform/Runpod/Targon put resp.text[:300] into ProvisionError.

    round_job raises RoundInfraError with str(exc); void_round sanitizes
    that detail before the column (and the public round response) sees it.
    """
    from gpu.errors import ProvisionError
    from round.store import VOID_POD_PROVISION_FAILED
    from worker.round_job import RoundInfraError

    escaped = json.dumps({"password": 'prefix"SYNTHETIC_SECRET'})
    encoded = json.dumps({"body": json.dumps({"api_key": "SYNTHETIC_SECRET"})})
    for body in (escaped, encoded):
        exc = ProvisionError(
            f"Shadeform POST /instances/create failed HTTP 403: {body[:300]}"
        )
        infra = RoundInfraError(VOID_POD_PROVISION_FAILED, str(exc))
        published = sanitize_void_detail(infra.detail)
        assert "SYNTHETIC_SECRET" not in published
        assert REDACTED in published
        assert "failed HTTP 403" in published


def test_terminal_escapes_and_newlines_are_flattened():
    """Provider output arrives coloured and multi-line; a round row is neither."""
    out = sanitize_void_detail("boom \x1b[31mred\x1b[0m\nsecond line\r\tthird")
    assert out == "boom red second line third"
    assert "\x1b" not in out


def test_null_bytes_and_control_characters_go():
    out = sanitize_void_detail("bad\x00detail\x07here")
    assert "\x00" not in out and "\x07" not in out
    assert out == "bad detail here"


def test_a_long_detail_is_truncated_within_the_limit():
    """A stack trace must not turn a round row into a log sink."""
    out = sanitize_void_detail("x" * 5000)
    assert len(out) <= MAX_VOID_DETAIL
    assert out.endswith("...")


def test_the_limit_is_honoured_exactly():
    out = sanitize_void_detail("y" * 1000, limit=20)
    assert len(out) == 20


def test_a_detail_at_the_limit_is_not_marked_truncated():
    out = sanitize_void_detail("z" * MAX_VOID_DETAIL)
    assert len(out) == MAX_VOID_DETAIL
    assert not out.endswith("...")
