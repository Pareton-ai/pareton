"""Offline guards for Neon e2e DB binding (no network)."""

from __future__ import annotations

import pytest

from e2e_db import require_e2e_database_url

pytestmark = pytest.mark.unit


def test_e2e_skips_without_test_url(monkeypatch):
    monkeypatch.delenv("PARETON_TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("PARETON_DATABASE_URL", "postgresql://u:p@main.example/neondb")
    with pytest.raises(pytest.skip.Exception):
        require_e2e_database_url()


def test_e2e_refuses_same_host_as_main(monkeypatch):
    monkeypatch.setenv(
        "PARETON_DATABASE_URL",
        "postgresql://u:p@ep-main.example/neondb?sslmode=require",
    )
    monkeypatch.setenv(
        "PARETON_TEST_DATABASE_URL",
        "postgresql://u:p@ep-main.example/neondb?sslmode=require",
    )
    with pytest.raises(pytest.fail.Exception, match="same host"):
        require_e2e_database_url()


def test_e2e_accepts_distinct_test_host(monkeypatch):
    monkeypatch.setenv(
        "PARETON_DATABASE_URL",
        "postgresql://u:p@ep-main.example/neondb?sslmode=require",
    )
    monkeypatch.setenv(
        "PARETON_TEST_DATABASE_URL",
        "postgresql://u:p@ep-test.example/neondb?sslmode=require",
    )
    assert require_e2e_database_url().startswith("postgresql://")
