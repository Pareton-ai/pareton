"""Environment configuration for validator submission fee exemptions."""

import runpy

import pytest

import config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, frozenset()),
        ("", frozenset()),
        (" , , ", frozenset()),
        (" hk1, hk2,,hk1, HK1 ", frozenset({"hk1", "hk2", "HK1"})),
    ],
)
def test_submission_fee_exempt_hotkeys(monkeypatch, raw, expected):
    key = "PARETON_SUBMISSION_FEE_EXEMPT_HOTKEYS"
    if raw is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, raw)
    settings = runpy.run_path(str(config.REPO_ROOT / "config.py"))
    assert settings["SUBMISSION_FEE_EXEMPT_HOTKEYS"] == expected


def test_legacy_fee_environment_is_only_read_when_seeding(monkeypatch):
    monkeypatch.setenv("PARETON_SUBMISSION_FEE_TAO", "not-a-fee")
    settings = runpy.run_path(str(config.REPO_ROOT / "config.py"))
    assert "SUBMISSION_FEE_TAO" not in settings
    assert settings["seed_submission_fee_tao"]() == "not-a-fee"
