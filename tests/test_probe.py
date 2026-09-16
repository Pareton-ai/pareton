"""Offline tests for the deployment probe poller (stage-2 spec 7.2)."""

import json
import logging
from pathlib import Path

import pytest

from observability import probe


@pytest.fixture()
def probe_file(tmp_path, monkeypatch):
    path = tmp_path / "probe.json"
    monkeypatch.setenv("PARETON_PROBE_FILE", str(path))
    return path


def events(caplog):
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "pareton.lifecycle"
    ]


def test_missing_file_is_silent(probe_file, caplog):
    with caplog.at_level(logging.INFO, logger="pareton.lifecycle"):
        assert probe.poll_once("pareton-api", None) is None
    assert events(caplog) == []


def test_emits_once_per_probe_id(probe_file, caplog):
    probe_file.write_text(
        json.dumps(
            {
                "probe_id": "p1",
                "target_commit": "c2",
                "host": "h",
                "issued_at": "2026-09-12T00:00:00Z",
            }
        )
    )
    with caplog.at_level(logging.INFO, logger="pareton.lifecycle"):
        seen = probe.poll_once("pareton-api", None)
        probe.poll_once("pareton-api", seen)
        seen = probe.poll_once("pareton-api", seen)
    emitted = events(caplog)
    assert len(emitted) == 1
    assert emitted[0]["event"] == "deployment_probe"
    assert emitted[0]["probe_id"] == "p1"
    assert emitted[0]["unit"] == "pareton-api"
    assert emitted[0]["target_commit"] == "c2"


def test_new_probe_id_emits_again(probe_file, caplog):
    probe_file.write_text(json.dumps({"probe_id": "p1"}))
    with caplog.at_level(logging.INFO, logger="pareton.lifecycle"):
        seen = probe.poll_once("pareton-worker", None)
        probe_file.write_text(json.dumps({"probe_id": "p2"}))
        seen = probe.poll_once("pareton-worker", seen)
    emitted = events(caplog)
    assert [e["probe_id"] for e in emitted] == ["p1", "p2"]


def test_corrupt_file_keeps_previous_id(probe_file, caplog):
    probe_file.write_text(json.dumps({"probe_id": "p1"}))
    with caplog.at_level(logging.INFO, logger="pareton.lifecycle"):
        seen = probe.poll_once("pareton-api", None)
        probe_file.write_text("not json{")
        assert probe.poll_once("pareton-api", seen) == "p1"
    assert len(events(caplog)) == 1
