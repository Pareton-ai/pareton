"""Pull deploy activity probe."""

from contextlib import contextmanager

import ops.deploy_probe as probe


class FakeCursor:
    def __init__(self, row):
        self.row = row
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql):
        self.sql = sql

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, row):
        self.cursor_value = FakeCursor(row)

    def cursor(self):
        return self.cursor_value


def _connection(row):
    @contextmanager
    def fake_connection(*, readonly=False):
        assert readonly is True
        yield FakeConnection(row)

    return fake_connection


def test_probe_is_busy_for_submission_build(monkeypatch):
    monkeypatch.setattr(probe, "db_connection", _connection((True, False)))

    active = probe.probe_active_work()

    assert active.busy is True
    assert active.labels() == "submission_job"


def test_probe_is_busy_for_running_round(monkeypatch):
    monkeypatch.setattr(probe, "db_connection", _connection((False, True)))

    active = probe.probe_active_work()

    assert active.busy is True
    assert active.labels() == "round"


def test_probe_is_idle_only_when_both_work_types_are_idle(monkeypatch):
    monkeypatch.setattr(probe, "db_connection", _connection((False, False)))

    assert probe.probe_active_work().busy is False
    assert probe.main() == probe.EXIT_IDLE


def test_probe_failure_is_fail_closed(monkeypatch, capsys):
    @contextmanager
    def broken_connection(*, readonly=False):
        raise RuntimeError("database unavailable")
        yield

    monkeypatch.setattr(probe, "db_connection", broken_connection)

    assert probe.main() == probe.EXIT_ERROR
    assert capsys.readouterr().err == "deploy: database activity probe failed\n"
