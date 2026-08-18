"""Watcher loop: keep scanning, reconnect after a failed scan, drain cleanly."""

import threading

from worker.watcher import scan_cycle


def test_scan_cycle_returns_same_client_on_success(monkeypatch):
    client = object()
    calls = []

    monkeypatch.setattr(
        "worker.watcher.scan_chain",
        lambda *_a, **_k: calls.append(1) or (["sid"], ["hk"]),
    )

    drain = threading.Event()
    assert scan_cycle(client, drain) is client
    assert calls == [1]


def test_scan_cycle_always_ready_to_sleep(monkeypatch):
    """A successful ingest must not look like 'did work' to _run_loop."""
    from worker.main import _run_loop

    monkeypatch.setattr(
        "worker.watcher.scan_chain",
        lambda *_a, **_k: (["sid"], ["hk"]),
    )
    drain = threading.Event()
    client = object()
    cycles = []

    def cycle():
        cycles.append(scan_cycle(client, drain))
        if len(cycles) == 2:
            drain.set()
        return False

    _run_loop(cycle, drain, poll_interval_s=0.0)
    assert cycles == [client, client]


def test_failed_scan_closes_client_before_reconnect(monkeypatch):
    closed = []

    class FakeClient:
        def close(self):
            closed.append(1)

    monkeypatch.setattr(
        "worker.watcher.scan_chain",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("websocket dead")),
    )
    drain = threading.Event()
    assert scan_cycle(FakeClient(), drain) is None
    assert closed == [1]


def test_failed_scan_drops_client_so_next_cycle_reconnects(monkeypatch):
    live = object()
    connects = []

    def boom(*_a, **_k):
        raise RuntimeError("websocket dead")

    monkeypatch.setattr("worker.watcher.scan_chain", boom)
    monkeypatch.setattr(
        "worker.watcher._connect_subtensor",
        lambda: connects.append(1) or live,
    )

    drain = threading.Event()
    assert scan_cycle(None, drain) is None
    assert connects == [1]

    monkeypatch.setattr(
        "worker.watcher.scan_chain",
        lambda *_a, **_k: ([], ["hk"]),
    )
    assert scan_cycle(None, drain) is live
    assert connects == [1, 1]


def test_drained_cycle_does_not_scan(monkeypatch):
    monkeypatch.setattr(
        "worker.watcher.scan_chain",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("scanned after drain")),
    )
    monkeypatch.setattr(
        "worker.watcher._connect_subtensor",
        lambda: (_ for _ in ()).throw(AssertionError("connected after drain")),
    )
    drain = threading.Event()
    drain.set()
    client = object()
    assert scan_cycle(client, drain) is client
