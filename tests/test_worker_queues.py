"""Execution queues stay independent even while a build waits on flock."""

import multiprocessing
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from builder.lock import builder_storage_lock
from worker import main as worker_main

JOB = {"id": "submission-1", "hotkey": "hk1", "patch_hash": "abc"}
ROUND = {"id": "round-1", "ordinal": 32, "campaign_id": "campaign-1"}
RUN_ARGS = {
    "mock_build": False,
    "mock_bench": False,
    "mock_correctness_fail": False,
    "registered_hotkeys": None,
}


@pytest.fixture
def handlers(monkeypatch):
    mocks = {
        "claim_next_job": Mock(return_value=JOB),
        "claim_pending_round": Mock(return_value=ROUND),
        "process_submission": Mock(
            return_value=SimpleNamespace(ok=True, state="built", reason=None)
        ),
        "process_round": Mock(return_value="settled"),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(worker_main, name, mock)
    return mocks


@pytest.mark.parametrize("queue", ["submissions", "rounds"])
@pytest.mark.parametrize("has_work", [True, False])
def test_dedicated_worker_never_claims_other_queue(handlers, queue, has_work):
    own_claim, other_claim = (
        ("claim_next_job", "claim_pending_round")
        if queue == "submissions"
        else ("claim_pending_round", "claim_next_job")
    )
    if not has_work:
        handlers[own_claim].return_value = None
    assert worker_main.run_once(queue=queue, **RUN_ARGS) is has_work
    handlers[own_claim].assert_called_once_with()
    handlers[other_claim].assert_not_called()
    other_handler = "process_round" if queue == "submissions" else "process_submission"
    handlers[other_handler].assert_not_called()
    if has_work and queue == "submissions":
        handlers["process_submission"].assert_called_once_with(
            JOB, registered_hotkeys=["hk1"], mock_build=False
        )
    elif has_work:
        handlers["process_round"].assert_called_once_with(
            ROUND, mock_bench=False, mock_correctness_fail=False
        )


def test_combined_worker_prioritizes_pending_round(handlers):
    assert worker_main.run_once(**RUN_ARGS)
    handlers["process_round"].assert_called_once()
    handlers["claim_next_job"].assert_not_called()


def test_combined_worker_builds_when_no_round_is_pending(handlers):
    handlers["claim_pending_round"].return_value = None
    assert worker_main.run_once(**RUN_ARGS)
    handlers["process_submission"].assert_called_once()


@pytest.mark.parametrize("queue", ["all", "submissions", "rounds"])
def test_cli_selects_queue(monkeypatch, queue):
    run_once = Mock(return_value=False)
    monkeypatch.setattr(worker_main, "run_once", run_once)
    monkeypatch.setattr(worker_main, "_heartbeat_loop", lambda *args, **kwargs: None)
    assert worker_main.main(["--queue", queue, "--once"]) == 0
    assert run_once.call_args.kwargs["queue"] == queue


def _isolated_worker(queue, lock_path, build_started, done):
    """Spawn-safe child with fake persistence and the real builder lock."""
    import config
    from builder.lock import serialized_build_storage

    config.BUILDER_LOCK_PATH = lock_path

    def wrong_queue():
        raise AssertionError(f"{queue} worker touched the other queue")

    worker_main.claim_next_job = lambda: JOB
    worker_main.claim_pending_round = lambda: ROUND
    if queue == "submissions":
        worker_main.claim_pending_round = wrong_queue
    else:
        worker_main.claim_next_job = wrong_queue

    @serialized_build_storage
    def build():
        return SimpleNamespace(ok=True, state="built", reason=None)

    def process_submission(*args, **kwargs):
        build_started.set()
        return build()

    worker_main.process_submission = process_submission
    worker_main.process_round = lambda *args, **kwargs: "settled"
    worker_main.count_pending_jobs = lambda: 0
    worker_main.count_pending_rounds = lambda: 0
    if queue == "submissions":
        # Exercise the production drain handler while flock is blocked. The
        # parent signals only this process, leaving its build to finish.
        assert worker_main.main(["--queue", queue]) == 0
    else:
        assert worker_main.run_once(queue=queue, **RUN_ARGS)
    done.set()


def test_round_completes_while_submission_process_waits_on_builder_lock(tmp_path: Path):
    ctx = multiprocessing.get_context("spawn")
    lock_path = tmp_path / "builder-storage.lock"
    build_started = ctx.Event()
    build_done, round_done = ctx.Event(), ctx.Event()
    build_worker = ctx.Process(
        target=_isolated_worker,
        args=("submissions", lock_path, build_started, build_done),
    )
    round_worker = ctx.Process(
        target=_isolated_worker,
        args=("rounds", lock_path, build_started, round_done),
    )
    try:
        # A manual baseline build owns storage before a submission is claimed.
        with builder_storage_lock(blocking=True, path=lock_path) as acquired:
            assert acquired
            build_worker.start()
            assert build_started.wait(timeout=15)
            assert not build_done.is_set()
            os.kill(build_worker.pid, signal.SIGTERM)
            # The round arrives only after the submission entered its build.
            round_worker.start()
            round_worker.join(timeout=15)
            assert round_worker.exitcode == 0
            assert round_done.is_set()
            assert build_worker.is_alive()
            assert not build_done.is_set()
        build_worker.join(timeout=15)
        assert build_worker.exitcode == 0
        assert build_done.is_set()
    finally:
        for process in (build_worker, round_worker):
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
