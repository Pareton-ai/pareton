"""A round arriving during a blocked build runs in the separate worker."""

import multiprocessing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from builder.lock import builder_storage_lock


@pytest.mark.parametrize("pending_round", [True, False])
def test_combined_mode_checks_rounds_before_submissions(monkeypatch, pending_round):
    from worker import main as worker_main

    claim_job = Mock(return_value={"id": "s1", "hotkey": "hk1", "patch_hash": "abc"})
    round_row = {"id": "r32", "ordinal": 32, "campaign_id": "c1"}
    run_round = Mock()
    run_submission = Mock(
        return_value=SimpleNamespace(ok=True, state="built", reason=None)
    )
    monkeypatch.setattr(worker_main, "claim_next_job", claim_job)
    monkeypatch.setattr(
        worker_main, "claim_pending_round", lambda: round_row if pending_round else None
    )
    monkeypatch.setattr(worker_main, "process_round", run_round)
    monkeypatch.setattr(worker_main, "process_submission", run_submission)
    assert worker_main.run_once(
        mock_build=False,
        mock_bench=False,
        mock_correctness_fail=False,
        registered_hotkeys=None,
    )
    assert run_round.call_count == int(pending_round)
    assert claim_job.call_count == int(not pending_round)
    assert run_submission.call_count == int(not pending_round)


def _worker(queue, lock_path, started, finished):
    import config
    from builder.lock import serialized_build_storage
    from worker import main as worker_main

    config.BUILDER_LOCK_PATH = lock_path

    def wrong_queue():
        raise AssertionError(f"{queue} worker claimed the other queue")

    @serialized_build_storage
    def build():
        return SimpleNamespace(ok=True, state="built", reason=None)

    def submit(*args, **kwargs):
        started.set()
        result = build()
        finished.set()
        return result

    def evaluate(*args, **kwargs):
        finished.set()
        return "settled"

    worker_main.claim_next_job = lambda: {
        "id": "s1",
        "hotkey": "hk1",
        "patch_hash": "abc",
    }
    worker_main.claim_pending_round = lambda: {
        "id": "r32",
        "ordinal": 32,
        "campaign_id": "c1",
    }
    if queue == "submissions":
        worker_main.claim_pending_round = wrong_queue
    else:
        worker_main.claim_next_job = wrong_queue
    worker_main.process_submission = submit
    worker_main.process_round = evaluate
    worker_main.count_pending_jobs = lambda: 0
    assert worker_main.main(["--queue", queue, "--once"]) == 0


def test_round_runs_while_submission_waits_on_builder_lock(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    lock_path = tmp_path / "builder-storage.lock"
    started, built, evaluated = ctx.Event(), ctx.Event(), ctx.Event()
    submission = ctx.Process(
        target=_worker, args=("submissions", lock_path, started, built)
    )
    round_worker = ctx.Process(
        target=_worker, args=("rounds", lock_path, started, evaluated)
    )
    try:
        with builder_storage_lock(blocking=True, path=lock_path):
            submission.start()
            assert started.wait(timeout=15)
            round_worker.start()
            round_worker.join(timeout=15)
            assert round_worker.exitcode == 0
            assert evaluated.is_set()
            assert submission.is_alive()
            assert not built.is_set()
        submission.join(timeout=15)
        assert submission.exitcode == 0
        assert built.is_set()
    finally:
        for process in (submission, round_worker):
            if process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(timeout=5)
