"""Attempt-scoped phase + heartbeat writer. Failures are logged and dropped."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Protocol

import config
from bench.phases import coerce_phase, coerce_progress

logger = logging.getLogger(__name__)

PhaseWriter = Callable[..., bool]
HeartbeatWriter = Callable[..., bool]


class PhaseSink(Protocol):
    """What a bench run needs in order to report progress."""

    def set(self, phase: Any, /, **progress: Any) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class NullPhaseReporter:
    """No-op when there is no claimed attempt to write against."""

    def set(self, phase: Any, /, **progress: Any) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def __enter__(self) -> NullPhaseReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class PhaseReporter:
    """Writes phase changes and heartbeats for one attempt of one job."""

    def __init__(
        self,
        *,
        job_id: int,
        attempt: int,
        phase_writer: PhaseWriter | None = None,
        heartbeat_writer: HeartbeatWriter | None = None,
        interval_s: float | None = None,
    ) -> None:
        self.job_id = int(job_id)
        self.attempt = int(attempt)
        self._interval_s = (
            config.JOB_HEARTBEAT_INTERVAL_S if interval_s is None else interval_s
        )
        self._phase_writer = phase_writer
        self._heartbeat_writer = heartbeat_writer
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: str | None = None
        # Set once a write is rejected: this attempt no longer owns the row.
        self._orphaned = False

    def _resolve_phase_writer(self) -> PhaseWriter:
        if self._phase_writer is None:
            from campaign.store import set_job_phase

            self._phase_writer = set_job_phase
        return self._phase_writer

    def _resolve_heartbeat_writer(self) -> HeartbeatWriter:
        if self._heartbeat_writer is None:
            from campaign.store import touch_job_heartbeat

            self._heartbeat_writer = touch_job_heartbeat
        return self._heartbeat_writer

    def set(self, phase: Any, /, **progress: Any) -> None:
        """Record the current phase. Unknown names and failures are dropped."""
        name = coerce_phase(phase)
        if name is None:
            logger.debug("ignoring unknown bench phase %r", phase)
            return
        detail = coerce_progress(progress)
        with self._lock:
            if self._orphaned:
                return
            # Same phase, no new detail: heartbeat already covers liveness.
            if name == self._current and not detail:
                return
        try:
            landed = self._resolve_phase_writer()(
                job_id=self.job_id,
                attempt=self.attempt,
                phase=name,
                progress=detail,
            )
        except Exception as exc:  # noqa: BLE001 - progress must never fail a bench
            logger.warning("phase write failed job=%s %s: %s", self.job_id, name, exc)
            return
        if not landed:
            self._mark_orphaned("phase")
            return
        with self._lock:
            if not self._orphaned:
                self._current = name
        logger.info("bench phase job=%s attempt=%s %s", self.job_id, self.attempt, name)

    def _mark_orphaned(self, source: str) -> None:
        with self._lock:
            if self._orphaned:
                return
            self._orphaned = True
        logger.info(
            "job %s attempt %s stopped accepting progress (%s write rejected); "
            "the job has settled or another attempt owns it",
            self.job_id,
            self.attempt,
            source,
        )

    def _beat(self) -> None:
        with self._lock:
            if self._orphaned:
                return
        try:
            landed = self._resolve_heartbeat_writer()(
                job_id=self.job_id, attempt=self.attempt
            )
        except Exception as exc:  # noqa: BLE001 - liveness must never fail a bench
            logger.warning("heartbeat write failed job=%s: %s", self.job_id, exc)
            return
        if not landed:
            self._mark_orphaned("heartbeat")

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._beat()

    def start(self) -> None:
        """Heartbeat in the background; the bench call itself can block for an hour."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name=f"phase-heartbeat-{self.job_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> PhaseReporter:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def reporter_for_job(row: dict[str, Any]) -> PhaseReporter | NullPhaseReporter:
    """Real reporter when `claim_next_job` set `job_id`/`job_attempt`, else no-op."""
    job_id = row.get("job_id")
    attempt = row.get("job_attempt")
    if job_id is None or attempt is None:
        return NullPhaseReporter()
    try:
        return PhaseReporter(job_id=int(job_id), attempt=int(attempt))
    except (TypeError, ValueError):
        return NullPhaseReporter()
