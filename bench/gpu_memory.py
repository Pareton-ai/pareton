"""Peak GPU memory of a serving engine, for the scoring rule's memory cap.

A campaign can cap how much more GPU memory a candidate may use than the
round's baseline (``scoring_rule.max_extra_gpu_memory_mib``). The harness is
the only trusted witness: anything the candidate engine reports about itself
comes from code the miner can patch. So the round polls ``nvidia-smi`` on the
host while each engine serves its replay and keeps every GPU's peak.

Sampling starts once the engine is healthy and ends before it is torn down,
so the previous engine's memory, still being released, is never counted. The
window is the engine's whole replay, warmup included, for the baseline and the
candidates alike: the cap is on the peak, whenever it happens.

This is a sampled limit, not an instantaneous one. A spike that comes and goes
between two polls is not seen. A window with a stretch longer than
``MAX_BLIND_S`` without a usable sample has no peak at all, so a cap is never
judged on what was seen before the sampler went blind.
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_QUERY = ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
_QUERY_TIMEOUT_S = 10.0
SAMPLE_INTERVAL_S = 2.0
# One missed poll is noise. Longer than this without a usable sample and the
# peak could have happened unseen, so the window counts as unmeasured.
MAX_BLIND_S = 30.0

MemoryReader = Callable[[], "list[float] | None"]


def read_gpu_memory_mib() -> list[float] | None:
    """Memory in use on each GPU, in MiB and in nvidia-smi's order.

    None when nvidia-smi is missing, fails, times out or prints something
    else, including a negative or non-finite number: a sample that can't be
    trusted is dropped, never guessed.
    """
    try:
        result = subprocess.run(
            list(_QUERY),
            capture_output=True,
            text=True,
            timeout=_QUERY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        used = [float(line) for line in result.stdout.splitlines() if line.strip()]
    except ValueError:
        return None
    if any(not math.isfinite(value) or value < 0 for value in used):
        return None
    return used or None


class GpuMemorySampler:
    """Polls GPU memory in a background thread and keeps each GPU's peak."""

    def __init__(
        self,
        *,
        read: MemoryReader = read_gpu_memory_mib,
        interval_s: float = SAMPLE_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._read = read
        self._interval_s = interval_s
        self._clock = clock
        # When the current stretch without a usable sample began: the start
        # of the window, then the time of each usable sample.
        self._last_usable: float | None = None
        self._blind = False
        self._peak: list[float] | None = None
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def peak_mib(self) -> list[float] | None:
        """Per-GPU peak over every sample.

        None when no sample was usable, or when the sampler was blind for
        longer than ``MAX_BLIND_S`` at any point of the window.
        """
        return None if self._peak is None or self._blind else list(self._peak)

    @property
    def samples(self) -> int:
        return self._samples

    def _sample(self) -> None:
        used = self._read()
        now = self._clock()
        if self._last_usable is not None and now - self._last_usable > MAX_BLIND_S:
            if not self._blind:
                logger.warning(
                    "no usable GPU memory sample for %.0f s; the window is unmeasured",
                    now - self._last_usable,
                )
            self._blind = True
        if used is None:
            return
        if self._peak is None:
            self._peak = list(used)
        elif len(used) != len(self._peak):
            # The GPU set changed under the engine; the peaks no longer line up.
            logger.warning(
                "GPU memory sample has %d GPUs, expected %d; dropped",
                len(used),
                len(self._peak),
            )
            return
        else:
            self._peak = [max(a, b) for a, b in zip(self._peak, used, strict=True)]
        self._last_usable = now
        self._samples += 1

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def __enter__(self) -> GpuMemorySampler:
        self._last_usable = self._clock()
        self._sample()
        self._thread = threading.Thread(
            target=self._loop, name="gpu-memory-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_QUERY_TIMEOUT_S + self._interval_s)
            if self._thread.is_alive():
                # A hung nvidia-smi: leave the daemon thread. Nothing was
                # sampled while it hung, so the window is unmeasured.
                self._blind = True
                return
        # The engine is still up here, so one last sample closes the window.
        self._sample()


class FixedGpuMemory:
    """A sampler with a known answer: mock engines, or a rule without a cap."""

    def __init__(self, peak_mib: Sequence[float] | None = None) -> None:
        self.peak_mib = None if peak_mib is None else [float(v) for v in peak_mib]

    def __enter__(self) -> FixedGpuMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def merge_peaks(
    first: Sequence[float] | None, second: Sequence[float] | None
) -> list[float] | None:
    """Per-GPU maximum of two peak lists; None when they can't be compared."""
    if first is None or second is None:
        return None if second is None else list(second)
    if len(first) != len(second):
        return None
    return [max(a, b) for a, b in zip(first, second, strict=True)]


def gpu_memory_report(
    baseline_peak_mib: Sequence[float] | None,
    candidate_peak_mib: Sequence[float] | None,
    cap_mib: float,
) -> dict[str, Any] | None:
    """A candidate's memory against the baseline's, GPU by GPU.

    ``extra_mib`` is the largest per-GPU excess (negative when the candidate
    uses less everywhere). None when either side has no usable peak or the
    GPU counts differ, so the cap is never applied to a guess.
    """
    if (
        not baseline_peak_mib
        or not candidate_peak_mib
        or len(baseline_peak_mib) != len(candidate_peak_mib)
    ):
        return None
    extra = max(
        c - b for b, c in zip(baseline_peak_mib, candidate_peak_mib, strict=True)
    )
    return {
        "baseline_peak_mib": list(baseline_peak_mib),
        "candidate_peak_mib": list(candidate_peak_mib),
        "extra_mib": extra,
        "max_extra_gpu_memory_mib": cap_mib,
        "over_cap": extra > cap_mib,
    }
