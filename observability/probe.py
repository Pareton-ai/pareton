"""Deployment log probe: read the current probe file and emit on change.

The release coordinator writes /run/pareton-deploy/probe.json before log
verification (stage-2 spec section 4.5). Long-running services poll it from
a read-only daemon thread every 5 seconds and emit one ``deployment_probe``
lifecycle event whenever the probe_id changes, so a verify against already
running services never needs to restart them.

This module must stay side-effect free apart from the emitted event: no
database, GPU, or chain access, and no business retries.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from observability.events import _emit

logger = logging.getLogger(__name__)

PROBE_POLL_S = 5.0
DEFAULT_PROBE_FILE = "/run/pareton-deploy/probe.json"


def probe_file() -> Path:
    override = os.environ.get("PARETON_PROBE_FILE")
    return Path(override) if override else Path(DEFAULT_PROBE_FILE)


def current_probe() -> dict | None:
    """Read the probe file; None when absent or unparseable."""
    try:
        data = json.loads(probe_file().read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def poll_once(unit: str, last_probe_id: str | None) -> str | None:
    """Emit one probe event if the probe_id changed. Returns the last seen id.

    A missing or unparseable file keeps the previous id: the next verify
    writes a fresh one and the change re-triggers emission.
    """
    probe = current_probe()
    if probe is None:
        return last_probe_id
    probe_id = probe.get("probe_id")
    if not probe_id or probe_id == last_probe_id:
        return last_probe_id
    _emit(
        "deployment_probe",
        probe_id=probe_id,
        unit=unit,
        target_commit=probe.get("target_commit"),
        host=probe.get("host"),
    )
    return probe_id


def run_probe_loop(unit: str) -> None:
    """Daemon thread body: poll the probe file forever."""
    last: str | None = None
    while True:
        last = poll_once(unit, last)
        threading.Event().wait(PROBE_POLL_S)
