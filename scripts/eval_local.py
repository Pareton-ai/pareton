#!/usr/bin/env python3
"""Run ``validator.gpu_eval`` locally without S3 (GPU pod smoke test).

Requires ``CACHEON_DATABASE_URL`` with a pending eval job in Postgres. Set
``CACHEON_SKIP_S3=1`` and run ``python -m validator.gpu_eval`` (or this wrapper).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("CACHEON_SKIP_S3", "1")

from validator.gpu_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
