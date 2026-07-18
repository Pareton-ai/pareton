"""Output / evidence-bundle directory layout."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bench.schemas import BenchReport

logger = logging.getLogger(__name__)


class OutputLayout:
    """
    output/
    ├── bench_report.json
    ├── harness.log
    └── evidence/
        ├── env/
        ├── weights/
        ├── correctness/
        ├── perf_screen/
        └── sla_bench/
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.evidence = self.root / "evidence"
        self.env_dir = self.evidence / "env"
        self.weights_dir = self.evidence / "weights"
        self.correctness_dir = self.evidence / "correctness"
        self.perf_screen_dir = self.evidence / "perf_screen"
        self.sla_bench_dir = self.evidence / "sla_bench"
        self.report_path = self.root / "bench_report.json"
        self.log_path = self.root / "harness.log"

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for d in (
            self.env_dir,
            self.weights_dir,
            self.correctness_dir,
            self.perf_screen_dir,
            self.sla_bench_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def write_weights_manifest(
        self, manifest: dict[str, Any], *, aggregate: str
    ) -> Path:
        """Write evidence/weights/weights_manifest.json (manifest + aggregate)."""
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo": manifest.get("repo"),
            "revision": manifest.get("revision"),
            "files": manifest.get("files", []),
            "aggregate": aggregate,
        }
        path = self.weights_dir / "weights_manifest.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return path

    def write_report(self, report: BenchReport) -> Path:
        data = report.to_dict()
        self.report_path.write_text(
            json.dumps(data, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return self.report_path

    def write_env_dumps(self, dumps: dict[str, str]) -> None:
        for name, text in dumps.items():
            safe = name.replace("/", "_")
            (self.env_dir / safe).write_text(text, encoding="utf-8")

    def append_log(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


class JsonlFileHandler(logging.Handler):
    """Emit structured JSON lines into harness.log via OutputLayout."""

    def __init__(self, layout: OutputLayout) -> None:
        super().__init__()
        self.layout = layout

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.layout.append_log(
                {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "time": self.format(record) if self.formatter else record.created,
                }
            )
        except Exception:  # noqa: BLE001 — logging must not raise
            self.handleError(record)
