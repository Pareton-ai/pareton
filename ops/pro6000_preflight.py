"""Check host-side campaign Python dependencies without Docker, GPU, or downloads."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "tokenizers",
    "jinja2",
    "datasets",
    "huggingface_hub",
    "requests",
    "psycopg2",
    "boto3",
    "bench.qualify_longform",
    "bench.preview_longform",
    "bench.main",
    "worker.round_job",
    "campaign.seed",
    "ops.pro6000_model_volume",
)


def check(import_module=importlib.import_module):
    errors = []
    pins = dict(
        line.strip().split("==", 1)
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.startswith(("tokenizers==", "jinja2=="))
    )
    for name in MODULES:
        try:
            module = import_module(name)
            if name in pins and module.__version__ != pins[name]:
                errors.append(
                    f"{name}: require {pins[name]}, found {module.__version__}"
                )
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return errors


def main():
    print(f"Checking host Python: {sys.executable}", flush=True)
    errors = check()
    if errors:
        print(
            "PRO6000 dependency preflight failed:\n- " + "\n- ".join(errors),
            file=sys.stderr,
        )
        print(
            "Activate the intended virtualenv, then run python -m pip install -r "
            "requirements.txt and python -m pip check. No GPU work was started.",
            file=sys.stderr,
        )
        return 1
    print(
        "Host imports and pinned tokenizer/template versions passed (no GPU validation)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
