"""Print a patch's commitment hash locally, without wallet or network access."""

import argparse
import hashlib
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path, help="Exact diff bytes uploaded to Pareton")
    args = parser.parse_args(argv)
    digest = hashlib.sha256()
    try:
        with args.patch.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        parser.error(str(exc))
    print(f"sha256:{digest.hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
