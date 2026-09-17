"""Stage the pinned AOT FetchContent archives in the trusted image build."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

SOURCE = Path("/src/python/sglang/kernels/aot")
DESTINATION = Path("/opt/sglang-deps")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    declarations = []
    for path in [
        SOURCE / "CMakeLists.txt",
        *sorted((SOURCE / "cmake").glob("*.cmake")),
    ]:
        declarations.extend(
            re.findall(
                r"FetchContent_Declare\(\s*(repo-[\w-]+)\s+(.*?)\n\)",
                path.read_text(),
                re.DOTALL,
            )
        )
    sources: dict[str, Path] = {}
    receipts = []
    for name, declaration in declarations:
        url = re.search(r"\bURL\s+(\S+)", declaration)
        sha = re.search(r"\bURL_HASH\s+SHA256=([a-f0-9]{64})", declaration)
        if url is None or sha is None:
            raise RuntimeError(f"AOT dependency {name} lacks a URL/SHA256 pin")
        address = url[1].replace("${GITHUB_ARTIFACTORY}", "github.com")
        destination = DESTINATION / name
        source_dir = re.search(r"\bSOURCE_DIR\s+(\S+)", declaration)
        if source_dir:
            expanded = source_dir[1]
            for other_name, other_path in sources.items():
                expanded = expanded.replace(
                    "${" + other_name + "_SOURCE_DIR}", str(other_path)
                )
            destination = Path(expanded)
            if not destination.is_relative_to(DESTINATION) or "$" in expanded:
                raise RuntimeError(
                    f"Unsupported dependency source directory: {expanded}"
                )
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "source.tar.gz"
            digest = hashlib.sha256()
            with (
                urlopen(address, timeout=300) as response,
                archive.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != sha[1]:
                raise RuntimeError(f"SHA256 mismatch for {name}")
            extracted = Path(tmp) / "source"
            extracted.mkdir()
            with tarfile.open(archive) as source:
                source.extractall(extracted, filter="data")
            roots = list(extracted.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError(f"Unexpected archive layout for {name}")
            shutil.copytree(roots[0], destination, dirs_exist_ok=True)
        sources[name] = destination
        receipts.append(
            {"name": name, "url": address, "sha256": sha[1], "path": str(destination)}
        )
        print(f"Staged {name} at {sha[1]}", flush=True)
    if len(sources) != 7:
        raise RuntimeError(
            f"Expected seven dependencies at this source pin, found {len(sources)}"
        )
    cache = [
        'set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "" FORCE)',
        'set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)',
    ]
    cache.extend(
        f'set(FETCHCONTENT_SOURCE_DIR_{name.upper()} "{path}" CACHE PATH "" FORCE)'
        for name, path in sources.items()
    )
    (DESTINATION / "offline.cmake").write_text("\n".join(cache) + "\n")
    (DESTINATION / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
