# Back up and restore compiler caches

`python -m builder.cache` copies Pareton's commit-scoped `ccache` between a
BuildKit builder and a registry such as GHCR. Run it after a successful trusted
`--empty-patch` build to avoid repeating all compilation on a fresh VPS.
It works with both `--engine vllm` and `--engine sglang`.

This command transfers compiler cache entries. It does not transfer Docker layer
caches, Rust build directories, or inference-time caches. Pull a previously
published engine image if you only need to run that exact engine build.

## Back up a completed build

Use the same full commit SHA, base-image digest, Docker builder, and platform
as the build that populated the cache. The default platform is `linux/amd64`;
pass `--platform linux/arm64` for an ARM builder.

```bash
COMMIT=4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc
BASE=ghcr.io/pareton-ai/pareton-baseline@sha256:97e1f4e868fc988355f91bb20a6d6f3a9b90c3a901d030730a2646ecbdf00688

python -m builder.cache backup \
  --engine sglang \
  --baseline-commit "$COMMIT" \
  --base-image "$BASE" \
  --image-ref "ghcr.io/pareton-ai/pareton-ccache:sglang-${COMMIT}-$(date -u +%Y%m%dT%H%M%SZ)" \
  > ccache-ref.txt
```

Backup always pushes the snapshot. It prints the resulting digest-pinned image
reference to stdout; Docker progress goes to stderr. Keep `ccache-ref.txt` with
your build records. A cache with no compiler entries fails instead of publishing
an empty snapshot. If this happens, check that the trusted build used `ccache`
and that `PARETON_BUILDER_NAME` selects the builder that ran it.

The snapshot contains cache files plus labels identifying the engine, source
commit, source base image, platform, and snapshot format version. Machine-local
`ccache.conf` and temporary files are excluded. The labels record operator inputs;
they are not proof of how a cache was produced.

## Restore on a new VPS

Copy the saved reference to the VPS, then restore before running the usual
`python -m builder` command:

```bash
python -m builder.cache restore \
  --engine sglang \
  --baseline-commit "$COMMIT" \
  --base-image "$BASE" \
  --image-ref "$(cat ccache-ref.txt)"
```

Restore requires a digest-pinned reference and verifies the snapshot's format,
engine, and platform before copying. It merges files into the target commit's
cache, replacing entries with the same path and retaining other entries. Repeating
the command executes the copy again, including after local cache eviction.

Both commands use `PARETON_BUILDER_NAME`, `PARETON_BUILDER_LOCK_PATH`, and
`PARETON_BUILD_TIMEOUT_S`. Use the same values as the engine builder. They acquire
the existing storage lock and a locked BuildKit cache mount, so they wait for
active Pareton builds or cleanup. Registry operations use Docker's credentials,
including the existing `PARETON_GHCR_USERNAME` and `PARETON_GHCR_TOKEN` login when
configured. Loading `.env` follows the usual shell setup; this CLI does not load it.

Budget disk space for both the cache and its snapshot layers. Registry downloads
and local copies can still take time for a large cache.

## Seed a later commit

Use a previous snapshot as `--image-ref` and set `--baseline-commit` to the full
SHA of the new target commit. Set `--base-image` to the new build base. The engine
and CPU platform must still match; a different base image produces a notice.
Run the new trusted baseline build afterward to populate any missing entries,
then back up that commit under a new tag.

`ccache` checks its normal compiler, source, header, and option keys. Unchanged
compilation units may hit across commits; changed toolchains, CUDA targets, or
build flags can reduce reuse. Restoring a cache does not guarantee a speedup or
skip linking and packaging.

Only restore snapshots produced by a trusted baseline builder. A digest pins
content, not its trustworthiness. Miner builds keep their existing read-only
cache access, and neither worker configuration nor campaign manifests gain a
cache-import setting.

## Smoke test without an engine build

From the repository root, run:

```bash
python scripts/smoke_build_cache.py
```

This requires a local Docker Engine and Buildx, with access to pull Alpine,
BuildKit, and `registry:2` from Docker Hub. It needs no GPU or extra Python
packages. The test publishes only to a temporary registry bound to loopback.

The script compiles and runs a tiny C program, checks a cold compiler-cache miss,
and backs up snapshots under both engine labels. It then removes the source
builder and its cache volume. For each label, it restores into separate fresh
builders using the original and a different synthetic commit SHA, verifies direct
compiler-cache hits, and checks the executable's output. No vLLM/SGLang code is
built. It removes its builders and registry on exit and leaves the selected
Docker builder unchanged.

Verified on 2026-09-09 with Docker Desktop ARM64, Alpine 3.22, GCC 14.2.0, and
ccache 4.11.3: both backups succeeded, and all four restores produced direct
cache hits and correct executable output after the source builder was deleted.
All test containers, builders, and cache volumes were cleaned up. This validates
cache transfer and reuse with unchanged C source across synthetic commit IDs;
it does not measure CUDA build speedups or test Docker Hub/GHCR publication.
