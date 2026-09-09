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
