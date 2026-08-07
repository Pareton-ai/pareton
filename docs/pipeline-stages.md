# Pipeline stages

The durable timeline of a submission lives in `submission_events` and is served
by `GET /v1/submissions/{patch_hash}`. The dashboard pipeline view renders these
states in order.

## Happy path

| Stage | State | Emitted when |
| --- | --- | --- |
| Submitted | `committed` | Watcher ingests the on-chain commitment |
| Picked up | `picked_up` | Worker claims the gates job |
| Fetched | `fetched` | Patch artifact pulled from object storage |
| Verified | `verified` | Integrity gate passed (hash, allowlist, hotkey bind) |
| Applied | `applied` | Patch applies onto the pinned baseline |
| Surface OK | `surface_ok` | Public surface compatibility holds |
| Building | `building` | Hermetic docker build started |
| Image pushed | `image_pushed` | Engine image pushed to GHCR (ref + digest) |
| Built | `built` | Build complete; digest-pinned ref recorded |
| Bench queued | `bench_queued` | Bench job row inserted |
| Correct | `correct` | Correctness gate passed vs baseline |
| Screened | `screened` | Performance screen cleared |
| Benched | `benched` | Full benchmark complete; verdict available |

## Terminal failure

`rejected` can follow any stage. The `detail` of the rejection event carries
`reason` plus a sanitized, size-capped `build_log_tail` for build failures.

## Build log drill-down

`GET /v1/submissions/{patch_hash}/build-log?tail=N` returns the last `N` lines
(default 200, max 2000) of the durable build log at
`$PARETON_BUILD_LOG_DIR/<submission_id>/build.log`, ANSI/control-stripped,
`text/plain`. 404 when the submission or its log does not exist (e.g. mock
builds, or pre-PAR-37 logs).

## Notes

- The event log is append-only; a retried job appends a fresh run of states.
- `bench_queued` only appears for campaigns with a bench spec.
- `image_pushed` only appears for real (non-mock, pushed) builds.
