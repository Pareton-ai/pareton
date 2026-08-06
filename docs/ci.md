# Continuous integration

## Offline tests

The `Tests` workflow runs on pull requests and pushes to `main`. It runs:

```bash
pytest -q -m "not docker"
```

These tests do not require a database, chain, GPU, or Docker daemon. Tests marked
`e2e` skip when `PARETON_TEST_DATABASE_URL` is not set.

## Testnet smoke

The `Testnet Smoke` workflow checks the live commitment and watcher path. It
runs every night at 03:17 UTC. An operator can also start it with
`workflow_dispatch`.

The job uses the protected GitHub Environment `pareton-test`. That environment
must contain:

- `PARETON_TEST_DATABASE_URL`: connection URL for the Neon `test` branch.
- `CI_TESTNET_WALLET_SEED`: seed or mnemonic for the CI-only hotkey registered
  on Bittensor testnet netuid 543. Do not store a coldkey.

The helper also checks a pinned SHA-256 hash of the Neon test-branch hostname.
It refuses a database URL for any other endpoint, including the main branch.
The hash does not expose the database hostname or credentials.

The workflow performs these steps:

1. Check the test database schema.
2. Restore the temporary CI hotkey on the GitHub runner.
3. Reuse or create an open campaign with no bench configuration.
4. Create a run-specific patch under `vllm/**`.
5. Serve the patch from a local HTTP server.
6. Start the API and worker against the test database.
7. Submit the commitment on Bittensor testnet netuid 543.
8. Poll the test database for:

   ```text
   committed -> fetched -> verified -> applied -> surface_ok -> built
   ```

9. Upload API and worker logs as a workflow artifact.

The timeout is 25 minutes. The database poll waits up to 15 minutes. A scheduled
run reports a skip when no revealed submission appears before the timeout. A
manual run fails in the same case.

The job uses `--mock-build`. It does not write to production Postgres, S3, GHCR,
or a GPU provider. It does not test the presigned S3 upload path or the real
hermetic vLLM build.

### Run manually

Open **Actions**, select **Testnet Smoke**, and select **Run workflow**. The
protected environment supplies both secrets. Do not paste wallet material into
workflow inputs or logs.

### Failure diagnostics

A red run prints the ordered submission events and the last 200 API, worker, and
patch-server log lines. Download the `testnet-smoke-logs-*` artifact for the
complete process logs.
