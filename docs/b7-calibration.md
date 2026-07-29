# B7 calibration runbook

Operator path after A2b (baseline engine image) exits 0. Prep code is offline;
only `python -m gpu bench` rents a GPU.

## Prerequisites

1. A2b RepoDigest for `ghcr.io/pareton-ai/pareton-engine:baseline`
   (capture from the build host / GHCR after push).
2. Lium credentials and `PARETON_GPU_*` env as for prior live GPU runs.
3. `PARETON_GHCR_TOKEN` / `PARETON_GHCR_USER` so the pod can pull the engine.
4. `HF_TOKEN` (or `PARETON_HF_TOKEN`) for Qwen2.5-7B-Instruct weights.
5. Budget: one pod for N harness repetitions (default 5); set TTL with margin.

## Steps

### 1. Capture digest

```bash
DIGEST="sha256:<A2B_REPODIGEST_HEX>"
# Engine image is CUDA arch 9.0 (Hopper only). Non-Hopper SKUs fail at engine start.
GPU_SKU="<Lium H100/H200 offer name, e.g. H200>"
RUN="out/b7/$(date -u +%Y%m%dT%H%M%SZ)"
```

### 2. Prepare request

```bash
python -m bench.calibrate prepare \
  --engine-digest "$DIGEST" \
  --gpu-sku "$GPU_SKU" \
  --output-dir "$RUN"
```

Expect:

- `$RUN/bench_request.json` with `mode=all`
- baseline image equals candidate image (pullable `@sha256:…` ref)
- model `Qwen/Qwen2.5-7B-Instruct` at the pinned revision
- A3a workload trace materialized and hash-checked
- permissive measurement thresholds (do not treat these as production gates)

Optional: `--trace-url` / `--trace-sha256` if not using the default A3a S3 object.

### 3. Run five self-checks on one pod

```bash
python -m gpu bench \
  --provider lium \
  --gpu-type "$GPU_SKU" \
  --gpu-count 1 \
  --request "$RUN/bench_request.json" \
  --output-dir "$RUN/runs" \
  --repetitions 5 \
  --ttl-hours 8
```

Default pod TTL is 2h; five `mode=all` reps need more. Pass `--ttl-hours 8` so
reap cannot destroy the pod mid-run.

Expect `$RUN/runs/run-001` … `run-005` each with `bench_report.json` and evidence.
Exit `75` means destroy failed: destroy the pod manually on the provider dashboard,
then `python -m gpu reap` if needed.

### 4. Analyze

```bash
python -m bench.calibrate analyze \
  --runs-dir "$RUN/runs" \
  --safety-factor 2.0 \
  --output "$RUN/calibration_summary.json"
```

Review:

- suggested correctness thresholds (`max observed × factor`, floored by current
  `config.py` placeholders when all observations are zero — see `all_observed_zero`)
- SLA repro suggestion (informational; does not edit `REPRO_BAR_MAX_REL_RANGE`)
- response-shape match vs `fixtures/bench/vllm_completion_response_shape.json`

### 5. Record decision (local only)

1. Manually accept or adjust suggested thresholds.
2. Apply accepted values to `config.py` in a follow-up change (not automatic).
3. Append hardware, digest, observations, margin, and final numbers to local
   `docs/technical-decisions.md` (gitignored).
4. Mark B7 complete in local `docs/roadmap.md` only after the live run passes
   acceptance (schema-valid reports + adversarial fixtures behave correctly).

## Emergency cleanup

```bash
python -m gpu list
python -m gpu destroy <pod_name>
python -m gpu reap
```

## Non-goals of the prep PR

No CI GPU rental, no A2b VPS changes, no automatic `config.py` writes, no fixture
auto-rewrite.
