# Pareton Roadmap & Agent Task Briefs

> Audience: AI coding agents (and future humans) working on this repo.
> Purpose: single source of truth for **what is done, what is not, and how to finish it**.
> Last updated: 2026-07-14. Update this doc whenever a workstream changes state.


## 1. Big picture

Pareton runs **inference-optimization campaigns** on Bittensor SN10. Miners submit git
patches against a pinned vLLM baseline; Pareton verifies, builds, benchmarks, and
(eventually) pays for real performance gains.

Full pipeline, end to end:

```
Stage 0  Campaign manifest → patch commitment → provenance gates → hermetic build
         [DONE — code complete, some ops remain]
Stage 1  Correctness gate (teacher-forced logprob comparison)      [NOT STARTED]
Stage 2  Perf screen (cheap throughput smoke test)                 [NOT STARTED]
Stage 3  SLA benchmark (workload trace replay, TTFT/ITL/p99)       [NOT STARTED]
Stage 4  Cross-environment validation (multi-GPU-SKU robustness)   [NOT STARTED]
Stage 5  Scoring → reward weights on chain                         [DESIGN ONLY]
```

Stages 1–3 are implemented as the **bench harness** (`bench/` package, workstream B).
Stage 4 reuses the same harness across GPU SKUs (workstream E). GPU machines are
rented on demand from **Targon / Lium / Shadeform** (workstream C).

Key repo docs:

- `docs/technical-decisions.md` — every decision made so far, with reasons. **Read first.**
- `docs/stage0-ops-checklist.md` — Stage 0 manual ops steps (partially done).
- `docs/pareton-bench-outsource-spec.md` — detailed bench spec (in Chinese; originally
  written for outsourcing, now the reference spec for workstream B. Schemas, thresholds,
  and method definitions in it are authoritative unless this doc says otherwise).


## 2. Current state — what is DONE

### Code (all tested, `pytest tests -q` green)

| Component | Path | Notes |
|---|---|---|
| Campaign domain + seed CLI | `campaign/` | Synthetic campaign, manifest hashing, Neon store |
| Chain commitment parse/watch | `chain/` | v1 patch payload, revealed-commitment polling |
| Provenance gates a–d | `gate/` | identity, integrity, base-apply, surface |
| Hermetic builder | `builder/` | patch → build in pinned base image → GHCR push (+ mock mode) |
| S3 presign + bounded fetch | `storage/s3.py` | AWS S3 `pareton-s3` (us-east-2), allowlisted URLs |
| Postgres schema + store | `db/schema.sql`, `campaign/store.py` | campaigns, submissions, events, jobs |
| Worker loop | `worker/main.py`, `worker/pipeline.py` | gates a–e fail-fast; `--scan-chain` polls SN10 |
| HTTP API | `api/server.py` | campaigns/submissions reads + presign endpoint |
| Miner CLI | `miner/commit_patch.py` | presign upload + `set_reveal_commitment` |
| Tests | `tests/` | 24 passing: surface adversarial diffs, commitment parse, e2e mock |

### Infra / ops

| Item | Status |
|---|---|
| GitHub repo `Pareton-ai/pareton` (private) | DONE — pushed |
| Neon Postgres (`PARETON_DATABASE_URL` in `.env`) | DONE — schema applied |
| AWS S3: bucket `pareton-s3` (us-east-2), IAM user `pareton-api`, public-read policy on `stage0/campaigns/*`, presign smoke test | DONE |
| vLLM baseline pin: v0.24.0 = `ee0da84ab9e04ac7610e28580af62c365e898389` | DONE (seed default) |
| GHCR namespace decided: `ghcr.io/pareton-ai/...` | DONE (config default) |

### State machine today (gates a–e)

`committed → fetched → verified → applied → surface_ok → built` (or `rejected`),
append-only in `submission_events`. See `gate/types.py`.


## 3. Conventions agents MUST follow

1. **Read `docs/technical-decisions.md` before changing architecture.** If you make a
   new decision, append it there with a date.
2. Env vars are all `PARETON_*`, defined in `config.py` with defaults; `.env` is
   gitignored and holds real secrets. Never commit secrets; never print them.
3. Top-level domain packages (`campaign/`, `gate/`, `bench/`, ...) — no nested
   `pareton/pareton/` style. New entrypoints are `python -m <package>`.
4. Audit trail is append-only `submission_events`; never mutate past events.
5. Everything content-addressed: images by digest, patches by sha256, traces by sha256.
6. Tests: unit tests must run without DB/GPU/network. DB-backed and GPU-backed tests
   get their own markers. Keep `pytest tests -q` green.
7. Legacy Cacheon code lives in `/Users/xavierlu/Desktop/cacheon` (read-only reference).
   Reuse patterns from it, but adapt into Pareton packages — do not import from it.


## 4. Workstreams (the TODOs)

Ordering / dependency graph:

```
WS-A (Stage 0 ops)  ──────────────┐
WS-B (bench harness, mock-first) ─┼─→ WS-D (pipeline integration) ─→ WS-E (cross-env) ─→ WS-F (scoring)
WS-C (GPU pod orchestration) ─────┘
WS-G (frontend API), WS-H (miner docs) — independent, any time
```

WS-B and WS-C can be built in parallel; WS-D needs both. WS-A is independent and
mostly human-ops (agent-assisted).


### WS-A — Finish Stage 0 ops  `[PARTIALLY DONE — mostly human + agent-assisted]`

**Goal:** go from "mock build on laptop" to "real pipeline on a VPS watching SN10".

**Context:** `docs/stage0-ops-checklist.md` has the full step-by-step with exact
commands. Remaining items:

1. ~~GHCR PAT with `write:packages`~~ **DONE 2026-07-14** — in `.env` as
   `PARETON_GHCR_TOKEN`, verified against ghcr.io.
2. Build + push the baseline **build** image (`images/baseline/Dockerfile`), capture
   its digest. Agent can do this on any amd64 Docker box once the PAT exists.
2b. Build + push the baseline **serving engine** image
   `ghcr.io/pareton-ai/pareton-engine:baseline` — vanilla vLLM at the pinned
   commit, serving-ready (`/v1/completions`). This is the "before" side of every
   bench comparison (WS-B) and the `baseline_engine_image_digest` the campaign
   manifest needs (WS-D item 4). Simplest path: run the hermetic builder with an
   empty patch, which also dogfoods the builder. Distinct from item 2 — the build
   base image is a compile environment, not a runnable engine.
3. Upload the workload trace fixture to S3 and re-seed the campaign with the real
   `base_image_digest` and an https trace URL (currently `file://...` placeholder).
4. Rent the CPU VPS, deploy API + worker (`--scan-chain`) with systemd, TLS via Caddy.
5. Create Bittensor coldkey/hotkey (human only — mnemonics must never touch an agent).
6. End-to-end test on finney: presign → upload → commit → watch → gates → build.

**Acceptance:** a real patch committed on SN10 ends `built` with an image on GHCR and
a full `submission_events` trail.


### WS-B — Bench harness (Stages 1–3)  `[IN PROGRESS — steps 1–2 done]`

**Goal:** a `bench/` package that answers: *does candidate engine image X beat baseline
image Y on workload trace T without changing model outputs?*

**Authoritative spec:** `docs/pareton-bench-outsource-spec.md` (Chinese). Implement
exactly the three modules, JSON contracts, and evidence-bundle layout defined there:

- **Module A — correctness gate:** greedy teacher-forced logprob comparison.
  Baseline generates greedy outputs; both engines score the identical forced sequence
  via OpenAI-compatible `/v1/completions` with echo+logprobs; compare per-position
  logprobs against thresholds (`mean_abs_logprob_diff`, `max_abs_logprob_diff`,
  `argmax_mismatch_rate`). Serial requests. Must include `baseline vs baseline`
  self-check mode.
- **Module B — perf screen:** closed-loop, fixed concurrency, small trace subset;
  candidate/baseline output-token throughput ratio ≥ threshold.
- **Module C — SLA benchmark:** open-loop trace replay by `arrival_offset_ms`,
  streaming; TTFT / ITL / e2e p50-p95-p99, tokens/s, goodput; 3 repetitions,
  median-aggregated; reproducibility bar: p99 TTFT relative range ≤ 10%.

**Placement decisions (differ from the outsource spec — these override it):**

- Lives in this repo as top-level `bench/` (not a separate repo).
- Entry point `python -m bench --request bench_request.json --output-dir out/`.
  The worker will invoke this **remotely on rented GPU pods** (WS-C/WS-D), which is
  why it keeps a process entrypoint rather than being import-only: the worker runs on
  a CPU VPS, the harness runs where the GPUs are. Report comes back as files.
- Shares `config.py` conventions but must be runnable standalone on a fresh GPU pod
  with just the repo + venv + Docker (no DB, no S3 creds, no chain).

**Engine-as-black-box rule:** talk to engines only via OpenAI-compatible HTTP.
Engine images are inputs (by digest). No hardcoded vLLM version or model name —
model comes from `bench_request.json` (`hf_repo` + pinned `hf_revision`, gated models
via `HF_TOKEN` env). Harness downloads weights, mounts read-only, runs engine
containers on an internal Docker network with no egress.

**Build order for agents (each step = one agent task, keep PRs reviewable):**

1. ~~`bench/` skeleton: request/report dataclasses + JSON schema validation, environment
   fingerprinting, output/evidence layout. Pure-Python unit tests.~~
   **DONE 2026-07-14** — `bench/{schemas,validate,env,output,main}.py`, fixtures under
   `fixtures/bench/`, stub CLI writes schema-valid report (`verdict=error` + `stub_note`).
2. ~~Mock engine: in-process OpenAI-compatible server with configurable logprobs and
   token latencies.~~ **DONE 2026-07-14** — `bench/mock_engine.py` + shape fixture
   `fixtures/bench/vllm_completion_response_shape.json` (re-validate vs real vLLM in B7).
   Tampered mode offsets logprobs for Module A adversarial tests.
3. Engine lifecycle manager: docker run/health-check/teardown, internal network,
   weights mount. Test against mock engine in a container (or subprocess).
4. Module A (correctness) end-to-end vs mock engine, incl. adversarial fixtures
   (tampered-logprob mock must fail, baseline-vs-baseline must pass).
5. HF weights manager: download, revision pin, sha256 manifest, cache dir.
6. Modules B and C vs mock engine (mock token latency makes TTFT/ITL testable
   deterministically). Independent recompute check: p99 from `requests.jsonl` must
   match report.
7. First real-GPU run: rent one pod (WS-C helper or manually), small model
   (e.g. Qwen2.5-7B-Instruct, 1 GPU), baseline-vs-baseline pass; calibrate default
   correctness thresholds from measured jitter; record them in
   `docs/technical-decisions.md`.

**Acceptance:** mock-engine e2e green in CI without GPU; on one rented GPU pod,
`mode=all` runs with a real small model and produces a schema-valid report + evidence
bundle; adversarial fixtures produce the right failures.


### WS-C — GPU pod orchestration  `[NOT STARTED — resurrect from Cacheon]`

**Goal:** programmatically rent/provision/destroy GPU machines on Targon, Lium, and
Shadeform, and run bench jobs on them over SSH.

**Resurrection source (read-only reference, adapt don't import):**

- `/Users/xavierlu/Desktop/cacheon/validator/providers/` — `targon_provider.py`,
  `lium_provider.py`, `shadeform_provider.py`, `static_ssh_provider.py`, `ssh_keys.py`.
  These already handle pod create/poll/ssh/destroy per provider.
- `/Users/xavierlu/Desktop/cacheon/scripts/gpu_setup/` — `create_targon_pod.py`,
  `create_lium_pod.py`, `shared.py`.

**Shape:** new `gpu/` package with a provider-agnostic interface, roughly:

```
provision(spec: PodSpec) -> Pod          # sku, gpu_count, region prefs, image
run(pod, argv, files_in, files_out)      # rsync/scp + ssh exec, stream logs
destroy(pod)                             # ALWAYS called, even on failure
```

plus a `static_ssh` provider (point at any existing box) for dev, mirroring the old
Cacheon pattern. Env vars: `PARETON_TARGON_API_KEY`, `PARETON_LIUM_API_KEY`,
`PARETON_SHADEFORM_API_KEY`, SSH key path.

**Key requirements:**

- Cost safety: hard TTL on every pod (auto-destroy after N hours no matter what),
  and a `gpu list`/`gpu reap` CLI to find and kill orphans. This is the most
  important requirement in the workstream — orphaned H200 pods burn real money.
- Pod bootstrap script: install Docker + nvidia toolkit if missing, clone repo at a
  pinned commit, venv, pull engine images. Idempotent.
- Unit tests mock the provider HTTP APIs; a `static_ssh` integration path allows
  testing against any box you already have.

**Acceptance:** one command rents a pod on at least one provider (whichever has
availability), runs `python -m bench` on it with the mock engine, copies the report
back, destroys the pod, and proves TTL reaping works.


### WS-D — Pipeline integration (worker runs Stages 1–3)  `[NOT STARTED — after B and C]`

**Goal:** the CPU-VPS worker, after `built`, automatically benchmarks each submission
on a rented GPU pod and records verdicts.

**Changes:**

1. **State machine** (`gate/types.py`): extend with
   `correct → screened → benched` (and reuse `rejected` with stage-specific reasons).
   Terminal happy state becomes `benched`.
2. **Schema** (`db/schema.sql` + migration): add `bench_reports` table —
   `submission_id`, `stage` (correctness|perf_screen|sla_bench), `verdict`,
   `report` JSONB, `evidence_s3_url`, `gpu_sku`, `created_at`. Evidence bundles
   upload to S3 under `stage0/evidence/<submission_id>/` (extend the IAM policy
   prefix if needed).
3. **Worker** (`worker/pipeline.py` + `worker/main.py`): after `built`, enqueue a
   bench job (extend `submission_jobs` with a `stage` column or add a
   `bench_jobs` table — pick one, document in decisions). A bench job:
   provision pod (WS-C) → ship `bench_request.json` built from the campaign manifest
   (trace URL/sha, SLA thresholds, model spec, baseline image from campaign,
   candidate image from submission) → run → collect report → persist → destroy pod.
4. **Campaign manifest**: it already carries `sla`, `workload_trace_*`, `gpu_skus`;
   add the missing piece — the **baseline engine image digest** (distinct from the
   build base image) and the model spec (`hf_repo`/`hf_revision`). Manifest is
   immutable, so this lands with the next seeded campaign; keep old campaigns valid.
5. **API**: expose bench verdicts/reports in `/v1/submissions/{patch_hash}` and
   `/v1/campaigns/{id}/submissions`.
6. **Mock mode all the way down:** `--mock-bench` flag that fakes the pod and runs
   bench with the mock engine locally, so the full state machine
   `committed → ... → benched` is testable in CI without GPUs. Extend
   `tests/test_e2e_mock.py` accordingly.

**Acceptance:** e2e mock test walks a submission to `benched` with a stored report;
on real infra, one submission goes chain-commit → build → GPU pod → SLA report,
fully unattended.


### WS-E — Stage 4: cross-environment validation  `[NOT STARTED — thin layer on D]`

**Goal:** a submission's gains must hold across GPU SKUs/configs, not just one
environment — this is the product's core anti-overfitting claim.

**Shape:** for campaigns with multiple `gpu_skus`, WS-D's bench job fans out to one
pod per SKU (possibly across different providers depending on availability). Add an
aggregation rule to the campaign manifest, e.g.
`cross_env: {"min_speedup_each": 1.0, "aggregate": "min"}` — a candidate's effective
speedup is the **minimum** across environments (conservative default; decide and
record). Store one `bench_reports` row per SKU plus an aggregated verdict event.

**Acceptance:** mock e2e with two fake SKUs produces per-SKU reports + aggregate;
design note added to `docs/technical-decisions.md`.


### WS-F — Stage 5: scoring & reward weights  `[DESIGN ONLY — do not build yet]`

**Goal (eventual):** turn bench results into miner payouts: score per submission →
normalize per campaign → `set_weights` on SN10.

Not implemented, deliberately. What exists to build on: append-only audit trail,
content-addressed reports, `commit_block` for first-seen tie-breaks, per-campaign
dedupe on `patch_hash`. Legacy scoring/weights patterns to reference (not copy
blindly): Cacheon `validator/scoring.py`, weight-setting in the old validator loop.

**Open design questions (answer before any code):**

- Winner-take-all per campaign vs. proportional-to-speedup with a floor?
- How to reward incremental improvements over a previous winner (patch stacking)?
- Anti-sybil: does first-seen-wins on `patch_hash` suffice when patches are public
  after reveal? (Likely needs commit-reveal timing analysis — noted in decisions doc.)
- Weight-setting cadence and the wallet/hotkey security model on the VPS.

**Trigger to start:** WS-D running unattended on a real campaign with ≥1 external miner.


### WS-G — Frontend read API alignment  `[SMALL — anytime]`

`pareton-frontend` (separate repo, Next.js) needs read-only visibility: campaign list
+ manifest detail, submission list with states, per-submission event timeline, and
(post WS-D) bench verdicts. The API in `api/server.py` already serves most of this;
remaining work is CORS origin pinning for the deployed frontend domain, pagination
on submission lists, and a `/v1/stats` summary endpoint (campaign counts, submission
counts by state). Keep all endpoints unauthenticated reads; anything mutating stays
out of the public API.


### WS-H — Miner-facing docs  `[SMALL — before opening a real campaign]`

A single `docs/miner-guide.md`: what a campaign is, how to read the manifest
(allowed/denied paths!), exact commit flow (`presign → PUT → commit_patch.py`),
what each gate checks and the exact rejection reasons, how to reproduce the
correctness gate locally against their own patch before committing (reuse the
bench `baseline vs baseline` mode), and where to see status (API/frontend).
Written last-mile once WS-B thresholds are calibrated.


## 5. Suggested execution order

| # | Task | Workstream | Needs GPU? | Needs human? |
|---|---|---|---|---|
| 1 | GHCR PAT + baseline image build/push | A | no | PAT creation |
| 2 | bench skeleton + mock engine + Module A | B1–B4 | no | no |
| 3 | GPU provider package + TTL reaper | C | cheap pod to verify | API keys |
| 4 | HF weights manager + Modules B/C | B5–B6 | no | no |
| 5 | First real-GPU bench run + threshold calibration | B7 | yes (1 small pod) | no |
| 6 | Trace to S3 + real campaign re-seed | A3 | no | no |
| 7 | Worker/schema/API integration + mock-bench e2e | D | no | no |
| 8 | VPS deploy + wallet + finney e2e | A4–A6 | no | wallet, VPS |
| 9 | Cross-env fan-out | E | yes (2 pods) | no |
| 10 | Scoring design doc → implementation | F | no | design sign-off |

Items 2–4 are safe to run as parallel agent tasks; they touch disjoint packages.


## 6. Verification checklist for every agent task

- [ ] `pytest tests -q` green locally (no DB/GPU/network needed for unit suite)
- [ ] New behavior has tests, including at least one adversarial/failure-path case
- [ ] No secrets in code, logs, or commits (`.env` only)
- [ ] New decisions appended to `docs/technical-decisions.md` with date
- [ ] This roadmap updated: workstream status + checkboxes
- [ ] Commit messages follow existing style (`feat:`, `docs:`, `ops:`, `fix:`)
