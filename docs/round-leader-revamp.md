# Round and leader revamp

Status: approved design, not yet implemented.
Date: 2026-08-19.
Linear: see [Tickets](#tickets).

## Purpose

Replace per-submission benchmarking and z-score promotion with round-based
benchmarking against a leader.

Today each submission is benched alone on its own GPU pod, against its own
sampled prompts, and is promoted if its z-scores beat a calibrated
distribution. Scores from two submissions are not comparable, and the pipeline
rents one pod per submission.

The new design batches submissions into rounds. One round rents one pod, draws
one prompt set, and runs every image in the round against that prompt set. The
best image becomes the leader. The leader runs again in every later round.

## What changes

| Area | Today | After |
| --- | --- | --- |
| Bench unit | one submission | one round of 5 challengers plus the leader |
| GPU pods | one per submission | one per round |
| Prompts | per submission, seeded by commit block | per round, seeded by round block |
| Baseline | started per submission, per module | started once per round |
| Correctness | candidate engine scores a forced sequence | one shared scorer grades captured output |
| Ranking | z-score vs calibrated distribution | median e2e speedup vs baseline, in-round |
| Outcome | promoted or not | leader seated, held, or vacated |
| Modules | correctness, perf_screen, sla_bench | correctness, sla_bench |

## Design decisions

Each decision below is settled. Do not reopen without a new design session.

### Scope and triggering

1. Rounds and leaders are per campaign. A campaign pins one engine and one
   baseline, so scores are only meaningful inside one campaign.
2. A round starts when 5 submissions reach `bench_queued`, or when the oldest
   `bench_queued` submission has waited longer than `PARETON_ROUND_MAX_WAIT_S`.
   A round needs at least 1 challenger.
3. `bench_queued` means the submission passed all gates and its image built and
   pushed. Do not bench an image that does not build.
4. The cohort is the oldest `bench_queued` submissions by `commit_block`.
   Submissions that do not fit stay queued for the next round.
5. Inside a cohort, submissions with the same `engine_image_ref` digest collapse
   to the earliest `commit_block`. The rest are rejected as `duplicate_image`.

### Prompts

6. Seed is `sha256(block_hash(B) + campaign_id)`. `B` is the chain head when the
   round is created. Wait for `B` to finalize before dispatch.
7. The sampling rule stays on `campaigns.sampling_rule`. The realized trace is
   snapshotted onto the round.
8. Every round draws a different prompt set. Scores are comparable inside one
   round only. State this in the public docs.

### Scoring

9. Rank on median per-prompt end-to-end speedup against the baseline:
   `(baseline_e2e - candidate_e2e) / baseline_e2e`. A score of 0.35 means 35
   percent faster.
10. The formula is a named rule in `campaigns.scoring_rule`. Ship one
    implementation, `median_e2e_speedup`. Dispatch by name in `bench/score.py`.
11. `scoring_rule` is fixed once a campaign leaves `draft`. Enforce this in
    `campaign/store.py`. `manifest_hash` covers `scoring_rule`.
12. Store absolute metrics in `round_entries.report`, not only the ratio. This
    keeps cross-campaign questions answerable later.

### Correctness

13. Correctness does not start its own container. Each candidate runs once, in
    production configuration. The SLA run captures its output text.
14. After all candidates finish, start one scorer and teacher-force every
    captured output through it. Stop the scorer.
15. The scorer is the campaign's own pinned baseline engine image, started with
    the correctness serve args. The scorer is per campaign.
16. Correctness is a hard gate. A failed image is `disqualified`, records a
    reason, gets no score, and cannot become leader.
17. Trade-off accepted: this drops strict per-position logprob equivalence
    against the baseline engine. An output that differs from the baseline but
    stays plausible now passes.

### Leader

18. One leader per campaign. No runner-up. A vacant leader has no row.
19. To take the crown, a challenger must beat `leader_score * (1 + epsilon)`.
    Default epsilon is 0.01.
20. To hold the crown at all, an image must pass correctness and score strictly
    above 0. If no image clears this, the crown stays vacant.
21. `campaigns.success_threshold` is a display flag on the round. It is not the
    crown bar.
22. The leader runs in every round as a normal entry.
23. If the leader loses on score, the challenger takes the crown.
24. If the leader fails correctness, the crown goes to the best passing
    challenger, or is vacated.
25. If the leader fails on infrastructure, void the round. No leader comparison
    is possible.

### Failure handling

26. Void the round on: pod provision failure, pod death, baseline failure,
    leader infrastructure failure, zero surviving challengers, drift over the
    ceiling, round duration over `PARETON_ROUND_MAX_DURATION_S`.
27. A void round changes no submission state and no leader state. It records
    `void_reason`, requeues its entries, and consumes its ordinal. There is no
    attempt counter and no retry cap. An operator watches each round.
28. A single challenger that fails to start is `infra_failed`. That entry gets
    one requeue. The round continues with the rest.
29. The watcher voids any `running` round whose `heartbeat_at` is older than
    `PARETON_ROUND_STALE_S`.

### Ordering and bias

30. Run the baseline once at the start. Its per-prompt timings are the fixed
    reference for every candidate in the round.
31. Mount the engine compile cache read-write for the baseline only. Never
    mount it for a candidate. Each candidate container starts in the same cache
    state and its cache dies with the container.
32. Discard warmup requests in every container. Keep
    `SlaBenchConfig.warmup_requests` above 0.
33. Run the baseline a second time at the end, SLA only. Record
    `drift = last_baseline_score - first_baseline_score`. Void the round if
    `abs(drift)` is over the ceiling.
34. Do not randomize entry order. Cache isolation removes the bias. Randomizing
    would only hide it and would make rounds harder to reproduce.

### Hardware

35. One GPU SKU per round, recorded on the round. Default to `gpu_skus[0]`.
36. Round starts: 1 baseline, 6 candidates, 1 scorer, 1 drift baseline. Total 9.
    Budget 6 hours per round.

## Schema

Apply as one edit to `db/schema.sql`. Write no migration files. The project is
pre-launch and `db/schema.sql` is the single source of truth.

### New tables

```sql
CREATE TABLE IF NOT EXISTS rounds (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  campaign_id UUID NOT NULL REFERENCES campaigns(id),
  ordinal INTEGER NOT NULL,
  gpu_sku TEXT NOT NULL,
  seed_block INTEGER NOT NULL,
  seed_block_hash TEXT NOT NULL,
  seed_hex TEXT NOT NULL,
  sampled_trace_sha256 TEXT NOT NULL,
  sampling_receipt JSONB NOT NULL DEFAULT '{}'::jsonb,
  scoring_rule JSONB NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'complete', 'void')),
  void_reason TEXT,
  incumbent_submission_id UUID REFERENCES submissions(id),
  winner_submission_id UUID REFERENCES submissions(id),
  leader_changed BOOLEAN,
  baseline_drift NUMERIC,
  phase TEXT CHECK (phase IS NULL OR phase IN (
    'provisioning', 'bootstrapping', 'pulling_image', 'downloading_model',
    'starting_engine', 'sla_bench', 'correctness', 'teardown')),
  phase_started_at TIMESTAMPTZ,
  heartbeat_at TIMESTAMPTZ,
  progress JSONB,
  current_entry_id BIGINT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  UNIQUE (campaign_id, ordinal)
);

-- At most one live round per campaign.
CREATE UNIQUE INDEX IF NOT EXISTS rounds_one_live_per_campaign_idx
  ON rounds (campaign_id) WHERE status IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS round_entries (
  id BIGSERIAL PRIMARY KEY,
  round_id UUID NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
  submission_id UUID REFERENCES submissions(id),
  role TEXT NOT NULL CHECK (role IN ('baseline', 'leader', 'challenger')),
  engine_image_ref TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'scored', 'disqualified',
                      'infra_failed')),
  score NUMERIC,
  disqualify_reason TEXT,
  report JSONB NOT NULL DEFAULT '{}'::jsonb,
  evidence_s3_url TEXT,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- The baseline row carries submission_id IS NULL, and a default UNIQUE
  -- treats every NULL as distinct, so without NULLS NOT DISTINCT one round
  -- could hold many NULL-id rows. Needs Postgres 15+; Neon runs 17.
  UNIQUE NULLS NOT DISTINCT (round_id, submission_id),
  -- The baseline is the campaign's pinned engine, not a submission. Every
  -- other role is a submission. Both directions are held here so a writer
  -- cannot insert a NULL-id challenger or a submission-backed baseline.
  CHECK (
    (role = 'baseline' AND submission_id IS NULL)
    OR (role <> 'baseline' AND submission_id IS NOT NULL)
  )
);

CREATE UNIQUE INDEX IF NOT EXISTS round_entries_one_baseline_idx
  ON round_entries (round_id) WHERE role = 'baseline';

CREATE TABLE IF NOT EXISTS leaders (
  campaign_id UUID PRIMARY KEY REFERENCES campaigns(id),
  submission_id UUID NOT NULL REFERENCES submissions(id),
  engine_image_ref TEXT NOT NULL,
  hotkey TEXT NOT NULL,
  won_at_round_id UUID NOT NULL REFERENCES rounds(id),
  won_at_ordinal INTEGER NOT NULL,
  last_score NUMERIC NOT NULL,
  last_scored_round_id UUID REFERENCES rounds(id),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS leader_history (
  id BIGSERIAL PRIMARY KEY,
  campaign_id UUID NOT NULL REFERENCES campaigns(id),
  round_id UUID NOT NULL REFERENCES rounds(id),
  ordinal INTEGER NOT NULL,
  event TEXT NOT NULL CHECK (event IN ('seated', 'overtaken', 'vacated')),
  new_submission_id UUID REFERENCES submissions(id),
  new_hotkey TEXT,
  new_score NUMERIC,
  prev_submission_id UUID REFERENCES submissions(id),
  prev_hotkey TEXT,
  prev_score NUMERIC,
  overtake_threshold NUMERIC,
  epsilon NUMERIC NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Notes on the shape:

- `round_entries.score` is NULL for a disqualified or infra-failed entry.
  0.0 is a real score and means the image matched baseline speed.
- The baseline entry stores `score = 0.0`. It anchors the chart zero line.
- A void round keeps its ordinal. The chart shows honest gaps.
- `role` and `submission_id` are held in lockstep by a CHECK: the baseline is
  the campaign's pinned engine and has no submission, every other role has one.
  With that CHECK in place, `round_entries_one_baseline_idx` is logically
  redundant. It is kept because it costs nothing and it survives an edit to the
  CHECK.

### Changes to existing tables

Drop `bench_reports` entirely. `round_entries` replaces it.

`campaigns`:

- Drop `calibration`.
- Drop `z_threshold`.
- Drop `bench.cross_env`.
- Add `scoring_rule JSONB NOT NULL`.

`submissions`:

- Drop `sample_seed_block`.
- Drop `sample_seed_block_hash`.
- Drop `sampled_trace_sha256`.
- Drop `sampling_receipt`.

`submission_jobs`:

- Drop `kind` and its CHECK. Only gates jobs remain.
- Change `UNIQUE (submission_id, kind)` to `UNIQUE (submission_id)`.
- Drop `perf_screen` from the `phase` CHECK.

### Submission states

`gate/types.py` `SubmissionState` changes:

- Drop `sampled`, `correct`, `screened`. Sampling is per round. The two
  surviving modules are not milestones.
- Rename `benched` to `scored`. Terminal.
- Add `round_assigned`. Not terminal.
- Add `disqualified`. Terminal. The image ran and was wrong.
- Add `infra_failed`. Not terminal. One requeue.
- Keep `rejected`. Terminal. The submission never ran.

## Round execution

The watcher creates rounds. The worker runs them.

### Creation, by the watcher

1. Read the head block `B` and its hash. Wait for finality.
2. In one transaction, select the oldest `bench_queued` submissions for one
   campaign, `ORDER BY commit_block`, `LIMIT PARETON_ROUND_SIZE`,
   `FOR UPDATE SKIP LOCKED`.
3. Collapse duplicate image digests.
4. Sample the trace from `campaigns.sampling_rule` with
   `seed = sha256(block_hash(B) + campaign_id)`.
5. Insert `rounds` with `status='pending'`. Insert `round_entries` for the
   baseline, the leader if one exists, and each challenger.
6. Append a `round_assigned` event to each challenger submission.
7. Commit. The partial unique index rejects a second live round.

### Execution, by the worker

1. Claim a `pending` round with `FOR UPDATE SKIP LOCKED`. Set `running`.
2. Assert the leader image digest still resolves in the registry. Void if not.
3. Rent one pod for `rounds.gpu_sku`. Stage model weights.
4. Start the baseline with the engine compile cache mounted read-write.
   Run SLA. Record per-prompt timings and output text. Stop it.
5. For each candidate in cohort order, start it with no cache mount, in
   production configuration. Run SLA. Capture timings and output text. Stop it.
6. Start the scorer once. Teacher-force every captured output. Stop it.
7. Start the baseline again. Run SLA only. Compute `baseline_drift`.
8. Score each entry with the round's `scoring_rule`.
9. Decide the leader. Write `leaders` and `leader_history` in the same
   transaction as `running -> complete`.
10. Tear down the pod.

Write `heartbeat_at`, `phase`, and `current_entry_id` throughout.

### Status transitions

| Transition | Writer | Trigger |
| --- | --- | --- |
| none to `pending` | watcher | cohort full, or oldest wait over limit |
| `pending` to `running` | worker | claim |
| `running` to `complete` | worker | all entries terminal, leader decided |
| `running` to `void` | worker | see decision 26 |
| `running` to `void` | watcher | heartbeat stale |

`leaders` and `leader_history` are written only by the worker, only on
`running -> complete`. A void round never touches leader state.

## Ranking

`round/rank.py` holds one pure function. It takes a list of
`(role, submission_id, score, status)` and the current leader score. It returns
the new leader, the event type, and the void decision. It touches no database,
no Docker, and no pod.

Order of checks:

1. If the baseline entry failed, void.
2. If the leader entry is `infra_failed`, void.
3. If no challenger survives, void.
4. If `abs(drift)` is over the ceiling, void.
5. Drop every entry that is not `scored`.
6. If the leader is present and scored, the crown needs
   `challenger_score > leader_score * (1 + epsilon)`.
7. If the leader is absent or disqualified, the crown needs
   `score > 0`.
8. The best qualifying entry takes the crown. If none qualifies and the leader
   was disqualified, vacate.

## Deletions

Hard delete. Write no compatibility shims.

Code:

- `bench/calibrate.py`
- `bench/promote.py`
- `bench/perf_screen.py`
- `campaign/cross_env.py`
- `worker/bench_job.py`
- `bench/sampler.py`: `calib_seed`, and `patch_hash` in the seed
- `out/calib/`

Tests:

- `tests/test_calibrate.py`
- `tests/test_z_calibration.py`
- `tests/test_promote.py`
- `tests/test_apply_calibration.py`

Docs:

- `content/docs/platform/calibration.mdx` in pareton-frontend

Config: every `PARETON_*` calibration variable in `config.py`.

## New code layout

| Path | Role |
| --- | --- |
| `round/create.py` | cohort selection, seeding, round and entry insert |
| `round/rank.py` | pure ranking and leader decision |
| `round/store.py` | round, entry, and leader persistence |
| `worker/round_job.py` | runs one claimed round |
| `bench/score.py` | scoring rule dispatch |

`bench/` keeps its harness role. `EnginesSpec` becomes a baseline plus a list of
candidates. `InputsFingerprint.candidate_image_digest` becomes a list.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARETON_ROUND_SIZE` | 5 | challengers per round |
| `PARETON_ROUND_MAX_WAIT_S` | tbd | starvation timeout |
| `PARETON_ROUND_STALE_S` | 1800 | heartbeat staleness before void |
| `PARETON_ROUND_MAX_DURATION_S` | 21600 | hard cap, 6 hours |
| `PARETON_OVERTAKE_EPSILON` | 0.01 | leader moat |
| `PARETON_BASELINE_DRIFT_CEILING` | tbd | void threshold |

These are operator knobs. They are not per campaign and they are not in
`manifest_hash`. Cohort size does not change any miner's odds, because
overtaking is a pairwise comparison against the leader.

## API

All endpoints are public and read only. `POST /v1/uploads/patch` stays the only
write endpoint.

| Endpoint | Returns |
| --- | --- |
| `GET /v1/campaigns/{id}/leader` | current leader, or 404 when vacant |
| `GET /v1/campaigns/{id}/rounds` | round list |
| `GET /v1/rounds/{round_id}` | round detail with entries |
| `GET /v1/campaigns/{id}/score-progress` | chart series, ready to plot |

Identity is the hotkey. Truncate it in tables. Show it in full on detail pages.
Build logs and evidence URLs stay behind their current gates.

## Frontend

Keep the RSC plus `router.refresh()` pattern. Extend the live poll host to the
campaign page and the round detail page while a round is running.

New:

- Leader panel on the campaign page.
- Score progress chart. X axis is round ordinal. The green line is the leader
  score per round. Grey dots are every other entry in that round. Draw it as
  inline SVG in a server component. Do not add a chart library.
- Rounds table on the campaign page.
- Round detail page at `/dashboard/campaigns/[id]/rounds/[n]`.

Changed:

- Submission detail timeline becomes queued, assigned to round N, then scored
  or disqualified.
- `bench-reports.tsx` becomes a round entry view.

Removed: all z-score, calibration, and promotion UI.

## Cutover

Nothing is live. Wipe and reseed.

1. `systemctl stop pareton-worker`
2. `systemctl stop pareton-watcher`
3. Apply the new `db/schema.sql` to a fresh database.
4. Reseed campaigns with the same model, the same hardware, and the same engine
   profile. Keep the engine choice, vLLM or SGLang, unchanged.
5. Verify each new `manifest_hash` by reading it back over psycopg2. Do not
   trust a hash read through an MCP tool, because timestamps truncate.
6. `systemctl start pareton-watcher`
7. `systemctl start pareton-worker`

Recreate the Neon `test` branch from `main` and truncate domain tables before
running the DB e2e suite.

## Tests

- `round/rank.py` is unit tested with no engines, no pod, and no database. This
  is the correctness core.
- `bench/mock_engine.py` gains a per-image speed factor. A mock round scripts
  the whole matrix: baseline 1.0, two close challengers, a leader, one
  correctness failure, one infra failure.
- One `--mock-bench` end-to-end round lives in `tests/test_e2e_mock.py`.
- One unit test asserts the start count from the engine profile.
- A real SGLang round is a manual smoke test. It is not CI.

## Open items owned elsewhere

1. On-chain weights. Pareton sets no weights today. The round design records
   enough for a weight vector to be derived. Wiring `set_weights` needs its own
   design session: runner-up policy, burn UID, emission ramp, and version key.
2. SGLang teacher-forcing. The scorer path needs `echo` and `prompt_logprobs`.
   Verify SGLang supports this before an SGLang campaign runs. If it does not,
   the fallback is a pinned vLLM scorer, which works because it only sees text.
3. Read-only shared compile cache for candidates. This would give uniform warm
   starts without cross-contamination. Verify that vLLM torch.compile tolerates
   a read-only cache directory before relying on it.
4. Round wall time. 9 starts at 6 hours is one round per shift. The lever is
   `PARETON_ROUND_SIZE`, not the drift run.

## Tickets

Filed 2026-08-19 across three Linear projects. Every ticket links back to this
file.

### Mainnet MVP

| Ticket | Slice | Blocked by |
| --- | --- | --- |
| PAR-75 | Schema, calibration deletion, manifest change, reseed | none |
| PAR-76 | Pure ranking function and scoring rule seam | none |
| PAR-77 | Harness round mode, drop module B, batched scorer | none |
| PAR-79 | Watcher creates rounds and reaps stale ones | PAR-75 |
| PAR-80 | Public read API | PAR-75 |
| PAR-83 | Worker round runner, delete bench_job | PAR-75, PAR-76, PAR-77, PAR-79 |
| PAR-85 | Ops cutover, wipe and reseed | PAR-75, PAR-79, PAR-83 |

### Website

| Ticket | Slice | Blocked by |
| --- | --- | --- |
| PAR-84 | API client, types, and endpoints | PAR-80 |
| PAR-86 | Leader panel and score progress chart | PAR-84 |
| PAR-87 | Rounds table on the campaign page | PAR-84 |
| PAR-88 | Round detail page | PAR-84 |
| PAR-89 | Submission detail rework, remove promotion UI | PAR-84 |
| PAR-78 | Docs rewrite for rounds and leaders | none |

### Multi-engine: SGLang + B300

| Ticket | Slice | Blocked by |
| --- | --- | --- |
| PAR-81 | Add `cache_dir` to the engine profile | PAR-75 |
| PAR-82 | Verify SGLang can serve as the scorer | PAR-77 |

### Start here

PAR-75, PAR-76, and PAR-77 have no blockers and can run in parallel. PAR-81
must land with PAR-77, or the cache mount makes the ordering bias worse than it
is today.

### Decisions left open inside tickets

| Setting | Owner |
| --- | --- |
| `PARETON_ROUND_MAX_WAIT_S` default | PAR-79 |
| `PARETON_BASELINE_DRIFT_CEILING` default | PAR-83 |
| Whether the campaign page keeps a flat submissions table | PAR-87 |
