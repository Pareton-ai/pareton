-- Pareton schema (Neon Postgres): Stage 0 + rounds.
-- Source of truth for campaigns, submissions, provenance events, gate jobs,
-- rounds, round entries, and leaders.
-- Single canonical schema file: no migration files while the project is pre-launch.
-- Apply wholesale to a fresh database: psql "$PARETON_DATABASE_URL" -f db/schema.sql
-- Schema changes pre-launch: edit this file and apply the delta by hand.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  data JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS campaigns (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  profile_id UUID REFERENCES profiles(id),
  baseline_repo TEXT NOT NULL,
  baseline_commit TEXT NOT NULL,
  base_image_digest TEXT NOT NULL,
  gpu_skus JSONB NOT NULL DEFAULT '[]'::jsonb,
  -- Unused when sampling_rule is set. Nullable so new seeds do not store a
  -- leftover file:// path. Live rows may still hold a value until an operator
  -- ALTER; do not remanifest a live campaign to clear them.
  workload_trace_sha256 TEXT,
  workload_trace_url TEXT,
  sla JSONB NOT NULL DEFAULT '{}'::jsonb,
  scoring_config_sha256 TEXT,
  scoring_config_url TEXT,
  allowed_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
  denied_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
  -- No submission window: campaigns accept patches until status flips to
  -- 'closed'. Dropped columns: window_opens_at, window_closes_at.
  priority_metric TEXT NOT NULL
    CHECK (priority_metric IN ('throughput', 'gpu_hours', 'latency',
                               'utilization', 'cost_per_request')),
  success_threshold TEXT NOT NULL,
  manifest_hash TEXT NOT NULL,
  customer_signoff JSONB,
  status TEXT NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft', 'open', 'closed')),
  bench JSONB,
  -- Build/launch recipe (campaign.engine):
  -- {name, install_cmd, entrypoint, cache_dir}.
  -- NULL means the vLLM default and is excluded from manifest_hash, so
  -- campaigns pinned before engine profiles existed keep their hash.
  engine JSONB,
  workload_pool JSONB,
  -- Sampler pin: {type: "hf_rows", dataset, revision, n_rows, n_prompts, ...}.
  -- Required for new campaigns. NULL is a leftover; the watcher will not
  -- create a round for that campaign.
  sampling_rule JSONB,
  -- Named ranking rule, pinned in manifest_hash: {name: "median_e2e_speedup"}.
  -- Fixed once the campaign leaves 'draft'.
  scoring_rule JSONB NOT NULL,
  -- Pay schedule, pinned in manifest_hash:
  -- {name: "linear_decay", start_weight, floor_weight, decay_blocks}.
  -- NULL means the campaign pays nothing and is left out of the weight
  -- vector, which keeps campaigns pinned before emission rules on their hash.
  emission_rule JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (manifest_hash)
);

CREATE INDEX IF NOT EXISTS campaigns_status_idx ON campaigns (status);
CREATE INDEX IF NOT EXISTS campaigns_profile_id_idx ON campaigns (profile_id);

-- The first trigger in this schema. The open campaigns must not promise more
-- emission than the subnet has: sum(start_weight) over status='open' <= 1.0.
-- A CHECK constraint sees only its own row, so it cannot express this, and an
-- application-level gate would be bypassed by the manual UPDATEs run against
-- campaigns. Checking start_weight rather than the decayed weight is
-- deliberate: decayed is always lower, so the start values bound the sum at
-- every block without the trigger needing to know the block.
-- A rule's start_weight, or an exception. Never NULL: `->>` on a missing key
-- yields SQL NULL, and NULL propagates through the comparison below as
-- unknown, so the guard would silently pass the row it exists to stop.
-- Treating a malformed rule as 0 was considered and rejected: an open campaign
-- whose rule cannot be read is broken, not free, and admitting it here just
-- moves the failure to the weight builder that reads the same field later.
CREATE OR REPLACE FUNCTION campaigns_emission_start_weight(rule JSONB)
RETURNS NUMERIC AS $$
DECLARE
  raw JSONB;
  weight NUMERIC;
BEGIN
  raw := rule -> 'start_weight';
  IF raw IS NULL OR jsonb_typeof(raw) <> 'number' THEN
    RAISE EXCEPTION
      'emission_rule.start_weight must be a number, got %',
      COALESCE(jsonb_typeof(raw), 'missing');
  END IF;
  weight := raw::TEXT::NUMERIC;
  IF weight < 0.0 OR weight > 1.0 THEN
    RAISE EXCEPTION
      'emission_rule.start_weight must be in [0, 1], got %', weight;
  END IF;
  RETURN weight;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION campaigns_emission_sum_guard() RETURNS trigger AS $$
DECLARE
  new_start NUMERIC;
  others NUMERIC;
BEGIN
  IF NEW.status <> 'open' OR NEW.emission_rule IS NULL THEN
    RETURN NEW;
  END IF;
  new_start := campaigns_emission_start_weight(NEW.emission_rule);
  -- SUM skips NULLs, so a malformed sibling row would under-count the budget.
  -- The helper raises instead, which can only fire on a row that predates this
  -- trigger or was written with it disabled: exactly when you want to know.
  SELECT COALESCE(SUM(campaigns_emission_start_weight(emission_rule)), 0)
    INTO others
    FROM campaigns
    WHERE status = 'open' AND emission_rule IS NOT NULL AND id <> NEW.id;
  IF new_start + others > 1.0 THEN
    RAISE EXCEPTION
      'emission_rule.start_weight % would take the open campaigns to %, over 1.0',
      new_start, new_start + others;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER campaigns_emission_sum_trg
  BEFORE INSERT OR UPDATE ON campaigns
  FOR EACH ROW EXECUTE FUNCTION campaigns_emission_sum_guard();

CREATE TABLE IF NOT EXISTS submissions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  campaign_id UUID NOT NULL REFERENCES campaigns(id),
  patch_hash TEXT NOT NULL,
  hotkey TEXT NOT NULL,
  baseline_commit TEXT NOT NULL,
  retrieval_url TEXT NOT NULL,
  commit_block INTEGER,
  payment_block INTEGER,
  payment_tx INTEGER,
  committed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  engine_image_ref TEXT,
  -- Workload sampling is per round now; see rounds.seed_block and friends.
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (campaign_id, patch_hash)
);

CREATE INDEX IF NOT EXISTS submissions_campaign_id_idx ON submissions (campaign_id);
CREATE INDEX IF NOT EXISTS submissions_hotkey_idx ON submissions (hotkey);
CREATE INDEX IF NOT EXISTS submissions_patch_hash_idx ON submissions (patch_hash);

-- Replay guard: one fee payment backs exactly one submission.
CREATE UNIQUE INDEX IF NOT EXISTS submissions_payment_ref_idx
  ON submissions (payment_block, payment_tx)
  WHERE payment_block IS NOT NULL;

CREATE TABLE IF NOT EXISTS submission_events (
  id BIGSERIAL PRIMARY KEY,
  submission_id UUID NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
  state TEXT NOT NULL,
  evidence_ref TEXT,
  detail JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS submission_events_submission_id_idx
  ON submission_events (submission_id, created_at);

CREATE TABLE IF NOT EXISTS submission_jobs (
  id BIGSERIAL PRIMARY KEY,
  submission_id UUID NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'done', 'failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  -- Live activity of the current attempt; cleared when the job ends.
  -- Keep the CHECK list in sync with bench/phases.py.
  phase TEXT
    CHECK (phase IS NULL OR phase IN ('provisioning', 'bootstrapping',
                                      'pulling_image', 'downloading_model',
                                      'starting_engine', 'correctness',
                                      'sla_bench', 'teardown')),
  phase_started_at TIMESTAMPTZ,
  heartbeat_at TIMESTAMPTZ,
  progress JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (submission_id)
);

CREATE INDEX IF NOT EXISTS submission_jobs_status_idx ON submission_jobs (status, created_at);

-- Round-based benchmarking. One round rents one pod, draws one prompt set, and
-- runs the baseline, the leader, and every challenger against that set.
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

-- One image run inside one round. score is NULL for a disqualified or
-- infra-failed entry; 0.0 is a real score and means baseline speed. The
-- baseline entry stores score 0.0 and anchors the chart zero line.
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

CREATE INDEX IF NOT EXISTS round_entries_submission_id_idx
  ON round_entries (submission_id);

-- One leader per campaign. A vacant crown has no row.
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

CREATE INDEX IF NOT EXISTS leader_history_campaign_id_idx
  ON leader_history (campaign_id, created_at);

-- Append-only: one row per weight compute cycle. Never updated after its
-- set_ok is written.
CREATE TABLE IF NOT EXISTS weight_sets (
  id BIGSERIAL PRIMARY KEY,
  computed_at_block INTEGER NOT NULL,
  version_key INTEGER NOT NULL,
  burn_uid INTEGER NOT NULL,
  -- The full dense vector, index 0..n-1, floats summing to 1.0.
  weights JSONB NOT NULL,
  -- Per-campaign inputs, so any row can be recomputed and checked:
  -- [{campaign_id, hotkey, uid, blocks_held, weight, note}, ...]
  -- `uid` is what the metagraph said at compute time and is NOT authoritative
  -- afterwards. `note` records why a share was withheld (vacant, dereg, closed).
  breakdown JSONB NOT NULL,
  -- Null until the chain call returns. Non-null with `set_ok=false` records a
  -- rejected attempt, so a run of failures is visible without reading logs.
  set_ok BOOLEAN,
  set_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS weight_sets_created_at_idx
  ON weight_sets (created_at DESC);
