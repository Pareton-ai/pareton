-- Pareton schema (Neon Postgres): Stage 0 + bench (WS-D/WS-E).
-- Source of truth for campaigns, submissions, provenance events, and bench jobs.
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
  workload_trace_sha256 TEXT NOT NULL,
  workload_trace_url TEXT NOT NULL,
  sla JSONB NOT NULL DEFAULT '{}'::jsonb,
  scoring_config_sha256 TEXT,
  scoring_config_url TEXT,
  allowed_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
  denied_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
  window_opens_at TIMESTAMPTZ NOT NULL,
  window_closes_at TIMESTAMPTZ NOT NULL,
  priority_metric TEXT NOT NULL
    CHECK (priority_metric IN ('throughput', 'gpu_hours', 'latency',
                               'utilization', 'cost_per_request')),
  success_threshold TEXT NOT NULL,
  manifest_hash TEXT NOT NULL,
  customer_signoff JSONB,
  status TEXT NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft', 'open', 'closed')),
  bench JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (manifest_hash)
);

CREATE INDEX IF NOT EXISTS campaigns_status_idx ON campaigns (status);
CREATE INDEX IF NOT EXISTS campaigns_profile_id_idx ON campaigns (profile_id);

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
  kind TEXT NOT NULL DEFAULT 'gates'
    CHECK (kind IN ('gates', 'bench')),
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'done', 'failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (submission_id, kind)
);

CREATE INDEX IF NOT EXISTS submission_jobs_status_idx ON submission_jobs (status, created_at);

CREATE TABLE IF NOT EXISTS bench_reports (
  id BIGSERIAL PRIMARY KEY,
  submission_id UUID NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
  task_id TEXT NOT NULL,
  stage TEXT NOT NULL
    CHECK (stage IN ('correctness', 'perf_screen', 'sla_bench')),
  verdict TEXT NOT NULL,
  report JSONB NOT NULL DEFAULT '{}'::jsonb,
  evidence_s3_url TEXT,
  gpu_sku TEXT NOT NULL DEFAULT 'unknown',
  mock BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (submission_id, stage, gpu_sku)
);

CREATE INDEX IF NOT EXISTS bench_reports_submission_id_idx
  ON bench_reports (submission_id);

CREATE TABLE IF NOT EXISTS watcher_meta (
  id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  last_scan_block INTEGER NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO watcher_meta (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
