-- Pareton Stage 0 schema (Neon Postgres).
-- Source of truth for campaigns, submissions, and the provenance gate audit log.

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
  manifest_hash TEXT NOT NULL,
  customer_signoff JSONB,
  status TEXT NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft', 'open', 'closed')),
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
  committed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  engine_image_ref TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (campaign_id, patch_hash)
);

CREATE INDEX IF NOT EXISTS submissions_campaign_id_idx ON submissions (campaign_id);
CREATE INDEX IF NOT EXISTS submissions_hotkey_idx ON submissions (hotkey);
CREATE INDEX IF NOT EXISTS submissions_patch_hash_idx ON submissions (patch_hash);

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
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (submission_id)
);

CREATE INDEX IF NOT EXISTS submission_jobs_status_idx ON submission_jobs (status, created_at);

CREATE TABLE IF NOT EXISTS watcher_meta (
  id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  last_scan_block INTEGER NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO watcher_meta (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
