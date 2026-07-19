-- WS-D: bench job kind, bench_reports, campaigns.bench
-- Apply manually: psql "$PARETON_DATABASE_URL" -f db/migrations/20260718_bench.sql

ALTER TABLE submission_jobs
  ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'gates';

ALTER TABLE submission_jobs
  DROP CONSTRAINT IF EXISTS submission_jobs_kind_check;

ALTER TABLE submission_jobs
  ADD CONSTRAINT submission_jobs_kind_check
  CHECK (kind IN ('gates', 'bench'));

ALTER TABLE submission_jobs
  DROP CONSTRAINT IF EXISTS submission_jobs_submission_id_key;

ALTER TABLE submission_jobs
  DROP CONSTRAINT IF EXISTS submission_jobs_submission_id_kind_key;

ALTER TABLE submission_jobs
  ADD CONSTRAINT submission_jobs_submission_id_kind_key
  UNIQUE (submission_id, kind);

CREATE TABLE IF NOT EXISTS bench_reports (
  id BIGSERIAL PRIMARY KEY,
  submission_id UUID NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
  task_id TEXT NOT NULL,
  stage TEXT NOT NULL
    CHECK (stage IN ('correctness', 'perf_screen', 'sla_bench')),
  verdict TEXT NOT NULL,
  report JSONB NOT NULL DEFAULT '{}'::jsonb,
  evidence_s3_url TEXT,
  gpu_sku TEXT,
  mock BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (submission_id, stage)
);

CREATE INDEX IF NOT EXISTS bench_reports_submission_id_idx
  ON bench_reports (submission_id);

ALTER TABLE campaigns
  ADD COLUMN IF NOT EXISTS bench JSONB;
