-- WS-E: per-SKU bench_reports (N SKUs x 3 stages)
-- Apply manually: psql "$PARETON_DATABASE_URL" -f db/migrations/20260719_cross_env.sql

UPDATE bench_reports SET gpu_sku = 'unknown' WHERE gpu_sku IS NULL;

ALTER TABLE bench_reports
  ALTER COLUMN gpu_sku SET DEFAULT 'unknown';

ALTER TABLE bench_reports
  ALTER COLUMN gpu_sku SET NOT NULL;

ALTER TABLE bench_reports
  DROP CONSTRAINT IF EXISTS bench_reports_submission_id_stage_key;

ALTER TABLE bench_reports
  DROP CONSTRAINT IF EXISTS bench_reports_submission_id_stage_gpu_sku_key;

ALTER TABLE bench_reports
  ADD CONSTRAINT bench_reports_submission_id_stage_gpu_sku_key
  UNIQUE (submission_id, stage, gpu_sku);
