-- Apply BEFORE deploying: claim_next_job reads submission_jobs.retry_after.
-- Additive and nullable; existing jobs keep NULL and are claimable as before.
BEGIN;
ALTER TABLE submission_jobs ADD COLUMN IF NOT EXISTS retry_after TIMESTAMPTZ;
COMMIT;
