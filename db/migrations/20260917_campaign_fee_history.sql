-- Run BEFORE deploying this branch. Does not change manifest hashes or signoffs.
-- Initial migration policy only: closed campaigns receive 0.1 TAO; the open
-- RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead campaign receives 0.15 TAO.
-- These genesis entries start at block zero; they are not reconstructed prices.
-- Existing histories are preserved. Unmapped campaigns abort the transaction.
BEGIN;
LOCK TABLE campaigns IN SHARE ROW EXCLUSIVE MODE;
-- Fee amounts are decimal strings, never JSON floating-point numbers. NUMERIC
-- has no scale here: a NUMERIC(p,9) cast would silently round fractional RAO.
CREATE OR REPLACE FUNCTION valid_campaign_fee_history(history JSONB)
RETURNS BOOLEAN LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
  entry JSONB;
  amount NUMERIC;
  block NUMERIC;
  previous_block NUMERIC := -1;
BEGIN
  IF history IS NULL OR jsonb_typeof(history) <> 'array'
     OR jsonb_array_length(history) = 0 THEN
    RETURN FALSE;
  END IF;
  FOR entry IN SELECT value FROM jsonb_array_elements(history) LOOP
    IF jsonb_typeof(entry) <> 'object'
       OR NOT (entry ?& ARRAY['amount_tao', 'recipient', 'effective_from_block'])
       OR (entry - ARRAY['amount_tao', 'recipient', 'effective_from_block']) <> '{}'::jsonb
       OR jsonb_typeof(entry->'amount_tao') <> 'string'
       OR (entry->>'amount_tao') !~ '^[0-9]+(\.[0-9]+)?$'
       OR jsonb_typeof(entry->'recipient') <> 'string'
       OR (entry->>'recipient') !~ '^[^[:space:]]+$'
       OR jsonb_typeof(entry->'effective_from_block') <> 'number'
       OR (entry->>'effective_from_block') !~ '^[0-9]+$' THEN
      RETURN FALSE;
    END IF;
    amount := (entry->>'amount_tao')::NUMERIC;
    block := (entry->>'effective_from_block')::NUMERIC;
    IF amount < 0 OR amount * 1000000000 <> trunc(amount * 1000000000)
       OR amount * 1000000000 > 18446744073709551615
       OR block <> trunc(block) OR block <= previous_block
       OR block > 9223372036854775807
       OR (previous_block = -1 AND block <> 0) THEN
      RETURN FALSE;
    END IF;
    previous_block := block;
  END LOOP;
  RETURN TRUE;
EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN
  RETURN FALSE;
END;
$$;

CREATE OR REPLACE FUNCTION preserve_campaign_fee_history()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.submission_fee_history IS NOT NULL AND
     (jsonb_array_length(NEW.submission_fee_history) < jsonb_array_length(OLD.submission_fee_history)
      OR EXISTS (
        SELECT 1 FROM jsonb_array_elements(OLD.submission_fee_history) WITH ORDINALITY old_fee(value, idx)
        WHERE NEW.submission_fee_history -> (idx::int - 1) IS DISTINCT FROM value
      )) THEN
    RAISE EXCEPTION 'campaign fee history is append-only';
  END IF;
  RETURN NEW;
END;
$$;

ALTER TABLE campaigns ADD COLUMN IF NOT EXISTS submission_fee_history JSONB;
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM campaigns
    WHERE submission_fee_history IS NULL
      AND NOT (
        status = 'closed'
        OR (status = 'open' AND COALESCE(bench #>> '{model,hf_repo}', '') =
            'RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead')
      )
  ) THEN
    RAISE EXCEPTION 'initial fee backfill only covers closed campaigns and the open RadixArk campaign; explicitly configure other campaigns first';
  END IF;
END;
$$;
UPDATE campaigns SET submission_fee_history = jsonb_build_array(jsonb_build_object(
  'amount_tao', CASE WHEN status = 'closed' THEN '0.1' ELSE '0.15' END,
  'recipient', '5CiieAa5nzSMbw4LPkh2hqv9rfMPZX9ZfEcSjh3SYWNBzk3K',
  'effective_from_block', 0
)) WHERE submission_fee_history IS NULL;
ALTER TABLE campaigns ALTER COLUMN submission_fee_history SET NOT NULL;
ALTER TABLE campaigns DROP CONSTRAINT IF EXISTS campaigns_submission_fee_history_check;
ALTER TABLE campaigns ADD CONSTRAINT campaigns_submission_fee_history_check
  CHECK (valid_campaign_fee_history(submission_fee_history));
DROP TRIGGER IF EXISTS campaigns_fee_history_append_only ON campaigns;
CREATE TRIGGER campaigns_fee_history_append_only
BEFORE UPDATE OF submission_fee_history ON campaigns
FOR EACH ROW EXECUTE FUNCTION preserve_campaign_fee_history();
COMMIT;

SELECT id, status, manifest_hash, submission_fee_history FROM campaigns ORDER BY created_at;
