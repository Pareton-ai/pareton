-- Apply BEFORE deploying. Existing campaigns remain private; hashes/signoffs stay intact.
BEGIN;
-- Canonical disclosure policy. Kept outside manifest_hash and customer_signoff.
CREATE OR REPLACE FUNCTION valid_campaign_patch_visibility(policy JSONB)
RETURNS BOOLEAN LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
  IF policy IS NULL OR jsonb_typeof(policy) <> 'object' THEN
    RETURN FALSE;
  END IF;
  IF policy = '{"mode":"private"}'::jsonb THEN
    RETURN TRUE;
  END IF;
  RETURN COALESCE(
    policy->>'mode' = 'public_after_reveal'
    AND policy ?& ARRAY['mode', 'reveal_delay_s']
    AND (policy - ARRAY['mode', 'reveal_delay_s']) = '{}'::jsonb
    AND jsonb_typeof(policy->'reveal_delay_s') = 'number'
    AND policy->>'reveal_delay_s' ~ '^[0-9]+$'
    AND (policy->>'reveal_delay_s')::NUMERIC BETWEEN 0 AND 2147483647,
    FALSE
  );
EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN
  RETURN FALSE;
END;
$$;

ALTER TABLE campaigns ADD COLUMN IF NOT EXISTS patch_visibility JSONB
  NOT NULL DEFAULT '{"mode":"private"}'::jsonb;
ALTER TABLE campaigns DROP CONSTRAINT IF EXISTS campaigns_patch_visibility_check;
ALTER TABLE campaigns ADD CONSTRAINT campaigns_patch_visibility_check
  CHECK (valid_campaign_patch_visibility(patch_visibility));
COMMIT;
