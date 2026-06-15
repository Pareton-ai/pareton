-- Allow the same submission to appear in multiple evaluation rounds.
-- Previously UNIQUE (hotkey, commit_block) kept only the latest round row.

ALTER TABLE evaluations
  DROP CONSTRAINT IF EXISTS evaluations_hotkey_commit_block_key;

ALTER TABLE evaluations
  DROP CONSTRAINT IF EXISTS evaluations_pkey_hotkey_commit;

ALTER TABLE evaluations
  ADD CONSTRAINT evaluations_hotkey_commit_block_evaluation_block_key
  UNIQUE (hotkey, commit_block, evaluation_block);
