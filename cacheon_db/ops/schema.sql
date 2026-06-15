-- Cacheon validator structured state schema (Neon Postgres).

CREATE TABLE IF NOT EXISTS validator_meta (
  id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  last_scan_block INTEGER NOT NULL DEFAULT 0,
  last_weights_set_block INTEGER NOT NULL DEFAULT 0,
  schema_version INTEGER NOT NULL DEFAULT 1,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO validator_meta (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS evaluations (
  id BIGSERIAL PRIMARY KEY,
  hotkey TEXT NOT NULL,
  commit_block INTEGER NOT NULL,
  uid INTEGER NOT NULL,
  image TEXT NOT NULL,
  digest TEXT NOT NULL,
  score DOUBLE PRECISION NOT NULL DEFAULT 0,
  speed_improvement DOUBLE PRECISION NOT NULL DEFAULT 0,
  token_match_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
  disqualified BOOLEAN NOT NULL DEFAULT false,
  disqualify_reason TEXT,
  evaluated_at DOUBLE PRECISION NOT NULL,
  evaluation_block INTEGER NOT NULL,
  per_prompt JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (hotkey, commit_block, evaluation_block)
);

CREATE INDEX IF NOT EXISTS evaluations_evaluated_at_idx ON evaluations (evaluated_at DESC);
CREATE INDEX IF NOT EXISTS evaluations_evaluation_block_idx ON evaluations (evaluation_block);
CREATE INDEX IF NOT EXISTS evaluations_hotkey_idx ON evaluations (hotkey);
CREATE INDEX IF NOT EXISTS evaluations_uid_idx ON evaluations (uid);
CREATE INDEX IF NOT EXISTS evaluations_disqualified_idx ON evaluations (disqualified);

CREATE TABLE IF NOT EXISTS leader_state (
  role TEXT PRIMARY KEY CHECK (role IN ('leader', 'runner_up')),
  uid INTEGER NOT NULL,
  hotkey TEXT NOT NULL,
  commit_block INTEGER NOT NULL,
  image TEXT NOT NULL,
  digest TEXT NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  speed_improvement DOUBLE PRECISION NOT NULL,
  token_match_rate DOUBLE PRECISION NOT NULL,
  evaluated_at DOUBLE PRECISION NOT NULL,
  evaluation_block INTEGER NOT NULL,
  won_at_block INTEGER NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS leader_history (
  id BIGSERIAL PRIMARY KEY,
  ts DOUBLE PRECISION NOT NULL,
  block INTEGER NOT NULL,
  new_leader_uid INTEGER NOT NULL,
  new_leader_hotkey TEXT NOT NULL,
  new_leader_score DOUBLE PRECISION NOT NULL,
  new_leader_image TEXT NOT NULL,
  new_leader_digest TEXT NOT NULL,
  overtake_threshold DOUBLE PRECISION NOT NULL,
  prev_leader_uid INTEGER,
  prev_leader_hotkey TEXT,
  prev_leader_score DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS leader_history_block_idx ON leader_history (block DESC);

CREATE TABLE IF NOT EXISTS precheck_failures (
  hotkey TEXT NOT NULL,
  commit_block INTEGER NOT NULL,
  reason TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (hotkey, commit_block)
);

CREATE TABLE IF NOT EXISTS eval_jobs (
  id BIGSERIAL PRIMARY KEY,
  block INTEGER NOT NULL,
  block_hash TEXT NOT NULL,
  challengers JSONB NOT NULL,
  leader JSONB,
  runner_up JSONB,
  created_at DOUBLE PRECISION NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'complete', 'failed')),
  created_db_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_progress (
  id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  status TEXT NOT NULL DEFAULT 'idle' CHECK (status IN ('idle', 'running', 'complete')),
  phase TEXT,
  detail TEXT,
  round_block INTEGER,
  current_idx INTEGER,
  challengers JSONB,
  leader JSONB,
  runner_up JSONB,
  gpu JSONB,
  steps JSONB,
  started_at DOUBLE PRECISION,
  updated_at DOUBLE PRECISION,
  completed_at DOUBLE PRECISION
);

INSERT INTO eval_progress (id, status) VALUES (1, 'idle') ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS content_fingerprints (
  fingerprint TEXT PRIMARY KEY,
  uid INTEGER NOT NULL,
  hotkey TEXT NOT NULL,
  commit_block INTEGER NOT NULL,
  image TEXT NOT NULL,
  digest TEXT NOT NULL,
  registered_at DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
