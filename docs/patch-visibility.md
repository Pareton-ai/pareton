# Campaign patch visibility

The manifest exposes an operational `patch_visibility` object. It is persisted
on the campaign and deliberately excluded from `manifest_hash`, like submission
fees. Changing it does not alter pinned benchmark terms, commitments, or customer
signoffs.

```json
{"patch_visibility": {"mode": "private"}}
```

This is the default for existing and new campaigns. Patches stay private while
the policy remains private, including after closing or archiving the campaign.
To enable delayed public disclosure:

```json
{"patch_visibility": {"mode": "public_after_reveal", "reveal_delay_s": 172800}}
```

Public mode defaults to 172800 seconds (two days). The delay must be an integer
from 0 through 2147483647; zero still requires a finalized evaluation. Private
mode accepts no delay. Unknown modes, extra fields, booleans, and fractional
seconds are rejected.

The clock starts at the first **complete** round whose entry is **scored** or
**disqualified**. Re-evaluating a leader does not restart it. Pending/running/void
rounds, infrastructure failures, build failures, and operator bans do not start
the clock. This applies to all submissions in the campaign, including those
created while reveal support was removed; no event backfill is needed.

## Configure

New campaigns accept `--patch-visibility private` (default), or add these options
to the existing `python -m campaign.seed` invocation:

```sh
--patch-visibility public_after_reveal --patch-reveal-delay-s 172800
```

For an existing campaign, use the operator command with database credentials:

```sh
python -m campaign.set_patch_visibility --campaign-id "$CAMPAIGN_ID" \
  --mode public_after_reveal --reveal-delay-s 172800

python -m campaign.set_patch_visibility --campaign-id "$CAMPAIGN_ID" --mode private
```

Updates apply to all existing submissions immediately on subsequent API reads.
Enabling reveal or shortening the delay can therefore publish already-evaluated
patches immediately. Publication is permanent: switching back to private hides
API locations and stops further publication, but **cannot revoke existing public
copies or downloads**. Retiring old public objects and caches is a separate
storage operation.

## API

Every submission response includes the effective `patch_visibility` policy from
the same database read. This lets the frontend distinguish permanent privacy
from a missing evaluation timestamp without a separate cached campaign lookup.
Private submissions omit `retrieval_url`, `patch_reveal_at`, and
`patch_download_url`, as before. Public-mode list/detail responses return a nullable
reveal time, an empty retrieval URL and null download URL. Once eligible, an
availability/download request verifies the object's checksum/size and copies it
into the public prefix. The availability response then includes the permanent
public location and a campaign-scoped download route.
Publication happens only on an explicit availability or download request, not
on ordinary list/detail reads or a background timer. List/detail always withhold
download fields and perform no S3 calls, even for already-revealed patches, so
a cold cache or storage outage cannot delay those responses.
The original private locator is still scrubbed from event/job diagnostics.

Both routes are restored:

- `/v1/campaigns/{campaign_id}/submissions/{patch_hash}/patch`
- `/v1/submissions/{patch_hash}/patch` (409 if the hash spans campaigns)

The frontend polls the campaign-scoped
`/v1/campaigns/{campaign_id}/submissions/{patch_hash}/patch-availability` endpoint.
It returns `{"submission": ...}` with the policy, reveal time, and verified public
location when available. Private or not-yet-eligible patches return 200 with
locations withheld. Each availability request can publish at most one patch.

The download routes return 403 with `patch_private` or `patch_not_revealed` while
withheld and 307 to the public object once eligible. Publication failures return retryable
503 responses from the availability/download routes; list/detail remain available
with download fields withheld. List/detail, availability, and download responses
use `no-store`.

## Rollout

Run these commands on the Linux validator host. They assume the deployed checkout
is `/opt/pareton`, systemd uses `pareton-deploy`, and `psql`, `jq`, and `curl` are
installed. Use the existing protected `.env`; never paste database or S3 secrets
into the commands. This procedure is for operators to execute after review; the
PR itself does not migrate, deploy, merge, or change live campaign policy.

### 1. Pause automatic deployments and load the service environment

Start a root Bash shell and keep it open through the rollout:

```bash
sudo -i
```

```bash
set -e
cd /opt/pareton
systemctl stop pareton-deploy.timer
while systemctl is-active --quiet pareton-deploy.service; do sleep 5; done
/usr/local/lib/pareton-ops/release.py status
jq -e '.phase == "idle" and .failure_step == null' \
  /var/lib/pareton-deploy/release-state.json
set -a
. /opt/pareton/.env
set +a
umask 077
ROLLOUT_DIR=$(mktemp -d /var/tmp/pareton-patch-visibility.XXXXXX)
printf 'Rollout evidence: %s\n' "$ROLLOUT_DIR"
```

Proceed only when the release is idle with no failure. Stopping the timer does
not stop a deployment already in progress. Do not force-reset the live checkout
or restart workers to bypass the release coordinator.

### 2. Apply the reviewed migration before merging/deploying the backend

Fetch the PR branch without checking it out in `/opt/pareton`. Inspect the resolved
commit against the reviewed PR head before running the migration:

```bash
git fetch origin arpan/campaign-patch-visibility
BACKEND_COMMIT=$(git rev-parse FETCH_HEAD)
git show --no-patch --format=fuller "$BACKEND_COMMIT"
git show "$BACKEND_COMMIT:db/migrations/20261001_campaign_patch_visibility.sql" \
  > "$ROLLOUT_DIR/migration.sql"
psql "$PARETON_DATABASE_URL" -X -v ON_ERROR_STOP=1 -c \
  'COPY (SELECT id, manifest_hash, customer_signoff FROM campaigns ORDER BY id) TO STDOUT WITH CSV HEADER' \
  > "$ROLLOUT_DIR/pins-before.csv"
psql "$PARETON_DATABASE_URL" -X -v ON_ERROR_STOP=1 \
  -f "$ROLLOUT_DIR/migration.sql"
psql "$PARETON_DATABASE_URL" -X -v ON_ERROR_STOP=1 -c \
  'COPY (SELECT id, manifest_hash, customer_signoff FROM campaigns ORDER BY id) TO STDOUT WITH CSV HEADER' \
  > "$ROLLOUT_DIR/pins-after.csv"
diff -u "$ROLLOUT_DIR/pins-before.csv" "$ROLLOUT_DIR/pins-after.csv"
psql "$PARETON_DATABASE_URL" -X -v ON_ERROR_STOP=1 -c \
  'SELECT id, status, patch_visibility FROM campaigns ORDER BY created_at;'
```

The migration defaults existing campaigns to private, preserves hashes/signoffs,
and is rerunnable without overwriting policies. Old application code ignores
the column. Avoid concurrent campaign creation/administrative edits while taking
the before/after snapshots. Fresh installations use `db/schema.sql` instead.

### 3. Merge the backend PR, then deploy through the coordinator

After the backend PR is reviewed and merged, run:

```bash
cd /opt/pareton
git fetch origin main
EXPECTED_COMMIT=$(git rev-parse origin/main)
git merge-base --is-ancestor "$BACKEND_COMMIT" "$EXPECTED_COMMIT"
systemctl start pareton-deploy.timer
systemctl start pareton-deploy.service
/usr/local/lib/pareton-ops/release.py status
journalctl -u pareton-deploy.service -n 60 --no-pager
```

The ancestry check assumes a merge commit or fast-forward. For a squash merge,
verify the reviewed changes in the squash commit before proceeding. The normal
release may first drain active work; a successful systemd tick alone does not
mean deployment has completed. Wait for the timer to finish the release, then
run this verification block. Do not continue until it succeeds:

```bash
jq -e --arg expected "$EXPECTED_COMMIT" \
  '.phase == "idle" and .verified_commit == $expected and .failure_step == null' \
  /var/lib/pareton-deploy/release-state.json
systemctl is-active pareton-api pareton-watcher pareton-worker pareton-round-worker
curl -fsS https://api.pareton.ai/health | jq -e '.ok == true'
```

If deployment fails, leave campaign policies private and use `ops/runbook.md`
for recovery. An application rollback can leave this additive migration in place.

### 4. Verify the default private policy before enabling public disclosure

Use an existing campaign and its submission hash; these prompts do not ask for
credentials. Keep the variables for the remaining steps:

```bash
read -r -p 'Campaign UUID: ' CAMPAIGN_ID
read -r -p 'Submission patch hash (sha256:...): ' PATCH_HASH
API_BASE=https://api.pareton.ai
SUBMISSION_URL="$API_BASE/v1/campaigns/$CAMPAIGN_ID/submissions/$PATCH_HASH"
curl -fsS "$API_BASE/v1/campaigns/$CAMPAIGN_ID" \
  | jq '{campaign_id, manifest_hash, customer_signoff, patch_visibility}'
curl -fsS "$SUBMISSION_URL" \
  | jq -e '.submission.patch_visibility.mode == "private" and
           (.submission | has("retrieval_url") | not) and
           (.submission | has("patch_download_url") | not)'
test "$(curl -sS -o "$ROLLOUT_DIR/private-download.json" -w '%{http_code}' \
  "$SUBMISSION_URL/patch")" = 403
jq -e '.detail.reason == "patch_private"' "$ROLLOUT_DIR/private-download.json"
```

On a rerun, use a campaign intentionally configured private for this check.
Public mode requires authenticated read/copy access to the private source and
public read access to `<prefix>/campaigns/...`. Keep
`<prefix>/private/campaigns/...` inaccessible anonymously. Verify the actual
bucket/CDN permissions separately; API denial alone does not prove S3 privacy.

### 5. Deploy the companion frontend and verify its proxy

Merge the reviewed frontend PR and deploy through that repository's existing
hosting pipeline. No frontend process needs installing on the validator. Once
the new frontend release is live:

```bash
FRONTEND_BASE=https://pareton.ai
PATCH_PROXY="$FRONTEND_BASE/api/campaigns/$CAMPAIGN_ID/submissions/$PATCH_HASH/patch"
curl -fsS -D "$ROLLOUT_DIR/proxy-headers.txt" "$PATCH_PROXY" \
  | jq -e '.mode == "private" and .url == "" and .downloadable == false'
cat "$ROLLOUT_DIR/proxy-headers.txt"
```

Check the submission page: the artifact row should say **Private for this
campaign**. The proxy must return `Cache-Control: no-store`. Missing policy from
an older backend is intentionally interpreted as private during mixed versions.

### 6. Opt a chosen campaign into delayed publication

Only run the following when that campaign's patches are intended to become
public. This affects **existing submissions** too: already-qualified patches can
publish immediately if their delay has elapsed. Once published, changing policy
cannot revoke public copies or downloads.

```bash
cd /opt/pareton
/opt/pareton/.venv/bin/python -m campaign.set_patch_visibility \
  --campaign-id "$CAMPAIGN_ID" --mode public_after_reveal --reveal-delay-s 172800
curl -fsS "$SUBMISSION_URL/patch-availability" \
  | jq '.submission | {patch_visibility, patch_reveal_at, patch_download_url, retrieval_url}'
curl -sS -D "$ROLLOUT_DIR/download-headers.txt" \
  -o "$ROLLOUT_DIR/download-response.txt" "$SUBMISSION_URL/patch"
cat "$ROLLOUT_DIR/download-headers.txt"
curl -fsS "$PATCH_PROXY" | jq '{mode, revealAt, downloadable, url}'
```

Before eligibility, expect API 403 `patch_not_revealed`. At/after the timestamp,
expect 307 with the public destination in `Location`; the frontend should show
**Download diff** only after the API supplies a safe public URL. Publication is
lazy on availability/download reads. An S3 publication failure returns 503 from
those routes, while ordinary list/detail reads never contact S3. Use a
controlled test campaign to exercise both sides of the deadline without
shortening a real campaign's disclosure period.

To stop further disclosure for the chosen campaign:

```bash
/opt/pareton/.venv/bin/python -m campaign.set_patch_visibility \
  --campaign-id "$CAMPAIGN_ID" --mode private
curl -fsS "$PATCH_PROXY" | jq -e '.mode == "private" and .url == ""'
```

Existing public objects and downloads remain public. This feature does not
change build-log, registry, or evidence access policies. Local tests use mocked
S3 and do not establish deployed storage permissions.
