# Campaign fee rollout

Fees live in `campaigns.submission_fee_history`, an append-only JSONB array. Each
entry has a decimal-string `amount_tao`, `recipient`, and `effective_from_block`.
The first entry starts at block zero. Both Python and Postgres reject non-whole
RAO amounts. No fee field participates in `manifest_hash`; signed hashes and
customer signoffs remain unchanged when fees change.

The API publishes the fee active at the chain height it reads, plus history and
`submission_fee_at_block`. A single genesis entry needs no chain lookup. Once a
campaign has scheduled changes, chain lookup failure returns HTTP 503 rather than
guessing the active fee. Both list and detail routes use this rule. The API
service therefore needs access to the configured Subtensor network.

The watcher selects by **payment block**, not ingestion time. For example, if
0.15 TAO starts at block 1000, a 0.05 TAO transfer at block 999 remains valid when
its commitment is ingested after block 1000. A 0.05 transfer at block 1000 is
underpaid. A previously consumed reference remains unusable. An unconsumed retry
must meet the target campaign's terms at its payment block. Commitments without
a proof are free only if the campaign fee was zero at their commitment block.

## Before merging

Auto-deploy can reach production about a minute after a merge. **Apply the
migration before merging**, or pause `pareton-deploy.timer` until migration and
deployment finish. `ops/deploy.sh` does not migrate. Reapplying `db/schema.sql`
does not add columns to existing campaigns.

From a checkout containing the reviewed migration, with the production database
URL already loaded securely:

```sh
psql "$PARETON_DATABASE_URL" -v ON_ERROR_STOP=1 \
  -f db/migrations/20260917_campaign_fee_history.sql
```

The migration backfills existing campaigns, including the live SGLang Qwen3.8
campaign, at **0.15 TAO** with the known recipient. It never changes manifest
hashes or signoffs. It is safe to rerun and leaves existing histories intact.
The old code ignores the new column, so migrating before merge is compatible.

The genesis backfill assumes 0.15 TAO for all historical blocks. Before rollout,
check for unconsumed payments that must retain older global fees. If any exist,
backfill their campaigns with the actual historical amounts and activation
blocks instead. Do this in the migration transaction before enforcing immutable
history; do not invent block boundaries or rewrite history after launch.

Save and compare `id`, `manifest_hash`, and `customer_signoff` before and after the
migration. Verify the actual live campaign ID and fee rather than using the old
closed campaign ID from PR #126. After deployment:

```sh
curl -fsS "https://api.pareton.ai/v1/campaigns/$CAMPAIGN_ID" \
  | jq '{campaign_id, manifest_hash, submission_fee, submission_fee_history}'
sudo systemctl is-active pareton-api pareton-watcher pareton-deploy.timer
```

If the timer was paused, restart it once the migration and release are verified.
Do not revert to the global-fee watcher after scheduling differing campaign fees;
its verification rules no longer match. The migration itself is additive.

## Change a fee

Run from the deployed checkout with validator database access. Select a future
activation height at least 100 blocks ahead of the current chain head and after
all previously scheduled changes:

```sh
python -m campaign.set_fee --campaign-id "$CAMPAIGN_ID" \
  --amount-tao 0.20 --effective-from-block "$ACTIVATION_BLOCK"
```

The command locks the campaign row, validates the amount, and appends history in
one transaction. Existing entries cannot be changed or removed. Only trusted
operators should have write access; the database cannot independently verify
chain height for arbitrary administrative SQL. Use this command for changes.
`PARETON_SUBMISSION_FEE_TAO` remains a seed-only input (default 0.15). It is never
read by the miner or watcher to determine a runtime payment. The seed recipient
must match the recipient pinned in `campaign/fees.py` for the public CLI.

## Miner announcement

> Update your miner checkout before submitting. The CLI now reads the campaign
> fee from the API, shows the amount and recipient, and asks for confirmation
> before uploading or paying. Automated submitters must add `--yes`; optionally
> add `--max-fee-tao 0.15` to cap new payments. Remove local fee environment
> overrides. For a failed commitment after payment, reuse `--payment-block` and
> `--payment-tx` to avoid paying twice. The validator checks the fee active at
> your payment's block.

The recipient is pinned locally in the miner. An API recipient mismatch stops
payment even with `--yes`. Updating a recipient requires a reviewed CLI release.
