# Campaign fee rollout

Fees live in `campaigns.submission_fee_history`, an append-only JSONB array. Each
entry has a decimal-string `amount_tao`, `recipient`, and `effective_from_block`.
The first entry starts at block zero. Both Python and Postgres reject non-whole
RAO amounts. No fee field participates in `manifest_hash`; signed hashes and
customer signoffs remain unchanged when fees change.

The API publishes the latest stored fee and its history, using only Neon. There
are no campaign-endpoint chain calls, watermark, or `submission_fee_at_block`.
Fee updates take effect immediately at the block observed by the admin command;
future activation blocks are not supported.

The watcher selects by **payment block**, not ingestion time. For example, if
0.15 TAO starts at block 1000, a 0.05 TAO transfer at block 999 remains valid when
its commitment is ingested after block 1000. A 0.05 transfer at block 1000 is
underpaid. A previously consumed reference remains unusable. An unconsumed retry
must meet the target campaign's terms at its payment block. Commitments without
a proof are free only if the campaign fee was zero at their commitment block.

## Before merging

Auto-deploy can reach production about a minute after a merge. **Apply the
migration before merging**, or pause `pareton-deploy.timer` until migration and
deployment finish. `ops/deploy.sh` does not migrate. Use the dedicated migration
for the existing VPS. `db/schema.sql` defines the final schema for fresh databases.
Existing databases must run the migration before reapplying it; the schema reports
that prerequisite if the fee column is missing. Only the migration assigns the
initial backfill amounts.

From a checkout containing the reviewed migration, with the production database
URL already loaded securely:

```sh
psql "$PARETON_DATABASE_URL" -v ON_ERROR_STOP=1 \
  -f db/migrations/20260917_campaign_fee_history.sql
```

The initial migration assigns **0.1 TAO to closed campaigns** and **0.15 TAO to
the open `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` campaign**, matched by
`bench.model.hf_repo`. These are explicit initial backfill amounts, effective
from block zero, rather than reconstructed historical prices. This policy
applies only to the migration; future seeds and fee changes keep their existing
behavior.

Existing fee histories, manifest hashes, and signoffs remain unchanged, including
on reruns. If a campaign without fee history is draft or an open campaign for
another model, the migration aborts the transaction instead of guessing a fee.
Configure its initial history explicitly before rerunning. The old code ignores
the new column, so migrating before merge is compatible.

Save and compare `id`, `manifest_hash`, and `customer_signoff` before and after the
migration. Verify the actual live campaign ID and fee rather than using the old
closed campaign ID from PR #126. After deployment:

```sh
curl -fsS "https://api.pareton.ai/v1/campaigns/$CAMPAIGN_ID" \
  | jq '{campaign_id, manifest_hash, submission_fee, submission_fee_history}'
sudo systemctl is-active pareton-api pareton-watcher pareton-deploy.timer
```

If the timer was paused, restart it once the migration and release are verified.
Do not revert to the global-fee watcher after publishing differing campaign fees;
its verification rules no longer match. The migration itself is additive.

## Change a fee

Run from the deployed checkout with validator database and Subtensor access:

```sh
python -m campaign.set_fee --campaign-id "$CAMPAIGN_ID" --amount-tao 0.20
```

The command reads the current chain height, locks the campaign row, validates the
amount, and appends the new fee with that block in one transaction. The API quotes
it immediately. Existing entries cannot be changed or removed. A second update
in the same block is rejected; retry after the chain advances. Only trusted
operators should have write access; the database cannot independently verify
chain height for arbitrary administrative SQL. Use this command for changes.

A payment included before the change retains its original terms. A transfer
included at or after the change must meet the new fee. The miner refreshes the
API quote before paying and validates fresh history at the actual payment block
before submitting a commitment; it never automatically sends a second payment.

For a new campaign, pass `--submission-fee-tao DECIMAL` to `python -m campaign.seed`,
or supply the second argument to the launch helper:

```sh
bash ops/seed-sglang-qwen38-27b.sh "$NATIVE_ENGINE_REF" 0.15 \
  /workspace/longwriter-qualification/sampling_rule.json
```

The initial fee is inserted with the campaign at block zero. Choose it during
seeding; use `campaign.set_fee` for subsequent immediate changes.
`--submission-fee-tao` is required. There is no environment fallback or default
amount; remove the obsolete global fee variable from the validator environment. The seed recipient
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
