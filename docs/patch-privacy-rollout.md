# Permanent patch privacy rollout

The API code and offline tests do not establish that deployed artifacts are
private. Record deployment revisions, verification times, and results in a
restricted operator record. This checklist remains pending until verified live.
Do not attach patch contents, credentials, or evidence bundles to public PRs.

1. Deploy authenticated legacy patch readers to the watcher and every worker
   before restricting storage. Confirm each running process uses the new
   `storage.s3.fetch_patch_bytes` implementation. Using its service identity,
   fetch both a legacy patch and a private upload and check their committed
   SHA-256 hashes. Preserve existing object keys and stored audit evidence.
2. Inventory patch and evidence objects, including versions and uploads without
   submission rows. Merge `ops/aws/s3-bucket-policy.json` into the live policy,
   preserving unrelated statements. Confirm the configured account matches all
   intended service readers before applying the deny. Inspect alternate buckets,
   public ACLs, and public copies too; the checked-in policy covers only its named
   bucket and prefixes. Retain originals and historical versions privately.
3. Against known existing legacy/private patches and evidence objects, verify
   anonymous GET, HEAD, ranged GET, and versioned GET are denied. A nonexistent
   object or a network failure is not evidence of access control. Repeat signed
   reads and checksum checks from the deployed workers after restriction, and
   verify a new signed upload and internal read succeed. Realized traces remain
   public under the intended policy.
4. Inventory registries and mirrors containing candidate images. Restrict their
   packages, which contain patched source, and verify anonymous manifest and
   layer access fail for actual candidate digests. Use an empty credential store
   for anonymous checks. Verify an authenticated pull from the worker/GPU runtime
   succeeds after restriction, forcing registry access instead of relying on a
   locally cached image. Keep trusted baseline publication separate.
5. Purge CDN/proxy caches for patch and evidence objects and prior public API or
   build-log responses. Check alternate domains and origin credentials; origin
   denial alone does not invalidate cached copies. Record invalidation completion
   and repeat anonymous checks through each public delivery path.

Mark rollout complete only when every applicable check passes. Record any
inapplicable cache or mirror checks with the inventory supporting that finding.
Previously downloaded copies and browser caches cannot be recalled. If internal
access fails, repair reader credentials or permissions without restoring public
patch access.
