# Permanent patch privacy

Every patch stays private, including leaders, former leaders, rejected patches,
and submissions predating signed uploads. Neither elapsed time nor
`PARETON_PATCH_REVEAL_DELAY_S=0` permits publication. That setting remains only
as a rollback safeguard, with a ten-year default.

The public API returns empty `retrieval_url`, null `patch_download_url`, and null
`patch_reveal_at`. Existing patch and build-log routes return 403 with no-store.
Submission events retain their states and timestamps, but their raw detail and
evidence references are withheld. Job status and progress remain available;
raw job errors are withheld. Nested report evidence and raw error/log fields
are also removed while score and timing metrics remain available. Operators
still have the original logs and
append-only audit events. No database migration or event rewrite is needed.

## Legacy objects and rollout order

Retain legacy objects at their existing keys and revoke public reads in place.
Watcher and builder fetches use authenticated S3 GetObject for both old and new
locators. This preserves on-chain commitments, content hashes, and internal
retrieval without copying or deleting the only retained source of a patch.

1. Extend the live delay before the existing deadline and restart the API.
   This buys time for unrevealed private uploads; existing public objects remain
   accessible until the bucket policy changes.
2. Deploy the permanent-privacy code to the API, watcher, and both workers through
   the normal coordinated deployment. Verify that the workers have loaded the
   new authenticated legacy-reader code before revoking public reads.
3. With an S3 administrator identity, inventory all objects and object versions
   beneath `stage0/`, including uploads with no database row. Verify that all
   database-known patches can be read with the service identity and match their
   committed SHA-256. Keep the inventory outside the repository and protect it
   as operator data. Check lifecycle rules before retaining originals in place.
4. Merge `PrivatePatchAndEvidenceReads` from
   `ops/aws/s3-bucket-policy.json` into the live bucket policy, preserving
   unrelated statements. The checked-in account is the verified production
   service account, `820451690806`. Confirm that all internal S3 readers belong
   to this account before applying. The explicit deny covers anonymous and
   other-account reads, including object versions, regardless of public ACLs.
   It does not grant any new reader permissions or block signed uploads.
5. Verify signed GetObject and checksum validation for private and legacy
   patches, then verify unsigned GET, HEAD, ranged GET, and versioned GET cannot
   retrieve them. Repeat for evidence bundles. Workload traces remain public.
   Verify that a newly issued signed upload and its internal read still work.
6. Purge any CDN copies of patch objects and old API/build-log responses. Check
   distributions, alternate domains, and origin credentials, not only the
   configured S3 public base. Browser caches and previously downloaded copies
   cannot be recalled. Remove obsolete public ACL grants after the deny is in
   effect; do not delete retained patch objects or historical versions.

The AWS CLI preparation below is read-only and writes only local policy files.
Use an administrator profile in account `820451690806`; the application's IAM
user does not currently have ListBucket or bucket-policy administration rights.
Run from the repository root with `AWS_PROFILE` set to that profile.

```bash
umask 077
aws s3api get-bucket-policy --bucket pareton-s3 \
  --query Policy --output text > /tmp/pareton-policy-before.json
aws s3api list-object-versions --bucket pareton-s3 --prefix stage0/ \
  > /tmp/pareton-object-versions.json
python3 - <<'PY'
import json
from pathlib import Path
before = json.loads(Path('/tmp/pareton-policy-before.json').read_text())
source = json.loads(Path('ops/aws/s3-bucket-policy.json').read_text())
sid = 'PrivatePatchAndEvidenceReads'
restriction = next(s for s in source['Statement'] if s.get('Sid') == sid)
statements = before['Statement']
if isinstance(statements, dict):
    statements = [statements]
before['Statement'] = [s for s in statements if s.get('Sid') != sid] + [restriction]
Path('/tmp/pareton-policy-after.json').write_text(json.dumps(before, indent=2) + '\n')
PY
```

After the deployment and reader checks above, review the merged policy and apply:

```bash
aws s3api put-bucket-policy --bucket pareton-s3 \
  --expected-bucket-owner 820451690806 \
  --policy file:///tmp/pareton-policy-after.json
aws s3api get-bucket-policy --bucket pareton-s3 --query Policy --output text
```

AWS documents the anonymous `aws:PrincipalAccount` value and account restrictions
in its [global condition keys reference](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html#condition-keys-principalaccount).

## Containers and evidence

Patch hashes and image digests are public identifiers, not access controls.
Candidate images contain patched source, so their GHCR package must remain
private and GPU pulls must use the existing registry credentials. Keep trusted
baseline publication separate from candidate package visibility. Test anonymous
pulls against an actual candidate digest, not just the package metadata.

Evidence archives and raw compiler diagnostics can include source or paths.
Keep their objects restricted and do not publish signed GETs through public API
responses. Public score and timing metrics remain available. This change does
not introduce an operator authentication system or a miner diagnostics portal.
