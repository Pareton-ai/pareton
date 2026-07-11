# Pareton Technical Decisions (Stage 0)

Record of decisions made during the Cacheon → Pareton pivot and Stage 0 build.  
Revisit this doc when something feels wrong in hindsight.

**Scope:** Stage 0 only — campaign manifest → patch commitment → provenance & build gate → content-addressed engine image.

---

## Product & scope

| Decision | Choice |
|---|---|
| Product identity | **Pareton on Bittensor SN10.** Not a rename of Cacheon SN14. |
| Cacheon SN14 | **Do not keep running.** Archive/reference only; not a parallel product path. |
| Repo strategy | **New repo** seeded from Cacheon history (`/Users/xavierlu/Desktop/pareton`). Do not develop in `/Users/xavierlu/Desktop/cacheon`. |
| Legacy Cacheon code | **Stripped** from the Pareton repo after Stage 0 foundation (validator, cacheon_db, old api, scripts, etc.). Full copy remains in the Cacheon repo if needed. |
| Frontend | **`pareton-frontend` is separate.** Stage 0 backend exposes frontend-ready read APIs only; no campaign/sign-off UI in this milestone. |
| Submission fee / payment gate | **Deferred.** No fee gate in v0. Schema/hooks should stay easy to add later. |
| Stages 1–4 | **Out of scope for v0:** correctness, perf screen, SLA benchmark, cross-env validation, scoring, weights. |

---

## Repository layout

| Decision | Choice | Why (if non-obvious) |
|---|---|---|
| Package structure | **Top-level domain packages** at repo root: `campaign/`, `chain/`, `gate/`, `builder/`, `storage/`, `db/`, `worker/`, `api/`, `config.py` | Avoided `pareton/pareton/` — repo name already *is* the product; nested package added ambiguity. |
| Entrypoints | `python -m campaign.seed`, `python -m api`, `python -m worker.main` | Matches flat layout. |
| Miner CLI | **`miner/commit_patch.py`** only. `miner/commit.py` (Docker image + payment) removed. | Stage 0 artifact is a git patch, not a container image. |
| `example-miner/` | **Deleted.** | Cacheon Docker-image sample; irrelevant to patch-based submissions. |

---

## Domain & data

| Decision | Choice |
|---|---|
| Campaign source of truth | **Neon Postgres** (`campaigns` table + related rows). Not files on disk, not on-chain. |
| Database isolation | **Dedicated Neon project** in org Pareton (`PARETON_DATABASE_URL`). Not shared with any Cacheon DB. |
| Env var | `PARETON_DATABASE_URL` — never reuse `CACHEON_DATABASE_URL`. |
| First campaign | **Pareton-owned synthetic fixture** (Qwen-style workload trace). No design partner required for v0. |
| Pin resolution (v0) | **Manual ops** via `python -m campaign.seed` (baseline commit, base image digest, trace URL/SHA pasted or defaulted). |
| Campaign immutability | Once `status = open`, manifest is frozen. Any change → new campaign. |
| Workload trace | **Minimal JSON schema now** (`schema_version`, `requests[]`, `meta`). Content-addressed (`sha256` + URL). Stage 0 does not execute it. |
| Customer sign-off (v0) | **Admin/seed only** (CLI sets sign-off on synthetic campaign). No customer UI in v0. |

### Default path globs (first synthetic campaign)

**`allowed_paths`**
- `vllm/**`

**`denied_paths`**
- `tests/**`
- `benchmarks/**`
- `.github/**`
- `docker/**`
- `**/Dockerfile*`
- `**/pyproject.toml`
- `**/setup.py`
- `**/setup.cfg`
- `**/requirements*.txt`
- `**/CMakeLists.txt`

Miners may touch engine code under `vllm/` only; cannot rewrite packaging, CI, Docker, or tests (“grades their own exam”).

---

## Chain & miner

| Decision | Choice | Why (if non-obvious) |
|---|---|---|
| Commitment API | **`set_reveal_commitment` / `get_all_revealed_commitments`** (same as Cacheon) | Reuse battle-tested chain plumbing; only the JSON payload changes. |
| Reveal scheme | **Keep reveal** (`blocks_until_reveal=1` OK for v0). | Not switching to plain `set_commitment` for v0. |
| Commitment payload | v1 JSON: `campaign_id`, `baseline_commit`, `patch_hash`, `retrieval_url` | See `chain/commitment.py`. |
| `campaign_id` on-chain | **Yes** — included in every commitment. | Supports multiple campaigns over time on one subnet. |
| Copy-attack dedupe | **Per-campaign** on `patch_hash`, first-seen wins. | Global dedupe would block legitimate reuse across campaigns. |
| Hotkey trust | From **extrinsic signer only**, never self-reported in payload. | Standard Bittensor commitment hygiene. |
| Patch hosting | **Pareton-presigned S3 upload only** in v0. | No IPFS. No bring-your-own arbitrary HTTPS URL. |
| Presign flow | Miner calls `POST /v1/uploads/patch` → PUT bytes → commits `retrieval_url` → `set_reveal_commitment`. | Miner does not need their own bucket. |
| Payment fields in commitment | **Omitted in v0.** Optional `payment_tx` / `payment_block` shape reserved for later. | Fee gate deferred. |

---

## Gates (Stage 0)

Fail-fast order: **identity → integrity → base apply → surface → hermetic build.**

| Gate | Decision |
|---|---|
| **Identity** | Registered on SN10; campaign `open` and within window; `baseline_commit` matches manifest. No whitelist / min-stake in v0. |
| **Integrity** | HTTPS GET from allowlisted Pareton S3 URL only; ≤5 MB; timeout; ≤3 retries; `sha256` must match on-chain `patch_hash`. |
| **Base apply** | Fresh checkout at pinned commit; `git apply --check` with no fuzz. |
| **Surface** | Reject: path traversal (`../`, absolute paths), symlinks, renames into denied paths, files outside `allowed_paths`, `.gitmodules` / submodule changes, binary patches, empty diffs. |
| **Build** | Apply patch inside pinned base image; **`network=none`**; push engine image tagged by `patch_hash`. |

---

## Build & registry

| Decision | Choice | Why (if non-obvious) |
|---|---|---|
| Base image owner | **Pareton** — `images/baseline/Dockerfile` | Ops pins `base_image_digest` into campaign manifest. |
| Base image contents | CUDA runtime + Python + vLLM build deps pre-baked for pinned baseline commit | Hermetic build cannot fetch during compile. |
| Build model | **Apply patch → install/rebuild vLLM inside base image** (not “miner brings full image”, not pip-only shortcut) | vLLM needs compiled extensions; full in-image rebuild is closer to production and more hermetic. |
| Registry | **GHCR** (`ghcr.io/<owner>/pareton-engine:<patch_hash>`) | Simple for a small team; ECR later if needed. |
| Build artifact cache | Built image tagged by `patch_hash`; later stages must run **exactly this artifact**, never rebuild ad hoc. | Content-addressed downstream stages. |

---

## Architecture & ops

| Decision | Choice | Why (if non-obvious) |
|---|---|---|
| Service topology (v0) | **Single worker process** + Postgres job queue; modules split cleanly (`campaign/`, `chain/`, `gate/`, `builder/`, `worker/`, `api/`) | Split into separate deployables only when builder isolation or scale forces it. |
| Queue | **Postgres-backed** (`submission_jobs`). No Redis for v0. | Low volume; Neon already in stack. |
| Chain watcher | Polling revealed commitments → enqueue submissions (idempotent on `patch_hash`). | Can stub with mock commitments for local dev. |
| Audit log | **Append-only** `submission_events` (not a mutable status column). | Dispute-resolution substrate. |
| State machine | `committed → fetched → verified → applied → surface_ok → built` (or `rejected`). | Matches engineering architecture v0 doc. |

---

## Ops decisions (2026-07-09)

| Decision | Choice | Why (if non-obvious) |
|---|---|---|
| Patch storage | **AWS S3, bucket `pareton-s3` in `us-east-2`**, prefix `stage0/` | Bucket already existed in us-east-2; config default region updated to match. Public read on `stage0/campaigns/*` via bucket policy. |
| GHCR owner | **`Pareton-ai` GitHub org** (`ghcr.io/pareton-ai/pareton-engine`) | Org exists and holds the vLLM fork; GHCR namespaces are lowercase. |
| vLLM baseline pin | **v0.24.0** = `ee0da84ab9e04ac7610e28580af62c365e898389` | Latest stable at pin time; customer had no specific version. Wired as seed default. |
| GPU SKUs in manifest | Keep `["H200-SXM-141GB", "B200"]` placeholder | Meaningless until Stage 1 benchmarks. |
| Campaign window | Open whenever ready; no fixed cadence | Ops decides when to flip `status = open`. |
| Build host | **One dedicated CPU VPS** running worker + builder + chain watcher (single process, `--scan-chain`) | vLLM compiles are CPU-heavy (30–60+ min); Mac is smoke-test only. See `docs/stage0-ops-checklist.md`. |
| SN10 wallet | **Not created yet.** Chain reads need no wallet; miner commits and future weight-setting do. | Setup steps in the ops checklist. |

---

## Stage 0 “done” bar

**Required for v0 done:**
- Synthetic campaign seeded in Neon
- Miner presign upload → patch commitment (mock or real)
- Worker runs gates a–e
- Engine image tagged by `patch_hash` (mock build OK for dev; real Docker/GHCR for prod hardening)
- Full `submission_events` audit trail populated
- Unit tests for surface-check adversarial diffs + e2e mock path

**Not required for v0 done:**
- Customer sign-off UI
- Real design-partner profile
- Live SN10 testnet commitment (nice-to-have before “testnet ready”)
- GPU eval / scoring / weights
- Payment fee gate

---

## Explicit non-goals (v0)

- Correctness / quality gate
- Performance screen + full SLA benchmark
- Cross-environment validation (multi-GPU robustness filter from product pitch)
- Reward calculation and weight-setting
- Commit-reveal upgrade for patch content (noted as future path if mempool sniping becomes a problem)
- Multi-campaign scheduling / GPU fleet orchestration beyond single open campaign
- IPFS or arbitrary miner-hosted URLs
- Sharing infrastructure with Cacheon SN14

---

## Long-term product context (not implemented yet)

From business / inference optimization docs — **for orientation only:**

- **Near-term:** Design-partner pilots (2–4 weeks) — workload profile → optimization run → before/after GPU-hours report → savings-share contract.
- **Core method:** Cross-environment robustness — gains must hold across GPU types / configs, not just one strawman baseline.
- **Long-term:** Optimization/routing layer above providers (“which stack should this workload run on?”).
- **Network status (as of docs):** Previously live on mainnet; **paused while protocol is rebuilt** — aligns with greenfield Pareton Stage 0 work.

---

## Revision history

| Date | Note |
|---|---|
| 2026-07-09 | Initial record after Stage 0 foundation + repo flatten + legacy strip. |
| 2026-07-09 | Ops decisions: AWS S3 `pareton-s3`, vLLM v0.24.0 pin, single-VPS deploy, worker `--scan-chain`. |
| 2026-07-10 | S3 ops complete: IAM `pareton-api`, bucket policy, region `us-east-2`, smoke test passed. |
