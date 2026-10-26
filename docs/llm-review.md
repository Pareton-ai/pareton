# LLM review gate (d2)

An optional, reject-only gate that asks an LLM whether a patch lowers
activation or KV-cache precision. It runs after the surface check (d) and
before the hermetic build (e).

It is a deterrent, not a proof. Dtypes chosen through string lookups, enum
integers or casts inside kernels are easy to miss, and a legitimate FP8-KV
kernel looks a lot like a cheat. The calibrated correctness bars stay the
numerical authority; treat every `llm_review_rejected` as a candidate for
manual review.

## What it does

- One tool-free `POST <base_url>/chat/completions` call with a fixed system
  prompt and `temperature: 0`.
- The diff is sent with comments removed (`#` in Python, `//` and `/* */` in
  C, C++, CUDA and Rust), inside a fence tag derived from the patch hash so
  the patch cannot close the fence itself.
- The answer must be one JSON object:
  `{"verdict": "pass" | "reject", "reason": str, "offending_hunks": [{"file", "hunk", "why"}]}`.
  Every cited hunk must exist in the diff. Anything else is invalid output.
- The patch is never modified. `patch_hash` binds the exact bytes the miner
  committed, so the gate can only pass or reject.

## Failure handling

The gate fails closed. These all reject the submission:

| reason | cause |
| --- | --- |
| `llm_review_rejected` | the model returned `reject`; `llm_reason` and `offending_hunks` are in the event detail |
| `llm_review_unconfigured` | base URL or API key is not set |
| `llm_review_unavailable` | network error, timeout or non-200 response |
| `llm_review_invalid_output` | off-schema JSON, unknown hunk, or `finish_reason` other than `stop` |
| `llm_review_patch_too_large` | stripped diff longer than `PARETON_LLM_REVIEW_MAX_INPUT_CHARS` |
| `llm_review_invalid_patch` | patch is not UTF-8 |

Valid verdicts are cached on disk, keyed by patch hash, policy, model and a
hash of the system prompt, so a retry of the same patch gets the same answer.
Failures are not cached.

## Enabling it for a campaign

The review is a `bench.llm_review` field, added only when set, so existing
campaigns keep their manifest hashes:

```bash
python -m campaign.seed ... \
  --llm-review dtype_precision_v1 \
  --llm-review-model <model id as the provider names it>
```

The model id is pinned in the manifest. The endpoint and key live in `.env`:

| variable | default | notes |
| --- | --- | --- |
| `PARETON_LLM_REVIEW_BASE_URL` | unset | the part before `/chat/completions` |
| `PARETON_LLM_REVIEW_API_KEY` | unset | sent as `Authorization: Bearer` |
| `PARETON_LLM_REVIEW_TIMEOUT_S` | 300 | |
| `PARETON_LLM_REVIEW_MAX_OUTPUT_TOKENS` | 4096 | |
| `PARETON_LLM_REVIEW_MAX_INPUT_CHARS` | 1000000 | larger patches are rejected, not truncated |
| `PARETON_LLM_REVIEW_JSON_MODE` | false | sends `response_format: {"type": "json_object"}` |
| `PARETON_LLM_REVIEW_CACHE_DIR` | `$PARETON_WORK_DIR/llm-review` | |
