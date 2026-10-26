# LLM review gate (d2)

An optional, reject-only gate that asks an LLM whether a patch lowers
activation or KV-cache precision. It runs after the surface check (d) and
before the hermetic build (e).

It is a deterrent, not a proof. The gate always ends in pass, reject or (on
an outage) skip, but the model's judgement is not guaranteed: it can miss a
change or flag a legitimate one, and `temperature: 0` is not deterministic
across calls, which is why verdicts are cached. Dtypes chosen through string lookups, enum
integers or casts inside kernels are easy to miss, and a legitimate FP8-KV
kernel looks a lot like a cheat. The calibrated correctness bars stay the
numerical authority; treat every `llm_review_rejected` as a candidate for
manual review.

## What it does

- One tool-free `POST <base_url>/chat/completions` call with a fixed system
  prompt and `temperature: 0`.
- The diff is sent verbatim inside a fence tag derived from the patch hash,
  so the patch cannot close the fence itself. Comments are not stripped: a
  line-level stripper can be fooled into hiding real code (a line starting
  with `#` inside a triple-quoted string that closes and continues as code).
  Instead the prompt treats comments as untrusted and rejects text addressed
  to the reviewer.
- The answer must be one JSON object:
  `{"verdict": "pass" | "reject", "reason": str, "offending_hunks": [{"file", "hunk", "why"}]}`.
  Every cited hunk must exist in the diff. Anything else is invalid output.
- The patch is never modified. `patch_hash` binds the exact bytes the miner
  committed, so the gate can only pass or reject.

## Failure handling

If the provider is down, the gate is **skipped** and the submission goes on to
the build, so an outage doesn't cost miners their fee. "Down" means a
connection error, a timeout, HTTP 429, any 5xx, or an auth or credit error on
the operator's key (401/402/403/404). The worker logs a warning, and the skip
is not cached, so a later submission of the same patch is reviewed.

Everything the patch itself could cause still rejects:

| reason | cause |
| --- | --- |
| `llm_review_rejected` | the model returned `reject`; `llm_reason` and `offending_hunks` are in the event detail |
| `llm_review_unconfigured` | base URL or API key is not set (a deploy error, not an outage) |
| `llm_review_request_rejected` | HTTP 400, 413 or 422 on the request |
| `llm_review_invalid_output` | off-schema JSON, unknown hunk, or `finish_reason` other than `stop` |
| `llm_review_patch_too_large` | stripped diff longer than `PARETON_LLM_REVIEW_MAX_INPUT_CHARS` |
| `llm_review_invalid_patch` | patch is not UTF-8 |

Valid verdicts are cached on disk, keyed by patch hash, policy, model and a
hash of the system prompt, so a retry of the same patch gets the same answer.
Failures and skips are not cached.

## Engy

Engy serves an OpenAI-compatible API at `https://api.engy.ai/v1` with
`Authorization: Bearer` auth, and Kimi K3 as `kimi-k3` with a 1M-token
context ([docs](https://engy.ai/docs)). Its docs don't say whether
`response_format` is honoured, so JSON mode stays off by default.

## Enabling it for a campaign

The review is a `bench.llm_review` field, added only when set, so existing
campaigns keep their manifest hashes:

```bash
python -m campaign.seed ... \
  --llm-review dtype_precision_v1 \
  --llm-review-model kimi-k3
```

The model id is pinned in the manifest. The endpoint and key live in `.env`:

| variable | default | notes |
| --- | --- | --- |
| `PARETON_LLM_REVIEW_BASE_URL` | unset | the part before `/chat/completions`; Engy is `https://api.engy.ai/v1` |
| `PARETON_LLM_REVIEW_API_KEY` | unset | sent as `Authorization: Bearer` |
| `PARETON_LLM_REVIEW_TIMEOUT_S` | 300 | |
| `PARETON_LLM_REVIEW_MAX_OUTPUT_TOKENS` | 16384 | leaves room if the model thinks before answering |
| `PARETON_LLM_REVIEW_MAX_INPUT_CHARS` | 1000000 | larger patches are rejected, not truncated |
| `PARETON_LLM_REVIEW_JSON_MODE` | false | sends `response_format: {"type": "json_object"}` |
| `PARETON_LLM_REVIEW_CACHE_DIR` | `$PARETON_WORK_DIR/llm-review` | |
