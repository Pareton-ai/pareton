# Review gates (d1, d2)

Two optional, reject-only gates run after the surface check (d) and before the
hermetic build (e):

- **d1, obfuscation scan** (`patch_limits.reject_obfuscation`): a
  deterministic scan of added lines that rejects code which hides what it does.
- **d2, LLM review** (`bench.llm_review`): asks an LLM whether a patch lowers
  activation or KV-cache precision.

Enable them together. d1 forces a dtype change to be written plainly, and d2
looks for it. Neither rewrites the patch.

## Obfuscation scan (d1)

`gate/obfuscation.py` scans only the lines a patch adds, so upstream code that
already uses these constructs is not penalised. Any hit rejects with
`obfuscation_detected`; the event detail lists `rules` and up to 50
`findings` (file, new line number, rule, snippet).

| rule | files | catches |
| --- | --- | --- |
| `dynamic_code` | Python | `exec`/`eval`/builtin `compile`, `__import__`, `importlib.import_module`, `pickle`/`marshal` loads, `builtins`, `types.CodeType` |
| `dynamic_code` | C family | `dlopen`, `dlsym`, `system`, `popen`, `exec*` |
| `encoded_payload` | Python | base64/hex/codecs/rot13 decoding, `zlib`/`lzma`/`bz2`/`gzip` decompress |
| `encoded_payload` | C family | universal character names (`\u0061`), trigraphs |
| `reflection` | Python | `getattr`/`setattr` on `torch`, `sys`, `os`, `config`, `server_args` with a non-literal name; writes through `globals()`, `sys.modules[...]`, `__dict__[...]`, `__code__`, `operator.attrgetter` |
| `runtime_native_code` | Python | `load_inline`, `cpp_extension`, `ctypes.CDLL`, `cffi`, `nvrtc`, `cupy.RawKernel`, `torch.ops.load_library` |
| `env_mutation` | Python | writing `os.environ` or `os.putenv` |
| `encoded_blob` | all | a 120+ character base64 run or 128+ character hex run |
| `reviewer_directed_text` | all | "ignore previous instructions", "you are an LLM/reviewer", "reviewer:", `"verdict":`, "precision-neutral" |
| `invisible_or_bidi_char` | all | bidi controls, zero-width and other format characters (Trojan Source) |
| `non_ascii_letter` | all | non-ASCII letters, including look-alike homoglyphs |

Python covers PyTorch, Triton, TileLang and CuTe DSL code. The C family
(`.c .cc .cpp .cxx .h .hh .hpp .cu .cuh .inl .tcc`) covers CUDA/C++ such as
FlashInfer, DeepGEMM and CUTLASS. Inline PTX `asm`, token pasting and plain
literal `getattr(config, "x")` are allowed.

Calibration: treating every line of upstream `sglang/srt` 0.5.22 (881k lines)
as added produces 318 findings, mostly in `environ.py`, lazy-import plumbing
and chat templates, so ordinary kernel and scheduler work rarely trips it. Two
real Kimi-K3 candidate patches produce none. A legitimate patch that needs one
of these constructs has to be written another way.

It is deterministic, so resubmitting a tweaked patch is not a fresh roll. It
is still a blocklist: a construct it doesn't know about gets through, and then
belongs in the rules.

# LLM review (d2)

It is a deterrent, not a proof. The gate always ends in pass or reject (an
outage defers the job until it can decide), but the model's judgement is not guaranteed: it can miss a
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

The review is never skipped. A patch can provoke some provider failures itself
(a huge diff, or text that makes the model think until the timeout), so a skip
would be a way around the gate.

If the provider looks down (a connection error, a timeout, HTTP 429, any 5xx,
or 401/402/403/404 on the operator's key), the worker **defers** the job: it
goes back to `pending` with `retry_after` set, and the submission stays at
`surface_ok` with no new event. The delay is
`RETRY_BASE_S * 2**(attempt - 1)`, capped at `RETRY_MAX_S` (5, 10, 20, 40, 60
minutes by default). After `MAX_ATTEMPTS` claims the submission is rejected
with `llm_review_unavailable`. Deferrals are not cached.

Requires `db/migrations/20261026_submission_jobs_retry_after.sql`, applied
before deploying (and on the Neon test branch).

Everything else rejects at once:

| reason | cause |
| --- | --- |
| `llm_review_unavailable` | provider still down after `MAX_ATTEMPTS` deferrals |
| `llm_review_rejected` | the model returned `reject`; `llm_reason` and `offending_hunks` are in the event detail |
| `llm_review_unconfigured` | base URL or API key is not set (a deploy error, not an outage) |
| `llm_review_request_rejected` | HTTP 400, 413 or 422 on the request |
| `llm_review_invalid_output` | off-schema JSON, unknown hunk, or `finish_reason` other than `stop` |
| `llm_review_patch_too_large` | diff longer than `PARETON_LLM_REVIEW_MAX_INPUT_CHARS` |
| `llm_review_invalid_patch` | patch is not UTF-8 |

Valid verdicts are cached on disk, keyed by patch hash, policy, model and a
hash of the system prompt, so a retry of the same patch gets the same answer.
Failures and deferrals are not cached.

## Engy

Engy serves an OpenAI-compatible API at `https://api.engy.ai/v1` with
`Authorization: Bearer` auth, and Kimi K3 as `kimi-k3` with a 1M-token
context ([docs](https://engy.ai/docs)). Its docs don't mention
`response_format`, structured outputs or grammar-constrained decoding, and its
request schema accepts unknown fields, so an unsupported `response_format`
may be silently ignored. Probe it with `scripts/llm_review_manual.py
--probe-response-format` before turning it on. Even with constrained
decoding, the gate still validates the verdict: a schema can't check that a
cited hunk exists in the diff.

## Enabling it for a campaign

Both are campaign fields added only when set (`bench.patch_limits.reject_obfuscation`
and `bench.llm_review`), so existing campaigns keep their manifest hashes:

```bash
python -m campaign.seed ... \
  --reject-obfuscation \
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
| `PARETON_LLM_REVIEW_RESPONSE_FORMAT` | off | `json_object`, or `json_schema` to send the strict verdict schema |
| `PARETON_LLM_REVIEW_MAX_ATTEMPTS` | 6 | claims before an outage becomes a reject |
| `PARETON_LLM_REVIEW_RETRY_BASE_S` | 300 | first deferral; doubles each attempt |
| `PARETON_LLM_REVIEW_RETRY_MAX_S` | 3600 | cap on one deferral |
| `PARETON_LLM_REVIEW_CACHE_DIR` | `$PARETON_WORK_DIR/llm-review` | |
