# Review gates (d1, d2)

Two reject-only gates run after the surface check (d) and before the hermetic
build (e). Neither rewrites the patch.

- **d1, obfuscation scan**: runs for every campaign. A deterministic scan
  that rejects added code which hides what it does.
- **d2, LLM review** (`bench.llm_review`, optional per campaign): asks an
  LLM whether a patch lowers activation or KV-cache precision. It reads the
  patch with comments removed.

d1 forces a dtype change to be written plainly, and d2 looks for it.

## Source view

Both gates read the files the patch touches, not just the diff.
`gate/source_view.py` applies the patch to a scratch git index over the
baseline checkout left by gate c (`git apply --cached`, so the working tree is
not touched) and reads the baseline and patched version of each file. If that
fails the submission is rejected with `source_view_failed`.

### Comment stripping

`gate/comment_strip.py` removes comments from whole files with a real lexer,
never from diff fragments, and checks every result before using it:

| files | lexer | check |
| --- | --- | --- |
| `.py .pyi` (PyTorch, Triton, TileLang, CuTe DSL, FlashInfer/DeepGEMM Python) | `tokenize` COMMENT tokens | the result parses to the same AST |
| `.c` | C translation phases 1 to 3 | same token sequence, no comments left |
| `.h .cc .cpp .cxx .hpp .hh .hxx .cu .cuh .inl .ipp .tcc .tpp` (CUDA, CUTLASS, CuTe, FlashInfer, DeepGEMM, sgl-kernel) | same, plus C++ raw strings and digit separators | same |

The C-family lexer handles line splices (including a `//` comment that a
trailing backslash continues onto the next line, as the compiler does),
string and character literals with encoding prefixes, raw strings
(`R"delim(...)delim"`), `#include <...>` header names and block comments
inside directives.

Stripping keeps every line, so hunk headers in the review diff point at the
real file. A file that is not UTF-8, does not tokenize or parse, or fails its
check is shown with its comments and a `# note:` line under its header, and
the gate records it in `comment_strip.comments_kept`. Other file types (YAML,
Markdown, CMake, shell) are shown as they are. Leaving a comment in is always
safe; removing code is not, so there is no fallback to a line-based stripper.

The review diff is the diff between the stripped baseline and stripped
patched files, so a comment-only change does not appear at all, and a patch
whose only changes are comments passes d2 without a model call (d1 still
scans the comments).

Validation on upstream trees (every file, not just samples):

| corpus | files | result |
| --- | --- | --- |
| Python: SGLang `srt`, FlashInfer, TileLang, Triton, CuTe DSL (CUTLASS `python/`), DeepGEMM, PyTorch, nvidia-cutlass | 8,142 | 6,979 stripped, 1,163 without comments, 0 unparseable; token stream identical apart from comments in every file |
| C/C++/CUDA: FlashInfer (incl. CCCL), DeepGEMM, CUTLASS/CuTe, PyTorch, sgl-kernel | 13,859 | 13,767 identical to `g++ -fpreprocessed -E` after removing `#pragma` lines (g++ drops `#pragma once` in a main file); 51 that g++ itself rejects pass our own check; 40 without comments |

`tests/test_comment_strip.py` runs the tricky cases against `g++` when it is
installed.

## What the public API shows

Submission events are public, and a patch can be private. Text that describes
the patch goes under `detail.private_until_reveal` in the reject event: the
d1 `findings` with their snippets, and the d2 `llm_reason` and
`offending_hunks` with the model's `why`. The API replaces it with
`"withheld"` until the patch is revealed, which never happens for a campaign in
`private` patch mode or for a patch rejected before evaluation. The public
part keeps file, line and rule for d1 and file and hunk header for d2, which
the miner can match against their own patch.

## Obfuscation scan (d1)

`gate/obfuscation.py` scans only the lines a patch adds, so upstream code that
already uses these constructs is not penalised. Any hit rejects with
`obfuscation_detected`; the event detail lists `rules` and up to 50
`findings` (file, new line number, rule, snippet).

Each rule is matched twice: on every added line, and over the whole patched
file with comments stripped and line splices joined, so a construct split
across lines (`dlopen \` then `("x", RTLD_NOW)`, `dlopen /* c */ (`, a call
broken after its name, `getattr(` with its arguments on the next line) is
caught when any of its lines is added. `token_splice` is checked per line only.

| rule | files | catches |
| --- | --- | --- |
| `dynamic_code` | Python | `exec`/`eval`/builtin `compile`, `__import__`, `importlib.import_module`, `pickle`/`marshal` loads, `builtins`, `types.CodeType`; also any bare reference to `exec`, `eval`, `compile`, `__import__`, `__builtins__` or `breakpoint` on an added line (an alias like `run = eval`, or a call split across lines), found on the token stream of the patched file |
| `dynamic_code` | C family | `dlopen`, `dlsym`, `system`, `popen`, `exec*` |
| `encoded_payload` | Python | base64/hex/codecs/rot13 decoding, `zlib`/`lzma`/`bz2`/`gzip` decompress |
| `encoded_payload` | C family | universal character names (`\u0061`), trigraphs |
| `reflection` | Python | `getattr`/`setattr` on `torch`, `sys`, `os`, `config`, `server_args` with a non-literal name; writes through `globals()`, `sys.modules[...]`, `__dict__[...]`, `__code__`, `operator.attrgetter` |
| `runtime_native_code` | Python | `load_inline`, `cpp_extension`, `ctypes.CDLL`, `cffi`, `nvrtc`, `cupy.RawKernel`, `torch.ops.load_library` |
| `env_mutation` | Python | writing `os.environ` or `os.putenv` |
| `dtype_rebinding` | Python | assigning into a framework module at the start of a statement: `torch.bfloat16 = ...`, `tl.float32 = ...`, `torch.bmm = ...`, `torch.backends...allow_tf32 = ...` (also `T.`, `cute.`, `cutlass.`, `flashinfer.`, `deep_gemm.`) |
| `dtype_rebinding` | C family | `#define`/`#undef` of `half`, `float`, `__nv_bfloat16`, `__nv_fp8_*`, `bfloat16_t`, `half_t`, `float_e4m3_t` and the like, or `using`/`typedef` that redefines one of those names |
| `token_splice` | C family | a backslash line splice right after an identifier or number character, which splits one token across two lines |
| `encoded_blob` | all | a 120+ character base64 run or 128+ character hex run |
| `reviewer_directed_text` | all | "ignore previous instructions", "you are an LLM/reviewer", "reviewer:", `"verdict":`, "precision-neutral" |
| `invisible_or_bidi_char` | all | bidi controls, zero-width and other format characters (Trojan Source) |
| `non_ascii_letter` | all | non-ASCII letters, including look-alike homoglyphs |

Python rules cover PyTorch, Triton, TileLang and CuTe DSL code; the C family
covers CUDA/C++ such as FlashInfer, DeepGEMM and CUTLASS. Inline PTX `asm`,
macro token pasting, aliases such as `using Element = cutlass::half_t;`,
`model.eval()`, `torch.compile(...)` and plain literal `getattr(config, "x")`
are allowed.

Calibration: treating every line of upstream `sglang/srt` 0.5.22 (881k lines)
as added produces about 340 findings, mostly in `environ.py`, lazy-import
plumbing, chat templates and `torch._dynamo.config` tuning, so ordinary kernel
and scheduler work rarely trips it. Two real Kimi-K3 candidate patches produced
none under the earlier rule set (before `dtype_rebinding`, `token_splice` and
the token-level check); `scripts/llm_review_manual.py` shows d1 findings for
any saved patch. A legitimate patch that needs one of these constructs has to be written
another way.

It is deterministic, so resubmitting a tweaked patch is not a fresh roll. It
is still a blocklist: a construct it doesn't know about gets through, and then
belongs in the rules.

## LLM review (d2)

It is a deterrent, not a proof. The gate always ends in pass or reject (an
outage defers the job until it can decide), but the model's judgement is not
guaranteed: it can miss a change or flag a legitimate one, and `temperature: 0`
is not deterministic across calls, which is why verdicts are cached. Dtypes
chosen through string lookups, enum integers or casts inside kernels are easy
to miss, and a legitimate FP8-KV kernel looks a lot like a cheat. The
calibrated correctness bars stay the numerical authority; treat every
`llm_review_rejected` as a candidate for manual review.

### What it does

- One tool-free `POST <base_url>/chat/completions` call with a fixed system
  prompt and `temperature: 0`. The base URL must be `https` (plain `http`
  only to this host), and redirects are not followed.
- The comment-stripped review diff goes inside a fence tag derived from the
  patch hash, so the patch cannot close the fence itself. The prompt says that
  comments were removed, and treats anything left (strings, and files marked
  `# note:`) as untrusted.
- The answer must be one JSON object:
  `{"verdict": "pass" | "reject", "reason": str, "offending_hunks": [{"file", "hunk", "why"}]}`.
  Every cited hunk must exist in the review diff. Anything else is invalid
  output.
- The patch is never modified. `patch_hash` binds the exact bytes the miner
  committed, so the gate can only pass or reject.

### Failure handling

The review is never skipped. A patch can provoke some provider failures itself
(a huge diff, or text that makes the model think until the timeout), so a skip
would be a way around the gate.

If the provider looks down (a connection error, a timeout, HTTP 429, any 5xx,
or 401/402/403/404 on the operator's key), the worker **defers** the job: it
goes back to `pending` with `last_error` starting `llm_review_deferred:`, and
the submission stays at `surface_ok` with no new event. `claim_next_job`
leaves such a job alone until `PARETON_LLM_REVIEW_RETRY_AFTER_S` (600 s by
default) has passed since it was deferred, using the existing `updated_at`
column, so no migration is needed. After `PARETON_LLM_REVIEW_MAX_ATTEMPTS`
claims the submission is rejected with `llm_review_unavailable`. Deferrals are
not cached.

Everything else rejects at once:

| reason | cause |
| --- | --- |
| `llm_review_unavailable` | provider still down after `MAX_ATTEMPTS` claims |
| `llm_review_rejected` | the model returned `reject`; `llm_reason` and `offending_hunks` are in the event detail |
| `llm_review_unconfigured` | base URL or API key is not set (a deploy error, not an outage) |
| `llm_review_request_rejected` | HTTP 400, 413 or 422 on the request |
| `llm_review_invalid_output` | off-schema JSON, unknown hunk, or `finish_reason` other than `stop` |
| `llm_review_patch_too_large` | review diff longer than `PARETON_LLM_REVIEW_MAX_INPUT_CHARS` |

Valid verdicts are cached on disk, keyed by patch hash, policy, model, a hash
of the system prompt, the review-view version (bumped when stripping
changes) and a hash of the review diff itself (the same patch can apply to two
baselines with different context), so a retry of the same patch gets the same answer. Failures and
deferrals are not cached.

### Engy

Engy serves an OpenAI-compatible API at `https://api.engy.ai/v1` with
`Authorization: Bearer` auth, and Kimi K3 as `kimi-k3` with a 1M-token
context ([docs](https://engy.ai/docs)). Its docs don't mention
`response_format`, structured outputs or grammar-constrained decoding, and its
request schema accepts unknown fields, so an unsupported `response_format`
may be silently ignored. Probe it with `scripts/llm_review_manual.py
--probe-response-format` before turning it on. Even with constrained
decoding, the gate still validates the verdict: a schema can't check that a
cited hunk exists in the diff.

## Enabling d2 for a campaign

d1 needs no setting. d2 is a campaign field added only when set
(`bench.llm_review`), so existing campaigns keep their manifest hashes:

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
| `PARETON_LLM_REVIEW_MAX_INPUT_CHARS` | 1000000 | larger review diffs are rejected, not truncated |
| `PARETON_LLM_REVIEW_RESPONSE_FORMAT` | off | `json_object`, or `json_schema` to send the strict verdict schema |
| `PARETON_LLM_REVIEW_MAX_ATTEMPTS` | 6 | claims before an outage becomes a reject |
| `PARETON_LLM_REVIEW_RETRY_AFTER_S` | 600 | wait after a deferral before the job is claimable again |
| `PARETON_LLM_REVIEW_CACHE_DIR` | `$PARETON_WORK_DIR/llm-review` | |
