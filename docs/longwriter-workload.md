# LongWriter campaign workload

The Qwen SGLang launch helper uses `zai-org/LongWriter-6k` at revision
`0db15c0624f19d63e2efe1021595af933cc5b6cc` (6000 rows). Sampler version 4
reads the `messages` user/assistant pair. It renders only the original user
request with the pinned model template and thinking disabled. It measures real
input lengths without padding or the SWE trajectory sampler's fixed input tiers.
Versions 1 through 3 keep their existing behavior and receipt formats.

The model, engine, hardware, serving arguments, 32-request workload, 2 ms
arrivals, fees and emissions remain as configured in the seed helper.
`max_tokens=5120` remains a ceiling. Requests respect EOS and never set a
minimum generation length. Long reference answers are a source filter, not
evidence that the deployed model will generate long answers.

## Qualify the source pool

Start a trusted baseline endpoint with the exact model, published image and
serving arguments from `ops/seed-sglang-qwen38-27b.sh`. The endpoint must expose
`/v1/completions`, `/tokenize` and `/server_info`. Use a dedicated endpoint:
qualification submits real inference work. From the repository root:

```bash
python -m bench.qualify_longform \
  --base-url http://127.0.0.1:8000 \
  --engine-ref "$NATIVE_ENGINE_REF" \
  --output-dir /workspace/longwriter-qualification \
  --pool-size 128 --repetitions 2
```

The qualifier checks input tokenization and capacity against the server. The
operator must ensure the endpoint runs the specified trusted image and serving
settings; the HTTP API does not attest to the image digest. It scans rows in a
fixed hash order, skips malformed rows and short reference answers, and rejects
duplicate rendered prompts. Each accepted row must produce at least 5000 tokens
in every repetition, with nonempty text that passes the harness's repetition
checks. Generation stops normally or reaches its 5120-token ceiling. Reaching
the ceiling without suppressing EOS qualifies the row, but does not establish
where it would naturally end with a larger allowance.

`qualification.jsonl` records the contract and generated responses for review.
`sampling_rule.json` pins the accepted row indices and hashes the evidence and
campaign settings. `summary.json` records the scope of the run. There is no
automatic semantic-quality judge in this step; review the saved outputs and run
the full correctness benchmark before launch. If fewer than the requested pool
size qualify, evidence is retained but no launch rule is written. Use a fresh
directory for another attempt; failures never silently fall back to short rows.

This is sequential output qualification. It does not validate concurrent
latency, full correctness, or candidate performance. Run the standalone sample
with the qualified rule and a fresh directory:

```bash
bash ops/sglang-sample-round/run.sh \
  /workspace/pareton-sample-round-longwriter \
  /workspace/longwriter-qualification/sampling_rule.json
```

The standalone runner uses the fixture's published baseline image. Qualify that
same image when using this runner. Inspect `bench_report.json` and its evidence.
Every measured baseline and drift repetition must still meet the output-length
floor. If one stops short under load, the round fails as a baseline workload
error. It does not force continuation, drop the request or penalize a miner.
Candidate outputs continue through the existing length, correctness and timing
checks.

## Open the campaign

After qualification and full GPU validation, run on the configured controller:

```bash
bash ops/seed-sglang-qwen38-27b.sh "$NATIVE_ENGINE_REF" "$INITIAL_FEE_TAO" \
  /workspace/longwriter-qualification/sampling_rule.json
```

The optional third argument selects the qualified rule. The two-argument form
still reads the fixture rule, which is an unqualified template until replaced
with qualification output. Open-campaign preflight rejects missing or stale
qualification before inserting a profile or campaign. A different model, image,
serving configuration or sampling rule requires requalification.

Rounds sample distinct rows from the frozen eligible pool using the chain seed.
Receipts pin selected rows, reference hashes, rendered input token hashes,
tokenizer/template metadata and the trace hash. Worker replay fetches only the
selected rows and fails if the source or rendering contract changes. No database
migration is required, and existing campaigns are not rewritten.

## Natural-output repetition enforcement

For every retained correctness prompt, every measured natural-output repetition
must pass the absolute repetition checks on its full reconstructed text, including
text beyond the baseline's output length. This also applies when generation hits
the 5120-token ceiling. A looping non-median repetition fails correctness even
when the latency-median response is clean. The policy applies to all normal-EOS
workloads, not only LongWriter; it is deliberately stricter than median-only
grading. Logprob checks and existing candidate length checks still use the
latency-median response.

Correctness evidence records per-repetition outcomes in `repetition_degeneracy`;
full texts remain in the corresponding SLA `rep_N/requests.jsonl` evidence. The
existing character n-gram and repeated-span thresholds, including the thinking /
answer split, are unchanged. These are heuristic repetition checks, not a
semantic-quality guarantee or a detector for every short repeated passage.

The existing baseline instability policy is unchanged: if any measured natural
baseline repetition is degenerate, that correctness prompt is excluded for all
engines with an audited reason. More than four exclusions fail the round; an
empty retained set cannot pass. Qualification already rejects degenerate outputs,
but qualification alone does not establish behavior under concurrent round load.

Speculative decoding receives the same text checks. Stream deltas are joined
before grading; token counts and chunk boundaries cannot truncate the inspected
text. Existing timing validation rejects streams with fewer inter-chunk gaps than
reported completion tokens require, so some coalesced speculative streams remain
incompatible with SLA timing. Forced-tail diagnostic exemptions require both a
forced baseline probe reference and the original request's `ignore_eos=true`;
normal-EOS requests cannot inherit them.
