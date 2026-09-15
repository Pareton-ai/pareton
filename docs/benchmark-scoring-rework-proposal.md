**Campaign workload coverage and reliability scoring**

Approved design · Implementation updated on 15 September 2026 · GPU calibration pending

Pareton compares miner patches with a pinned baseline using the same sampled requests within each round. This proposal makes request arrival spacing configurable, samples conversation histories across input lengths, exposes thinking mode, and adds a modest reliability deduction to the existing median E2E speedup score. It retains one workload trace and the current prompt count, repetitions, and engine startup structure.

**Behavior before this change and motivation**

The sampler schedules request `i` at `i × 200 ms`. Requests dispatch independently without waiting for earlier responses. Baseline, incumbent, and challengers receive identical prompts, generation settings, order, and scheduled arrivals. See [trace construction](../bench/sampler.py) and [replay](../bench/sla_bench.py).

Round 43 of the 8,192-token campaign illustrates why spacing matters. Median full-request latency was approximately 764 ms for the baseline, 128 ms for the incumbent, and 104 ms for challenger entry 510. The selected per-request timings for both miners were below the 200 ms arrival interval, consistent with approximately one outstanding request at a time. Entry 510's selected outputs contained 20–70 tokens despite a 5,120-token output ceiling. Sources: [baseline report](https://api.pareton.ai/v1/rounds/c0ac84e8-e6c1-406e-a2b4-04859591d9d8/entries/508/report), [incumbent report](https://api.pareton.ai/v1/rounds/c0ac84e8-e6c1-406e-a2b4-04859591d9d8/entries/509/report), [challenger report](https://api.pareton.ai/v1/rounds/c0ac84e8-e6c1-406e-a2b4-04859591d9d8/entries/510/report).

The current `median_e2e_speedup` rule takes the median of per-request latency improvements. With 32 requests, the result is the average of the 16th and 17th sorted values. Six requests in entry 510 received zero for insufficient output length, yet the entry scored 0.863488. There is no separate failure-rate deduction. See [scoring](../bench/score.py).

**Configurable arrival spacing**

Add `sampling_rule.request_interval_ms`, a nonnegative integer. It belongs to the workload description; `bench.model` continues to describe the model.

`arrival_offset_ms[i] = i × request_interval_ms`

Apply the resolved schedule to every replay of the trace, including warmup. The harness continues dispatching independently without waiting for earlier responses. The proposed sampler version 3 supports this setting, defaulting to 200 when omitted. Historical sampler versions retain their current schedule and trace hashes; unsupported versions must reject the new field.

For 32 requests:

| Interval | Final scheduled arrival | Intended workload effect |
| ---: | ---: | --- |
| 200 ms | 6.2 seconds | Limited overlap for responses completing in roughly 100 ms |
| 2 ms | 62 ms | Encourage requests to accumulate before earlier responses finish |
| 0 ms | 0 ms | Release all 32 requests as one burst |

Lower spacing implicitly emphasizes performance under concurrent load. Client dispatch spread, response duration, memory capacity, and engine scheduling determine actual overlap and internal GPU batches. A serving limit of 32 sequences is an upper bound, not a measured batch size.

Use the same interval for baseline and every miner. Faster completion can still produce different concurrency under the same arrival schedule; the score measures the resulting serving performance. Input length and output allowances remain identical across engines and do not shrink when spacing is reduced.

**Median E2E score with a reliability deduction**

Retain `scoring_rule.name = "median_e2e_speedup"` and its current per-request calculation:

1. Select each engine's median-latency repetition for the request, preserving its observed timing and output.
2. Apply the existing output-length tolerance.
3. Align both engines to `K = min(baseline_tokens, candidate_tokens)`.
4. Measure `T` from dispatch through output token `K`, including queueing, prefill, and decode.
5. Calculate `s_i = 1 - T_candidate,i / T_baseline,i`, or zero when a candidate-attributable scoring check fails.

Take the median across the full request set, including zeroed requests. A valid zero improvement or negative speedup remains a successful measurement. TTFT, inter-token latency, and aggregate output throughput remain diagnostics.

Add optional `scoring_rule.failure_penalty`, a finite nonnegative coefficient. Omission or zero preserves the existing score exactly. The proposed value for new campaigns is `0.1`.

For an otherwise scoreable evaluation:

`failure_rate = failed_requests / scheduled_requests`

`penalty = failure_penalty × failure_rate`

`final_score = median(s_i) - penalty`

Count each request once using the same selected timings and per-request failure reasons as the median calculation. The denominator is the full scheduled request count; failures cannot be removed from it. Repetitions do not multiply the count. Invalid baseline measurements cannot count against a miner.

With six failures among 32 requests, coefficient 0.1 subtracts 0.01875. Applying only this deduction to entry 510's existing report changes 0.863488 to 0.844738. This illustrates the scoring change using the old timings; it does not predict performance at a shorter interval.

The deduction makes failures affect the score even when the median barely changes. Correctness disqualification and infrastructure-invalid outcomes retain their existing handling and do not become softly penalized scores.

**What the 90% output-length rule does**

The existing rule determines eligibility for speed credit on a request. It is not automatic entry rejection. For a baseline producing at least two tokens, the required candidate count is `max(2, ceil(tolerance × baseline_tokens))`. The default `scoring_rule.tolerance` is 0.9; a one-token baseline requires one candidate token.

| Baseline output | Candidate output | Existing outcome |
| ---: | ---: | --- |
| 100 tokens | 89 tokens | Zero request score with reason `candidate output below tolerance` |
| 100 tokens | 90 tokens | Eligible for speed credit; compare both timings at token 90 |
| 100 tokens | 100 tokens | Eligible for speed credit; compare both timings at token 100 |

The threshold uses the baseline's actual output, not `sampling_rule.max_tokens`. A shorter answer can be valid; this check withholds speed credit when it does materially less output work.

Separate correctness checks can disqualify an entry for empty captured outputs, degenerate repetition, or failed logprob thresholds. Request errors fail the replay under existing engine/infrastructure handling. Scorer coverage measures how many output positions could be graded, independently of the output-length tolerance. Sources: [length gate](../bench/score.py), [correctness checks](../bench/correctness.py), [entry eligibility](../bench/main.py).

**Constructing longer inputs from SWE-agent trajectories**

A larger `bench.model.max_model_len` does not create longer inputs. The sampler currently selects only the first nonempty user message, rejects it above 8,000 characters, and renders it as a single user turn. The source dataset already contains longer histories: `ai` messages hold recorded reasoning and actions, and `user` messages hold environment observations. See [current extraction](../bench/sampler.py) and the [dataset structure](https://huggingface.co/datasets/nebius/SWE-agent-trajectories#dataset-structure).

Sampler version 3 should select a conversation prefix from each sampled row:

1. Parse `trajectory` whether stored as a list or JSON string. Continue discarding dataset `system` messages and their `system_prompt` fields.
2. Map `ai` to `assistant`; preserve `user` and already normalized `assistant` roles. Read nonempty string `text`, falling back to string `content`. Preserve recorded commands and observations as text. Do not infer structured tool calls, execute commands, or stringify malformed content; skip invalid rows.
3. Retain the first user turn and subsequent complete turns through a chosen user observation, stopping before the next recorded assistant response. Exclude later turns and row-level `generated_patch` and `eval_logs`. This produces one next-response request, not an entire agent execution.
4. Render the message list once with the campaign's pinned chat template and `add_generation_prompt=true`. Count the final rendered input with the matching pinned tokenizer, including role markers and any template-generated instructions. Match the engine's special-token handling so markers are not added twice.

Store the rendered string in the existing trace prompt and continue using `/v1/completions`. The serving endpoint does not need to change for conversation-history inputs.

Replace the 8,000-character acceptance check in the new sampler path with this final token-count check. Raising `MAX_PROMPT_CHARS` to another fixed character count would still mismeasure context use. Keep the old ceiling for historical sampler versions. Do not reject a whole trajectory for its character length before selecting a valid prefix, or cut serialized text through a message or template marker.

Use the existing 32 requests across four fixed input targets: 4,096, 8,192, 16,384, and 32,768 tokens, with eight requests each. The receipt labels are `4k`, `8k`, `16k`, and `32k`. Select a complete prefix within 90–100% of its target; skip rows that cannot supply one. Use distinct rows and deterministically shuffle the final requests so input length is not tied to arrival order. These construction rules belong to sampler version 3 and do not add scoring weights or separate benchmark runs.

Fixed targets avoid requiring SWE-agent histories near the model's 262K limit. This workload measures inputs up to 32K; it does not establish performance across the entire declared context window. Profile eligible prefix lengths with the pinned tokenizer before opening a campaign. If a group cannot be filled, report the missing coverage and require a suitable pinned source or an explicitly revised workload. Do not silently substitute short prompts, repeat text, or join unrelated trajectories to claim coverage.

**Input length and output headroom**

`sampling_rule.max_tokens` remains the single adjustable output ceiling, including 5,120. `bench.model.max_model_len` is the combined input/output limit. Resolve the actual allowance separately for each request:

`request.max_tokens = min(sampling_rule.max_tokens, context_limit - input_tokens - engine_reserve)`

Validate the pinned engine's additional input and capacity limits before accepting the trace. Resolve headroom against the campaign baseline configuration and freeze it for every miner; do not let each engine silently truncate inputs or choose its own allowance. Reject inputs with no valid output headroom.

At context 262,144, all four input tiers leave room for the full 5,120-token output ceiling, including SGLang's two-token replay reserve. A context too small for the 32K input band, such as an 8K campaign, fails validation before source scanning. Version 3 does not rescale its targets to fit a smaller model window; historical sampler versions retain their existing workloads.

Actual inputs depend on available message boundaries. The pinned SGLang scheduler can impose tighter KV-capacity bounds, which the runtime preflight checks separately. See [SGLang request limits](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/sglang/srt/managers/tp_worker.py#L542) and [output clamping](https://github.com/sgl-project/sglang/blob/4c3d47f1df9dee2d77794f6fc5ef11c64817e4fc/python/sglang/srt/managers/scheduler.py#L2514). EOS can still end any response earlier. Arrival spacing changes neither the chosen inputs nor their output allowances, even when more requests overlap.

**Thinking mode and template behavior**

Add `sampling_rule.enable_thinking`, defaulting to `false` to preserve current campaign behavior. The formatter already accepts this boolean and records it in the sampling receipt, but round creation currently uses the hardcoded false default. Expose the setting through campaign parsing and trace creation; use the recorded value during reconstruction. See [formatter](../bench/sampler.py) and [round creation](../round/create.py).

For the pinned Qwen3.8-27B template, `true` opens a `<think>` block; `false` supplies an empty, closed block before generation. Enabling thinking also uses the template's default `xhigh` reasoning instruction, inserted as a system block even when the supplied history has no system message. Dataset system prompts remain excluded; this instruction comes from the pinned model template. Count it in the input length. See the [pinned template](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/tokenizer_config.json).

The same template defaults to preserving historical thinking and wraps assistant turns in thinking delimiters, including an empty block when `reasoning_content` is absent. Keep SWE-agent's recorded reasoning/actions in `content`; do not invent a separate reasoning field or split ordinary text heuristically. Pin and record the effective template defaults alongside the thinking flag. This proposal does not add separate reasoning-effort or history-preservation campaign controls.

Thinking output and the final answer share the request's output allowance. The existing completion-stream timings and length gate cover generated output together; they do not measure time to the first final-answer token separately. Enabling thinking may produce longer outputs, but does not guarantee 5,120 tokens or a completed final answer before the cap. Baseline and every miner must use the same mode, and existing correctness checks remain applicable. Calibrate thinking with the harness's existing greedy generation settings before opening the campaign; the model's recommended sampling settings differ. See [stream timing](../bench/http.py) and [Qwen guidance](https://huggingface.co/Qwen/Qwen3.8-27B-FP8#best-practices).

**What prefix warmup measures**

The current warmup replays the measured prompts before scoring, once for vLLM and twice for SGLang, for each evaluated engine. Prefix caching can then skip prefill work for matching initial tokens. Warmup timings are unscored; faster prefix reuse and residual prefill can improve measured E2E. Cache residency depends on capacity and eviction, so a warmup does not guarantee a full hit for every request. See [warmup implementation](../bench/sla_bench.py).

The longer history inputs and chosen thinking mode also apply to warmup. This expands context coverage while retaining the existing emphasis on performance after cache preparation. It does not establish cold-prefill TTFT. If every measured request actually takes about 30 seconds before its first token, 200 ms spacing accumulates all 32 requests before the first completes; a long declared context alone does not guarantee this after warmup.

**Example configuration**

This configuration fragment selects trajectory sampling, thinking-enabled generation, 32 requests spaced 2 ms apart, a 262,144-token context, and a modest reliability deduction while retaining median E2E scoring. Merge it into the existing campaign configuration, preserving the dataset, revision, row count, and model pins. Sampler version 3 fixes the four input targets independently of the context limit; output allowances derive from the existing output ceiling and available headroom. These settings are implemented; opening a new campaign also requires successful source-coverage preflight and GPU calibration.

```json
{
  "sampling_rule": {
    "algo_version": 3,
    "n_prompts": 32,
    "max_tokens": 5120,
    "request_interval_ms": 2,
    "enable_thinking": true
  },
  "bench": {
    "model": {
      "max_model_len": 262144
    }
  },
  "scoring_rule": {
    "name": "median_e2e_speedup",
    "failure_penalty": 0.1
  }
}
```

**Qwen3.8-27B seed profile**

The campaign allocates 10% of subnet emissions to a fresh leader through `emission_rule.start_weight: 0.1`. The existing linear decay reaches `floor_weight: 0` after 201600 blocks held; the starting allocation resets when a new leader takes over.

[The seed script](../ops/seed-sglang-qwen38-27b.sh) selects four RTX 5090 GPUs with tensor parallelism 4, a 262,144-token context, the sampling settings above, and the reliability coefficient 0.1. It pins memory fraction 0.85, FlashInfer attention, chunked prefill 8,192, Mamba radix caching `extra_buffer`, a 40-request serving limit, the Qwen reasoning/tool parsers, and cache reporting.

| Input group and target | Accepted rendered input tokens | Requests | Output ceiling |
| ---: | ---: | ---: | ---: |
| 4K | 3,687–4,096 | 8 | 5,120 |
| 8K | 7,373–8,192 | 8 | 5,120 |
| 16K | 14,746–16,384 | 8 | 5,120 |
| 32K | 29,492–32,768 | 8 | 5,120 |

All tiers leave room for the full output ceiling, including SGLang's two-token reserve. Complete message boundaries determine actual input length within each range. EOS may still end output early. `enable_thinking: true` controls template rendering; the server's reasoning parser alone does not enable thinking in these pre-rendered completion prompts.

The script requires a published Pareton engine digest matching its source pin. The upstream `lmsysorg/sglang:v0.5.19` runtime image is a serving reference and cannot replace that build image without the trusted offline miner-build installer. The harness stages pinned weights at `/model` and controls networking, listen address, port, and GPU allocation. `OMP_NUM_THREADS=8` and Docker `--shm-size=32g` are not campaign controls in this change; the current harness uses host IPC and a `16g` shared-memory argument. Validate the actual harness configuration during GPU calibration.

**Reporting and implementation**

Keep `GET /v1/rounds/{round_id}/entries/{entry_id}/report` and its flat `prompts` array. Preserve each request's existing speedup, aligned timings, and failure reason. Add an overall breakdown containing the unpenalized median, failure rate, and applied deduction; the existing top-level score is the final result. Existing prompt summary counts explain the failures.

The frontend should display the campaign interval, thinking mode, actual input tokens and output allowance per prompt, and the calculation from median score to final score. Keep these as additions to the existing flat report. Update the explicit API response builder and documented models to expose them. Historical reports retain their current interpretation. See [report endpoint](../api/server.py).

Implementation covers versioned trajectory selection, token counting, per-request output allowances and arrivals, thinking-mode plumbing, the reliability deduction, and additive reporting. Extend both round creation and worker reconstruction to support version 3; their current version-2-only template checks cannot simply be reused unchanged. Record dataset and tokenizer revisions, template hash/settings, context and resolved engine limits, selected row/cut points, input token counts, resolved output allowances, and interval in the sampling receipt. Identical inputs must reproduce an identical trace hash across the round creator and worker. Preserve historical manifests and traces; unsupported settings must not be silently ignored.

Validate system-message exclusion, role normalization, complete-turn selection, both thinking modes, token counts after rendering, context boundaries, missing length-group coverage, and receipt-based trace reconstruction. Also verify omitted and zero penalties against legacy scores, zero and nonzero intervals, the 89/90-token boundary, failure counting, and report-based score reconstruction. Record actual dispatch and completion timestamps per replay to verify overlap. Calibrate the selected workload and recheck promotion and drift thresholds against the resulting score distribution.

Each measured repetition still executes 32 requests under the example configuration, or 96 with the current three-repetition default. Warmup and engine startup structure remain intact. Pin the settings in a new campaign.

**Implementation and rollout**

The backend now supports explicit `algo_version: 3`; the default remains version 2. Version 3 requires full dataset and model commit revisions, uses the pinned `tokenizers` library, and records token IDs by hash alongside the rendered input count. Round creation selects distinct rows and complete cut points; workers reconstruct those exact selections from the receipt and verify the trace hash. Versions 1 and 2 retain their existing trace bytes and 8,000-character ceiling, and reject the new spacing and thinking fields rather than ignoring them.

The version 3 input targets are fixed at 4K, 8K, 16K, and 32K tokens. Each group accepts 90–100% of its target. The seeding CLI checks that the pinned source can fill all groups before inserting an open campaign. This check renders and tokenizes source histories without starting an engine. Missing coverage prevents opening; it does not silently reduce input-length coverage. Campaigns with other prompt counts divide requests across the same four groups and require at least four requests. Remainders go to the shortest groups first; requests are shuffled after selection.

For the current replay contracts, vLLM reserves no additional output tokens and SGLang reserves two. Before warmup, the harness checks each running engine's resolved context and capacity limits. The trusted baseline also verifies the sampled token IDs through its tokenization endpoint. A mismatch or a capacity limit that would shorten the workload fails validation; it does not alter the frozen requests. Live limits are recorded in `workload_preflight.json` within the engine's evidence directory because they become available after startup. The sampling receipt records the context and engine reservation contract used to construct the trace.

The score report now includes `score_breakdown` with the median, scheduled and failed request counts, failure rate, coefficient, and deduction. Each scored request records whether its failure was attributable to the candidate, classified alongside its gate reason. The existing report route adds workload settings and per-request input tokens, output allowance, and length group. Baseline timing rows receive the same input details without an invented candidate score. The dashboard shows input lengths and output allowances once in the request trace table, for both baseline and candidate reports; historical reports leave missing details absent. Baseline drift remains the existing median latency metric without the miner reliability deduction.

Replay evidence also records actual dispatch and completion offsets for every request. Version 3 checks streamed input usage against the sampled count and rejects a length-limited response that reports fewer generated tokens than its frozen allowance. Natural EOS remains allowed, with the existing output-length tolerance applied during scoring.

One version 3 validator checks context bounds, token metadata, scheduled arrivals, and length-group coverage during construction and replay loading. Receipt reconstruction uses that same validator and exact receipt/hash comparison. Strict numeric checks apply only to version 3; historical trace parsing retains its prior behavior.

Implementation verification covers role normalization and system exclusion, both thinking modes, inputs longer than 8,000 characters, context headroom, missing source coverage, exact receipt reconstruction, historical trace hashes, zero and nonzero intervals, runtime capacity failures, reliability arithmetic, report parsing, and the dashboard.

Local verification of the fixed input tiers passed 1,398 backend tests, with 39 skipped and five Docker tests deselected. Backend formatting and lint passed. Earlier frontend verification passed 196 tests, with two live-contract tests skipped, plus formatting, lint, TypeScript and the production build. The production-built report was checked in the browser using mock campaign data. The frontend implementation is in [frontend PR #79](https://github.com/Pareton-ai/pareton-frontend/pull/79).

Before activating a campaign, run GPU calibration on its pinned engine and hardware with the intended interval, context, and thinking mode. Inspect actual overlap, output lengths, cache behavior, and correctness, then review promotion and drift thresholds. Local tests do not establish that the pinned source fills every 4–32K tier or validate production performance. No campaign or production service was changed by this implementation.
