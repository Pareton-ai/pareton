# Reuse the original PRO6000 qualification

If the original 32-prompt/four-tier qualification succeeded, retain its whole
`qualification/` directory. The conversion checks `sampling_rule.json`,
`qualification.jsonl` and `summary.json`, verifies their hashes and unchanged
model/image/serving/generation pins, rechecks recorded outputs, and rebuilds the
selected prompts with the pinned tokenizer. It may download cached source/tokenizer
files but makes **no Docker or GPU requests**.

It retains the 32 already-qualified 8k/16k rows (16 per tier) for the new
16-prompt C4 campaign. Original evidence is copied into `qualification/source/`;
`reuse.json` records the derivation. Only this exact narrowing is supported.
A rule alone, edited hashes, missing evidence, or changed execution pins cannot
use this path. A fresh **step 3 shadow round is still required**.

From the existing checkout with the original run's `env.sh` loaded:

```bash
cd /workspace/pareton
git pull --ff-only
source .venv/bin/activate
umask 077
export PRO6000_OLD_RUN_DIR="${PRO6000_RUN_DIR:?Load the original run env.sh first}"
export PRO6000_RUN_DIR=$(mktemp -d /var/tmp/pareton-pro6000-XXXXXX)
export PRO6000_FIELDS=fixtures/campaigns/sglang_qwen38_27b_pro6000/campaign-fields.json
export PARETON_BENCH_HEALTH_TIMEOUT_S=3600
declare -p PRO6000_OLD_RUN_DIR PRO6000_RUN_DIR PRO6000_FIELDS \
  PARETON_BENCH_HEALTH_TIMEOUT_S > "$PRO6000_RUN_DIR/env.sh"
printf 'New run: %s\n' "$PRO6000_RUN_DIR"
nohup bash ops/qualify-pro6000.sh \
  --reuse-qualification "$PRO6000_OLD_RUN_DIR/qualification" \
  >> "$PRO6000_RUN_DIR/step2.log" 2>&1 < /dev/null &
```

Watch and check the result; Ctrl-C stops only `tail`:

```bash
tail -f "$PRO6000_RUN_DIR/step2.log" || true
cat "$PRO6000_RUN_DIR/step2.exit-code"
```

After exit `0`, use [step 3 of the runbook](README.md#3-run-the-c4-shadow-benchmark)
with this **new** `PRO6000_RUN_DIR`, once any old GPU run has exited.
After reconnecting, load the new run's
`env.sh` and activate `.venv`. Do not overwrite or reuse the old shadow directory.
Preserve original artifacts if verification fails; inspect the error instead of
manually changing the rule or writing a success exit code.
