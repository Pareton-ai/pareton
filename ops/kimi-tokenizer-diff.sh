#!/usr/bin/env bash
# Compare the sampler's Kimi K3 token ids with the engine tokenizer's on the
# smoke's saved 128k prompts. CPU only; run on the GPU host after a failed
# "trusted engine tokenization differs" preflight. Prints one JSON line per
# prompt and the engine tokenizer's encode code; all files land in $OUT.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LAUNCH=${LAUNCH:-/workspace/kimi-exgentic}
OUT=${OUT:-/workspace/tokdiff}
FIELDS=$LAUNCH/campaign-fields.json
mkdir -p "$OUT"
cd "$REPO" && . .venv/bin/activate
image=$(jq -r .bench.baseline_engine_image_digest "$FIELDS")
weights=$(jq -r .weights_dir "$LAUNCH/preflight/summary.json")

# 1. The sampler's ids, exactly as candidates are encoded.
PYTHONPATH=. python - "$FIELDS" "$LAUNCH/smoke-prep/sequences.json" "$OUT/ours.json" <<'PY'
import json, sys
from bench.sampler import build_prompt_formatter, parse_sampling_rule
fields = json.load(open(sys.argv[1]))
rule = parse_sampling_rule(fields["sampling_rule"])
model = fields["bench"]["model"]
fmt = build_prompt_formatter(rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"])
rows = [
    {"id": s["id"], "prompt": s["prompt"], "ours": fmt.encode(s["prompt"])}
    for s in json.load(open(sys.argv[2]))["sequences"]
]
json.dump(rows, open(sys.argv[3], "w"))
print(f"encoded {len(rows)} prompts locally")
PY

# 2. The engine image's tokenizer, as SGLang loads it.
cat > "$OUT/engine_side.py" <<'PY'
import inspect, json, os
import sglang
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
open("/out/tokenizer_source.py", "w").write(inspect.getsource(type(tok)))
print("tokenizer class:", [c.__name__ for c in type(tok).__mro__][:4])
root = os.path.dirname(sglang.__file__)
for dirpath, _, files in os.walk(os.path.join(root, "srt")):
    for name in files:
        path = os.path.join(dirpath, name)
        if name.endswith(".py"):
            text = open(path, errors="replace").read()
            if "def tokenize" in text or '"/tokenize"' in text:
                print("sglang tokenize code in", os.path.relpath(path, root))


def first_diff(x, y):
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


for row in json.load(open("/out/ours.json")):
    prompt, ours = row["prompt"], row["ours"]
    kw = tok.encode(prompt, add_special_tokens=True)
    plain = tok.encode(prompt)
    # What the trusted check now sends: one /tokenize call per 400k-char window.
    windowed = [
        t
        for i in range(0, len(prompt), 400_000)
        for t in tok.encode(prompt[i : i + 400_000], add_special_tokens=True)
    ]
    i = first_diff(kw, ours)
    info = {
        "id": row["id"], "chars": len(prompt), "ours": len(ours),
        "engine_add_special": len(kw), "engine_plain": len(plain),
        "add_special_matches_ours": kw == ours, "plain_matches_ours": plain == ours,
        "plain_matches_add_special": plain == kw,
        "windowed_check_matches_ours": windowed == ours, "first_diff": i,
    }
    if i is not None:
        offset = len(tok.decode(kw[:i]))
        info.update(
            char_offset=offset,
            context=prompt[max(0, offset - 60): offset + 60],
            engine_next=[tok.decode([t]) for t in kw[i: i + 6]],
            ours_next=[tok.decode([t]) for t in ours[i: i + 6]],
        )
    print(json.dumps(info, ensure_ascii=False))
PY
docker run --rm --entrypoint python3 -e HF_HUB_OFFLINE=1 \
  -v "$weights:/model:ro" -v "$OUT:/out" "$image" /out/engine_side.py | tee "$OUT/report.txt"
echo "--- engine tokenizer encode path ---"
grep -n "def encode\|def _tokenize\|def tokenize\|kwargs\|MAX\|split_whitespaces\|pre_tokenizer" "$OUT/tokenizer_source.py" | head -40
echo "Full files: $OUT/report.txt $OUT/tokenizer_source.py"
