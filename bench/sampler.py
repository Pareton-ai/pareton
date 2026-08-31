"""Deterministic on-the-fly workload traces from a pinned HuggingFace dataset.

Seed = sha256(block_hash(seed block) || campaign_id).
Row i = sha256(seed || counter) % n_rows. Empty/too-long rows skip via counter.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

_SHA256_HEX_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

ALGO_VERSION = 1
MAX_PROMPT_CHARS = 8000
DEFAULT_N_PROMPTS = 32
DEFAULT_MAX_TOKENS = 128
DEFAULT_SEED_BLOCK_OFFSET = 1
CHAT_TEMPLATE_ENABLE_THINKING = False

# process-local cache: one Arrow split per pinned (dataset, revision, config, split)
_HF_SPLIT_CACHE: dict[tuple[str, str, str, str], Any] = {}


class SamplerError(ValueError):
    """Invalid sampler inputs or generation failure."""


@dataclass(frozen=True)
class SampledTrace:
    sha256: str
    body: bytes
    seed_hex: str
    sample_seed_block: int
    sample_seed_block_hash: str
    row_indices: tuple[int, ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class PromptFormatter:
    """Deterministic conversion from a dataset user message to model input."""

    render: Callable[[str], str]
    receipt: dict[str, Any]


def normalize_sha256(value: str) -> str:
    m = _SHA256_HEX_RE.fullmatch(str(value).strip())
    if not m:
        raise SamplerError(f"expected sha256:<64 hex> or 64 hex, got {value!r}")
    return f"sha256:{m.group(1).lower()}"


def parse_sampling_rule(rule: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize and validate an hf_rows sampling_rule."""
    if not isinstance(rule, dict):
        raise SamplerError("sampling_rule must be an object")
    rtype = str(rule.get("type") or "")
    if rtype != "hf_rows":
        raise SamplerError(f"unsupported sampling_rule.type: {rtype!r}")
    offset = int(rule.get("seed_block_offset", DEFAULT_SEED_BLOCK_OFFSET))
    if offset < 0:
        raise SamplerError("seed_block_offset must be >= 0")
    dataset = str(rule.get("dataset") or "").strip()
    revision = str(rule.get("revision") or "").strip()
    if not dataset or not revision:
        raise SamplerError("hf_rows rule requires dataset and revision")
    n_rows = int(rule.get("n_rows") or 0)
    n_prompts = int(rule.get("n_prompts") or DEFAULT_N_PROMPTS)
    max_tokens = int(rule.get("max_tokens") or DEFAULT_MAX_TOKENS)
    ignore_eos = rule.get("ignore_eos", False)
    if ignore_eos is None:
        ignore_eos = False
    if not isinstance(ignore_eos, bool):
        raise SamplerError("ignore_eos must be a boolean")
    algo_version = int(rule.get("algo_version") or ALGO_VERSION)
    if n_rows < 1:
        raise SamplerError("n_rows must be >= 1")
    if n_prompts < 1:
        raise SamplerError("n_prompts must be >= 1")
    if n_prompts > n_rows:
        raise SamplerError("n_prompts must be <= n_rows")
    if max_tokens < 1:
        raise SamplerError("max_tokens must be >= 1")
    if algo_version != ALGO_VERSION:
        raise SamplerError(f"unsupported algo_version: {algo_version}")
    parsed = {
        "type": "hf_rows",
        "seed_block_offset": offset,
        "dataset": dataset,
        "revision": revision,
        "config": str(rule.get("config") or "default"),
        "split": str(rule.get("split") or "train"),
        "n_rows": n_rows,
        "n_prompts": n_prompts,
        "max_tokens": max_tokens,
        "algo_version": algo_version,
    }
    if ignore_eos:
        parsed["ignore_eos"] = True
    return parsed


def _call_load_tokenizer_config(**kwargs: Any) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(filename="tokenizer_config.json", **kwargs)
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise TypeError("tokenizer_config.json must contain an object")
    return config


def _resolve_chat_template(config: dict[str, Any]) -> str:
    value = config.get("chat_template")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list):
        templates = {
            str(item.get("name") or ""): item.get("template")
            for item in value
            if isinstance(item, dict)
        }
        default = templates.get("default")
        if isinstance(default, str) and default:
            return default
        usable = [item for item in templates.values() if isinstance(item, str) and item]
        if len(usable) == 1:
            return usable[0]
    raise ValueError("tokenizer_config.json has no unambiguous chat template")


def _special_token_value(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return value


def _compile_chat_template(template: str) -> Any:
    import jinja2
    from jinja2.ext import LoopControlExtension
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(
        value: Any,
        ensure_ascii: bool = False,
        indent: int | None = None,
        separators: tuple[str, str] | None = None,
        sort_keys: bool = False,
    ) -> str:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[LoopControlExtension],
    )
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    return environment.from_string(template)


def build_prompt_formatter(
    rule: dict[str, Any],
    *,
    model_repo: str | None = None,
    model_revision: str | None = None,
    expected_template_sha256: str | None = None,
    enable_thinking: bool = CHAT_TEMPLATE_ENABLE_THINKING,
    config_loader: Callable[..., dict[str, Any]] | None = None,
) -> PromptFormatter:
    """Build a formatter from the campaign's pinned tokenizer config."""
    parse_sampling_rule(rule)
    if not isinstance(enable_thinking, bool):
        raise SamplerError("enable_thinking must be a boolean")

    repo = str(model_repo or "").strip()
    revision = str(model_revision or "").strip()
    if not repo or not revision:
        raise SamplerError("chat template formatting requires model repo and revision")
    loader = config_loader or _call_load_tokenizer_config
    try:
        config = loader(
            repo_id=repo,
            revision=revision,
            token=_hf_token(),
        )
        template = _resolve_chat_template(config)
        compiled = _compile_chat_template(template)
    except Exception as exc:
        raise SamplerError(
            f"failed to load chat template for {repo}@{revision}: {type(exc).__name__}"
        ) from exc
    if not isinstance(template, str) or not template:
        raise SamplerError(f"model {repo}@{revision} has no usable chat template")
    template_sha256 = "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()
    if expected_template_sha256:
        expected = normalize_sha256(expected_template_sha256)
        if template_sha256 != expected:
            raise SamplerError(
                f"chat template sha256 mismatch: expected {expected}, "
                f"got {template_sha256}"
            )

    def render(prompt: str) -> str:
        try:
            special_tokens = {
                key: _special_token_value(value)
                for key, value in config.items()
                if key.endswith("_token") or key == "additional_special_tokens"
            }
            rendered = compiled.render(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                documents=None,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **special_tokens,
            )
        except Exception as exc:
            raise SamplerError(
                f"chat template render failed for {repo}@{revision}: "
                f"{type(exc).__name__}"
            ) from exc
        if not isinstance(rendered, str) or not rendered:
            raise SamplerError(
                f"chat template for {repo}@{revision} rendered an empty prompt"
            )
        return rendered

    return PromptFormatter(
        render=render,
        receipt={
            "chat_template": {
                "model_repo": repo,
                "model_revision": revision,
                "sha256": template_sha256,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            },
        },
    )


def compute_sample_seed(
    *,
    block_hash: str,
    campaign_id: str,
) -> str:
    """Return 64-char lowercase hex seed (no sha256: prefix).

    The seed is per campaign and per block, so every image in one round draws
    the same prompt set. The patch hash is deliberately not in the material.
    """
    bh = str(block_hash).strip().lower()
    if bh.startswith("0x"):
        bh = bh[2:]
    if not bh or not _HEX_RE.fullmatch(bh):
        raise SamplerError(f"block_hash must be hex, got {block_hash!r}")
    material = (bh + str(campaign_id).strip().lower()).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def extract_prompt(row: dict[str, Any]) -> str:
    """First non-empty user message from a SWE-agent trajectory row."""
    traj = row.get("trajectory")
    if traj is None:
        return ""
    msgs = traj if isinstance(traj, list) else json.loads(traj)
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("text") or m.get("content") or ""
        content = str(content)
        if content:
            return content
    return ""


def prompt_ok(prompt: str) -> bool:
    return bool(prompt) and len(prompt) <= MAX_PROMPT_CHARS


def _index_at(seed_hex: str, counter: int, n_rows: int) -> int:
    material = f"{seed_hex}:{counter}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest(), 16) % n_rows


def select_row_indices(
    *,
    seed_hex: str,
    n_rows: int,
    n_prompts: int,
    row_ok: Callable[[int], bool],
) -> list[int]:
    """Pick n_prompts distinct row indices; skip rows where row_ok is False."""
    cleaned = str(seed_hex).strip().lower()
    if cleaned.startswith("sha256:"):
        cleaned = cleaned[len("sha256:") :]
    if not _HEX_RE.fullmatch(cleaned):
        raise SamplerError(f"seed_hex must be hex, got {seed_hex!r}")
    if n_rows < 1 or n_prompts < 1:
        raise SamplerError("n_rows and n_prompts must be >= 1")
    chosen: list[int] = []
    seen: set[int] = set()
    counter = 0
    max_tries = n_rows * 8
    while len(chosen) < n_prompts:
        if counter >= max_tries:
            raise SamplerError(f"could not fill {n_prompts} prompts from {n_rows} rows")
        idx = _index_at(cleaned, counter, n_rows)
        counter += 1
        if idx in seen:
            continue
        seen.add(idx)
        if not row_ok(idx):
            continue
        chosen.append(idx)
    return chosen


def build_trace_json(
    prompts: Sequence[str],
    *,
    max_tokens: int,
    meta_name: str,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    requests = []
    for j, p in enumerate(prompts):
        sampling: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0}
        if ignore_eos:
            sampling["ignore_eos"] = True
        requests.append(
            {
                "id": f"hf-{j:03d}",
                "arrival_offset_ms": j * 200,
                "prompt": p,
                "max_tokens": int(max_tokens),
                "sampling": sampling,
            }
        )
    return {
        "schema_version": 1,
        "meta": {
            "name": meta_name,
            "description": f"{len(requests)} prompts sampled on the fly",
        },
        "requests": requests,
    }


def encode_trace(trace: dict[str, Any]) -> bytes:
    return (json.dumps(trace, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _hf_token() -> str | None:
    token = (
        os.environ.get("HF_TOKEN") or os.environ.get("PARETON_HF_TOKEN") or ""
    ).strip()
    return token or None


def _call_load_dataset(**kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(**kwargs)


def _cached_hf_split(rule: dict[str, Any]) -> Any:
    key = (
        str(rule["dataset"]),
        str(rule["revision"]),
        str(rule["config"]),
        str(rule["split"]),
    )
    cached = _HF_SPLIT_CACHE.get(key)
    if cached is not None:
        return cached
    print(
        f"loading HF dataset {key[0]}@{key[1][:12]} "
        f"config={key[2]} split={key[3]} (once, then index locally)",
        file=sys.stderr,
        flush=True,
    )
    try:
        ds = _call_load_dataset(
            path=key[0],
            name=key[2],
            split=key[3],
            revision=key[1],
            token=_hf_token(),
        )
    except Exception as exc:
        raise SamplerError(f"hf load_dataset failed: {exc}") from exc
    _HF_SPLIT_CACHE[key] = ds
    return ds


def fetch_hf_row(rule: dict[str, Any], index: int) -> dict[str, Any]:
    """Return one row from the pinned HF split. Downloads parquet once per process."""
    ds = _cached_hf_split(rule)
    idx = int(index)
    try:
        row = ds[idx]
    except IndexError as exc:
        raise SamplerError(f"hf row missing at offset {index}") from exc
    if not isinstance(row, dict):
        try:
            row = dict(row)
        except Exception as exc:
            raise SamplerError(f"hf row missing at offset {index}") from exc
    return row


def generate_trace(
    *,
    rule: dict[str, Any],
    seed_hex: str,
    row_fetcher: Callable[[int], dict[str, Any]],
    prompt_formatter: PromptFormatter | None = None,
    sample_seed_block: int = 0,
    sample_seed_block_hash: str = "",
) -> SampledTrace:
    """Build a trace from hash-selected rows. row_fetcher is injected in tests."""
    parsed = parse_sampling_rule(rule)
    cache: dict[int, str] = {}

    def row_ok(idx: int) -> bool:
        if idx not in cache:
            try:
                cache[idx] = extract_prompt(row_fetcher(idx))
            except SamplerError:
                raise
            except Exception as exc:
                raise SamplerError(f"row fetch failed at {idx}: {exc}") from exc
        return prompt_ok(cache[idx])

    indices = select_row_indices(
        seed_hex=seed_hex,
        n_rows=int(parsed["n_rows"]),
        n_prompts=int(parsed["n_prompts"]),
        row_ok=row_ok,
    )
    formatter = prompt_formatter or PromptFormatter(
        render=lambda prompt: prompt,
        receipt={},
    )
    prompts = [formatter.render(cache[i]) for i in indices]
    trace = build_trace_json(
        prompts,
        max_tokens=int(parsed["max_tokens"]),
        meta_name=f"hf-rows-{seed_hex[:12]}",
        ignore_eos=bool(parsed.get("ignore_eos", False)),
    )
    body = encode_trace(trace)
    sha = "sha256:" + hashlib.sha256(body).hexdigest()
    receipt = {
        "type": "hf_rows",
        "algo_version": parsed["algo_version"],
        "dataset": parsed["dataset"],
        "revision": parsed["revision"],
        "config": parsed["config"],
        "split": parsed["split"],
        "n_rows": parsed["n_rows"],
        "n_prompts": parsed["n_prompts"],
        "max_tokens": parsed["max_tokens"],
        "seed_block_offset": parsed["seed_block_offset"],
        "sample_seed_block": int(sample_seed_block),
        "sample_seed_block_hash": str(sample_seed_block_hash).strip().lower(),
        "seed_hex": str(seed_hex).strip().lower(),
        "row_indices": list(indices),
        "sampled_trace_sha256": sha,
    }
    receipt.update(formatter.receipt)
    if parsed.get("ignore_eos", False):
        receipt["ignore_eos"] = True
    return SampledTrace(
        sha256=sha,
        body=body,
        seed_hex=str(seed_hex).strip().lower(),
        sample_seed_block=int(sample_seed_block),
        sample_seed_block_hash=str(sample_seed_block_hash).strip().lower(),
        row_indices=tuple(indices),
        receipt=receipt,
    )


def sample_workload(
    *,
    rule: dict[str, Any],
    commit_block: int,
    block_hash: str,
    campaign_id: str,
    row_fetcher: Callable[[int], dict[str, Any]] | None = None,
    prompt_formatter: PromptFormatter | None = None,
) -> SampledTrace:
    """Generate one round trace from a future-block seed."""
    if commit_block is None:
        raise SamplerError("commit_block is required for sampling")
    parsed = parse_sampling_rule(rule)
    seed_block = int(commit_block) + int(parsed["seed_block_offset"])
    seed_hex = compute_sample_seed(
        block_hash=block_hash,
        campaign_id=campaign_id,
    )
    fetcher = row_fetcher or (lambda idx: fetch_hf_row(parsed, idx))
    return generate_trace(
        rule=parsed,
        seed_hex=seed_hex,
        row_fetcher=fetcher,
        prompt_formatter=prompt_formatter,
        sample_seed_block=seed_block,
        sample_seed_block_hash=str(block_hash).strip().lower(),
    )
