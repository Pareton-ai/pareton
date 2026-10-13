"""Chat formatting for pinned models without tokenizer.json or a Jinja template.

Kimi K3 ships a tiktoken vocabulary, a transformers tokenizer class and a
standard-library chat encoder. The harness rebuilds the same tiktoken encoding
from the pinned files and runs only the stdlib chat encoder, so it needs neither
transformers nor the model's tokenizer class. Each round still checks every
prompt's token IDs against the trusted engine's /tokenize endpoint.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from typing import Any

from bench.sampler import PromptFormatter, PromptRenderError, SamplerError

# Pinned revisions only: the chat encoder below is executed as Python.
TIKTOKEN_CHAT_MODELS: dict[tuple[str, str], dict[str, str]] = {
    ("moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"): {
        "vocab": "tiktoken.model",
        "tokenizer_code": "tokenization_kimi.py",
        "chat_code": "encoding_k3.py",
        "chat_function": "build_chat_segments",
    },
}

# The model tokenizer encodes longer inputs in windows; plain whole-string
# encoding is identical only below these limits, so refuse anything larger.
_MAX_ENCODE_CHARS = 400_000
_MAX_RUN_CHARS = 25_000
_LONG_RUN_RE = re.compile(rf"\s{{{_MAX_RUN_CHARS + 1},}}|\S{{{_MAX_RUN_CHARS + 1},}}")
# Stdlib only: anything else means the pinned encoder changed shape.
_ALLOWED_CHAT_IMPORTS = frozenset(
    {"__future__", "dataclasses", "json", "typing", "re", "enum", "collections"}
)


def tiktoken_chat_spec(repo: str, revision: str) -> dict[str, str] | None:
    return TIKTOKEN_CHAT_MODELS.get((repo, revision.lower()))


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _call_load_file(*, repo_id: str, revision: str, filename: str, token: Any) -> bytes:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id, revision=revision, filename=filename, token=token
    )
    with open(path, "rb") as fh:
        return fh.read()


def parse_tokenizer_constants(source: str) -> tuple[str, int]:
    """Read pat_str and the reserved special-token count without importing it."""
    tree = ast.parse(source)
    pat_str = reserved = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        name = target.id if isinstance(target, ast.Name) else None
        if name == "pat_str":
            call = node.value
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "join"
                and isinstance(call.func.value, ast.Constant)
                and len(call.args) == 1
                and not call.keywords
            ):
                raise SamplerError("pinned tokenizer pat_str has an unexpected form")
            separator = ast.literal_eval(call.func.value)
            parts = ast.literal_eval(call.args[0])
            if not isinstance(separator, str) or not all(
                isinstance(p, str) for p in parts
            ):
                raise SamplerError("pinned tokenizer pat_str has an unexpected form")
            if pat_str is not None:
                raise SamplerError("pinned tokenizer defines pat_str more than once")
            pat_str = separator.join(parts)
        elif name == "num_reserved_special_tokens":
            value = ast.literal_eval(node.value)
            if type(value) is not int or value <= 0:
                raise SamplerError("pinned tokenizer has an invalid reserved count")
            if reserved is not None and reserved != value:
                raise SamplerError("pinned tokenizer reserved count is ambiguous")
            reserved = value
    if pat_str is None or reserved is None:
        raise SamplerError("pinned tokenizer lacks pat_str or its reserved count")
    return pat_str, reserved


def parse_tiktoken_ranks(data: bytes) -> dict[bytes, int]:
    ranks: dict[bytes, int] = {}
    for line in data.splitlines():
        if not line:
            continue
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    if not ranks or len(set(ranks.values())) != len(ranks):
        raise SamplerError("pinned tiktoken vocabulary is empty or has duplicate ranks")
    return ranks


def special_tokens(config: dict[str, Any], base: int, reserved: int) -> dict[str, int]:
    decoder = config.get("added_tokens_decoder") or {}
    if not isinstance(decoder, dict):
        raise SamplerError("tokenizer_config added_tokens_decoder must be an object")
    names = {}
    for key, value in decoder.items():
        content = value.get("content") if isinstance(value, dict) else None
        if not isinstance(content, str) or not content:
            raise SamplerError("tokenizer_config has an invalid added token")
        names[int(key)] = content
    tokens = {
        names.get(i, f"<|reserved_token_{i}|>"): i for i in range(base, base + reserved)
    }
    if len(tokens) != reserved:
        raise SamplerError("tokenizer_config special tokens are not unique")
    return tokens


def load_chat_function(source: str, filename: str, function: str) -> Callable[..., Any]:
    tree = ast.parse(source, filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""] if node.level == 0 else [""]
        else:
            continue
        if any(m.split(".")[0] not in _ALLOWED_CHAT_IMPORTS for m in modules):
            raise SamplerError(f"pinned chat encoder {filename} imports beyond stdlib")
    name = "_pareton_pinned_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_loader(name, loader=None)
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolves annotations through sys.modules.
        sys.modules[name] = module
        try:
            # Allowlisted, revision-pinned, stdlib-only encoder.
            exec(compile(tree, filename, "exec"), module.__dict__)  # noqa: S102
        except BaseException:
            sys.modules.pop(name, None)
            raise
    fn = getattr(module, function, None)
    if not callable(fn):
        raise SamplerError(f"pinned chat encoder {filename} lacks {function}")
    return fn


def build_tiktoken_chat_formatter(
    *,
    repo: str,
    revision: str,
    enable_thinking: bool,
    expected_template_sha256: str | None,
    token: Any,
    file_loader: Callable[..., bytes] | None = None,
) -> PromptFormatter:
    import tiktoken

    spec = tiktoken_chat_spec(repo, revision)
    if spec is None:
        raise SamplerError(f"{repo}@{revision} is not a pinned tiktoken chat model")
    load = file_loader or _call_load_file
    try:
        files = {
            name: load(repo_id=repo, revision=revision, filename=name, token=token)
            for name in (
                "tokenizer_config.json",
                spec["tokenizer_code"],
                spec["chat_code"],
                spec["vocab"],
            )
        }
        config = json.loads(files["tokenizer_config.json"])
        if not isinstance(config, dict):
            raise TypeError("tokenizer_config.json must contain an object")
        chat_source = files[spec["chat_code"]].decode("utf-8")
        pat_str, reserved = parse_tokenizer_constants(
            files[spec["tokenizer_code"]].decode("utf-8")
        )
        ranks = parse_tiktoken_ranks(files[spec["vocab"]])
        encoding = tiktoken.Encoding(
            name=f"{repo}@{revision}",
            pat_str=pat_str,
            mergeable_ranks=ranks,
            special_tokens=special_tokens(config, len(ranks), reserved),
        )
        build_segments = load_chat_function(
            chat_source, spec["chat_code"], spec["chat_function"]
        )
    except SamplerError:
        raise
    except Exception as exc:
        raise SamplerError(
            f"failed to load pinned tiktoken chat files for {repo}@{revision}: "
            f"{type(exc).__name__}"
        ) from exc

    # The rendering contract is the code that renders and the config it reads.
    code_pins = {
        name: _sha256(files[name])
        for name in ("tokenizer_config.json", spec["tokenizer_code"], spec["chat_code"])
    }
    template_sha256 = _sha256(
        json.dumps(code_pins, sort_keys=True, separators=(",", ":")).encode()
    )
    if expected_template_sha256:
        from bench.sampler import normalize_sha256

        expected = normalize_sha256(expected_template_sha256)
        if template_sha256 != expected:
            raise SamplerError(
                f"chat template sha256 mismatch: expected {expected}, "
                f"got {template_sha256}"
            )
    # Match the model tokenizer's apply_chat_template default.
    kwargs = {"thinking_effort": "max"} if enable_thinking else {}

    def render(prompt: str | list[dict[str, str]]) -> str:
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        try:
            segments = build_segments(
                messages,
                None,
                add_generation_prompt=True,
                thinking=enable_thinking,
                **kwargs,
            )
            rendered = "".join(segment.text for segment in segments)
        except Exception as exc:
            raise PromptRenderError(
                f"chat encoder render failed for {repo}@{revision}: "
                f"{type(exc).__name__}"
            ) from exc
        if not isinstance(rendered, str) or not rendered:
            raise PromptRenderError(
                f"chat encoder for {repo}@{revision} rendered an empty prompt"
            )
        return rendered

    def encode(prompt: str) -> list[int]:
        # The engine's tokenizer parses special-token text and adds no BOS.
        if len(prompt) > _MAX_ENCODE_CHARS or _LONG_RUN_RE.search(prompt):
            raise PromptRenderError(
                "prompt exceeds the pinned tokenizer's single-window limits"
            )
        return encoding.encode(prompt, allowed_special="all")

    return PromptFormatter(
        render=render,
        receipt={
            "chat_template": {
                "model_repo": repo,
                "model_revision": revision,
                "sha256": template_sha256,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
                "kwargs": kwargs,
                "encoder": {"function": spec["chat_function"], "files": code_pins},
            },
            "tokenizer": {
                "model_repo": repo,
                "model_revision": revision,
                "sha256": _sha256(files[spec["vocab"]]),
                "library": "tiktoken",
                "library_version": tiktoken.__version__,
                "add_special_tokens": True,
            },
        },
        encode=encode,
    )
