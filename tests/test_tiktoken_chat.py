import base64
import json

import pytest

from bench import tiktoken_chat
from bench.sampler import PromptRenderError, SamplerError, build_prompt_formatter
from bench.tiktoken_chat import (
    build_tiktoken_chat_formatter,
    parse_tokenizer_constants,
)
from bench.weights import WeightsError, assert_complete

REPO = "moonshotai/Kimi-K3"
REVISION = "f831ab66814297da540d832a5235f8e904f29d06"

TOKENIZER_CODE = '''
import tiktoken


class TikTokenTokenizer:
    num_reserved_special_tokens = 4
    pat_str = "|".join(
        [
            r"""\\p{L}+""",
            r"""\\p{N}{1,3}""",
            r""" ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*""",
            r"""\\s+(?!\\S)""",
            r"""\\s+""",
        ]
    )
'''

CHAT_CODE = """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EncodeSegment:
    text: str
    allow_special: bool


def build_chat_segments(messages, tools=None, *, add_generation_prompt=True,
                        thinking=True, **kwargs: Any):
    out = []
    for message in messages:
        out.append(EncodeSegment("<|open|>", True))
        out.append(EncodeSegment("role=" + message["role"], False))
        out.append(EncodeSegment("<|sep|>", True))
        out.append(EncodeSegment(message["content"], False))
        out.append(EncodeSegment("<|end_of_msg|>", True))
    if add_generation_prompt:
        out.append(EncodeSegment("<|open|>", True))
        out.append(EncodeSegment("think" if thinking else "response", False))
        if thinking:
            out.append(EncodeSegment(kwargs["thinking_effort"], False))
    return out
"""

CONFIG = {
    "tokenizer_class": "TikTokenTokenizer",
    "added_tokens_decoder": {
        "256": {"content": "<|open|>"},
        "257": {"content": "<|sep|>"},
        "258": {"content": "<|end_of_msg|>"},
    },
}

VOCAB = b"".join(
    base64.b64encode(bytes([i])) + b" " + str(i).encode() + b"\n" for i in range(256)
)


def files(**overrides):
    data = {
        "tokenizer_config.json": json.dumps(CONFIG).encode(),
        "tokenization_kimi.py": TOKENIZER_CODE.encode(),
        "encoding_k3.py": CHAT_CODE.encode(),
        "tiktoken.model": VOCAB,
        **overrides,
    }
    return lambda *, repo_id, revision, filename, token: data[filename]


def formatter(enable_thinking=False, expected=None, **overrides):
    return build_tiktoken_chat_formatter(
        repo=REPO,
        revision=REVISION,
        enable_thinking=enable_thinking,
        expected_template_sha256=expected,
        token=None,
        file_loader=files(**overrides),
    )


def test_renders_with_pinned_encoder_and_encodes_special_tokens():
    fmt = formatter()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
    ]
    rendered = fmt.render(messages)
    assert rendered == (
        "<|open|>role=user<|sep|>hi<|end_of_msg|>"
        "<|open|>role=assistant<|sep|>ok<|end_of_msg|>"
        "<|open|>role=user<|sep|>more<|end_of_msg|>"
        "<|open|>response"
    )
    ids = fmt.encode(rendered)
    assert ids[0] == 256 and ids.count(258) == 3
    assert 0 not in ids  # no BOS or padding is added
    assert fmt.receipt["chat_template"]["enable_thinking"] is False
    assert fmt.receipt["chat_template"]["kwargs"] == {}
    assert fmt.receipt["tokenizer"]["library"] == "tiktoken"


def test_thinking_matches_apply_chat_template_default_effort():
    fmt = formatter(enable_thinking=True)
    assert fmt.render("hi").endswith("<|open|>thinkmax")
    assert fmt.receipt["chat_template"]["kwargs"] == {"thinking_effort": "max"}


def test_template_pin_covers_rendering_code():
    pin = formatter().receipt["chat_template"]["sha256"]
    assert formatter(expected=pin).receipt["chat_template"]["sha256"] == pin
    changed = CHAT_CODE.replace("role=", "role:").encode()
    with pytest.raises(SamplerError, match="sha256 mismatch"):
        formatter(expected=pin, **{"encoding_k3.py": changed})
    # The vocabulary is pinned separately by the tokenizer receipt.
    assert formatter().receipt["tokenizer"]["sha256"].startswith("sha256:")


def test_rejects_non_stdlib_chat_encoder():
    code = "import os\n" + CHAT_CODE
    with pytest.raises(SamplerError, match="beyond stdlib"):
        formatter(**{"encoding_k3.py": code.encode()})


def test_refuses_inputs_the_model_tokenizer_would_window():
    fmt = formatter()
    with pytest.raises(PromptRenderError):
        fmt.encode("a" * 25_001)


def test_parse_tokenizer_constants_requires_literal_pattern():
    pat, reserved = parse_tokenizer_constants(TOKENIZER_CODE)
    assert pat.startswith(r"\p{L}+|") and reserved == 4
    with pytest.raises(SamplerError):
        parse_tokenizer_constants("pat_str = make()\nnum_reserved_special_tokens = 4\n")


def test_build_prompt_formatter_routes_pinned_kimi(monkeypatch):
    monkeypatch.setattr(tiktoken_chat, "_call_load_file", files())
    rule = {
        "algo_version": 5,
        "type": "hf_rows",
        "dataset": "zai-org/LongWriter-6k",
        "revision": "0" * 40,
        "config": "default",
        "split": "train",
        "n_rows": 6000,
        "n_prompts": 16,
        "max_tokens": 5120,
        "enable_thinking": False,
        "min_output_tokens": 3000,
        "followup_prompt": "Write.",
        "temperature_range": [0.1, 1.01],
        "request_concurrency": 4,
        "request_timeout_s": 600,
        "input_tiers": ["8k", "16k"],
        "max_baseline_prompt_drops": 4,
    }
    fmt = build_prompt_formatter(rule, model_repo=REPO, model_revision=REVISION)
    assert fmt.receipt["tokenizer"]["library"] == "tiktoken"


def test_staged_weights_accept_tiktoken_vocabulary(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x")
    with pytest.raises(WeightsError, match="tokenizer"):
        assert_complete(tmp_path)
    (tmp_path / "tiktoken.model").write_bytes(VOCAB)
    assert_complete(tmp_path)
