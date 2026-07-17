"""Shared HTTP completions clients: streaming usage + usage-only chunks."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from bench.http import post_completion_stream
from bench.lifecycle import EngineError


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._buf = io.BytesIO(body)

    def __iter__(self):
        return self._buf.__iter__()

    def close(self) -> None:
        return None


def _sse(*chunks: dict[str, Any], done: bool = True) -> bytes:
    parts = [f"data: {json.dumps(c)}\n\n".encode("utf-8") for c in chunks]
    if done:
        parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def test_stream_requests_include_usage(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=60):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        body = _sse(
            {
                "id": "c1",
                "object": "text_completion",
                "choices": [
                    {
                        "index": 0,
                        "text": "hi",
                        "finish_reason": "length",
                        "logprobs": None,
                    }
                ],
            },
            {
                "id": "c1",
                "object": "text_completion",
                "choices": [],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )
        return _FakeResp(body)

    monkeypatch.setattr("bench.http.urlopen", fake_urlopen)
    res = post_completion_stream("http://example", prompt="p", max_tokens=1)
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert res.completion_tokens == 1
    assert res.text == "hi"
    assert res.finish_reason == "length"
    assert len(res.itl_s) == 0  # usage-only chunk must not add an ITL sample


def test_stream_usage_only_chunk_without_choices(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(req, timeout=60):
        body = _sse(
            {
                "choices": [
                    {"index": 0, "text": "a", "finish_reason": None, "logprobs": None}
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "text": "b",
                        "finish_reason": "length",
                        "logprobs": None,
                    }
                ],
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                    "total_tokens": 4,
                },
            },
        )
        return _FakeResp(body)

    monkeypatch.setattr("bench.http.urlopen", fake_urlopen)
    res = post_completion_stream("http://example", prompt="p", max_tokens=2)
    assert res.completion_tokens == 2
    assert res.text == "ab"
    assert len(res.itl_s) == 1


def test_stream_missing_done_is_engine_error(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(req, timeout=60):
        return _FakeResp(
            _sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "x",
                            "finish_reason": "length",
                            "logprobs": None,
                        }
                    ],
                },
                done=False,
            )
        )

    monkeypatch.setattr("bench.http.urlopen", fake_urlopen)
    with pytest.raises(EngineError, match="without \\[DONE\\]"):
        post_completion_stream("http://example", prompt="p", max_tokens=1)


def test_stream_done_only_is_engine_error(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(req, timeout=60):
        return _FakeResp(_sse(done=True))

    monkeypatch.setattr("bench.http.urlopen", fake_urlopen)
    with pytest.raises(EngineError, match="no choice chunks"):
        post_completion_stream("http://example", prompt="p", max_tokens=1)


def test_stream_coalesced_multi_token_is_engine_error(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(req, timeout=60):
        body = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": "hello",
                        "finish_reason": "length",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 5,
                    "total_tokens": 6,
                },
            },
        )
        return _FakeResp(body)

    monkeypatch.setattr("bench.http.urlopen", fake_urlopen)
    with pytest.raises(EngineError, match="no inter-token gaps"):
        post_completion_stream("http://example", prompt="p", max_tokens=5)
