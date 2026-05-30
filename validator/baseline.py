"""Baseline measurement types for containerized evaluation.

The vLLM baseline is run once per GPU eval round; results are held in memory
so every challenger is compared against identical baseline numbers.

No Docker, no HTTP -- datatypes and cache-key derivation for log labels only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass
class BaselinePromptResult:
    """Baseline measurements for a single prompt."""

    tokens: list[str]
    top_logprobs: list[list[dict[str, Any]]]
    ttft_s: float
    throughput_tps: float
    output_tokens: int
    decode_elapsed_secs: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tokens": self.tokens,
            "top_logprobs": self.top_logprobs,
            "ttft_s": self.ttft_s,
            "throughput_tps": self.throughput_tps,
            "output_tokens": self.output_tokens,
        }
        if self.decode_elapsed_secs is not None:
            out["decode_elapsed_secs"] = self.decode_elapsed_secs
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselinePromptResult:
        raw_elapsed = data.get("decode_elapsed_secs")
        decode_elapsed_secs = list(raw_elapsed) if raw_elapsed is not None else []
        return cls(
            tokens=list(data["tokens"]),
            top_logprobs=list(data["top_logprobs"]),
            ttft_s=float(data["ttft_s"]),
            throughput_tps=float(data["throughput_tps"]),
            output_tokens=int(data["output_tokens"]),
            decode_elapsed_secs=decode_elapsed_secs,
        )


@dataclass
class BaselineCache:
    """Pass 1 baseline run for a prompt set (in-memory only)."""

    cache_key: str
    results: list[BaselinePromptResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineCache:
        return cls(
            cache_key=str(data["cache_key"]),
            results=[BaselinePromptResult.from_dict(r) for r in data["results"]],
        )


def derive_cache_key(
    block_hash: str,
    baseline_digest: str = "",
    regime: str = "stress",
) -> str:
    """Short id for a baseline run (container log filenames).

    Derived from block_hash, baseline image digest, prompt engine version, and
    regime so logs from different configs do not collide.
    """
    from .prompts import PROMPT_ENGINE_VERSION

    raw = f"{block_hash}:{baseline_digest}:v{PROMPT_ENGINE_VERSION}:{regime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
