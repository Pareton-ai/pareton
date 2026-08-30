"""Round creation policy: dedupe, the fire trigger, and the hf_rows path."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

import config
from bench.sampler import PromptFormatter
from round.create import dedupe_by_digest, should_create, try_create_round

pytestmark = pytest.mark.unit

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _row(sid: str, ref: str, block: int) -> dict:
    return {"id": sid, "engine_image_ref": ref, "commit_block": block}


def test_dedupe_keeps_the_earliest_commit_block():
    cohort = [
        _row("s1", f"ghcr.io/o/i@{DIGEST_A}", 10),
        _row("s2", f"ghcr.io/other/img@{DIGEST_A.upper()}", 11),
        _row("s3", f"ghcr.io/o/i@{DIGEST_B}", 12),
    ]
    keepers, duplicates = dedupe_by_digest(cohort)

    assert [r["id"] for r in keepers] == ["s1", "s3"]
    assert duplicates == [{"id": "s2", "kept_id": "s1"}]


def test_dedupe_keeps_a_digestless_ref_on_its_own():
    """A tag-pinned ref cannot be compared by digest, and must not fail the round."""
    cohort = [
        _row("s1", "ghcr.io/o/i:deadbeef", 10),
        _row("s2", "ghcr.io/o/i:cafe", 11),
    ]
    keepers, duplicates = dedupe_by_digest(cohort)

    assert [r["id"] for r in keepers] == ["s1", "s2"]
    assert duplicates == []


@pytest.mark.parametrize(
    "queued,waited_s,expected",
    [
        (0, 999_999, False),
        (5, 0, True),
        (6, 0, True),
        (1, 0, False),
        (1, 21_599, False),
        (1, 21_600, True),
        (4, 30_000, True),
    ],
)
def test_should_create_trigger_matrix(queued, waited_s, expected):
    oldest = NOW - timedelta(seconds=waited_s)
    assert should_create(queued, oldest, size=5, max_wait_s=21_600, now=NOW) is expected


def test_should_create_needs_a_queue_timestamp_below_size():
    assert should_create(2, None, size=5, max_wait_s=1, now=NOW) is False


def _campaign(**over):
    fields = {
        "campaign_id": uuid4(),
        "bench": {"baseline_engine_image_digest": DIGEST_A},
        "gpu_skus": ["H200", "B200"],
        "sampling_rule": None,
        "workload_trace_sha256": "sha256:" + "e" * 64,
        "workload_trace_url": "https://cdn.test/trace.json",
        "scoring_rule": {"name": "median_e2e_speedup"},
    }
    fields.update(over)
    return SimpleNamespace(**fields)


def _capture(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        "round.create.create_round",
        lambda **kw: calls.append(kw) or {"round_id": "r1", "ordinal": 1},
    )
    return calls


def test_no_sampling_rule_skips_the_round(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(config, "ROUND_SIZE", 5)

    out = try_create_round(
        _campaign(),
        {"queued": 5, "oldest_queued_at": NOW},
        seed_block=1000,
        seed_block_hash="0x" + "ab" * 32,
    )

    assert out is None
    assert calls == []


def test_no_round_before_the_queue_earns_one(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(config, "ROUND_SIZE", 5)
    monkeypatch.setattr(config, "ROUND_MAX_WAIT_S", 21_600)

    out = try_create_round(
        _campaign(),
        {"queued": 1, "oldest_queued_at": datetime.now(timezone.utc)},
        seed_block=1000,
        seed_block_hash="ab" * 32,
    )

    assert out is None
    assert calls == []


def test_no_round_without_a_baseline_engine_pin(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(config, "ROUND_SIZE", 1)

    out = try_create_round(
        _campaign(bench={}),
        {"queued": 5, "oldest_queued_at": NOW},
        seed_block=1000,
        seed_block_hash="ab" * 32,
    )

    assert out is None
    assert calls == []


def test_minimal_hf_rule_fetches_with_parsed_defaults(monkeypatch):
    """fetch_hf_row indexes config/split; a minimal rule must be parsed first."""
    calls = _capture(monkeypatch)
    monkeypatch.setattr(config, "ROUND_SIZE", 1)
    rule = {
        "type": "hf_rows",
        "dataset": "d",
        "revision": "r",
        "n_rows": 5,
        "n_prompts": 2,
    }

    def fake_fetch(rule_arg, idx):
        # Mirror _cached_hf_split: these keys must exist after parsing.
        assert rule_arg["config"] == "default"
        assert rule_arg["split"] == "train"
        return {"trajectory": [{"role": "user", "content": "hello"}]}

    monkeypatch.setattr("round.create.fetch_hf_row", fake_fetch)

    out = try_create_round(
        _campaign(sampling_rule=rule),
        {"queued": 5, "oldest_queued_at": NOW},
        seed_block=1000,
        seed_block_hash="ab" * 32,
        prompt_formatter=PromptFormatter(render=lambda prompt: prompt, receipt={}),
    )

    assert out == {"round_id": "r1", "ordinal": 1}
    (kw,) = calls
    assert kw["sampling_receipt"]["type"] == "hf_rows"
    assert kw["sampling_receipt"]["config"] == "default"
    assert kw["sampling_receipt"]["split"] == "train"


def test_round_creation_hashes_chat_formatted_prompts(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(config, "ROUND_SIZE", 1)
    rule = {
        "type": "hf_rows",
        "dataset": "d",
        "revision": "r",
        "n_rows": 2,
        "n_prompts": 1,
        "algo_version": 1,
    }
    formatter = PromptFormatter(
        render=lambda prompt: f"<chat>{prompt}</chat><assistant>",
        receipt={
            "chat_template": {
                "model_repo": "org/model",
                "model_revision": "a" * 40,
                "sha256": "sha256:" + "b" * 64,
                "add_generation_prompt": True,
            },
        },
    )
    formatter_calls = []

    def fake_build_formatter(rule_arg, **kwargs):
        formatter_calls.append((rule_arg, kwargs))
        return formatter

    monkeypatch.setattr("round.create.build_prompt_formatter", fake_build_formatter)

    out = try_create_round(
        _campaign(
            sampling_rule=rule,
            bench={
                "baseline_engine_image_digest": DIGEST_A,
                "model": {
                    "hf_repo": "org/model",
                    "hf_revision": "a" * 40,
                },
            },
        ),
        {"queued": 5, "oldest_queued_at": NOW},
        seed_block=1000,
        seed_block_hash="ab" * 32,
        row_fetcher=lambda _idx: {
            "trajectory": [{"role": "user", "content": "issue"}]
        },
    )

    assert out == {"round_id": "r1", "ordinal": 1}
    (kw,) = calls
    assert kw["sampling_receipt"]["chat_template"]["add_generation_prompt"] is True
    assert formatter_calls[0][1] == {
        "model_repo": "org/model",
        "model_revision": "a" * 40,
    }
