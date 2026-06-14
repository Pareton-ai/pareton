"""Unit tests for validator.eval_schema -- no GPU, no Docker, no bittensor."""

from __future__ import annotations

import pytest

from validator.eval_schema import (
    EVAL_JOB_FILE,
    ChallengerInfo,
    ChatMessage,
    EvalJob,
    PerPromptResult,
    Prompt,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# ChatMessage
# --------------------------------------------------------------------------- #


class TestChatMessage:
    def test_round_trip(self):
        msg = ChatMessage(role="user", content="Hello")
        assert ChatMessage.from_dict(msg.to_dict()) == msg


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


class TestPrompt:
    def test_round_trip(self):
        prompt = Prompt(
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Summarize this."),
            ],
            max_tokens=512,
        )
        restored = Prompt.from_dict(prompt.to_dict())
        assert restored == prompt


# --------------------------------------------------------------------------- #
# PerPromptResult
# --------------------------------------------------------------------------- #


class TestPerPromptResult:
    def test_round_trip(self):
        r = PerPromptResult(
            ttft_s=0.045,
            e2e_s=12.5,
            output_tokens=256,
            token_match_rate=0.998,
            baseline_e2e_s=13.2,
        )
        assert PerPromptResult.from_dict(r.to_dict()) == r

    def test_from_dict_defaults_baseline_e2e(self):
        data = {
            "ttft_s": 0.04,
            "e2e_s": 12.0,
            "output_tokens": 256,
            "token_match_rate": 0.998,
        }
        assert PerPromptResult.from_dict(data).baseline_e2e_s == 0.0


# --------------------------------------------------------------------------- #
# ChallengerInfo
# --------------------------------------------------------------------------- #


class TestChallengerInfo:
    def test_round_trip(self):
        ci = ChallengerInfo(
            uid=3,
            hotkey="5F3a",
            commit_block=4500100,
            image="foo/bar:v1",
            digest="sha256:" + "c" * 64,
        )
        restored = ChallengerInfo.from_dict(ci.to_dict())
        assert restored == ci

    def test_from_dict_coerces_types(self):
        ci = ChallengerInfo.from_dict(
            {
                "uid": "3",
                "hotkey": 5,
                "commit_block": "100",
                "image": "img",
                "digest": "d",
            }
        )
        assert ci.uid == 3
        assert ci.hotkey == "5"
        assert ci.commit_block == 100


# --------------------------------------------------------------------------- #
# EvalJob
# --------------------------------------------------------------------------- #


class TestEvalJob:
    def _make_job(self, n_challengers: int = 2) -> EvalJob:
        challengers = [
            ChallengerInfo(
                uid=i,
                hotkey=f"hk{i}",
                commit_block=100 + i,
                image=f"img{i}:v1",
                digest=f"sha256:{'a' * 64}",
            )
            for i in range(n_challengers)
        ]
        return EvalJob(
            block=4501234,
            block_hash="0xdeadbeef",
            challengers=challengers,
            created_at=1700000000.0,
        )

    def test_round_trip(self):
        job = self._make_job()
        restored = EvalJob.from_dict(job.to_dict())
        assert restored.block == job.block
        assert restored.block_hash == job.block_hash
        assert len(restored.challengers) == 2
        assert restored.challengers[0] == job.challengers[0]

    def test_save_and_load(self, tmp_path):
        job = self._make_job(3)
        job.save(tmp_path)
        assert (tmp_path / EVAL_JOB_FILE).exists()
        loaded = EvalJob.load(tmp_path)
        assert loaded is not None
        assert loaded.block == 4501234
        assert len(loaded.challengers) == 3

    def test_load_missing_returns_none(self, tmp_path):
        assert EvalJob.load(tmp_path) is None

    def test_load_corrupt_returns_none(self, tmp_path):
        (tmp_path / EVAL_JOB_FILE).write_text("{bad json")
        assert EvalJob.load(tmp_path) is None

    def test_empty_challengers(self):
        job = EvalJob(block=1, block_hash="0x0", challengers=[])
        restored = EvalJob.from_dict(job.to_dict())
        assert restored.challengers == []

    def test_default_created_at(self):
        job = EvalJob(block=1, block_hash="0x0", challengers=[])
        assert job.created_at == 0.0

    def test_leader_runner_up_defaults_none(self):
        job = EvalJob(block=1, block_hash="0x0", challengers=[])
        assert job.leader is None
        assert job.runner_up is None

    def test_leader_runner_up_round_trip(self):
        leader = ChallengerInfo(
            uid=10,
            hotkey="hk_leader",
            commit_block=50,
            image="leader:v1",
            digest="sha256:" + "l" * 64,
        )
        ru = ChallengerInfo(
            uid=20,
            hotkey="hk_ru",
            commit_block=60,
            image="ru:v1",
            digest="sha256:" + "r" * 64,
        )
        job = EvalJob(
            block=1000,
            block_hash="0xabc",
            challengers=[],
            leader=leader,
            runner_up=ru,
        )
        restored = EvalJob.from_dict(job.to_dict())
        assert restored.leader == leader
        assert restored.runner_up == ru

    def test_leader_none_runner_up_present(self):
        ru = ChallengerInfo(
            uid=20,
            hotkey="hk_ru",
            commit_block=60,
            image="ru:v1",
            digest="sha256:" + "r" * 64,
        )
        job = EvalJob(
            block=1000,
            block_hash="0xabc",
            challengers=[],
            runner_up=ru,
        )
        restored = EvalJob.from_dict(job.to_dict())
        assert restored.leader is None
        assert restored.runner_up == ru

    def test_backward_compat_no_leader_runner_up_keys(self):
        data = {
            "block": 1000,
            "block_hash": "0xabc",
            "challengers": [],
            "created_at": 0.0,
        }
        job = EvalJob.from_dict(data)
        assert job.leader is None
        assert job.runner_up is None
