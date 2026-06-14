"""Unit tests for validator.content_fingerprint -- mocked subprocess only."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from validator.content_fingerprint import (
    FINGERPRINT_FILE_NAME,
    FingerprintCheckResult,
    apply_fingerprint_supersede_records,
    check_and_register_fingerprint,
    check_duplicate,
    composite_fingerprint,
    compute_content_fingerprint,
    duplicate_submission_reason,
    find_registry_supersession,
    load_fingerprint_registry,
    register_fingerprint,
    retro_dq_record,
    save_fingerprint_registry,
)
from validator.chain import CommitmentRecord
from validator.state import DUPLICATE_SUBMISSION_REASON, EvaluationRecord

pytestmark = pytest.mark.unit


def _make_commitment(**overrides) -> CommitmentRecord:
    defaults = dict(
        uid=1,
        hotkey="hk_owner",
        coldkey="ck",
        commit_block=100,
        image="user/miner:v1",
        digest="sha256:" + "a" * 64,
        raw="{}",
    )
    defaults.update(overrides)
    return CommitmentRecord(**defaults)


def _make_eval_record(**overrides) -> EvaluationRecord:
    defaults = dict(
        uid=5,
        hotkey="hk_copy",
        commit_block=200,
        image="user/copy:v1",
        digest="sha256:" + "b" * 64,
        score=0.42,
        speed_improvement=0.42,
        token_match_rate=1.0,
        disqualified=False,
        disqualify_reason=None,
        evaluated_at=1.0,
        evaluation_block=500,
    )
    defaults.update(overrides)
    return EvaluationRecord(**defaults)


class TestCompositeFingerprint:
    def test_empty_files_sentinel(self):
        assert composite_fingerprint({}) == composite_fingerprint({})

    def test_order_independent(self):
        files = {"/start.sh": "aa" * 32, "/cacheon/foo.bin": "bb" * 32}
        assert composite_fingerprint(files) == composite_fingerprint(
            dict(reversed(files.items()))
        )

    def test_different_content_different_fp(self):
        a = composite_fingerprint({"/start.sh": "a" * 64})
        b = composite_fingerprint({"/start.sh": "b" * 64})
        assert a != b


class TestRegistryRules:
    def _registry_with_owner(self, fp: str, **owner) -> dict:
        defaults = dict(
            hotkey="hk_owner",
            commit_block=100,
            uid=1,
            image="user/a:v1",
            digest="sha256:" + "a" * 64,
            registered_at=1.0,
        )
        defaults.update(owner)
        return {"entries": {fp: defaults}}

    def test_no_entry_not_duplicate(self):
        assert check_duplicate({"entries": {}}, "fp1", "hk2", 200) is None

    def test_same_hotkey_allowed(self):
        reg = self._registry_with_owner("fp1", hotkey="hk1", commit_block=100)
        assert check_duplicate(reg, "fp1", "hk1", 200) is None

    def test_later_commit_different_hotkey_is_duplicate(self):
        reg = self._registry_with_owner("fp1", hotkey="hk_owner", commit_block=100)
        owner = check_duplicate(reg, "fp1", "hk_copy", 200)
        assert owner is not None
        assert owner["uid"] == 1

    def test_earlier_commit_not_duplicate_vs_registry(self):
        reg = self._registry_with_owner("fp1", hotkey="hk_late", commit_block=200)
        assert check_duplicate(reg, "fp1", "hk_early", 100) is None

    def test_find_registry_supersession_when_earlier(self):
        reg = self._registry_with_owner("fp1", hotkey="hk_late", commit_block=200)
        entry = find_registry_supersession(reg, "fp1", "hk_early", 100)
        assert entry is not None
        assert entry["commit_block"] == 200

    def test_register_replaces_later_owner_with_earlier(self):
        reg = self._registry_with_owner(
            "fp1", hotkey="hk_late", commit_block=200, uid=9
        )
        register_fingerprint(
            reg,
            "fp1",
            hotkey="hk_early",
            commit_block=100,
            uid=2,
            image="user/b:v1",
            digest="sha256:" + "b" * 64,
        )
        assert reg["entries"]["fp1"]["hotkey"] == "hk_early"
        assert reg["entries"]["fp1"]["uid"] == 2

    def test_duplicate_submission_reason_format(self):
        reason = duplicate_submission_reason(
            {"hotkey": "5GrwvaEF5zXb26Fz9rcQp", "commit_block": 12345}
        )
        assert reason.startswith(f"{DUPLICATE_SUBMISSION_REASON}:")
        assert "5GrwvaEF5zXb26Fz9rcQp" in reason
        assert "12345" in reason


class TestRegistryPersistence:
    def test_load_missing_returns_empty(self, tmp_path):
        assert load_fingerprint_registry(tmp_path) == {"entries": {}}

    def test_round_trip(self, tmp_path):
        reg = {"entries": {"abc": {"hotkey": "hk", "commit_block": 1, "uid": 0}}}
        save_fingerprint_registry(tmp_path, reg)
        path = tmp_path / FINGERPRINT_FILE_NAME
        assert path.exists()
        loaded = load_fingerprint_registry(tmp_path)
        assert loaded["entries"]["abc"]["hotkey"] == "hk"


class TestComputeContentFingerprint:
    @patch("validator.content_fingerprint._hash_file")
    @patch("validator.content_fingerprint._file_size_bytes")
    @patch("validator.content_fingerprint._list_fingerprint_files")
    def test_hashes_listed_files(self, mock_list, mock_size, mock_hash):
        mock_list.return_value = ["/start.sh", "/draft/model.safetensors"]
        mock_size.return_value = 1024
        mock_hash.side_effect = ["a" * 64, "b" * 64]
        fp = compute_content_fingerprint("cid123")
        assert fp == composite_fingerprint(
            {"/start.sh": "a" * 64, "/draft/model.safetensors": "b" * 64}
        )

    @patch("validator.content_fingerprint._list_fingerprint_files")
    def test_infra_failure_returns_none(self, mock_list):
        mock_list.return_value = None
        assert compute_content_fingerprint("cid123") is None


class TestCheckAndRegister:
    @patch("validator.content_fingerprint.compute_content_fingerprint")
    def test_duplicate_returns_reason_without_register(self, mock_compute, tmp_path):
        fp = composite_fingerprint({"/start.sh": "a" * 64})
        mock_compute.return_value = fp
        registry = {
            "entries": {
                fp: {
                    "hotkey": "hk_owner",
                    "commit_block": 100,
                    "uid": 5,
                    "image": "user/a:v1",
                    "digest": "sha256:" + "a" * 64,
                    "registered_at": 1.0,
                }
            }
        }
        com = _make_commitment(hotkey="hk_copy", commit_block=200, uid=9)
        result = check_and_register_fingerprint(
            registry, container_id="cid", com=com, state_dir=tmp_path
        )
        assert result.dq_reason is not None
        assert DUPLICATE_SUBMISSION_REASON in result.dq_reason
        assert len(registry["entries"]) == 1

    @patch("validator.content_fingerprint.compute_content_fingerprint")
    def test_success_registers_and_saves(self, mock_compute, tmp_path):
        fp = composite_fingerprint({"/start.sh": "c" * 64})
        mock_compute.return_value = fp
        registry: dict = {"entries": {}}
        com = _make_commitment()
        result = check_and_register_fingerprint(
            registry, container_id="cid", com=com, state_dir=tmp_path
        )
        assert result.dq_reason is None
        assert result.superseded_owner is None
        assert fp in registry["entries"]
        assert (tmp_path / FINGERPRINT_FILE_NAME).exists()

    @patch("validator.content_fingerprint.compute_content_fingerprint")
    def test_earlier_commit_supersedes_later_registry_owner(
        self, mock_compute, tmp_path
    ):
        fp = composite_fingerprint({"/start.sh": "a" * 64})
        mock_compute.return_value = fp
        registry = {
            "entries": {
                fp: {
                    "hotkey": "hk_copy",
                    "commit_block": 200,
                    "uid": 5,
                    "image": "user/copy:v1",
                    "digest": "sha256:" + "b" * 64,
                    "registered_at": 1.0,
                }
            }
        }
        com = _make_commitment(hotkey="hk_owner", commit_block=100, uid=1)
        result = check_and_register_fingerprint(
            registry, container_id="cid", com=com, state_dir=tmp_path
        )
        assert result.dq_reason is None
        assert result.superseded_owner is not None
        assert result.superseded_owner["uid"] == 5
        assert registry["entries"][fp]["commit_block"] == 100


class TestRetroDq:
    def test_retro_dq_record_zeros_score(self):
        record = _make_eval_record()
        canonical = {"uid": 1, "commit_block": 100}
        dq = retro_dq_record(record, canonical)
        assert dq.disqualified is True
        assert dq.score == 0.0
        assert DUPLICATE_SUBMISSION_REASON in (dq.disqualify_reason or "")


class TestApplySupersede:
    def test_retro_dqs_matching_leader(self):
        fp = composite_fingerprint({"/start.sh": "a" * 64})
        leader = _make_eval_record(uid=5, hotkey="hk_copy", commit_block=200)
        canonical_com = _make_commitment(hotkey="hk_owner", commit_block=100, uid=1)
        result = FingerprintCheckResult(
            fingerprint=fp,
            superseded_owner={
                "hotkey": "hk_copy",
                "commit_block": 200,
                "uid": 5,
            },
        )
        new_leader, new_ru, challengers, removed = apply_fingerprint_supersede_records(
            result,
            canonical_com,
            leader_record=leader,
            ru_record=None,
            challenger_records=[],
        )
        assert new_leader is not None
        assert new_leader.disqualified is True
        assert new_ru is None
        assert challengers == []
        assert removed == [leader.eval_key]

    def test_retro_dqs_leader_when_runner_up_supersedes(self):
        leader = _make_eval_record(uid=5, hotkey="hk_copy", commit_block=200)
        ru = _make_eval_record(uid=2, hotkey="hk_owner", commit_block=100)
        canonical_com = _make_commitment(hotkey="hk_owner", commit_block=100, uid=2)
        result = FingerprintCheckResult(
            fingerprint="fp",
            superseded_owner={
                "hotkey": "hk_copy",
                "commit_block": 200,
                "uid": 5,
            },
        )
        new_leader, new_ru, _, removed = apply_fingerprint_supersede_records(
            result,
            canonical_com,
            leader_record=leader,
            ru_record=ru,
            challenger_records=[],
        )
        assert new_leader is not None
        assert new_leader.disqualified is True
        assert new_ru is not None
        assert new_ru.disqualified is False
        assert removed == [leader.eval_key]

    def test_leaves_unrelated_leader(self):
        leader = _make_eval_record(uid=9, hotkey="hk_other", commit_block=50)
        canonical_com = _make_commitment(hotkey="hk_owner", commit_block=100, uid=1)
        result = FingerprintCheckResult(
            fingerprint="fp",
            superseded_owner={
                "hotkey": "hk_copy",
                "commit_block": 200,
                "uid": 5,
            },
        )
        new_leader, _, _, removed = apply_fingerprint_supersede_records(
            result,
            canonical_com,
            leader_record=leader,
            ru_record=None,
            challenger_records=[],
        )
        assert removed == []
        assert new_leader is not None
        assert new_leader.disqualified is False
