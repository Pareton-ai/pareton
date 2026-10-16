"""Restamp a qualified Kimi K3 C16 rule for the C32 fixture without GPU work."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from bench.longform import qualification_contract, require_qualification
from bench.sampler import SamplerError, parse_sampling_rule

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "restamp_k3_c32", ROOT / "ops/restamp_k3_c32.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
FIELDS = json.loads(helper.FIELDS.read_text())
POOL = list(range(0, 3200, 100))


def qualified_c16(**changes):
    rule = {**FIELDS["sampling_rule"], **helper.SOURCE_PINS, **changes}
    bench = copy.deepcopy(FIELDS["bench"])
    bench["correctness"]["num_prompts"] = helper.SOURCE_CORRECTNESS_PROMPTS
    rule = parse_sampling_rule({**rule, "eligible_row_indices": POOL})
    rule["qualification"] = {
        "contract_sha256": qualification_contract(rule, bench, FIELDS["engine"]),
        "repetitions": 2,
    }
    return rule


def test_fixture_is_the_c32_contract():
    rule = FIELDS["sampling_rule"]
    fixture = json.loads((helper.FIELDS.parent / "sampling_rule.json").read_text())
    assert rule == fixture
    assert (rule["n_prompts"], rule["request_concurrency"]) == (32, 32)
    assert rule["max_baseline_prompt_drops"] == 8
    assert rule["request_timeout_s"] == 1200
    assert FIELDS["bench"]["correctness"]["num_prompts"] == 32


def test_restamp_keeps_pool_and_qualifies_for_fixture():
    new = helper.restamp(qualified_c16(), FIELDS)
    require_qualification(new, FIELDS["bench"], FIELDS["engine"])
    assert new["eligible_row_indices"] == POOL
    assert new["qualification"]["repetitions"] == 2
    assert {
        k: v
        for k, v in new.items()
        if k not in ("eligible_row_indices", "qualification")
    } == FIELDS["sampling_rule"]


@pytest.mark.parametrize(
    "rule",
    [
        qualified_c16(min_output_tokens=1000),
        qualified_c16(max_tokens=4096),
        {
            **qualified_c16(),
            "qualification": {
                "contract_sha256": "sha256:" + "0" * 64,
                "repetitions": 2,
            },
        },
        {k: v for k, v in qualified_c16().items() if k != "qualification"},
    ],
)
def test_restamp_refuses_other_changes_or_stale_hash(rule):
    with pytest.raises(SamplerError):
        helper.restamp(rule, FIELDS)


def test_restamp_refuses_a_pool_smaller_than_the_round():
    rule = qualified_c16()
    rule = parse_sampling_rule(
        {k: v for k, v in rule.items() if k != "qualification"}
        | {"eligible_row_indices": POOL[:28]}
    )
    bench = copy.deepcopy(FIELDS["bench"])
    bench["correctness"]["num_prompts"] = 16
    rule["qualification"] = {
        "contract_sha256": qualification_contract(rule, bench, FIELDS["engine"]),
        "repetitions": 2,
    }
    with pytest.raises(SamplerError, match="cannot fill"):
        helper.restamp(rule, FIELDS)


class FakeCursor:
    def __init__(self, row, active_rounds=0):
        self.row, self.active, self.queries = row, active_rounds, []

    def execute(self, sql, args):
        self.queries.append(sql)

    def fetchone(self):
        if "FROM rounds" in self.queries[-1]:
            return {"n": self.active}
        return self.row


def campaign_row(**changes):
    bench = copy.deepcopy(FIELDS["bench"])
    bench["correctness"]["num_prompts"] = 16
    return {
        "id": "cid",
        "status": "open",
        "bench": bench,
        "engine": FIELDS["engine"],
        "sampling_rule": qualified_c16(),
        "manifest_hash": "sha256:old",
        **changes,
    }


def test_campaign_row_is_restamped_and_rehashed():
    cur = FakeCursor(campaign_row())
    old, row = helper.restamp_campaign_row(cur, "cid", FIELDS)
    assert "FOR UPDATE" in cur.queries[0]
    assert old["manifest_hash"] == "sha256:old" and row["manifest_hash"] is None
    assert row["bench"] == FIELDS["bench"]
    require_qualification(row["sampling_rule"], row["bench"], FIELDS["engine"])
    assert row["sampling_rule"]["eligible_row_indices"] == POOL


@pytest.mark.parametrize(
    "row, active",
    [
        (campaign_row(status="closed"), 0),
        (campaign_row(), 1),
        (campaign_row(bench=FIELDS["bench"]), 0),
        (campaign_row(sampling_rule=qualified_c16(min_output_tokens=1000)), 0),
        (None, 0),
    ],
)
def test_campaign_row_refuses_unsafe_states(row, active):
    with pytest.raises(SamplerError):
        helper.restamp_campaign_row(FakeCursor(row, active), "cid", FIELDS)
