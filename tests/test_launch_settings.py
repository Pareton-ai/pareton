"""Launch settings a candidate must leave as the baseline has them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bench.main as bm
from bench.launch_settings import (
    OKC_SERVING_DENIED_FLAGS,
    UNREADABLE,
    changed_settings,
    denied_reason,
    fixed_settings,
    stable_settings,
    validate_policy,
)
from bench.lifecycle import EngineError
from bench.main import EXIT_OK, main
from bench.schemas import BenchRequest
from tests.test_bench_cli import SAMPLE_REQUEST, _write_request
from worker.round_job import entry_results_from_report

pytestmark = pytest.mark.unit

POLICY = "okc_serving_v1"

# The shape of SGLang's /server_info: resolved arguments by flag name, some
# nested, next to state no flag sets.
SERVER_INFO = {
    "model_path": "/model",
    "dtype": "bfloat16",
    "context_length": 131072,
    "max_running_requests": 32,
    "cuda_graph_bs_decode": None,
    "cuda_graph_max_bs_decode": None,
    "cuda_graph_config": {"decode": {"bs": [1, 2, 4, 8], "backend": "full"}},
    "speculative_accept_threshold_single": 1.0,
    "enable_lora": None,
    "log_requests": False,
    "mem_fraction_static": 0.83,
    "max_total_num_tokens": 900000,
    "internal_states": [{"cuda_graph_bs_decode": [1, 2]}],
}


# -- The policy --------------------------------------------------------------


def test_policy_names_are_checked():
    assert validate_policy(None) is None
    assert validate_policy(POLICY) == POLICY
    for bad in ("lossless_v1", "", 1, ["okc_serving_v1"]):
        with pytest.raises(ValueError, match="fixed_launch_settings"):
            validate_policy(bad)


def test_request_carries_the_policy():
    raw = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    assert BenchRequest.from_dict(raw).fixed_launch_settings is None
    raw["fixed_launch_settings"] = POLICY
    assert BenchRequest.from_dict(raw).fixed_launch_settings == POLICY
    raw["fixed_launch_settings"] = "anything"
    with pytest.raises(ValueError, match="fixed_launch_settings"):
        BenchRequest.from_dict(raw)


def test_the_judge_set_flags_are_fixed():
    """The group OKC failed 628aaf15 under."""
    assert "--cuda-graph-bs-decode" in OKC_SERVING_DENIED_FLAGS["is set by the judge"]
    assert "--max-running-requests" in OKC_SERVING_DENIED_FLAGS["is set by the judge"]


# -- Which settings are fixed ------------------------------------------------


def test_only_settings_behind_denied_flags_are_kept():
    assert fixed_settings(SERVER_INFO, POLICY) == {
        "model_path": "/model",
        "dtype": "bfloat16",
        "context_length": 131072,
        "max_running_requests": 32,
        "cuda_graph_bs_decode": None,
        "cuda_graph_max_bs_decode": None,
        "speculative_accept_threshold_single": 1.0,
        "enable_lora": None,
        "log_requests": False,
    }


def test_a_resolved_setting_is_not_fixed():
    """OKC lets ``cuda_graph_config.decode.bs`` move as a new default."""
    kept = fixed_settings(SERVER_INFO, POLICY)
    assert not any(name.startswith("cuda_graph_config") for name in kept)
    # Whatever shape the server reports it in.
    for value in ("decode.bs=1", {}, None, {"decode": {"dtype": "x"}}):
        assert fixed_settings({"cuda_graph_config": value}, POLICY) == {}


def test_a_mapping_valued_setting_is_compared_whole():
    """A change under a denied flag's own mapping is a change of the flag."""
    info = dict(SERVER_INFO, preferred_sampling_params={"temperature": 0.7})
    baseline = fixed_settings(info, POLICY)
    assert baseline["preferred_sampling_params"] == {"temperature": 0.7}
    patched = dict(info, preferred_sampling_params={"temperature": 1.0})
    assert changed_settings(baseline, fixed_settings(patched, POLICY)) == {
        "preferred_sampling_params": {
            "baseline": "{'temperature': 0.7}",
            "candidate": "{'temperature': 1.0}",
        }
    }
    # A setting emptied out, or filled in from nothing, differs too.
    for other in ({}, None, {"temperature": 0.7, "top_p": 0.9}):
        patched = dict(info, preferred_sampling_params=other)
        assert list(changed_settings(baseline, fixed_settings(patched, POLICY))) == [
            "preferred_sampling_params"
        ]


def test_patterns_match_dotted_names_under_an_unfixed_mapping():
    """SGLang reports its arguments flat; a nested name matches as written."""
    assert fixed_settings({"serving": {"dtype": "bfloat16"}}, POLICY) == {}
    kept = fixed_settings({"serving": {"lora_paths": ["/a"], "x": 1}}, POLICY)
    assert kept == {"serving.lora_paths": ["/a"]}


# -- Comparison --------------------------------------------------------------


def test_a_changed_default_is_reported_with_both_values():
    baseline = fixed_settings(SERVER_INFO, POLICY)
    patched = dict(SERVER_INFO, cuda_graph_bs_decode=[1, 2, 4, 8, 16, 32])
    changed = changed_settings(baseline, fixed_settings(patched, POLICY))
    assert changed == {
        "cuda_graph_bs_decode": {
            "baseline": "None",
            "candidate": "[1, 2, 4, 8, 16, 32]",
        }
    }
    reason = denied_reason(changed)
    assert reason.startswith("launch_setting_denied: ")
    assert "cuda_graph_bs_decode: None -> [1, 2, 4, 8, 16, 32]" in reason


def test_unfixed_settings_may_change():
    baseline = fixed_settings(SERVER_INFO, POLICY)
    patched = dict(
        SERVER_INFO,
        mem_fraction_static=0.9,
        max_total_num_tokens=1,
        cuda_graph_config={"decode": {"bs": list(range(1, 33))}},
        enable_layernorm_sp=True,
    )
    assert changed_settings(baseline, fixed_settings(patched, POLICY)) == {}


def test_a_setting_added_or_dropped_counts_as_changed():
    baseline = fixed_settings(SERVER_INFO, POLICY)
    added = fixed_settings(dict(SERVER_INFO, lora_paths=["/x"]), POLICY)
    assert changed_settings(baseline, added) == {
        "lora_paths": {"baseline": "<absent>", "candidate": "['/x']"}
    }
    dropped = {k: v for k, v in baseline.items() if k != "dtype"}
    assert changed_settings(baseline, dropped) == {
        "dtype": {"baseline": "'bfloat16'", "candidate": "<absent>"}
    }


def test_a_setting_two_baselines_disagree_on_is_never_compared():
    first = {"dtype": "bfloat16", "port": 30000, "host": "0.0.0.0"}
    second = {"dtype": "bfloat16", "port": 30001}
    reference = stable_settings(first, second)
    assert reference["dtype"] == "bfloat16"
    candidate = {"dtype": "bfloat16", "port": 30002, "host": "::"}
    assert changed_settings(reference, candidate) == {}
    assert list(changed_settings(reference, {"dtype": "float16"})) == ["dtype"]
    # A third baseline keeps an unstable setting unstable.
    again = stable_settings(reference, first)
    assert changed_settings(again, candidate) == {}


def test_long_values_and_long_lists_are_cut_in_the_reason():
    changed = changed_settings(
        {f"kt_{i}": 0 for i in range(20)},
        {f"kt_{i}": "x" * 500 for i in range(20)},
    )
    assert all(len(v["candidate"]) <= 120 for v in changed.values())
    reason = denied_reason(changed)
    assert "and 12 more" in reason
    assert len(reason) < 2500


# -- Reading the settings ----------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        # A body shorter than its Content-Length: http.client.IncompleteRead.
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b'Content-Length: 100\r\n\r\n{"dty',
        # Closed without an answer: http.client.RemoteDisconnected.
        b"",
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n[]",
        b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n{x}",
    ],
)
def test_a_broken_server_info_answer_is_an_engine_error(response):
    """Every way the read fails must reach the round as an EngineError."""
    import socket
    import threading

    from bench.http import get_json

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(response)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.getsockname()[1]}"
        with pytest.raises(EngineError):
            get_json(url, "/server_info", timeout=10)
    finally:
        thread.join(5)
        server.close()


# -- The round ---------------------------------------------------------------


def _run(tmp_path: Path, candidates: list[dict], **overrides) -> tuple[int, dict]:
    req = _write_request(tmp_path, candidates=len(candidates), **overrides)
    out = tmp_path / "out"
    code = main(
        [
            "--request",
            str(req),
            "--output-dir",
            str(out),
            "--mock-engine",
            "--mock-baseline-token-latency-s",
            "0.004",
            "--mock-candidates",
            json.dumps(candidates),
        ]
    )
    report = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    statuses = json.loads((out / "entry_status.json").read_text(encoding="utf-8"))
    report["_statuses"] = statuses.get("entries", statuses)
    report["_log"] = [
        record
        for line in (out / "harness.log").read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
        and (record := json.loads(line)).get("event") == "launch_settings"
    ]
    return code, report


def test_round_disqualifies_a_candidate_that_changed_a_fixed_setting(tmp_path: Path):
    code, report = _run(
        tmp_path,
        [
            # 628aaf15: a judge-set flag filled in by the patch.
            {
                "speed_factor": 3.0,
                "server_info": {"cuda_graph_bs_decode": [1, 2, 4, 8, 16, 32]},
            },
            # A new default behind no denied flag, and a derived value.
            {
                "speed_factor": 1.5,
                "server_info": {
                    "enable_layernorm_sp": True,
                    "cuda_graph_config": {"decode": {"bs": [1, 2, 4, 8, 16, 32]}},
                },
            },
            {"speed_factor": 1.2},
        ],
        fixed_launch_settings=POLICY,
    )
    assert code == EXIT_OK
    statuses = report.pop("_statuses")
    from bench.validate import validate_report_dict

    validate_report_dict(report)
    denied, tuned, plain = report["entries"]
    assert denied["status"] == "disqualified"
    assert denied["score"] is None
    assert denied["reason"].startswith("launch_setting_denied")
    assert "cuda_graph_bs_decode: None -> [1, 2, 4, 8, 16, 32]" in denied["reason"]
    assert denied["launch_settings"] == {
        "cuda_graph_bs_decode": {
            "baseline": "None",
            "candidate": "[1, 2, 4, 8, 16, 32]",
        }
    }
    # Caught before it served anything, and not mistaken for a startup crash.
    assert "engine_crashed" not in denied
    assert "sla" not in denied
    assert statuses["0"]["status"] == "disqualified"
    assert tuned["status"] == "scored"
    assert "launch_settings" not in tuned
    assert plain["status"] == "scored"
    # The round's log says the check ran, and on what, for every engine.
    assert report.pop("_log") == [
        {"event": "launch_settings", "role": "baseline", "fixed": 3},
        {"event": "launch_settings", "role": "baseline-drift", "fixed": 3},
        {
            "event": "launch_settings",
            "role": "candidate-0",
            "fixed": 3,
            "changed": ["cuda_graph_bs_decode"],
        },
        {"event": "launch_settings", "role": "candidate-1", "fixed": 3, "changed": []},
        {"event": "launch_settings", "role": "candidate-2", "fixed": 3, "changed": []},
    ]


def test_round_without_the_policy_never_asks_for_settings(tmp_path: Path, monkeypatch):
    def no_read(*args, **kwargs):
        raise AssertionError("/server_info must not be read without the policy")

    monkeypatch.setattr(bm, "get_json", no_read)
    code, report = _run(tmp_path, [{"server_info": {"cuda_graph_bs_decode": [1, 2]}}])
    assert code == EXIT_OK
    assert report["entries"][0]["status"] == "scored"


def test_a_candidate_that_cannot_be_asked_its_settings_is_disqualified(
    tmp_path: Path, monkeypatch
):
    """The baseline answered, so a silent candidate is the patch's doing."""
    real = bm.get_json
    reads = []

    def get_json(url, path, **kwargs):
        reads.append(path)
        if len(reads) == 3:  # baseline, baseline-drift, then candidate 0
            raise EngineError("cannot read engine metadata")
        return real(url, path, **kwargs)

    monkeypatch.setattr(bm, "get_json", get_json)
    code, report = _run(
        tmp_path,
        [{"speed_factor": 2.0}, {"speed_factor": 1.5}],
        fixed_launch_settings=POLICY,
    )
    assert code == EXIT_OK
    silent, other = report["entries"]
    assert silent["status"] == "disqualified"
    assert silent["reason"].startswith("launch_setting_denied")
    assert list(silent["launch_settings"]) == [UNREADABLE]
    assert "engine_crashed" not in silent
    assert other["status"] == "scored"
    assert reads == ["/server_info"] * 4


def test_a_silent_candidate_is_disqualified_before_the_workload_preflight(
    tmp_path: Path, monkeypatch
):
    """Preflight reads /server_info too. Its failure is an infra failure, which
    would void the round for a leader, so the settings read has to come first."""
    import bench.workload_preflight as wp

    real = bm.get_json
    reads = []

    def get_json(url, path, **kwargs):
        reads.append(path)
        if len(reads) == 3:  # baseline, baseline-drift, then candidate 0
            raise EngineError("cannot read engine metadata")
        return real(url, path, **kwargs)

    def preflight(url, trace, *, evidence_dir, **kwargs):
        if evidence_dir.name == "candidate-0":
            raise EngineError("cannot read engine metadata")

    monkeypatch.setattr(bm, "get_json", get_json)
    monkeypatch.setattr(wp, "validate_engine_workload", preflight)
    code, report = _run(
        tmp_path,
        [{"speed_factor": 2.0}, {"speed_factor": 1.5}],
        fixed_launch_settings=POLICY,
        leader_candidate_index=0,
    )
    assert code == EXIT_OK
    silent, other = report["entries"]
    assert silent["status"] == "disqualified"
    assert list(silent["launch_settings"]) == [UNREADABLE]
    assert "engine_crashed" not in silent
    assert other["status"] == "scored"


def test_round_fails_when_the_baseline_cannot_be_asked(tmp_path: Path, monkeypatch):
    """No reference, no verdicts: the round fails as the harness's own."""

    def get_json(url, path, **kwargs):
        raise EngineError("cannot read engine metadata")

    monkeypatch.setattr(bm, "get_json", get_json)
    code, report = _run(tmp_path, [{"speed_factor": 2.0}], fixed_launch_settings=POLICY)
    assert code == bm.EXIT_ENGINE
    assert report["verdict"] != "pass"


def test_a_leader_with_a_changed_setting_stays_disqualified():
    """Only a startup crash maps an incumbent to infra; a rule breach does not."""
    entries = [
        {"id": 1, "role": "baseline"},
        {"id": 2, "role": "leader", "submission_id": "s"},
    ]
    changed = {"cuda_graph_bs_decode": {"baseline": "None", "candidate": "[1]"}}
    report = {
        "baseline": {"role": "baseline"},
        "entries": [
            {
                "index": 0,
                "status": "disqualified",
                "score": None,
                "reason": denied_reason(changed),
                "launch_settings": changed,
            }
        ],
    }
    leader = entry_results_from_report(entries, report)[1]
    assert leader["status"] == "disqualified"
    assert leader["disqualify_reason"].startswith("launch_setting_denied")
    assert leader["report"]["launch_settings"] == changed


def test_round_fails_when_the_baseline_reports_no_fixed_setting(
    tmp_path: Path, monkeypatch
):
    """A server info of another shape must not turn the check into a pass."""
    monkeypatch.setattr(bm, "get_json", lambda url, path, **kw: {"args": {}})
    code, report = _run(tmp_path, [{"speed_factor": 2.0}], fixed_launch_settings=POLICY)
    assert code == bm.EXIT_ENGINE
    assert "none of the launch settings" in json.dumps(report)
