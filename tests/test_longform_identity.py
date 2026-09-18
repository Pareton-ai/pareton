"""Qualification must bind its endpoint to the running baseline image. No Docker."""

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_longform_sampling import fields, formatter, response, row

from bench.lifecycle import EngineError
from bench.qualify_longform import qualify, verify_baseline_image

pytestmark = pytest.mark.unit
REF = "ghcr.io/pareton-ai/pareton-baseline@sha256:" + "a" * 64
IMAGE = "sha256:" + "b" * 64
URL = "http://127.0.0.1:8000"


@pytest.fixture
def docker(monkeypatch):
    state = {
        "container": {
            "Id": "c" * 64,
            "Image": IMAGE,
            "RestartCount": 0,
            "State": {"Running": True, "StartedAt": "2026-09-17T12:00:00Z"},
            "HostConfig": {"NetworkMode": "bridge"},
            "NetworkSettings": {
                "Ports": {"30000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}]}
            },
        },
        "image": {"Id": IMAGE, "RepoDigests": [REF]},
    }
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        assert cmd[:3] == ["docker", "--host", "unix:///var/run/docker.sock"]
        assert cmd[4:6] == ["inspect", "--"]
        if cmd[3] == "image":
            assert cmd[-1] == IMAGE
        return SimpleNamespace(stdout=json.dumps([state[cmd[3]]]))

    monkeypatch.setattr("bench.qualify_longform.subprocess.run", run)
    monkeypatch.setattr("bench.qualify_longform.getproxies", dict)
    return state, calls


def verify(**kw):
    return verify_baseline_image(
        **{"engine_ref": REF, "container": "baseline", "base_url": URL, **kw}
    )


def test_identity_uses_running_image_and_published_port(docker):
    state, calls = docker
    state["container"]["Config"] = {"Image": "mutable-tag"}
    assert verify() == {
        "engine_ref": REF,
        "image_id": IMAGE,
        "container_id": "c" * 64,
        "started_at": "2026-09-17T12:00:00Z",
        "restart_count": 0,
        "base_url": URL,
        "container_port": "30000/tcp",
    }
    assert len(calls) == 2


@pytest.mark.parametrize(
    "change",
    ["digest", "missing_digest", "image_id", "port", "ip", "stopped", "host_network"],
)
def test_unverified_identity_is_rejected(docker, change):
    state, _ = docker
    if change == "digest":
        state["image"]["RepoDigests"] = [REF.replace("a" * 64, "d" * 64)]
    elif change == "missing_digest":
        state["image"]["RepoDigests"] = []
    elif change == "image_id":
        state["image"]["Id"] = "sha256:" + "d" * 64
    elif change in ("port", "ip"):
        state["container"]["NetworkSettings"]["Ports"]["30000/tcp"][0][
            "HostPort" if change == "port" else "HostIp"
        ] = "9000" if change == "port" else "192.0.2.1"
    elif change == "stopped":
        state["container"]["State"]["Running"] = False
    else:
        state["container"]["HostConfig"]["NetworkMode"] = "host"
    with pytest.raises(EngineError):
        verify()


@pytest.mark.parametrize(
    "url",
    [
        "http://remote:8000",
        "http://localhost:8000",
        "https://127.0.0.1:8000",
        URL + "/proxy",
        URL + "?x=1",
    ],
)
def test_unbound_endpoint_rejected_before_inspection(docker, url):
    with pytest.raises(EngineError, match="direct"):
        verify(base_url=url)
    assert not docker[1]


def test_mutable_image_and_proxied_endpoint_are_rejected(docker, monkeypatch):
    with pytest.raises(EngineError, match="by digest"):
        verify(engine_ref="baseline:latest")
    monkeypatch.setattr(
        "bench.qualify_longform.getproxies", lambda: {"http": "http://proxy"}
    )
    monkeypatch.setattr("bench.qualify_longform.proxy_bypass", lambda host: False)
    with pytest.raises(EngineError, match="bypass"):
        verify()
    assert not docker[1]


def test_docker_inspection_failure_is_not_trusted(monkeypatch):
    monkeypatch.setattr("bench.qualify_longform.getproxies", dict)

    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1, a[0])

    monkeypatch.setattr("bench.qualify_longform.subprocess.run", fail)
    with pytest.raises(EngineError, match="cannot verify"):
        verify()


@pytest.mark.parametrize("change", [None, "wrong_image", "restart"])
def test_qualification_only_writes_rule_for_verified_unchanged_image(
    docker, monkeypatch, tmp_path, change
):
    state, calls = docker
    f = fields()
    original = copy.deepcopy(f)
    fmt = formatter()
    if change == "wrong_image":
        state["image"]["RepoDigests"] = []

    def post(url, path, body, **kw):
        if change == "restart":
            state["container"]["State"]["StartedAt"] = "2026-09-17T13:00:00Z"
        result = response()
        result["usage"]["prompt_tokens"] = len(fmt.encode(body["prompt"]))
        return result

    monkeypatch.setattr("bench.qualify_longform.post_json", post)
    monkeypatch.setattr(
        "bench.qualify_longform.validate_engine_workload", lambda *a, **k: None
    )
    kwargs = {
        "fields": f,
        "base_url": URL,
        "container": "baseline",
        "engine_ref": REF,
        "output_dir": tmp_path,
        "pool_size": 4,
        "max_rows": 40,
        "row_fetcher": row,
        "formatter": fmt,
    }
    if change:
        with pytest.raises(EngineError):
            qualify(**kwargs)
        assert not (tmp_path / "sampling_rule.json").exists()
    else:
        qualified = qualify(**kwargs)
        from bench.longform import require_qualification

        verified_bench = {**f["bench"], "baseline_engine_image_digest": REF}
        require_qualification(qualified, verified_bench, f["engine"])
        evidence = json.loads(
            (tmp_path / "qualification.jsonl").read_text().splitlines()[0]
        )
        assert evidence["bench"]["baseline_engine_image_digest"] == REF
        assert evidence["baseline_identity"]["image_id"] == IMAGE
        assert calls[-2][-1] == "c" * 64
    assert f == original


@pytest.mark.parametrize(
    "script,args",
    [
        ("ops/seed-sglang-qwen38-27b.sh", [REF, "0.15"]),
        ("ops/seed-sglang-qwen38-27b.sh", [REF, "0.15", "missing.json"]),
        ("ops/sglang-sample-round/run.sh", []),
        ("ops/sglang-sample-round/run.sh", ["output"]),
        ("ops/sglang-sample-round/run.sh", ["output", "missing.json"]),
    ],
)
def test_launchers_reject_missing_rules_before_side_effects(tmp_path, script, args):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / script), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert (
        "QUALIFIED_SAMPLING_RULE_JSON" in result.stderr
        or "readable file" in result.stderr
    )
    assert not list(tmp_path.iterdir())
