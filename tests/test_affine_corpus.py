"""Affine public corpus (schema 3, duel_turns@v4): pins, parsing, sampling, replay."""

import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import bench.affine_corpus as affine
from bench.affine_corpus import (
    AffineCorpus,
    CorpusIntegrityError,
    require_qualification,
    sampling_context_for_campaign,
    turn_messages,
    write_gzip_jsonl,
)
from bench.sampler import (
    SamplerError,
    build_prompt_formatter,
    generate_trace,
    parse_sampling_rule,
)
from bench.validate import RequestValidationError, validate_workload_trace_dict
from round.create import try_create_round
from worker.round_job import RoundInfraError, materialize_round_trace

pytestmark = pytest.mark.unit

BASE = "https://corpus.example"
VIEW = "duel_turns@v4"
# History sizes in tokens under the test formatter, one per input tier.
TIER_TOKENS = (1000, 3000, 6000, 12000)


# --------------------------------------------------------------------------
# Synthetic corpus
# --------------------------------------------------------------------------


def linear_record(k, words, *, source="affine_agent", kind="tool_call"):
    """system -> user -> assistant(target). History tokens = words + 4."""
    nodes = [
        {"parent": None, "role": "system", "content": f"p{k}"},
        {"parent": 0, "role": "user", "content": " ".join(["w"] * words)},
        {"parent": 1, "role": "assistant", "content": f"REF{k}"},
    ]
    return {
        "view": VIEW,
        "rollout_id": f"r{k}",
        "traj_id": f"traj.{k}",
        "instance_id": f"i{k}",
        "repo": "repo",
        "model": "engy/qwen3.8-27b",
        "policy": {"id": "teacher", "harness": "null", "action_kind": kind},
        "source": source,
        "language": "chat",
        "action_kind": kind,
        "generated_at": "2026-09-30T00:00:00+00:00",
        "nodes": nodes,
        "turns": [
            {
                "turn_idx": 0,
                "node_id": 2,
                "phase": "early",
                "suffix_len": len(nodes[2]["content"]),
                "n_prefix_chars": len(nodes[0]["content"]) + len(nodes[1]["content"]),
                "action_kind": kind,
            }
        ],
    }


def branched_record():
    """A forked graph with a second root, baked tool text and later nodes."""
    nodes = [
        {"parent": None, "role": "system", "content": "<tools>baked</tools>"},  # 0
        {"parent": 0, "role": "user", "content": "question"},  # 1
        {"parent": 1, "role": "assistant", "content": "SIBLING"},  # 2
        {"parent": 2, "role": "user", "content": "SIBLINGFOLLOWUP"},  # 3
        {
            "parent": 1,
            "role": "assistant",
            "content": "<tool_call>search</tool_call>",
        },  # 4
        {
            "parent": 4,
            "role": "user",
            "content": "<tool_response>hit</tool_response>",
        },  # 5
        {"parent": 5, "role": "assistant", "content": "TARGET"},  # 6
        {"parent": 6, "role": "user", "content": "LATER"},  # 7
        {"parent": None, "role": "system", "content": "otherroot"},  # 8
        {"parent": 8, "role": "user", "content": "side"},  # 9
        {"parent": 9, "role": "assistant", "content": "SIDEREPLY"},  # 10
    ]

    def meta(turn_idx, node):
        chain, j = [], nodes[node]["parent"]
        while j is not None:
            chain.append(nodes[j])
            j = nodes[j]["parent"]
        return {
            "turn_idx": turn_idx,
            "node_id": node,
            "phase": "mid",
            "suffix_len": len(nodes[node]["content"]),
            "n_prefix_chars": sum(len(n["content"]) for n in chain),
            "action_kind": "tool_call",
        }

    return {
        **linear_record("b", 1),
        "traj_id": "traj.b",
        "nodes": nodes,
        "turns": [meta(0, 6), meta(1, 10)],
    }


def index_row(record, turn, chunk_key, line):
    return {
        "turn_id": f"{record['traj_id']}:{turn['turn_idx']}",
        "traj_id": record["traj_id"],
        "rollout_id": record["rollout_id"],
        "turn_idx": turn["turn_idx"],
        "node_id": turn["node_id"],
        "stratum": "s",
        "phase": turn["phase"],
        "source": record["source"],
        "language": record["language"],
        "action_kind": turn["action_kind"],
        "chunk_key": chunk_key,
        "traj_line": line,
        "n_prefix_chars": turn["n_prefix_chars"],
        "stratum_src": "extra column",
    }


def parquet_bytes(rows):
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


class Corpus:
    """In-memory published corpus served through an injected fetch."""

    def __init__(self, chunks=None, *, extra_rows=(), epoch=7, manifest_patch=None):
        if chunks is None:
            chunks = [
                [linear_record(k, TIER_TOKENS[k % 4] - 4) for k in range(0, 8)],
                [linear_record(k, TIER_TOKENS[k % 4] - 4) for k in range(8, 16)],
            ]
        self.objects, shards, rows = {}, [], []
        for c, records in enumerate(chunks):
            key = f"views/{VIEW}/chunks/view_{epoch:04d}_{c:04d}.jsonl.gz"
            blob, sha = write_gzip_jsonl(records)
            self.objects[key] = blob
            shards.append(
                {
                    "key": key,
                    "sha256": sha,
                    "format": "view_v4",
                    "active": True,
                    "n_trajectories": len(records),
                    "n_turns": sum(len(r["turns"]) for r in records),
                }
            )
            for line, record in enumerate(records):
                rows += [index_row(record, t, key, line) for t in record["turns"]]
        rows += list(extra_rows)
        index_key = f"views/{VIEW}/index/turns_{epoch:04d}.parquet"
        self.objects[index_key] = parquet_bytes(rows)
        manifest = {
            "schema_version": 3,
            "view_spec": VIEW,
            "corpus_epoch": epoch,
            "published_at": "2026-10-01T00:00:00+00:00",
            "index": {
                "key": index_key,
                "sha256": hashlib.sha256(self.objects[index_key]).hexdigest(),
                "n_turns": len(rows),
            },
            "shards": shards,
        }
        if manifest_patch:
            manifest_patch(manifest)
        self.publish(manifest)
        self.rows = rows
        self.requested = []

    def publish(self, manifest):
        raw = json.dumps(manifest, sort_keys=True).encode()
        self.manifest = manifest
        self.manifest_sha = hashlib.sha256(raw).hexdigest()
        self.objects[f"corpus/manifests/{self.manifest_sha}.json"] = raw
        self.objects["corpus/manifest.json"] = raw

    def fetch(self, url, max_bytes):
        key = url.removeprefix(BASE + "/")
        self.requested.append(key)
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def rule(self, **kw):
        return {
            "type": "affine_corpus",
            "algo_version": 5,
            "base_url": BASE,
            "manifest_sha256": "sha256:" + self.manifest_sha,
            "index_sha256": "sha256:" + self.manifest["index"]["sha256"],
            "corpus_epoch": self.manifest["corpus_epoch"],
            "n_turns": self.manifest["index"]["n_turns"],
            "n_prompts": 4,
            "max_tokens": 16,
            "request_interval_ms": 2,
            "enable_thinking": False,
            "max_prefix_chars": 100000,
            **kw,
        }

    def open(self, tmp_path, rule=None, **kw):
        parsed = parse_sampling_rule(rule or self.rule(**kw))
        return AffineCorpus(parsed, fetch=self.fetch, cache_dir=tmp_path / "cache")


def formatter(enable_thinking=False):
    vocab = {"[UNK]": 0, "system": 1, "user": 2, "assistant": 3, "w": 4}
    vocab.update({f"p{i}": 10 + i for i in range(2000)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return build_prompt_formatter(
        {
            "type": "affine_corpus",
            "algo_version": 5,
            "base_url": BASE,
            "manifest_sha256": "sha256:" + "0" * 64,
            "index_sha256": "sha256:" + "0" * 64,
            "corpus_epoch": 0,
            "n_turns": 1,
            "n_prompts": 4,
            "max_tokens": 1,
            "enable_thinking": enable_thinking,
            "max_prefix_chars": 1,
        },
        model_repo="test/model",
        model_revision="b" * 40,
        config_loader=lambda **_: {
            "chat_template": "{% for m in messages %}{{ m.role }} {{ m.content }} "
            "{% endfor %}assistant{% if enable_thinking %} THINK{% endif %}"
        },
        tokenizer_loader=lambda **_: tokenizer.to_str(),
    )


def fields(rule):
    return {
        "sampling_rule": rule,
        "engine": {"name": "sglang"},
        "bench": {
            "model": {
                "hf_repo": "test/model",
                "hf_revision": "b" * 40,
                "max_model_len": 32768,
            },
            "baseline_engine_image_digest": "sha256:" + "e" * 64,
        },
    }


def sample(corpus, rule, **kwargs):
    f = fields(rule)
    return generate_trace(
        **{
            "rule": rule,
            "seed_hex": "c" * 64,
            "row_fetcher": corpus,
            "prompt_formatter": formatter(),
            "sampling_context": sampling_context_for_campaign(f["bench"], f["engine"]),
            **kwargs,
        }
    )


@pytest.fixture
def published():
    return Corpus()


@pytest.fixture(autouse=True)
def no_retry_pause(monkeypatch):
    monkeypatch.setattr(affine, "RETRY_DELAYS_S", (0, 0))


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def test_history_is_the_root_to_parent_path_without_reference_siblings_or_later_nodes():
    record = branched_record()
    rows = [index_row(record, t, "k", 0) for t in record["turns"]]
    messages, reference = turn_messages(record, rows[0])
    assert messages == [
        {"role": "system", "content": "<tools>baked</tools>"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<tool_call>search</tool_call>"},
        {"role": "user", "content": "<tool_response>hit</tool_response>"},
    ]
    assert reference == "TARGET"
    text = json.dumps(messages)
    for excluded in ("TARGET", "SIBLING", "LATER", "otherroot", "SIDEREPLY"):
        assert excluded not in text
    side, reply = turn_messages(record, rows[1])
    assert side == [
        {"role": "system", "content": "otherroot"},
        {"role": "user", "content": "side"},
    ]
    assert reply == "SIDEREPLY"


def _break(record, change):
    nodes = record["nodes"]
    if change == "cycle":
        nodes[0]["parent"] = 2
    elif change == "dangling_parent":
        nodes[1]["parent"] = 99
    elif change == "target_not_assistant":
        nodes[2]["role"] = "user"
    elif change == "history_ends_on_assistant":
        nodes[1]["role"] = "assistant"
    elif change == "unknown_role":
        nodes[0]["role"] = "tool"
    elif change == "non_text_content":
        nodes[1]["content"] = ["w"]
    elif change == "prefix_length":
        nodes[1]["content"] += " w"
    elif change == "turn_meta":
        record["turns"][0]["node_id"] = 1
    elif change == "traj_id":
        record["traj_id"] = "other"
    elif change == "view":
        record["view"] = "duel_turns@v3"
    elif change == "no_nodes":
        record["nodes"] = []


@pytest.mark.parametrize(
    "change",
    [
        "cycle",
        "dangling_parent",
        "target_not_assistant",
        "history_ends_on_assistant",
        "unknown_role",
        "non_text_content",
        "prefix_length",
        "turn_meta",
        "traj_id",
        "view",
        "no_nodes",
    ],
)
def test_malformed_records_are_rejected(change):
    record = linear_record(1, 5)
    row = index_row(record, record["turns"][0], "k", 0)
    _break(record, change)
    with pytest.raises(CorpusIntegrityError):
        turn_messages(record, row)


# --------------------------------------------------------------------------
# Sampling and replay
# --------------------------------------------------------------------------


def test_trace_fills_every_tier_and_replays_byte_identically(published, tmp_path):
    corpus = published.open(tmp_path)
    sampled = sample(corpus, corpus.rule)
    trace = validate_workload_trace_dict(json.loads(sampled.body))
    assert {r.input_length_group for r in trace.requests} == {"2k", "4k", "8k", "16k"}
    assert [r.arrival_offset_ms for r in trace.requests] == [0, 2, 4, 6]
    assert all(not r.sampling.ignore_eos and r.max_tokens == 16 for r in trace.requests)
    receipt = sampled.receipt
    assert receipt["manifest_sha256"] == "sha256:" + published.manifest_sha
    assert receipt["chat_template"]["enable_thinking"] is False
    assert receipt["tokenizer"]["model_revision"] == "b" * 40
    assert len(receipt["turn_ids"]) == 4
    pool = corpus.eligible_rows()
    for request, entry in zip(trace.requests, receipt["requests"], strict=True):
        row = pool[entry["turn_id"]]
        messages, reference = turn_messages(corpus.record(row), row)
        assert request.prompt == formatter().render(messages)
        assert reference not in request.prompt
        assert entry["request_id"] == request.id
        assert (
            entry["reference_sha256"]
            == "sha256:" + hashlib.sha256(reference.encode()).hexdigest()
        )
        assert entry["chunk_sha256"] == "sha256:" + corpus.chunk_sha256(
            row["chunk_key"]
        )
    # A fresh process with only the cache and the receipt reproduces the bytes.
    offline = AffineCorpus(
        corpus.rule,
        fetch=lambda *a: (_ for _ in ()).throw(OSError("offline")),
        cache_dir=tmp_path / "cache",
    )
    rebuilt = sample(offline, offline.rule, sampling_receipt=receipt)
    assert rebuilt.body == sampled.body and rebuilt.receipt == receipt
    again = sample(published.open(tmp_path / "again"), corpus.rule)
    assert again.body == sampled.body and again.receipt == receipt


def test_published_pointer_is_never_read(published, tmp_path):
    first = sample(
        published.open(tmp_path / "a"), parse_sampling_rule(published.rule())
    )
    pinned = published.rule()
    newer = copy.deepcopy(published.manifest)
    newer["corpus_epoch"] = 8
    published.publish(newer)  # the pointer now names a different revision
    second_corpus = published.open(tmp_path / "b", rule=pinned)
    second = sample(second_corpus, second_corpus.rule)
    assert second.body == first.body
    assert "corpus/manifest.json" not in published.requested


def test_selection_reaches_the_full_eligible_pool(published, tmp_path):
    corpus = published.open(tmp_path)
    chosen = set()
    for i in range(48):
        seed = hashlib.sha256(str(i).encode()).hexdigest()
        chosen.update(sample(corpus, corpus.rule, seed_hex=seed).receipt["turn_ids"])
    assert chosen == set(corpus.eligible_rows())


def test_duplicate_turn_ids_are_excluded_and_duplicate_prompts_skipped(tmp_path):
    base = Corpus()
    duplicate = dict(base.rows[0])
    extra = Corpus(extra_rows=[duplicate])
    corpus = extra.open(tmp_path)
    assert base.rows[0]["turn_id"] not in corpus.eligible_rows()
    assert len(corpus.eligible_rows()) == 15
    # Two different turns rendering to the same prompt: only one may be drawn.
    twin = linear_record(0, TIER_TOKENS[0] - 4)
    twin["traj_id"] = "traj.twin"
    chunks = [
        [linear_record(k, TIER_TOKENS[k % 4] - 4) for k in range(4, 8)]
        + [linear_record(k, TIER_TOKENS[k % 4] - 4) for k in range(4)]
        + [twin]
    ]
    twins = Corpus(chunks)
    corpus = twins.open(tmp_path / "twin")
    hashes = [
        r["input_ids_sha256"] for r in sample(corpus, corpus.rule).receipt["requests"]
    ]
    assert len(set(hashes)) == len(hashes)
    # Every tier has two distinct prompts; the twin cannot become a third 2K one.
    wide = twins.open(tmp_path / "wide", n_prompts=12)
    with pytest.raises(SamplerError, match="insufficient"):
        sample(wide, wide.rule)


def test_missing_tier_fails_without_fallback(tmp_path):
    chunks = [[linear_record(k, TIER_TOKENS[0] - 4) for k in range(8)]]
    corpus = Corpus(chunks).open(tmp_path)
    with pytest.raises(SamplerError, match="missing by input tier"):
        sample(corpus, corpus.rule)


def test_filters_and_prefix_cap_define_the_pool(tmp_path):
    chunks = [
        [linear_record(k, TIER_TOKENS[k % 4] - 4, source="a") for k in range(4)]
        + [
            linear_record(k, TIER_TOKENS[k % 4] - 4, source="b", kind="bash")
            for k in range(4, 8)
        ]
    ]
    published = Corpus(chunks)
    assert {
        r["source"]
        for r in published.open(tmp_path / "s", sources=["b"]).eligible_rows().values()
    } == {"b"}
    assert {
        r["action_kind"]
        for r in published.open(tmp_path / "k", action_kinds=["tool_call"])
        .eligible_rows()
        .values()
    } == {"tool_call"}
    capped = published.open(tmp_path / "c", max_prefix_chars=4000)
    assert all(r["n_prefix_chars"] <= 4000 for r in capped.eligible_rows().values())


@pytest.mark.parametrize("change", ["seed", "turn", "context", "thinking", "manifest"])
def test_replay_rejects_any_changed_selection_or_contract(published, tmp_path, change):
    corpus = published.open(tmp_path)
    sampled = sample(corpus, corpus.rule)
    receipt = copy.deepcopy(sampled.receipt)
    kwargs = {"sampling_receipt": receipt}
    if change == "seed":
        kwargs["seed_hex"] = "d" * 64
    elif change == "turn":
        receipt["turn_ids"][0] = next(
            t for t in corpus.eligible_rows() if t not in receipt["turn_ids"]
        )
    elif change == "context":
        f = fields(corpus.rule)
        f["bench"]["model"]["max_model_len"] = 65536
        kwargs["sampling_context"] = sampling_context_for_campaign(
            f["bench"], f["engine"]
        )
    elif change == "thinking":
        kwargs["prompt_formatter"] = formatter(enable_thinking=True)
    else:
        receipt["manifest_sha256"] = "sha256:" + "1" * 64
    with pytest.raises(SamplerError):
        sample(corpus, corpus.rule, **kwargs)


@pytest.mark.parametrize(
    "mutation", ["label", "quota", "workload", "interval", "sampling"]
)
def test_trace_validation_rejects_tampering(published, tmp_path, mutation):
    corpus = published.open(tmp_path)
    trace = json.loads(sample(corpus, corpus.rule).body)
    if mutation == "label":
        trace["requests"][0]["input_length_group"] = "32k"
    elif mutation == "quota":
        for request in trace["requests"]:
            request.update(input_tokens=1000, input_length_group="2k")
    elif mutation == "workload":
        trace["meta"]["sampling"]["workload"] = "other"
    elif mutation == "interval":
        trace["requests"][1]["arrival_offset_ms"] = 7
    else:
        trace["requests"][0]["sampling"]["ignore_eos"] = True
    with pytest.raises(RequestValidationError):
        validate_workload_trace_dict(trace)


# --------------------------------------------------------------------------
# Integrity of pinned objects
# --------------------------------------------------------------------------


def _corrupt(published, change, monkeypatch):
    objects = published.objects
    index_key = published.manifest["index"]["key"]
    chunk_key = published.manifest["shards"][0]["key"]
    if change == "manifest_bytes":
        objects[f"corpus/manifests/{published.manifest_sha}.json"] += b" "
    elif change == "index_bytes":
        objects[index_key] += b"\0"
    elif change == "chunk_content":
        objects[chunk_key], _ = write_gzip_jsonl([linear_record(0, 1)])
    elif change == "chunk_not_gzip":
        objects[chunk_key] = b"not gzip"
    elif change == "chunk_truncated":
        objects[chunk_key] = objects[chunk_key][:-12]
    elif change == "chunk_bomb":
        monkeypatch.setattr(affine, "MAX_CHUNK_JSONL_BYTES", 64)
    elif change == "chunk_missing":
        del objects[chunk_key]
    elif change == "index_missing":
        del objects[index_key]


@pytest.mark.parametrize(
    "change",
    [
        "manifest_bytes",
        "index_bytes",
        "chunk_content",
        "chunk_not_gzip",
        "chunk_truncated",
        "chunk_bomb",
        "chunk_missing",
        "index_missing",
    ],
)
def test_corrupt_or_missing_objects_fail_closed(
    published, tmp_path, monkeypatch, change
):
    _corrupt(published, change, monkeypatch)
    corpus = published.open(tmp_path)
    with pytest.raises(CorpusIntegrityError):
        corpus.eligible_rows()
        for row in corpus.eligible_rows().values():
            corpus.record(row)


def test_cut_transfers_are_retried_and_persistent_damage_still_fails(
    published, tmp_path
):
    chunk_key = published.manifest["shards"][0]["key"]
    good = published.objects[chunk_key]
    served = []

    def flaky(url, max_bytes):
        data = published.fetch(url, max_bytes)
        if url.endswith(chunk_key):
            served.append(url)
            if len(served) == 1:
                return data[: len(data) // 2]  # cut mid-stream
        return data

    corpus = published.open(tmp_path)
    corpus.fetch = flaky
    rows = [r for r in corpus.eligible_rows().values() if r["chunk_key"] == chunk_key]
    assert corpus.record(rows[0])["traj_id"] == rows[0]["traj_id"]
    assert len(served) == 2
    published.objects[chunk_key] = good[:-12]
    attempts = []
    damaged = published.open(tmp_path / "damaged")
    damaged.fetch = lambda url, n: (attempts.append(url), published.fetch(url, n))[1]
    with pytest.raises(CorpusIntegrityError):
        damaged.record(rows[0])
    assert attempts.count(f"{BASE}/{chunk_key}") == 1 + len(affine.RETRY_DELAYS_S)


def test_incomplete_http_body_is_detected(monkeypatch):
    class Response:
        headers = {"Content-Length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def geturl(self):
            return BASE + "/x"

        def read(self, n):
            return b"12345"

    monkeypatch.setattr(affine.urllib.request, "urlopen", lambda *a, **k: Response())
    with pytest.raises(CorpusIntegrityError, match="incomplete"):
        affine.http_get(BASE + "/x", 100)


def test_replay_fails_when_a_selected_object_is_corrupt(published, tmp_path):
    corpus = published.open(tmp_path / "first")
    receipt = sample(corpus, corpus.rule).receipt
    for key in {e["chunk_key"] for e in receipt["requests"]}:
        published.objects[key], _ = write_gzip_jsonl([linear_record(99, 1)])
    fresh = published.open(tmp_path / "second")
    with pytest.raises(CorpusIntegrityError):
        sample(fresh, fresh.rule, sampling_receipt=receipt)


def test_cached_objects_are_reverified(published, tmp_path):
    corpus = published.open(tmp_path)
    corpus.index_rows()
    cached = tmp_path / "cache" / "index" / published.manifest["index"]["sha256"]
    cached.write_bytes(b"tampered")
    fresh = published.open(tmp_path)
    assert len(fresh.index_rows()) == len(published.rows)  # refetched and verified
    assert cached.read_bytes() == published.objects[published.manifest["index"]["key"]]
    row = next(iter(fresh.eligible_rows().values()))
    record = fresh.record(row)
    gz = tmp_path / "cache" / "chunk-gz" / fresh.chunk_sha256(row["chunk_key"])
    assert gz.read_bytes() == published.objects[row["chunk_key"]]  # stored compressed
    gz.write_bytes(write_gzip_jsonl([linear_record(5, 1)])[0])
    assert published.open(tmp_path).record(row) == record
    assert gz.read_bytes() == published.objects[row["chunk_key"]]


@pytest.mark.parametrize(
    "patch",
    [
        lambda m: m.update(schema_version=2),
        lambda m: m.update(view_spec="duel_turns@v3"),
        lambda m: [s.update(active=False) for s in m["shards"]],
        lambda m: m["shards"][0].update(active=False),
        lambda m: m["shards"][0].update(format="traj_v1"),
        lambda m: m["index"].update(n_turns=m["index"]["n_turns"] + 1),
    ],
    ids=["schema", "view", "no_active", "inactive_indexed", "format", "row_count"],
)
def test_unsupported_or_inconsistent_manifests_are_rejected(tmp_path, patch):
    published = Corpus(manifest_patch=patch)
    rule = published.rule()
    with pytest.raises(CorpusIntegrityError):
        published.open(tmp_path, rule=rule).eligible_rows()


def test_rule_pins_must_match_the_manifest(published, tmp_path):
    for change in (
        {"corpus_epoch": 8},
        {"n_turns": 1},
        {"index_sha256": "sha256:" + "2" * 64},
    ):
        with pytest.raises(CorpusIntegrityError):
            published.open(tmp_path, **change).eligible_rows()


# --------------------------------------------------------------------------
# Rule contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"base_url": "http://corpus.example"},
        {"base_url": "https://corpus.example/path"},
        {"manifest_sha256": "abc"},
        {"n_prompts": 6},
        {"algo_version": 4},
        {"ignore_eos": True},
        {"followup_prompt": "x"},
        {"enable_thinking": "no"},
        {"sources": ["b", "a"]},
        {"temperature": 3},
        {"qualification": {"contract_sha256": "sha256:" + "a" * 64}},
    ],
)
def test_invalid_rules_are_rejected(published, change):
    with pytest.raises(SamplerError):
        parse_sampling_rule(published.rule(**change))


def test_existing_sampler_versions_are_untouched():
    rule = {
        "type": "hf_rows",
        "dataset": "d",
        "revision": "r",
        "n_rows": 10,
        "n_prompts": 2,
    }
    assert parse_sampling_rule(rule)["algo_version"] == 2
    with pytest.raises(SamplerError, match="unsupported algo_version"):
        parse_sampling_rule({**rule, "algo_version": 5})


def test_repository_fixture_pins_the_live_epoch_103_corpus():
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures/workloads/affine_corpus_e103/sampling_rule.json"
    )
    rule = parse_sampling_rule(json.loads(path.read_text()))
    assert rule["base_url"] == "https://data.affine.io"
    assert rule["manifest_sha256"].endswith(
        "4f1d5945dcbb260a0493ca3f3060ca09c0a295d285ba386099fd46fe4cbb9778"
    )
    assert rule["index_sha256"].endswith(
        "adfc84f948254f238e990426afd88d6a3ce58eff7530acb77c106b1dd4798dc7"
    )
    assert (rule["corpus_epoch"], rule["n_turns"]) == (103, 300209)
    assert rule["enable_thinking"] is False and "qualification" not in rule


# --------------------------------------------------------------------------
# Rounds, qualification and preview
# --------------------------------------------------------------------------


def test_round_creation_and_worker_replay_use_the_pinned_corpus(
    published, tmp_path, monkeypatch
):
    corpus = published.open(tmp_path)
    f = fields(published.rule())
    campaign = SimpleNamespace(
        campaign_id=uuid4(),
        gpu_skus=["H100"],
        sampling_rule=published.rule(),
        scoring_rule={"name": "median_e2e_speedup", "failure_penalty": 0.1},
        bench=f["bench"],
        engine=f["engine"],
    )
    monkeypatch.setattr("round.create.create_round", lambda **kw: kw)
    monkeypatch.setattr("bench.affine_corpus._CORPORA", {})
    monkeypatch.setattr(
        "bench.affine_corpus.default_cache_dir", lambda: tmp_path / "cache"
    )
    monkeypatch.setattr(affine, "http_get", published.fetch)
    result = try_create_round(
        campaign,
        {"queued": 100, "oldest_queued_at": None},
        seed_block=10,
        seed_block_hash="a" * 64,
        prompt_formatter=formatter(),
    )
    assert result["sampling_receipt"]["type"] == "affine_corpus"
    path = materialize_round_trace(
        result,
        campaign,
        tmp_path / "trace",
        row_fetcher=corpus,
        prompt_formatter=formatter(),
    )
    assert (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        == result["sampled_trace_sha256"]
    )
    # The default worker path rebuilds the rule from the receipt alone.
    path = materialize_round_trace(
        result, campaign, tmp_path / "default", prompt_formatter=formatter()
    )
    assert (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        == result["sampled_trace_sha256"]
    )
    campaign.bench["model"]["max_model_len"] *= 2
    with pytest.raises(RoundInfraError, match="sampling receipt"):
        materialize_round_trace(
            result,
            campaign,
            tmp_path / "changed",
            row_fetcher=corpus,
            prompt_formatter=formatter(),
        )


def test_qualification_binds_pins_records_evidence_and_preview_matches_rounds(
    published, tmp_path
):
    from bench.qualify_affine import preview, qualify
    from bench.sampler import compute_sample_seed

    f = fields(published.rule())
    corpus = published.open(tmp_path)
    rule = qualify(
        fields=f,
        output_dir=tmp_path / "q",
        scan_turns=16,
        corpus=corpus,
        formatter=formatter(),
    )
    assert rule["qualification"]["scanned_turns"] == 16
    assert rule["qualification"]["eligible_by_tier"] == {
        "2k": 4,
        "4k": 4,
        "8k": 4,
        "16k": 4,
    }
    require_qualification(rule, f["bench"], f["engine"])
    changed = copy.deepcopy(f["bench"])
    changed["model"]["hf_revision"] = "c" * 40
    with pytest.raises(SamplerError, match="qualification"):
        require_qualification(rule, changed, f["engine"])
    with pytest.raises(SamplerError, match="qualification"):
        require_qualification({**rule, "max_tokens": 32}, f["bench"], f["engine"])
    lines = [
        json.loads(x)
        for x in (tmp_path / "q/qualification.jsonl").read_text().splitlines()
    ]
    assert lines[0]["type"] == "contract"
    assert lines[0]["corpus"]["manifest_sha256"] == "sha256:" + published.manifest_sha
    assert {x["input_length_group"] for x in lines[1:]} == {"2k", "4k", "8k", "16k"}
    summary = json.loads((tmp_path / "q/summary.json").read_text())
    assert summary["accepted_turns"] == 16
    assert summary["demo_draw"]["receipt_replay"] == "exact"
    assert summary["input_tokens"]["16k"]["min"] >= 8193

    campaign_id = uuid4()
    actual = preview(
        fields={**f, "sampling_rule": rule},
        output_dir=tmp_path / "p",
        campaign_id=campaign_id,
        seed_block=10,
        block_hash="a" * 64,
        corpus=published.open(tmp_path, rule=rule),
        formatter=formatter(),
    )
    expected = sample(
        published.open(tmp_path, rule=rule),
        parse_sampling_rule(rule),
        seed_hex=compute_sample_seed(block_hash="a" * 64, campaign_id=campaign_id),
        sample_seed_block=10,
        sample_seed_block_hash="a" * 64,
    )
    assert actual.body == expected.body and actual.receipt == expected.receipt
    assert len((tmp_path / "p/index.tsv").read_text().splitlines()) == 5
    for i in range(4):
        messages = json.loads((tmp_path / f"p/af-{i:03d}.messages.json").read_text())
        assert [m["role"] for m in messages] == ["system", "user"]
        assert (
            tmp_path / f"p/af-{i:03d}.prompt.txt"
        ).read_text() == formatter().render(messages)


def test_qualification_without_headroom_writes_evidence_but_no_rule(tmp_path):
    chunks = [[linear_record(k, TIER_TOKENS[k % 4] - 4) for k in range(8)]]
    published = Corpus(chunks)
    from bench.qualify_affine import qualify

    with pytest.raises(SamplerError, match="insufficient"):
        qualify(
            fields=fields(published.rule()),
            output_dir=tmp_path / "q",
            scan_turns=8,
            corpus=published.open(tmp_path),
            formatter=formatter(),
        )
    assert (tmp_path / "q/qualification.jsonl").exists()
    assert not (tmp_path / "q/sampling_rule.json").exists()


def test_qualification_reports_rejection_reasons(tmp_path):
    from bench.qualify_affine import qualify

    chunks = [
        [linear_record(k, TIER_TOKENS[k % 4] - 4) for k in range(16)]
        + [linear_record(100 + k, 20000) for k in range(4)]
    ]
    published = Corpus(chunks)
    qualify(
        fields=fields(published.rule()),
        output_dir=tmp_path / "q",
        scan_turns=20,
        corpus=published.open(tmp_path),
        formatter=formatter(),
    )
    summary = json.loads((tmp_path / "q/summary.json").read_text())
    assert summary["rejections"] == {"outside_input_tiers": 4}
    assert summary["accepted_turns"] == 16
