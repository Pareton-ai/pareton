"""Version 5: replay turns from Affine's public corpus (schema 3, duel_turns@v4).

The corpus is append-only objects behind a mutable pointer. A rule pins one
immutable manifest revision by hash and never reads the pointer, so a newly
published corpus cannot change a campaign's rounds. Every object is verified
against the hash the manifest publishes for it:

* manifest: SHA-256 of its bytes, stored at corpus/manifests/{sha256}.json
* index: SHA-256 of the raw Parquet bytes
* chunk: SHA-256 of the *uncompressed* JSONL (the object itself is gzip)

A view record holds a message graph (`nodes[i] = {parent, role, content}`) and
one meta per scorable reply. A turn's prompt is the root-to-parent path of its
reply node; the reply itself is the Affine reference and is never sent. Later
nodes and sibling branches are not on that path. Tool calls and tool results
are already baked into message text; nothing in the corpus is executed.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import time
import urllib.request
import zlib
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from bench.longform import (
    digest,
    generation_fields,
    generation_sampling,
    sampling_context_for_campaign as _longform_context,
    validate_context,
)
from bench.sampler import (
    PromptRenderError,
    SampledTrace,
    SamplerError,
    encode_trace,
    normalize_sha256,
)
from bench.trajectory import token_ids_sha256

RULE_TYPE = "affine_corpus"
ALGO_VERSION = 5
SCHEMA_VERSION = 3
VIEW_SPEC = "duel_turns@v4"
CHUNK_FORMAT = "view_v4"
ROLES = frozenset({"system", "user", "assistant"})
INDEX_COLUMNS = (
    "turn_id",
    "traj_id",
    "turn_idx",
    "node_id",
    "phase",
    "source",
    "action_kind",
    "chunk_key",
    "traj_line",
    "n_prefix_chars",
)
RULE_FIELDS = frozenset(
    {
        "type",
        "algo_version",
        "base_url",
        "manifest_sha256",
        "index_sha256",
        "corpus_epoch",
        "n_turns",
        "n_prompts",
        "max_tokens",
        "request_interval_ms",
        "enable_thinking",
        "seed_block_offset",
        "max_prefix_chars",
        "sources",
        "action_kinds",
        "temperature",
        "temperature_range",
        "qualification",
    }
)

# Fields that decide which corpus objects and turns a rule can draw from.
POOL_FIELDS = (
    "base_url",
    "manifest_sha256",
    "index_sha256",
    "corpus_epoch",
    "n_turns",
    "max_prefix_chars",
    "sources",
    "action_kinds",
)

# Download and decompression ceilings. The live corpus is far below them
# (manifest ~100 KiB, index ~5 MiB, chunks ~0.5 MiB compressed).
MAX_MANIFEST_BYTES = 16 << 20
MAX_INDEX_BYTES = 512 << 20
MAX_CHUNK_BYTES = 128 << 20
MAX_CHUNK_JSONL_BYTES = 1 << 30
FETCH_TIMEOUT_S = 120
# Pauses between download attempts. Objects are content-addressed, so a retry
# can only recover a transfer cut short, never accept different content.
RETRY_DELAYS_S = (1.0, 3.0)

_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
_CHUNK_KEY_RE = re.compile(
    r"views/" + re.escape(VIEW_SPEC) + r"/chunks/[A-Za-z0-9_.-]+\.jsonl\.gz"
)
_INDEX_KEY_RE = re.compile(
    r"views/" + re.escape(VIEW_SPEC) + r"/index/[A-Za-z0-9_.-]+\.parquet"
)


class CorpusIntegrityError(SamplerError):
    """A pinned corpus object is missing, corrupt or inconsistent."""


# --------------------------------------------------------------------------
# Rule
# --------------------------------------------------------------------------


def length_groups(n_prompts):
    """Contiguous input tiers up to 16K tokens, with equal quotas."""
    bounds = ((1, 2048), (2049, 4096), (4097, 8192), (8193, 16384))
    return [
        {
            "name": f"{high // 1024}k",
            "min_tokens": low,
            "max_tokens": high,
            "count": n_prompts // 4,
        }
        for low, high in bounds
    ]


def input_group(input_tokens):
    return next(
        (
            g["name"]
            for g in length_groups(4)
            if g["min_tokens"] <= input_tokens <= g["max_tokens"]
        ),
        None,
    )


def _int_field(rule, name, minimum, default=None):
    value = rule.get(name, default)
    if type(value) is not int or value < minimum:
        raise SamplerError(f"affine_corpus {name} must be an integer >= {minimum}")
    return value


def _name_list(rule, name):
    values = rule[name]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(v, str) or not v.strip() for v in values)
        or values != sorted(set(values))
    ):
        raise SamplerError(f"affine_corpus {name} must be sorted unique names")
    return list(values)


def parse_affine_rule(rule):
    """Validate a version 5 rule. Unknown fields are rejected, not ignored."""
    unknown = set(rule) - RULE_FIELDS
    if unknown:
        raise SamplerError(f"unknown affine_corpus sampling fields: {sorted(unknown)}")
    if (
        rule.get("algo_version") != ALGO_VERSION
        or type(rule["algo_version"]) is not int
    ):
        raise SamplerError(f"affine_corpus requires algo_version {ALGO_VERSION}")
    url = urlsplit(str(rule.get("base_url") or ""))
    if (
        url.scheme != "https"
        or not url.hostname
        or url.path not in ("", "/")
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        raise SamplerError("affine_corpus base_url must be an https origin")
    try:
        manifest = normalize_sha256(rule.get("manifest_sha256", ""))
        index = normalize_sha256(rule.get("index_sha256", ""))
    except SamplerError as exc:
        raise SamplerError("affine_corpus requires manifest and index hashes") from exc
    thinking = rule.get("enable_thinking", False)
    if not isinstance(thinking, bool):
        raise SamplerError("enable_thinking must be a boolean")
    parsed = {
        "type": RULE_TYPE,
        "algo_version": ALGO_VERSION,
        "base_url": f"https://{url.netloc}",
        "manifest_sha256": manifest,
        "index_sha256": index,
        "corpus_epoch": _int_field(rule, "corpus_epoch", 0),
        "n_turns": _int_field(rule, "n_turns", 1),
        "n_prompts": _int_field(rule, "n_prompts", 4),
        "max_tokens": _int_field(rule, "max_tokens", 1),
        "request_interval_ms": _int_field(rule, "request_interval_ms", 0, 200),
        "enable_thinking": thinking,
        "seed_block_offset": _int_field(rule, "seed_block_offset", 0, 1),
        "max_prefix_chars": _int_field(rule, "max_prefix_chars", 1),
    }
    if parsed["n_prompts"] % 4:
        raise SamplerError("affine_corpus n_prompts must be a multiple of 4")
    for name in ("sources", "action_kinds"):
        if name in rule:
            parsed[name] = _name_list(rule, name)
    # Absent generation fields stay absent so the rule hash does not move.
    parsed.update(generation_fields(rule))
    if "qualification" in rule:
        q = rule["qualification"]
        if (
            not isinstance(q, dict)
            or set(q) != {"contract_sha256", "scanned_turns", "eligible_by_tier"}
            or not _SHA_RE.fullmatch(str(q["contract_sha256"]))
            or type(q["scanned_turns"]) is not int
            or q["scanned_turns"] < 1
            or not isinstance(q["eligible_by_tier"], dict)
            or set(q["eligible_by_tier"]) != {g["name"] for g in length_groups(4)}
            or any(
                type(v) is not int or v < parsed["n_prompts"] // 4
                for v in q["eligible_by_tier"].values()
            )
        ):
            raise SamplerError("invalid affine_corpus qualification metadata")
        parsed["qualification"] = dict(q)
    return parsed


def sampling_context_for_campaign(bench, engine=None):
    try:
        return _longform_context(bench, engine)
    except SamplerError as exc:
        raise SamplerError(
            "affine_corpus sampling requires valid model context and engine"
        ) from exc


def qualification_contract(rule, bench, engine):
    return digest(
        {
            "sampling_rule": {k: v for k, v in rule.items() if k != "qualification"},
            "bench": bench,
            "engine": engine,
        }
    )


def require_qualification(rule, bench, engine):
    """Check a trusted operator's rule for stale settings, not authenticity."""
    q = rule.get("qualification")
    if not q or q["contract_sha256"] != qualification_contract(rule, bench, engine):
        raise SamplerError(
            "affine_corpus campaign requires qualification for these exact pins; "
            "run python -m bench.qualify_affine qualify and pass its sampling rule"
        )


# --------------------------------------------------------------------------
# Pinned corpus access
# --------------------------------------------------------------------------


def _hex(value):
    return normalize_sha256(value).removeprefix("sha256:")


def pool_key(rule):
    return tuple(json.dumps(rule.get(name), sort_keys=True) for name in POOL_FIELDS)


def http_get(url, max_bytes):
    """GET with a hard size cap. Redirects stay on https."""
    request = urllib.request.Request(url, headers={"User-Agent": "pareton-sampler"})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
        if urlsplit(response.geturl()).scheme != "https":
            raise CorpusIntegrityError(f"corpus fetch left https: {url}")
        expected = response.headers.get("Content-Length")
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise CorpusIntegrityError(f"corpus object exceeds {max_bytes} bytes: {url}")
    if expected is not None and expected.isdigit() and int(expected) != len(body):
        raise CorpusIntegrityError(
            f"corpus transfer incomplete: {len(body)} of {expected} bytes: {url}"
        )
    return body


def gunzip_bounded(blob, max_bytes):
    out = io.BytesIO()
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data = inflater.decompress(blob, max_bytes + 1)
        out.write(data)
        if out.tell() > max_bytes or inflater.unconsumed_tail:
            raise CorpusIntegrityError(f"chunk inflates beyond {max_bytes} bytes")
        out.write(inflater.flush())
    except zlib.error as exc:
        raise CorpusIntegrityError("chunk is not valid gzip") from exc
    if not inflater.eof or inflater.unused_data:
        raise CorpusIntegrityError(
            "chunk gzip stream is truncated or has trailing data"
        )
    return out.getvalue()


def default_cache_dir():
    configured = os.environ.get("PARETON_AFFINE_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "pareton" / "affine-corpus"


class AffineCorpus:
    """Read-only, content-verified view of one pinned manifest revision.

    `fetch(url, max_bytes) -> bytes` is injected in tests. Verified objects are
    cached by content hash and re-verified on every load. Chunks stay gzip on
    disk (about 1.4 GiB for the whole epoch 103 corpus, versus 5 GiB inflated);
    only the requested record line of a loaded chunk is parsed.
    """

    def __init__(self, rule, *, fetch=None, cache_dir=None):
        self.rule = rule
        self.fetch = fetch or http_get
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self._manifest = None
        self._shards = None
        self._pool = None
        self.index_stats = None
        self._chunks = {}

    # -- content-addressed objects ------------------------------------------
    def _cached(self, sha_hex, kind):
        path = self.cache_dir / kind / sha_hex
        if path.is_file():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == sha_hex:
                return data
        return None

    def _store(self, sha_hex, kind, data):
        folder = self.cache_dir / kind
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".{sha_hex}.{os.getpid()}.tmp"
        tmp.write_bytes(data)
        tmp.replace(folder / sha_hex)

    def _download_verified(self, key, max_bytes, verify):
        """Fetch and verify one object, retrying failed or damaged transfers."""
        error = None
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            if attempt:
                time.sleep(RETRY_DELAYS_S[attempt - 1])
            try:
                blob = self.fetch(f"{self.rule['base_url']}/{key}", max_bytes)
                return blob, verify(blob)
            except CorpusIntegrityError as exc:
                error = exc
            except Exception as exc:
                error = CorpusIntegrityError(
                    f"corpus object unavailable: {key} ({type(exc).__name__})"
                )
                error.__cause__ = exc
        raise error

    def _object(self, key, sha_hex, *, kind, max_bytes):
        data = self._cached(sha_hex, kind)
        if data is not None:
            return data

        def verify(blob):
            got = hashlib.sha256(blob).hexdigest()
            if got != sha_hex:
                raise CorpusIntegrityError(f"{key} sha256 mismatch: {got} != {sha_hex}")
            return blob

        data, _ = self._download_verified(key, max_bytes, verify)
        self._store(sha_hex, kind, data)
        return data

    def _chunk_jsonl(self, key, sha_hex):
        """Inflated chunk bytes. The gzip object is cached under the hash of
        its uncompressed content, which is what the manifest publishes."""
        path = self.cache_dir / "chunk-gz" / sha_hex
        if path.is_file():
            try:
                data = gunzip_bounded(path.read_bytes(), MAX_CHUNK_JSONL_BYTES)
            except CorpusIntegrityError:
                data = None
            if data is not None and hashlib.sha256(data).hexdigest() == sha_hex:
                return data

        def verify(blob):
            data = gunzip_bounded(blob, MAX_CHUNK_JSONL_BYTES)
            got = hashlib.sha256(data).hexdigest()
            if got != sha_hex:
                raise CorpusIntegrityError(f"{key} sha256 mismatch: {got} != {sha_hex}")
            return data

        blob, data = self._download_verified(key, MAX_CHUNK_BYTES, verify)
        self._store(sha_hex, "chunk-gz", blob)
        return data

    # -- manifest -----------------------------------------------------------
    def manifest(self):
        if self._manifest is not None:
            return self._manifest
        sha = _hex(self.rule["manifest_sha256"])
        raw = self._object(
            f"corpus/manifests/{sha}.json",
            sha,
            kind="manifest",
            max_bytes=MAX_MANIFEST_BYTES,
        )
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusIntegrityError("pinned manifest is not JSON") from exc
        if not isinstance(manifest, dict):
            raise CorpusIntegrityError("pinned manifest is not an object")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CorpusIntegrityError(
                f"unsupported corpus schema_version {manifest.get('schema_version')!r}"
            )
        if manifest.get("view_spec") != VIEW_SPEC:
            raise CorpusIntegrityError(
                f"unsupported corpus view {manifest.get('view_spec')!r}"
            )
        if manifest.get("corpus_epoch") != self.rule["corpus_epoch"]:
            raise CorpusIntegrityError("manifest epoch does not match the rule pin")
        index = manifest.get("index")
        if (
            not isinstance(index, dict)
            or not isinstance(index.get("key"), str)
            or not _INDEX_KEY_RE.fullmatch(index["key"])
            or "sha256:" + str(index.get("sha256", "")).lower()
            != self.rule["index_sha256"]
            or index.get("n_turns") != self.rule["n_turns"]
        ):
            raise CorpusIntegrityError("manifest index does not match the rule pins")
        shards = {}
        for shard in manifest.get("shards") or []:
            if not isinstance(shard, dict):
                raise CorpusIntegrityError("manifest shard is not an object")
            key = shard.get("key")
            if not isinstance(key, str) or key in shards:
                raise CorpusIntegrityError(f"manifest shard key invalid: {key!r}")
            shards[key] = shard
        active = {
            key: str(shard.get("sha256", "")).lower()
            for key, shard in shards.items()
            if shard.get("active") is True
            and shard.get("format") == CHUNK_FORMAT
            and _CHUNK_KEY_RE.fullmatch(key)
            and re.fullmatch(r"[0-9a-f]{64}", str(shard.get("sha256", "")).lower())
        }
        if not active:
            raise CorpusIntegrityError("manifest has no active view_v4 chunks")
        self._manifest, self._shards = manifest, active
        return manifest

    # -- index --------------------------------------------------------------
    def index_rows(self):
        """Every verified index row. Not retained: see eligible_rows()."""
        import sys

        import pyarrow as pa
        import pyarrow.parquet as pq

        manifest = self.manifest()
        sha = _hex(self.rule["index_sha256"])
        raw = self._object(
            manifest["index"]["key"], sha, kind="index", max_bytes=MAX_INDEX_BYTES
        )
        try:
            table = pq.read_table(pa.BufferReader(raw))
        except Exception as exc:
            raise CorpusIntegrityError("pinned index is not readable Parquet") from exc
        missing = [c for c in INDEX_COLUMNS if c not in table.column_names]
        if missing:
            raise CorpusIntegrityError(f"pinned index lacks columns {missing}")
        if table.num_rows != self.rule["n_turns"]:
            raise CorpusIntegrityError(
                f"index has {table.num_rows} rows, manifest claims {self.rule['n_turns']}"
            )
        columns = {c: table.column(c).to_pylist() for c in INDEX_COLUMNS}
        # Few distinct values repeat on every row; share one string object each.
        for c in ("phase", "source", "action_kind", "chunk_key"):
            columns[c] = [
                sys.intern(v) if isinstance(v, str) else v for v in columns[c]
            ]
        rows = [
            {c: columns[c][i] for c in INDEX_COLUMNS} for i in range(table.num_rows)
        ]
        for row in rows:
            if (
                not isinstance(row["turn_id"], str)
                or not row["turn_id"]
                or not isinstance(row["traj_id"], str)
                or row["turn_id"] != f"{row['traj_id']}:{row['turn_idx']}"
                or any(
                    type(row[c]) is not int or row[c] < 0
                    for c in ("turn_idx", "node_id", "traj_line", "n_prefix_chars")
                )
            ):
                raise CorpusIntegrityError(
                    f"malformed index row {row.get('turn_id')!r}"
                )
            if row["chunk_key"] not in self._shards:
                raise CorpusIntegrityError(
                    f"index row {row['turn_id']} points at an inactive or unknown chunk"
                )
        return rows

    def eligible_rows(self):
        """Index rows a version 5 rule may draw, before tokenization.

        Turn IDs that appear more than once are ambiguous and excluded."""
        if self._pool is not None:
            return self._pool
        rows = self.index_rows()
        counts = Counter(row["turn_id"] for row in rows)
        self.index_stats = {
            "indexed_turns": len(rows),
            "duplicate_turn_rows_excluded": sum(n for n in counts.values() if n > 1),
        }
        sources = set(self.rule.get("sources") or ())
        kinds = set(self.rule.get("action_kinds") or ())
        self._pool = {
            row["turn_id"]: row
            for row in rows
            if counts[row["turn_id"]] == 1
            and row["n_prefix_chars"] <= self.rule["max_prefix_chars"]
            and (not sources or row["source"] in sources)
            and (not kinds or row["action_kind"] in kinds)
        }
        return self._pool

    # -- chunks -------------------------------------------------------------
    def chunk_sha256(self, key):
        self.manifest()
        if key not in self._shards:
            raise CorpusIntegrityError(f"chunk {key} is not active in the manifest")
        return self._shards[key]

    def chunk_lines(self, key):
        """Raw record lines of one verified chunk (most recent few kept)."""
        if key in self._chunks:
            self._chunks[key] = self._chunks.pop(key)
            return self._chunks[key]
        data = self._chunk_jsonl(key, self.chunk_sha256(key))
        lines = [line for line in data.split(b"\n") if line]
        while len(self._chunks) >= 2:
            self._chunks.pop(next(iter(self._chunks)))
        self._chunks[key] = lines
        return lines

    def record(self, row):
        lines = self.chunk_lines(row["chunk_key"])
        if not 0 <= row["traj_line"] < len(lines):
            raise CorpusIntegrityError(
                f"index row {row['turn_id']} points past the end of its chunk"
            )
        try:
            return json.loads(lines[row["traj_line"]])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusIntegrityError(
                f"{row['turn_id']}: chunk line is not a JSON record"
            ) from exc


# Rounds reuse one verified corpus per pin within a worker process.
_CORPORA: dict[tuple[str, str], AffineCorpus] = {}


def corpus_for_rule(rule):
    key = (rule["base_url"], rule["manifest_sha256"])
    corpus = _CORPORA.get(key)
    if corpus is None:
        corpus = _CORPORA[key] = AffineCorpus(rule)
    elif pool_key(corpus.rule) != pool_key(rule):
        # Same corpus, different filters: share verified objects, not the pool.
        corpus = AffineCorpus(rule, fetch=corpus.fetch, cache_dir=corpus.cache_dir)
    return corpus


# --------------------------------------------------------------------------
# Turn parsing
# --------------------------------------------------------------------------


def turn_messages(record, row):
    """Root-to-parent history of one reply node, and that reply's text.

    Raises CorpusIntegrityError for anything that is not a well-formed v4
    record agreeing with its index row."""
    if not isinstance(record, dict) or record.get("view") != VIEW_SPEC:
        raise CorpusIntegrityError(f"{row['turn_id']}: record is not {VIEW_SPEC}")
    if record.get("traj_id") != row["traj_id"]:
        raise CorpusIntegrityError(f"{row['turn_id']}: record traj_id differs")
    nodes, turns = record.get("nodes"), record.get("turns")
    if not isinstance(nodes, list) or not nodes or not isinstance(turns, list):
        raise CorpusIntegrityError(f"{row['turn_id']}: record lacks nodes or turns")
    metas = [
        t for t in turns if isinstance(t, dict) and t.get("turn_idx") == row["turn_idx"]
    ]
    if len(metas) != 1 or metas[0].get("node_id") != row["node_id"]:
        raise CorpusIntegrityError(f"{row['turn_id']}: turn meta disagrees with index")
    chain, seen, j = [], set(), row["node_id"]
    while j is not None:
        if type(j) is not int or not 0 <= j < len(nodes) or j in seen:
            raise CorpusIntegrityError(f"{row['turn_id']}: node graph broken at {j!r}")
        seen.add(j)
        node = nodes[j]
        if (
            not isinstance(node, dict)
            or node.get("role") not in ROLES
            or not isinstance(node.get("content"), str)
        ):
            raise CorpusIntegrityError(f"{row['turn_id']}: malformed node {j}")
        chain.append(node)
        j = node.get("parent")
    chain.reverse()
    reply, history = chain[-1], chain[:-1]
    if reply["role"] != "assistant":
        raise CorpusIntegrityError(
            f"{row['turn_id']}: target is not an assistant reply"
        )
    if not history or history[-1]["role"] != "user":
        raise CorpusIntegrityError(f"{row['turn_id']}: history does not end on user")
    if not any(m["content"].strip() for m in history):
        raise CorpusIntegrityError(f"{row['turn_id']}: history is empty")
    if sum(len(m["content"]) for m in history) != row["n_prefix_chars"]:
        raise CorpusIntegrityError(
            f"{row['turn_id']}: prefix length disagrees with index"
        )
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    return messages, reply["content"]


def candidate_for_row(row, corpus, formatter, rule, context):
    """Render one indexed turn, or None when it is outside the workload."""
    messages, reference = turn_messages(corpus.record(row), row)
    prompt = formatter.render(messages)
    ids = formatter.encode(prompt)
    group = input_group(len(ids))
    if (
        group is None
        or len(ids) > context["max_input_tokens"]
        or len(ids) + rule["max_tokens"] + context["engine_reserve"]
        > context["max_model_len"]
    ):
        return None
    return {
        "turn_id": row["turn_id"],
        "chunk_key": row["chunk_key"],
        "chunk_sha256": "sha256:" + corpus.chunk_sha256(row["chunk_key"]),
        "traj_line": row["traj_line"],
        "node_id": row["node_id"],
        "source": row["source"],
        "action_kind": row["action_kind"],
        "phase": row["phase"],
        "history_messages": len(messages),
        "reference_sha256": "sha256:" + hashlib.sha256(reference.encode()).hexdigest(),
        "prompt": prompt,
        "input_tokens": len(ids),
        "input_ids_sha256": token_ids_sha256(ids),
        "input_length_group": group,
    }


def ordered_turn_ids(turn_ids, seed):
    return sorted(
        turn_ids, key=lambda t: (hashlib.sha256(f"{seed}:{t}".encode()).digest(), t)
    )


# --------------------------------------------------------------------------
# Trace generation and replay
# --------------------------------------------------------------------------


def request_for_candidate(candidate, rule, index, *, generation_seed=""):
    return {
        "id": f"af-{index:03d}",
        "arrival_offset_ms": index * rule["request_interval_ms"],
        "prompt": candidate["prompt"],
        "max_tokens": rule["max_tokens"],
        "sampling": generation_sampling(
            rule, seed_key=f"{generation_seed}:{index}" if generation_seed else ""
        ),
        "input_tokens": candidate["input_tokens"],
        "input_ids_sha256": candidate["input_ids_sha256"],
        "input_length_group": candidate["input_length_group"],
    }


def _check_formatter(formatter, rule):
    if (
        formatter is None
        or formatter.encode is None
        or not formatter.receipt.get("tokenizer")
    ):
        raise SamplerError(
            "affine_corpus requires a pinned chat formatter and tokenizer"
        )
    if (
        formatter.receipt.get("chat_template", {}).get("enable_thinking")
        is not rule["enable_thinking"]
    ):
        raise SamplerError("formatter thinking mode does not match sampling_rule")


def generate_affine_trace(
    *,
    rule,
    seed_hex,
    corpus,
    formatter,
    context,
    receipt,
    sample_seed_block,
    sample_seed_block_hash,
):
    _check_formatter(formatter, rule)
    validate_context(context)
    if not isinstance(corpus, AffineCorpus) or pool_key(corpus.rule) != pool_key(rule):
        raise SamplerError("affine_corpus sampling requires the rule's pinned corpus")
    seed = seed_hex.removeprefix("sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", seed):
        raise SamplerError("affine_corpus sampling requires a 64-character hex seed")
    pool = corpus.eligible_rows()
    if receipt is None:
        turn_ids = ordered_turn_ids(pool, seed)
    else:
        turn_ids = receipt.get("turn_ids")
        if (
            not isinstance(turn_ids, list)
            or len(turn_ids) != rule["n_prompts"]
            or len(set(turn_ids)) != len(turn_ids)
            or any(t not in pool for t in turn_ids)
        ):
            raise SamplerError("invalid affine_corpus receipt turn selections")
    groups = length_groups(rule["n_prompts"])
    remaining = {g["name"]: g["count"] for g in groups}
    selected, seen_prompts = [], set()
    for turn_id in turn_ids:
        try:
            candidate = candidate_for_row(
                pool[turn_id], corpus, formatter, rule, context
            )
        except (CorpusIntegrityError, PromptRenderError):
            # A fresh draw skips a bad turn; a replay must reproduce exactly.
            if receipt is not None:
                raise
            continue
        if candidate is None or candidate["input_ids_sha256"] in seen_prompts:
            if receipt is not None:
                raise SamplerError("affine_corpus receipt selected an ineligible turn")
            continue
        group = candidate["input_length_group"]
        if remaining[group] == 0:
            if receipt is not None:
                raise SamplerError("affine_corpus receipt exceeds its input tier quota")
            continue
        remaining[group] -= 1
        selected.append(candidate)
        seen_prompts.add(candidate["input_ids_sha256"])
        if len(selected) == rule["n_prompts"]:
            break
    if len(selected) != rule["n_prompts"]:
        raise SamplerError(
            f"insufficient eligible Affine turns; missing by input tier: {remaining}"
        )
    requests = [
        request_for_candidate(item, rule, i, generation_seed=seed)
        for i, item in enumerate(selected)
    ]
    workload = workload_contract(rule, context)
    if "temperature_range" in rule:
        workload["generation_seed"] = seed
    validate_affine_trace(requests, workload)
    body = encode_trace(
        {
            "schema_version": 1,
            "meta": {"name": f"affine-corpus-{seed[:12]}", "sampling": workload},
            "requests": requests,
        }
    )
    sha = "sha256:" + hashlib.sha256(body).hexdigest()
    result = {
        **rule,
        **formatter.receipt,
        "context": context,
        "sample_seed_block": sample_seed_block,
        "sample_seed_block_hash": sample_seed_block_hash.strip().lower(),
        "seed_hex": seed,
        "turn_ids": [item["turn_id"] for item in selected],
        "requests": [
            {
                "request_id": f"af-{i:03d}",
                "max_tokens": rule["max_tokens"],
                **{k: v for k, v in item.items() if k != "prompt"},
            }
            for i, item in enumerate(selected)
        ],
        "sampled_trace_sha256": sha,
    }
    if receipt is not None and result != receipt:
        raise SamplerError("sampling receipt does not reproduce the selected trace")
    return SampledTrace(
        sha,
        body,
        seed,
        sample_seed_block,
        sample_seed_block_hash.strip().lower(),
        (),
        result,
    )


def workload_contract(rule, context):
    workload = {
        "workload": RULE_TYPE,
        "algo_version": ALGO_VERSION,
        "enable_thinking": rule["enable_thinking"],
        "context": context,
        "request_interval_ms": rule["request_interval_ms"],
        "max_tokens": rule["max_tokens"],
        "length_groups": length_groups(rule["n_prompts"]),
    }
    workload.update(generation_fields(rule))
    return workload


def validate_affine_trace(requests, sampling):
    if (
        sampling.get("workload") != RULE_TYPE
        or sampling.get("algo_version") != ALGO_VERSION
        or not isinstance(sampling.get("enable_thinking"), bool)
    ):
        raise SamplerError("invalid affine_corpus trace mode")
    context = sampling.get("context")
    validate_context(context)
    generation_fields(sampling)
    generation_seed = sampling.get("generation_seed", "")
    if "temperature_range" in sampling and not re.fullmatch(
        r"[0-9a-f]{64}", str(generation_seed)
    ):
        raise SamplerError("invalid affine_corpus generation seed")
    for key in ("request_interval_ms", "max_tokens"):
        value = sampling.get(key)
        if type(value) is not int or value < (0 if key == "request_interval_ms" else 1):
            raise SamplerError(f"invalid affine_corpus {key}")
    groups = length_groups(len(requests))
    if (
        len(requests) < 4
        or len(requests) % 4
        or sampling.get("length_groups") != groups
    ):
        raise SamplerError("invalid affine_corpus input tier contract")
    counts = {g["name"]: 0 for g in groups}
    for i, request in enumerate(requests):
        settings = generation_sampling(
            sampling, seed_key=f"{generation_seed}:{i}" if generation_seed else ""
        )
        size = request.get("input_tokens")
        if (
            type(size) is not int
            or size < 1
            or size > context["max_input_tokens"]
            or request.get("max_tokens") != sampling["max_tokens"]
            or size + request["max_tokens"] + context["engine_reserve"]
            > context["max_model_len"]
            or not isinstance(request.get("prompt"), str)
            or not request["prompt"].strip()
            or request.get("prompt_token_ids") is not None
            or not _SHA_RE.fullmatch(str(request.get("input_ids_sha256")))
            or request.get("sampling")
            not in (settings, {**settings, "ignore_eos": False})
            or request.get("arrival_offset_ms") != i * sampling["request_interval_ms"]
        ):
            raise SamplerError("invalid affine_corpus request or generation settings")
        group = input_group(size)
        if group is None or request.get("input_length_group") != group:
            raise SamplerError("affine_corpus request is outside its input tier")
        counts[group] += 1
    if counts != {g["name"]: g["count"] for g in groups}:
        raise SamplerError("affine_corpus trace does not fill every input tier")


def rule_from_receipt(receipt):
    """The sampling rule a receipt was drawn under, for worker replay."""
    return {key: receipt[key] for key in RULE_FIELDS if key in receipt}


def preflight_affine_campaign(rule, bench, engine):
    from bench.sampler import build_prompt_formatter, generate_trace

    require_qualification(rule, bench, engine)
    model = bench["model"]
    return generate_trace(
        rule=rule,
        seed_hex="0" * 64,
        row_fetcher=corpus_for_rule(rule),
        prompt_formatter=build_prompt_formatter(
            rule, model_repo=model["hf_repo"], model_revision=model["hf_revision"]
        ),
        sampling_context=sampling_context_for_campaign(bench, engine),
    )


def write_gzip_jsonl(records):
    """Test and fixture helper: (gzip bytes, sha256 of the uncompressed JSONL)."""
    raw = "".join(
        json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in records
    ).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(raw)
    return buffer.getvalue(), hashlib.sha256(raw).hexdigest()
