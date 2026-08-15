"""DB round-trip for the campaign engine profile (PAR-55).

Binds to the Neon test branch via ``e2e_db`` — never the partner/main database.
The ``engine`` column is additive and nullable, so the null case here also proves
rows written before engine profiles existed still load.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytestmark = pytest.mark.e2e

from campaign.engine import SGLANG_ENGINE, VLLM_ENGINE
from campaign.manifest import build_manifest
from campaign.models import SLA
from campaign.store import get_campaign, insert_campaign, insert_profile
from e2e_db import cleanup_e2e_rows, require_e2e_database_url

# SGLang v0.5.17 — the tag PAR-54 verified against lmsysorg/sglang.
SGLANG_COMMIT = "29481685462732237d80d86076d6563e1f658102"


@pytest.fixture(autouse=True)
def _bind_e2e_database(monkeypatch: pytest.MonkeyPatch):
    """Point store/connection code at the Neon test branch for this module."""
    url = require_e2e_database_url()
    monkeypatch.setenv("PARETON_DATABASE_URL", url)
    import db.connection as conn

    monkeypatch.setattr(conn, "DATABASE_URL", url)
    yield
    cleanup_e2e_rows()


def _insert(engine):
    campaign_id = uuid4()
    profile_id = insert_profile("e2e-engine", {"fixture": True})
    manifest = build_manifest(
        campaign_id=campaign_id,
        profile_id=profile_id,
        baseline_repo="https://github.com/sgl-project/sglang.git",
        baseline_commit=SGLANG_COMMIT,
        base_image_digest="sha256:" + "d" * 64,
        gpu_skus=["B300"],
        workload_trace_sha256="sha256:" + "e" * 64,
        workload_trace_url="https://cdn.test/trace.json",
        sla=SLA(),
        scoring_config_sha256=None,
        scoring_config_url=None,
        allowed_paths=["python/sglang/**"],
        denied_paths=["rust/**"],
        priority_metric="throughput",
        success_threshold=">=10% at SLA",
        status="draft",
        engine=engine,
    )
    insert_campaign(manifest)
    return manifest, get_campaign(campaign_id)


def test_engine_round_trips_through_postgres():
    written, read_back = _insert(SGLANG_ENGINE)
    assert read_back is not None
    assert read_back.engine == SGLANG_ENGINE
    # The stored profile must reproduce the hash it was pinned with — that is
    # what makes engine part of the frozen manifest rather than loose metadata.
    assert read_back.manifest_hash == written.manifest_hash
    assert read_back.to_public_dict()["engine"] == SGLANG_ENGINE


def test_absent_engine_round_trips_as_null():
    written, read_back = _insert(None)
    assert read_back is not None
    assert read_back.engine is None
    assert read_back.manifest_hash == written.manifest_hash


def test_vllm_engine_round_trips():
    _written, read_back = _insert(VLLM_ENGINE)
    assert read_back is not None
    assert read_back.engine == VLLM_ENGINE
