"""Regression tests for the multi-agent permission groundwork fields.

Verifies that owner_id / visibility_scope / source_type / session_id are
stamped onto every stored payload on the write path and survive an
export -> import round-trip - WITHOUT any query/filter logic reading them
(behavior-neutral until a future permission layer is built on top).

Uses an in-process fake embedder so the suite runs without Google / BGE-M3
credentials or network access.
"""
import hashlib
import os
import shutil
import time
import uuid

import pytest

from deepmem.interface import Tenant


class _FakeEmbedder:
    """Deterministic in-process embedder - no network, no model download."""

    def __init__(self, dim=8):
        self.dim = dim

    def _vec(self, text: str):
        h = hashlib.md5(text.encode("utf-8")).digest()
        v = [(h[i % len(h)] / 255.0) for i in range(self.dim)]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    async def aembed(self, text, memory_action=None):
        return self._vec(text)

    async def aembed_batch(self, texts):
        return [self._vec(t) for t in texts]

    def embed(self, text):
        return self._vec(text)


@pytest.fixture
def store():
    test_path = f"./data/test_qdrant_gw_{uuid.uuid4().hex[:8]}"
    from deepmem.vector_store import VectorStore
    s = VectorStore(qdrant_path=test_path, embedding_dims=8)
    # Bypass the configured embedder (Google/BGE-M3) with a deterministic fake.
    s._embedder = _FakeEmbedder(dim=8)
    yield s
    time.sleep(0.3)
    if os.path.exists(test_path):
        try:
            shutil.rmtree(test_path)
        except PermissionError:
            pass


def _payload(store, mem_id):
    pts = store._client.retrieve(
        collection_name=store.collection_name, ids=[mem_id], with_payload=True,
    )
    return pts[0].payload


class TestGroundworkFields:
    """owner_id / visibility_scope / source_type / session_id are write-only
    payload groundwork today; queries must not depend on them."""

    @pytest.mark.asyncio
    async def test_explicit_fields_stamped_on_write(self, store):
        tenant = Tenant(
            user_id="alice",
            owner_id="agent-42",
            visibility_scope="team",
            source_type="agent",
            session_id="sess-7",
        )
        ids = await store.add(
            [{"role": "user", "content": "I prefer dark mode."}],
            tenant, infer=False,
        )
        assert len(ids) == 1
        payload = _payload(store, ids[0])
        assert payload["owner_id"] == "agent-42"
        assert payload["visibility_scope"] == "team"
        assert payload["source_type"] == "agent"
        assert payload["session_id"] == "sess-7"

    @pytest.mark.asyncio
    async def test_defaults_when_unset(self, store):
        # Tenant with no groundwork fields -> payload still carries every field
        # with safe defaults (visibility_scope / source_type are never null).
        tenant = Tenant(user_id="bob")
        ids = await store.add(
            [{"role": "user", "content": "I like Python."}],
            tenant, infer=False,
        )
        payload = _payload(store, ids[0])
        assert payload["owner_id"] is None
        assert payload["visibility_scope"] == "private"
        assert payload["source_type"] == "user"
        assert payload["session_id"] is None

    @pytest.mark.asyncio
    async def test_search_still_works_and_is_neutral(self, store):
        # Queries must ignore the new fields - search by content still hits.
        tenant = Tenant(user_id="carol", owner_id="agent-1",
                        visibility_scope="team", source_type="agent")
        await store.add(
            [{"role": "user", "content": "My timezone is UTC+8."}],
            tenant, infer=False,
        )
        results = await store.search("timezone", tenant, top_k=3)
        assert len(results) == 1
        # The groundwork fields surface in result metadata but are not filtered on.
        assert results[0].metadata.get("owner_id") == "agent-1"

    @pytest.mark.asyncio
    async def test_export_import_round_trips_fields(self, store):
        tenant = Tenant(
            user_id="dave",
            owner_id="agent-9",
            visibility_scope="org",
            source_type="agent",
            session_id="sess-1",
        )
        await store.add(
            [{"role": "user", "content": "I drive an EV."}],
            tenant, infer=False,
        )

        exported = await store.export(tenant)
        assert len(exported) == 1
        # Groundwork fields ride inside the metadata sub-dict on export.
        meta = exported[0]["metadata"]
        assert meta.get("owner_id") == "agent-9"
        assert meta.get("visibility_scope") == "org"
        assert meta.get("source_type") == "agent"
        assert meta.get("session_id") == "sess-1"

        # Re-import under a fresh tenant; the fields must come back.
        tenant2 = Tenant(user_id="dave2")
        result = await store.import_records(tenant2, exported, skip_existing=False)
        assert result["inserted"] == 1

        listed = await store.list(tenant2, limit=10)
        assert len(listed) == 1
        payload = _payload(store, listed[0].id)
        assert payload["owner_id"] == "agent-9"
        assert payload["visibility_scope"] == "org"
        assert payload["source_type"] == "agent"
        assert payload["session_id"] == "sess-1"
