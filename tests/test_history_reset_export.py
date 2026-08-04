"""Tests for B (Memory History) and C (Reset / Export / Import).

Uses a fresh VectorStore wired to a per-class HistoryStore so we can
verify ADD/UPDATE/DELETE events land in sqlite, and round-trip the
export/import path.
"""
import os
import shutil
import uuid
import pytest

from deepmem.interface import Tenant


@pytest.fixture(scope="class")
def store_with_history():
    import time
    test_qdrant = f"./data/test_hist_qdrant_{uuid.uuid4().hex[:8]}"
    test_db = f"./data/test_hist_{uuid.uuid4().hex[:8]}.db"

    from deepmem.config import config as _cfg
    from deepmem.vector_store import VectorStore
    from deepmem.history_store import HistoryStore

    history = HistoryStore(db_path=test_db)
    store = VectorStore(
        qdrant_path=test_qdrant,
        embedding_dims=_cfg.embedding_dims,
        bge_m3_path=_cfg.bge_m3_path,
        history_store=history,
    )
    yield store, history

    history.close()
    time.sleep(0.3)
    if os.path.exists(test_qdrant):
        try:
            shutil.rmtree(test_qdrant)
        except PermissionError:
            pass
    for f in (test_db, test_db + "-wal", test_db + "-shm"):
        if os.path.exists(f):
            try:
                os.remove(f)
            except PermissionError:
                pass


class TestMemoryHistory:
    """B — every ADD/UPDATE/DELETE writes one history row."""

    @pytest.mark.asyncio
    async def test_add_writes_history_event(self, store_with_history):
        store, history = store_with_history
        tenant = Tenant(user_id=f"hist_add_{uuid.uuid4().hex[:6]}")
        ids = await store.add(
            [{"role": "user", "content": "fact about cats"}],
            tenant, infer=False,
        )
        assert ids
        events = await store.history(ids[0], tenant)
        assert len(events) == 1
        assert events[0]["event"] == "ADD"
        assert events[0]["new_memory"]
        assert events[0]["prev_memory"] is None

    @pytest.mark.asyncio
    async def test_update_appends_event_with_prev(self, store_with_history):
        store, _ = store_with_history
        tenant = Tenant(user_id=f"hist_upd_{uuid.uuid4().hex[:6]}")
        ids = await store.add(
            [{"role": "user", "content": "favorite color is blue"}],
            tenant, infer=False,
        )
        ok = await store.update(ids[0], "favorite color is green", tenant)
        assert ok

        events = await store.history(ids[0], tenant)
        assert len(events) == 2
        assert [e["event"] for e in events] == ["ADD", "UPDATE"]
        assert events[1]["prev_memory"] and "blue" in events[1]["prev_memory"]
        assert "green" in events[1]["new_memory"]

    @pytest.mark.asyncio
    async def test_delete_appends_event(self, store_with_history):
        store, _ = store_with_history
        tenant = Tenant(user_id=f"hist_del_{uuid.uuid4().hex[:6]}")
        ids = await store.add(
            [{"role": "user", "content": "temporary fact"}],
            tenant, infer=False,
        )
        await store.delete(ids[0], tenant)
        events = await store.history(ids[0], tenant)
        assert [e["event"] for e in events] == ["ADD", "DELETE"]

    @pytest.mark.asyncio
    async def test_history_isolated_by_tenant(self, store_with_history):
        store, history = store_with_history
        tenant_a = Tenant(user_id=f"hist_iso_a_{uuid.uuid4().hex[:6]}")
        tenant_b = Tenant(user_id=f"hist_iso_b_{uuid.uuid4().hex[:6]}")
        ids = await store.add(
            [{"role": "user", "content": "private to A"}],
            tenant_a, infer=False,
        )
        # Bob can't read Alice's history even with the id
        events = await store.history(ids[0], tenant_b)
        assert events == []


class TestReset:
    """C — reset hard-deletes vectors and history for a user."""

    @pytest.mark.asyncio
    async def test_reset_removes_vectors_and_history(self, store_with_history):
        store, _ = store_with_history
        tenant = Tenant(user_id=f"reset_{uuid.uuid4().hex[:6]}")

        ids = await store.add(
            [
                {"role": "user", "content": "first reset fact"},
                {"role": "user", "content": "second reset fact"},
            ],
            tenant, infer=False,
        )
        assert len(ids) == 2

        result = await store.reset(tenant)
        assert result["deleted_vectors"] == 2
        assert result["deleted_history_events"] >= 2  # at least the ADD events

        # Vectors gone
        listed = await store.list(tenant)
        assert listed == []
        # History gone for this user
        events = await store.history(ids[0], tenant)
        assert events == []

    @pytest.mark.asyncio
    async def test_reset_does_not_touch_other_users(self, store_with_history):
        store, _ = store_with_history
        keeper = Tenant(user_id=f"keeper_{uuid.uuid4().hex[:6]}")
        nuked = Tenant(user_id=f"nuked_{uuid.uuid4().hex[:6]}")

        keep_ids = await store.add(
            [{"role": "user", "content": "keep me around"}],
            keeper, infer=False,
        )
        await store.add(
            [{"role": "user", "content": "delete me"}],
            nuked, infer=False,
        )

        await store.reset(nuked)

        survivors = await store.list(keeper)
        assert len(survivors) == 1
        events = await store.history(keep_ids[0], keeper)
        assert any(e["event"] == "ADD" for e in events)


class TestExportImport:
    """C — export to JSON-friendly dicts, import re-ingests."""

    @pytest.mark.asyncio
    async def test_export_returns_payload_fields(self, store_with_history):
        store, _ = store_with_history
        tenant = Tenant(user_id=f"exp_{uuid.uuid4().hex[:6]}",
                        agent_id="agent-X")

        await store.add(
            [{"role": "user", "content": "exportable fact one"}],
            tenant, infer=False,
        )
        await store.add(
            [{"role": "user", "content": "exportable fact two"}],
            tenant, infer=False,
        )

        records = await store.export(tenant)
        assert len(records) == 2
        for r in records:
            assert r["user_id"] == tenant.user_id
            assert r["agent_id"] == "agent-X"
            assert r["memory"]
            assert r["hash"]

    @pytest.mark.asyncio
    async def test_import_round_trip_with_skip_existing(self, store_with_history):
        store, _ = store_with_history
        src = Tenant(user_id=f"src_{uuid.uuid4().hex[:6]}")
        dst = Tenant(user_id=f"dst_{uuid.uuid4().hex[:6]}")

        await store.add(
            [{"role": "user", "content": "round-trip fact A"},
             {"role": "user", "content": "round-trip fact B"}],
            src, infer=False,
        )
        records = await store.export(src)
        assert len(records) == 2

        # First import: everything new
        first = await store.import_records(dst, records, skip_existing=True)
        assert first["inserted"] == 2
        assert first["skipped"] == 0

        # Second import of the same dump: all skipped
        second = await store.import_records(dst, records, skip_existing=True)
        assert second["inserted"] == 0
        assert second["skipped"] == 2

        # And dst now has the facts listed
        listed = await store.list(dst)
        assert len(listed) == 2

    @pytest.mark.asyncio
    async def test_import_skip_existing_false_inserts_anyway(self, store_with_history):
        store, _ = store_with_history
        src = Tenant(user_id=f"dup_src_{uuid.uuid4().hex[:6]}")

        await store.add(
            [{"role": "user", "content": "duplicate-friendly fact"}],
            src, infer=False,
        )
        records = await store.export(src)

        # Re-import without skip: a fresh row gets added (new uuid).
        result = await store.import_records(src, records, skip_existing=False)
        assert result["inserted"] == 1

        listed = await store.list(src)
        assert len(listed) == 2  # original + re-imported
