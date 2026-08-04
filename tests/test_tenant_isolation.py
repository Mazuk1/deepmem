"""Cross-account isolation regression tests.

Locks down the account_id boundary in the engine: the same user_id string
("alice") appearing under two different accounts must NEVER cross-contaminate.
(The HTTP layer runs in open mode with account_id=None, but the engine still
supports account_id scoping, so this guards the boundary directly.)

- writes don't get dedup'd against the other account
- get / update / delete by id from account B can't touch account A's record
- delete_all and reset on account A leave account B's data alone
- semantic cache buckets by (account_id, user_id), not just user_id
- history rows are read-fenced by account_id

Mirrors the smoke-test scenario verified manually before commit e4bf0c2.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid

import pytest

from deepmem.interface import Tenant


# ── Shared fixture ─────────────────────────────────────────────────────
# Class-scoped so each test class gets a clean Qdrant + sqlite, and we
# don't pay the embedder/Qdrant init cost per test method.

@pytest.fixture(scope="class")
def store_with_history():
    test_qdrant = f"./data/test_iso_qdrant_{uuid.uuid4().hex[:8]}"
    test_db = f"./data/test_iso_{uuid.uuid4().hex[:8]}.db"

    from deepmem.config import config as _cfg
    from deepmem.history_store import HistoryStore
    from deepmem.vector_store import VectorStore

    history = HistoryStore(db_path=test_db)
    store = VectorStore(
        qdrant_path=test_qdrant,
        embedding_dims=_cfg.embedding_dims,
        bge_m3_path=_cfg.bge_m3_path,
        history_store=history,
    )
    yield store, history

    history.close()
    # Windows: WAL keeps a handle around briefly after close().
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


def _tenant(account: str, user: str = "alice") -> Tenant:
    return Tenant(user_id=user, account_id=account)


# ── Cross-account write/read isolation ─────────────────────────────────

class TestCrossAccountIsolation:
    """Two accounts sharing the same user_id must remain independent."""

    @pytest.mark.asyncio
    async def test_dedup_does_not_cross_accounts(self, store_with_history):
        """Account B writing the same fact text as A should NOT be deduped."""
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_A_{suffix}")
        b = _tenant(f"acct_B_{suffix}")
        text = f"shared fact text {suffix}"

        a_ids = await store.add(
            [{"role": "user", "content": text}], a, infer=False,
        )
        b_ids = await store.add(
            [{"role": "user", "content": text}], b, infer=False,
        )

        assert len(a_ids) == 1
        assert len(b_ids) == 1
        assert a_ids[0] != b_ids[0]

        # Each account sees exactly its own one row.
        assert len(await store.list(a)) == 1
        assert len(await store.list(b)) == 1

    @pytest.mark.asyncio
    async def test_list_does_not_cross_accounts(self, store_with_history):
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_listA_{suffix}")
        b = _tenant(f"acct_listB_{suffix}")

        await store.add([{"role": "user", "content": "A row"}], a, infer=False)
        await store.add([{"role": "user", "content": "B row 1"}], b, infer=False)
        await store.add([{"role": "user", "content": "B row 2"}], b, infer=False)

        a_rows = await store.list(a)
        b_rows = await store.list(b)
        assert len(a_rows) == 1 and "A row" in a_rows[0].memory
        b_texts = sorted(r.memory for r in b_rows)
        assert len(b_texts) == 2
        assert "B row 1" in b_texts[0] and "B row 2" in b_texts[1]

    @pytest.mark.asyncio
    async def test_get_returns_none_across_accounts(self, store_with_history):
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_getA_{suffix}")
        b = _tenant(f"acct_getB_{suffix}")

        ids = await store.add(
            [{"role": "user", "content": "private to A"}], a, infer=False,
        )
        assert ids

        # B knows the id but is on a different account — must be invisible.
        assert await store.get(ids[0], b) is None
        # A still sees it.
        assert "private to A" in (await store.get(ids[0], a)).memory

    @pytest.mark.asyncio
    async def test_update_blocked_across_accounts(self, store_with_history):
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_updA_{suffix}")
        b = _tenant(f"acct_updB_{suffix}")

        ids = await store.add(
            [{"role": "user", "content": "A's data"}], a, infer=False,
        )
        # B cannot tamper with A's row.
        ok = await store.update(ids[0], "B overwrote it", b)
        assert ok is False
        # And the value is unchanged.
        got = await store.get(ids[0], a)
        assert "A's data" in got.memory

    @pytest.mark.asyncio
    async def test_delete_blocked_across_accounts(self, store_with_history):
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_delA_{suffix}")
        b = _tenant(f"acct_delB_{suffix}")

        ids = await store.add(
            [{"role": "user", "content": "A's data"}], a, infer=False,
        )
        assert await store.delete(ids[0], b) is False
        # Still alive for A.
        assert await store.get(ids[0], a) is not None

    @pytest.mark.asyncio
    async def test_delete_all_does_not_touch_other_account(self, store_with_history):
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_dallA_{suffix}")
        b = _tenant(f"acct_dallB_{suffix}")

        await store.add([{"role": "user", "content": "A1"}], a, infer=False)
        await store.add([{"role": "user", "content": "A2"}], a, infer=False)
        await store.add([{"role": "user", "content": "B1"}], b, infer=False)

        nuked = await store.delete_all(a)
        assert nuked == 2
        # B's row survives unscathed.
        b_rows = await store.list(b)
        assert len(b_rows) == 1 and "B1" in b_rows[0].memory
        # A is empty now.
        assert await store.list(a) == []

    @pytest.mark.asyncio
    async def test_search_does_not_cross_accounts(self, store_with_history):
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_searchA_{suffix}")
        b = _tenant(f"acct_searchB_{suffix}")

        await store.add(
            [{"role": "user", "content": f"alpha-fact-{suffix}"}], a, infer=False,
        )
        await store.add(
            [{"role": "user", "content": f"alpha-fact-{suffix}"}], b, infer=False,
        )

        a_hits = await store.search(f"alpha-fact-{suffix}", a, top_k=10, threshold=0.0)
        b_hits = await store.search(f"alpha-fact-{suffix}", b, top_k=10, threshold=0.0)
        assert len(a_hits) == 1
        assert len(b_hits) == 1
        # Different ids — distinct rows for each account.
        assert a_hits[0].id != b_hits[0].id


# ── Reset isolation ────────────────────────────────────────────────────

class TestCrossAccountReset:
    """reset() must hard-delete only its own account's slice."""

    @pytest.mark.asyncio
    async def test_reset_only_wipes_own_account(self, store_with_history):
        store, history = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_resetA_{suffix}")
        b = _tenant(f"acct_resetB_{suffix}")

        a_ids = await store.add(
            [{"role": "user", "content": "A's data"}], a, infer=False,
        )
        b_ids = await store.add(
            [{"role": "user", "content": "B's data"}], b, infer=False,
        )

        result = await store.reset(a)
        assert result["deleted_vectors"] == 1
        assert result["deleted_history_events"] >= 1

        # A is gone (vectors + history).
        assert await store.list(a) == []
        assert await store.history(a_ids[0], a) == []

        # B unchanged.
        b_rows = await store.list(b)
        assert len(b_rows) == 1 and "B's data" in b_rows[0].memory
        b_events = await store.history(b_ids[0], b)
        assert any(e["event"] == "ADD" for e in b_events)

    @pytest.mark.asyncio
    async def test_reset_legacy_user_does_not_touch_account_scoped(self, store_with_history):
        """Legacy (account_id=None) writes must not be wiped by an account-scoped reset."""
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        legacy = Tenant(user_id=f"legacy_{suffix}")  # account_id=None
        scoped = Tenant(user_id=f"legacy_{suffix}", account_id=f"acct_{suffix}")

        await store.add(
            [{"role": "user", "content": "legacy row"}], legacy, infer=False,
        )
        await store.add(
            [{"role": "user", "content": "scoped row"}], scoped, infer=False,
        )

        result = await store.reset(scoped)
        assert result["deleted_vectors"] == 1

        # Legacy row survives — different bucket.
        legacy_rows = await store.list(legacy)
        assert len(legacy_rows) == 1 and "legacy row" in legacy_rows[0].memory


# ── History + cache isolation ──────────────────────────────────────────

class TestCrossAccountHistoryAndCache:
    @pytest.mark.asyncio
    async def test_history_read_fenced_by_account(self, store_with_history):
        """Knowing the memory_id is not enough — account_id must match too."""
        store, _ = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_histA_{suffix}")
        b = _tenant(f"acct_histB_{suffix}")

        ids = await store.add(
            [{"role": "user", "content": "A's audit-trail row"}], a, infer=False,
        )

        a_events = await store.history(ids[0], a)
        b_events = await store.history(ids[0], b)
        assert len(a_events) == 1
        assert b_events == []

    @pytest.mark.asyncio
    async def test_history_for_user_scoped_by_account(self, store_with_history):
        """history_for_user (used by /v1/history) must filter by account too."""
        store, history = store_with_history
        suffix = uuid.uuid4().hex[:6]
        a = _tenant(f"acct_huserA_{suffix}")
        b = _tenant(f"acct_huserB_{suffix}")

        await store.add([{"role": "user", "content": "A1"}], a, infer=False)
        await store.add([{"role": "user", "content": "A2"}], a, infer=False)
        await store.add([{"role": "user", "content": "B1"}], b, infer=False)

        a_rows = history.history_for_user(
            a.user_id, account_id=a.account_id,
        )
        b_rows = history.history_for_user(
            b.user_id, account_id=b.account_id,
        )
        # Both accounts share user_id "alice" — only the account_id fence
        # keeps them separate.
        assert len(a_rows) == 2
        assert len(b_rows) == 1
        # _raw_facts prefixes content with "user: " / "assistant: "; check via substring.
        assert all(("A1" in r["new_memory"]) or ("A2" in r["new_memory"]) for r in a_rows)
        assert "B1" in b_rows[0]["new_memory"]

    @pytest.mark.asyncio
    async def test_semantic_cache_bucketed_by_account(self):
        """Cache value stored under (acct_A, alice) must not be visible to (acct_B, alice)."""
        from deepmem.engine.semantic_cache import SemanticCache

        cache = SemanticCache(similarity_threshold=0.98, ttl_seconds=300)
        # A unit vector of any size works for cosine similarity tests.
        vec = [1.0] + [0.0] * 1023
        messages = [{"role": "user", "content": "what's my favorite color"}]

        await cache.store(messages, vec, "alice", ["A's facts"], account_id="acct-A")
        await cache.store(messages, vec, "alice", ["B's facts"], account_id="acct-B")

        # Same query vector + same user_id, different account: no cross-talk.
        a_hit = await cache.check(messages, vec, "alice", account_id="acct-A")
        b_hit = await cache.check(messages, vec, "alice", account_id="acct-B")
        legacy_miss = await cache.check(messages, vec, "alice")  # account_id=None bucket

        assert a_hit == ["A's facts"]
        assert b_hit == ["B's facts"]
        assert legacy_miss is None

    @pytest.mark.asyncio
    async def test_semantic_cache_clear_scoped(self):
        """clear(user_id, account_id) wipes only that bucket."""
        from deepmem.engine.semantic_cache import SemanticCache

        cache = SemanticCache(similarity_threshold=0.98, ttl_seconds=300)
        vec = [1.0] + [0.0] * 1023
        messages = [{"role": "user", "content": "q"}]

        await cache.store(messages, vec, "alice", ["A facts"], account_id="acct-A")
        await cache.store(messages, vec, "alice", ["B facts"], account_id="acct-B")

        cache.clear(user_id="alice", account_id="acct-A")

        assert await cache.check(messages, vec, "alice", account_id="acct-A") is None
        # B's bucket untouched.
        assert await cache.check(messages, vec, "alice", account_id="acct-B") == ["B facts"]

    @pytest.mark.asyncio
    async def test_semantic_cache_clear_legacy_user_clears_all_buckets(self):
        """clear(user_id) without account_id wipes the user across every account
        bucket — preserves legacy reset semantics used during tenant reset."""
        from deepmem.engine.semantic_cache import SemanticCache

        cache = SemanticCache(similarity_threshold=0.98, ttl_seconds=300)
        vec = [1.0] + [0.0] * 1023
        messages = [{"role": "user", "content": "q"}]

        await cache.store(messages, vec, "alice", ["A"], account_id="acct-A")
        await cache.store(messages, vec, "alice", ["B"], account_id="acct-B")
        await cache.store(messages, vec, "bob", ["bob's"], account_id="acct-A")

        cache.clear(user_id="alice")

        assert await cache.check(messages, vec, "alice", account_id="acct-A") is None
        assert await cache.check(messages, vec, "alice", account_id="acct-B") is None
        # Bob is on a different user — survives.
        assert await cache.check(messages, vec, "bob", account_id="acct-A") == ["bob's"]
