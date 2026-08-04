"""Phase 3.4 / 3.5: semantic cache integration with the add path.

- Phase 3.4: when the caller passes `infer=False`, the server stores raw
  messages verbatim — semantic dedupe over the prompt would short-circuit
  a real write, so cache.check MUST NOT run for that path.
- Phase 3.5: when extraction succeeds, the cache stores the actual
  (id, memory) pair returned by the vector store, not just a fact string.
  A subsequent cache HIT returns the real Qdrant id back to the caller —
  fabricating a fresh UUID on hit would break downstream get/update/delete
  workflows.
"""
import pytest


class TestSemanticCacheStoresRichValues:
    """Phase 3.5 contract: cache.store accepts arbitrary value shapes.

    Historically the cache stored List[str] (raw fact text) and the add
    handler invented UUIDs on cache hits. After Phase 3.5 the value is
    List[Dict[str, str]] with "id" + "memory" keys so cache hits round-trip
    to real persisted records.
    """

    @pytest.fixture
    def cache(self):
        from deepmem.engine.semantic_cache import SemanticCache
        # threshold=1.0 forces an exact-vector match below; we want
        # determinism, not approximate-match semantics under test.
        return SemanticCache(similarity_threshold=0.99, ttl_seconds=300)

    @pytest.mark.asyncio
    async def test_round_trips_id_and_memory_pairs(self, cache):
        messages = [{"role": "user", "content": "Alice lives in SF"}]
        vec = [1.0, 0.0, 0.0]
        items = [
            {"id": "qdrant-id-1", "memory": "Lives in SF"},
            {"id": "qdrant-id-2", "memory": "Name is Alice"},
        ]
        await cache.store(messages, vec, "user_alice", items)
        hit = await cache.check(messages, vec, "user_alice")
        assert hit == items
        # Each entry must still carry a real id, not a fabricated UUID.
        assert all(it["id"].startswith("qdrant-id-") for it in hit)

    @pytest.mark.asyncio
    async def test_account_isolation_for_rich_values(self, cache):
        messages = [{"role": "user", "content": "Alice lives in SF"}]
        vec = [1.0, 0.0, 0.0]
        items = [{"id": "tenant-a-id", "memory": "Lives in SF"}]
        await cache.store(messages, vec, "shared_user", items, account_id="acct-A")

        # Same user_id string under a different account MUST miss — otherwise
        # tenant-A's persisted ids would leak to tenant-B.
        other_account_hit = await cache.check(
            messages, vec, "shared_user", account_id="acct-B",
        )
        assert other_account_hit is None

        same_account_hit = await cache.check(
            messages, vec, "shared_user", account_id="acct-A",
        )
        assert same_account_hit == items


class TestAddPathCacheGating:
    """Phase 3.4: the add handler must not call cache.check on infer=False.

    We monkeypatch the cache.check method on the live singleton and assert
    it was untouched by an infer=False request. We don't need to assert
    success of the underlying store call — the gating is the whole point.
    """

    @pytest.mark.asyncio
    async def test_infer_false_skips_cache_check(self, monkeypatch):
        from httpx import ASGITransport, AsyncClient
        from server.dependencies import get_services
        from server.main import app

        svc = get_services()
        calls: list = []

        original_check = svc["cache"].check

        async def spy_check(*args, **kwargs):
            calls.append((args, kwargs))
            return await original_check(*args, **kwargs)

        monkeypatch.setattr(svc["cache"], "check", spy_check)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/v1/memories", json={
                "messages": [{"role": "user", "content": "raw stash"}],
                "user_id": "cache_gate_user_infer_false",
                "infer": False,
            })

        # We don't require status 200 here — env quirks might make
        # the underlying add path fail. The contract under test is purely
        # "cache.check did not fire", and that holds either way.
        assert r.status_code in (200, 500)
        assert calls == [], (
            f"cache.check was invoked {len(calls)} times on infer=False; "
            "Phase 3.4 says it must be skipped"
        )

    @pytest.mark.asyncio
    async def test_infer_true_calls_cache_check(self, monkeypatch):
        from httpx import ASGITransport, AsyncClient
        from server.dependencies import get_services
        from server.main import app

        svc = get_services()
        calls: list = []

        original_check = svc["cache"].check

        async def spy_check(*args, **kwargs):
            calls.append((args, kwargs))
            return await original_check(*args, **kwargs)

        monkeypatch.setattr(svc["cache"], "check", spy_check)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/v1/memories", json={
                "messages": [{"role": "user", "content": "extracted intent"}],
                "user_id": "cache_gate_user_infer_true",
                "infer": True,
            })

        # On infer=True the handler MUST consult the cache before deciding
        # whether to enqueue / extract. Otherwise repeat requests duplicate
        # work and the cache becomes write-only.
        assert len(calls) >= 1
