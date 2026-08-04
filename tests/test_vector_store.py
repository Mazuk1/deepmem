import os
import shutil
import uuid
import pytest

from deepmem.interface import Tenant


@pytest.fixture(scope="class")
def store():
    """Create one VectorStore for all tests in this class (avoids Qdrant lock conflicts)."""
    import time
    test_path = f"./data/test_qdrant_{uuid.uuid4().hex[:8]}"
    from deepmem.config import config as _cfg
    from deepmem.vector_store import VectorStore
    store = VectorStore(
        qdrant_path=test_path,
        embedding_dims=_cfg.embedding_dims,
        bge_m3_path=_cfg.bge_m3_path,
    )
    yield store
    time.sleep(0.5)
    if os.path.exists(test_path):
        try:
            shutil.rmtree(test_path)
        except PermissionError:
            pass


class TestVectorStore:

    @pytest.fixture
    def tenant_alice(self):
        return Tenant(user_id="alice_test")

    @pytest.fixture
    def tenant_bob(self):
        return Tenant(user_id="bob_test")

    # ── Basic add & search (infer=False, no LLM) ──────────────────────

    @pytest.mark.asyncio
    async def test_add_and_search_single_memory(self, store, tenant_alice):
        messages = [
            {"role": "user", "content": "My name is Alice, I live in San Francisco."},
        ]
        ids = await store.add(messages, tenant_alice, infer=False)
        assert len(ids) > 0

        results = await store.search("Where does Alice live?", tenant_alice, top_k=3)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_add_and_search_multiple_memories(self, store, tenant_alice):
        messages = [
            {"role": "user", "content": "My name is Alice. I work at Google as an engineer."},
            {"role": "assistant", "content": "Nice! Google is a great place to work."},
            {"role": "user", "content": "I also love hiking and Italian food."},
        ]
        ids = await store.add(messages, tenant_alice, infer=False)
        assert len(ids) > 0

        results = await store.search("What is Alice's job?", tenant_alice, top_k=3)
        assert len(results) > 0

    # ── Tenant isolation ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, store, tenant_alice, tenant_bob):
        await store.add(
            [{"role": "user", "content": "My name is Alice, I like cats."}],
            tenant_alice, infer=False,
        )
        await store.add(
            [{"role": "user", "content": "My name is Bob, I like dogs."}],
            tenant_bob, infer=False,
        )

        alice_results = await store.search("What do I like?", tenant_alice, top_k=5)
        alice_memories = " ".join(r.memory for r in alice_results)
        assert "cat" in alice_memories.lower()
        assert "dog" not in alice_memories.lower()

        bob_results = await store.search("What do I like?", tenant_bob, top_k=5)
        bob_memories = " ".join(r.memory for r in bob_results)
        assert "dog" in bob_memories.lower()
        assert "cat" not in bob_memories.lower()

    # ── Delete & soft-delete ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_delete_single_memory(self, store, tenant_alice):
        messages = [
            {"role": "user", "content": "My name is Alice and my secret code is 99999."},
        ]
        ids = await store.add(messages, tenant_alice, infer=False)
        assert len(ids) > 0

        deleted = await store.delete(ids[0], tenant_alice)
        assert deleted is True

        results = await store.search("secret code", tenant_alice, top_k=10)
        assert all(r.id != ids[0] for r in results)

    @pytest.mark.asyncio
    async def test_delete_all_for_tenant(self, store, tenant_alice):
        await store.add(
            [{"role": "user", "content": "My name is Alice and I live in Beijing."}],
            tenant_alice, infer=False,
        )
        await store.add(
            [{"role": "user", "content": "I work as a product manager at ByteDance."}],
            tenant_alice, infer=False,
        )

        count = await store.delete_all(tenant_alice)
        assert count > 0

        results = await store.search("Alice", tenant_alice, top_k=10)
        assert len(results) == 0

    # ── Edge cases ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_search_empty_for_nonexistent_user(self, store):
        tenant = Tenant(user_id="nonexistent_user_xyz")
        results = await store.search("anything", tenant, top_k=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, store, tenant_alice):
        messages = [
            {"role": "user", "content": "My favorite color is blue."},
        ]
        ids = await store.add(messages, tenant_alice, metadata={"source": "test", "priority": 1}, infer=False)
        assert len(ids) > 0

        results = await store.search("favorite color", tenant_alice, top_k=5)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_memory_not_cross_contaminate_after_delete(self, store, tenant_alice, tenant_bob):
        await store.add(
            [{"role": "user", "content": "I am Alice. My favorite book is Dune."}],
            tenant_alice, infer=False,
        )
        await store.add(
            [{"role": "user", "content": "I am Bob. My favorite book is The Hobbit."}],
            tenant_bob, infer=False,
        )

        count = await store.delete_all(tenant_alice)
        assert count > 0

        bob_results = await store.search("favorite book", tenant_bob, top_k=5)
        bob_text = " ".join(r.memory for r in bob_results)
        assert "Hobbit" in bob_text or "book" in bob_text.lower()
        assert "Dune" not in bob_text

    @pytest.mark.asyncio
    async def test_search_result_sorting(self, store, tenant_alice):
        await store.add([
            {"role": "user", "content": "I am Alice. I work at Google. I live in SF. I like cats."},
        ], tenant_alice, infer=False)

        results = await store.search("Where does Alice work?", tenant_alice, top_k=5)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    # ── New methods: get, update, list ────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_single_memory(self, store, tenant_alice):
        ids = await store.add(
            [{"role": "user", "content": "I love Python."}],
            tenant_alice, infer=False,
        )
        result = await store.get(ids[0], tenant_alice)
        assert result is not None
        assert result.id == ids[0]
        assert "Python" in result.memory

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, store, tenant_alice):
        result = await store.get("nonexistent-id", tenant_alice)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_wrong_tenant(self, store, tenant_alice, tenant_bob):
        ids = await store.add(
            [{"role": "user", "content": "Alice's secret."}],
            tenant_alice, infer=False,
        )
        result = await store.get(ids[0], tenant_bob)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_memory(self, store, tenant_alice):
        ids = await store.add(
            [{"role": "user", "content": "I like cats."}],
            tenant_alice, infer=False,
        )
        success = await store.update(ids[0], "I love dogs.", tenant_alice)
        assert success is True

        result = await store.get(ids[0], tenant_alice)
        assert result.memory == "I love dogs."

    @pytest.mark.asyncio
    async def test_list_memories(self, store, tenant_alice):
        await store.add(
            [{"role": "user", "content": "Memory A"}, {"role": "user", "content": "Memory B"}],
            tenant_alice, infer=False,
        )
        results = await store.list(tenant_alice, limit=10)
        assert len(results) >= 2
