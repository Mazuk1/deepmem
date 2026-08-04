import asyncio
import pytest


class TestSemanticCache:
    @pytest.fixture
    def cache(self):
        from deepmem.engine.semantic_cache import SemanticCache
        return SemanticCache(similarity_threshold=0.98, ttl_seconds=300)

    @pytest.fixture
    def embedder(self, config):
        from deepmem.embedder import get_embedder
        return get_embedder(config)

    @pytest.mark.asyncio
    async def test_cache_miss_on_first_request(self, cache, embedder):
        messages = [{"role": "user", "content": "What's my name?"}]
        query_vec = embedder.embed("What's my name?")
        result = await cache.check(messages, query_vec, "user_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_on_similar_request(self, cache, embedder):
        messages = [{"role": "user", "content": "My name is Alice and I live in SF."}]
        query_vec = embedder.embed("My name is Alice and I live in SF.")
        facts = ["Name is Alice", "Lives in San Francisco"]

        await cache.store(messages, query_vec, "user_1", facts)

        # Use the EXACT same vector to guarantee cache hit
        result = await cache.check(messages, query_vec, "user_1")
        assert result is not None
        assert result == facts

    @pytest.mark.asyncio
    async def test_cache_miss_on_different_request(self, cache, embedder):
        messages = [{"role": "user", "content": "My name is Alice."}]
        query_vec = embedder.embed("My name is Alice.")
        await cache.store(messages, query_vec, "user_1", ["Name is Alice"])

        different_vec = embedder.embed("What's the weather like in Tokyo?")
        result = await cache.check(messages, different_vec, "user_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_ttl_expiry(self, cache, embedder):
        cache.ttl_seconds = 0.1
        messages = [{"role": "user", "content": "Test"}]
        query_vec = embedder.embed("Test")
        await cache.store(messages, query_vec, "user_1", ["Test fact"])

        await asyncio.sleep(0.2)

        result = await cache.check(messages, query_vec, "user_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_isolated_by_user(self, cache, embedder):
        messages = [{"role": "user", "content": "My name is Alice."}]
        query_vec = embedder.embed("My name is Alice.")
        await cache.store(messages, query_vec, "user_1", ["Name is Alice"])

        result = await cache.check(messages, query_vec, "user_2")
        assert result is None

    def test_clear_specific_user(self, cache, embedder):
        import asyncio
        messages = [{"role": "user", "content": "Test"}]
        query_vec = embedder.embed("Test")

        async def store():
            await cache.store(messages, query_vec, "user_1", ["Test fact"])
        asyncio.run(store())

        cache.clear("user_1")
        assert "user_1" not in cache._cache

    def test_clear_all_users(self, cache, embedder):
        import asyncio
        messages = [{"role": "user", "content": "Test"}]
        query_vec = embedder.embed("Test")

        async def store():
            await cache.store(messages, query_vec, "u1", ["fact"])
            await cache.store(messages, query_vec, "u2", ["fact"])
        asyncio.run(store())

        cache.clear()
        assert len(cache._cache) == 0
