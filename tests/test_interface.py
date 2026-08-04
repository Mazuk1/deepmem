import pytest


class TestMemoryEngineInterface:
    def test_cannot_instantiate_abstract_engine(self):
        from deepmem.interface import MemoryEngine
        with pytest.raises(TypeError):
            MemoryEngine()

    def test_concrete_must_implement_all_abstract_methods(self):
        from deepmem.interface import MemoryEngine
        class IncompleteEngine(MemoryEngine):
            pass
        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_full_implementation_instantiates(self):
        from deepmem.interface import MemoryEngine, Tenant, SearchResult, MemoryItem
        from typing import Any, Dict, List, Optional

        class FullEngine(MemoryEngine):
            async def add(self, messages, tenant, metadata=None, infer=True,
                          llm_provider=None):
                return ["id-1"]

            async def search(self, query, tenant, top_k=10, threshold=0.3, **kwargs):
                return [SearchResult(id="id-1", memory="test", score=0.95)]

            async def get(self, memory_id, tenant):
                return SearchResult(id=memory_id, memory="test", score=1.0)

            async def update(self, memory_id, new_memory, tenant):
                return True

            async def list(self, tenant, limit=100, offset=0):
                return [SearchResult(id="id-1", memory="test", score=0.0)]

            async def delete(self, memory_id, tenant):
                return True

            async def delete_all(self, tenant):
                return 5

        engine = FullEngine()
        assert engine is not None
