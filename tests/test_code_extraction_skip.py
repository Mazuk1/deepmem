import pytest

from deepmem.interface import Tenant
from deepmem.vector_store import VectorStore


BIG_CODE = """```python
def solve():
    board = []
    rows = []
    cols = []
    diag1 = []
    diag2 = []
    result = []
    for i in range(100):
        result.append(i)
    return result
```"""


class DummyLLM:
    def __init__(self, facts=None, raises=False):
        self.facts = facts if facts is not None else []
        self.raises = raises
        self.seen_messages = None

    async def extract_facts(self, messages):
        self.seen_messages = messages
        if self.raises:
            raise RuntimeError("LLM down")
        return self.facts


class DummyEmbedder:
    def embed_batch(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    # Phase 3.2: VectorStore.process_batch awaits aembed_batch so the
    # event loop isn't blocked on the BGE-M3 forward pass. Tests need a
    # matching async surface; the no-op embedder doesn't need a real
    # asyncio.to_thread offload.
    async def aembed_batch(self, texts, memory_action="add"):
        return self.embed_batch(texts)

    async def aembed(self, text, memory_action=None):
        return self.embed(text)


@pytest.fixture
def store(tmp_path):
    s = VectorStore(qdrant_path=str(tmp_path / "qdrant"), embedding_dims=4)
    s._embedder = DummyEmbedder()
    return s


@pytest.mark.asyncio
async def test_code_only_input_skips_extraction_and_does_not_store_raw(store):
    llm = DummyLLM(facts=["should not be used"])
    tenant = Tenant(user_id="code_skip_user")

    ids = await store.add(
        [{"role": "user", "content": BIG_CODE}],
        tenant,
        infer=True,
        llm_provider=llm,
    )

    assert ids == []
    assert llm.seen_messages is None
    assert await store.list(tenant) == []


@pytest.mark.asyncio
async def test_mixed_text_strips_code_before_llm(store):
    llm = DummyLLM(facts=["User prefers all algorithm examples to use Rust."])
    tenant = Tenant(user_id="code_mixed_user")

    ids = await store.add(
        [{"role": "user", "content": f"I prefer all algorithm examples in Rust.\n\n{BIG_CODE}"}],
        tenant,
        infer=True,
        llm_provider=llm,
    )

    assert len(ids) == 1
    assert llm.seen_messages is not None
    sent = llm.seen_messages[0]["content"]
    assert "I prefer all algorithm examples in Rust" in sent
    assert "[code block omitted]" in sent
    assert "def solve" not in sent

    memories = await store.list(tenant)
    assert len(memories) == 1
    assert "Rust" in memories[0].memory


@pytest.mark.asyncio
async def test_llm_empty_facts_does_not_fallback_to_raw_task(store):
    llm = DummyLLM(facts=[])
    tenant = Tenant(user_id="empty_fact_user")

    ids = await store.add(
        [{"role": "user", "content": "Write a Rust program that solves N-queens."}],
        tenant,
        infer=True,
        llm_provider=llm,
    )

    assert ids == []
    assert await store.list(tenant) == []


@pytest.mark.asyncio
async def test_llm_failure_fallback_is_sanitized(store):
    llm = DummyLLM(raises=True)
    tenant = Tenant(user_id="fallback_user")

    ids = await store.add(
        [{"role": "user", "content": f"The project requires Python 3.11.\n\n{BIG_CODE}"}],
        tenant,
        infer=True,
        llm_provider=llm,
    )

    assert len(ids) == 1
    memories = await store.list(tenant)
    text = "\n".join(m.memory for m in memories)
    assert "Python 3.11" in text
    assert "[code block omitted]" in text
    assert "def solve" not in text
