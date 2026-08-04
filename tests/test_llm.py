import pytest


class TestDeepSeekAdapter:
    @pytest.fixture
    def adapter(self, config):
        from deepmem.llm import DeepSeekAdapter
        return DeepSeekAdapter(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            model=config.deepseek_model,
        )

    def test_adapter_initializes_with_config(self, adapter):
        assert adapter.model is not None
        assert adapter.client is not None

    @pytest.mark.asyncio
    async def test_extract_facts_from_messages(self, adapter, sample_messages):
        result = await adapter.extract_facts(sample_messages)
        assert isinstance(result, list)
        assert len(result) > 0
        # Facts are now {"text": ..., "attributed_to": ...} dicts
        texts = [f["text"] if isinstance(f, dict) else f for f in result]
        assert any("Alice" in t or "San Francisco" in t for t in texts)

    @pytest.mark.asyncio
    async def test_extract_facts_empty_messages(self, adapter):
        result = await adapter.extract_facts([{"role": "user", "content": "Hi"}])
        assert isinstance(result, list)

    def test_parse_facts_from_json(self, adapter):
        response = '{"facts": ["Alice lives in San Francisco", "Alice is a software engineer"]}'
        facts = adapter._parse_facts(response)
        assert len(facts) == 2
        texts = [f["text"] for f in facts]
        assert "Alice lives in San Francisco" in texts
        assert all(f["attributed_to"] == "user" for f in facts)

    def test_parse_facts_with_attribution(self, adapter):
        response = (
            '{"facts": ['
            '{"text": "Alice lives in SF", "attributed_to": "user"},'
            '{"text": "Recommend Rust to Alice", "attributed_to": "assistant"}'
            ']}'
        )
        facts = adapter._parse_facts(response)
        assert len(facts) == 2
        assert facts[0]["attributed_to"] == "user"
        assert facts[1]["attributed_to"] == "assistant"

    def test_parse_facts_empty(self, adapter):
        response = '{"facts": []}'
        facts = adapter._parse_facts(response)
        assert facts == []

    def test_parse_facts_invalid_json(self, adapter):
        response = 'not json at all'
        facts = adapter._parse_facts(response)
        assert isinstance(facts, list)
