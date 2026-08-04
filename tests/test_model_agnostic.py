"""Test model-agnostic LLM adapter — any OpenAI-compatible API should work."""

import pytest


class TestUniversalLLMAdapter:
    """Verify adapter works with different providers via config."""

    def test_deepseek_provider_auto_detected(self, config):
        from deepmem.llm import create_llm_adapter
        adapter = create_llm_adapter(
            api_key=config.deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )
        assert adapter is not None
        assert adapter.model == "deepseek-v4-flash"
        assert "deepseek" in adapter.base_url

    def test_openai_provider_auto_detected(self):
        from deepmem.llm import create_llm_adapter
        adapter = create_llm_adapter(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        assert adapter is not None
        assert adapter.model == "gpt-4o"
        assert "openai" in adapter.base_url

    def test_custom_openai_compatible_endpoint(self):
        """Any OpenAI-compatible endpoint (vLLM, Ollama, Groq, etc.) should work."""
        from deepmem.llm import create_llm_adapter
        adapter = create_llm_adapter(
            api_key="not-needed",
            base_url="http://localhost:11434/v1",
            model="llama3",
        )
        assert adapter is not None
        assert adapter.model == "llama3"
        assert "localhost:11434" in adapter.base_url

    def test_anthropic_provider_auto_detected(self):
        from deepmem.llm import create_llm_adapter
        adapter = create_llm_adapter(
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-6",
        )
        assert adapter is not None
        assert "claude" in adapter.model

    def test_factory_from_config(self, config):
        """Factory should create adapter from DeepMemoryConfig."""
        from deepmem.llm import create_llm_from_config
        adapter = create_llm_from_config(config)
        assert adapter is not None
        assert adapter.model == config.deepseek_model

    @pytest.mark.asyncio
    async def test_fact_extraction_works_with_any_adapter(self, config):
        """Fact extraction should work identically regardless of provider."""
        from deepmem.llm import create_llm_adapter
        adapter = create_llm_adapter(
            api_key=config.deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
            model=config.deepseek_model,
        )
        messages = [
            {"role": "user", "content": "My name is Alice, I live in San Francisco."},
        ]
        facts = await adapter.extract_facts(messages)
        assert isinstance(facts, list)
        assert len(facts) > 0

    def test_adapter_stores_provider_name(self):
        from deepmem.llm import create_llm_adapter
        adapter = create_llm_adapter(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        # Base URL should be stored for downstream use
        assert adapter.base_url is not None
        assert adapter.api_key is not None
        assert adapter.model is not None


class TestProviderAutoDetection:
    def test_detect_deepseek(self):
        from deepmem.llm import _detect_provider
        assert _detect_provider("https://api.deepseek.com/v1") == "deepseek"
        assert _detect_provider("https://api.deepseek.com") == "deepseek"

    def test_detect_openai(self):
        from deepmem.llm import _detect_provider
        assert _detect_provider("https://api.openai.com/v1") == "openai"

    def test_detect_anthropic(self):
        from deepmem.llm import _detect_provider
        assert _detect_provider("https://api.anthropic.com") == "anthropic"

    def test_detect_ollama(self):
        from deepmem.llm import _detect_provider
        assert _detect_provider("http://localhost:11434/v1") == "ollama"

    def test_detect_unknown(self):
        from deepmem.llm import _detect_provider
        assert _detect_provider("https://my-custom-llm.example.com") == "openai_compatible"
