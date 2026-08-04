"""Multi-provider LLM tests.

Verifies create_llm_from_config selects the right adapter for each provider
(openai / anthropic / openai_compatible / deepseek auto-fallback) and that
native Anthropic + OpenAI-compatible extract_facts both parse facts correctly.
All network is mocked - no real API calls.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deepmem.llm import (
    AnthropicLLMAdapter,
    UniversalLLMAdapter,
    create_llm_from_config,
)


class FakeConfig:
    """Minimal config stand-in - create_llm_from_config only reads these attrs."""

    def __init__(self, **kw):
        self.llm_provider = "auto"
        self.llm_api_key = ""
        self.llm_base_url = ""
        self.llm_model = ""
        self.anthropic_api_key = ""
        self.anthropic_model = "claude-3-5-sonnet-20241022"
        self.deepseek_api_key = ""
        self.deepseek_base_url = "https://api.deepseek.com/v1"
        self.deepseek_model = "deepseek-v4-flash"
        for k, v in kw.items():
            setattr(self, k, v)


class TestProviderSelection:
    def test_explicit_anthropic(self):
        cfg = FakeConfig(llm_provider="anthropic", anthropic_api_key="sk-ant")
        adapter = create_llm_from_config(cfg)
        assert isinstance(adapter, AnthropicLLMAdapter)
        assert adapter.provider == "anthropic"

    def test_auto_prefers_anthropic_when_only_anthropic_key(self):
        cfg = FakeConfig(llm_provider="auto", anthropic_api_key="sk-ant")
        adapter = create_llm_from_config(cfg)
        assert isinstance(adapter, AnthropicLLMAdapter)

    def test_explicit_openai(self):
        cfg = FakeConfig(llm_provider="openai", llm_api_key="sk-oa",
                         llm_base_url="https://api.openai.com/v1", llm_model="gpt-4o")
        adapter = create_llm_from_config(cfg)
        assert isinstance(adapter, UniversalLLMAdapter)
        assert adapter.base_url == "https://api.openai.com/v1"
        assert adapter.model == "gpt-4o"

    def test_explicit_openai_compatible(self):
        cfg = FakeConfig(llm_provider="openai_compatible", llm_api_key="sk",
                         llm_base_url="http://localhost:11434/v1", llm_model="llama3")
        adapter = create_llm_from_config(cfg)
        assert isinstance(adapter, UniversalLLMAdapter)
        assert "11434" in adapter.base_url

    def test_auto_falls_back_to_deepseek_backward_compat(self):
        # No llm_* / anthropic_* set - must use deepseek_* (existing behavior).
        cfg = FakeConfig(llm_provider="auto", deepseek_api_key="sk-ds")
        adapter = create_llm_from_config(cfg)
        assert isinstance(adapter, UniversalLLMAdapter)
        assert adapter.api_key == "sk-ds"
        assert "deepseek" in adapter.base_url

    def test_llm_fields_override_deepseek_when_both_set(self):
        cfg = FakeConfig(llm_provider="openai_compatible",
                         llm_api_key="sk-llm", llm_base_url="http://vllm:8000/v1",
                         llm_model="qwen", deepseek_api_key="sk-ds")
        adapter = create_llm_from_config(cfg)
        assert adapter.api_key == "sk-llm"
        assert adapter.base_url == "http://vllm:8000/v1"
        assert adapter.model == "qwen"


class TestExtractFacts:
    """extract_facts must parse the {facts:[...]} envelope for every provider."""

    @pytest.mark.asyncio
    async def test_anthropic_extract_facts_native_api_shape(self):
        cfg = FakeConfig(llm_provider="anthropic", anthropic_api_key="sk-ant")
        adapter = create_llm_from_config(cfg)
        fake_resp = SimpleNamespace(content=[SimpleNamespace(
            text='{"facts":[{"text":"User likes ramen","attributed_to":"user"}]}')])
        adapter.client.messages.create = AsyncMock(return_value=fake_resp)

        facts = await adapter.extract_facts(
            [{"role": "user", "content": "I like ramen"}])

        assert facts == [{"text": "User likes ramen", "attributed_to": "user"}]
        # Native Anthropic API: system is a top-level param, not in messages.
        kwargs = adapter.client.messages.create.call_args.kwargs
        assert "system" in kwargs and kwargs["system"]
        assert all(m["role"] != "system" for m in kwargs["messages"])

    @pytest.mark.asyncio
    async def test_openai_compatible_extract_facts(self):
        cfg = FakeConfig(llm_provider="openai_compatible", llm_api_key="sk",
                         llm_base_url="http://localhost:11434/v1", llm_model="llama3")
        adapter = create_llm_from_config(cfg)
        fake_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"facts":[{"text":"User codes in Rust","attributed_to":"user"}]}'))],
            usage=None)
        adapter.client.chat.completions.create = AsyncMock(return_value=fake_resp)

        facts = await adapter.extract_facts(
            [{"role": "user", "content": "I code in Rust"}])
        assert facts == [{"text": "User codes in Rust", "attributed_to": "user"}]

    @pytest.mark.asyncio
    async def test_anthropic_empty_facts(self):
        cfg = FakeConfig(llm_provider="anthropic", anthropic_api_key="sk-ant")
        adapter = create_llm_from_config(cfg)
        adapter.client.messages.create = AsyncMock(
            return_value=SimpleNamespace(content=[SimpleNamespace(text='{"facts":[]}')]))
        facts = await adapter.extract_facts([{"role": "user", "content": "hi"}])
        assert facts == []
