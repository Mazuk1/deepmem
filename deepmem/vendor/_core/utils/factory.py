"""Provider factories — trimmed to DeepMemory's needs (DeepSeek/OpenAI/Anthropic/BGE-M3/Qdrant)."""

import importlib
from typing import Dict, Optional, Union

from _core.configs.embeddings.base import BaseEmbedderConfig
from _core.configs.llms.anthropic import AnthropicConfig
from _core.configs.llms.base import BaseLlmConfig
from _core.configs.llms.deepseek import DeepSeekConfig
from _core.configs.llms.openai import OpenAIConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "openai": ("mem0.llms.openai.OpenAILLM", OpenAIConfig),
        "anthropic": ("mem0.llms.anthropic.AnthropicLLM", AnthropicConfig),
        "deepseek": ("mem0.llms.deepseek.DeepSeekLLM", DeepSeekConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseLlmConfig, Dict]] = None, **kwargs):
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")

        class_type, config_class = cls.provider_to_class[provider_name]
        llm_class = load_class(class_type)

        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config.update(kwargs)
            config = config_class(**config)
        elif isinstance(config, BaseLlmConfig) and config_class != BaseLlmConfig:
            config_dict = {
                "model": config.model, "temperature": config.temperature,
                "api_key": config.api_key, "max_tokens": config.max_tokens,
                "top_p": config.top_p, "top_k": config.top_k,
                "enable_vision": config.enable_vision,
                "vision_details": config.vision_details,
                "http_client_proxies": config.http_client,
            }
            config_dict.update(kwargs)
            config = config_class(**config_dict)

        return llm_class(config)


class EmbedderFactory:
    provider_to_class = {
        "openai": "mem0.embeddings.openai.OpenAIEmbedding",
        "bge_m3": "mem0.embeddings.bge_m3.BGEM3Embedding",
    }

    @classmethod
    def create(cls, provider_name, config, vector_config: Optional[dict] = None):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            if provider_name == "bge_m3":
                return embedder_instance()
            base_config = BaseEmbedderConfig(**config) if isinstance(config, dict) else config
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "mem0.vector_stores.qdrant.Qdrant",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance


class RerankerFactory:
    provider_to_class = {}

    @classmethod
    def create(cls, provider_name: str, config=None, **kwargs):
        raise ValueError(f"Unsupported reranker provider: {provider_name}")
