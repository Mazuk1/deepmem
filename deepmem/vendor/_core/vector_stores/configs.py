from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class VectorStoreConfig(BaseModel):
    provider: str = Field(
        description="Provider of the vector store",
        default="qdrant",
    )
    config: Optional[Dict] = Field(description="Configuration for the vector store", default=None)

    _provider_configs: Dict[str, str] = {
        "qdrant": "QdrantConfig",
    }

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "VectorStoreConfig":
        provider = self.provider
        config = self.config

        if provider not in self._provider_configs:
            raise ValueError(f"Unsupported vector store provider: {provider}")

        from _core.configs.vector_stores.qdrant import QdrantConfig
        config_class = QdrantConfig

        if config is None:
            config = {}

        if not isinstance(config, dict):
            if not isinstance(config, config_class):
                raise ValueError(f"Invalid config type for provider {provider}")
            return self

        if "path" not in config and "path" in config_class.__annotations__:
            config["path"] = "/tmp/qdrant"

        self.config = config_class(**config)
        return self
