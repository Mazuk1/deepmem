import logging
from typing import Optional

from deepmem.llm import DeepSeekAdapter, create_llm_adapter, create_llm_from_config

logger = logging.getLogger(__name__)


# BYOK defaults — kept here so the two call sites (direct add path in
# server/main.py, batch path via VectorStore.process_batch) cannot drift.
BYOK_DEFAULT_BASE_URL = "https://api.openai.com/v1"
BYOK_DEFAULT_MODEL = "gpt-4o"


def build_byok_config(api_key: Optional[str],
                       base_url: Optional[str] = None,
                       model: Optional[str] = None) -> Optional[dict]:
    """Return the BYOK config dict, or None when no api_key is provided.

    The same dict shape is consumed by VectorStore.process_batch via the
    distiller pipeline and by adapter_from_byok_config below — keeping a
    single producer prevents the two from drifting.
    """
    if not api_key:
        return None
    return {
        "api_key": api_key,
        "base_url": base_url or BYOK_DEFAULT_BASE_URL,
        "model": model or BYOK_DEFAULT_MODEL,
    }


def adapter_from_byok_config(cfg: dict) -> DeepSeekAdapter:
    return create_llm_adapter(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url", BYOK_DEFAULT_BASE_URL),
        model=cfg.get("model", BYOK_DEFAULT_MODEL),
    )


class ModelRouter:
    """Routes memory extraction to the configured LLM provider.

    Multi-provider: when wired with the full config (dependencies.get_services
    passes config=...), route() honors LLM_PROVIDER / ANTHROPIC_* / LLM_* via
    create_llm_from_config - so OpenAI, native Anthropic, and any OpenAI-
    compatible backend (DeepSeek / vLLM / Ollama / Groq / ...) are all
    selectable without code changes. The legacy deepseek_* constructor kwargs
    are kept so existing call sites and tests keep working when no config is
    supplied. BYOK always overrides everything.
    """

    def __init__(self, deepseek_api_key: Optional[str] = None,
                 deepseek_base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-v4-flash",
                 config=None):
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_base_url = deepseek_base_url
        self.model = model
        self.config = config

    def route(self, tenant, tier: str = "free",
              custom_api_key: str = None,
              custom_base_url: str = None) -> DeepSeekAdapter:
        """Select the LLM provider based on tenant tier and configuration."""

        if custom_api_key:
            logger.info(f"Routing user={tenant.user_id} to BYOK provider")
            return DeepSeekAdapter(
                api_key=custom_api_key,
                base_url=custom_base_url or "https://api.openai.com/v1",
            )

        if self.config is not None:
            # Multi-provider path: honor LLM_PROVIDER config.
            logger.info(f"Routing user={tenant.user_id} via configured LLM provider")
            return create_llm_from_config(self.config)

        # Legacy path (no config wired) - DeepSeek via constructor kwargs.
        logger.info(f"Routing user={tenant.user_id} to DeepSeek model={self.model} tier={tier}")
        return DeepSeekAdapter(
            api_key=self.deepseek_api_key,
            base_url=self.deepseek_base_url,
            model=self.model,
        )
