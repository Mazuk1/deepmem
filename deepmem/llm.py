"""Model-Agnostic LLM Adapter — any OpenAI-compatible API, one line to switch.

Supports: DeepSeek, OpenAI, Anthropic (via OpenAI-compat), Ollama, vLLM,
Groq, Together, and any custom OpenAI-compatible endpoint.

Usage:
    from deepmem.config import DeepMemoryConfig
    from deepmem.llm import create_llm_from_config

    cfg = DeepMemoryConfig.from_json()
    adapter = create_llm_from_config(cfg)
    facts = await adapter.extract_facts(messages)

Or directly:
    from deepmem.llm import create_llm_adapter
    adapter = create_llm_adapter(api_key="sk-xxx", base_url="https://api.openai.com/v1", model="gpt-4o")

To switch providers, just change config.json:
    {"base_url": "https://api.openai.com/v1", "api_key": "sk-xxx", "model": "gpt-4o"}
    {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "llama3"}
"""

import json
import logging
import re
from typing import Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger("deepmem.llm")

# Prompt template for fact extraction — works across all models
FACT_EXTRACTION_PROMPT = """You are a Personal Information Organizer. Extract relevant facts, user memories, and preferences from conversations into distinct, manageable facts.

Types of Information to Remember:
1. Personal Preferences: likes, dislikes, preferences in food, products, activities
2. Important Personal Details: names, relationships, important dates
3. Plans and Intentions: upcoming events, trips, goals
4. Professional Details: job titles, work habits, career goals
5. Durable Technical Preferences and Project Conventions: preferred programming languages, framework/library versions, architecture constraints, coding style preferences, testing/deployment conventions, and workflow habits
6. Miscellaneous: favorite books, movies, brands, other durable details

Do NOT extract implementation details from code. Do NOT memorize full code, algorithms, function bodies, logs, stack traces, or one-off task instructions.
If the input is only code or implementation detail with no durable preference, configuration, convention, or habit, return {"facts": []}.

For every fact, attribute it to either the user or the assistant:
- "user": the user stated this about themselves, their preferences, plans, or context.
- "assistant": the assistant produced this (e.g. recommendation, summary). Assistant facts are usually NOT user profile facts and most of the time should be omitted unless they encode a durable agreement or convention the user accepted.

Return ONLY a JSON object in this shape:
{"facts": [{"text": "fact 1", "attributed_to": "user"}, {"text": "fact 2", "attributed_to": "user"}]}

Only extract factual, durable information that would be useful in future conversations. Ignore generic statements and small talk.
If no facts are found, return {"facts": []}.
"""

# Provider-specific model name aliases (normalized → actual).
# DeepSeek currently exposes only `deepseek-v4-flash` — legacy `deepseek-chat`
# and `deepseek-reasoner` model ids no longer route, so any caller that still
# passes `deepseek-v3` / `deepseek-r1` (or the old direct names) lands on flash.
_MODEL_ALIASES = {
    "deepseek-v3": "deepseek-v4-flash",
    "deepseek-r1": "deepseek-v4-flash",
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
    "gpt-4": "gpt-4o",
    "claude-3.5": "claude-3-5-sonnet-20241022",
}


def _detect_provider(base_url: str) -> str:
    """Auto-detect LLM provider from base_url.

    Returns one of: 'deepseek', 'openai', 'anthropic', 'ollama', 'openai_compatible'
    """
    url_lower = base_url.lower()
    if "deepseek" in url_lower:
        return "deepseek"
    if "openai" in url_lower:
        return "openai"
    if "anthropic" in url_lower:
        return "anthropic"
    if "ollama" in url_lower or ":11434" in url_lower:
        return "ollama"
    if "groq" in url_lower:
        return "groq"
    # Any other OpenAI-compatible endpoint (vLLM, Together, local, etc.)
    return "openai_compatible"


def _normalize_model(model: str) -> str:
    """Resolve model aliases to actual API model names."""
    return _MODEL_ALIASES.get(model.lower(), model)


class UniversalLLMAdapter:
    """Model-agnostic LLM adapter — works with any OpenAI-compatible API.

    DeepSeek, OpenAI, Anthropic (via compatible proxy), Ollama, vLLM,
    Groq, Together — all work by just changing base_url and model.
    """

    def __init__(self, api_key: str,
                 base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-v4-flash",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 extra_headers: Optional[Dict[str, str]] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = _normalize_model(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = _detect_provider(base_url)

        client_kwargs = {"api_key": api_key, "base_url": base_url}
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
        # AsyncOpenAI: native asyncio, doesn't block the event loop. Critical
        # because extract_facts() is on the request hot path under FastAPI.
        self.client = AsyncOpenAI(**client_kwargs)

        logger.info(
            "LLM adapter initialized: provider=%s model=%s base_url=%s",
            self.provider, self.model, base_url,
        )

    async def extract_facts(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extract memory facts from a conversation using the configured LLM.

        Returns a list of {"text": str, "attributed_to": "user"|"assistant"} dicts.
        Plain-string facts (legacy / dummy adapters) are normalized to user-attributed.

        Raises on LLM transport / API errors so callers can trigger the raw
        fallback path. Previously this method swallowed everything and returned
        [], which the caller could not distinguish from "no facts found" and
        the user's content was lost silently.
        """
        system_msg = {"role": "system", "content": FACT_EXTRACTION_PROMPT}
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        user_msg = {"role": "user", "content": f"Extract facts from:\n{conversation}"}

        # Provider-specific optimizations
        extra_kwargs = {}
        if self.provider == "deepseek":
            # DeepSeek supports OpenAI-style JSON mode
            extra_kwargs["response_format"] = {"type": "json_object"}
        elif self.provider == "openai":
            extra_kwargs["response_format"] = {"type": "json_object"}
        elif self.provider == "anthropic":
            # Anthropic via OpenAI-compat proxy may need different params
            extra_kwargs["max_tokens"] = min(self.max_tokens, 4096)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[system_msg, user_msg],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **extra_kwargs,
            )
        except Exception as e:
            logger.error("LLM API error [provider=%s model=%s]: %s",
                         self.provider, self.model, e)
            raise

        content = response.choices[0].message.content
        facts = self._parse_facts(content)

        if facts:
            logger.debug(
                "LLM extracted %d facts: provider=%s model=%s tokens_in=%d",
                len(facts), self.provider, self.model,
                response.usage.prompt_tokens if response.usage else 0,
            )
        return facts

    def _parse_facts(self, response: str) -> List[Dict[str, str]]:
        """Parse facts from LLM JSON response.

        Accepts both the new {text, attributed_to} dict shape and the legacy
        plain-string list shape (defaults attribution to "user"). Tolerates
        markdown fences and JSON with leading/trailing prose.
        """
        if not response:
            return []
        # Strip markdown fences (```json ... ``` or ``` ... ```)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]
            cleaned = cleaned.strip()

        # Try direct JSON parse first
        try:
            data = json.loads(cleaned)
            parsed = self._normalize_facts_payload(data)
            if parsed is not None:
                return parsed
        except json.JSONDecodeError:
            pass

        # Find first balanced {...} block — manual scan handles nesting
        start = cleaned.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:i + 1]
                        try:
                            data = json.loads(candidate)
                            parsed = self._normalize_facts_payload(data)
                            if parsed is not None:
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break
            start = cleaned.find("{", start + 1)

        logger.warning("Failed to parse facts from response (first 300 chars): %s",
                       response[:300])
        return []

    @staticmethod
    def _normalize_facts_payload(data) -> Optional[List[Dict[str, str]]]:
        """Coerce a parsed JSON object into a list of {text, attributed_to} dicts.
        Returns None if the payload doesn't look like a facts envelope."""
        if not isinstance(data, dict) or "facts" not in data:
            return None
        items = data.get("facts") or []
        out: List[Dict[str, str]] = []
        for item in items:
            if isinstance(item, str):
                if item.strip():
                    out.append({"text": item.strip(), "attributed_to": "user"})
            elif isinstance(item, dict):
                text = item.get("text") or item.get("fact")
                if not isinstance(text, str) or not text.strip():
                    continue
                attributed = item.get("attributed_to", "user")
                if attributed not in ("user", "assistant"):
                    attributed = "user"
                out.append({"text": text.strip(), "attributed_to": attributed})
        return out

    async def generate_response(self, messages: List[Dict[str, str]],
                                **kwargs) -> str:
        """General-purpose response generation (for chat, not fact extraction)."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content


# ── Native Anthropic adapter ───────────────────────────────────────────────


class AnthropicLLMAdapter(UniversalLLMAdapter):
    """Native Anthropic LLM adapter - uses the anthropic SDK directly, NOT an
    OpenAI-compatibility proxy. Same extract_facts contract as
    UniversalLLMAdapter, so the memory pipeline doesn't care which provider
    is wired.

    Anthropic has no JSON response mode, so we rely on the extraction prompt
    plus the tolerant _parse_facts (inherited) to handle prose / code fences.
    """

    def __init__(self, api_key: str,
                 model: str = "claude-3-5-sonnet-20241022",
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        import anthropic  # lazy import keeps the dep optional at module import
        self.api_key = api_key
        self.base_url = ""  # native Anthropic - no OpenAI-style base_url
        self.model = _normalize_model(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = "anthropic"
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info("LLM adapter initialized: provider=anthropic model=%s", self.model)

    async def extract_facts(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        user_msg = {"role": "user", "content": f"Extract facts from:\n{conversation}"}
        try:
            response = await self.client.messages.create(
                model=self.model,
                system=FACT_EXTRACTION_PROMPT,
                messages=[user_msg],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error("Anthropic LLM API error [model=%s]: %s", self.model, e)
            raise
        content = response.content[0].text if response.content else ""
        return self._parse_facts(content)

    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Anthropic takes system as a top-level param, not in the messages list.
        system = ""
        convo = []
        for m in messages:
            if m.get("role") == "system":
                system += m.get("content", "")
            else:
                convo.append(m)
        kwargs.setdefault("max_tokens", self.max_tokens)
        response = await self.client.messages.create(
            model=self.model,
            system=system or None,
            messages=convo,
            **kwargs,
        )
        return response.content[0].text if response.content else ""


# ── Factory Functions ─────────────────────────────────────────────────────


def create_llm_adapter(api_key: str,
                       base_url: str = "https://api.deepseek.com/v1",
                       model: str = "deepseek-v4-flash",
                       **kwargs) -> UniversalLLMAdapter:
    """Create a model-agnostic LLM adapter.

    Change base_url to switch between providers:
        DeepSeek:  base_url="https://api.deepseek.com/v1"
        OpenAI:    base_url="https://api.openai.com/v1"
        Ollama:    base_url="http://localhost:11434/v1"
        vLLM:      base_url="http://localhost:8000/v1"
        Groq:      base_url="https://api.groq.com/openai/v1"
    """
    return UniversalLLMAdapter(api_key=api_key, base_url=base_url, model=model, **kwargs)


def create_llm_from_config(config) -> UniversalLLMAdapter:
    """Create an LLM adapter from DeepMemoryConfig.

    This is the recommended way — reads base_url, api_key, model from config.
    To switch providers, just change your config.json.
    """
    provider = (getattr(config, "llm_provider", "auto") or "auto").lower()

    if provider == "anthropic":
        return AnthropicLLMAdapter(
            api_key=config.anthropic_api_key,
            model=config.anthropic_model,
        )

    if provider == "auto":
        # If only an Anthropic key is configured (no OpenAI-compatible key),
        # prefer native Anthropic; otherwise the OpenAI-compatible path below
        # handles OpenAI / DeepSeek / vLLM / Ollama / Groq / etc.
        if getattr(config, "anthropic_api_key", "") and not (
            getattr(config, "llm_api_key", "") or config.deepseek_api_key
        ):
            return AnthropicLLMAdapter(
                api_key=config.anthropic_api_key,
                model=config.anthropic_model,
            )

    # openai / openai_compatible / deepseek / auto-fallback
    api_key = getattr(config, "llm_api_key", "") or config.deepseek_api_key
    base_url = getattr(config, "llm_base_url", "") or config.deepseek_base_url
    model = getattr(config, "llm_model", "") or config.deepseek_model
    return UniversalLLMAdapter(api_key=api_key, base_url=base_url, model=model)


# ── Backward Compatibility ─────────────────────────────────────────────────


# DeepSeekAdapter alias for existing code
DeepSeekAdapter = UniversalLLMAdapter
