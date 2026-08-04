import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Auto-load .env from the repo root so the app reads secrets from the
# environment without the caller having to `source .env` first. override=False
# means a real environment variable (systemd / Docker / CI) always wins over
# the value in .env - .env only fills in what the environment hasn't set.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)


@dataclass
class DeepMemoryConfig:
    # LLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"
    # Multi-provider LLM selection. llm_provider: auto | openai | anthropic |
    # openai_compatible. "auto" detects from base_url (DeepSeek/vLLM/Ollama =
    # openai_compatible) and falls back to the deepseek_* fields, preserving
    # existing behavior. Set "anthropic" to use the native Anthropic API.
    llm_provider: str = "auto"
    llm_api_key: str = ""        # generic key for openai / openai_compatible
    llm_base_url: str = ""       # generic base_url for openai / openai_compatible
    llm_model: str = ""          # generic model for openai / openai_compatible
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Embedding
    default_embedder: str = "bge-m3"  # "bge-m3" | "google" | "openai"
    google_api_key: str = ""
    google_embedding_model: str = "gemini-embedding-001"
    bge_m3_path: str = ""  # local BGE-M3 dir OR HuggingFace model id
    # OpenAI-compatible embedding (OpenAI / vLLM / Ollama / LM Studio / etc.)
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    # Semantic Cache
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.98
    cache_ttl_seconds: int = 300
    cache_persist_path: str = "./data/cache.json"

    # Async Batch Distiller
    batch_enabled: bool = True
    batch_silence_window_seconds: int = 180
    batch_max_size: int = 50

    # Storage
    qdrant_path: str = "./data/qdrant"
    qdrant_url: str = ""  # Remote Qdrant; when set, overrides qdrant_path.
    qdrant_api_key: str = ""  # Required by Qdrant Cloud; ignored on local file mode.
    qdrant_collection: str = "memories"
    embedding_dims: int = 1024

    # GDPR
    soft_delete_retention_days: int = 30

    @classmethod
    def from_json(cls, path: str = "config.json") -> "DeepMemoryConfig":
        """Load config from JSON, then let env vars override.

        Precedence: env var > config.json > dataclass default.
        Env vars win so production secrets never have to live in a file
        that could end up in git.
        """
        cfg = cls()
        data: dict = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # ── DeepSeek ────────────────────────────────────────────────
        cfg.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or data.get("api_key", "")
        cfg.deepseek_model = os.getenv("DEEPSEEK_MODEL") or data.get("model", cfg.deepseek_model)

        # Trust the user-supplied base_url. The previous heuristic
        # (`"api" in raw_url`) was too loose — it matched any url
        # containing the substring "api" while silently dropping
        # internal endpoints that happened not to.
        raw_url = os.getenv("DEEPSEEK_BASE_URL") or data.get("base_url", "")
        if raw_url:
            cfg.deepseek_base_url = raw_url

        # ── Multi-provider LLM ───────────────────────────────────────
        # llm_provider selects the backend; the generic llm_* fields feed the
        # OpenAI-compatible path, anthropic_* feed the native Anthropic path.
        # When unset, falls back to deepseek_* so existing deployments keep
        # working unchanged.
        cfg.llm_provider = (
            os.getenv("LLM_PROVIDER") or data.get("llm_provider") or cfg.llm_provider
        )
        cfg.llm_api_key = os.getenv("LLM_API_KEY") or data.get("llm_api_key", "")
        cfg.llm_base_url = os.getenv("LLM_BASE_URL") or data.get("llm_base_url", "")
        cfg.llm_model = os.getenv("LLM_MODEL") or data.get("llm_model", "")
        cfg.anthropic_api_key = (
            os.getenv("ANTHROPIC_API_KEY") or data.get("anthropic_api_key", "")
        )
        cfg.anthropic_model = (
            os.getenv("ANTHROPIC_MODEL")
            or data.get("anthropic_model", cfg.anthropic_model)
        )

        # ── Embedding ───────────────────────────────────────────────
        # Provider selection: "bge-m3" (default), "google", or "openai"
        cfg.default_embedder = (
            os.getenv("EMBEDDING_PROVIDER")
            or data.get("default_embedder")
            or cfg.default_embedder
        )

        # Google embedding
        cfg.google_api_key = os.getenv("GOOGLE_API_KEY") or data.get("google_api_key", "")
        cfg.google_embedding_model = (
            os.getenv("GOOGLE_EMBEDDING_MODEL")
            or data.get("google_embedding_model", cfg.google_embedding_model)
        )

        # OpenAI-compatible embedding endpoint (also vLLM / Ollama / LM Studio).
        cfg.openai_api_key = os.getenv("OPENAI_API_KEY") or data.get("openai_api_key", "")
        cfg.openai_base_url = os.getenv("OPENAI_BASE_URL") or data.get("openai_base_url", "")
        cfg.openai_embedding_model = (
            os.getenv("OPENAI_EMBEDDING_MODEL")
            or data.get("openai_embedding_model", cfg.openai_embedding_model)
        )

        # BGE-M3 local model path. Env: BGE_M3_PATH. JSON key: bge_m3_path
        # (embedding_model_path is a legacy alias).
        emb_path = (
            os.getenv("BGE_M3_PATH")
            or data.get("bge_m3_path", "")
            or data.get("embedding_model_path", "")
        )
        if emb_path:
            cfg.bge_m3_path = emb_path

        # ── Storage ─────────────────────────────────────────────────
        # QDRANT_URL takes precedence over qdrant_path when set; both are
        # forwarded to dependencies.py which picks the right QdrantClient
        # constructor (path= for local file, url= for remote service).
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        if qdrant_url:
            cfg.qdrant_url = qdrant_url
        # Qdrant Cloud requires an api_key; self-hosted Docker often doesn't.
        # Empty string keeps the QdrantClient default (no auth header).
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if qdrant_api_key:
            cfg.qdrant_api_key = qdrant_api_key
        # Local file mode also honors QDRANT_PATH so e2e tests / parallel
        # dev sandboxes can point at a scratch dir instead of polluting
        # the default ./data/qdrant.
        qdrant_path_env = os.getenv("QDRANT_PATH", "").strip()
        if qdrant_path_env:
            cfg.qdrant_path = qdrant_path_env
        elif data.get("qdrant_path"):
            cfg.qdrant_path = data["qdrant_path"]

        # ── Cache / Batch tunables ──────────────────────────────────
        # Each tunable is best-effort: a typo'd env var falls back to the
        # dataclass default rather than crashing boot.
        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def _i(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        cfg.cache_similarity_threshold = _f(
            "CACHE_SIMILARITY_THRESHOLD", cfg.cache_similarity_threshold,
        )
        cfg.cache_ttl_seconds = _i("CACHE_TTL_SECONDS", cfg.cache_ttl_seconds)
        cfg.batch_silence_window_seconds = _i(
            "BATCH_SILENCE_WINDOW_SECONDS", cfg.batch_silence_window_seconds,
        )
        cfg.batch_max_size = _i("BATCH_MAX_SIZE", cfg.batch_max_size)

        return cfg


# Global config instance for easy import
config = DeepMemoryConfig.from_json()
