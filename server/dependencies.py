import logging
import os

from deepmem.config import (
    DeepMemoryConfig,
    config,
)
from deepmem.embedder import get_embedder
from deepmem.engine.semantic_cache import SemanticCache
from deepmem.engine.batch_distiller import AsyncBatchDistiller
from deepmem.engine.model_router import ModelRouter
from deepmem.history_store import HistoryStore
from deepmem.middleware import TenantValidator
from deepmem.vector_store import VectorStore

logger = logging.getLogger(__name__)

_embedder = None
_cache = None
_distiller = None
_router = None
_store = None
_history = None


def get_services():
    """Lazy-initialize and return all service singletons."""
    global _embedder, _cache, _distiller, _router, _store, _history

    if _embedder is None:
        _embedder = get_embedder(config)

    if _cache is None:
        _cache = SemanticCache(
            similarity_threshold=config.cache_similarity_threshold,
            ttl_seconds=config.cache_ttl_seconds,
            persist_path=getattr(config, "cache_persist_path", ""),
        )

    if _router is None:
        # config= wires the multi-provider LLM path (LLM_PROVIDER /
        # ANTHROPIC_* / LLM_*); BYOK still overrides per-request.
        _router = ModelRouter(config=config)

    if _history is None:
        history_db = getattr(config, "history_db_path", None) \
            or os.path.join(getattr(config, "qdrant_path", "./data/qdrant"),
                            "..", "history.db")
        _history = HistoryStore(db_path=os.path.normpath(history_db))

    if _store is None:
        _store = VectorStore(
            qdrant_path=getattr(config, "qdrant_path", "./data/qdrant"),
            qdrant_url=getattr(config, "qdrant_url", "") or "",
            qdrant_api_key=getattr(config, "qdrant_api_key", "") or "",
            embedding_dims=config.embedding_dims,
            bge_m3_path=getattr(config, "bge_m3_path", "") or None,
            history_store=_history,
        )

    if _distiller is None:
        # Wire on_batch_ready inline at construction so anybody calling
        # get_services() can immediately enqueue messages and trust they
        # will reach VectorStore.process_batch - no need for a separate
        # lifespan hook to attach the callback later.
        _distiller = AsyncBatchDistiller(
            silence_window=config.batch_silence_window_seconds,
            max_batch_size=config.batch_max_size,
        )
        _distiller.on_batch_ready = _store.process_batch

    return {
        "embedder": _embedder,
        "cache": _cache,
        "distiller": _distiller,
        "router": _router,
        "store": _store,
        "history": _history,
        "validator": TenantValidator(),
        "config": config,
    }
