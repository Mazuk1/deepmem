"""Embedding provider factory for DeepMemory.

Supports three backends:
  - google:   Google Gemini embedding API. Requires GOOGLE_API_KEY.
  - bge-m3:   BGE-M3 via sentence-transformers. BGE_M3_PATH may be a local
              model dir (loaded offline) or a HuggingFace model id (auto-
              downloaded on first use); when unset it defaults to the
              "BAAI/bge-m3" HuggingFace model.
  - openai:   Any OpenAI-compatible /v1/embeddings endpoint (OpenAI, vLLM,
              Ollama, LM Studio, ...). Requires OPENAI_API_KEY; set
              OPENAI_BASE_URL for non-OpenAI backends.

Select via config.default_embedder ("google" / "bge-m3" / "openai") or the
env var EMBEDDING_PROVIDER.
"""

import logging
import os
import sys
from typing import List

import numpy as np

# Ensure vendored _core package is importable
_vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _vendor not in sys.path:
    sys.path.insert(0, _vendor)

logger = logging.getLogger(__name__)

_embedder_instance = None


def get_embedder(config=None):
    """Get or create the global embedder instance.

    Selects backend based on config.default_embedder:
      - "google": GoogleEmbedding (Gemini API, cloud)
      - "bge-m3": BGEM3Embedding (local dir or HuggingFace model id)
      - "openai": OpenAIEmbedding (any OpenAI-compatible /v1/embeddings endpoint)

    Also updates config.embedding_dims to match the actual model output
    dimensions (Google gemini-embedding-001 returns 3072; BGE-M3 returns 1024;
    OpenAI text-embedding-3-small returns 1536).
    """
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    provider = getattr(config, 'default_embedder', 'google') if config else 'google'

    if provider == "bge-m3":
        _embedder_instance = _create_bge_m3(config)
    elif provider == "openai":
        _embedder_instance = _create_openai(config)
    else:
        _embedder_instance = _create_google(config)

    # Auto-detect embedding dimensions from the embedder
    if config is not None and hasattr(config, 'embedding_dims'):
        try:
            test_vec = _embedder_instance.embed("dim-probe")
            actual_dims = len(test_vec)
            if config.embedding_dims != actual_dims:
                logger.info(
                    "Embedding dims updated: %d -> %d (model: %s)",
                    config.embedding_dims, actual_dims,
                    getattr(_embedder_instance, 'model', provider),
                )
                config.embedding_dims = actual_dims
        except Exception:
            pass

    return _embedder_instance


def _create_google(config) -> "GoogleEmbedding":
    api_key = getattr(config, 'google_api_key', '') if config else ''
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Google embedding selected but GOOGLE_API_KEY is not set. "
            "Set it via config.json key 'google_api_key', the env var "
            "GOOGLE_API_KEY, or switch to bge-m3 via config.default_embedder."
        )

    model = (getattr(config, 'google_embedding_model', 'gemini-embedding-001')
             if config else 'gemini-embedding-001')

    from _core.embeddings.google import GoogleEmbedding
    instance = GoogleEmbedding(api_key=api_key, model=model)
    logger.info("Embedder initialized: Google %s", model)
    return instance


def _create_bge_m3(config) -> "BGEM3Embedding":
    path = getattr(config, 'bge_m3_path', '') if config else ''
    if not path:
        path = os.environ.get("BGE_M3_PATH", "")
    # No local path -> fetch BGE-M3 from HuggingFace Hub (sentence-transformers
    # downloads & caches it on first use). BGE_M3_PATH accepts either a local
    # model dir (loaded fully offline) or a HuggingFace model id (e.g.
    # "BAAI/bge-m3", pulled from the Hub). Defaulting to the Hub model makes
    # bge-m3 work out of the box without a pre-staged model directory.
    if not path:
        path = "BAAI/bge-m3"
        logger.info(
            "BGE-M3: no local model path configured - will fetch 'BAAI/bge-m3' "
            "from HuggingFace Hub on first use (set BGE_M3_PATH to a local dir "
            "to skip the download)."
        )
    from _core.embeddings.bge_m3 import BGEM3Embedding
    instance = BGEM3Embedding(model_path=path)
    logger.info("Embedder initialized: BGE-M3 from %s", path)
    return instance


def _create_openai(config) -> "OpenAIEmbedding":
    """OpenAI-compatible embedding endpoint.

    Works with the OpenAI API and any backend that speaks its /v1/embeddings
    contract (vLLM, Ollama's OpenAI shim, LM Studio, ...). Point
    OPENAI_BASE_URL at the compatible server for non-OpenAI backends.
    """
    api_key = (getattr(config, 'openai_api_key', '') if config else '') \
        or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OpenAI embedding selected but OPENAI_API_KEY is not set. "
            "Set the env var OPENAI_API_KEY (or config.openai_api_key), "
            "or switch providers via EMBEDDING_PROVIDER."
        )
    base_url = (getattr(config, 'openai_base_url', '') if config else '') \
        or os.environ.get("OPENAI_BASE_URL", "")
    model = (getattr(config, 'openai_embedding_model', 'text-embedding-3-small')
             if config else 'text-embedding-3-small') or 'text-embedding-3-small'

    from _core.configs.embeddings.base import BaseEmbedderConfig
    from _core.embeddings.openai import OpenAIEmbedding
    # embedding_dims=None -> don't send `dimensions` to the API (non-matryoshka
    # backends like vLLM/Voyage reject the param). get_embedder() auto-detects
    # the real dimensionality via a probe call right after construction.
    cfg = BaseEmbedderConfig(
        model=model,
        api_key=api_key,
        openai_base_url=base_url or None,
        embedding_dims=None,
    )
    instance = OpenAIEmbedding(cfg)
    logger.info(
        "Embedder initialized: OpenAI-compatible %s @ %s",
        model, base_url or "https://api.openai.com/v1",
    )
    return instance


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two normalized vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np))
