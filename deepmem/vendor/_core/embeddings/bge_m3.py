"""BGE-M3 self-hosted embedding provider — zero embedding cost.

Auto-detects GPU via torch.cuda.is_available().
"""

import asyncio
import logging
import os
from typing import List, Literal, Optional

# IMPORTANT: import torch BEFORE sentence_transformers. On Windows the reverse
# order has been observed to crash the interpreter (SIGSEGV during DLL load)
# when sentence_transformers triggers torch's lazy CUDA init from inside a
# host process that has already loaded fastapi/uvicorn. test_bge.py works
# precisely because it imports torch first; matching that order here keeps
# `python -m uvicorn server.main:app` from segfaulting on boot.
import torch  # noqa: F401  (load order matters even though we use it lazily)

# When the user supplies a fully populated local model dir (the common case
# for self-hosted deploys), do NOT let sentence_transformers / huggingface_hub
# phone home — a flaky network can otherwise hang the boot for minutes while
# it tries to refresh the model card. The env vars are read at HF's import
# time, so set them before the import below.
_local_path = os.environ.get("BGE_M3_PATH") or os.environ.get("GTE_MODEL_PATH", "")
if _local_path and os.path.isdir(_local_path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

from _core.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__)


# BGE-style retrieval instruction. Applied to the query side only when the
# caller passes memory_action="search" — passages stored at write time are
# embedded without the prefix, matching how the model was trained for
# instruction-aware retrieval. Costs nothing at inference, lifts recall
# 2-3pt on Chinese/multilingual queries.
_QUERY_INSTRUCTION_EN = "Represent this sentence for searching relevant passages: "
_QUERY_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


def _has_cjk(text: str) -> bool:
    for ch in text:
        if "一" <= ch <= "鿿":
            return True
    return False


def _apply_query_instruction(text: str) -> str:
    if not text:
        return text
    prefix = _QUERY_INSTRUCTION_ZH if _has_cjk(text) else _QUERY_INSTRUCTION_EN
    return prefix + text


def _resolve_device() -> str:
    """Pick the BGE-M3 device from the BGE_DEVICE env var.

    BGE_DEVICE=cpu   -> force CPU (avoids GPU contention / OOM on small VRAM;
                       load is faster, inference ~0.3s/batch - fine for tests
                       and low-throughput deploys).
    BGE_DEVICE=cuda  -> force CUDA (fails if unavailable).
    BGE_DEVICE=auto  (default) -> CUDA if available, else CPU.

    On a single 8GB GPU, two processes each loading BGE-M3 (~3.1GB VRAM each)
    can OOM or stall; forcing CPU on one of them removes that contention.
    """
    env = os.environ.get("BGE_DEVICE", "auto").strip().lower()
    if env == "cpu":
        return "cpu"
    if env in ("cuda", "gpu"):
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEM3Embedding(EmbeddingBase):
    """BGE-M3 — 1024-dim, multilingual, self-hosted."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(None)
        if model_path is None:
            model_path = os.environ.get("BGE_M3_PATH") or os.environ.get("GTE_MODEL_PATH", "")
        if not model_path:
            raise RuntimeError(
                "BGE-M3 model_path is required. Pass model_path explicitly or "
                "set the BGE_M3_PATH env var."
            )
        self.model_path = model_path
        self._model: Optional[SentenceTransformer] = None
        # Serialize GPU access across concurrent async requests. SentenceTransformer
        # .encode() is sync and holds the GIL while the GPU runs (50-500ms on CPU,
        # 5-30ms on GPU). The lock is lazily attached on first aembed() call so
        # the embedder can still be constructed outside a running event loop.
        self._aio_lock: Optional[asyncio.Lock] = None

    def _get_aio_lock(self) -> asyncio.Lock:
        if self._aio_lock is None:
            self._aio_lock = asyncio.Lock()
        return self._aio_lock

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            import time
            device = _resolve_device()
            logger.info(
                "Loading BGE-M3 from %s on %s (BGE_DEVICE=%s, HF_HUB_OFFLINE=%s)",
                self.model_path, device,
                os.environ.get("BGE_DEVICE", "auto"),
                os.environ.get("HF_HUB_OFFLINE", "0"),
            )
            t0 = time.monotonic()
            self._model = SentenceTransformer(
                self.model_path,
                device=device,
            )
            logger.info("BGE-M3 loaded on %s in %.1fs", device, time.monotonic() - t0)
        return self._model

    def embed(self, text: str,
              memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[float]:
        encoded_text = _apply_query_instruction(text) if memory_action == "search" else text
        embedding = self.model.encode(
            encoded_text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str],
                    memory_action: str = "add") -> List[List[float]]:
        if memory_action == "search":
            texts = [_apply_query_instruction(t) for t in texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()

    async def aembed(self, text: str,
                     memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[float]:
        """Async wrapper around .embed for use inside FastAPI request handlers.

        Holds an asyncio.Lock around asyncio.to_thread so concurrent requests
        serialize on the GPU/CPU encoder (one inference at a time) without
        blocking the event loop while the call is in flight.
        """
        async with self._get_aio_lock():
            return await asyncio.to_thread(self.embed, text, memory_action)

    async def aembed_batch(self, texts: List[str],
                            memory_action: str = "add") -> List[List[float]]:
        async with self._get_aio_lock():
            return await asyncio.to_thread(self.embed_batch, texts, memory_action)
