"""Google Gemini embedding adapter.

Supports Gemini embedding models via the google-genai SDK:
  - gemini-embedding-001  (V1, uses task_type in EmbedContentConfig)
  - gemini-embedding-2    (V2, uses text prefix format)

API reference: https://ai.google.dev/gemini-api/docs/embeddings
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Literal, Optional

from _core.embeddings.base import EmbeddingBase

logger = logging.getLogger("deepmem.embedder.google")


# Text prefix helpers for Gemini Embedding V2
def _v2_query(text: str) -> str:
    return f"task: search result | query: {text}"


def _v2_doc(text: str) -> str:
    return f"title: none | text: {text}"


class GoogleEmbedding(EmbeddingBase):
    """Google Gemini embedding via google-genai SDK."""

    _LOCK = asyncio.Lock()

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
    ):
        super().__init__()
        self._api_key = api_key
        self.model = model
        self._client = None
        self._is_v2 = "-2" in model or "embedding-2" in model
        logger.info(
            "GoogleEmbedding: model=%s api_key=%s... v2=%s",
            model, api_key[:8] if api_key else "<unset>", self._is_v2,
        )

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def embed(
        self,
        text: str,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> List[float]:
        """Embed a single text. Not the hot path — use embed_batch for bulk."""
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(
        self,
        texts: List[str],
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> List[List[float]]:
        """Batch embed multiple texts (native Google API call)."""
        if not texts:
            return []

        mode = _memory_action_to_mode(memory_action)

        if self._is_v2:
            if mode == "query":
                texts = [_v2_query(t) for t in texts]
            elif mode == "doc":
                texts = [_v2_doc(t) for t in texts]
            from google.genai import types
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in texts
            ]
            result = self.client.models.embed_content(
                model=self.model, contents=contents,
            )
        else:
            task = "RETRIEVAL_QUERY" if mode == "query" else "RETRIEVAL_DOCUMENT"
            from google.genai import types
            result = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task),
            )

        return [e.values for e in result.embeddings]

    async def aembed(
        self,
        text: str,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> List[float]:
        """Async embed — serialized by lock to avoid flooding the API."""
        async with GoogleEmbedding._LOCK:
            return await asyncio.to_thread(self.embed, text, memory_action)

    async def aembed_batch(
        self,
        texts: List[str],
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> List[List[float]]:
        """Async batch embed — serialized by lock."""
        async with GoogleEmbedding._LOCK:
            return await asyncio.to_thread(self.embed_batch, texts, memory_action)


def _memory_action_to_mode(
    action: Optional[Literal["add", "search", "update"]],
) -> str:
    """Map DeepMemory's memory_action to the Google embedding query/doc mode.

    - "search" → "query" (user is searching, the input is a query)
    - "add" / "update" / None → "doc" (storing a memory, the input is a document)
    """
    if action == "search":
        return "query"
    return "doc"
