"""Mem0 compatibility layer - drop-in replacement for `from mem0 import Memory`.

Goal: a Mem0 user can switch to DeepMem by changing one import line and
keeping the same method calls:

    # before
    from mem0 import Memory
    m = Memory()
    m.add(messages, user_id="pat")

    # after
    from deepmem.compat import mem0_client as m
    m.add(messages, user_id="pat")

The method signatures mirror Mem0's Memory client (add / search / get_all /
get / update / delete / delete_all / reset). Responses are reshaped to Mem0's
`{"results": [...]}` envelope so existing parsing code keeps working.

What is NOT identical (and can't be, honestly):
  - `add(infer=True)` (default) fires DeepMem's LLM fact extraction. Under
    DeepMem's async batch distiller the extracted facts may land a few seconds
    later (silence window), so `add` can return with `results=[]` and the
    facts appearing on the next `search`. Pass `infer=False` for synchronous
    raw-text storage that returns immediately.
  - Mem0's graph relations are not synthesized - `"relations": []` is always
    returned. DeepMem uses hybrid vector retrieval, not a knowledge graph.

Talks to a running DeepMem server over HTTP (open mode, no API key needed).
Configure with DEEPMEMORY_BASE_URL (default http://localhost:8000).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

__all__ = ["Mem0CompatClient", "mem0_client"]


class Mem0CompatClient:
    """Sync, Mem0-shaped client over the DeepMem HTTP API."""

    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key or os.environ.get("DEEPMEMORY_API_KEY", "")
        self._client = httpx.Client(timeout=timeout)

    # ── internal ──

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        r = self._client.post(f"{self.base_url}{path}", json=body,
                              headers=self._headers())
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = self._client.get(f"{self.base_url}{path}",
                             headers=self._headers())
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        r = self._client.delete(f"{self.base_url}{path}",
                                headers=self._headers())
        r.raise_for_status()
        return r.json()

    # ── Mem0-shaped public API ──

    def add(self, messages: List[Dict[str, str]], user_id: Optional[str] = None,
            agent_id: Optional[str] = None, run_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            infer: bool = True) -> Dict[str, Any]:
        """Write messages. Mirrors `mem0.Memory.add`.

        infer=True (default) runs LLM fact extraction (may be batched/async);
        infer=False stores raw text synchronously and returns immediately.
        """
        body: dict = {"messages": messages, "infer": infer}
        if user_id is not None:
            body["user_id"] = user_id
        if agent_id is not None:
            body["agent_id"] = agent_id
        if run_id is not None:
            body["run_id"] = run_id
        if metadata is not None:
            body["metadata"] = metadata
        resp = self._post("/v1/memories", body)
        # Mem0 envelope: {"results": [...], "relations": [...]}
        resp.setdefault("relations", [])
        return resp

    def search(self, query: str, user_id: Optional[str] = None,
               limit: int = 100, agent_id: Optional[str] = None,
               run_id: Optional[str] = None) -> Dict[str, Any]:
        """Semantic search. Mirrors `mem0.Memory.search`."""
        body: dict = {"query": query, "top_k": limit}
        if user_id is not None:
            body["user_id"] = user_id
        if agent_id is not None:
            body["agent_id"] = agent_id
        if run_id is not None:
            body["run_id"] = run_id
        return self._post("/v1/memories/search", body)

    def get_all(self, user_id: Optional[str] = None,
                agent_id: Optional[str] = None,
                run_id: Optional[str] = None,
                limit: int = 100) -> Dict[str, Any]:
        """List memories. Mirrors `mem0.Memory.get_all`."""
        params = {"limit": limit}
        if user_id is not None:
            params["user_id"] = user_id
        if agent_id is not None:
            params["agent_id"] = agent_id
        if run_id is not None:
            params["run_id"] = run_id
        # DeepMem GET /v1/memories takes query params.
        r = self._client.get(f"{self.base_url}/v1/memories", params=params,
                             headers=self._headers())
        r.raise_for_status()
        return r.json()

    def get(self, memory_id: str) -> Dict[str, Any]:
        """Get one memory by id. Mirrors `mem0.Memory.get`."""
        return self._get(f"/v1/memories/{memory_id}")

    def update(self, memory_id: str, text: str) -> Dict[str, Any]:
        """Update one memory's text. Mirrors `mem0.Memory.update`."""
        r = self._client.put(f"{self.base_url}/v1/memories/{memory_id}",
                             json={"memory": text}, headers=self._headers())
        r.raise_for_status()
        return {"results": [r.json()], "relations": []}

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Soft-delete one memory. Mirrors `mem0.Memory.delete`."""
        self._delete(f"/v1/memories/{memory_id}")
        return {"results": [{"id": memory_id, "status": "deleted"}]}

    def delete_all(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Soft-delete all memories for a user. Mirrors `mem0.Memory.delete_all`."""
        path = "/v1/memories"
        if user_id is not None:
            path += f"?user_id={user_id}"
        self._delete(path)
        return {"status": "done"}

    def reset(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Hard-delete all records + history for a user. Mirrors `mem0.Memory.reset`.

        DeepMem's reset requires confirm_user_id == user_id as a safety guard.
        """
        if user_id is None:
            raise ValueError(
                "DeepMem reset requires a user_id (the confirm guard needs it)."
            )
        self._post("/v1/reset", {"user_id": user_id, "confirm_user_id": user_id})
        return {"status": "done"}

    def close(self) -> None:
        self._client.close()

    # Context-manager convenience so users can `with mem0_client as m:`
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# Pre-built singleton - the one-line migration target.
# `from deepmem.compat import mem0_client` and use it directly.
mem0_client = Mem0CompatClient(
    base_url=os.environ.get("DEEPMEMORY_BASE_URL", "http://localhost:8000"),
)
