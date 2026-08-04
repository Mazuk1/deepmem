"""Memory-backend adapters for the benchmark harness.

Every backend (DeepMem, Mem0, Zep, ...) is wrapped behind the same
`MemoryAdapter` protocol so `run_benchmark.py` can compare them apples-to-
apples: the same workload, the same latency instrumentation, the same
percentile math.

Only the DeepMem adapter is implemented today - it talks to a running
DeepMem HTTP server. The Mem0 / Zep adapters are sketched as stubs with
install hints, so anyone who wants a three-way comparison can fill them in
without touching the runner.

Why an adapter and not a direct HTTP client: Mem0 and Zep are SDKs / servers
with their own connection setup, batching, and async semantics. Isolating
that behind `add()` / `search()` keeps the benchmark honest - we measure the
backend's real cost, not our client's cleverness.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import httpx


@runtime_checkable
class MemoryAdapter(Protocol):
    """The two operations every memory backend must expose for the benchmark."""

    name: str

    async def add(self, messages: List[dict], user_id: str,
                  infer: bool = False) -> None: ...

    async def search(self, query: str, user_id: str,
                     top_k: int = 5) -> List[dict]: ...

    async def close(self) -> None: ...


class DeepMemAdapter:
    """Talks to a running DeepMem server over HTTP (open mode, no API key).

    Point it at your server with DEEPMEM_BASE_URL (default http://localhost:8000).
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 timeout: float = 60.0):
        self.name = "deepmem"
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def add(self, messages: List[dict], user_id: str,
                  infer: bool = False) -> None:
        r = await self._client.post(
            f"{self.base_url}/v1/memories",
            json={"messages": messages, "user_id": user_id, "infer": infer},
        )
        r.raise_for_status()

    async def search(self, query: str, user_id: str,
                     top_k: int = 5) -> List[dict]:
        r = await self._client.post(
            f"{self.base_url}/v1/memories/search",
            json={"query": query, "user_id": user_id, "top_k": top_k},
        )
        r.raise_for_status()
        return r.json().get("results", [])

    async def close(self) -> None:
        await self._client.aclose()


# ── Cloud adapters (DeepMem cloud vs Mem0 cloud) ───────────────────────
# These talk to the hosted services over HTTPS with API-key auth. Register
# to get keys (see benchmarks/README.md "Cloud comparison"), then use them
# for an apples-to-apples hosted-vs-hosted comparison. httpx honors
# HTTPS_PROXY/HTTP_PROXY env vars, so set those if you're behind a proxy.


class DeepMemCloudAdapter:
    """DeepMem hosted API (https://deepmem.dev). Mem0-compatible endpoints.

    Needs DEEPMEM_API_KEY (dm_live_...). add(infer=False) stores raw (sync,
    immediately searchable) - the fair storage-latency comparison; add(infer=True)
    queues for LLM extraction (async, lands after the batch silence window).
    """

    def __init__(self, api_key: str, base_url: str = "https://deepmem.dev",
                 timeout: float = 60.0):
        self.name = "deepmem-cloud"
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def add(self, messages, user_id, infer=False):
        r = await self._client.post(
            f"{self.base_url}/v1/memories",
            headers=self._headers,
            json={"messages": messages, "user_id": user_id, "infer": infer},
        )
        r.raise_for_status()

    async def search(self, query, user_id, top_k=5):
        r = await self._client.post(
            f"{self.base_url}/v1/memories/search",
            headers=self._headers,
            json={"query": query, "user_id": user_id, "top_k": top_k},
        )
        r.raise_for_status()
        payload = r.json()
        return payload.get("results", []) if isinstance(payload, dict) else payload

    async def close(self):
        await self._client.aclose()


class Mem0CloudAdapter:
    """Mem0 hosted API (https://api.mem0.ai). Needs MEM0_API_KEY (m0-...).

    Mem0 always runs LLM extraction on add (no infer=False), so add latency
    includes extraction. search returns a bare JSON list.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.mem0.ai",
                 timeout: float = 60.0):
        self.name = "mem0-cloud"
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        self._headers = {"Authorization": f"Token {api_key}"}

    async def add(self, messages, user_id, infer=False):
        # Mem0 has no infer flag; it always extracts. The `infer` arg is
        # accepted for protocol compatibility and ignored.
        r = await self._client.post(
            f"{self.base_url}/v1/memories/",
            headers=self._headers,
            json={"messages": messages, "user_id": user_id},
        )
        r.raise_for_status()

    async def search(self, query, user_id, top_k=5):
        r = await self._client.post(
            f"{self.base_url}/v1/memories/search/",
            headers=self._headers,
            json={"query": query, "user_id": user_id},
        )
        r.raise_for_status()
        payload = r.json()
        return payload if isinstance(payload, list) else (payload.get("results") or payload.get("memories") or [])

    async def close(self):
        await self._client.aclose()


class ZepAdapter:
    """Stub. Install: pip install zep-cloud (or run a self-hosted Zep server).
    """
    name = "zep"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ZepAdapter not wired. See benchmarks/README.md for the Zep "
            "client setup and implement add/search."
        )
