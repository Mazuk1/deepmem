from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    text: str
    user_id: str
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    memory: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tenant:
    user_id: str
    agent_id: Optional[str] = None
    # Session / workflow scope. When set, writes are partitioned and reads
    # are filtered to this run only. None means "whole user". Mirrors mem0's
    # user_id / agent_id / run_id three-level isolation.
    run_id: Optional[str] = None
    # Optional account-level isolation. When None, behavior is single-tenant
    # mode — Qdrant filter ignores account_id entirely so existing self-hosted
    # data keeps matching. When set, every read/write is hard-filtered by it
    # so customer A literally cannot read customer B's records, even if both
    # of them happen to use the same downstream user_id string.
    account_id: Optional[str] = None
    # ── Multi-agent shared-memory groundwork (not enforced yet) ───────
    # These four fields are stamped onto every stored payload so a future
    # permission layer ("which agents can see which memories") can be built
    # on top without a schema migration. They are WRITE-TIME attributes only:
    # NO query / scope filter reads them today, so existing behavior is
    # unchanged. Stamping them now is cheap; adding them after the storage
    # layer is set in stone would require a backfill over every point.
    # owner_id        - which agent instance wrote this memory
    # visibility_scope - who may read it (default "private")
    # source_type     - "user" (stated by user) vs "agent" (inferred by agent)
    # session_id      - which conversation/session this memory belongs to
    owner_id: Optional[str] = None
    visibility_scope: str = "private"
    source_type: str = "user"
    session_id: Optional[str] = None


class MemoryEngine(ABC):
    """Universal memory engine interface — dependency inversion boundary.

    Business logic depends on this interface, not on concrete implementations.
    """

    @abstractmethod
    async def add(self, messages: List[Dict[str, str]], tenant: Tenant,
                  metadata: Optional[Dict[str, Any]] = None,
                  infer: bool = True,
                  llm_provider: Optional[Any] = None) -> List[str]:
        """Write memories extracted from messages. Returns memory IDs.

        When infer=False, stores raw message content without LLM extraction.
        llm_provider lets callers (BYOK / ModelRouter) override the LLM adapter
        without binding the implementation to a single concrete provider.
        """
        ...

    @abstractmethod
    async def search(self, query: str, tenant: Tenant,
                     top_k: int = 10, threshold: float = 0.3,
                     **kwargs) -> List[SearchResult]:
        """Semantic search for memories. Returns ranked results."""
        ...

    @abstractmethod
    async def get(self, memory_id: str, tenant: Tenant) -> Optional[SearchResult]:
        """Retrieve a single memory by ID."""
        ...

    @abstractmethod
    async def update(self, memory_id: str, new_memory: str, tenant: Tenant) -> bool:
        """Update the text of a single memory."""
        ...

    @abstractmethod
    async def list(self, tenant: Tenant, limit: int = 100, offset: int = 0) -> List[SearchResult]:
        """List all memories for a tenant with pagination."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str, tenant: Tenant) -> bool:
        """Delete a single memory by ID."""
        ...

    @abstractmethod
    async def delete_all(self, tenant: Tenant) -> int:
        """Delete all memories for a tenant (GDPR Right to Erasure). Returns count."""
        ...
