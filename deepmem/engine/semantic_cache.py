import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from deepmem.embedder import cosine_similarity

logger = logging.getLogger("deepmem.engine.cache")


class SemanticCache:
    """Semantic cache with cosine similarity matching + optional disk persistence.

    Stores an arbitrary `value` (List[str] of facts for the add path,
    List[dict] of search items for the search path) so callers can put
    real ids/scores in the cache rather than re-fabricating them on hit.

    Tenant isolation: entries are bucketed by (account_id, user_id) so two
    accounts that happen to share a user_id string never see each
    other's cached results. account_id=None preserves the single-
    tenant bucket.

    Disk persistence: when `persist_path` is set, the cache is written as
    JSON on every store (debounced: max one write per 5 seconds) and loaded
    from disk on init. Survives server restarts.
    """

    _PERSIST_DEBOUNCE = 5.0  # seconds between disk writes

    def __init__(
        self,
        similarity_threshold: float = 0.98,
        ttl_seconds: int = 300,
        persist_path: str = "",
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[Tuple[Optional[str], str], Dict] = {}
        self._persist_path = persist_path
        self._persist_task: Optional[asyncio.Task] = None
        self._dirty = False

        if self._persist_path:
            self._load_from_disk()

    # ── Disk I/O ──────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if not os.path.exists(self._persist_path):
            logger.info("Cache persist file not found, starting empty: %s",
                        self._persist_path)
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            loaded = 0
            expired = 0
            now = time.time()
            for bucket_str, entries in raw.items():
                # Restore tuple key from JSON string repr
                parts = bucket_str.split("\x00", 1)
                if len(parts) == 2:
                    bk = (parts[0] if parts[0] != "__none__" else None, parts[1])
                else:
                    bk = (None, bucket_str)
                bucket: dict = {}
                for key, entry in entries.items():
                    if now - entry.get("timestamp", 0) > self.ttl_seconds:
                        expired += 1
                        continue
                    bucket[key] = entry
                if bucket:
                    self._cache[bk] = bucket
                    loaded += len(bucket)
            logger.info(
                "Cache loaded from disk: %d entries (%d expired), %d buckets: %s",
                loaded, expired, len(self._cache), self._persist_path,
            )
        except Exception as e:
            logger.warning("Failed to load cache from %s: %s", self._persist_path, e)

    def _schedule_persist(self) -> None:
        """Schedule a debounced disk write."""
        if not self._persist_path:
            return
        self._dirty = True
        if self._persist_task is None or self._persist_task.done():
            self._persist_task = asyncio.ensure_future(self._persist_after_delay())

    async def _persist_after_delay(self) -> None:
        await asyncio.sleep(self._PERSIST_DEBOUNCE)
        self._write_to_disk()

    def _write_to_disk(self) -> None:
        if not self._dirty or not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            # Serialize tuple keys as strings (JSON only allows string keys)
            raw: Dict[str, dict] = {}
            for (account_id, user_id), bucket in self._cache.items():
                bucket_str = f"{account_id or '__none__'}\x00{user_id}"
                raw[bucket_str] = bucket
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False)
            self._dirty = False
            logger.debug("Cache persisted: %d buckets to %s",
                         len(self._cache), self._persist_path)
        except Exception as e:
            logger.warning("Failed to persist cache to %s: %s",
                          self._persist_path, e)

    def flush(self) -> None:
        """Force immediate disk write (for graceful shutdown)."""
        self._write_to_disk()

    # ── Bucket key ────────────────────────────────────────────────────

    @staticmethod
    def _bucket_key(user_id: str,
                    account_id: Optional[str]) -> Tuple[Optional[str], str]:
        return (account_id, user_id)

    def _make_key(self, messages: List[Dict[str, str]]) -> str:
        content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # ── Public API ────────────────────────────────────────────────────

    async def check(self, messages: List[Dict[str, str]],
                    query_vector: List[float], user_id: str,
                    account_id: Optional[str] = None) -> Optional[Any]:
        """Check if a semantically similar request exists in cache.

        Returns the cached value (List[str] or List[dict]) or None.
        """
        bucket = self._cache.get(self._bucket_key(user_id, account_id), {})
        now = time.time()

        for key, entry in list(bucket.items()):
            if now - entry["timestamp"] > self.ttl_seconds:
                del bucket[key]
                self._schedule_persist()
                continue
            similarity = cosine_similarity(query_vector, entry["vector"])
            if similarity >= self.similarity_threshold:
                logger.info(
                    "Cache HIT user=%s account=%s similarity=%.4f",
                    user_id, account_id or "-", similarity,
                )
                return entry["value"]

        return None

    async def store(self, messages: List[Dict[str, str]],
                    query_vector: List[float], user_id: str,
                    value: Any,
                    account_id: Optional[str] = None) -> None:
        """Store a result in the cache. `value` may be List[str] or List[dict]."""
        bk = self._bucket_key(user_id, account_id)
        if bk not in self._cache:
            self._cache[bk] = {}
        key = self._make_key(messages)
        self._cache[bk][key] = {
            "vector": query_vector,
            "value": value,
            "timestamp": time.time(),
        }
        self._schedule_persist()

    def clear(self, user_id: Optional[str] = None,
              account_id: Optional[str] = None):
        """Clear cache for a user (within an account) or everything.

        - clear(): wipe everything
        - clear(user_id="alice"): wipe alice across every account bucket
        - clear(user_id="alice", account_id="acct-123"): wipe just that bucket
        """
        if user_id is None:
            self._cache.clear()
        elif account_id is not None:
            self._cache.pop(self._bucket_key(user_id, account_id), None)
        else:
            for bk in [k for k in self._cache if k[1] == user_id]:
                self._cache.pop(bk, None)
        self._schedule_persist()
