"""VectorStore — MemoryEngine implementation backed by Qdrant + LLM extraction."""

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from deepmem.interface import MemoryEngine, SearchResult, Tenant
from deepmem.embedder import get_embedder
from deepmem.extraction_filter import prepare_messages_for_fact_extraction

logger = logging.getLogger(__name__)

# Default values for the multi-agent permission groundwork fields stamped
# onto every payload. See Tenant.owner_id / visibility_scope / source_type /
# session_id - these are write-only today (no query reads them); they exist
# so a future shared-memory permission layer can be added without a backfill.
DEFAULT_VISIBILITY_SCOPE = "private"
DEFAULT_SOURCE_TYPE = "user"


class VectorStore(MemoryEngine):
    """Qdrant-backed memory engine with LLM fact extraction and semantic search.

    Uses local file-based Qdrant for development. Multi-tenant isolation
    via user_id payload filter — single collection, hard-filtered per user.
    """

    def __init__(self, qdrant_path: str = "./data/qdrant",
                 collection_name: str = "memories",
                 embedding_dims: int = 1024,
                 bge_m3_path: Optional[str] = None,
                 history_store: Optional[Any] = None,
                 qdrant_url: str = "",
                 qdrant_api_key: str = ""):
        self.qdrant_path = qdrant_path
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_dims = embedding_dims
        self._bge_m3_path = bge_m3_path
        # Optional HistoryStore — when set, ADD/UPDATE/DELETE events get
        # recorded for audit / time-travel reads. None disables history
        # so unit tests that don't care about audit aren't forced to mock it.
        self._history = history_store

        # QdrantClient takes either path= (embedded file mode) or url= (remote
        # service); the two are mutually exclusive. QDRANT_URL wins when set
        # so production can swap to the docker-compose qdrant container without
        # any code change.
        if qdrant_url:
            client_kwargs: Dict[str, Any] = {"url": qdrant_url}
            if qdrant_api_key:
                client_kwargs["api_key"] = qdrant_api_key
            self._client = QdrantClient(**client_kwargs)
            logger.info(
                "Qdrant: connected to remote %s (api_key=%s)",
                qdrant_url, "set" if qdrant_api_key else "none",
            )
        else:
            os.makedirs(qdrant_path, exist_ok=True)
            self._client = QdrantClient(path=qdrant_path)
            logger.info("Qdrant: opened local file store at %s", qdrant_path)
        self._ensure_collection()

        # Lazy-loaded embedder and LLM
        self._embedder = None
        self._llm = None

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self.collection_name not in names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dims,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
        # Create payload index for user_id (only works on remote, ignored on local)
        try:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema="keyword",
            )
        except Exception:
            pass  # Local Qdrant doesn't support payload indexes

    @property
    def embedder(self):
        # The embedder singleton is normally pre-created by
        # dependencies.get_services() with the FULL config, so this lazy
        # fallback is only hit when a VectorStore is constructed directly
        # (e.g. in tests) without that wiring. The synthetic config here MUST
        # carry the configured provider (default_embedder) - otherwise
        # get_embedder falls back to its 'google' dataclass default and
        # silently ignores EMBEDDING_PROVIDER=bge-m3, producing 3072-dim
        # vectors into a 1024-dim collection (ValueError on upsert).
        if self._embedder is None:
            from deepmem.config import config as _cfg
            if self._bge_m3_path:
                class _EmbedderConfig:
                    default_embedder = _cfg.default_embedder
                    bge_m3_path = self._bge_m3_path
                    google_api_key = _cfg.google_api_key
                    google_embedding_model = _cfg.google_embedding_model
                    openai_api_key = _cfg.openai_api_key
                    openai_base_url = _cfg.openai_base_url
                    openai_embedding_model = _cfg.openai_embedding_model
                cfg = _EmbedderConfig()
            else:
                cfg = _cfg
            self._embedder = get_embedder(cfg)
        return self._embedder

    @property
    def llm(self):
        if self._llm is None:
            from deepmem.config import config as _cfg
            from deepmem.llm import create_llm_from_config
            # Honor LLM_PROVIDER / ANTHROPIC_* / LLM_* config (multi-provider);
            # falls back to deepseek_* when only those are set.
            self._llm = create_llm_from_config(_cfg)
        return self._llm

    def _raw_facts(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Fallback when LLM extraction yields nothing — store the raw turns.

        Returns dict facts so the rest of the pipeline is uniform. Attribution
        comes from the message role (user / assistant), defaulting to user.
        """
        out: List[Dict[str, str]] = []
        for m in messages:
            content = (m.get("content") or "").strip()
            if not content:
                continue
            role = m.get("role", "user")
            attributed = "assistant" if role == "assistant" else "user"
            out.append({
                "text": f"{role}: {content}",
                "attributed_to": attributed,
            })
        return out

    @staticmethod
    def _normalize_fact(item) -> Optional[Dict[str, str]]:
        """Coerce a fact (dict or string) into a normalized dict."""
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return {"text": text, "attributed_to": "user"}
        if isinstance(item, dict):
            text = (item.get("text") or item.get("fact") or "").strip()
            if not text:
                return None
            attributed = item.get("attributed_to", "user")
            if attributed not in ("user", "assistant"):
                attributed = "user"
            return {"text": text, "attributed_to": attributed}
        return None

    @staticmethod
    def _fact_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _existing_hashes(self, user_id: str,
                         candidate_hashes: List[str],
                         account_id: Optional[str] = None) -> set:
        """Lookup which of the given hashes already exist for this tenant scope.

        Uses a hash MatchAny filter so we only fetch the relevant points
        instead of scrolling the whole user partition. Caller deduplicates
        before upsert (1.2 hash precise dedup). When account_id is set,
        dedup is scoped to that account so two accounts can each
        store the same fact text without colliding.
        """
        if not candidate_hashes:
            return set()
        try:
            must_clauses: List[FieldCondition] = []
            if account_id:
                must_clauses.append(FieldCondition(
                    key="account_id", match=MatchValue(value=account_id),
                ))
            must_clauses.extend([
                FieldCondition(key="user_id",
                               match=MatchValue(value=user_id)),
                FieldCondition(key="hash",
                               match=MatchAny(any=candidate_hashes)),
                FieldCondition(key="deleted",
                               match=MatchValue(value=False)),
            ])
            page, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must_clauses),
                limit=len(candidate_hashes),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            # Hash filter failed (e.g. legacy points without hash payload on
            # an older local Qdrant). Fall back to an empty set so we don't
            # silently lose new writes — duplicates are tolerable, data loss
            # is not.
            logger.warning("Hash dedup lookup failed for user=%s: %s", user_id, e)
            return set()
        return {p.payload.get("hash") for p in page if p.payload.get("hash")}

    def _dedup_facts(self, fact_dicts: List[Dict[str, str]],
                     user_id: str,
                     account_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Drop facts whose text hash already exists in this tenant's store
        or appears earlier in this same batch."""
        if not fact_dicts:
            return []
        for fd in fact_dicts:
            fd.setdefault("hash", self._fact_hash(fd["text"]))
        existing = self._existing_hashes(
            user_id, [fd["hash"] for fd in fact_dicts], account_id=account_id,
        )
        seen: set = set()
        kept: List[Dict[str, str]] = []
        skipped = 0
        for fd in fact_dicts:
            h = fd["hash"]
            if h in existing or h in seen:
                skipped += 1
                continue
            seen.add(h)
            kept.append(fd)
        if skipped:
            logger.info(
                "Hash dedup user=%s — skipped %d duplicate fact(s) of %d",
                user_id, skipped, len(fact_dicts),
            )
        return kept

    def _build_points(self, fact_dicts: List[Dict[str, str]], vectors, *,
                      user_id: str,
                      agent_id: Optional[str] = None,
                      run_id: Optional[str] = None,
                      account_id: Optional[str] = None,
                      llm_used: bool,
                      metadata: Optional[Dict[str, Any]] = None,
                      owner_id: Optional[str] = None,
                      visibility_scope: str = DEFAULT_VISIBILITY_SCOPE,
                      source_type: str = DEFAULT_SOURCE_TYPE,
                      session_id: Optional[str] = None):
        import time as _time
        ids: List[str] = []
        points: List[PointStruct] = []
        now = _time.time()
        # Normalize groundwork fields - callers (HTTP / distiller) may pass
        # None when unset; payload always carries concrete values. owner_id
        # and session_id may legitimately stay None.
        if not visibility_scope:
            visibility_scope = DEFAULT_VISIBILITY_SCOPE
        if not source_type:
            source_type = DEFAULT_SOURCE_TYPE
        for fd, vector in zip(fact_dicts, vectors):
            mem_id = str(uuid.uuid4())
            ids.append(mem_id)
            payload = {
                "user_id": user_id,
                "memory": fd["text"],
                "attributed_to": fd.get("attributed_to", "user"),
                "hash": fd.get("hash") or self._fact_hash(fd["text"]),
                "llm_extracted": llm_used,
                "deleted": False,
                "created_at": now,
                # Multi-agent permission groundwork - stamped on every write,
                # not read by any query today. See Tenant docstring.
                "owner_id": owner_id,
                "visibility_scope": visibility_scope,
                "source_type": source_type,
                "session_id": session_id,
            }
            if account_id:
                payload["account_id"] = account_id
            if agent_id:
                payload["agent_id"] = agent_id
            if run_id:
                payload["run_id"] = run_id
            if metadata:
                payload.update(metadata)
            points.append(PointStruct(id=mem_id, vector=vector, payload=payload))
        return ids, points

    @staticmethod
    def _scope_filter_clauses(tenant: Tenant) -> List[FieldCondition]:
        """Build the account/user/agent/run scope filter for tenant-bounded queries.

        account_id is the outermost account-level isolation; only
        added to the filter when the tenant carries one, so legacy
        single-tenant deployments whose payloads don't have account_id keep
        matching. agent_id / run_id are only enforced when the tenant
        supplies them, matching mem0's "scope down on demand" semantics.
        """
        clauses: List[FieldCondition] = []
        if tenant.account_id:
            clauses.append(FieldCondition(key="account_id",
                                          match=MatchValue(value=tenant.account_id)))
        clauses.append(FieldCondition(key="user_id",
                                      match=MatchValue(value=tenant.user_id)))
        if tenant.agent_id:
            clauses.append(FieldCondition(key="agent_id",
                                          match=MatchValue(value=tenant.agent_id)))
        if tenant.run_id:
            clauses.append(FieldCondition(key="run_id",
                                          match=MatchValue(value=tenant.run_id)))
        return clauses

    def _record_add_history(self, ids: List[str], points,
                            user_id: str,
                            agent_id: Optional[str],
                            run_id: Optional[str],
                            account_id: Optional[str] = None) -> None:
        """Append ADD events for a freshly upserted batch (no-op if disabled)."""
        if not self._history or not ids:
            return
        events = []
        for mid, pt in zip(ids, points):
            events.append({
                "memory_id": mid,
                "user_id": user_id,
                "account_id": account_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "event": "ADD",
                "new_memory": pt.payload.get("memory", ""),
                "timestamp": pt.payload.get("created_at"),
            })
        self._history.record_many(events)

    async def add(self, messages: List[Dict[str, str]], tenant: Tenant,
                  metadata: Optional[Dict[str, Any]] = None,
                  infer: bool = True,
                  llm_provider: Optional[Any] = None) -> List[str]:
        """Extract facts via LLM (or store raw), vectorize, and persist to Qdrant.

        When infer=False, stores raw message content directly without LLM extraction.
        When LLM is unreachable, falls back to raw storage automatically.
        llm_provider: optional adapter (BYOK / ModelRouter). If None, uses self.llm.
        """
        fact_dicts: List[Dict[str, str]] = []
        llm_used = False
        llm_failed = False
        adapter = llm_provider if llm_provider else self.llm
        extraction_messages = messages

        if infer:
            extraction_messages, should_skip, stripped_code = prepare_messages_for_fact_extraction(messages)
            if should_skip:
                logger.info(
                    "Skipping fact extraction for code-only input user=%s msg_count=%d",
                    tenant.user_id, len(messages),
                )
                return []
            if stripped_code:
                logger.info(
                    "Stripped large code blocks before fact extraction user=%s msg_count=%d",
                    tenant.user_id, len(messages),
                )

            try:
                logger.debug("LLM extracting facts for user=%s msg_count=%d",
                             tenant.user_id, len(extraction_messages))
                raw_facts = await adapter.extract_facts(extraction_messages)
                fact_dicts = [fd for fd in (self._normalize_fact(f) for f in raw_facts) if fd]
                llm_used = bool(fact_dicts)
                if not fact_dicts:
                    logger.info("LLM extracted 0 durable facts for user=%s", tenant.user_id)
                    return []
            except Exception as e:
                llm_failed = True
                logger.warning(
                    "LLM extraction failed for user=%s: %s. Falling back to sanitized raw storage.",
                    tenant.user_id, e,
                )

        if not fact_dicts:
            # infer=False stores raw by contract. infer=True only falls back on
            # LLM failures, and uses sanitized messages so large code blocks are
            # not persisted as raw memories.
            raw_source = extraction_messages if (infer and llm_failed) else messages
            fact_dicts = self._raw_facts(raw_source)
            if not fact_dicts:
                return []

        fact_dicts = self._dedup_facts(fact_dicts, tenant.user_id,
                                        account_id=tenant.account_id)
        if not fact_dicts:
            logger.info("All facts already present for user=%s — nothing to upsert",
                        tenant.user_id)
            return []

        vectors = await self.embedder.aembed_batch([fd["text"] for fd in fact_dicts])
        ids, points = self._build_points(
            fact_dicts, vectors,
            user_id=tenant.user_id, agent_id=tenant.agent_id,
            run_id=tenant.run_id, account_id=tenant.account_id,
            llm_used=llm_used, metadata=metadata,
            owner_id=tenant.owner_id,
            visibility_scope=tenant.visibility_scope,
            source_type=tenant.source_type,
            session_id=tenant.session_id,
        )
        self._client.upsert(collection_name=self.collection_name, points=points)
        self._record_add_history(ids, points, tenant.user_id,
                                 tenant.agent_id, tenant.run_id,
                                 account_id=tenant.account_id)
        logger.info(
            "Stored %d memories for user=%s agent=%s run=%s (%s)",
            len(fact_dicts), tenant.user_id, tenant.agent_id or "-",
            tenant.run_id or "-",
            "LLM extracted" if llm_used else "raw",
        )
        return ids

    async def process_batch(self, batch: List[Dict[str, str]],
                            user_id: str = None, metadata: Dict = None,
                            llm_config: Optional[Dict] = None,
                            agent_id: Optional[str] = None,
                            run_id: Optional[str] = None,
                            account_id: Optional[str] = None,
                            owner_id: Optional[str] = None,
                            visibility_scope: str = DEFAULT_VISIBILITY_SCOPE,
                            source_type: str = DEFAULT_SOURCE_TYPE,
                            session_id: Optional[str] = None) -> None:
        """Callback for AsyncBatchDistiller — process accumulated messages.

        Falls back to raw storage when LLM extraction fails or returns empty,
        so the user's messages are never silently dropped.

        llm_config (BYOK): {"api_key": "...", "base_url": "...", "model": "..."}.
        Never logged or persisted.
        """
        if not batch or not user_id:
            return

        # Pick adapter: BYOK > self.llm
        adapter = self.llm
        if llm_config:
            from deepmem.engine.model_router import adapter_from_byok_config
            adapter = adapter_from_byok_config(llm_config)

        extraction_batch, should_skip, stripped_code = prepare_messages_for_fact_extraction(batch)
        if should_skip:
            logger.info(
                "process_batch user=%s — skipping code-only input msg_count=%d",
                user_id, len(batch),
            )
            return
        if stripped_code:
            logger.info(
                "process_batch user=%s — stripped large code blocks before extraction",
                user_id,
            )

        logger.info(
            "process_batch user=%s msg_count=%d byok=%s — starting LLM extraction",
            user_id, len(extraction_batch), bool(llm_config),
        )

        fact_dicts: List[Dict[str, str]] = []
        llm_used = False
        llm_failed = False
        try:
            raw_facts = await adapter.extract_facts(extraction_batch)
            fact_dicts = [fd for fd in (self._normalize_fact(f) for f in raw_facts) if fd]
            llm_used = bool(fact_dicts)
            if not fact_dicts:
                logger.info("process_batch user=%s — LLM returned 0 durable facts", user_id)
                return
        except Exception as e:
            llm_failed = True
            logger.warning(
                "process_batch user=%s LLM extraction failed: %s — falling back to sanitized raw",
                user_id, e,
            )

        if not fact_dicts and llm_failed:
            fact_dicts = self._raw_facts(extraction_batch)
            if not fact_dicts:
                logger.info("process_batch user=%s — nothing to store after fallback", user_id)
                return

        fact_dicts = self._dedup_facts(fact_dicts, user_id, account_id=account_id)
        if not fact_dicts:
            logger.info(
                "process_batch user=%s — all %d fact(s) already stored, skipping upsert",
                user_id, 0,
            )
            return

        try:
            vectors = await self.embedder.aembed_batch([fd["text"] for fd in fact_dicts])
            ids, points = self._build_points(
                fact_dicts, vectors,
                user_id=user_id, agent_id=agent_id, run_id=run_id,
                account_id=account_id,
                llm_used=llm_used, metadata=metadata,
                owner_id=owner_id,
                visibility_scope=visibility_scope,
                source_type=source_type,
                session_id=session_id,
            )
            self._client.upsert(collection_name=self.collection_name, points=points)
            self._record_add_history(ids, points, user_id, agent_id, run_id,
                                     account_id=account_id)
            logger.info(
                "process_batch user=%s agent=%s run=%s — stored %d facts from %d messages (%s)",
                user_id, agent_id or "-", run_id or "-",
                len(fact_dicts), len(batch),
                "LLM extracted" if llm_used else "raw fallback",
            )
        except Exception as e:
            # Truly catastrophic — embedding or Qdrant down. We've already
            # exhausted the fallback. Surface loudly so an operator can act.
            logger.error(
                "process_batch user=%s upsert FAILED — %d messages may be lost: %s",
                user_id, len(batch), e,
            )

    async def search(self, query: str, tenant: Tenant,
                     top_k: int = 10, threshold: float = 0.3,
                     **kwargs) -> List[SearchResult]:
        """Semantic search with tenant isolation filter.

        Supports hybrid mode (hybrid=True): combines vector similarity,
        BM25 keyword match, entity boost, and time decay scoring.
        """
        import time as _time
        from deepmem.hybrid_search import (
            compute_time_decay, compute_bm25_scores,
            compute_entity_boost, fuse_scores,
        )

        use_hybrid = kwargs.get("hybrid", False)
        # memory_action="search" tells BGE-M3 to apply the query-side
        # retrieval instruction prefix (2.3). Passages stored at write time
        # are embedded without the prefix.
        try:
            query_vec = await self.embedder.aembed(query, memory_action="search")
        except TypeError:
            # Custom embedders (tests) without the memory_action kwarg.
            query_vec = await self.embedder.aembed(query)

        query_filter = Filter(
            must=self._scope_filter_clauses(tenant) + [
                FieldCondition(key="deleted", match=MatchValue(value=False)),
            ]
        )

        # Over-fetch more for hybrid since we re-rank
        fetch_limit = top_k * 3 if use_hybrid else top_k * 2
        hits = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=fetch_limit,
        )

        candidate_points = list(hits.points)

        # Bulk BM25 over candidate set — IDF needs a corpus, so we compute it
        # against the over-fetched candidates rather than per-doc.
        keyword_scores: List[float] = []
        if use_hybrid and candidate_points:
            candidate_texts = [p.payload.get("memory", "") for p in candidate_points]
            keyword_scores = compute_bm25_scores(query, candidate_texts)

        now = _time.time()
        results = []
        for idx, hit in enumerate(candidate_points):
            payload = hit.payload
            memory_text = payload.get("memory", "")
            vector_score = hit.score

            if use_hybrid:
                keyword_score = keyword_scores[idx] if idx < len(keyword_scores) else 0.0
                entity_boost = compute_entity_boost(query, memory_text)
                created_at = payload.get("created_at", 0)
                time_decay = compute_time_decay(created_at, now)
                final_score = fuse_scores(vector_score, keyword_score, entity_boost, time_decay)
            else:
                final_score = vector_score

            if final_score < threshold:
                continue

            results.append(SearchResult(
                id=hit.id,
                memory=memory_text,
                score=final_score,
                metadata={k: v for k, v in payload.items()
                         if k not in ("user_id", "memory", "agent_id", "deleted",
                                      "llm_extracted", "deleted_at", "created_at")},
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _payload_in_scope(payload: Dict[str, Any], tenant: Tenant) -> bool:
        """Check that a stored payload matches the tenant's account/user/agent/run scope.
        account_id only narrows scope when tenant has it set (legacy data without
        account_id still matches when caller is in legacy mode). agent_id / run_id
        only narrow scope when the tenant has them set."""
        if tenant.account_id and payload.get("account_id") != tenant.account_id:
            return False
        if payload.get("user_id") != tenant.user_id:
            return False
        if tenant.agent_id and payload.get("agent_id") != tenant.agent_id:
            return False
        if tenant.run_id and payload.get("run_id") != tenant.run_id:
            return False
        return True

    async def get(self, memory_id: str, tenant: Tenant) -> Optional[SearchResult]:
        """Retrieve a single memory by ID (verified against tenant)."""
        try:
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
            )
            if not points:
                return None
            point = points[0]
            payload = point.payload
            if not self._payload_in_scope(payload, tenant):
                return None
            if payload.get("deleted"):
                return None
            return SearchResult(
                id=point.id,
                memory=payload.get("memory", ""),
                score=1.0,
                metadata={k: v for k, v in payload.items()
                         if k not in ("user_id", "memory", "agent_id", "deleted",
                                      "llm_extracted", "deleted_at")},
            )
        except Exception as e:
            logger.error(f"Get failed for {memory_id}: {e}")
            return None

    async def update(self, memory_id: str, new_memory: str, tenant: Tenant) -> bool:
        """Update the text of a single memory (re-embeds the new text)."""
        try:
            # Verify ownership
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
            )
            if not points:
                return False
            point = points[0]
            if not self._payload_in_scope(point.payload, tenant):
                logger.warning(
                    f"Tenant user={tenant.user_id} agent={tenant.agent_id} "
                    f"run={tenant.run_id} attempted to update memory {memory_id} "
                    f"owned by user={point.payload.get('user_id')} "
                    f"agent={point.payload.get('agent_id')} "
                    f"run={point.payload.get('run_id')}"
                )
                return False

            prev_memory = point.payload.get("memory", "")

            # Re-embed and upsert. Refresh the hash so future dedup matches
            # the new text and we don't accidentally re-add the old fact.
            new_vec = await self.embedder.aembed(new_memory)
            payload = dict(point.payload)
            payload["memory"] = new_memory
            payload["hash"] = self._fact_hash(new_memory)

            self._client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=memory_id, vector=new_vec, payload=payload)],
            )
            if self._history:
                self._history.record(
                    memory_id=memory_id,
                    user_id=tenant.user_id,
                    account_id=tenant.account_id,
                    agent_id=payload.get("agent_id") or tenant.agent_id,
                    run_id=payload.get("run_id") or tenant.run_id,
                    event="UPDATE",
                    prev_memory=prev_memory,
                    new_memory=new_memory,
                )
            return True
        except Exception as e:
            logger.error(f"Update failed for {memory_id}: {e}")
            return False

    async def list(self, tenant: Tenant, limit: int = 100,
                   offset: int = 0) -> List[SearchResult]:
        """List all non-deleted memories for a tenant with pagination.

        Sorted by created_at descending (newest first) — matches user
        expectations and what every dashboard caller wants.
        """
        try:
            all_points, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=self._scope_filter_clauses(tenant) + [
                        FieldCondition(key="deleted",
                                      match=MatchValue(value=False)),
                    ]
                ),
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            # Sort newest-first by payload.created_at. Qdrant scroll
            # has no ORDER BY; client-side sort is fine for typical limits.
            all_points.sort(
                key=lambda p: p.payload.get("created_at", 0),
                reverse=True,
            )

            results = []
            for point in all_points:
                payload = point.payload
                results.append(SearchResult(
                    id=point.id,
                    memory=payload.get("memory", ""),
                    score=0.0,
                    metadata={k: v for k, v in payload.items()
                             if k not in ("user_id", "memory", "agent_id",
                                         "deleted", "llm_extracted", "deleted_at")},
                ))
            return results
        except Exception as e:
            logger.error(f"List failed for {tenant.user_id}: {e}")
            return []

    async def delete(self, memory_id: str, tenant: Tenant) -> bool:
        """Soft-delete a single memory by ID (verified against tenant).

        Uses set_payload so we never re-upsert a zero vector when
        point.vector isn't returned.
        """
        try:
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
            )
            if not points:
                return False
            point = points[0]
            if not self._payload_in_scope(point.payload, tenant):
                logger.warning(
                    f"Tenant user={tenant.user_id} agent={tenant.agent_id} "
                    f"run={tenant.run_id} attempted to delete memory {memory_id} "
                    f"owned by user={point.payload.get('user_id')} "
                    f"agent={point.payload.get('agent_id')} "
                    f"run={point.payload.get('run_id')}"
                )
                return False

            import time
            self._client.set_payload(
                collection_name=self.collection_name,
                payload={"deleted": True, "deleted_at": time.time()},
                points=[memory_id],
            )
            if self._history:
                self._history.record(
                    memory_id=memory_id,
                    user_id=tenant.user_id,
                    account_id=tenant.account_id,
                    agent_id=point.payload.get("agent_id") or tenant.agent_id,
                    run_id=point.payload.get("run_id") or tenant.run_id,
                    event="DELETE",
                    prev_memory=point.payload.get("memory", ""),
                )
            return True
        except Exception as e:
            logger.error(f"Soft-delete failed for {memory_id}: {e}")
            return False

    async def delete_all(self, tenant: Tenant) -> int:
        """Soft-delete all memories for a tenant (GDPR Right to Erasure).

        Iterates with pagination — Qdrant's scroll has a default page size,
        so a single call can miss records. Uses set_payload to avoid
        re-upserting vectors (cheap) and to avoid the zero-vector hazard.
        Honours tenant.agent_id / tenant.run_id when supplied for scoped
        erasure (e.g. clear just one workflow run).
        """
        import time
        all_ids: List[str] = []
        # Capture (id, payload) pairs so the history log records who got nuked
        # without having to re-fetch by id afterwards.
        history_rows: List[Dict[str, Any]] = []
        offset = None
        while True:
            page, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=self._scope_filter_clauses(tenant)),
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in page:
                all_ids.append(p.id)
                if self._history:
                    history_rows.append({
                        "memory_id": p.id,
                        "user_id": tenant.user_id,
                        "account_id": tenant.account_id,
                        "agent_id": p.payload.get("agent_id") or tenant.agent_id,
                        "run_id": p.payload.get("run_id") or tenant.run_id,
                        "event": "DELETE",
                        "prev_memory": p.payload.get("memory", ""),
                    })
            if not next_offset:
                break
            offset = next_offset
        total = len(all_ids)

        if total > 0:
            self._client.set_payload(
                collection_name=self.collection_name,
                payload={"deleted": True, "deleted_at": time.time()},
                points=all_ids,
            )
            if self._history and history_rows:
                self._history.record_many(history_rows)

        logger.info(
            "Soft-deleted %d memories for user=%s agent=%s run=%s",
            total, tenant.user_id, tenant.agent_id or "-", tenant.run_id or "-",
        )
        return total

    async def hard_delete_account(self, account_id: str) -> int:
        """Physically delete every memory belonging to an account.

        Used by the account-deletion endpoint — unlike delete_all (which
        soft-flags records for the soft-delete retention window), this
        actually removes the points from Qdrant so the data is gone the
        moment the user pulls the plug. account_id is the only scope; we
        don't know all of the account's user_ids, and don't need to.
        """
        if not account_id:
            return 0
        flt = Filter(must=[FieldCondition(key="account_id",
                                          match=MatchValue(value=account_id))])
        all_ids: List[str] = []
        offset = None
        while True:
            page, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=512,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for p in page:
                all_ids.append(p.id)
            if not next_offset:
                break
            offset = next_offset
        if all_ids:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=all_ids,
            )
            if self._history:
                self._history.delete_for_account(account_id)
        logger.info("Hard-deleted %d memories for account=%s", len(all_ids), account_id)
        return len(all_ids)

    # ── History (B) ────────────────────────────────────────────────────

    async def history(self, memory_id: str, tenant: Tenant,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Return the audit trail for a memory id, scoped to the tenant.

        Returns [] when history is disabled, when the memory doesn't exist,
        or when the tenant doesn't own it. The ownership check happens against
        the live record so a tenant can't read history for a memory they
        don't own even if they know the id.
        """
        if not self._history:
            return []
        try:
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
            )
        except Exception as e:
            logger.error("history retrieve failed for %s: %s", memory_id, e)
            return []
        if not points:
            # Memory may have been hard-deleted by reset(); fall back to a
            # user_id check on the history table itself.
            return [
                row for row in self._history.history_for_memory(
                    memory_id, tenant.user_id, limit,
                    account_id=tenant.account_id,
                )
            ]
        if not self._payload_in_scope(points[0].payload, tenant):
            return []
        return self._history.history_for_memory(
            memory_id, tenant.user_id, limit,
            account_id=tenant.account_id,
        )

    # ── Reset / Export / Import (C) ───────────────────────────────────

    async def reset(self, tenant: Tenant) -> Dict[str, int]:
        """Hard-delete every record (and history) for a user.

        Differs from delete_all in two ways:
        1. Physical delete from Qdrant, not soft delete.
        2. Wipes the history table for the user too.

        Always operates at user scope — agent_id / run_id on the tenant are
        ignored for reset; partial wipes go through delete_all.
        """
        all_ids: List[str] = []
        offset = None
        # Preserve account_id when scoping the wipe — agent_id / run_id are
        # explicitly dropped (reset is whole-user), but two accounts that
        # happen to share a user_id string must not erase each other.
        user_only = Tenant(user_id=tenant.user_id, account_id=tenant.account_id)
        while True:
            page, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=self._scope_filter_clauses(user_only)),
                limit=512,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            all_ids.extend(p.id for p in page)
            if not next_offset:
                break
            offset = next_offset
        deleted_vectors = len(all_ids)

        if all_ids:
            try:
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=all_ids,
                )
            except Exception as e:
                logger.error("reset hard-delete failed user=%s: %s", tenant.user_id, e)

        deleted_history = 0
        if self._history:
            deleted_history = self._history.reset_user(
                tenant.user_id, account_id=tenant.account_id,
            )

        logger.info(
            "Reset user=%s vectors=%d history=%d",
            tenant.user_id, deleted_vectors, deleted_history,
        )
        return {
            "deleted_vectors": deleted_vectors,
            "deleted_history_events": deleted_history,
        }

    async def export(self, tenant: Tenant) -> List[Dict[str, Any]]:
        """Dump every live (non-deleted) record for the tenant scope.

        Used by /v1/export for portability / migration. Returns plain dicts
        so the response body is JSON-friendly without needing pydantic on
        the engine layer. Vectors are NOT included — re-embedding on import
        is cheaper than carting around 1024-dim float arrays.
        """
        records: List[Dict[str, Any]] = []
        offset = None
        while True:
            page, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=self._scope_filter_clauses(tenant) + [
                        FieldCondition(key="deleted",
                                       match=MatchValue(value=False)),
                    ]
                ),
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in page:
                payload = p.payload or {}
                rec = {
                    "id": p.id,
                    "memory": payload.get("memory", ""),
                    "user_id": payload.get("user_id"),
                    "agent_id": payload.get("agent_id"),
                    "run_id": payload.get("run_id"),
                    "attributed_to": payload.get("attributed_to"),
                    "hash": payload.get("hash"),
                    "created_at": payload.get("created_at"),
                    "metadata": {
                        k: v for k, v in payload.items()
                        if k not in (
                            "user_id", "memory", "agent_id", "run_id",
                            "attributed_to", "hash", "deleted", "deleted_at",
                            "llm_extracted", "created_at",
                        )
                    },
                }
                records.append(rec)
            if not next_offset:
                break
            offset = next_offset
        logger.info("Exported %d records for user=%s agent=%s run=%s",
                    len(records), tenant.user_id,
                    tenant.agent_id or "-", tenant.run_id or "-")
        return records

    async def import_records(self, tenant: Tenant,
                             records: List[Dict[str, Any]],
                             skip_existing: bool = True) -> Dict[str, int]:
        """Re-ingest exported records under tenant.user_id.

        - Records always get re-embedded with the current model so the result
          is searchable even if the export came from a different embedder.
        - When skip_existing=True (default) we drop records whose content hash
          already exists for the user — re-importing the same dump is a no-op.
        - The original id from the export is NOT preserved; new uuids are
          assigned. Preserving ids would let an attacker overwrite arbitrary
          point ids in the collection.
        """
        if not records:
            return {"inserted": 0, "skipped": 0}

        fact_dicts: List[Dict[str, str]] = []
        per_record_meta: List[Dict[str, Any]] = []
        for r in records:
            text = (r.get("memory") or "").strip()
            if not text:
                continue
            fact_dicts.append({
                "text": text,
                "attributed_to": r.get("attributed_to") or "user",
                "hash": self._fact_hash(text),
            })
            per_record_meta.append({
                "agent_id": r.get("agent_id") or tenant.agent_id,
                "run_id": r.get("run_id") or tenant.run_id,
                "metadata": r.get("metadata") or {},
            })

        if not fact_dicts:
            return {"inserted": 0, "skipped": 0}

        skipped = 0
        if skip_existing:
            existing = self._existing_hashes(
                tenant.user_id, [fd["hash"] for fd in fact_dicts],
                account_id=tenant.account_id,
            )
            kept_facts: List[Dict[str, str]] = []
            kept_meta: List[Dict[str, Any]] = []
            for fd, m in zip(fact_dicts, per_record_meta):
                if fd["hash"] in existing:
                    skipped += 1
                    continue
                kept_facts.append(fd)
                kept_meta.append(m)
            fact_dicts = kept_facts
            per_record_meta = kept_meta

        if not fact_dicts:
            return {"inserted": 0, "skipped": skipped}

        vectors = await self.embedder.aembed_batch([fd["text"] for fd in fact_dicts])

        # Build points one-by-one so each carries its own agent_id/run_id from
        # the source export — we can't reuse _build_points's single-tenant API.
        import time as _time
        ids: List[str] = []
        points: List[PointStruct] = []
        now = _time.time()
        for fd, vec, m in zip(fact_dicts, vectors, per_record_meta):
            mem_id = str(uuid.uuid4())
            ids.append(mem_id)
            payload: Dict[str, Any] = {
                "user_id": tenant.user_id,
                "memory": fd["text"],
                "attributed_to": fd.get("attributed_to", "user"),
                "hash": fd["hash"],
                "llm_extracted": False,  # imported, not freshly extracted
                "deleted": False,
                "created_at": now,
                # Groundwork fields - defaults; if the source export carried
                # them (inside metadata), payload.update(m["metadata"]) below
                # overwrites these with the original values.
                "owner_id": None,
                "visibility_scope": DEFAULT_VISIBILITY_SCOPE,
                "source_type": DEFAULT_SOURCE_TYPE,
                "session_id": None,
            }
            if tenant.account_id:
                payload["account_id"] = tenant.account_id
            if m["agent_id"]:
                payload["agent_id"] = m["agent_id"]
            if m["run_id"]:
                payload["run_id"] = m["run_id"]
            if m["metadata"]:
                payload.update(m["metadata"])
            points.append(PointStruct(id=mem_id, vector=vec, payload=payload))

        self._client.upsert(collection_name=self.collection_name, points=points)
        self._record_add_history(ids, points, tenant.user_id,
                                 tenant.agent_id, tenant.run_id,
                                 account_id=tenant.account_id)
        logger.info(
            "Imported user=%s inserted=%d skipped=%d",
            tenant.user_id, len(points), skipped,
        )
        return {"inserted": len(points), "skipped": skipped}
