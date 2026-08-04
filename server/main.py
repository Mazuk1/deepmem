import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Optional

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deepmem.config import config
from deepmem.engine.model_router import adapter_from_byok_config, build_byok_config
from deepmem.interface import Tenant
from deepmem.middleware import GDPRLogMasker
from server.dependencies import get_services
from server.models import (
    AddMemoriesRequest,
    AddMemoriesResponse,
    DeleteResponse,
    ErrorResponse,
    ExportRecord,
    ExportResponse,
    HealthResponse,
    ImportRequest,
    ImportResponse,
    MemoryHistoryEvent,
    MemoryHistoryResponse,
    MemoryResult,
    ResetRequest,
    ResetResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    UpdateMemoryRequest,
)

# ── Logging: stdout + rotating file ────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, "deepmem.log")

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_root = logging.getLogger()
if not any(isinstance(h, RotatingFileHandler) for h in _root.handlers):
    _root.setLevel(logging.INFO)
    # Stdout
    _stdout = logging.StreamHandler()
    _stdout.setFormatter(_fmt)
    _root.addHandler(_stdout)
    # Rotating file: 10 MB × 5 files
    _file = RotatingFileHandler(_LOG_PATH, maxBytes=10 * 1024 * 1024,
                                backupCount=5, encoding="utf-8")
    _file.setFormatter(_fmt)
    _root.addHandler(_file)

logger = logging.getLogger("deepmem.server")
logger.info("File logging enabled: %s", _LOG_PATH)

# Globally shared instances
_masker = GDPRLogMasker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MCP_AVAILABLE, _MCP_PORT

    logger.info("=== DeepMemory server starting ===")

    logger.info("Initializing service singletons (this triggers BGE-M3 import)…")
    t_svc = time.monotonic()
    svc = get_services()
    logger.info(
        "Services initialized in %.1fs: embedder=%s cache=%s distiller=%s router=%s store=%s",
        time.monotonic() - t_svc,
        type(svc["embedder"]).__name__,
        type(svc["cache"]).__name__,
        type(svc["distiller"]).__name__,
        type(svc["router"]).__name__,
        type(svc["store"]).__name__,
    )

    # Embedder warmup — BGE-M3 lazy-loads on first use (10–20s on CPU). Pay
    # that cost during boot, not on the first user request, so /ready can
    # advertise true readiness and the first POST doesn't time out.
    try:
        logger.info("Warming up BGE-M3 embedder (first call loads weights)…")
        t = time.monotonic()
        svc["embedder"].embed("warmup")
        logger.info("Embedder warmup complete in %.1fs", time.monotonic() - t)
    except Exception as e:
        logger.error("Embedder warmup failed: %s", e, exc_info=True)

    # ── MCP Server ────────────────────────────────────────────────────
    # Deferred import: load mcp AFTER BGE-M3 is warm so the mcp package's
    # C-extensions (zstandard, rpds, pydantic-core) don't interfere with
    # sklearn/pyarrow DLL loading on Windows.
    # Run on a separate port (default 8001) to avoid path-mount issues
    # with streamable-http session redirects under FastAPI.
    _mcp_task: Optional[asyncio.Task] = None

    if os.environ.get("DEEPMEMORY_NO_MCP", "").strip() != "1":
        try:
            from server.mcp_server import mcp as _mcp
            import uvicorn

            _mcp_port = int(os.environ.get("MCP_PORT", "8001"))
            _mcp.settings.host = "0.0.0.0"
            _mcp.settings.port = _mcp_port
            _mcp_cfg = uvicorn.Config(
                _mcp.streamable_http_app(),
                host="0.0.0.0",
                port=_mcp_port,
                log_level="info",
                proxy_headers=True,
                forwarded_allow_ips="*",
            )
            _mcp_server = uvicorn.Server(_mcp_cfg)

            async def _run_mcp():
                logger.info("MCP server starting on port %d", _mcp_port)
                await _mcp_server.serve()

            _mcp_task = asyncio.create_task(_run_mcp())
            _MCP_AVAILABLE = True
            _MCP_PORT = _mcp_port
        except Exception:
            _MCP_AVAILABLE = False
            _MCP_PORT = 0

    # Report which services are starting
    logger.info("Services:")
    logger.info("  HTTP API         http://0.0.0.0:8000")
    if _MCP_AVAILABLE:
        logger.info("  MCP (streamable) http://0.0.0.0:%d/mcp", _MCP_PORT)
        logger.info("    Tools: deepmem_write, deepmem_search, deepmem_delete")

    logger.info(
        "=== DeepMemory server ready (MCP=%s) ===",
        f"on :{_MCP_PORT}" if _MCP_AVAILABLE else "off",
    )

    yield
    # ── Graceful shutdown ────────────────────────────────────────────
    logger.info("=== DeepMemory server shutting down ===")

    # Stop MCP server first so no new connections arrive
    if _mcp_task is not None:
        _mcp_server.should_exit = True
        _mcp_task.cancel()
        try:
            await asyncio.wait_for(_mcp_task, timeout=5)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logger.info("MCP server stopped")

    # Persist cache to disk before shutting down
    try:
        svc["cache"].flush()
        logger.info("Cache persisted to disk")
    except Exception as e:
        logger.warning("Cache persist failed: %s", e)

    try:
        await asyncio.wait_for(svc["distiller"].flush(), timeout=30)
        logger.info("Distiller flushed")
    except asyncio.TimeoutError:
        logger.error("Distiller flush timed out after 30s - queued messages may be lost")
    except Exception as e:
        logger.error("Distiller flush error: %s", e)


app = FastAPI(
    title="DeepMemory",
    version="0.1.0",
    lifespan=lifespan,
)

# MCP is started during lifespan - import deferred to after BGE-M3 warmup.
_MCP_AVAILABLE = False
_MCP_PORT = 0


def _account_id(http: Request) -> Optional[str]:
    """Open mode: no account/registration. Always None (legacy single-tenant)."""
    return None


def _resolve_user_id(http: Request, request_user_id: Optional[str], validator) -> str:
    """Resolve effective user_id.

    Open mode (no API key, no registration): callers MAY pass an explicit
    user_id to scope writes/reads to a downstream end-user partition. When
    omitted (None) a shared "default" partition is used so the API works out
    of the box. An explicitly-empty or invalid user_id is rejected with 400.
    """
    if request_user_id is None:
        return "default"
    try:
        return validator.validate_user_id(request_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── CORS ───────────────────────────────────────────────────────────────
# Set CORS_ORIGINS to a comma-separated list in prod; default "*" is
# fine for dev + curl smoke tests but should NOT ship to a public host.
# Browsers also forbid the (allow_origins="*", allow_credentials=True)
# combination outright — they drop the Access-Control-Allow-Origin header
# and the request silently fails. Detect that combo at boot and downgrade
# to allow_credentials=False so the dev experience matches the prod one.
_CORS_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
] or ["*"]
_CORS_ALLOW_CREDENTIALS = True
if "*" in _CORS_ORIGINS:
    if os.environ.get("DEEPMEMORY_DEBUG") != "1":
        raise RuntimeError(
            "CORS_ORIGINS='*' is not allowed in production — set "
            "CORS_ORIGINS to a comma-separated list of trusted origins "
            "(e.g. CORS_ORIGINS=https://app.example.com,https://example.com). "
            "Set DEEPMEMORY_DEBUG=1 only for local development."
        )
    logger.warning(
        "CORS_ORIGINS=* with credentials is forbidden by browsers — "
        "running with allow_credentials=False; dev mode only.",
    )
    _CORS_ALLOW_CREDENTIALS = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate limiting ──────────────────────────────────────────────────────
# Per-IP token bucket implemented as middleware so we don't have to rename
# every endpoint's Pydantic body parameter (slowapi's decorator demands a
# parameter literally called `request: Request`, which clashes with
# `request: AddMemoriesRequest`). Disabled when RATE_LIMIT_DISABLED=1.
_RATE_LIMIT_DISABLED = os.environ.get("RATE_LIMIT_DISABLED") == "1"


def _parse_rate(spec: str, default_per_min: int) -> int:
    """Parse '30/minute' / '60/minute' style rate spec → per-minute integer.

    Anything we don't recognize falls back to default_per_min so a typo in
    the env var doesn't bring the service down.
    """
    try:
        n, _ = spec.split("/", 1)
        return max(1, int(n.strip()))
    except Exception:
        return default_per_min


_RATE_LIMIT_ADD_PM = _parse_rate(os.environ.get("RATE_LIMIT_ADD", "30/minute"), 30)
_RATE_LIMIT_SEARCH_PM = _parse_rate(os.environ.get("RATE_LIMIT_SEARCH", "60/minute"), 60)

# In-process counters: {(ip, route_key, minute_bucket): count}.
# Cleared lazily — when a new minute_bucket is seen we delete all entries
# from older buckets so memory stays bounded under sustained load.
_rl_counters: dict = {}
_rl_lock = asyncio.Lock()


async def _rate_limit_check(ip: str, route_key: str, limit_pm: int) -> bool:
    """Returns True if the request is allowed, False if rate-limited."""
    bucket = int(time.time() // 60)
    async with _rl_lock:
        # Drop counters older than the current minute
        if _rl_counters and any(b != bucket for (_, _, b) in _rl_counters.keys()):
            for k in [k for k in _rl_counters if k[2] != bucket]:
                _rl_counters.pop(k, None)
        key = (ip, route_key, bucket)
        cur = _rl_counters.get(key, 0)
        if cur >= limit_pm:
            return False
        _rl_counters[key] = cur + 1
        return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _RATE_LIMIT_DISABLED:
        return await call_next(request)
    path = request.url.path
    method = request.method
    route_key, limit = None, None
    if path == "/v1/memories" and method in ("POST", "PUT"):
        route_key, limit = "memories_add", _RATE_LIMIT_ADD_PM
    elif path == "/v1/memories/search":
        route_key, limit = "memories_search", _RATE_LIMIT_SEARCH_PM
    if route_key is None:
        return await call_next(request)
    ip = request.client.host if request.client else "unknown"
    if not await _rate_limit_check(ip, route_key, limit):
        return JSONResponse(
            status_code=429,
            content={"detail": f"rate limit exceeded: {limit}/minute on {route_key}"},
        )
    return await call_next(request)


logger.info(
    "Rate limiting: %s (add=%d/min, search=%d/min)",
    "disabled" if _RATE_LIMIT_DISABLED else "enabled",
    _RATE_LIMIT_ADD_PM, _RATE_LIMIT_SEARCH_PM,
)


# ── Auth (open mode) ───────────────────────────────────────────────────
# No API key / no registration. All endpoints are public. Multi-tenant
# isolation is still enforced via user_id in the request body (defaults to
# "default" when omitted). No auth middleware is registered.


# ── Health & Readiness ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe — always 200 if the process is alive.

    Use /ready when the load balancer needs to know dependencies are wired.
    """
    return HealthResponse(status="ok")


@app.get("/ready")
async def ready_check():
    """Readiness probe — verifies critical dependencies are reachable.

    Returns 503 if any dependency is down, with per-component status so
    operators can see which one regressed. The embedder and LLM are NOT
    probed here: the embedder is verified at startup (model load) and the
    LLM is allowed to be degraded — process_batch falls back to raw storage.
    """
    svc = get_services()
    components = {}
    healthy = True

    # Qdrant — the only dependency on the request hot path.
    try:
        svc["store"]._client.get_collections()
        components["qdrant"] = "ok"
    except Exception as e:
        components["qdrant"] = f"error: {type(e).__name__}: {e}"
        healthy = False

    # Embedder — already loaded at startup; just confirm the singleton exists.
    components["embedder"] = "ok" if svc["embedder"] is not None else "not_initialized"
    if components["embedder"] != "ok":
        healthy = False

    if not healthy:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "components": components},
        )
    return {"status": "ready", "components": components}


# ── Debug: force-flush pending batches ────────────────────────────────
# Enabled in dev (DEEPMEMORY_DEBUG=1 or batch_silence_window short).
# Lets test_pipeline.py write → flush → search round-trip without
# waiting the 180s silence window.

@app.post("/v1/_debug/flush")
async def debug_flush(user_id: str = Query(None)):
    if not (os.environ.get("DEEPMEMORY_DEBUG") == "1"
            or config.batch_silence_window_seconds < 30):
        raise HTTPException(status_code=404, detail="Not found")
    svc = get_services()
    await svc["distiller"].flush(user_id)
    logger.info("DEBUG flush user_id=%s", user_id or "<all>")
    return {"flushed": True, "user_id": user_id}


# ── Add Memories ──────────────────────────────────────────────────────

def _byok_config(request: AddMemoriesRequest) -> "dict | None":
    """Extract BYOK config from a request, or None if no key supplied."""
    return build_byok_config(request.llm_api_key, request.llm_base_url)


@app.post("/v1/memories", response_model=AddMemoriesResponse,
          responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def add_memories(request: AddMemoriesRequest, http: Request):
    """Write and extract memories from conversation messages."""
    t_start = time.monotonic()
    svc = get_services()

    user_id = _resolve_user_id(http, request.user_id, svc["validator"])

    user_hash = _masker._hash_str(user_id)
    account_id = _account_id(http)
    logger.info(
        "add_memories user_hash=%s account_hash=%s agent=%s run=%s msg_count=%d infer=%s byok=%s",
        user_hash, _masker.mask_id(account_id),
        request.agent_id, request.run_id,
        len(request.messages), request.infer,
        bool(request.llm_api_key),
    )

    tenant = Tenant(user_id=user_id, agent_id=request.agent_id,
                    run_id=request.run_id, account_id=account_id,
                    owner_id=request.owner_id,
                    visibility_scope=request.visibility_scope,
                    source_type=request.source_type,
                    session_id=request.session_id)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Check semantic cache (add flow) — only when LLM extraction is on. With
    # infer=False the caller is persisting raw messages, so semantic dedupe
    # over the prompt has no meaning and would short-circuit a real write.
    query_text = " ".join(m["content"] for m in messages)
    query_vec = await svc["embedder"].aembed(query_text)

    if config.cache_enabled and request.infer:
        cached = await svc["cache"].check(messages, query_vec, user_id,
                                          account_id=account_id)
        if cached:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "add_memories CACHE HIT user_hash=%s facts=%d elapsed=%.1fms",
                user_hash, len(cached), elapsed,
            )
            # cached is List[{"id": ..., "memory": ...}] — real Qdrant ids
            results = [MemoryResult(id=item["id"], memory=item["memory"], event="ADD")
                       for item in cached]
            return AddMemoriesResponse(results=results)

    # Route through BatchDistiller for async batching (or direct if batch disabled)
    if config.batch_enabled and request.infer:
        # BYOK config rides along with the queued batch so the
        # distiller's worker uses the user's key, not ours.
        await svc["distiller"].enqueue(
            user_id, messages,
            metadata=request.metadata,
            llm_config=_byok_config(request),
            agent_id=request.agent_id,
            run_id=request.run_id,
            account_id=account_id,
            owner_id=request.owner_id,
            visibility_scope=request.visibility_scope,
            source_type=request.source_type,
            session_id=request.session_id,
        )
        logger.debug(
            "add_memories enqueued user_hash=%s queue_size=%d",
            user_hash,
            svc["distiller"].get_queue_size(
                user_id, agent_id=request.agent_id, run_id=request.run_id,
            ),
        )
        elapsed = (time.monotonic() - t_start) * 1000
        logger.info("add_memories BATCHED user_hash=%s elapsed=%.1fms", user_hash, elapsed)
        # tell the caller this was queued, not 0-extracted.
        return AddMemoriesResponse(
            results=[],
            pending=True,
            message=(
                f"Queued for batched extraction; facts will be available after the "
                f"{config.batch_silence_window_seconds}s silence window or when the "
                f"batch fills (max {config.batch_max_size} messages)."
            ),
        )

    # Direct processing (infer=False or batch disabled)
    # Select LLM provider: BYOK > ModelRouter
    byok_cfg = _byok_config(request)
    if byok_cfg:
        llm_provider = adapter_from_byok_config(byok_cfg)
        logger.info("add_memories using BYOK provider for user_hash=%s", user_hash)
    else:
        llm_provider = svc["router"].route(tenant)

    mem_ids = await svc["store"].add(
        messages, tenant, metadata=request.metadata,
        infer=request.infer, llm_provider=llm_provider,
    )

    # Fetch by id rather than via list(): list() truncates *before* the
    # client-side created_at sort, so when the user already has more than
    # `limit` memories it can return an arbitrary record instead of the
    # one we just stored.
    facts = []
    if mem_ids:
        for mid in mem_ids:
            rec = await svc["store"].get(mid, tenant)
            if rec is not None:
                facts.append(rec.memory)
        if not facts and request.infer:
            search_results = await svc["store"].search(query_text, tenant, top_k=len(mem_ids) + 5)
            facts = [r.memory for r in search_results[:len(mem_ids)]]

    if config.cache_enabled and request.infer and mem_ids and facts:
        cache_items = [{"id": mid, "memory": fact}
                       for mid, fact in zip(mem_ids, facts)]
        await svc["cache"].store(messages, query_vec, user_id, cache_items,
                                 account_id=account_id)

    results = [MemoryResult(id=mid, memory=fact, event="ADD")
               for mid, fact in zip(mem_ids, facts)] if mem_ids and facts else []

    elapsed = (time.monotonic() - t_start) * 1000
    logger.info(
        "add_memories DONE user_hash=%s facts=%d elapsed=%.1fms",
        user_hash, len(facts), elapsed,
    )
    return AddMemoriesResponse(results=results)


# ── Search Memories ───────────────────────────────────────────────────

@app.post("/v1/memories/search", response_model=SearchResponse,
          responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def search_memories(request: SearchRequest, http: Request):
    """Semantic search for memories in Qdrant."""
    t_start = time.monotonic()
    svc = get_services()

    user_id = _resolve_user_id(http, request.user_id, svc["validator"])

    user_hash = _masker._hash_str(user_id)
    account_id = _account_id(http)
    logger.info(
        "search_memories user_hash=%s account_hash=%s agent=%s run=%s query='%s' top_k=%d threshold=%.2f",
        user_hash, _masker.mask_id(account_id),
        request.agent_id, request.run_id,
        request.query[:80], request.top_k, request.threshold,
    )

    tenant = Tenant(user_id=user_id, agent_id=request.agent_id,
                    run_id=request.run_id, account_id=account_id)

    # Check semantic cache for search queries.
    # cache stores the actual search items (id+memory+score), so on
    # hit we return real Qdrant ids and real scores rather than fabricating
    # uuid4()s and a 0.99 placeholder.
    query_vec = await svc["embedder"].aembed(request.query, memory_action="search")
    cache_key = [{"role": "user", "content": request.query}]

    if config.cache_enabled:
        cached = await svc["cache"].check(cache_key, query_vec, user_id,
                                          account_id=account_id)
        if cached:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "search_memories CACHE HIT user_hash=%s results=%d elapsed=%.1fms",
                user_hash, len(cached), elapsed,
            )
            items = [SearchResultItem(**item) for item in cached]
            return SearchResponse(results=items[:request.top_k])

    # Actual Qdrant search with hybrid scoring
    results = await svc["store"].search(
        request.query, tenant,
        top_k=request.top_k,
        threshold=request.threshold,
        hybrid=True,
    )

    items = [SearchResultItem(id=r.id, memory=r.memory, score=r.score) for r in results]

    if config.cache_enabled and items:
        # Store as list of plain dicts so the cache stays JSON-serializable
        # if a future Redis backend wants to persist it.
        cached_value = [
            {"id": it.id, "memory": it.memory, "score": it.score} for it in items
        ]
        await svc["cache"].store(cache_key, query_vec, user_id, cached_value,
                                 account_id=account_id)

    elapsed = (time.monotonic() - t_start) * 1000
    logger.info(
        "search_memories DONE user_hash=%s results=%d elapsed=%.1fms",
        user_hash, len(items), elapsed,
    )
    return SearchResponse(results=items)


# ── Get Single Memory ─────────────────────────────────────────────────

@app.get("/v1/memories/{memory_id}", response_model=SearchResultItem,
         responses={404: {"model": ErrorResponse}})
async def get_memory(
    http: Request,
    memory_id: str = Path(..., description="UUID of the memory"),
    user_id: Optional[str] = Query(None, description="Optional. Defaults to account_id."),
    agent_id: Optional[str] = Query(None, description="Optional agent scope"),
    run_id: Optional[str] = Query(None, description="Optional session/workflow scope"),
):
    """Retrieve a single memory by ID."""
    svc = get_services()

    user_id = _resolve_user_id(http, user_id, svc["validator"])

    logger.info("get_memory id=%s user_hash=%s agent=%s run=%s",
                memory_id, _masker._hash_str(user_id), agent_id, run_id)

    tenant = Tenant(user_id=user_id, agent_id=agent_id, run_id=run_id,
                    account_id=_account_id(http))
    result = await svc["store"].get(memory_id, tenant)
    if result is None:
        logger.info("get_memory NOT_FOUND id=%s", memory_id)
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    return SearchResultItem(id=result.id, memory=result.memory, score=result.score)


# ── Update Memory ─────────────────────────────────────────────────────

@app.put("/v1/memories/{memory_id}", response_model=SearchResultItem,
         responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def update_memory(
    http: Request,
    memory_id: str = Path(..., description="UUID of the memory to update"),
    request: UpdateMemoryRequest = ...,
):
    """Update the text of a single memory."""
    svc = get_services()

    user_id = _resolve_user_id(http, request.user_id, svc["validator"])

    logger.info(
        "update_memory id=%s user_hash=%s agent=%s run=%s new_len=%d",
        memory_id, _masker._hash_str(user_id),
        request.agent_id, request.run_id, len(request.memory),
    )

    tenant = Tenant(user_id=user_id, agent_id=request.agent_id,
                    run_id=request.run_id, account_id=_account_id(http))
    success = await svc["store"].update(memory_id, request.memory, tenant)
    if not success:
        logger.info("update_memory NOT_FOUND id=%s", memory_id)
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    svc["cache"].clear(user_id, account_id=tenant.account_id)
    logger.info("update_memory OK id=%s cache_cleared=true", memory_id)
    return SearchResultItem(id=memory_id, memory=request.memory, score=1.0)


# ── List Memories ─────────────────────────────────────────────────────

@app.get("/v1/memories", response_model=SearchResponse)
async def list_memories(
    http: Request,
    user_id: Optional[str] = Query(None, description="Optional. Defaults to account_id."),
    agent_id: Optional[str] = Query(None, description="Optional agent scope"),
    run_id: Optional[str] = Query(None, description="Optional session/workflow scope"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List all non-deleted memories for a tenant."""
    svc = get_services()

    user_id = _resolve_user_id(http, user_id, svc["validator"])

    logger.info(
        "list_memories user_hash=%s agent=%s run=%s limit=%d offset=%d",
        _masker._hash_str(user_id), agent_id, run_id, limit, offset,
    )

    tenant = Tenant(user_id=user_id, agent_id=agent_id, run_id=run_id,
                    account_id=_account_id(http))
    results = await svc["store"].list(tenant, limit=limit, offset=offset)
    items = [SearchResultItem(id=r.id, memory=r.memory, score=r.score) for r in results]
    logger.info("list_memories DONE user_hash=%s count=%d", _masker._hash_str(user_id), len(items))
    return SearchResponse(results=items)


# ── Delete Single Memory ──────────────────────────────────────────────

@app.delete("/v1/memories/{memory_id}", response_model=DeleteResponse,
            responses={404: {"model": ErrorResponse}})
async def delete_memory(
    http: Request,
    memory_id: str = Path(..., description="UUID of the memory to delete"),
    user_id: Optional[str] = Query(None, description="Optional. Defaults to account_id."),
    agent_id: Optional[str] = Query(None, description="Optional agent scope"),
    run_id: Optional[str] = Query(None, description="Optional session/workflow scope"),
):
    """Soft-delete a single memory by ID."""
    svc = get_services()

    user_id = _resolve_user_id(http, user_id, svc["validator"])

    logger.info("delete_memory id=%s user_hash=%s agent=%s run=%s",
                memory_id, _masker._hash_str(user_id), agent_id, run_id)

    tenant = Tenant(user_id=user_id, agent_id=agent_id, run_id=run_id,
                    account_id=_account_id(http))
    success = await svc["store"].delete(memory_id, tenant)
    if not success:
        logger.info("delete_memory NOT_FOUND id=%s", memory_id)
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    svc["cache"].clear(user_id, account_id=tenant.account_id)
    logger.info("delete_memory OK id=%s cache_cleared=true", memory_id)
    return DeleteResponse(deleted_count=1, user_id=user_id)


# ── Delete All Memories (GDPR) ────────────────────────────────────────

@app.delete("/v1/memories", response_model=DeleteResponse)
async def delete_all_memories(
    http: Request,
    user_id: Optional[str] = Query(None, description="Optional. Defaults to account_id."),
    agent_id: Optional[str] = Query(None, description="Optional agent scope"),
    run_id: Optional[str] = Query(None, description="Optional session/workflow scope"),
):
    """Soft-delete all memories for a user (GDPR Right to Erasure).

    With agent_id / run_id supplied the erasure is scoped — useful for
    "forget this session" without nuking the whole user.
    """
    svc = get_services()

    user_id = _resolve_user_id(http, user_id, svc["validator"])

    logger.info("delete_all_memories user_hash=%s agent=%s run=%s",
                _masker._hash_str(user_id), agent_id, run_id)

    tenant = Tenant(user_id=user_id, agent_id=agent_id, run_id=run_id,
                    account_id=_account_id(http))
    deleted = await svc["store"].delete_all(tenant)
    svc["cache"].clear(user_id, account_id=tenant.account_id)
    # aclear awaits timer cancellation so we don't race with a half-fired
    # batch from the very user we're trying to erase.
    await svc["distiller"].aclear(user_id)

    logger.info("delete_all_memories DONE user_hash=%s deleted=%d",
                _masker._hash_str(user_id), deleted)
    return DeleteResponse(deleted_count=deleted, user_id=user_id)


# ── Memory History (B) ────────────────────────────────────────────────

@app.get("/v1/memories/{memory_id}/history", response_model=MemoryHistoryResponse,
        responses={404: {"model": ErrorResponse}})
async def get_memory_history(
    http: Request,
    memory_id: str = Path(..., description="UUID of the memory"),
    user_id: Optional[str] = Query(None, description="Optional. Defaults to account_id."),
    agent_id: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Return the ADD/UPDATE/DELETE event log for a single memory.

    Useful for audit / time-travel — the live `memory` field only shows the
    current state; this shows what it used to be and when it changed.
    """
    svc = get_services()

    user_id = _resolve_user_id(http, user_id, svc["validator"])

    tenant = Tenant(user_id=user_id, agent_id=agent_id, run_id=run_id,
                    account_id=_account_id(http))
    events = await svc["store"].history(memory_id, tenant, limit=limit)
    if not events:
        # No events AND no live record -> tell the caller the id doesn't exist
        # (or isn't theirs). An empty history for an existing memory shouldn't
        # be possible because ADD events are recorded on every write.
        live = await svc["store"].get(memory_id, tenant)
        if live is None:
            raise HTTPException(
                status_code=404,
                detail=f"Memory {memory_id} not found",
            )

    logger.info(
        "get_memory_history id=%s user_hash=%s events=%d",
        memory_id, _masker._hash_str(user_id), len(events),
    )
    return MemoryHistoryResponse(
        memory_id=memory_id,
        events=[MemoryHistoryEvent(**ev) for ev in events],
    )


# ── Reset / Export / Import (C) ───────────────────────────────────────

@app.post("/v1/reset", response_model=ResetResponse,
          responses={400: {"model": ErrorResponse}})
async def reset_user(request: ResetRequest, http: Request):
    """Hard-delete every record (and history) for a user.

    Distinct from DELETE /v1/memories which soft-deletes. Reset is the
    "factory reset" path — vectors are physically removed, history wiped.
    Requires confirm_user_id == user_id to guard against accidental nukes.
    """
    svc = get_services()

    user_id = _resolve_user_id(http, request.user_id, svc["validator"])
    try:
        confirm_id = svc["validator"].validate_user_id(request.confirm_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if user_id != confirm_id:
        raise HTTPException(
            status_code=400,
            detail="confirm_user_id must equal user_id",
        )

    logger.warning("reset_user user_hash=%s — HARD DELETE", _masker._hash_str(user_id))

    tenant = Tenant(user_id=user_id, account_id=_account_id(http))
    result = await svc["store"].reset(tenant)
    svc["cache"].clear(user_id, account_id=tenant.account_id)
    await svc["distiller"].aclear(user_id)

    return ResetResponse(
        user_id=user_id,
        deleted_vectors=result["deleted_vectors"],
        deleted_history_events=result["deleted_history_events"],
    )


@app.get("/v1/export", response_model=ExportResponse)
async def export_memories(
    http: Request,
    user_id: Optional[str] = Query(None, description="Optional. Defaults to account_id."),
    agent_id: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
):
    """Export all live memories for a user (or scoped sub-set) as portable JSON."""
    svc = get_services()

    user_id = _resolve_user_id(http, user_id, svc["validator"])

    tenant = Tenant(user_id=user_id, agent_id=agent_id, run_id=run_id,
                    account_id=_account_id(http))
    records = await svc["store"].export(tenant)

    logger.info(
        "export user_hash=%s agent=%s run=%s count=%d",
        _masker._hash_str(user_id), agent_id, run_id, len(records),
    )
    return ExportResponse(
        user_id=user_id,
        count=len(records),
        records=[ExportRecord(**r) for r in records],
    )


@app.post("/v1/import", response_model=ImportResponse)
async def import_memories(request: ImportRequest, http: Request):
    """Import previously exported records into a user's memory store."""
    svc = get_services()

    user_id = _resolve_user_id(http, request.user_id, svc["validator"])

    tenant = Tenant(user_id=user_id, account_id=_account_id(http))
    result = await svc["store"].import_records(
        tenant,
        [r.model_dump() for r in request.records],
        skip_existing=request.skip_existing,
    )

    logger.info(
        "import user_hash=%s inserted=%d skipped=%d",
        _masker._hash_str(user_id), result["inserted"], result["skipped"],
    )
    return ImportResponse(
        user_id=user_id,
        inserted=result["inserted"],
        skipped=result["skipped"],
    )
