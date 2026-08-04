from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class AddMemoriesRequest(BaseModel):
    messages: List[Message]
    # user_id is the downstream end-user partition. Optional: when omitted, the
    # server defaults it to "default" (open mode has no API key / account).
    user_id: Optional[str] = Field(None, description="Optional end-user scope. Defaults to 'default' when missing.")
    agent_id: Optional[str] = None
    # Session / workflow scope. Mirrors mem0's run_id — when set, the write is
    # partitioned and reads can be filtered to this run only.
    run_id: Optional[str] = Field(None, description="Session/workflow scope")
    metadata: Optional[Dict[str, Any]] = None
    infer: bool = True
    # Multi-agent permission groundwork - stamped onto every stored memory.
    # Optional and unused by queries today; defaults applied server-side.
    owner_id: Optional[str] = Field(None, description="Agent instance that wrote this memory")
    visibility_scope: Optional[str] = Field(None, description="Who may read this memory (default: private)")
    source_type: Optional[str] = Field(None, description="'user' (stated) or 'agent' (inferred). Default: user")
    session_id: Optional[str] = Field(None, description="Conversation/session this memory belongs to")
    # BYOK: Bring Your Own Key — caller can override the LLM provider
    llm_api_key: Optional[str] = Field(None, description="Custom LLM API key (BYOK)")
    llm_base_url: Optional[str] = Field(None, description="Custom LLM base URL (BYOK)")


class MemoryResult(BaseModel):
    id: str
    memory: str
    event: str = "ADD"


class AddMemoriesResponse(BaseModel):
    results: List[MemoryResult]
    # When facts are extracted asynchronously by the batch distiller, the
    # request returns immediately with results=[] and pending=True so the
    # caller can tell "queued" apart from "no facts extracted".
    pending: bool = False
    message: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = Field(None, description="Optional. Defaults to 'default'.")
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class SearchResultItem(BaseModel):
    id: str
    memory: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


class UpdateMemoryRequest(BaseModel):
    memory: str = Field(..., min_length=1, description="New memory text")
    user_id: Optional[str] = Field(None, description="Optional. Defaults to 'default'.")
    agent_id: Optional[str] = None
    run_id: Optional[str] = None


class DeleteResponse(BaseModel):
    deleted_count: int
    user_id: str


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    detail: str


class MemoryHistoryEvent(BaseModel):
    id: int
    memory_id: str
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    event: str  # "ADD" | "UPDATE" | "DELETE"
    prev_memory: Optional[str] = None
    new_memory: Optional[str] = None
    timestamp: float


class MemoryHistoryResponse(BaseModel):
    memory_id: str
    events: List[MemoryHistoryEvent]


class ResetRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional. Defaults to 'default'.")
    # Confirmation guard — request is rejected unless this matches user_id.
    # Reset hard-deletes vectors and history for the user; no soft-delete
    # safety net, so we make the caller spell out the user twice.
    confirm_user_id: str = Field(..., min_length=1)


class ResetResponse(BaseModel):
    user_id: str
    deleted_vectors: int
    deleted_history_events: int


class ExportRecord(BaseModel):
    id: str
    memory: str
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    attributed_to: Optional[str] = None
    hash: Optional[str] = None
    created_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExportResponse(BaseModel):
    user_id: str
    count: int
    records: List[ExportRecord]


class ImportRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional. Defaults to 'default'.")
    records: List[ExportRecord]
    # When True, an existing record (matched by content hash) is left in place
    # and the imported one skipped. When False, the imported record is added
    # alongside (a new id is assigned).
    skip_existing: bool = True


class ImportResponse(BaseModel):
    user_id: str
    inserted: int
    skipped: int
