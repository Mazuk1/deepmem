"""DeepMemory MCP Server.

Exposes DeepMemory's write, search, and delete APIs as MCP tools so that
Claude Desktop, Cursor, and other MCP-compatible agents can use them.

Run directly (stdio transport, for MCP client config):

    python server/mcp_server.py

Or with SSE transport (for remote access):

    python server/mcp_server.py --transport sse --port 8001

Environment variables:
    DEEPMEMORY_BASE_URL  - DeepMemory server URL (default: http://localhost:8000)
    DEEPMEMORY_API_KEY   - optional bearer token (only if the target server
                           enforces auth; the default open-mode server needs none)

MCP client config example (claude_desktop_config.json):

    {
      "mcpServers": {
        "deepmem": {
          "command": "python",
          "args": ["server/mcp_server.py"],
          "env": {
            "DEEPMEMORY_BASE_URL": "http://localhost:8000"
          }
        }
      }
    }
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────

# Optional bearer token - only needed if the target server enforces API-key
# auth. The default open-mode DeepMemory server requires none.
API_KEY = os.environ.get("DEEPMEMORY_API_KEY", "").strip()
BASE_URL = os.environ.get("DEEPMEMORY_BASE_URL", "http://localhost:8000").rstrip("/")

mcp = FastMCP(
    "deepmem",
    instructions="DeepMemory - Mem0-compatible memory-as-a-service. "
    "Write conversation memories and search them semantically.",
)


# ── HTTP helpers ──────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


async def _post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{BASE_URL}{path}",
            json=body,
            headers=_headers(),
        )
        return r.json()


async def _delete(path: str, params: Optional[dict] = None) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.delete(
            f"{BASE_URL}{path}",
            params=params or {},
            headers=_headers(),
        )
        if r.status_code >= 400:
            return {"error": r.text, "status_code": r.status_code}
        return r.json()


# ── Tools ─────────────────────────────────────────────────────────────

@mcp.tool(
    name="deepmem_write",
    description=(
        "Write conversation messages to DeepMemory for fact extraction "
        "and persistent storage. Messages are processed by an LLM to "
        "extract structured memories, which are then embedded and stored "
        "in a vector database for later semantic search.\n\n"
        "Set infer=True to enable LLM fact extraction (produces richer "
        "memories but costs one LLM call). Set infer=False to store raw "
        "messages without extraction.\n\n"
        "Returns a list of memory IDs for successfully stored facts."
    ),
)
async def deepmem_write(
    messages: list[dict],
    user_id: str = "default",
    infer: bool = True,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    body: dict = {
        "messages": messages,
        "user_id": user_id,
        "infer": infer,
    }
    if agent_id:
        body["agent_id"] = agent_id
    if run_id:
        body["run_id"] = run_id
    return await _post("/v1/memories", body)


@mcp.tool(
    name="deepmem_search",
    description=(
        "Search memories stored in DeepMemory using semantic search. "
        "Returns the most relevant memories for the given query, ranked "
        "by hybrid scoring (vector similarity + BM25 keyword match + "
        "entity boost + time decay).\n\n"
        "Use this to retrieve context from past conversations before "
        "responding to the user. Memories are scoped to the user_id "
        "provided during write."
    ),
)
async def deepmem_search(
    query: str,
    user_id: str = "default",
    top_k: int = 10,
    threshold: float = 0.3,
) -> dict:
    body = {
        "query": query,
        "user_id": user_id,
        "top_k": top_k,
        "threshold": threshold,
    }
    return await _post("/v1/memories/search", body)


@mcp.tool(
    name="deepmem_delete",
    description=(
        "Delete a single memory from DeepMemory by its ID. Performs a "
        "soft-delete: the record is marked deleted and retained for the "
        "configured soft-delete retention window (audit/history still sees "
        "it). For a physical wipe of a whole user, use the HTTP /v1/reset "
        "endpoint instead.\n\n"
        "The memory must belong to the given user_id scope (and agent_id / "
        "run_id when supplied), otherwise the delete is rejected.\n\n"
        "Returns the deletion result with deleted_count."
    ),
)
async def deepmem_delete(
    memory_id: str,
    user_id: str = "default",
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    params = {"user_id": user_id}
    if agent_id:
        params["agent_id"] = agent_id
    if run_id:
        params["run_id"] = run_id
    return await _delete(f"/v1/memories/{memory_id}", params=params)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepMemory MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python server/mcp_server.py                           # stdio (for MCP clients)\n"
            "  python server/mcp_server.py --transport sse --port 8001  # HTTP SSE endpoint\n"
            "\n"
            "Environment:\n"
            "  DEEPMEMORY_BASE_URL  DeepMemory server URL (default http://localhost:8000)\n"
            "  DEEPMEMORY_API_KEY   optional bearer token (open-mode server needs none)\n"
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    args = parser.parse_args()

    mcp.run(transport=args.transport)
