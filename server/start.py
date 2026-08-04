"""DeepMemory unified startup entrypoint.

Starts both the HTTP API server and the MCP server in a single process.

Usage:
    python server/start.py                       # HTTP :8000 + MCP :8001
    python server/start.py --port 9000           # Custom HTTP port
    python server/start.py --mcp-port 9001       # Custom MCP port
    python server/start.py --no-mcp              # HTTP only, skip MCP

The MCP server runs on its own port (default 8001) independently of the
FastAPI app, started as a background task in the lifespan. This avoids
path-mount issues with streamable-http session redirects under FastAPI.
"""

from __future__ import annotations

import os


def _print_banner(host: str, port: int, mcp_available: bool, mcp_port: int) -> None:
    print()
    print("=" * 62)
    print("  DeepMemory Server")
    print("=" * 62)
    print(f"  HTTP API          http://{host}:{port}")
    print(f"  Health check      http://{host}:{port}/health")
    print(f"  API docs          http://{host}:{port}/docs")
    if mcp_available:
        print(f"  MCP endpoint      http://{host}:{mcp_port}/mcp")
        print("    Tools: deepmem_write, deepmem_search")
    print("-" * 62)
    qdrant = os.environ.get("QDRANT_URL", "").strip()
    print(f"  Qdrant:           {qdrant if qdrant else 'local file (./data/qdrant)'}")
    print("=" * 62)
    print()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="DeepMemory Server")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="HTTP API port (default: 8000)")
    parser.add_argument("--mcp-port", type=int, default=8001,
                        help="MCP streamable-http port (default: 8001)")
    parser.add_argument("--no-mcp", action="store_true",
                        help="Disable MCP server entirely")
    args = parser.parse_args()

    mcp_available = False
    if not args.no_mcp:
        try:
            from server.mcp_server import mcp as _mcp  # noqa: F401
            mcp_available = True
            os.environ["MCP_PORT"] = str(args.mcp_port)
        except Exception:
            mcp_available = False

    if args.no_mcp:
        os.environ["DEEPMEMORY_NO_MCP"] = "1"

    _print_banner(args.host, args.port, mcp_available, args.mcp_port)

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
