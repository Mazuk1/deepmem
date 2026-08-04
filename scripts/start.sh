#!/usr/bin/env bash
# ============================================
# DeepMemory backend start script
#
# Usage:
#   ./scripts/start.sh              # start (checks venv first, then runs uvicorn)
#   ./scripts/start.sh --debug      # dev mode (reload + debug)
#
# Logs:
#   Simultaneously to stdout + logs/deepmem.log (10MB x 5 rotated)
#   service.log: systemd redirect log
#   deepmem.log: application log (pure app, no systemd wrapper)
# ============================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Load .env ───────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
fi

MODE="${1:-}"

# ── Create required directories ─────────────────────────────────────
mkdir -p logs data

# ── venv ─────────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv venv
fi
PYTHON="$(pwd)/venv/bin/python"

if [ ! -f "venv/.install-stamp" ]; then
    echo "[setup] Installing dependencies..."
    "$PYTHON" -m pip install --upgrade pip -q
    "$PYTHON" -m pip install -r requirements.txt -q
    touch venv/.install-stamp
fi

# ── Startup info ─────────────────────────────────────────────────────
echo "==========================================="
echo "  DeepMemory backend"
echo "==========================================="
echo "  CORS_ORIGINS      = ${CORS_ORIGINS:-*}"
echo "  BGE_M3_PATH       = ${BGE_M3_PATH:-./data/bge-m3}"
echo "  LOGS              = stdout + logs/deepmem.log (10MB x 5)"
echo "==========================================="
echo ""

# ── Start ───────────────────────────────────────────────────────────
if [ "$MODE" = "--debug" ]; then
    echo "[debug] Starting with reload..."
    export DEEPMEMORY_DEBUG=1
    exec "$PYTHON" -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1 --reload --log-level info
else
    echo "[prod] Starting..."
    exec "$PYTHON" -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
fi
