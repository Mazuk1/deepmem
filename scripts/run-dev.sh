#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Load environment variables from .env
if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "==========================================="
echo "  DeepMemory backend starting (dev mode)"
echo "==========================================="
echo "  CORS_ORIGINS        = ${CORS_ORIGINS:-*}"
echo "  BGE_M3_PATH         = ${BGE_M3_PATH:-./data/bge-m3}"
echo "==========================================="
echo ""

# Create venv if missing and install deps
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

exec "$PYTHON" -m uvicorn server.main:app --host 0.0.0.0 --port 8000
