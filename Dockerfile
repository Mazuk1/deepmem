# DeepMem Docker image - CPU by default (works everywhere, smaller).
# For GPU: use a CUDA base image (e.g. nvidia/cuda:12.1.0-runtime-ubuntu22.04)
# and install torch from the CUDA index instead of the CPU line below.
FROM python:3.11-slim

# Build-time deps for sentence-transformers / numpy / rank_bm25.
# APT_MIRROR (optional): swap deb.debian.org for a faster mirror, e.g.
#   --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn
# Leave unset for the official Debian source.
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
        f=$(ls /etc/apt/sources.list.d/*.sources 2>/dev/null | head -1); \
        if [ -n "$f" ]; then sed -i "s|http://deb.debian.org|$APT_MIRROR|g; s|http://security.debian.org|$APT_MIRROR|g" "$f"; \
        else sed -i "s|http://deb.debian.org|$APT_MIRROR|g; s|http://security.debian.org|$APT_MIRROR|g" /etc/apt/sources.list; fi; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU torch FIRST (the default PyPI wheel is a multi-GB CUDA build on
# linux). Installing from the CPU index keeps the image small; the rest of
# requirements.txt then resolves against it without reinstalling.
# GPU alternative:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
# PIP_INDEX_URL / TORCH_INDEX_URL default to the official sources; pass a
# mirror via --build-arg in regions where PyPI / pytorch is slow.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_INDEX_URL=https://pypi.org/simple
# Install torch's pure-Python deps from the (fast) PyPI mirror first, then
# torch itself --no-deps from the CPU-only pytorch index. The pytorch index
# also serves those small dep wheels but very slowly from some regions; this
# split avoids a multi-minute stall on networkx/sympy/fsspec.
RUN pip install --no-cache-dir \
        filelock typing-extensions sympy networkx jinja2 fsspec \
        --index-url ${PIP_INDEX_URL} && \
    pip install --no-cache-dir torch --no-deps --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url ${PIP_INDEX_URL}

# Copy the application (vendor core, server, deepmem, scripts). .dockerignore
# keeps data/, logs/, venv/, .env, and the local model dir out of the image.
COPY . .

# Local-file Qdrant under /data by default; docker-compose overrides
# QDRANT_URL to point at the sidecar Qdrant service. HTTP on 8000, MCP on 8001.
ENV QDRANT_PATH=/data/qdrant \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache

EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run as a module (not `python server/start.py`) so the repo root /app is on
# sys.path and `import server.main` / `import server.mcp_server` resolve.
CMD ["python", "-m", "server.start"]
