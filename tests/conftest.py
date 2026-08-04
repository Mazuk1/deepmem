import os
import socket
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Use vendored core engine from deepmem/vendor/ (imports as _core package)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deepmem", "vendor"))

# DEEPMEMORY_DEBUG=1 unlocks the dev-only /v1/_debug/flush route AND tells
# the CORS_ORIGINS='*' startup guard in server.main to allow the wildcard.
# Production guarantees still hold because the env var must be set
# explicitly - the suite is opting in on purpose.
os.environ.setdefault("DEEPMEMORY_DEBUG", "1")

# Isolate the global Qdrant store from any stale ./data/qdrant collection
# (e.g. a 3072-dim collection left over from a different embedder backend).
# Tests that need their own store override this path; this is just the safe
# default for server-level tests that go through the get_services() singleton.
import tempfile
import uuid as _uuid
from deepmem.config import config as _cfg
_cfg.qdrant_path = os.path.join(
    tempfile.gettempdir(), f"deepmem_test_qdrant_{_uuid.uuid4().hex[:8]}"
)


@pytest.fixture
def config():
    from deepmem.config import DeepMemoryConfig
    return DeepMemoryConfig.from_json()


@pytest.fixture
def sample_messages():
    return [
        {"role": "user", "content": "My name is Alice, I live in San Francisco."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
    ]


@pytest.fixture
def sample_user_id():
    return "test_user_alice"


# ── Session-scoped uvicorn for e2e tests ──────────────────────────────
# Boots ONCE per pytest session; embedding model is loaded once.
# Activate by setting TEST_LIVE_SERVER=1 — without it the fixture skips,
# so the fast unit-test suite stays fast.

def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server():
    if os.environ.get("TEST_LIVE_SERVER") != "1":
        pytest.skip("set TEST_LIVE_SERVER=1 to run live-server e2e tests")

    import requests

    port = _find_free_port()
    env = os.environ.copy()
    env["DEEPMEMORY_DEBUG"] = "1"
    # Isolate the booted server's Qdrant store from the dev ./data/qdrant,
    # which may hold a collection created at a different embedding dim by a
    # previous provider (e.g. 3072 from Google vs 1024 from BGE-M3). A dim
    # mismatch makes every upsert fail and leaves local Qdrant in a bad state
    # where subsequent reads hang. A fresh temp store always matches the
    # currently-configured embedder.
    import tempfile as _tempfile
    import uuid as _uuid
    _e2e_qdrant = os.path.join(
        _tempfile.gettempdir(), f"deepmem_e2e_qdrant_{_uuid.uuid4().hex[:8]}")
    env["QDRANT_PATH"] = _e2e_qdrant
    # Force the e2e server's BGE-M3 onto CPU. The pytest process may already
    # hold BGE-M3 on the GPU for in-process unit tests; a second GPU instance
    # (~3.1GB VRAM each) on a single 8GB card contends/OOMs and makes the
    # e2e server stall. CPU BGE is ~0.3s/batch - fine for 5 e2e requests and
    # removes the contention entirely.
    env["BGE_DEVICE"] = "cpu"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "server.main:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        env=env,
        # DEVNULL, NOT PIPE: an undrained PIPE deadlocks once the server's log
        # output fills the OS pipe buffer (~64KB), at which point the server
        # blocks on its next log write and every in-flight request times out.
        # Server logs still go to logs/deepmem.log for diagnostics.
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    started = False
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base}/health", timeout=1)
            if r.status_code == 200:
                started = True
                break
        except requests.exceptions.RequestException:
            time.sleep(0.5)

    if not started:
        proc.terminate()
        raise RuntimeError(
            "Server failed to start within 60s - check logs/deepmem.log for errors."
        )

    yield base

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Clean up the isolated e2e Qdrant store.
    import shutil as _shutil
    if os.path.exists(_e2e_qdrant):
        try:
            _shutil.rmtree(_e2e_qdrant)
        except OSError:
            pass
