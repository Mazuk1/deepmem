#!/usr/bin/env python
"""Functional test - HTTP end-to-end. Boots a real DeepMem server and drives
every endpoint with real data, printing the actual responses. The companion
to func_test_direct.py: that one calls the engine functions in-process, this
one goes over HTTP through FastAPI + the full middleware stack (rate limiting,
cache, batch distiller, hybrid search).

Run:
    python scripts/func_test_http.py
    BGE_DEVICE=cpu python scripts/func_test_http.py   # force CPU BGE

The script process itself never loads BGE-M3 (it uses HTTP), so there's no
GPU contention - only the booted server does, and you can pick its device.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QDRANT = os.path.join(tempfile.gettempdir(), f"deepmem_http_functest_{uuid.uuid4().hex[:8]}")
_PORT = 8731
_BASE = f"http://127.0.0.1:{_PORT}"


def _hr(title):
    print(f"\n{'═' * 60}\n {title}\n{'═' * 60}")


def _show(label, r):
    body = r.text
    if len(body) > 240:
        body = body[:240] + "…"
    print(f"  {label}: {r.status_code}  {body}")


def main():
    env = os.environ.copy()
    env["QDRANT_PATH"] = _QDRANT
    env["DEEPMEMORY_DEBUG"] = "1"        # unlocks /v1/_debug/flush
    env["DEEPMEMORY_NO_MCP"] = "1"       # we test HTTP, not MCP; skip :8001
    env.setdefault("BGE_DEVICE", "auto")
    # Truncate the log so we can tail this run cleanly.
    try:
        open(os.path.join(REPO, "logs", "deepmem.log"), "w").close()
    except Exception:
        pass

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app",
         "--host", "127.0.0.1", "--port", str(_PORT), "--log-level", "warning"],
        cwd=REPO, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    def stop():
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    try:
        t0 = time.perf_counter()
        ready = False
        while time.perf_counter() - t0 < 90:
            try:
                if requests.get(f"{_BASE}/health", timeout=1).status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.5)
        if not ready:
            print("SERVER DID NOT START - check logs/deepmem.log")
            return 1
        print(f"server up in {time.perf_counter()-t0:.1f}s "
              f"(BGE_DEVICE={env['BGE_DEVICE']})")

        user = f"http_user_{uuid.uuid4().hex[:6]}"

        # 1. health / ready
        _hr("1. health & ready")
        _show("GET /health", requests.get(f"{_BASE}/health", timeout=5))
        _show("GET /ready", requests.get(f"{_BASE}/ready", timeout=5))

        # 2. write with LLM extraction (batched) + flush
        _hr("2. POST /v1/memories (infer=True, batched) + flush")
        msgs = [{"role": "user",
                 "content": "I'm Li Si, I do data science in Shanghai, commonly use Python and PyTorch, and I have a cat."}]
        r = requests.post(f"{_BASE}/v1/memories",
                          json={"user_id": user, "messages": msgs, "infer": True},
                          timeout=120)
        _show("POST /v1/memories", r)
        _show("POST /v1/_debug/flush",
              requests.post(f"{_BASE}/v1/_debug/flush",
                            params={"user_id": user}, timeout=120))
        time.sleep(0.3)

        # 3. list
        _hr("3. GET /v1/memories (list)")
        r = requests.get(f"{_BASE}/v1/memories", params={"user_id": user}, timeout=30)
        _show("GET /v1/memories", r)
        items = r.json().get("results", [])
        first_id = items[0]["id"] if items else None
        print(f"  -> {len(items)} memory(ies), first_id={first_id}")

        # 4. search
        _hr("4. POST /v1/memories/search")
        for q in ["Where does Li Si live?", "What tech stack does Li Si use?"]:
            r = requests.post(f"{_BASE}/v1/memories/search",
                              json={"user_id": user, "query": q, "top_k": 3},
                              timeout=30)
            hits = r.json().get("results", [])
            print(f"  q='{q}' -> {len(hits)} hit(s)")
            for h in hits:
                print(f"      [{h['score']:.3f}] {h['memory']}")

        # 5. get one by id
        if first_id:
            _hr("5. GET /v1/memories/{id}")
            _show(f"GET /{first_id[:8]}…",
                  requests.get(f"{_BASE}/v1/memories/{first_id}",
                               params={"user_id": user}, timeout=30))

            # 6. update
            _hr("6. PUT /v1/memories/{id} (update)")
            r = requests.put(f"{_BASE}/v1/memories/{first_id}",
                             json={"user_id": user,
                                   "memory": "Li Si does data science in Shanghai and has a cat named Orange."},
                             timeout=30)
            _show("PUT", r)

            # 7. delete one
            _hr("7. DELETE /v1/memories/{id} (soft)")
            _show("DELETE one",
                  requests.delete(f"{_BASE}/v1/memories/{first_id}",
                                  params={"user_id": user}, timeout=30))

        # 8. raw write (infer=False) - synchronous, no flush
        _hr("8. POST /v1/memories (infer=False, raw, synchronous)")
        r = requests.post(f"{_BASE}/v1/memories",
                          json={"user_id": user,
                                "messages": [{"role": "user", "content": "raw: I love ramen."}],
                                "infer": False},
                          timeout=30)
        _show("POST infer=False", r)

        # 9. invalid user_id rejected
        _hr("9. invalid user_id -> 400")
        r = requests.post(f"{_BASE}/v1/memories",
                          json={"user_id": "   ",
                                "messages": [{"role": "user", "content": "x"}]},
                          timeout=10)
        _show("POST bad user_id", r)

        # 10. delete all (GDPR)
        _hr("10. DELETE /v1/memories (delete_all, GDPR)")
        r = requests.delete(f"{_BASE}/v1/memories",
                            params={"user_id": user}, timeout=30)
        _show("DELETE all", r)
        r = requests.get(f"{_BASE}/v1/memories",
                         params={"user_id": user}, timeout=30)
        print(f"  list after delete_all -> {len(r.json().get('results', []))} (expect 0)")

        print(f"\n✓ HTTP functional test done.")
        return 0
    finally:
        stop()
        shutil.rmtree(_QDRANT, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
