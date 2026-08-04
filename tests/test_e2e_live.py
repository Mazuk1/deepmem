"""End-to-end tests against a real uvicorn process.

Run:
    set TEST_LIVE_SERVER=1
    pytest tests/test_e2e_live.py -v

The `live_server` fixture (in conftest.py) boots uvicorn ONCE for the whole
session, loading the model exactly once.
"""

import time
import uuid

import pytest
import requests


def _user():
    return f"e2e_user_{uuid.uuid4().hex[:8]}"


def test_health(live_server):
    r = requests.get(f"{live_server}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_pipeline_write_search_list_delete(live_server):
    """Write conversation → flush → search → list → delete."""
    user = _user()

    # 1. Write (returns immediately because of batching)
    msgs = [
        {"role": "user", "content": "I'm Zhang San, I live in Beijing, I'm a programmer, and I like playing basketball"},
        {"role": "assistant", "content": "Hello Zhang San!"},
    ]
    r = requests.post(f"{live_server}/v1/memories",
                      json={"user_id": user, "messages": msgs, "infer": True},
                      timeout=120)
    assert r.status_code == 200, r.text

    # 2. Flush — synchronously runs LLM extract + Qdrant upsert
    r = requests.post(f"{live_server}/v1/_debug/flush",
                      params={"user_id": user}, timeout=120)
    assert r.status_code == 200, r.text
    assert r.json()["flushed"] is True

    # Tiny buffer for Qdrant local-mode write visibility
    time.sleep(0.5)

    # 3. List — should have facts now
    r = requests.get(f"{live_server}/v1/memories",
                     params={"user_id": user}, timeout=30)
    assert r.status_code == 200
    items = r.json()["results"]
    assert len(items) > 0, f"Expected facts after flush, got: {r.json()}"

    # Sanity: at least one fact should mention Zhang San or Beijing
    memories = " ".join(it["memory"] for it in items)
    assert any(kw in memories for kw in ["Zhang San", "Beijing", "programmer", "basketball"]), \
        f"Memories missing expected keywords: {memories}"

    # 4. Search
    r = requests.post(f"{live_server}/v1/memories/search",
                      json={"user_id": user, "query": "Where does Zhang San live?", "top_k": 5},
                      timeout=30)
    assert r.status_code == 200
    hits = r.json()["results"]
    assert len(hits) > 0

    # 5. Delete all (GDPR)
    r = requests.delete(f"{live_server}/v1/memories",
                        params={"user_id": user}, timeout=30)
    assert r.status_code == 200
    assert r.json()["deleted_count"] >= len(items)

    # 6. List should now be empty
    r = requests.get(f"{live_server}/v1/memories",
                     params={"user_id": user}, timeout=30)
    assert r.json()["results"] == []


def test_user_isolation(live_server):
    """alice and bob should not see each other's memories."""
    alice = _user()
    bob = _user()

    # alice writes
    requests.post(f"{live_server}/v1/memories",
                  json={"user_id": alice,
                        "messages": [{"role": "user", "content": "I'm Alice, I live in Shanghai"}],
                        "infer": True}, timeout=60).raise_for_status()
    requests.post(f"{live_server}/v1/_debug/flush",
                  params={"user_id": alice}, timeout=60).raise_for_status()

    # bob writes
    requests.post(f"{live_server}/v1/memories",
                  json={"user_id": bob,
                        "messages": [{"role": "user", "content": "I'm Bob, I live in Tokyo"}],
                        "infer": True}, timeout=60).raise_for_status()
    requests.post(f"{live_server}/v1/_debug/flush",
                  params={"user_id": bob}, timeout=60).raise_for_status()

    time.sleep(0.5)

    # alice can only see Alice
    a = requests.get(f"{live_server}/v1/memories",
                     params={"user_id": alice}, timeout=30).json()["results"]
    a_text = " ".join(x["memory"] for x in a).lower()
    assert "tokyo" not in a_text and "bob" not in a_text

    # bob can only see Bob
    b = requests.get(f"{live_server}/v1/memories",
                     params={"user_id": bob}, timeout=30).json()["results"]
    b_text = " ".join(x["memory"] for x in b)
    assert "Shanghai" not in b_text and "Alice" not in b_text

    # cleanup
    requests.delete(f"{live_server}/v1/memories", params={"user_id": alice}, timeout=30)
    requests.delete(f"{live_server}/v1/memories", params={"user_id": bob}, timeout=30)


def test_invalid_user_id_rejected(live_server):
    r = requests.post(f"{live_server}/v1/memories",
                      json={"user_id": "   ",
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=10)
    assert r.status_code == 400


def test_infer_false_stores_raw_messages(live_server):
    """infer=False bypasses LLM and the silence window — synchronous."""
    user = _user()
    r = requests.post(f"{live_server}/v1/memories",
                      json={"user_id": user,
                            "messages": [{"role": "user", "content": "raw memory test"}],
                            "infer": False},
                      timeout=30)
    assert r.status_code == 200
    assert len(r.json()["results"]) > 0  # immediate, no flush needed

    requests.delete(f"{live_server}/v1/memories", params={"user_id": user}, timeout=30)
