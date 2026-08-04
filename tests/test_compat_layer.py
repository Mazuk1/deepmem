"""Tests for the Mem0 compatibility layer (deepmem.compat).

Verifies mem0_client issues the right HTTP calls to the DeepMem API and
reshapes responses into Mem0's {"results": [...]} envelope - without a
running server (httpx.MockTransport stands in for the network).
"""
import json

import httpx
import pytest

from deepmem.compat import Mem0CompatClient


def _client_with_handler(handler):
    """Build a Mem0CompatClient whose httpx.Client uses a mock transport."""
    c = Mem0CompatClient(base_url="http://test")
    c._client.close()
    c._client = httpx.Client(transport=httpx.MockTransport(handler), timeout=10)
    return c


def _recorder():
    """Return (handler, calls) - calls captures (method, path, params, body)."""
    calls = []

    def handler(request: httpx.Request):
        calls.append((
            request.method,
            request.url.path,
            dict(request.url.params),
            request.content.decode() if request.content else None,
        ))
        return httpx.Response(200, json={"results": []})

    return handler, calls


class TestCompatShape:
    def test_add_posts_to_memories_with_infer_default(self):
        h, calls = _recorder()
        c = _client_with_handler(h)
        c.add([{"role": "user", "content": "hi"}], user_id="u1")
        method, path, _, body = calls[0]
        assert method == "POST" and path == "/v1/memories"
        jb = json.loads(body)
        assert jb["infer"] is True
        assert jb["user_id"] == "u1"

    def test_add_infer_false_passes_through(self):
        h, calls = _recorder()
        c = _client_with_handler(h)
        c.add([{"role": "user", "content": "hi"}], user_id="u1", infer=False)
        assert json.loads(calls[0][3])["infer"] is False

    def test_add_sets_relations_envelope(self):
        def h(req):
            return httpx.Response(200, json={"results": [
                {"id": "x", "memory": "m", "event": "ADD"}]})
        c = _client_with_handler(h)
        r = c.add([{"role": "user", "content": "hi"}], user_id="u1")
        assert r["results"][0]["memory"] == "m"
        assert r["relations"] == []  # Mem0 envelope key always present

    def test_search_posts_to_search_endpoint(self):
        h, calls = _recorder()
        c = _client_with_handler(h)
        c.search("where?", user_id="u1", limit=5)
        method, path, _, body = calls[0]
        assert method == "POST" and path == "/v1/memories/search"
        assert json.loads(body)["top_k"] == 5

    def test_get_all_uses_query_params(self):
        h, calls = _recorder()
        c = _client_with_handler(h)
        c.get_all(user_id="u1", limit=50)
        method, path, params, _ = calls[0]
        assert method == "GET" and path == "/v1/memories"
        assert params["user_id"] == "u1" and params["limit"] == "50"

    def test_delete_uses_delete_method_and_shapes_status(self):
        h, calls = _recorder()
        c = _client_with_handler(h)
        r = c.delete("mem-123")
        method, path = calls[0][0], calls[0][1]
        assert method == "DELETE" and path == "/v1/memories/mem-123"
        assert r["results"][0] == {"id": "mem-123", "status": "deleted"}

    def test_delete_all_returns_done(self):
        h, calls = _recorder()
        c = _client_with_handler(h)
        r = c.delete_all(user_id="u1")
        assert calls[0][0] == "DELETE"
        assert r == {"status": "done"}

    def test_reset_requires_user_id(self):
        c = _client_with_handler(_recorder()[0])
        with pytest.raises(ValueError):
            c.reset()

    def test_reset_posts_confirm_user_id(self):
        h, calls = _recorder()
        c = _client_with_handler(h)
        c.reset(user_id="u1")
        method, path, _, body = calls[0]
        assert method == "POST" and path == "/v1/reset"
        jb = json.loads(body)
        assert jb["user_id"] == "u1" and jb["confirm_user_id"] == "u1"
