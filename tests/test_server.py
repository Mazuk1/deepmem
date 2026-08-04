import os
import shutil
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture(scope="class")
async def client():
    # Start the app ONCE for all server tests — avoids repeated model loading.
    test_path = f"./data/test_server_{uuid.uuid4().hex[:8]}"

    import server.dependencies as deps
    deps.config.qdrant_path = test_path
    deps._store = None  # Force re-init with new path

    from server.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test",
    ) as ac:
        yield ac

    # Cleanup
    import time
    time.sleep(0.3)
    if os.path.exists(test_path):
        try:
            shutil.rmtree(test_path)
        except PermissionError:
            pass


class TestServerEndpoints:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_add_memories_infer_false(self, client, sample_messages, sample_user_id):
        """infer=False should store raw messages without LLM."""
        response = await client.post("/v1/memories", json={
            "messages": sample_messages,
            "user_id": sample_user_id,
            "infer": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # infer=False stores raw messages, should have non-empty results
        assert len(data["results"]) > 0, f"Expected non-empty results, got {data}"

    @pytest.mark.asyncio
    async def test_add_memories_defaults_user_id_when_omitted(self, client, sample_messages):
        """Open mode: omitting user_id defaults to the shared "default" partition."""
        response = await client.post("/v1/memories", json={
            "messages": sample_messages,
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_add_then_list(self, client, sample_user_id):
        """Add memories, then list them."""
        await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "My name is Alice, I live in SF"}],
            "user_id": sample_user_id,
            "infer": False,
        })

        response = await client.get("/v1/memories", params={
            "user_id": sample_user_id,
            "limit": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0, f"Expected stored memories, got {data}"

    @pytest.mark.asyncio
    async def test_add_then_search(self, client, sample_user_id):
        """Add memories, then search for them."""
        await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "My name is Alice, I live in SF"}],
            "user_id": sample_user_id,
            "infer": False,
        })

        response = await client.post("/v1/memories/search", json={
            "query": "Where does Alice live?",
            "user_id": sample_user_id,
        })
        assert response.status_code == 200
        assert "results" in response.json()

    @pytest.mark.asyncio
    async def test_search_defaults_user_id_when_omitted(self, client):
        """Open mode: omitting user_id defaults to the shared "default" partition."""
        response = await client.post("/v1/memories/search", json={
            "query": "test query",
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_single_memory(self, client, sample_user_id):
        """Store a memory and retrieve it by ID."""
        add_resp = await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "I love Python programming."}],
            "user_id": sample_user_id,
            "infer": False,
        })
        assert add_resp.status_code == 200
        mem_id = add_resp.json()["results"][0]["id"]

        get_resp = await client.get(f"/v1/memories/{mem_id}", params={"user_id": sample_user_id})
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == mem_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory_returns_404(self, client, sample_user_id):
        response = await client.get("/v1/memories/nonexistent-id", params={"user_id": sample_user_id})
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_memory(self, client, sample_user_id):
        """Store a memory, then update its text."""
        add_resp = await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "I like cats."}],
            "user_id": sample_user_id,
            "infer": False,
        })
        mem_id = add_resp.json()["results"][0]["id"]

        update_resp = await client.put(f"/v1/memories/{mem_id}", json={
            "memory": "I love dogs.",
            "user_id": sample_user_id,
        })
        assert update_resp.status_code == 200
        assert update_resp.json()["memory"] == "I love dogs."

    @pytest.mark.asyncio
    async def test_delete_single_memory(self, client, sample_user_id):
        """Store and then soft-delete a memory."""
        add_resp = await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "Temporary memory to delete."}],
            "user_id": sample_user_id,
            "infer": False,
        })
        mem_id = add_resp.json()["results"][0]["id"]

        del_resp = await client.delete(f"/v1/memories/{mem_id}", params={"user_id": sample_user_id})
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted_count"] == 1

        # Verify it's no longer visible
        get_resp = await client.get(f"/v1/memories/{mem_id}", params={"user_id": sample_user_id})
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_all_endpoint(self, client, sample_user_id):
        """Delete all memories for a user."""
        await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "Test memory 1"}],
            "user_id": sample_user_id,
            "infer": False,
        })
        await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "Test memory 2"}],
            "user_id": sample_user_id,
            "infer": False,
        })

        response = await client.delete("/v1/memories", params={"user_id": sample_user_id})
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] > 0
        assert data["user_id"] == sample_user_id

    @pytest.mark.asyncio
    async def test_empty_user_id_rejected(self, client):
        """An explicitly-empty user_id is a validation error (400), not open mode."""
        response = await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "Test"}],
            "user_id": "",
        })
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_whitespace_user_id_rejected(self, client):
        response = await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "Test"}],
            "user_id": "   ",
        })
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_user_isolation_list(self, client):
        """Verify list is isolated per user."""
        await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "Alice secret: I like cats"}],
            "user_id": "alice_iso",
            "infer": False,
        })
        await client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "Bob secret: I like dogs"}],
            "user_id": "bob_iso",
            "infer": False,
        })

        alice_resp = await client.get("/v1/memories", params={"user_id": "alice_iso"})
        alice_memories = " ".join(r["memory"] for r in alice_resp.json()["results"])
        assert "cat" in alice_memories.lower()
        assert "dog" not in alice_memories.lower()

        bob_resp = await client.get("/v1/memories", params={"user_id": "bob_iso"})
        bob_memories = " ".join(r["memory"] for r in bob_resp.json()["results"])
        assert "dog" in bob_memories.lower()
        assert "cat" not in bob_memories.lower()
