"""Test hybrid search: vector + BM25 keyword + time decay + entity boost."""

import os
import time
import pytest
import numpy as np
from deepmem.interface import Tenant


class TestTimeDecayScoring:
    """Verify time decay weighting on search results."""

    def test_newer_memory_scores_higher(self):
        from deepmem.hybrid_search import compute_time_decay
        now = time.time()
        score_new = compute_time_decay(now - 60, now)        # 1 minute ago
        score_old = compute_time_decay(now - 86400 * 30, now)  # 30 days ago
        assert score_new > score_old

    def test_same_time_same_score(self):
        from deepmem.hybrid_search import compute_time_decay
        now = time.time()
        score1 = compute_time_decay(now - 3600, now)
        score2 = compute_time_decay(now - 3600, now)
        assert abs(score1 - score2) < 0.0001

    def test_decay_at_boundary(self):
        from deepmem.hybrid_search import compute_time_decay
        now = time.time()
        score_now = compute_time_decay(now, now)  # Just now
        assert 0.9 <= score_now <= 1.0

    def test_very_old_memory_near_zero(self):
        from deepmem.hybrid_search import compute_time_decay
        now = time.time()
        score_ancient = compute_time_decay(now - 86400 * 365, now)  # 1 year ago
        assert score_ancient < 0.1


class TestHybridScoreFusion:
    """Verify hybrid scoring combines vector + keyword + time + entity."""

    def test_fusion_with_all_components(self):
        from deepmem.hybrid_search import fuse_scores
        score = fuse_scores(
            vector_score=0.85,
            keyword_score=0.60,
            entity_boost=1.0,
            time_decay=0.95,
        )
        assert 0 < score <= 1.0

    def test_keyword_only_without_vector(self):
        from deepmem.hybrid_search import fuse_scores
        # When vector score is low but keyword is high
        score = fuse_scores(
            vector_score=0.2,
            keyword_score=0.9,
            entity_boost=1.0,
            time_decay=1.0,
        )
        assert score > 0.2  # Keyword should pull up the score

    def test_time_decay_penalizes(self):
        from deepmem.hybrid_search import fuse_scores
        score_fresh = fuse_scores(0.8, 0.8, 1.0, 1.0)
        score_stale = fuse_scores(0.8, 0.8, 1.0, 0.3)
        assert score_fresh > score_stale

    def test_entity_boost_amplifies(self):
        from deepmem.hybrid_search import fuse_scores
        score_normal = fuse_scores(0.8, 0.8, 1.0, 1.0)
        score_boosted = fuse_scores(0.8, 0.8, 1.5, 1.0)
        assert score_boosted > score_normal


class TestHybridSearchEndToEnd:
    """End-to-end: store memories with timestamps, search with hybrid scoring."""

    @pytest.fixture(scope="class")
    def store(self):
        import uuid, shutil
        from deepmem.config import config as _cfg
        test_path = f"./data/test_hybrid_{uuid.uuid4().hex[:8]}"
        from deepmem.vector_store import VectorStore
        store = VectorStore(
            qdrant_path=test_path,
            embedding_dims=_cfg.embedding_dims,
            bge_m3_path=_cfg.bge_m3_path,
        )
        yield store
        time.sleep(0.5)
        if os.path.exists(test_path):
            try:
                shutil.rmtree(test_path)
            except PermissionError:
                pass

    @pytest.fixture
    def tenant(self):
        return Tenant(user_id="hybrid_test_user")

    @pytest.mark.asyncio
    async def test_search_returns_ranked_results(self, store, tenant):
        """Hybrid search returns results sorted by combined score."""
        await store.add(
            [{"role": "user", "content": "My name is Alice. I work at Google as a software engineer."}],
            tenant, infer=False,
        )
        await store.add(
            [{"role": "user", "content": "I love hiking in the mountains on weekends."}],
            tenant, infer=False,
        )

        results = await store.search("What is Alice's job?", tenant, top_k=5, hybrid=True)
        assert len(results) > 0
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_keyword_match_boosts_result(self, store, tenant):
        """Exact keyword match should rank higher than pure vector match."""
        await store.add(
            [{"role": "user", "content": "My API key is dm-abc123xyz-secret-key-999"}],
            tenant, infer=False,
        )
        await store.add(
            [{"role": "user", "content": "I enjoy writing Python code on weekends."}],
            tenant, infer=False,
        )

        # Search with exact keyword "dm-abc123xyz" should rank that memory higher
        results = await store.search("dm-abc123xyz", tenant, top_k=3, hybrid=True)
        if len(results) >= 2:
            # First result should contain the keyword
            assert "dm-abc123xyz" in results[0].memory or "API key" in results[0].memory

    @pytest.mark.asyncio
    async def test_hybrid_flag_falls_back_to_pure_vector(self, store, tenant):
        """When hybrid=False, should use pure vector search (existing behavior)."""
        await store.add(
            [{"role": "user", "content": "I like cats more than dogs."}],
            tenant, infer=False,
        )

        results_hybrid = await store.search("cats", tenant, top_k=5, hybrid=True)
        results_pure = await store.search("cats", tenant, top_k=5, hybrid=False)
        assert len(results_hybrid) > 0
        assert len(results_pure) > 0
