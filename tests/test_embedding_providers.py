"""Embedding provider selection tests.

Verifies get_embedder routes to the right backend (google / openai / bge-m3)
based on config.default_embedder, and that BGE_DEVICE controls GPU/CPU. The
_create_* factories are monkeypatched so no model loads and no network runs.
"""
from types import SimpleNamespace

import pytest

import deepmem.embedder as emb


@pytest.fixture(autouse=True)
def reset_embedder_singleton():
    emb._embedder_instance = None
    yield
    emb._embedder_instance = None


class TestEmbedderProviderSelection:
    def test_selects_google(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(emb, "_create_google", lambda c: sentinel)
        cfg = SimpleNamespace(default_embedder="google")
        assert emb.get_embedder(cfg) is sentinel

    def test_selects_openai(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(emb, "_create_openai", lambda c: sentinel)
        cfg = SimpleNamespace(default_embedder="openai")
        assert emb.get_embedder(cfg) is sentinel

    def test_selects_bge_m3(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(emb, "_create_bge_m3", lambda c: sentinel)
        cfg = SimpleNamespace(default_embedder="bge-m3")
        assert emb.get_embedder(cfg) is sentinel

    def test_unknown_provider_falls_back_to_google(self, monkeypatch):
        # get_embedder's else branch -> _create_google (the dataclass default).
        sentinel = object()
        monkeypatch.setattr(emb, "_create_google", lambda c: sentinel)
        cfg = SimpleNamespace(default_embedder="something-unknown")
        assert emb.get_embedder(cfg) is sentinel

    def test_singleton_cached_across_calls(self, monkeypatch):
        # Once created, the same instance is returned without re-calling the
        # factory - critical so we don't reload BGE-M3 per request.
        calls = []
        def fake(c):
            calls.append(c)
            return object()
        monkeypatch.setattr(emb, "_create_bge_m3", fake)
        cfg = SimpleNamespace(default_embedder="bge-m3")
        first = emb.get_embedder(cfg)
        second = emb.get_embedder(cfg)
        assert first is second
        assert len(calls) == 1


class TestBGEDeviceResolution:
    """BGE_DEVICE env var controls GPU vs CPU (avoids VRAM contention)."""

    def test_force_cpu(self, monkeypatch):
        monkeypatch.setenv("BGE_DEVICE", "cpu")
        from _core.embeddings.bge_m3 import _resolve_device
        assert _resolve_device() == "cpu"

    def test_force_cuda(self, monkeypatch):
        monkeypatch.setenv("BGE_DEVICE", "cuda")
        from _core.embeddings.bge_m3 import _resolve_device
        assert _resolve_device() == "cuda"

    def test_auto_uses_available(self, monkeypatch):
        monkeypatch.delenv("BGE_DEVICE", raising=False)
        from _core.embeddings.bge_m3 import _resolve_device
        import torch
        assert _resolve_device() == ("cuda" if torch.cuda.is_available() else "cpu")

    def test_empty_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("BGE_DEVICE", "  ")
        from _core.embeddings.bge_m3 import _resolve_device
        import torch
        assert _resolve_device() == ("cuda" if torch.cuda.is_available() else "cpu")
