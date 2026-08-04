"""Phase 3.3: BYOK adapter helper unit tests.

`build_byok_config` / `adapter_from_byok_config` exist so the direct
`/v1/memories` request path and the batched `VectorStore.process_batch`
path cannot drift on BYOK defaults. These tests pin the contract.
"""
import pytest

from deepmem.engine.model_router import (
    BYOK_DEFAULT_BASE_URL,
    BYOK_DEFAULT_MODEL,
    adapter_from_byok_config,
    build_byok_config,
)
from deepmem.llm import UniversalLLMAdapter


class TestBuildByokConfig:
    def test_returns_none_when_api_key_missing(self):
        assert build_byok_config(None) is None
        assert build_byok_config("") is None

    def test_fills_defaults_when_caller_omits_them(self):
        cfg = build_byok_config("sk-test")
        assert cfg == {
            "api_key": "sk-test",
            "base_url": BYOK_DEFAULT_BASE_URL,
            "model": BYOK_DEFAULT_MODEL,
        }

    def test_caller_overrides_win_over_defaults(self):
        cfg = build_byok_config(
            "sk-test",
            base_url="https://api.openai.com/v2",
            model="gpt-5",
        )
        assert cfg["base_url"] == "https://api.openai.com/v2"
        assert cfg["model"] == "gpt-5"

    def test_empty_string_overrides_fall_back_to_defaults(self):
        # falsy string values should not poison the config — the helper's
        # job is to guarantee non-empty base_url / model so the adapter never
        # gets handed "" as a base url.
        cfg = build_byok_config("sk-test", base_url="", model="")
        assert cfg["base_url"] == BYOK_DEFAULT_BASE_URL
        assert cfg["model"] == BYOK_DEFAULT_MODEL


class TestAdapterFromByokConfig:
    def test_returns_universal_adapter_with_supplied_values(self):
        cfg = {
            "api_key": "sk-test",
            "base_url": "https://example.test/v1",
            "model": "custom-model",
        }
        adapter = adapter_from_byok_config(cfg)
        assert isinstance(adapter, UniversalLLMAdapter)
        assert adapter.base_url == "https://example.test/v1"
        assert adapter.model == "custom-model"

    def test_falls_back_to_module_defaults_when_keys_missing(self):
        adapter = adapter_from_byok_config({"api_key": "sk-test"})
        assert adapter.base_url == BYOK_DEFAULT_BASE_URL
        assert adapter.model == BYOK_DEFAULT_MODEL

    def test_pairs_with_build_byok_config_round_trip(self):
        # Phase 3.3 invariant: every consumer pipes build → adapter without
        # touching the dict in between. If round-tripping ever stops working
        # the two call sites WILL drift again.
        cfg = build_byok_config("sk-test",
                                 base_url="https://router.example/v1",
                                 model="route-1")
        adapter = adapter_from_byok_config(cfg)
        assert adapter.base_url == "https://router.example/v1"
        assert adapter.model == "route-1"
