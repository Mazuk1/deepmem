import pytest


class TestModelRouter:
    @pytest.fixture
    def router(self, config):
        from deepmem.engine.model_router import ModelRouter
        return ModelRouter(
            deepseek_api_key=config.deepseek_api_key,
            deepseek_base_url=config.deepseek_base_url,
        )

    def test_route_free_tier_gets_deepseek(self, router):
        from deepmem.interface import Tenant
        from deepmem.llm import UniversalLLMAdapter
        tenant = Tenant(user_id="test_user")
        provider = router.route(tenant, tier="free")
        assert isinstance(provider, UniversalLLMAdapter)
        assert "deepseek" in provider.base_url or "flash" in provider.model

    def test_route_premium_uses_same_model_as_free(self, router):
        # free and premium tiers currently share the same model. The tier knob
        # is kept as a forwarding API so a tier-specific model can be lit up
        # later without touching call sites; right now both must point at the
        # same adapter so behavior is deterministic.
        from deepmem.interface import Tenant
        free = router.route(Tenant(user_id="free_user"), tier="free")
        premium = router.route(Tenant(user_id="premium_user"), tier="premium")
        assert free.model == premium.model
        assert "deepseek" in premium.base_url or "deepseek" in premium.model

    def test_route_byok_gets_custom_provider(self, router):
        from deepmem.interface import Tenant
        tenant = Tenant(user_id="byok_user")
        provider = router.route(tenant, custom_api_key="sk-test",
                               custom_base_url="https://api.openai.com/v1")
        assert provider is not None

    def test_route_unknown_tier_falls_back(self, router):
        from deepmem.interface import Tenant
        tenant = Tenant(user_id="test_user")
        provider = router.route(tenant, tier="unknown")
        assert provider is not None
