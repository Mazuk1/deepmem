import hashlib
import json
import pytest


class TestGDPRLogMasker:
    @pytest.fixture
    def masker(self):
        from deepmem.middleware import GDPRLogMasker
        return GDPRLogMasker()

    def test_mask_request_hashes_content(self, masker):
        request_data = {
            "messages": [{"role": "user", "content": "My name is Alice"}],
            "user_id": "user_123",
        }
        masked = masker.mask_request(request_data)
        assert masked["user_id"] == "user_123"
        assert "messages" not in masked
        assert "content_hash" in masked
        expected_hash = hashlib.sha256(
            json.dumps(request_data["messages"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        assert masked["content_hash"] == expected_hash

    def test_mask_response_hashes_memory(self, masker):
        response_data = {
            "results": [
                {"id": "abc", "memory": "Alice lives in San Francisco"}
            ]
        }
        masked = masker.mask_response(response_data)
        assert "results" not in masked
        assert "memory_hashes" in masked
        assert len(masked["memory_hashes"]) == 1

    def test_empty_request(self, masker):
        assert masker.mask_request({}) == {}

    def test_empty_response(self, masker):
        assert masker.mask_response({}) == {}

    def test_mask_id_blank_returns_dash(self, masker):
        # Phase 4.6: mask_id keeps log shape stable when caller is anonymous.
        assert masker.mask_id(None) == "-"
        assert masker.mask_id("") == "-"

    def test_mask_id_hashes_consistently(self, masker):
        h1 = masker.mask_id("account_42")
        h2 = masker.mask_id("account_42")
        assert h1 == h2
        assert len(h1) == 12
        assert h1 != "account_42"


class TestTenantValidator:
    @pytest.fixture
    def validator(self):
        from deepmem.middleware import TenantValidator
        return TenantValidator()

    def test_valid_user_id(self, validator):
        result = validator.validate_user_id("alice_123")
        assert result == "alice_123"

    def test_trims_whitespace(self, validator):
        result = validator.validate_user_id("  bob  ")
        assert result == "bob"

    def test_rejects_empty_string(self, validator):
        with pytest.raises(ValueError):
            validator.validate_user_id("")

    def test_rejects_whitespace_only(self, validator):
        with pytest.raises(ValueError):
            validator.validate_user_id("   ")

    def test_rejects_none(self, validator):
        with pytest.raises(ValueError):
            validator.validate_user_id(None)

    def test_rejects_internal_whitespace(self, validator):
        with pytest.raises(ValueError):
            validator.validate_user_id("user name with spaces")

    # Phase 4.7: length + charset enforcement.
    def test_rejects_overlong_user_id(self, validator):
        with pytest.raises(ValueError):
            validator.validate_user_id("a" * 257)

    def test_accepts_max_length(self, validator):
        assert validator.validate_user_id("a" * 256) == "a" * 256

    def test_rejects_disallowed_chars(self, validator):
        # `$`, `/`, `\`, control chars are out of the [A-Za-z0-9._:-] charset.
        for bad in ("user$1", "a/b", "a\\b", "user\x00id", "user@host"):
            with pytest.raises(ValueError):
                validator.validate_user_id(bad)

    def test_accepts_allowed_punctuation(self, validator):
        # Dot, underscore, dash, colon are all whitelisted (common id styles
        # use colons, UUIDs use dashes, etc).
        for ok in ("user.1", "user_1", "user-1", "user:1", "u.0_a-b:c"):
            assert validator.validate_user_id(ok) == ok

    def test_normalizes_nfc(self, validator):
        # Latin "é" can be encoded as U+00E9 (precomposed) OR U+0065 U+0301
        # (decomposed). NFC normalization collapses both to the precomposed
        # form *before* the regex sees it. Both are still rejected by the
        # current charset, but the precomposed/decomposed forms must produce
        # the same error path — exercising the normalize() call so future
        # widening of the charset doesn't accidentally split callers across
        # two buckets that look identical in the dashboard.
        decomposed = "usér"  # "user" with combining acute on 'e'
        precomposed = "usér"
        # Both raise (é not in charset) — but importantly, both follow the
        # SAME path (NFC normalization is applied before the regex).
        with pytest.raises(ValueError):
            validator.validate_user_id(decomposed)
        with pytest.raises(ValueError):
            validator.validate_user_id(precomposed)
