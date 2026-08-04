"""GDPR-compliant logging and request processing."""

import hashlib
import json
import logging
import re
import unicodedata
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GDPRLogMasker:
    """Masks sensitive user content in logs — only hashes are logged."""

    @staticmethod
    def _hash_str(s: str) -> str:
        """Return a short SHA256 hash prefix for log-safe user identification."""
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    @classmethod
    def mask_id(cls, value: Optional[str]) -> str:
        """Hash an optional identifier (user_id, account_id) for log output.

        Returns ``"-"`` when the value is missing/blank so every log line has
        a stable shape regardless of whether the caller is signed in.
        """
        if not value:
            return "-"
        return cls._hash_str(value)

    @staticmethod
    def hash_content(content: Any) -> str:
        raw = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def mask_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a log-safe version of the request data."""
        if not request_data:
            return {}
        safe = {}
        for field in ("user_id", "agent_id", "run_id"):
            if field in request_data:
                safe[field] = request_data[field]
        if "messages" in request_data:
            safe["content_hash"] = self.hash_content(request_data["messages"])
        if "query" in request_data:
            safe["query_hash"] = self.hash_content(request_data["query"])
        return safe

    def mask_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a log-safe version of the response data."""
        if not response_data:
            return {}
        safe = {}
        if "results" in response_data:
            safe["memory_hashes"] = [
                self.hash_content(r.get("memory", ""))
                for r in response_data["results"]
            ]
            safe["result_count"] = len(response_data["results"])
        return safe


class TenantValidator:
    """Middleware to validate tenant isolation requirements."""

    # 1..256 chars; allow letters, digits, dash, underscore, dot, colon. The
    # character class is deliberately restrictive: user_id flows into Qdrant
    # payload filters, log lines, and the semantic-cache bucket key — control
    # characters / NBSPs / homoglyphs would either bypass the "no whitespace"
    # rule below or split callers across two buckets that look identical in
    # the dashboard. 256 is the upper bound because nothing legitimate
    # exceeds it and an unbounded string is a memory amplifier.
    _USER_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,256}$")

    @classmethod
    def validate_user_id(cls, user_id: Optional[str]) -> str:
        """Validate and normalize user_id for multi-tenant isolation."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")
        # Reject NFC-unsafe inputs before stripping so a homoglyph attack
        # can't survive normalization and bypass the regex.
        trimmed = unicodedata.normalize("NFC", user_id).strip()
        if any(c.isspace() for c in trimmed):
            raise ValueError("user_id cannot contain whitespace")
        if not cls._USER_ID_RE.fullmatch(trimmed):
            raise ValueError(
                "user_id must be 1..256 chars from [A-Za-z0-9._:-]; "
                "use a stable opaque id (uuid, hash, or your own user pk)."
            )
        return trimmed
