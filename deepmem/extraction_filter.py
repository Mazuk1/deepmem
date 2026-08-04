"""Lightweight filters for memory fact extraction.

The memory engine should remember durable user/project preferences, not dump
large code implementations into Qdrant. These helpers skip pure code-block
inputs and strip large fenced code blocks before asking the LLM to extract
facts.
"""

import re
from typing import Dict, List, Tuple

_CODE_BLOCK_RE = re.compile(r"```[\w.+#-]*\s*\n(?P<body>.*?)\n```", re.DOTALL)
_PURE_CODE_RE = re.compile(r"^```[\w.+#-]*\s*\n(?P<body>.*?)\n```$", re.DOTALL)

# Heuristic threshold: small snippets often appear inside natural language
# explanations and should not force a skip. Large standalone blocks are the
# pollution risk we're defending against.
MIN_CODE_LINES_FOR_SKIP = 8
MIN_CODE_CHARS_FOR_SKIP = 500


def _is_large_code_body(body: str) -> bool:
    lines = [line for line in body.splitlines() if line.strip()]
    return len(lines) >= MIN_CODE_LINES_FOR_SKIP or len(body.strip()) >= MIN_CODE_CHARS_FOR_SKIP


def is_pure_code_block(text: str) -> bool:
    """Return True when text is only one large fenced code block.

    Allows an optional language marker (```python / ```rust / ...). Small
    examples are not treated as pure-code pollution because users often paste
    tiny snippets together with durable preferences.
    """
    if not text:
        return False
    match = _PURE_CODE_RE.match(text.strip())
    if not match:
        return False
    return _is_large_code_body(match.group("body"))


def strip_large_code_blocks(text: str) -> Tuple[str, bool]:
    """Remove large fenced code blocks, preserving surrounding prose.

    Returns (cleaned_text, changed). Large blocks become a short placeholder so
    the LLM knows code was present without seeing implementation details.
    """
    changed = False

    def repl(match: re.Match) -> str:
        nonlocal changed
        body = match.group("body")
        if not _is_large_code_body(body):
            return match.group(0)
        changed = True
        return "[code block omitted]"

    cleaned = _CODE_BLOCK_RE.sub(repl, text)
    return cleaned.strip(), changed


def prepare_messages_for_fact_extraction(
    messages: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], bool, bool]:
    """Prepare messages before LLM fact extraction.

    Returns (prepared_messages, should_skip, stripped_code).

    Skip only when every non-empty user message is a pure large code block.
    Otherwise, strip large fenced blocks and keep natural language around them.
    """
    user_contents = [
        m.get("content", "") for m in messages
        if m.get("role") == "user" and m.get("content", "").strip()
    ]
    if user_contents and all(is_pure_code_block(content) for content in user_contents):
        return [], True, True

    prepared: List[Dict[str, str]] = []
    stripped_any = False
    for msg in messages:
        content = msg.get("content", "")
        cleaned, stripped = strip_large_code_blocks(content)
        stripped_any = stripped_any or stripped
        if cleaned.strip():
            prepared.append({
                "role": msg.get("role", "user"),
                "content": cleaned,
            })

    if stripped_any:
        has_user_text = any(
            m.get("role") == "user" and m.get("content", "").replace("[code block omitted]", "").strip()
            for m in prepared
        )
        if not has_user_text:
            return [], True, True

    return prepared, False, stripped_any
