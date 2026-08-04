"""Hybrid Search + Time Decay scoring for DeepMemory.

Combines four scoring signals:
1. Vector similarity (cosine distance from Qdrant)
2. BM25 keyword match (proper TF/IDF/length-norm via rank_bm25)
3. Entity boost (query↔memory shared named entities)
4. Time decay (exponential decay based on memory age)

The final score is a weighted fusion of all four components.
"""

import logging
import math
import re
from typing import Iterable, List, Optional, Sequence, Set

from rank_bm25 import BM25Okapi

logger = logging.getLogger("deepmem.hybrid_search")

# ── Time Decay ──────────────────────────────────────────────────────────────

# Half-life: memories lose half their freshness weight after 30 days
TIME_DECAY_HALF_LIFE_SECONDS = 30 * 24 * 3600  # 30 days in seconds


def compute_time_decay(created_at: float, now: float) -> float:
    """Compute time decay factor using exponential decay.

    Returns a value in [0, 1] where:
    - 1.0 = just created (full freshness)
    - 0.5 = 30 days old (half freshness)
    - ~0.0 = very old (no freshness bonus)

    The decay is gentle — old memories are still retrievable but rank
    lower than recent ones with similar relevance.
    """
    if created_at <= 0 or now <= 0:
        return 1.0  # No timestamp → assume fresh
    age_seconds = max(0, now - created_at)
    decay = math.exp(-math.log(2) * age_seconds / TIME_DECAY_HALF_LIFE_SECONDS)
    return float(decay)


# ── Tokenization & Stop-Words ──────────────────────────────────────────────

# Chat-scaffolding words that appear in nearly every memory; without
# filtering them, a query about the user will rank a memory containing just
# "the user is..." first purely on the "user" overlap.
_STOP: Set[str] = {
    "用户", "user", "users", "the", "a", "an", "is", "are", "was", "were",
    "what", "where", "who", "when", "which", "how", "why",
    "of", "in", "on", "at", "to", "for", "with", "and", "or",
    "什么", "哪里", "哪个", "怎么", "为什么", "几", "多少",
    "是", "的", "了", "在", "和", "或", "对",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, drop stop-words and 1-char tokens.
    No lemmatization — that requires spaCy and is the next-phase upgrade."""
    if not text:
        return []
    raw = _TOKEN_RE.findall(text.lower())
    return [t for t in raw if t not in _STOP and len(t) > 1]


# ── BM25 Keyword Scoring ────────────────────────────────────────────────────


def compute_bm25_scores(query: str, memory_texts: Sequence[str]) -> List[float]:
    """Real BM25 (rank_bm25 BM25Okapi) over the candidate set.

    The candidate set is the over-fetched top-K from the vector search, so
    IDF is computed against that local corpus rather than the full store.
    Scores are min-max normalized to [0, 1] for fusion with vector score.
    """
    if not query or not memory_texts:
        return [0.0] * len(memory_texts)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0] * len(memory_texts)

    tokenized_corpus = [_tokenize(t) or [""] for t in memory_texts]
    try:
        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = bm25.get_scores(query_tokens)
    except Exception as e:
        logger.warning("BM25 scoring failed: %s — falling back to zero", e)
        return [0.0] * len(memory_texts)

    raw_list = [float(s) for s in raw_scores]
    max_score = max(raw_list) if raw_list else 0.0
    if max_score <= 0:
        return [0.0] * len(memory_texts)

    normalized = [s / max_score for s in raw_list]

    # Exact phrase match: nudge that document up so a memory containing the
    # whole query string outranks one that just shares words.
    q_lower = query.lower()
    for i, mem in enumerate(memory_texts):
        if mem and q_lower in mem.lower():
            normalized[i] = min(1.0, normalized[i] + 0.15)
    return normalized


def compute_keyword_score(query: str, memory_text: str) -> float:
    """Single-document BM25-like score (kept for backward compatibility).

    For multi-doc scoring use compute_bm25_scores — IDF is meaningful only
    over a corpus. This scalar form computes BM25 with a 1-doc corpus,
    which collapses to TF * length-norm; mostly useful for unit tests.
    """
    if not query or not memory_text:
        return 0.0
    scores = compute_bm25_scores(query, [memory_text])
    return scores[0] if scores else 0.0


# ── Entity Boost ─────────────────────────────────────────────────────────────

# Lightweight entity patterns. Replaced by spaCy NER in a later phase, but the
# query-aware matching logic below is what actually moves the needle.
_ENTITY_PATTERNS = {
    "person": re.compile(
        r'\b(?:Alice|Bob|Charlie|David|Eve|Frank|Grace|Henry|Ivy|Jack|Kate|'
        r'Leo|Mia|Noah|Olivia|Peter|Quinn|Rose|Sam|Tom|Uma|Victor|Wendy|'
        r'Xavier|Yara|Zoe|John|Jane|Mary|James|Sarah|Michael|Emma|Daniel|'
        r'Sophia|William|Isabella|Alexander|Emily|Benjamin|Ava)\b'
    ),
    "organization": re.compile(
        r'\b(?:Google|Apple|Microsoft|Amazon|Meta|Netflix|Tesla|OpenAI|'
        r'Stripe|ByteDance|Tencent|Alibaba|Baidu|DeepSeek|Anthropic|'
        r'MIT|Stanford|Harvard|Berkeley)\b'
    ),
    "location": re.compile(
        r'\b(?:San Francisco|New York|London|Beijing|Shanghai|Tokyo|'
        r'Paris|Berlin|Singapore|Seattle|Los Angeles|Chicago|Boston|'
        r'Shenzhen|Hangzhou|Mountain View|Palo Alto|Cambridge)\b'
    ),
}


def extract_entities(text: str) -> Set[str]:
    """Return the set of recognized entities (lowercased) found in text."""
    if not text:
        return set()
    found: Set[str] = set()
    for pattern in _ENTITY_PATTERNS.values():
        for m in pattern.findall(text):
            found.add(m.lower())
    return found


def compute_entity_boost(query, memory_text: Optional[str] = None) -> float:
    """Compute entity boost using query↔memory entity overlap.

    Two call shapes are supported:

    - compute_entity_boost(query: str, memory_text: str) — computes overlap
      between entities in the query and entities in this memory. Memories
      that don't share any entity with the query get no boost.
    - compute_entity_boost(memory_text: str)  *(legacy)* — falls back to
      counting memory-only entities; kept so existing call sites keep working
      while the search path is migrated.

    Returns a multiplier in [1.0, 1.5].
    """
    # Two-arg query-aware shape
    if memory_text is not None:
        query_entities = extract_entities(query) if isinstance(query, str) else set()
        mem_entities = extract_entities(memory_text)
        if not query_entities or not mem_entities:
            return 1.0
        overlap = len(query_entities & mem_entities)
        if overlap == 0:
            return 1.0
        # Coverage: how much of the query's entity set is matched
        coverage = overlap / len(query_entities)
        boost = 1.0 + coverage * 0.5  # full coverage → 1.5x, partial scales linearly
        return min(boost, 1.5)

    # One-arg legacy shape: query parameter actually carries the memory text
    legacy_text = query if isinstance(query, str) else ""
    if not legacy_text:
        return 1.0
    entity_count = sum(len(p.findall(legacy_text)) for p in _ENTITY_PATTERNS.values())
    if entity_count == 0:
        return 1.0
    boost = 1.0 + math.log(1 + entity_count) * 0.1
    return min(boost, 1.5)


# ── Score Fusion ─────────────────────────────────────────────────────────────

# Weight configuration for each scoring component
WEIGHTS = {
    "vector": 0.50,     # Vector similarity is the primary signal
    "keyword": 0.25,    # Keyword match is secondary
    "entity": 0.10,     # Entity boost is tertiary
    "time": 0.15,       # Time decay is a freshness bonus
}


def fuse_scores(vector_score: float,
                keyword_score: float = 0.0,
                entity_boost: float = 1.0,
                time_decay: float = 1.0) -> float:
    """Fuse multiple scoring signals into a single relevance score.

    Args:
        vector_score: Cosine similarity from vector search [0, 1]
        keyword_score: BM25 keyword overlap score [0, 1]
        entity_boost: Entity multiplier [1.0, 1.5]
        time_decay: Time freshness factor [0, 1]

    Returns:
        Fused score [0, 1], higher is more relevant.
    """
    raw = (
        WEIGHTS["vector"] * vector_score +
        WEIGHTS["keyword"] * keyword_score +
        WEIGHTS["entity"] * entity_boost * vector_score +  # Entity amplifies vector
        WEIGHTS["time"] * time_decay * vector_score        # Time amplifies vector
    )
    # Clamp to [0, 1]
    return max(0.0, min(1.0, raw))
