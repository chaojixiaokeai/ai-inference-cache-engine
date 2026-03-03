"""Long context optimization: chunked caching, incremental tokens, topic anchoring.

Handles 32k/128k contexts efficiently by splitting into semantic chunks,
caching each independently, and reusing cached prefixes for multi-turn conversations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.cache.base import CacheEntry
from engine.utils.hashing import hash_token_sequence

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 2048


class ChunkedCacheManager:
    """Manages chunked caching for long context sequences."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self._chunk_size = chunk_size
        self._chunk_cache: Dict[int, CacheEntry] = {}

    def split_into_chunks(self, tokens: List[int]) -> List[List[int]]:
        """Split a long token sequence into fixed-size chunks."""
        chunks = []
        for i in range(0, len(tokens), self._chunk_size):
            chunks.append(tokens[i : i + self._chunk_size])
        return chunks

    def get_cached_prefix_length(
        self, tokens: List[int], model_hash: str
    ) -> int:
        """Find how many prefix tokens are already cached as chunks.

        Returns the number of tokens that can be skipped (already cached).
        """
        chunks = self.split_into_chunks(tokens)
        cached_length = 0

        for chunk in chunks:
            chunk_hash = hash_token_sequence(chunk)
            entry = self._chunk_cache.get(chunk_hash)
            if entry is None or entry.model_hash != model_hash:
                break
            cached_length += len(chunk)

        return cached_length

    def cache_chunk(self, chunk_tokens: List[int], entry: CacheEntry) -> None:
        """Cache a single chunk."""
        chunk_hash = hash_token_sequence(chunk_tokens)
        entry.context_hash = chunk_hash
        self._chunk_cache[chunk_hash] = entry

    def get_chunk(self, chunk_tokens: List[int], model_hash: str) -> Optional[CacheEntry]:
        chunk_hash = hash_token_sequence(chunk_tokens)
        entry = self._chunk_cache.get(chunk_hash)
        if entry and entry.model_hash == model_hash:
            entry.touch()
            return entry
        return None

    def clear(self):
        self._chunk_cache.clear()

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_cache)


class IncrementalTokenCache:
    """Tracks incremental token additions across multi-turn conversations.

    For multi-turn dialogs, only new tokens need cache computation;
    the existing prefix KV cache is fully reused.
    """

    def __init__(self):
        self._last_tokens: List[int] = []
        self._last_hash: int = 0

    def compute_delta(self, current_tokens: List[int]) -> Tuple[int, List[int]]:
        """Compute the incremental delta between last and current tokens.

        Returns:
            (prefix_length, new_tokens) where prefix_length tokens can be
            reused from the previous KV cache.
        """
        prefix_len = 0
        max_check = min(len(self._last_tokens), len(current_tokens))

        for i in range(max_check):
            if self._last_tokens[i] != current_tokens[i]:
                break
            prefix_len = i + 1

        new_tokens = current_tokens[prefix_len:]
        return prefix_len, new_tokens

    def update(self, tokens: List[int]):
        """Update the reference token sequence after generation."""
        self._last_tokens = list(tokens)
        self._last_hash = hash_token_sequence(tokens) if tokens else 0


class TopicAnchor:
    """Anchors long documents by their core topic embedding.

    When the topic of a document doesn't change, the global KV cache
    for the document body can be fully reused across questions.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self._threshold = similarity_threshold
        self._anchors: Dict[int, np.ndarray] = {}

    def set_anchor(self, doc_hash: int, embedding: np.ndarray):
        """Store a topic anchor for a document."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self._anchors[doc_hash] = embedding

    def check_anchor(self, doc_hash: int, current_embedding: np.ndarray) -> bool:
        """Check if current topic matches the anchored topic.

        Returns True if the topic hasn't changed significantly.
        """
        if doc_hash not in self._anchors:
            return False

        anchor = self._anchors[doc_hash]
        norm = np.linalg.norm(current_embedding)
        if norm > 0:
            current_embedding = current_embedding / norm

        similarity = float(np.dot(anchor, current_embedding))
        return similarity >= self._threshold

    def clear(self):
        self._anchors.clear()
