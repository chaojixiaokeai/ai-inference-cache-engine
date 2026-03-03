"""Base cache interfaces and shared data structures."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CacheLevel(Enum):
    L1_KV_MEMORY = "L1"
    L2_LOGITS_MEMORY = "L2"
    L3_SEMANTIC_DISK = "L3"
    L4_PERSISTENT_DISK = "L4"


class CacheHitType(Enum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    MISS = "miss"


@dataclass
class CacheEntry:
    """Universal cache entry stored across all cache levels."""
    context_hash: int
    model_hash: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    greedy_token: int = -1
    top_k_indices: Optional[np.ndarray] = None
    top_k_values: Optional[np.ndarray] = None
    kv_state: Optional[bytes] = None
    response_text: Optional[str] = None
    token_sequence: Optional[List[int]] = None
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 1.0

    def touch(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

    def estimated_size_bytes(self) -> int:
        """Estimate memory footprint of this entry."""
        size = 128  # base object overhead
        if self.kv_state is not None:
            size += len(self.kv_state)
        if self.top_k_indices is not None:
            size += self.top_k_indices.nbytes
        if self.top_k_values is not None:
            size += self.top_k_values.nbytes
        if self.response_text is not None:
            size += len(self.response_text.encode("utf-8"))
        if self.embedding is not None:
            size += self.embedding.nbytes
        if self.token_sequence is not None:
            size += len(self.token_sequence) * 4
        return size


@dataclass
class CacheResult:
    """Result from a cache lookup."""
    hit: bool
    level: CacheLevel
    hit_type: CacheHitType
    entry: Optional[CacheEntry] = None


class BaseCache(ABC):
    """Abstract base class for all cache levels."""

    def __init__(self, level: CacheLevel, enabled: bool = True):
        self.level = level
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

    @abstractmethod
    def get(self, context_hash: int, model_hash: str) -> Optional[CacheEntry]:
        """Look up a cache entry by context hash."""
        ...

    @abstractmethod
    def put(self, entry: CacheEntry) -> None:
        """Store a cache entry."""
        ...

    @abstractmethod
    def delete(self, context_hash: int) -> bool:
        """Delete a specific cache entry."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries in this cache level."""
        ...

    @abstractmethod
    def size_bytes(self) -> int:
        """Current size of cached data in bytes."""
        ...

    @abstractmethod
    def entry_count(self) -> int:
        """Number of entries in cache."""
        ...

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def record_hit(self):
        self._hits += 1

    def record_miss(self):
        self._misses += 1

    @property
    def stats(self) -> Dict:
        return {
            "level": self.level.value,
            "enabled": self.enabled,
            "entries": self.entry_count(),
            "size_bytes": self.size_bytes(),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }
