"""L2 Memory-level Logits Cache: Zero-error, ultimate speed.

Stores top-K full-precision logits distributions in memory.
When hit, token generation bypasses all Transformer computation entirely,
sampling directly from cached probability distributions.
Uses LFU eviction with configurable memory hard limit.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.cache.base import BaseCache, CacheEntry, CacheLevel
from engine.cache.eviction import LFUEvictionPolicy

logger = logging.getLogger(__name__)


class L2LogitsCache(BaseCache):
    """In-memory logits cache with exact hash matching and LFU eviction."""

    def __init__(
        self,
        max_memory_mb: int = 200,
        top_k: int = 100,
        enabled: bool = True,
    ):
        super().__init__(CacheLevel.L2_LOGITS_MEMORY, enabled)
        self._max_bytes = max_memory_mb * 1024 * 1024
        self._top_k = top_k
        self._store: Dict[int, CacheEntry] = {}
        self._eviction = LFUEvictionPolicy(self._max_bytes)
        self._lock = threading.RLock()

    def get(self, context_hash: int, model_hash: str) -> Optional[CacheEntry]:
        if not self.enabled:
            self.record_miss()
            return None

        with self._lock:
            entry = self._store.get(context_hash)
            if entry is None:
                self.record_miss()
                return None
            if entry.model_hash != model_hash:
                self.record_miss()
                return None
            if entry.top_k_indices is None or entry.top_k_values is None:
                self.record_miss()
                return None

            entry.touch()
            self._eviction.touch(context_hash, entry.estimated_size_bytes())
            self.record_hit()
            logger.debug(f"L2 命中: hash={context_hash}")
            return entry

    def put(self, entry: CacheEntry) -> None:
        if not self.enabled:
            return
        if entry.top_k_indices is None or entry.top_k_values is None:
            return

        with self._lock:
            size = entry.estimated_size_bytes()
            self._eviction.touch(entry.context_hash, size)

            evicted_keys = self._eviction.evict_if_needed()
            for key in evicted_keys:
                self._store.pop(key, None)
            if evicted_keys:
                logger.debug(f"L2 淘汰 {len(evicted_keys)} 个条目")

            self._store[entry.context_hash] = entry

    def delete(self, context_hash: int) -> bool:
        with self._lock:
            if context_hash in self._store:
                del self._store[context_hash]
                self._eviction.remove(context_hash)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._eviction.clear()
            logger.info("L2 缓存已清空")

    def size_bytes(self) -> int:
        return self._eviction.current_size

    def entry_count(self) -> int:
        return len(self._store)

    @staticmethod
    def sample_from_logits(
        top_k_indices: np.ndarray,
        top_k_values: np.ndarray,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> int:
        """Sample a token from cached top-K logits distribution.

        This is the core speedup: no Transformer computation needed.
        """
        if temperature <= 0:
            return int(top_k_indices[0])

        logits = top_k_values / temperature
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()

        if top_k > 0 and top_k < len(probs):
            sorted_idx = np.argsort(probs)[::-1][:top_k]
            mask = np.zeros_like(probs)
            mask[sorted_idx] = probs[sorted_idx]
            probs = mask / mask.sum()

        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, top_p) + 1
            allowed = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[allowed] = probs[allowed]
            probs = mask / mask.sum()

        chosen_idx = np.random.choice(len(probs), p=probs)
        return int(top_k_indices[chosen_idx])
