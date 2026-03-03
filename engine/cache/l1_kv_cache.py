"""L1 Memory-level KV Cache: Zero-error, ultra-fast response.

Stores full-precision KV cache states in memory for exact hash matching.
Uses LRU eviction with configurable memory hard limit.
Supports incremental prefix matching for multi-turn conversations.
"""

import logging
import threading
from typing import Dict, List, Optional

from engine.cache.base import BaseCache, CacheEntry, CacheLevel
from engine.cache.eviction import LRUEvictionPolicy

logger = logging.getLogger(__name__)


class L1KVCache(BaseCache):
    """In-memory KV cache with exact hash matching and LRU eviction."""

    def __init__(self, max_memory_mb: int = 100, enabled: bool = True):
        super().__init__(CacheLevel.L1_KV_MEMORY, enabled)
        self._max_bytes = max_memory_mb * 1024 * 1024
        self._store: Dict[int, CacheEntry] = {}
        self._eviction = LRUEvictionPolicy(self._max_bytes)
        self._lock = threading.RLock()
        self._prefix_index: Dict[int, List[int]] = {}

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
            entry.touch()
            self._eviction.touch(context_hash, entry.estimated_size_bytes())
            self.record_hit()
            logger.debug(f"L1 命中: hash={context_hash}")
            return entry

    def get_prefix_match(
        self, tokens: List[int], model_hash: str
    ) -> Optional[tuple]:
        """Find the longest cached prefix of the given token sequence.

        Returns (CacheEntry, prefix_length) or None if no prefix found.
        """
        if not self.enabled:
            return None

        with self._lock:
            best_entry = None
            best_len = 0

            for ctx_hash, entry in self._store.items():
                if entry.model_hash != model_hash:
                    continue
                if entry.token_sequence is None:
                    continue
                cached_tokens = entry.token_sequence
                prefix_len = 0
                for i in range(min(len(cached_tokens), len(tokens))):
                    if cached_tokens[i] != tokens[i]:
                        break
                    prefix_len = i + 1
                if prefix_len > best_len and prefix_len < len(tokens):
                    best_len = prefix_len
                    best_entry = entry

            if best_entry is not None and best_len > 0:
                best_entry.touch()
                return best_entry, best_len
            return None

    def put(self, entry: CacheEntry) -> None:
        if not self.enabled:
            return

        with self._lock:
            size = entry.estimated_size_bytes()
            self._eviction.touch(entry.context_hash, size)

            evicted_keys = self._eviction.evict_if_needed()
            for key in evicted_keys:
                self._store.pop(key, None)
                if evicted_keys:
                    logger.debug(f"L1 淘汰 {len(evicted_keys)} 个条目")

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
            self._prefix_index.clear()
            logger.info("L1 缓存已清空")

    def size_bytes(self) -> int:
        return self._eviction.current_size

    def entry_count(self) -> int:
        return len(self._store)

    def iter_entries(self):
        """Iterate all entries (for persist to disk at session end)."""
        with self._lock:
            for ctx_hash, entry in list(self._store.items()):
                yield ctx_hash, entry

    def evict_stale(self, max_idle_seconds: int = 1800) -> int:
        """Evict entries not accessed within max_idle_seconds (default 30 min)."""
        import time
        evicted = 0
        now = time.time()
        with self._lock:
            stale_keys = [
                k for k, e in self._store.items()
                if now - e.last_accessed > max_idle_seconds
            ]
            for key in stale_keys:
                del self._store[key]
                self._eviction.remove(key)
                evicted += 1
        if evicted:
            logger.debug(f"L1 清理过期条目: {evicted} 个")
        return evicted
