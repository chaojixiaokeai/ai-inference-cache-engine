"""Eviction strategy implementations: LRU, LFU, TTL+LRU."""

import time
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple


class LRUEvictionPolicy:
    """Least Recently Used eviction with memory-based hard limit."""

    def __init__(self, max_size_bytes: int):
        self._max_size = max_size_bytes
        self._order: OrderedDict[int, int] = OrderedDict()  # key -> size_bytes
        self._current_size = 0
        self._lock = threading.RLock()

    @property
    def current_size(self) -> int:
        return self._current_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def utilization(self) -> float:
        return self._current_size / self._max_size if self._max_size > 0 else 0

    def touch(self, key: int, size_bytes: int) -> None:
        with self._lock:
            if key in self._order:
                self._current_size -= self._order[key]
                self._order.move_to_end(key)
            self._order[key] = size_bytes
            self._current_size += size_bytes

    def remove(self, key: int) -> None:
        with self._lock:
            if key in self._order:
                self._current_size -= self._order.pop(key)

    def evict_if_needed(self) -> List[int]:
        """Evict least recently used entries until under limit. Returns evicted keys."""
        evicted = []
        with self._lock:
            while self._current_size > self._max_size and self._order:
                key, size = self._order.popitem(last=False)
                self._current_size -= size
                evicted.append(key)
        return evicted

    def should_evict(self) -> bool:
        return self._current_size > self._max_size * 0.8

    def clear(self) -> None:
        with self._lock:
            self._order.clear()
            self._current_size = 0


class LFUEvictionPolicy:
    """Least Frequently Used eviction."""

    def __init__(self, max_size_bytes: int):
        self._max_size = max_size_bytes
        self._freq: Dict[int, int] = {}    # key -> access_count
        self._sizes: Dict[int, int] = {}   # key -> size_bytes
        self._current_size = 0
        self._lock = threading.RLock()

    @property
    def current_size(self) -> int:
        return self._current_size

    def touch(self, key: int, size_bytes: int) -> None:
        with self._lock:
            if key in self._sizes:
                self._current_size -= self._sizes[key]
                self._freq[key] = self._freq.get(key, 0) + 1
            else:
                self._freq[key] = 1
            self._sizes[key] = size_bytes
            self._current_size += size_bytes

    def remove(self, key: int) -> None:
        with self._lock:
            if key in self._sizes:
                self._current_size -= self._sizes.pop(key)
                self._freq.pop(key, None)

    def evict_if_needed(self) -> List[int]:
        evicted = []
        with self._lock:
            while self._current_size > self._max_size and self._freq:
                min_key = min(self._freq, key=self._freq.get)
                self._current_size -= self._sizes.pop(min_key)
                self._freq.pop(min_key)
                evicted.append(min_key)
        return evicted

    def clear(self) -> None:
        with self._lock:
            self._freq.clear()
            self._sizes.clear()
            self._current_size = 0


class TTLLRUEvictionPolicy:
    """Combined TTL + LRU eviction for persistent cache."""

    def __init__(self, max_size_bytes: int, ttl_seconds: int):
        self._max_size = max_size_bytes
        self._ttl = ttl_seconds
        self._entries: OrderedDict[int, Tuple[int, float]] = OrderedDict()  # key -> (size, timestamp)
        self._current_size = 0
        self._lock = threading.RLock()

    @property
    def current_size(self) -> int:
        return self._current_size

    def touch(self, key: int, size_bytes: int) -> None:
        with self._lock:
            if key in self._entries:
                old_size, _ = self._entries[key]
                self._current_size -= old_size
                self._entries.move_to_end(key)
            self._entries[key] = (size_bytes, time.time())
            self._current_size += size_bytes

    def remove(self, key: int) -> None:
        with self._lock:
            if key in self._entries:
                size, _ = self._entries.pop(key)
                self._current_size -= size

    def evict_expired(self) -> List[int]:
        """Remove entries that have exceeded their TTL."""
        evicted = []
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, (_, ts) in self._entries.items()
                if now - ts > self._ttl
            ]
            for key in expired_keys:
                size, _ = self._entries.pop(key)
                self._current_size -= size
                evicted.append(key)
        return evicted

    def evict_if_needed(self) -> List[int]:
        evicted = self.evict_expired()
        with self._lock:
            while self._current_size > self._max_size and self._entries:
                key, (size, _) = self._entries.popitem(last=False)
                self._current_size -= size
                evicted.append(key)
        return evicted

    def should_evict(self) -> bool:
        return self._current_size > self._max_size * 0.8

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._current_size = 0
