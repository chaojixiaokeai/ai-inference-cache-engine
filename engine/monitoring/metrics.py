"""Real-time metrics collection and reporting."""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricsSnapshot:
    timestamp: float = field(default_factory=time.time)
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_generated: int = 0
    avg_tokens_per_second: float = 0.0
    memory_rss_mb: float = 0.0
    cpu_percent: float = 0.0


class MetricsCollector:
    """Collects and aggregates engine performance metrics."""

    def __init__(self):
        self._request_count = 0
        self._cache_hit_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0
        self._lock = threading.Lock()
        self._history: List[MetricsSnapshot] = []

    def record_request(self, cache_hit: bool, tokens: int, latency_ms: float):
        with self._lock:
            self._request_count += 1
            if cache_hit:
                self._cache_hit_count += 1
            self._total_tokens += tokens
            self._total_latency_ms += latency_ms

    @property
    def hit_rate(self) -> float:
        if self._request_count == 0:
            return 0.0
        return self._cache_hit_count / self._request_count

    @property
    def avg_speed(self) -> float:
        if self._total_latency_ms == 0:
            return 0.0
        return self._total_tokens / (self._total_latency_ms / 1000)

    def get_summary(self) -> Dict:
        return {
            "total_requests": self._request_count,
            "cache_hits": self._cache_hit_count,
            "hit_rate": round(self.hit_rate, 4),
            "total_tokens": self._total_tokens,
            "avg_tokens_per_second": round(self.avg_speed, 1),
        }
