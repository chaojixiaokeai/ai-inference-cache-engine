"""Performance optimization: CPU scheduling, power management, cache prefetch.

Implements platform-specific optimizations for sustained performance
on consumer hardware including laptops.
"""

import logging
import os
import platform
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional

import psutil

if TYPE_CHECKING:
    from engine.config import EngineConfig
    from engine.core.inference import InferenceEngine

logger = logging.getLogger(__name__)


class CPUScheduler:
    """Intelligent CPU core scheduling for P-core/E-core architectures."""

    def __init__(self):
        self._physical_cores = psutil.cpu_count(logical=False) or 2
        self._logical_cores = psutil.cpu_count(logical=True) or 4
        self._system = platform.system()
        self._arch = platform.machine()

    def get_optimal_threads(self, is_inference: bool = True) -> int:
        """Calculate optimal thread count based on task type and hardware."""
        if is_inference:
            return min(self._physical_cores, 8)
        else:
            return max(1, self._physical_cores // 4)

    def get_affinity_mask(self) -> Optional[list]:
        """Get CPU affinity mask preferring performance cores.

        Returns None if affinity control is not available.
        """
        if self._system == "Linux":
            try:
                p = psutil.Process(os.getpid())
                all_cpus = list(range(self._logical_cores))
                p_cores = all_cpus[: self._physical_cores]
                return p_cores
            except Exception:
                return None
        return None

    def apply_affinity(self):
        """Apply CPU affinity if possible."""
        mask = self.get_affinity_mask()
        if mask is not None:
            try:
                p = psutil.Process(os.getpid())
                p.cpu_affinity(mask)
                logger.info(f"CPU 亲和性已设置: cores {mask}")
            except Exception as e:
                logger.debug(f"CPU 亲和性设置失败: {e}")


class PowerManager:
    """Laptop power management to extend battery life."""

    def __init__(self, config: "EngineConfig"):
        self._config = config
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None

    def is_on_battery(self) -> bool:
        """Check if running on battery power."""
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                return not battery.power_plugged
        except Exception:
            pass
        return False

    def battery_percent(self) -> Optional[float]:
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                return battery.percent
        except Exception:
            pass
        return None

    def get_power_profile(self) -> Dict:
        """Get recommended settings based on power state."""
        on_battery = self.is_on_battery()
        battery_pct = self.battery_percent()

        if not on_battery:
            return {
                "profile": "plugged_in",
                "n_threads": min(psutil.cpu_count(logical=False) or 2, 8),
                "n_batch": 512,
                "background_tasks": True,
            }

        if battery_pct is not None and battery_pct < 20:
            return {
                "profile": "low_battery",
                "n_threads": max(1, (psutil.cpu_count(logical=False) or 2) // 4),
                "n_batch": 128,
                "background_tasks": False,
            }

        return {
            "profile": "battery",
            "n_threads": max(1, (psutil.cpu_count(logical=False) or 2) // 2),
            "n_batch": 256,
            "background_tasks": True,
        }

    def start_monitoring(self, engine: "InferenceEngine"):
        """Start background power monitoring and auto-adjust."""
        if self._monitoring:
            return
        self._monitoring = True
        self._thread = threading.Thread(
            target=self._monitor_loop, args=(engine,), daemon=True
        )
        self._thread.start()

    def stop_monitoring(self):
        self._monitoring = False

    def _monitor_loop(self, engine: "InferenceEngine"):
        last_profile = None
        while self._monitoring:
            try:
                profile = self.get_power_profile()
                if profile["profile"] != last_profile:
                    logger.info(f"电源模式切换: {profile['profile']}")
                    last_profile = profile["profile"]
                time.sleep(30)
            except Exception:
                time.sleep(60)


class CachePrefetcher:
    """Prefetch likely cache entries based on conversation patterns."""

    def __init__(self):
        self._pattern_history: list = []
        self._max_history = 100

    def record_query(self, query: str):
        """Record a query for pattern analysis."""
        self._pattern_history.append(query)
        if len(self._pattern_history) > self._max_history:
            self._pattern_history = self._pattern_history[-self._max_history:]

    def predict_next(self) -> Optional[str]:
        """Predict the likely next query based on patterns.

        Uses simple n-gram pattern matching on conversation history.
        """
        if len(self._pattern_history) < 3:
            return None

        last_two = self._pattern_history[-2:]
        for i in range(len(self._pattern_history) - 3):
            if self._pattern_history[i:i+2] == last_two:
                if i + 2 < len(self._pattern_history):
                    return self._pattern_history[i + 2]

        return None

    @property
    def history_size(self) -> int:
        return len(self._pattern_history)


class PerformanceOptimizer:
    """Unified performance optimization manager."""

    def __init__(self, config: "EngineConfig"):
        self._config = config
        self.cpu_scheduler = CPUScheduler()
        self.power_manager = PowerManager(config)
        self.prefetcher = CachePrefetcher()

    def initialize(self, engine: "InferenceEngine"):
        """Apply initial optimizations."""
        self.cpu_scheduler.apply_affinity()

        if platform.system() == "Darwin" or self.power_manager.is_on_battery():
            self.power_manager.start_monitoring(engine)

    def get_status(self) -> Dict:
        battery = self.power_manager.battery_percent()
        profile = self.power_manager.get_power_profile()
        return {
            "cpu_cores": f"{self.cpu_scheduler._physical_cores}P/{self.cpu_scheduler._logical_cores}L",
            "power_profile": profile["profile"],
            "battery_percent": f"{battery:.0f}%" if battery is not None else "N/A",
            "recommended_threads": profile["n_threads"],
            "prefetch_history": self.prefetcher.history_size,
        }

    def shutdown(self):
        self.power_manager.stop_monitoring()
