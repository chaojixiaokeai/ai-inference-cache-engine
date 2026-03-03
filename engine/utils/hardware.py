"""Hardware detection and resource monitoring utilities."""

import os
import platform
import shutil
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class HardwareInfo:
    cpu_count_physical: int
    cpu_count_logical: int
    total_memory_mb: int
    available_memory_mb: int
    disk_total_gb: float
    disk_free_gb: float
    os_name: str
    arch: str

    def optimal_thread_count(self, max_threads: int = 8) -> int:
        return min(self.cpu_count_physical // 2 or 1, max_threads)

    def optimal_ctx_size(self, max_memory_mb: int = 2048) -> int:
        safe_memory = min(self.available_memory_mb * 0.6, max_memory_mb)
        ctx = min(int(safe_memory / 2) * 512, 131072)
        return max(ctx, 512)


def detect_hardware(cache_dir: str = "~/.cache/ai_engine") -> HardwareInfo:
    """Detect current hardware capabilities."""
    mem = psutil.virtual_memory()
    cache_path = os.path.expanduser(cache_dir)
    if os.path.exists(cache_path):
        disk = shutil.disk_usage(cache_path)
    else:
        disk = shutil.disk_usage(os.path.expanduser("~"))

    return HardwareInfo(
        cpu_count_physical=psutil.cpu_count(logical=False) or 2,
        cpu_count_logical=psutil.cpu_count(logical=True) or 4,
        total_memory_mb=int(mem.total / (1024 * 1024)),
        available_memory_mb=int(mem.available / (1024 * 1024)),
        disk_total_gb=disk.total / (1024 ** 3),
        disk_free_gb=disk.free / (1024 ** 3),
        os_name=platform.system(),
        arch=platform.machine(),
    )


class ResourceMonitor:
    """Monitor process resource usage in real-time."""

    def __init__(self):
        self._process = psutil.Process(os.getpid())

    @property
    def memory_rss_mb(self) -> float:
        return self._process.memory_info().rss / (1024 * 1024)

    @property
    def memory_vms_mb(self) -> float:
        return self._process.memory_info().vms / (1024 * 1024)

    @property
    def cpu_percent(self) -> float:
        return self._process.cpu_percent(interval=0.1)

    def is_memory_critical(self, limit_mb: int = 2048) -> bool:
        return self.memory_rss_mb > limit_mb * 0.9

    def get_summary(self) -> dict:
        return {
            "memory_rss_mb": round(self.memory_rss_mb, 1),
            "memory_vms_mb": round(self.memory_vms_mb, 1),
            "cpu_percent": round(self.cpu_percent, 1),
        }
