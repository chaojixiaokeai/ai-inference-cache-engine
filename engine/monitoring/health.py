"""Health check and diagnostic utilities."""

import logging
import os
from typing import Dict, List

from engine.utils.hardware import detect_hardware
from engine.utils.storage import get_dir_size_mb, get_free_disk_gb

logger = logging.getLogger(__name__)


def run_health_check(config, engine=None, scheduler=None) -> Dict:
    """Run a comprehensive health check and return report."""
    results = {"status": "healthy", "checks": [], "warnings": []}

    # Hardware check
    hw = detect_hardware(config.cache.base_dir)
    results["checks"].append({
        "name": "硬件检测",
        "status": "pass",
        "detail": f"CPU {hw.cpu_count_physical}核, 内存 {hw.available_memory_mb}MB 可用",
    })

    if hw.available_memory_mb < 2048:
        results["warnings"].append("可用内存不足2GB，推理可能受限")

    # Disk check
    cache_dir = os.path.expanduser(config.cache.base_dir)
    free_gb = get_free_disk_gb(cache_dir)
    if free_gb < 5:
        results["warnings"].append(f"磁盘空间不足: 仅剩 {free_gb:.1f}GB")
        results["status"] = "warning"

    results["checks"].append({
        "name": "磁盘空间",
        "status": "pass" if free_gb >= 5 else "warning",
        "detail": f"空闲 {free_gb:.1f}GB",
    })

    # Cache check
    if os.path.exists(cache_dir):
        cache_size = get_dir_size_mb(cache_dir)
        results["checks"].append({
            "name": "缓存大小",
            "status": "pass",
            "detail": f"{cache_size:.1f}MB",
        })

    # Model check
    if engine and engine.is_loaded:
        results["checks"].append({
            "name": "模型状态",
            "status": "pass",
            "detail": f"已加载, hash={engine.model_hash[:16]}",
        })
    else:
        results["checks"].append({
            "name": "模型状态",
            "status": "info",
            "detail": "未加载",
        })

    # Scheduler check
    if scheduler:
        stats = scheduler.stats
        results["checks"].append({
            "name": "缓存调度",
            "status": "pass",
            "detail": f"L1={stats['l1']['entries']}条, L4={stats['l4']['entries']}条",
        })

    return results
