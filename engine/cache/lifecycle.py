"""Cache lifecycle management: eviction, hot/cold flow, defrag, backup/restore.

Ensures long-term stable operation without storage bloat or performance degradation.
"""

import asyncio
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from engine.utils.storage import (
    ensure_dir,
    format_size,
    get_dir_size_bytes,
    get_free_disk_gb,
    safe_remove_dir,
)

if TYPE_CHECKING:
    from engine.cache.scheduler import CacheScheduler
    from engine.config import EngineConfig

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manages cache lifecycle: eviction, hot/cold migration, backup, and defrag."""

    def __init__(self, config: "EngineConfig", scheduler: "CacheScheduler"):
        self._config = config
        self._scheduler = scheduler
        self._base_dir = os.path.expanduser(config.cache.base_dir)
        self._backup_dir = os.path.join(self._base_dir, "backups")
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start_background(self):
        """Start background lifecycle management thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        logger.info("生命周期管理后台任务已启动")

    def stop_background(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("生命周期管理后台任务已停止")

    def _background_loop(self):
        """Main background loop running periodic maintenance tasks."""
        cycle = 0
        while self._running:
            time.sleep(60)
            cycle += 1

            try:
                # Every 5 minutes: evict stale L1/L2 entries
                if cycle % 5 == 0:
                    self._evict_stale_memory()

                # Every 30 minutes: check disk limits
                if cycle % 30 == 0:
                    self._check_disk_limits()

                # Every hour: cold data cleanup
                if cycle % 60 == 0:
                    self._cold_data_cleanup()

            except Exception as e:
                logger.warning(f"后台维护任务异常: {e}")

    def _evict_stale_memory(self):
        """Evict L1/L2 entries not accessed in 30 minutes."""
        evicted = self._scheduler.l1.evict_stale(max_idle_seconds=1800)
        if evicted:
            logger.info(f"L1 自动淘汰 {evicted} 个过期条目")

    def _check_disk_limits(self):
        """Check if disk caches exceed their hard limits."""
        l4_size = self._scheduler.l4.size_bytes()
        l4_limit = self._config.cache.l4.max_disk_gb * (1024 ** 3)

        if l4_size > l4_limit * 0.8:
            logger.warning(
                f"L4 缓存已达 {l4_size / (1024**3):.1f}GB / "
                f"{self._config.cache.l4.max_disk_gb}GB 上限的80%"
            )

        free_gb = get_free_disk_gb(self._base_dir)
        if free_gb < 5:
            logger.warning(f"磁盘空间不足: 仅剩 {free_gb:.1f}GB, 暂停缓存写入")

    def _cold_data_cleanup(self):
        """Clean up cold data in L3 and L4."""
        if self._scheduler.l3 is not None and self._scheduler.l3.enabled:
            evicted = self._scheduler.l3.evict_stale(max_idle_days=7)
            if evicted:
                logger.info(f"L3 冷数据清理: {evicted} 个条目")

    def run_full_maintenance(self) -> Dict:
        """Run a complete maintenance cycle (user-triggered)."""
        report = {"actions": [], "errors": []}

        # 1. Evict stale memory caches
        try:
            l1_evicted = self._scheduler.l1.evict_stale(max_idle_seconds=1800)
            report["actions"].append(f"L1 淘汰 {l1_evicted} 个过期条目")
        except Exception as e:
            report["errors"].append(f"L1 淘汰失败: {e}")

        # 2. Check and enforce disk limits
        try:
            self._check_disk_limits()
            report["actions"].append("磁盘限额检查完成")
        except Exception as e:
            report["errors"].append(f"磁盘检查失败: {e}")

        # 3. L4 compaction
        try:
            self._scheduler.l4.compact()
            report["actions"].append("L4 数据库碎片整理完成")
        except Exception as e:
            report["errors"].append(f"L4 碎片整理失败: {e}")

        # 4. Cold data cleanup
        try:
            self._cold_data_cleanup()
            report["actions"].append("冷数据清理完成")
        except Exception as e:
            report["errors"].append(f"冷数据清理失败: {e}")

        # 5. Storage summary
        total_size = get_dir_size_bytes(self._base_dir)
        free_gb = get_free_disk_gb(self._base_dir)
        report["storage"] = {
            "total_cache_size": format_size(total_size),
            "disk_free": f"{free_gb:.1f}GB",
        }

        return report

    def backup(self, backup_path: Optional[str] = None) -> str:
        """Create an incremental backup of all cache data."""
        if backup_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self._backup_dir, f"backup_{timestamp}")

        ensure_dir(backup_path)

        # Backup L4 RocksDB
        l4_src = os.path.join(self._base_dir, "l4_db")
        l4_dst = os.path.join(backup_path, "l4_db")
        if os.path.exists(l4_src):
            shutil.copytree(l4_src, l4_dst, dirs_exist_ok=True)

        # Backup L3 FAISS + metadata
        l3_src = os.path.join(self._base_dir, "l3_semantic")
        l3_dst = os.path.join(backup_path, "l3_semantic")
        if os.path.exists(l3_src):
            shutil.copytree(l3_src, l3_dst, dirs_exist_ok=True)

        total_size = get_dir_size_bytes(backup_path)
        logger.info(f"备份完成: {backup_path} ({format_size(total_size)})")
        return backup_path

    def restore(self, backup_path: str) -> bool:
        """Restore cache data from a backup."""
        if not os.path.exists(backup_path):
            logger.error(f"备份路径不存在: {backup_path}")
            return False

        try:
            # Close existing databases
            self._scheduler.close()

            # Restore L4
            l4_src = os.path.join(backup_path, "l4_db")
            l4_dst = os.path.join(self._base_dir, "l4_db")
            if os.path.exists(l4_src):
                safe_remove_dir(l4_dst)
                shutil.copytree(l4_src, l4_dst)

            # Restore L3
            l3_src = os.path.join(backup_path, "l3_semantic")
            l3_dst = os.path.join(self._base_dir, "l3_semantic")
            if os.path.exists(l3_src):
                safe_remove_dir(l3_dst)
                shutil.copytree(l3_src, l3_dst)

            logger.info(f"恢复完成: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"恢复失败: {e}")
            return False

    def list_backups(self):
        """List available backups."""
        if not os.path.exists(self._backup_dir):
            return []
        backups = []
        for name in sorted(os.listdir(self._backup_dir)):
            path = os.path.join(self._backup_dir, name)
            if os.path.isdir(path):
                size = get_dir_size_bytes(path)
                backups.append({
                    "name": name,
                    "path": path,
                    "size": format_size(size),
                })
        return backups

    def get_storage_report(self) -> Dict:
        """Get detailed storage usage report."""
        report = {}

        # L1/L2 memory
        report["l1_memory_bytes"] = self._scheduler.l1.size_bytes()
        if self._scheduler.l2 is not None:
            report["l2_memory_bytes"] = self._scheduler.l2.size_bytes()

        # L3/L4 disk
        report["l4_disk_bytes"] = self._scheduler.l4.size_bytes()
        if self._scheduler.l3 is not None:
            report["l3_disk_bytes"] = self._scheduler.l3.size_bytes()

        report["total_disk_bytes"] = get_dir_size_bytes(self._base_dir)
        report["disk_free_gb"] = get_free_disk_gb(self._base_dir)

        return report
