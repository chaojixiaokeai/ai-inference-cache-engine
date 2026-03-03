"""Cross-device synchronization and model migration.

Enables cache export/import for device migration and
incremental cache re-computation on model upgrades.
"""

import json
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from engine.utils.storage import ensure_dir, format_size, get_dir_size_bytes

if TYPE_CHECKING:
    from engine.cache.scheduler import CacheScheduler
    from engine.config import EngineConfig

logger = logging.getLogger(__name__)


class SyncManager:
    """Manages cross-device cache export/import and model migration."""

    def __init__(self, config: "EngineConfig", scheduler: "CacheScheduler"):
        self._config = config
        self._scheduler = scheduler
        self._base_dir = os.path.expanduser(config.cache.base_dir)

    def export_cache(self, output_path: str) -> str:
        """Export all cache data to a portable tar.gz archive."""
        output = os.path.expanduser(output_path)
        if not output.endswith(".tar.gz"):
            output += ".tar.gz"

        ensure_dir(os.path.dirname(output))

        manifest = {
            "version": self._config.version,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_stats": self._scheduler.stats,
        }

        manifest_path = os.path.join(self._base_dir, "_export_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        try:
            with tarfile.open(output, "w:gz") as tar:
                tar.add(manifest_path, arcname="_export_manifest.json")

                l4_path = os.path.join(self._base_dir, "l4_db")
                if os.path.exists(l4_path):
                    tar.add(l4_path, arcname="l4_db")

                l3_path = os.path.join(self._base_dir, "l3_semantic")
                if os.path.exists(l3_path):
                    tar.add(l3_path, arcname="l3_semantic")

            size = os.path.getsize(output)
            logger.info(f"缓存导出完成: {output} ({format_size(size)})")
            return output
        finally:
            if os.path.exists(manifest_path):
                os.unlink(manifest_path)

    def import_cache(self, archive_path: str) -> bool:
        """Import cache data from a tar.gz archive."""
        archive = os.path.expanduser(archive_path)
        if not os.path.exists(archive):
            logger.error(f"导入文件不存在: {archive}")
            return False

        try:
            self._scheduler.close()

            with tarfile.open(archive, "r:gz") as tar:
                safe_members = []
                for member in tar.getmembers():
                    if member.name.startswith(("/", "..")):
                        continue
                    safe_members.append(member)

                tar.extractall(path=self._base_dir, members=safe_members)

            manifest_path = os.path.join(self._base_dir, "_export_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                logger.info(
                    f"缓存导入完成 (导出自 v{manifest.get('version')}, "
                    f"{manifest.get('exported_at')})"
                )
                os.unlink(manifest_path)

            return True
        except Exception as e:
            logger.error(f"缓存导入失败: {e}")
            return False

    def migrate_model_cache(
        self,
        old_model_hash: str,
        new_engine: "InferenceEngine",
    ) -> Dict:
        """Migrate high-frequency cache entries from old model to new model.

        Re-computes cached outputs using the new model for entries that
        had high access counts under the old model.
        """
        report = {"migrated": 0, "skipped": 0, "errors": 0}

        l4 = self._scheduler.l4
        if not l4.enabled:
            return report

        try:
            from engine.cache.l4_persistent_cache import _deserialize_entry, _serialize_entry
            from engine.cache.base import CacheEntry
            from engine.utils.hashing import make_cache_key

            new_model_hash = new_engine.model_hash
            if new_model_hash is None:
                return report

            entries_to_migrate = []
            for key in l4._db.keys():
                try:
                    data = l4._db[key]
                    entry = _deserialize_entry(data)
                    if entry.model_hash == old_model_hash and entry.access_count >= 3:
                        entries_to_migrate.append(entry)
                except Exception:
                    continue

            for entry in entries_to_migrate[:100]:
                try:
                    if entry.token_sequence is None:
                        report["skipped"] += 1
                        continue

                    greedy = new_engine.get_greedy_token(entry.token_sequence)
                    top_idx, top_val = new_engine.get_top_k_logits(entry.token_sequence)

                    new_entry = CacheEntry(
                        context_hash=entry.context_hash,
                        model_hash=new_model_hash,
                        greedy_token=greedy,
                        top_k_indices=top_idx,
                        top_k_values=top_val,
                        response_text=None,
                        token_sequence=entry.token_sequence,
                    )
                    l4.put(new_entry)
                    report["migrated"] += 1

                except Exception as e:
                    report["errors"] += 1
                    logger.debug(f"条目迁移失败: {e}")

        except Exception as e:
            logger.error(f"模型缓存迁移失败: {e}")

        logger.info(
            f"模型缓存迁移完成: 迁移{report['migrated']}, "
            f"跳过{report['skipped']}, 错误{report['errors']}"
        )
        return report
