"""L4 Persistent Exact Cache: Zero-error, long-term reuse.

Stores full-precision top-K logits and greedy tokens on SSD via RocksDB.
Uses TTL + LRU eviction with configurable disk hard limit.
Data persists across engine restarts.
"""

import io
import logging
import os
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.cache.base import BaseCache, CacheEntry, CacheLevel
from engine.cache.eviction import TTLLRUEvictionPolicy
from engine.utils.hashing import make_cache_key
from engine.utils.storage import ensure_dir, get_dir_size_bytes

logger = logging.getLogger(__name__)


def _serialize_entry(entry: CacheEntry) -> bytes:
    """Serialize a CacheEntry to bytes for RocksDB storage."""
    buf = io.BytesIO()

    buf.write(struct.pack("<Q", entry.context_hash))

    model_hash_bytes = entry.model_hash.encode("utf-8")
    buf.write(struct.pack("<H", len(model_hash_bytes)))
    buf.write(model_hash_bytes)

    buf.write(struct.pack("<d", entry.created_at))
    buf.write(struct.pack("<d", entry.last_accessed))
    buf.write(struct.pack("<I", entry.access_count))
    buf.write(struct.pack("<i", entry.greedy_token))

    if entry.top_k_indices is not None and entry.top_k_values is not None:
        idx_bytes = entry.top_k_indices.astype(np.int32).tobytes()
        val_bytes = entry.top_k_values.astype(np.float32).tobytes()
        buf.write(struct.pack("<I", len(idx_bytes)))
        buf.write(idx_bytes)
        buf.write(struct.pack("<I", len(val_bytes)))
        buf.write(val_bytes)
    else:
        buf.write(struct.pack("<I", 0))
        buf.write(struct.pack("<I", 0))

    if entry.response_text is not None:
        text_bytes = entry.response_text.encode("utf-8")
        buf.write(struct.pack("<I", len(text_bytes)))
        buf.write(text_bytes)
    else:
        buf.write(struct.pack("<I", 0))

    if entry.token_sequence is not None:
        seq_bytes = np.array(entry.token_sequence, dtype=np.int32).tobytes()
        buf.write(struct.pack("<I", len(seq_bytes)))
        buf.write(seq_bytes)
    else:
        buf.write(struct.pack("<I", 0))

    return buf.getvalue()


def _deserialize_entry(data: bytes) -> CacheEntry:
    """Deserialize bytes back to a CacheEntry."""
    buf = io.BytesIO(data)

    context_hash = struct.unpack("<Q", buf.read(8))[0]

    mh_len = struct.unpack("<H", buf.read(2))[0]
    model_hash = buf.read(mh_len).decode("utf-8")

    created_at = struct.unpack("<d", buf.read(8))[0]
    last_accessed = struct.unpack("<d", buf.read(8))[0]
    access_count = struct.unpack("<I", buf.read(4))[0]
    greedy_token = struct.unpack("<i", buf.read(4))[0]

    idx_len = struct.unpack("<I", buf.read(4))[0]
    if idx_len > 0:
        top_k_indices = np.frombuffer(buf.read(idx_len), dtype=np.int32).copy()
    else:
        top_k_indices = None

    val_len = struct.unpack("<I", buf.read(4))[0]
    if val_len > 0:
        top_k_values = np.frombuffer(buf.read(val_len), dtype=np.float32).copy()
    else:
        top_k_values = None

    text_len = struct.unpack("<I", buf.read(4))[0]
    if text_len > 0:
        response_text = buf.read(text_len).decode("utf-8")
    else:
        response_text = None

    seq_len = struct.unpack("<I", buf.read(4))[0]
    if seq_len > 0:
        token_sequence = np.frombuffer(buf.read(seq_len), dtype=np.int32).tolist()
    else:
        token_sequence = None

    return CacheEntry(
        context_hash=context_hash,
        model_hash=model_hash,
        created_at=created_at,
        last_accessed=last_accessed,
        access_count=access_count,
        greedy_token=greedy_token,
        top_k_indices=top_k_indices,
        top_k_values=top_k_values,
        response_text=response_text,
        token_sequence=token_sequence,
    )


class L4PersistentCache(BaseCache):
    """RocksDB-backed persistent cache with TTL+LRU eviction."""

    def __init__(
        self,
        db_path: str = "~/.cache/ai_engine/l4_db",
        max_disk_gb: int = 100,
        ttl_days: int = 90,
        enabled: bool = True,
    ):
        super().__init__(CacheLevel.L4_PERSISTENT_DISK, enabled)
        self._db_path = os.path.expanduser(db_path)
        self._max_bytes = max_disk_gb * (1024 ** 3)
        self._ttl_seconds = ttl_days * 86400
        self._db = None
        self._lock = threading.RLock()
        self._eviction = TTLLRUEvictionPolicy(self._max_bytes, self._ttl_seconds)
        self._initialized = False

    def _ensure_db(self):
        if self._db is not None:
            return
        ensure_dir(self._db_path)
        try:
            from rocksdict import Rdict
            self._db = Rdict(self._db_path)
            self._initialized = True
            self._rebuild_eviction_index()
            logger.info(f"L4 数据库已打开: {self._db_path}")
        except ImportError:
            logger.warning("rocksdict 未安装，L4 持久化缓存不可用")
            self.enabled = False
        except Exception as e:
            logger.error(f"L4 数据库打开失败: {e}")
            self.enabled = False

    def _rebuild_eviction_index(self):
        """Rebuild eviction index from existing database entries on startup."""
        if self._db is None:
            return
        count = 0
        try:
            for key in self._db.keys():
                try:
                    data = self._db[key]
                    entry = _deserialize_entry(data)
                    self._eviction.touch(entry.context_hash, len(data))
                    count += 1
                except Exception:
                    continue
            logger.info(f"L4 重建索引完成: {count} 个条目")
        except Exception as e:
            logger.warning(f"L4 索引重建部分失败: {e}")

    def get(self, context_hash: int, model_hash: str) -> Optional[CacheEntry]:
        if not self.enabled:
            self.record_miss()
            return None

        with self._lock:
            self._ensure_db()
            if self._db is None:
                self.record_miss()
                return None

            key = make_cache_key(model_hash, context_hash)
            try:
                data = self._db[key]
            except KeyError:
                self.record_miss()
                return None
            except Exception as e:
                logger.warning(f"L4 读取异常: {e}")
                self.record_miss()
                return None

            entry = _deserialize_entry(data)
            if entry.model_hash != model_hash:
                self.record_miss()
                return None

            now = time.time()
            if now - entry.created_at > self._ttl_seconds:
                self._safe_delete(key, context_hash)
                self.record_miss()
                return None

            entry.touch()
            self._db[key] = _serialize_entry(entry)
            self._eviction.touch(context_hash, len(data))
            self.record_hit()
            logger.debug(f"L4 命中: hash={context_hash}")
            return entry

    def put(self, entry: CacheEntry) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._ensure_db()
            if self._db is None:
                return

            key = make_cache_key(entry.model_hash, entry.context_hash)
            data = _serialize_entry(entry)

            if self._eviction.should_evict():
                evicted = self._eviction.evict_if_needed()
                for ctx_hash in evicted:
                    self._delete_by_context_hash(ctx_hash, entry.model_hash)
                if evicted:
                    logger.debug(f"L4 淘汰 {len(evicted)} 个条目")

            try:
                self._db[key] = data
                self._eviction.touch(entry.context_hash, len(data))
            except Exception as e:
                logger.error(f"L4 写入异常: {e}")

    def delete(self, context_hash: int) -> bool:
        with self._lock:
            self._ensure_db()
            if self._db is None:
                return False
            self._eviction.remove(context_hash)
            found = False
            try:
                for key in list(self._db.keys()):
                    key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                    if str(context_hash) in key_str:
                        del self._db[key]
                        found = True
            except Exception as e:
                logger.warning(f"L4 删除异常: {e}")
            return found

    def _delete_by_context_hash(self, context_hash: int, model_hash: str):
        key = make_cache_key(model_hash, context_hash)
        try:
            del self._db[key]
        except (KeyError, Exception):
            pass

    def _safe_delete(self, key: bytes, context_hash: int):
        try:
            del self._db[key]
            self._eviction.remove(context_hash)
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None
            import shutil
            if os.path.exists(self._db_path):
                shutil.rmtree(self._db_path, ignore_errors=True)
            self._eviction.clear()
            self._initialized = False
            logger.info("L4 缓存已清空")

    def size_bytes(self) -> int:
        return get_dir_size_bytes(self._db_path)

    def entry_count(self) -> int:
        if not self.enabled or self._db is None:
            return 0
        try:
            return len(list(self._db.keys()))
        except Exception:
            return 0

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None
                logger.info("L4 数据库已关闭")

    def compact(self) -> None:
        """Trigger RocksDB compaction for defragmentation."""
        with self._lock:
            if self._db is not None:
                try:
                    self._db.compact_range()
                    logger.info("L4 数据库碎片整理完成")
                except Exception as e:
                    logger.warning(f"L4 碎片整理失败: {e}")
