"""L3 SSD-level Semantic Cache: Controlled error, high hit rate.

Uses FAISS for vector similarity search and RocksDB for metadata storage.
Matches semantically similar queries (cosine similarity >= threshold).
Can be toggled on/off; when off, engine runs in strict zero-error mode.
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
from engine.cache.l4_persistent_cache import _serialize_entry, _deserialize_entry
from engine.utils.storage import ensure_dir, get_dir_size_bytes

logger = logging.getLogger(__name__)


class L3SemanticCache(BaseCache):
    """FAISS-backed semantic similarity cache with RocksDB metadata."""

    def __init__(
        self,
        base_path: str = "~/.cache/ai_engine/l3_semantic",
        max_disk_gb: int = 50,
        similarity_threshold: float = 0.98,
        embedding_dim: int = 4096,
        enabled: bool = False,
    ):
        super().__init__(CacheLevel.L3_SEMANTIC_DISK, enabled)
        self._base_path = os.path.expanduser(base_path)
        self._max_bytes = max_disk_gb * (1024 ** 3)
        self._threshold = similarity_threshold
        self._embedding_dim = embedding_dim
        self._faiss_index = None
        self._metadata_db = None
        self._id_to_hash: Dict[int, int] = {}
        self._next_id = 0
        self._lock = threading.RLock()
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            import faiss
            ensure_dir(self._base_path)
            index_path = os.path.join(self._base_path, "vectors.index")
            if os.path.exists(index_path):
                self._faiss_index = faiss.read_index(index_path)
                self._next_id = self._faiss_index.ntotal
            else:
                self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)

            from rocksdict import Rdict
            db_path = os.path.join(self._base_path, "metadata_db")
            self._metadata_db = Rdict(db_path)

            self._rebuild_id_map()
            self._initialized = True
            logger.info(
                f"L3 语义缓存初始化完成: {self._faiss_index.ntotal} 个向量, "
                f"dim={self._embedding_dim}"
            )
        except ImportError as e:
            logger.warning(f"L3 语义缓存依赖缺失: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"L3 语义缓存初始化失败: {e}")
            self.enabled = False

    def _rebuild_id_map(self):
        """Rebuild vector ID -> context hash mapping from metadata DB."""
        if self._metadata_db is None:
            return
        try:
            for key in self._metadata_db.keys():
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                if key_str.startswith("id:"):
                    vec_id = int(key_str[3:])
                    ctx_hash = int.from_bytes(self._metadata_db[key], "little")
                    self._id_to_hash[vec_id] = ctx_hash
                    self._next_id = max(self._next_id, vec_id + 1)
        except Exception as e:
            logger.warning(f"L3 ID映射重建部分失败: {e}")

    def search(
        self,
        tokens: List[int],
        model_hash: str,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[CacheEntry]:
        """Search for semantically similar cached entries.

        Args:
            tokens: Current context tokens (unused if embedding provided)
            model_hash: Current model hash for verification
            embedding: Pre-computed embedding vector

        Returns:
            CacheEntry if similarity >= threshold, else None
        """
        if not self.enabled:
            self.record_miss()
            return None

        with self._lock:
            self._ensure_initialized()
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                self.record_miss()
                return None
            if embedding is None:
                self.record_miss()
                return None

            query = embedding.reshape(1, -1).astype(np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            try:
                scores, indices = self._faiss_index.search(query, 1)
                score = float(scores[0][0])
                idx = int(indices[0][0])

                if score < self._threshold or idx < 0:
                    self.record_miss()
                    return None

                ctx_hash = self._id_to_hash.get(idx)
                if ctx_hash is None:
                    self.record_miss()
                    return None

                entry_key = f"entry:{ctx_hash}".encode("utf-8")
                data = self._metadata_db[entry_key]
                entry = _deserialize_entry(data)

                if entry.model_hash != model_hash:
                    self.record_miss()
                    return None

                entry.touch()
                entry.similarity_score = score
                self.record_hit()
                logger.debug(
                    f"L3 语义命中: similarity={score:.4f}, hash={ctx_hash}"
                )
                return entry

            except Exception as e:
                logger.warning(f"L3 搜索异常: {e}")
                self.record_miss()
                return None

    def get(self, context_hash: int, model_hash: str) -> Optional[CacheEntry]:
        """Exact hash lookup (not the primary use case for L3)."""
        if not self.enabled:
            self.record_miss()
            return None

        with self._lock:
            self._ensure_initialized()
            if self._metadata_db is None:
                self.record_miss()
                return None

            try:
                entry_key = f"entry:{context_hash}".encode("utf-8")
                data = self._metadata_db[entry_key]
                entry = _deserialize_entry(data)
                if entry.model_hash != model_hash:
                    self.record_miss()
                    return None
                entry.touch()
                self.record_hit()
                return entry
            except (KeyError, Exception):
                self.record_miss()
                return None

    def put(self, entry: CacheEntry, embedding: Optional[np.ndarray] = None) -> None:
        """Store entry with its embedding vector for semantic matching."""
        if not self.enabled or embedding is None:
            return

        with self._lock:
            self._ensure_initialized()
            if self._faiss_index is None or self._metadata_db is None:
                return

            try:
                vec = embedding.reshape(1, -1).astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                vec_id = self._next_id
                self._faiss_index.add(vec)
                self._next_id += 1

                self._id_to_hash[vec_id] = entry.context_hash

                self._metadata_db[f"id:{vec_id}".encode("utf-8")] = (
                    entry.context_hash.to_bytes(8, "little")
                )
                entry_key = f"entry:{entry.context_hash}".encode("utf-8")
                self._metadata_db[entry_key] = _serialize_entry(entry)

                logger.debug(
                    f"L3 写入: id={vec_id}, hash={entry.context_hash}"
                )
            except Exception as e:
                logger.error(f"L3 写入异常: {e}")

    def delete(self, context_hash: int) -> bool:
        with self._lock:
            if self._metadata_db is None:
                return False
            try:
                entry_key = f"entry:{context_hash}".encode("utf-8")
                del self._metadata_db[entry_key]
                return True
            except (KeyError, Exception):
                return False

    def clear(self) -> None:
        with self._lock:
            if self._metadata_db is not None:
                self._metadata_db.close()
                self._metadata_db = None
            self._faiss_index = None
            self._id_to_hash.clear()
            self._next_id = 0
            self._initialized = False

            import shutil
            if os.path.exists(self._base_path):
                shutil.rmtree(self._base_path, ignore_errors=True)
            logger.info("L3 语义缓存已清空")

    def size_bytes(self) -> int:
        return get_dir_size_bytes(self._base_path)

    def entry_count(self) -> int:
        if self._faiss_index is not None:
            return self._faiss_index.ntotal
        return 0

    def save_index(self) -> None:
        """Persist FAISS index to disk."""
        with self._lock:
            if self._faiss_index is not None:
                import faiss
                index_path = os.path.join(self._base_path, "vectors.index")
                faiss.write_index(self._faiss_index, index_path)
                logger.info("L3 FAISS索引已保存")

    def close(self) -> None:
        with self._lock:
            self.save_index()
            if self._metadata_db is not None:
                self._metadata_db.close()
                self._metadata_db = None
            logger.info("L3 语义缓存已关闭")

    def evict_stale(self, max_idle_days: int = 7) -> int:
        """Remove entries not accessed within max_idle_days."""
        if not self._initialized or self._metadata_db is None:
            return 0

        now = time.time()
        threshold = max_idle_days * 86400
        evicted = 0

        with self._lock:
            try:
                keys_to_delete = []
                for key in self._metadata_db.keys():
                    key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                    if key_str.startswith("entry:"):
                        try:
                            data = self._metadata_db[key]
                            entry = _deserialize_entry(data)
                            if now - entry.last_accessed > threshold:
                                keys_to_delete.append(key)
                        except Exception:
                            continue

                for key in keys_to_delete:
                    try:
                        del self._metadata_db[key]
                        evicted += 1
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"L3 过期清理异常: {e}")

        if evicted:
            logger.info(f"L3 清理过期条目: {evicted} 个")
        return evicted
