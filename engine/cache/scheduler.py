"""Cache scheduling engine: the brain of the hybrid cache system.

Controls the query/write flow across all cache levels.
Implements: cache-first, native-fallback, verify-always, seamless-switch.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from engine.cache.base import CacheEntry, CacheHitType, CacheLevel, CacheResult
from engine.cache.l1_kv_cache import L1KVCache
from engine.cache.l2_logits_cache import L2LogitsCache
from engine.cache.l3_semantic_cache import L3SemanticCache
from engine.cache.l4_persistent_cache import L4PersistentCache
from engine.config import EngineConfig
from engine.utils.hashing import hash_token_sequence
from engine.verification.consistency import ConsistencyVerifier, VerifyResult

if TYPE_CHECKING:
    from engine.core.inference import InferenceEngine, GenerationParams

logger = logging.getLogger(__name__)


@dataclass
class RequestStats:
    """Statistics for a single request processing."""
    cache_hit: bool = False
    cache_level: Optional[str] = None
    hit_type: CacheHitType = CacheHitType.MISS
    verify_result: Optional[str] = None
    tokens_generated: int = 0
    time_ms: float = 0
    from_cache: bool = False


class CacheScheduler:
    """Orchestrates cache lookup, verification, and writeback across levels."""

    def __init__(
        self,
        config: EngineConfig,
        inference_engine: "InferenceEngine",
    ):
        self._config = config
        self._engine = inference_engine
        self._verifier = ConsistencyVerifier(inference_engine)

        self._l1 = L1KVCache(
            max_memory_mb=config.cache.l1.max_memory_mb,
            enabled=config.cache.l1.enabled,
        )

        cache_base = config.cache.base_dir
        self._l4 = L4PersistentCache(
            db_path=f"{cache_base}/l4_db",
            max_disk_gb=config.cache.l4.max_disk_gb,
            ttl_days=config.cache.l4.ttl_days,
            enabled=config.cache.l4.enabled,
        )

        # L2 Logits Cache
        self._l2 = L2LogitsCache(
            max_memory_mb=config.cache.l2.max_memory_mb,
            top_k=config.cache.l2.top_k_logits,
            enabled=config.cache.l2.enabled,
        )

        # L3 Semantic Cache
        self._l3 = L3SemanticCache(
            base_path=f"{cache_base}/l3_semantic",
            max_disk_gb=config.cache.l3.max_disk_gb,
            similarity_threshold=config.cache.l3.similarity_threshold,
            enabled=config.cache.l3.enabled,
        )

        self._total_requests = 0
        self._cache_hits = 0
        self._lock = threading.RLock()

    @property
    def l1(self) -> L1KVCache:
        return self._l1

    @property
    def l2(self) -> L2LogitsCache:
        return self._l2

    @property
    def l3(self) -> L3SemanticCache:
        return self._l3

    @property
    def l4(self) -> L4PersistentCache:
        return self._l4

    @property
    def verifier(self) -> ConsistencyVerifier:
        return self._verifier

    def process_request(
        self,
        tokens: List[int],
        params: "GenerationParams",
        model_hash: str,
    ) -> Iterator[str]:
        """Main scheduling flow: cache lookup -> verify -> output or native inference.

        Yields text chunks for streaming output.
        """
        self._total_requests += 1
        start_time = time.time()
        context_hash = hash_token_sequence(tokens)

        # --- Phase 1: Cache Lookup (L1 -> L2 -> L4 -> L3) ---
        cache_result = self._lookup_caches(context_hash, tokens, model_hash)

        if cache_result.hit and cache_result.entry is not None:
            # --- Phase 2: Consistency Verification ---
            entry = cache_result.entry
            if cache_result.hit_type == CacheHitType.EXACT:
                vr = self._verifier.verify_exact(entry, tokens, model_hash)
            elif cache_result.hit_type == CacheHitType.SEMANTIC:
                vr = self._verifier.verify_semantic(entry, tokens, model_hash)
            else:
                vr = VerifyResult.PASS

            if vr == VerifyResult.PASS:
                self._cache_hits += 1
                logger.info(
                    f"缓存命中 [{cache_result.level.value}] "
                    f"hash={context_hash}, 类型={cache_result.hit_type.value}"
                )
                yield from self._output_from_cache(entry, cache_result)
                return
            else:
                logger.warning(
                    f"缓存校验失败 [{cache_result.level.value}]: {vr.value}, "
                    f"清除脏缓存并回退到原生推理"
                )
                self._invalidate_entry(context_hash, cache_result.level)

        # --- Phase 3: Native Inference Fallback ---
        logger.debug(f"缓存未命中, 启动原生全精度推理 (tokens={len(tokens)})")
        yield from self._native_inference_and_cache(
            tokens, params, model_hash, context_hash
        )

    def _lookup_caches(
        self,
        context_hash: int,
        tokens: List[int],
        model_hash: str,
    ) -> CacheResult:
        """Query caches in order: L1 -> L2 -> L4 -> L3."""

        # L1: Memory KV Cache
        entry = self._l1.get(context_hash, model_hash)
        if entry is not None:
            return CacheResult(
                hit=True,
                level=CacheLevel.L1_KV_MEMORY,
                hit_type=CacheHitType.EXACT,
                entry=entry,
            )

        # L2: Memory Logits Cache
        if self._l2 is not None and self._l2.enabled:
            entry = self._l2.get(context_hash, model_hash)
            if entry is not None:
                return CacheResult(
                    hit=True,
                    level=CacheLevel.L2_LOGITS_MEMORY,
                    hit_type=CacheHitType.EXACT,
                    entry=entry,
                )

        # L4: Persistent Exact Cache
        entry = self._l4.get(context_hash, model_hash)
        if entry is not None:
            return CacheResult(
                hit=True,
                level=CacheLevel.L4_PERSISTENT_DISK,
                hit_type=CacheHitType.EXACT,
                entry=entry,
            )

        # L3: Semantic Cache
        if self._l3 is not None and self._l3.enabled:
            try:
                embedding = self._engine.get_embedding(tokens)
                entry = self._l3.search(tokens, model_hash, embedding=embedding)
                if entry is not None:
                    return CacheResult(
                        hit=True,
                        level=CacheLevel.L3_SEMANTIC_DISK,
                        hit_type=CacheHitType.SEMANTIC,
                        entry=entry,
                    )
            except Exception as e:
                logger.debug(f"L3 语义搜索跳过: {e}")

        return CacheResult(
            hit=False,
            level=CacheLevel.L1_KV_MEMORY,
            hit_type=CacheHitType.MISS,
        )

    def _output_from_cache(
        self, entry: CacheEntry, result: CacheResult
    ) -> Iterator[str]:
        """Stream output from a verified cache entry."""
        if entry.response_text:
            chunk_size = 4
            text = entry.response_text
            for i in range(0, len(text), chunk_size):
                yield text[i : i + chunk_size]
        elif entry.top_k_indices is not None and entry.top_k_values is not None:
            token_id = int(entry.top_k_indices[0])
            text = self._engine.detokenize([token_id])
            yield text

    def _native_inference_and_cache(
        self,
        tokens: List[int],
        params: "GenerationParams",
        model_hash: str,
        context_hash: int,
    ) -> Iterator[str]:
        """Run native inference, stream output, and write results to cache."""
        generated_text = []
        generated_tokens = []

        try:
            for text_chunk in self._engine.generate_text(
                self._engine.detokenize(tokens), params
            ):
                generated_text.append(text_chunk)
                yield text_chunk
        except Exception as e:
            logger.error(f"原生推理异常: {e}")
            yield f"\n[推理错误: {e}]"
            return

        full_response = "".join(generated_text)

        # Write back to caches
        try:
            greedy_token = -1
            top_k_indices = None
            top_k_values = None

            try:
                greedy_token = self._engine.get_greedy_token(tokens)
                top_k_indices, top_k_values = self._engine.get_top_k_logits(tokens)
            except Exception as e:
                logger.debug(f"logits提取跳过: {e}")

            entry = CacheEntry(
                context_hash=context_hash,
                model_hash=model_hash,
                greedy_token=greedy_token,
                top_k_indices=top_k_indices,
                top_k_values=top_k_values,
                response_text=full_response,
                token_sequence=tokens,
            )

            # Always write to L4 persistent cache
            try:
                self._l4.put(entry)
            except Exception as e:
                logger.debug(f"L4 写入跳过: {e}")

            # Write to L1 if KV state available
            try:
                kv_state = self._engine.save_state()
                entry.kv_state = kv_state
                self._l1.put(entry)
            except Exception as e:
                logger.debug(f"L1 KV状态保存跳过: {e}")

            # Write to L2 if logits available
            if self._l2 is not None and self._l2.enabled:
                if entry.top_k_indices is not None:
                    try:
                        self._l2.put(entry)
                    except Exception as e:
                        logger.debug(f"L2 写入跳过: {e}")

            # Write to L3 if semantic cache enabled
            if self._l3 is not None and self._l3.enabled:
                try:
                    embedding = self._engine.get_embedding(tokens)
                    entry.embedding = embedding
                    self._l3.put(entry, embedding=embedding)
                except Exception as e:
                    logger.debug(f"L3 写入跳过: {e}")

        except Exception as e:
            logger.warning(f"缓存写回异常 (不影响输出): {e}")

    def _invalidate_entry(self, context_hash: int, level: CacheLevel):
        """Remove a dirty cache entry."""
        try:
            if level == CacheLevel.L1_KV_MEMORY:
                self._l1.delete(context_hash)
            elif level == CacheLevel.L4_PERSISTENT_DISK:
                self._l4.delete(context_hash)
        except Exception as e:
            logger.warning(f"脏缓存清除异常: {e}")

    @property
    def stats(self) -> dict:
        total = self._total_requests
        result = {
            "total_requests": total,
            "cache_hits": self._cache_hits,
            "hit_rate": round(self._cache_hits / total, 4) if total > 0 else 0,
            "l1": self._l1.stats,
            "l4": self._l4.stats,
            "verification": self._verifier.stats,
        }
        if self._l2 is not None:
            result["l2"] = self._l2.stats
        if self._l3 is not None:
            result["l3"] = self._l3.stats
        return result

    def clear_all(self):
        """Clear all cache levels."""
        self._l1.clear()
        if self._l2:
            self._l2.clear()
        if self._l3:
            self._l3.clear()
        self._l4.clear()
        logger.info("所有缓存已清空")

    def persist_l1_to_l4(self) -> int:
        """会话结束：将内存缓存持久化到硬盘。

        设计原则：每句话发完就存内存，会话结束只存硬盘。
        符合「逐 token 存/取」「只认单次指令」，缓存即时生效，不依赖会话是否结束。

        Returns:
            持久化到 L4 的条目数
        """
        count = 0
        try:
            for ctx_hash, entry in self._l1.iter_entries():
                if entry.response_text:
                    try:
                        self._l4.put(entry)
                        count += 1
                    except Exception as e:
                        logger.debug(f"L1->L4 持久化跳过 hash={ctx_hash}: {e}")
            if count > 0:
                logger.info(f"会话结束: L1 已持久化 {count} 条到 L4，保证跨重启生效")
        except Exception as e:
            logger.warning(f"L1->L4 持久化异常: {e}")
        return count

    def close(self):
        """Clean shutdown: persist L1 to L4, then close all backends."""
        self.persist_l1_to_l4()
        self._l4.close()
        if self._l3 is not None:
            try:
                self._l3.close()
            except Exception:
                pass
        logger.info("缓存调度引擎已关闭")
