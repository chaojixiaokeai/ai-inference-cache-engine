"""Tests for cache layer functionality."""

import time
import unittest

import numpy as np

from engine.cache.base import CacheEntry, CacheLevel
from engine.cache.eviction import LFUEvictionPolicy, LRUEvictionPolicy, TTLLRUEvictionPolicy
from engine.cache.l1_kv_cache import L1KVCache
from engine.cache.l2_logits_cache import L2LogitsCache
from engine.utils.hashing import hash_token_sequence


class TestLRUEviction(unittest.TestCase):
    def test_eviction_order(self):
        policy = LRUEvictionPolicy(max_size_bytes=100)
        policy.touch(1, 40)
        policy.touch(2, 40)
        policy.touch(3, 40)

        evicted = policy.evict_if_needed()
        self.assertIn(1, evicted)
        self.assertLessEqual(policy.current_size, 100)

    def test_touch_refreshes_order(self):
        policy = LRUEvictionPolicy(max_size_bytes=100)
        policy.touch(1, 40)
        policy.touch(2, 40)
        policy.touch(1, 40)  # refresh key 1
        policy.touch(3, 40)

        evicted = policy.evict_if_needed()
        self.assertIn(2, evicted)
        self.assertNotIn(1, evicted)


class TestLFUEviction(unittest.TestCase):
    def test_least_frequent_evicted(self):
        policy = LFUEvictionPolicy(max_size_bytes=100)
        policy.touch(1, 40)
        policy.touch(2, 40)
        policy.touch(2, 40)  # key 2 accessed twice
        policy.touch(3, 40)

        evicted = policy.evict_if_needed()
        self.assertTrue(1 in evicted or 3 in evicted)
        self.assertNotIn(2, evicted[:1])


class TestTTLLRUEviction(unittest.TestCase):
    def test_expired_entries_evicted(self):
        policy = TTLLRUEvictionPolicy(max_size_bytes=1000, ttl_seconds=1)
        policy.touch(1, 50)
        time.sleep(1.1)
        evicted = policy.evict_expired()
        self.assertIn(1, evicted)


class TestL1KVCache(unittest.TestCase):
    def test_put_and_get(self):
        cache = L1KVCache(max_memory_mb=10)
        tokens = [1, 2, 3]
        ctx_hash = hash_token_sequence(tokens)
        entry = CacheEntry(
            context_hash=ctx_hash,
            model_hash="test_hash",
            greedy_token=42,
        )
        cache.put(entry)
        result = cache.get(ctx_hash, "test_hash")
        self.assertIsNotNone(result)
        self.assertEqual(result.greedy_token, 42)

    def test_model_hash_mismatch(self):
        cache = L1KVCache(max_memory_mb=10)
        ctx_hash = hash_token_sequence([1, 2, 3])
        entry = CacheEntry(context_hash=ctx_hash, model_hash="hash_a")
        cache.put(entry)
        result = cache.get(ctx_hash, "hash_b")
        self.assertIsNone(result)

    def test_delete(self):
        cache = L1KVCache(max_memory_mb=10)
        ctx_hash = hash_token_sequence([1, 2, 3])
        entry = CacheEntry(context_hash=ctx_hash, model_hash="test")
        cache.put(entry)
        self.assertTrue(cache.delete(ctx_hash))
        self.assertIsNone(cache.get(ctx_hash, "test"))

    def test_clear(self):
        cache = L1KVCache(max_memory_mb=10)
        for i in range(5):
            ctx_hash = hash_token_sequence([i])
            entry = CacheEntry(context_hash=ctx_hash, model_hash="test")
            cache.put(entry)
        cache.clear()
        self.assertEqual(cache.entry_count(), 0)


class TestL2LogitsCache(unittest.TestCase):
    def test_put_and_get_with_logits(self):
        cache = L2LogitsCache(max_memory_mb=10)
        ctx_hash = hash_token_sequence([10, 20, 30])
        entry = CacheEntry(
            context_hash=ctx_hash,
            model_hash="test",
            top_k_indices=np.array([5, 10, 15], dtype=np.int32),
            top_k_values=np.array([0.9, 0.05, 0.03], dtype=np.float32),
        )
        cache.put(entry)
        result = cache.get(ctx_hash, "test")
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result.top_k_indices, entry.top_k_indices)

    def test_sample_from_logits_greedy(self):
        indices = np.array([100, 200, 300], dtype=np.int32)
        values = np.array([10.0, 5.0, 1.0], dtype=np.float32)
        token = L2LogitsCache.sample_from_logits(indices, values, temperature=0)
        self.assertEqual(token, 100)


class TestCacheStats(unittest.TestCase):
    def test_hit_rate_tracking(self):
        cache = L1KVCache(max_memory_mb=10)
        ctx_hash = hash_token_sequence([1, 2])
        entry = CacheEntry(context_hash=ctx_hash, model_hash="test")
        cache.put(entry)

        cache.get(ctx_hash, "test")      # hit
        cache.get(ctx_hash, "test")      # hit
        cache.get(999, "test")           # miss

        self.assertEqual(cache._hits, 2)
        self.assertEqual(cache._misses, 1)
        self.assertAlmostEqual(cache.hit_rate, 2/3, places=2)


if __name__ == "__main__":
    unittest.main()
