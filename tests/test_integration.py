"""Comprehensive end-to-end integration tests using a real Qwen3-8B F16 GGUF model.

Tests the full pipeline: model validation -> inference -> cache scheduling ->
consistency verification -> lifecycle management -> privacy -> tools.
"""

import os
import shutil
import sys
import time
import unittest

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Qwen3-8B-f16.gguf")
TEST_CACHE_DIR = os.path.expanduser("~/.cache/ai_engine_integration_test")

SKIP_REASON = "Qwen3-8B-f16.gguf not found - skipping integration tests"


def model_available():
    return os.path.exists(MODEL_PATH)


@unittest.skipUnless(model_available(), SKIP_REASON)
class TestModelValidation(unittest.TestCase):
    """Red-line compliance with real model."""

    def test_real_model_passes_validation(self):
        from engine.core.model_validator import validate_model
        metadata, model_hash = validate_model(MODEL_PATH)
        self.assertIn(metadata.get("general.file_type"), {0, 1, 26})
        self.assertTrue(len(model_hash) > 0)

    def test_model_architecture_detected(self):
        from engine.core.model_validator import validate_model
        metadata, _ = validate_model(MODEL_PATH)
        arch = metadata.get("general.architecture", "")
        self.assertTrue(len(arch) > 0, "Architecture should be detected")
        self.assertIsNotNone(metadata.get(f"{arch}.block_count"))
        self.assertIsNotNone(metadata.get(f"{arch}.attention.head_count"))


@unittest.skipUnless(model_available(), SKIP_REASON)
class TestInferenceEngine(unittest.TestCase):
    """Core inference engine with real model."""

    @classmethod
    def setUpClass(cls):
        from engine.config import load_config
        from engine.core.inference import InferenceEngine
        cls.config = load_config(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        )
        cls.config.model.path = MODEL_PATH
        cls.config.model.n_ctx = 512
        cls.config.model.n_batch = 64
        cls.engine = InferenceEngine(cls.config)
        cls.engine.load()

    @classmethod
    def tearDownClass(cls):
        cls.engine.unload()

    def test_tokenize_roundtrip(self):
        text = "Hello, 你好世界!"
        tokens = self.engine.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        decoded = self.engine.detokenize(tokens)
        self.assertIn("Hello", decoded)
        self.assertIn("你好", decoded)

    def test_greedy_token_deterministic(self):
        tokens = self.engine.tokenize("1+1=")
        gt1 = self.engine.get_greedy_token(tokens)
        gt2 = self.engine.get_greedy_token(tokens)
        self.assertEqual(gt1, gt2, "Greedy token should be deterministic")

    def test_generate_text_streaming(self):
        from engine.core.inference import GenerationParams
        params = GenerationParams(temperature=0, max_tokens=20)
        chunks = list(self.engine.generate_text("What is 2+2?", params))
        self.assertTrue(len(chunks) > 0, "Should generate at least one chunk")
        full_text = "".join(chunks)
        self.assertTrue(len(full_text) > 0, "Should produce non-empty output")

    def test_top_k_logits(self):
        tokens = self.engine.tokenize("Python is")
        indices, values = self.engine.get_top_k_logits(tokens, k=10)
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(values), 10)
        self.assertTrue(values[0] >= values[1], "Top logits should be sorted descending")

    def test_model_hash_stable(self):
        h1 = self.engine.model_hash
        h2 = self.engine.model_hash
        self.assertEqual(h1, h2)
        self.assertTrue(len(h1) == 64, "SHA256 hash should be 64 hex chars")


@unittest.skipUnless(model_available(), SKIP_REASON)
class TestCacheSchedulerIntegration(unittest.TestCase):
    """Full cache scheduling pipeline with real model."""

    @classmethod
    def setUpClass(cls):
        if os.path.exists(TEST_CACHE_DIR):
            shutil.rmtree(TEST_CACHE_DIR)

        from engine.config import load_config
        from engine.core.inference import InferenceEngine, GenerationParams
        from engine.cache.scheduler import CacheScheduler

        cls.config = load_config(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        )
        cls.config.model.path = MODEL_PATH
        cls.config.model.n_ctx = 512
        cls.config.model.n_batch = 64
        cls.config.cache.base_dir = TEST_CACHE_DIR
        cls.config.cache.l3.enabled = False

        cls.engine = InferenceEngine(cls.config)
        cls.engine.load()
        cls.scheduler = CacheScheduler(cls.config, cls.engine)
        cls.params = GenerationParams(temperature=0, max_tokens=20, seed=42)

    @classmethod
    def tearDownClass(cls):
        cls.scheduler.close()
        cls.engine.unload()
        shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)

    def test_01_first_request_miss(self):
        """First request should be a cache miss and trigger native inference."""
        tokens = self.engine.tokenize("什么是人工智能?")
        output = "".join(
            self.scheduler.process_request(tokens, self.params, self.engine.model_hash)
        )
        self.assertTrue(len(output) > 0)
        self.__class__._first_output = output

        stats = self.scheduler.stats
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["cache_hits"], 0)

    def test_02_second_request_hit(self):
        """Same request should hit cache."""
        tokens = self.engine.tokenize("什么是人工智能?")
        start = time.time()
        output = "".join(
            self.scheduler.process_request(tokens, self.params, self.engine.model_hash)
        )
        elapsed = time.time() - start

        self.assertEqual(output, self.__class__._first_output,
                         "Cached output must be identical to first output")
        stats = self.scheduler.stats
        self.assertEqual(stats["cache_hits"], 1)

    def test_03_different_request_miss(self):
        """Different prompt should miss cache."""
        tokens = self.engine.tokenize("什么是机器学习?")
        output = "".join(
            self.scheduler.process_request(tokens, self.params, self.engine.model_hash)
        )
        self.assertTrue(len(output) > 0)
        stats = self.scheduler.stats
        self.assertEqual(stats["total_requests"], 3)

    def test_04_cache_stats_accurate(self):
        stats = self.scheduler.stats
        self.assertGreater(stats["l1"]["entries"], 0)
        self.assertGreater(stats["l4"]["entries"], 0)
        self.assertGreaterEqual(stats["verification"]["pass_rate"], 0)

    def test_05_consistency_verification_works(self):
        stats = self.scheduler.stats
        v = stats["verification"]
        self.assertEqual(v["failed"], 0, "No consistency failures should occur")


@unittest.skipUnless(model_available(), SKIP_REASON)
class TestPrivacyModule(unittest.TestCase):
    """Privacy sanitization and encryption."""

    def test_sensitive_info_sanitized(self):
        from engine.safety.privacy import sanitize_sensitive_info
        cases = {
            "身份证110101199001011234": "[身份证号已脱敏]",
            "电话13912345678": "[手机号已脱敏]",
            "email: user@test.com": "[邮箱已脱敏]",
            "api_key: secret123": "[密钥已脱敏]",
        }
        for text, expected in cases.items():
            result = sanitize_sensitive_info(text)
            self.assertIn(expected, result, f"Failed for: {text}")

    def test_encryption_roundtrip(self):
        from engine.safety.privacy import EncryptionManager
        em = EncryptionManager()
        if em.setup(password="test_password_123"):
            original = b"sensitive data here"
            encrypted = em.encrypt(original)
            self.assertNotEqual(encrypted, original)
            decrypted = em.decrypt(encrypted)
            self.assertEqual(decrypted, original)


@unittest.skipUnless(model_available(), SKIP_REASON)
class TestToolsFramework(unittest.TestCase):
    """Local tool calling."""

    def test_calculator(self):
        from engine.extensions.tools import ToolRegistry
        tr = ToolRegistry()
        self.assertEqual(tr.call("calculator", expression="100*3+7"), "307")

    def test_python_exec(self):
        from engine.extensions.tools import ToolRegistry
        tr = ToolRegistry()
        result = tr.call("python", code="print('hello')")
        self.assertEqual(result.strip(), "hello")

    def test_system_info(self):
        from engine.extensions.tools import ToolRegistry
        tr = ToolRegistry()
        result = tr.call("system_info")
        self.assertIn("CPU", result)
        self.assertIn("Memory", result)


class TestStorageUtils(unittest.TestCase):
    """Storage and hardware utilities (no model needed)."""

    def test_format_size(self):
        from engine.utils.storage import format_size
        self.assertIn("B", format_size(100))
        self.assertIn("KB", format_size(1500))
        self.assertIn("MB", format_size(2_000_000))
        self.assertIn("GB", format_size(3_000_000_000))

    def test_hardware_detect(self):
        from engine.utils.hardware import detect_hardware
        hw = detect_hardware()
        self.assertGreater(hw.cpu_count_physical, 0)
        self.assertGreater(hw.total_memory_mb, 0)
        self.assertGreater(hw.disk_free_gb, 0)

    def test_resource_monitor(self):
        from engine.utils.hardware import ResourceMonitor
        rm = ResourceMonitor()
        self.assertGreater(rm.memory_rss_mb, 0)


if __name__ == "__main__":
    unittest.main()
