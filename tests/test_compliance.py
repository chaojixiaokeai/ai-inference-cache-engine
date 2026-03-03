"""Red-line compliance tests: ensures the three inviolable rules are never broken.

1. No Quantization: Only F16/BF16/F32 models accepted
2. No Distillation: Complete model structure preserved
3. No Performance Reduction: Output matches native model exactly
"""

import os
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from engine.core.model_validator import (
    ALLOWED_FTYPE_VALUES,
    GGUF_MAGIC,
    ModelValidationError,
    validate_model,
)

# GGUF magic bytes: 'G','G','U','F' -> little-endian uint32 = 0x46554747
from engine.utils.hashing import hash_token_sequence


class TestNoQuantizationRedLine(unittest.TestCase):
    """Verify that quantized models are rejected."""

    def _create_fake_gguf(self, ftype: int, path: str):
        """Create a minimal fake GGUF file with given file_type."""
        with open(path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))  # magic
            f.write(struct.pack("<I", 3))  # version
            f.write(struct.pack("<Q", 0))  # n_tensors
            f.write(struct.pack("<Q", 1))  # n_kv (1 metadata entry)

            key = "general.file_type"
            f.write(struct.pack("<Q", len(key)))
            f.write(key.encode("utf-8"))
            f.write(struct.pack("<I", 4))  # type = UINT32
            f.write(struct.pack("<I", ftype))

    def test_f16_accepted(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            self._create_fake_gguf(1, tmp.name)
            try:
                metadata, model_hash = validate_model(tmp.name)
                self.assertEqual(metadata.get("general.file_type"), 1)
                self.assertTrue(len(model_hash) > 0)
            finally:
                os.unlink(tmp.name)

    def test_bf16_accepted(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            self._create_fake_gguf(26, tmp.name)
            try:
                metadata, _ = validate_model(tmp.name)
                self.assertEqual(metadata.get("general.file_type"), 26)
            finally:
                os.unlink(tmp.name)

    def test_f32_accepted(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            self._create_fake_gguf(0, tmp.name)
            try:
                metadata, _ = validate_model(tmp.name)
                self.assertEqual(metadata.get("general.file_type"), 0)
            finally:
                os.unlink(tmp.name)

    def test_q4_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            self._create_fake_gguf(2, tmp.name)
            try:
                with self.assertRaises(ModelValidationError) as ctx:
                    validate_model(tmp.name)
                self.assertIn("不量化红线", str(ctx.exception))
            finally:
                os.unlink(tmp.name)

    def test_q8_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            self._create_fake_gguf(7, tmp.name)
            try:
                with self.assertRaises(ModelValidationError) as ctx:
                    validate_model(tmp.name)
                self.assertIn("不量化红线", str(ctx.exception))
            finally:
                os.unlink(tmp.name)

    def test_non_gguf_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(b"not a gguf file")
            tmp.flush()
            try:
                with self.assertRaises(ModelValidationError):
                    validate_model(tmp.name)
            finally:
                os.unlink(tmp.name)


class TestHashConsistency(unittest.TestCase):
    """Verify hash functions produce deterministic results."""

    def test_same_tokens_same_hash(self):
        tokens = [1, 2, 3, 4, 5]
        h1 = hash_token_sequence(tokens)
        h2 = hash_token_sequence(tokens)
        self.assertEqual(h1, h2)

    def test_different_tokens_different_hash(self):
        h1 = hash_token_sequence([1, 2, 3])
        h2 = hash_token_sequence([4, 5, 6])
        self.assertNotEqual(h1, h2)

    def test_order_matters(self):
        h1 = hash_token_sequence([1, 2, 3])
        h2 = hash_token_sequence([3, 2, 1])
        self.assertNotEqual(h1, h2)


class TestCacheConsistencyVerification(unittest.TestCase):
    """Verify consistency checks work correctly."""

    def test_model_hash_mismatch_detected(self):
        from engine.cache.base import CacheEntry
        from engine.verification.consistency import ConsistencyVerifier, VerifyResult

        mock_engine = MagicMock()
        verifier = ConsistencyVerifier(mock_engine)

        entry = CacheEntry(
            context_hash=hash_token_sequence([1, 2, 3]),
            model_hash="hash_a",
            greedy_token=42,
        )

        result = verifier.verify_exact(entry, [1, 2, 3], "hash_b")
        self.assertEqual(result, VerifyResult.FAIL_MODEL_HASH)

    def test_context_hash_mismatch_detected(self):
        from engine.cache.base import CacheEntry
        from engine.verification.consistency import ConsistencyVerifier, VerifyResult

        mock_engine = MagicMock()
        verifier = ConsistencyVerifier(mock_engine)

        entry = CacheEntry(
            context_hash=hash_token_sequence([1, 2, 3]),
            model_hash="hash_a",
            greedy_token=42,
        )

        # Different tokens but same model hash
        result = verifier.verify_exact(entry, [4, 5, 6], "hash_a")
        self.assertEqual(result, VerifyResult.FAIL_CONTEXT_HASH)

    def test_correct_entry_passes(self):
        from engine.cache.base import CacheEntry
        from engine.verification.consistency import ConsistencyVerifier, VerifyResult

        mock_engine = MagicMock()
        mock_engine.get_greedy_token.return_value = 42
        verifier = ConsistencyVerifier(mock_engine)

        tokens = [1, 2, 3]
        entry = CacheEntry(
            context_hash=hash_token_sequence(tokens),
            model_hash="hash_a",
            greedy_token=42,
        )

        result = verifier.verify_exact(entry, tokens, "hash_a")
        self.assertEqual(result, VerifyResult.PASS)


if __name__ == "__main__":
    unittest.main()
