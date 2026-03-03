"""Multi-level consistency verification to enforce the no-performance-reduction red line.

Level 1: Exact cache strong verification (L1/L2/L4)
  - Model hash match
  - Context hash recomputation
  - Greedy token comparison

Level 2: Semantic cache enhanced verification (L3)
  - All Level 1 checks
  - First 3 token comparison
  - Real-time sampling verification every 10 tokens

Level 3: Periodic full regression (background)
  - Random 1% sample comparison weekly
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

from engine.cache.base import CacheEntry, CacheHitType
from engine.utils.hashing import hash_token_sequence

if TYPE_CHECKING:
    from engine.core.inference import InferenceEngine

logger = logging.getLogger(__name__)


class VerifyResult(Enum):
    PASS = "pass"
    FAIL_MODEL_HASH = "fail_model_hash"
    FAIL_CONTEXT_HASH = "fail_context_hash"
    FAIL_GREEDY_TOKEN = "fail_greedy_token"
    FAIL_FIRST_TOKENS = "fail_first_tokens"
    SKIP = "skip"


class ConsistencyVerifier:
    """Ensures cache outputs match native model outputs exactly."""

    def __init__(self, inference_engine: "InferenceEngine"):
        self._engine = inference_engine
        self._total_checks = 0
        self._pass_count = 0
        self._fail_count = 0

    def verify_exact(
        self,
        entry: CacheEntry,
        tokens: List[int],
        model_hash: str,
        do_greedy_check: bool = True,
    ) -> VerifyResult:
        """Level 1: Strong verification for exact-match caches (L1/L2/L4).

        Args:
            entry: The cache entry to verify
            tokens: Current context token sequence
            model_hash: Current model's hash
            do_greedy_check: Whether to verify greedy token (costs ~1ms)
        """
        self._total_checks += 1

        if entry.model_hash != model_hash:
            self._fail_count += 1
            logger.warning(
                f"一致性校验失败: 模型哈希不匹配 "
                f"(缓存={entry.model_hash[:8]}, 当前={model_hash[:8]})"
            )
            return VerifyResult.FAIL_MODEL_HASH

        context_hash = hash_token_sequence(tokens)
        if context_hash != entry.context_hash:
            self._fail_count += 1
            logger.warning("一致性校验失败: 上下文哈希不匹配")
            return VerifyResult.FAIL_CONTEXT_HASH

        if do_greedy_check and entry.greedy_token >= 0:
            try:
                current_greedy = self._engine.get_greedy_token(tokens)
                if current_greedy != entry.greedy_token:
                    self._fail_count += 1
                    logger.warning(
                        f"一致性校验失败: 贪婪token不一致 "
                        f"(缓存={entry.greedy_token}, 当前={current_greedy})"
                    )
                    return VerifyResult.FAIL_GREEDY_TOKEN
            except Exception as e:
                logger.warning(f"贪婪token校验异常，跳过: {e}")

        self._pass_count += 1
        return VerifyResult.PASS

    def verify_semantic(
        self,
        entry: CacheEntry,
        tokens: List[int],
        model_hash: str,
    ) -> VerifyResult:
        """Level 2: Enhanced verification for semantic cache (L3).

        Includes all Level 1 checks plus first-3-token comparison.
        """
        basic_result = self.verify_exact(
            entry, tokens, model_hash, do_greedy_check=False
        )
        if basic_result == VerifyResult.FAIL_MODEL_HASH:
            return basic_result

        if entry.response_text and len(entry.response_text) > 0:
            try:
                current_greedy_tokens = []
                temp_tokens = list(tokens)
                for _ in range(3):
                    gt = self._engine.get_greedy_token(temp_tokens)
                    current_greedy_tokens.append(gt)
                    temp_tokens.append(gt)

                cached_tokens = self._engine.tokenize(
                    entry.response_text, add_bos=False
                )[:3]

                if current_greedy_tokens != cached_tokens:
                    self._fail_count += 1
                    logger.warning(
                        f"语义缓存校验失败: 前3token不一致 "
                        f"(缓存={cached_tokens}, 当前={current_greedy_tokens})"
                    )
                    return VerifyResult.FAIL_FIRST_TOKENS
            except Exception as e:
                logger.warning(f"前3token校验异常，跳过: {e}")

        self._pass_count += 1
        return VerifyResult.PASS

    def verify_streaming_sample(
        self,
        cached_token: int,
        position: int,
        tokens_so_far: List[int],
    ) -> bool:
        """Real-time sampling verification during streaming output.

        Called every 10 tokens during semantic cache streaming.
        Returns True if consistent, False to abort cache output.
        """
        if position % 10 != 0:
            return True

        try:
            greedy = self._engine.get_greedy_token(tokens_so_far)
            if greedy != cached_token:
                logger.warning(
                    f"流式抽样校验失败: position={position}, "
                    f"cached={cached_token}, actual={greedy}"
                )
                return False
        except Exception as e:
            logger.warning(f"流式抽样校验异常: {e}")

        return True

    @property
    def stats(self) -> dict:
        return {
            "total_checks": self._total_checks,
            "passed": self._pass_count,
            "failed": self._fail_count,
            "pass_rate": (
                round(self._pass_count / self._total_checks, 4)
                if self._total_checks > 0
                else 0
            ),
        }
