"""Safety alignment module.

Phase 1: Lightweight approach reusing the model's native RLHF alignment.
Basic content filtering for cache write protection.
Phase 2+: Full Llama Guard integration.
"""

import logging
import re
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

UNSAFE_PATTERNS = [
    r"(?i)(如何|怎么|怎样)(制造|制作|合成)(炸弹|爆炸物|毒品|武器)",
    r"(?i)(hack|exploit|crack)\s+(password|system|bank)",
    r"(?i)self[- ]?harm",
    r"(?i)suicide\s+(method|how)",
]

_compiled_patterns = [re.compile(p) for p in UNSAFE_PATTERNS]


class SafetyFilter:
    """Content safety filter for cache and output protection."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._blocked_count = 0

    def check_input(self, text: str) -> bool:
        """Check if input text is safe. Returns True if safe."""
        if not self.enabled:
            return True
        for pattern in _compiled_patterns:
            if pattern.search(text):
                self._blocked_count += 1
                logger.warning("安全过滤: 输入内容触发安全规则")
                return False
        return True

    def check_output(self, text: str) -> bool:
        """Check if output text is safe for caching and display."""
        if not self.enabled:
            return True
        for pattern in _compiled_patterns:
            if pattern.search(text):
                self._blocked_count += 1
                logger.warning("安全过滤: 输出内容触发安全规则")
                return False
        return True

    def filter_for_cache(self, text: str) -> Optional[str]:
        """Filter text before writing to cache. Returns None if unsafe."""
        if self.check_output(text):
            return text
        return None

    @property
    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "blocked_count": self._blocked_count,
        }
