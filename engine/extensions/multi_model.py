"""Multi-model collaborative inference: expert pool, semantic routing, cross-validation.

Enables loading multiple full-precision expert models and routing queries
to the most suitable one. Supports cross-validation for complex queries.
All models must pass red-line validation (no quantization/distillation).
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from engine.config import EngineConfig
from engine.core.inference import GenerationParams, InferenceEngine
from engine.core.model_validator import validate_model

logger = logging.getLogger(__name__)


class ExpertDomain(Enum):
    GENERAL = "general"
    CODE = "code"
    MATH = "math"
    CREATIVE = "creative"
    LONG_CONTEXT = "long_context"
    TRANSLATION = "translation"


@dataclass
class ExpertModel:
    name: str
    path: str
    domain: ExpertDomain
    engine: Optional[InferenceEngine] = None
    model_hash: Optional[str] = None
    loaded: bool = False
    priority: int = 0  # higher = preferred


DOMAIN_KEYWORDS = {
    ExpertDomain.CODE: [
        "代码", "编程", "函数", "bug", "debug", "python", "java", "javascript",
        "code", "program", "implement", "class", "def ", "import ",
    ],
    ExpertDomain.MATH: [
        "计算", "数学", "公式", "方程", "概率", "统计", "积分", "微分",
        "calculate", "math", "equation", "probability", "algebra",
    ],
    ExpertDomain.CREATIVE: [
        "写作", "创作", "故事", "诗", "小说", "文案", "剧本",
        "write", "story", "poem", "creative", "fiction",
    ],
    ExpertDomain.TRANSLATION: [
        "翻译", "translate", "translation", "转换",
    ],
}


class MultiModelRouter:
    """Routes queries to the most suitable expert model."""

    def __init__(self, config: EngineConfig):
        self._config = config
        self._experts: Dict[str, ExpertModel] = {}
        self._default_expert: Optional[str] = None
        self._lock = threading.RLock()

    def register_expert(
        self,
        name: str,
        model_path: str,
        domain: ExpertDomain = ExpertDomain.GENERAL,
        priority: int = 0,
    ) -> bool:
        """Register an expert model (validation only, lazy loading)."""
        try:
            metadata, model_hash = validate_model(model_path)
            expert = ExpertModel(
                name=name,
                path=model_path,
                domain=domain,
                model_hash=model_hash,
                priority=priority,
            )
            self._experts[name] = expert
            if self._default_expert is None or domain == ExpertDomain.GENERAL:
                self._default_expert = name
            logger.info(f"注册专家模型: {name} ({domain.value})")
            return True
        except Exception as e:
            logger.error(f"专家模型注册失败 ({name}): {e}")
            return False

    def _load_expert(self, name: str) -> bool:
        """Lazy-load an expert model."""
        expert = self._experts.get(name)
        if expert is None:
            return False
        if expert.loaded:
            return True

        try:
            engine = InferenceEngine(self._config)
            engine.load(expert.path)
            expert.engine = engine
            expert.loaded = True
            logger.info(f"专家模型已加载: {name}")
            return True
        except Exception as e:
            logger.error(f"专家模型加载失败 ({name}): {e}")
            return False

    def route(self, query: str) -> str:
        """Determine which expert model should handle the query.

        Uses keyword-based routing with fallback to default.
        """
        query_lower = query.lower()
        best_domain = ExpertDomain.GENERAL
        best_score = 0

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_domain = domain

        for name, expert in self._experts.items():
            if expert.domain == best_domain:
                return name

        return self._default_expert or next(iter(self._experts), "")

    def generate(
        self,
        query: str,
        tokens: List[int],
        params: GenerationParams,
    ) -> Iterator[str]:
        """Route query to best expert and generate response."""
        expert_name = self.route(query)
        expert = self._experts.get(expert_name)

        if expert is None:
            logger.warning("无可用专家模型")
            return

        with self._lock:
            if not expert.loaded:
                if not self._load_expert(expert_name):
                    return

            logger.info(f"路由到专家模型: {expert_name} ({expert.domain.value})")
            yield from expert.engine.generate_text(
                expert.engine.detokenize(tokens), params
            )

    def cross_validate(
        self,
        query: str,
        tokens: List[int],
        params: GenerationParams,
        n_models: int = 2,
    ) -> List[Tuple[str, str]]:
        """Run query through multiple models for cross-validation.

        Returns list of (model_name, response) tuples.
        """
        results = []
        candidates = sorted(
            self._experts.values(),
            key=lambda e: e.priority,
            reverse=True,
        )[:n_models]

        for expert in candidates:
            if not expert.loaded:
                self._load_expert(expert.name)
            if not expert.loaded or expert.engine is None:
                continue

            try:
                response_parts = []
                for chunk in expert.engine.generate_text(
                    expert.engine.detokenize(tokens), params
                ):
                    response_parts.append(chunk)
                results.append((expert.name, "".join(response_parts)))
            except Exception as e:
                logger.warning(f"交叉验证失败 ({expert.name}): {e}")

        return results

    def unload_all(self):
        """Unload all expert models."""
        for expert in self._experts.values():
            if expert.loaded and expert.engine is not None:
                expert.engine.unload()
                expert.loaded = False

    @property
    def expert_names(self) -> List[str]:
        return list(self._experts.keys())

    @property
    def stats(self) -> Dict:
        return {
            "total_experts": len(self._experts),
            "loaded_experts": sum(1 for e in self._experts.values() if e.loaded),
            "experts": {
                name: {
                    "domain": e.domain.value,
                    "loaded": e.loaded,
                    "priority": e.priority,
                }
                for name, e in self._experts.items()
            },
        }
