"""Native full-precision inference engine wrapping llama-cpp-python.

Core architecture: Model is a FACTORY that populates the cache.
Cache is the PRODUCT that serves users at disk-IO speed.

The model is loaded ON-DEMAND only when cache misses occur,
and automatically unloaded after inference to free all resources.
When serving from cache: ZERO CPU, ZERO GPU, ZERO memory — only disk reads.
"""

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from engine.config import EngineConfig
from engine.core.model_validator import ModelValidationError, validate_model
from engine.core.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    seed: int = -1
    stop: List[str] = field(default_factory=list)


@contextlib.contextmanager
def _suppress_stderr():
    """Suppress C-level stderr (llama.cpp Metal kernel warnings)."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


class InferenceEngine:
    """Native full-precision inference engine with on-demand loading.

    Key design: The model is NOT loaded at startup. It loads only when
    a cache miss requires actual inference, then unloads automatically
    after a configurable idle timeout to free ALL resources.
    """

    def __init__(self, config: EngineConfig):
        self._config = config
        self._model = None
        self._tokenizer: Optional[Tokenizer] = None
        self._model_hash: Optional[str] = None
        self._model_metadata: Optional[Dict] = None
        self._model_path: Optional[str] = None
        self._lock = threading.RLock()
        self._loaded = False
        self._last_used: float = 0
        self._auto_unload_seconds = 300  # auto-unload after 5 min idle
        self._unload_timer: Optional[threading.Timer] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_hash(self) -> Optional[str]:
        return self._model_hash

    @property
    def tokenizer(self) -> Optional[Tokenizer]:
        return self._tokenizer

    @property
    def model_metadata(self) -> Optional[Dict]:
        return self._model_metadata

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def is_validated(self) -> bool:
        """Model file is validated but not necessarily loaded into memory."""
        return self._model_hash is not None

    def validate_only(self, model_path: Optional[str] = None) -> None:
        """Validate model without loading it. Zero resource usage."""
        path = model_path or self._config.model.path
        if not path:
            raise ModelValidationError("未指定模型路径")
        metadata, model_hash = validate_model(path)
        self._model_metadata = metadata
        self._model_hash = model_hash
        self._model_path = path
        logger.info(f"模型校验通过 (未加载到内存): {path}")

    def load(self, model_path: Optional[str] = None) -> None:
        """Load model into memory for inference. Called on-demand."""
        with self._lock:
            if self._loaded:
                self._touch()
                return

            path = model_path or self._model_path or self._config.model.path
            if not path:
                raise ModelValidationError("未指定模型路径")

            if not self._model_hash:
                self.validate_only(path)

            os.environ["GGML_METAL_LOG_LEVEL"] = "0"

            try:
                with _suppress_stderr():
                    from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python 未安装。请运行: pip install llama-cpp-python"
                )

            n_threads = self._config.model.n_threads
            n_ctx = self._config.model.n_ctx
            n_gpu = self._config.model.n_gpu_layers
            if n_gpu == -1:
                n_gpu = 999

            logger.info(f"按需加载模型到内存: {path}")

            with _suppress_stderr():
                self._model = Llama(
                    model_path=path,
                    n_ctx=n_ctx,
                    n_batch=self._config.model.n_batch,
                    n_threads=n_threads,
                    n_threads_batch=n_threads,
                    use_mmap=self._config.model.use_mmap,
                    use_mlock=False,
                    n_gpu_layers=n_gpu,
                    seed=self._config.model.seed if self._config.model.seed > 0 else 0xFFFFFFFF,
                    verbose=False,
                )

            if getattr(self._config.cache, "prefix_cache", True):
                try:
                    from llama_cpp import LlamaRAMCache
                    cap_mb = getattr(self._config.cache, "prefix_cache_capacity_mb", 512)
                    self._model.set_cache(LlamaRAMCache(capacity_bytes=cap_mb * 1024 * 1024))
                    logger.info(f"前缀缓存已启用 (容量 {cap_mb}MB)")
                except Exception as e:
                    logger.warning(f"前缀缓存启用失败: {e}")

            self._tokenizer = Tokenizer(self._model)
            self._loaded = True
            self._model_path = path
            self._touch()
            logger.info("模型加载完成")

    def unload(self) -> None:
        """Release model from memory. Keeps validation info for cache lookups."""
        with self._lock:
            if self._unload_timer:
                self._unload_timer.cancel()
                self._unload_timer = None
            if self._model is not None:
                del self._model
                self._model = None
            self._tokenizer = None
            self._loaded = False
            import gc
            gc.collect()
            logger.info("模型已从内存卸载 (缓存数据保留在磁盘)")

    def _touch(self):
        """Mark model as recently used, reset auto-unload timer."""
        self._last_used = time.time()
        if self._unload_timer:
            self._unload_timer.cancel()
        self._unload_timer = threading.Timer(
            self._auto_unload_seconds, self._auto_unload
        )
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _auto_unload(self):
        """Automatically unload model after idle timeout."""
        if self._loaded and time.time() - self._last_used >= self._auto_unload_seconds:
            logger.info(f"模型空闲{self._auto_unload_seconds}秒，自动卸载释放资源")
            self.unload()

    def ensure_loaded(self):
        """Load model if not already loaded. Called before inference."""
        if not self._loaded:
            self.load()

    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        self.ensure_loaded()
        return self._tokenizer.encode(text, add_bos=add_bos)

    def detokenize(self, tokens: List[int]) -> str:
        self.ensure_loaded()
        return self._tokenizer.decode(tokens)

    def generate_text(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
    ) -> Iterator[str]:
        self.ensure_loaded()
        if params is None:
            params = GenerationParams()

        with self._lock:
            self._touch()
            result = self._model.create_completion(
                prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                repeat_penalty=params.repeat_penalty,
                seed=params.seed if params.seed > 0 else None,
                stop=params.stop or None,
                stream=True,
            )
            for chunk in result:
                text = chunk["choices"][0]["text"]
                if text:
                    yield text

    def chat(
        self,
        messages: List[Dict],
        params: Optional[GenerationParams] = None,
    ) -> Iterator[str]:
        """Stream chat response. Model loaded on-demand if needed."""
        self.ensure_loaded()
        if params is None:
            params = GenerationParams()

        with self._lock:
            self._touch()
            result = self._model.create_chat_completion(
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                repeat_penalty=params.repeat_penalty,
                seed=params.seed if params.seed > 0 else None,
                stream=True,
            )
            for chunk in result:
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text

    def get_greedy_token(self, tokens: List[int]) -> int:
        self.ensure_loaded()
        with self._lock:
            self._touch()
            self._model.reset()
            self._model.eval(tokens)
            logits = np.array(
                self._model.scores[self._model.n_tokens - 1], dtype=np.float32
            )
            return int(np.argmax(logits))

    def get_top_k_logits(
        self, tokens: List[int], k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.ensure_loaded()
        with self._lock:
            self._touch()
            self._model.reset()
            self._model.eval(tokens)
            logits = np.array(
                self._model.scores[self._model.n_tokens - 1], dtype=np.float32
            )
            top_indices = np.argpartition(logits, -k)[-k:]
            top_indices = top_indices[np.argsort(logits[top_indices])[::-1]]
            top_values = logits[top_indices]
            return top_indices, top_values

    def get_embedding(self, tokens: List[int]) -> np.ndarray:
        self.ensure_loaded()
        with self._lock:
            self._touch()
            text = self._tokenizer.decode(tokens)
            embedding = self._model.embed(text)
            return np.array(embedding, dtype=np.float32)

    def save_state(self) -> bytes:
        self.ensure_loaded()
        return self._model.save_state().llama_state

    def load_state(self, state: bytes) -> None:
        self.ensure_loaded()
        from llama_cpp import LlamaState
        ls = LlamaState(
            input_ids=self._model._input_ids.copy(),
            scores=self._model._scores.copy(),
            n_tokens=self._model.n_tokens,
            llama_state=state,
            llama_state_size=len(state),
        )
        self._model.load_state(ls)

    def get_context_size(self) -> int:
        self.ensure_loaded()
        return self._model.n_ctx()

    def _sample_token(self, logits: np.ndarray, params: GenerationParams) -> int:
        if params.temperature <= 0:
            return int(np.argmax(logits))

        logits = logits / params.temperature

        if params.top_k > 0:
            top_k = min(params.top_k, len(logits))
            indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full(logits.shape, float("-inf"))
            mask[indices] = logits[indices]
            logits = mask

        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()

        if params.top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, params.top_p) + 1
            allowed = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[allowed] = probs[allowed]
            probs = mask / mask.sum()

        return int(np.random.choice(len(probs), p=probs))
