"""Multimodal adaptation: image understanding, audio processing, multi-modal caching.

Supports LLaVA-series vision models and Whisper audio models.
All models use native full-precision weights (no quantization/distillation).
Multimodal results are cached for instant replay on repeated inputs.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import numpy as np

from engine.cache.base import CacheEntry
from engine.utils.hashing import hash_bytes
from engine.utils.storage import ensure_dir

if TYPE_CHECKING:
    from engine.config import EngineConfig
    from engine.core.inference import GenerationParams, InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class MultimodalInput:
    """Represents a multimodal input (image, audio, or video)."""
    input_type: str  # "image", "audio", "video"
    file_path: str
    file_hash: str = ""
    description: str = ""
    processed_text: str = ""
    cached: bool = False


class ImageProcessor:
    """Process images through LLaVA-compatible vision models."""

    def __init__(self, config: "EngineConfig"):
        self._config = config
        self._model = None
        self._loaded = False

    def load_model(self, model_path: str, clip_path: str = ""):
        """Load a LLaVA-compatible multimodal model."""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            chat_handler = Llava15ChatHandler(clip_model_path=clip_path) if clip_path else None

            self._model = Llama(
                model_path=model_path,
                n_ctx=self._config.model.n_ctx,
                n_threads=self._config.model.n_threads,
                use_mmap=True,
                n_gpu_layers=0,
                chat_handler=chat_handler,
                verbose=False,
            )
            self._loaded = True
            logger.info(f"视觉模型已加载: {model_path}")
        except ImportError:
            logger.warning("llama-cpp-python 多模态支持不可用")
        except Exception as e:
            logger.error(f"视觉模型加载失败: {e}")

    def process_image(
        self,
        image_path: str,
        prompt: str = "请描述这张图片的内容",
    ) -> str:
        """Process an image and return text description."""
        if not self._loaded or self._model is None:
            return "[视觉模型未加载]"

        if not os.path.exists(image_path):
            return f"[图片不存在: {image_path}]"

        try:
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            suffix = Path(image_path).suffix.lower()
            mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif", ".webp": "webp"}
            mime = mime_map.get(suffix, "jpeg")

            response = self._model.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{image_data}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=1024,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return f"[图像处理错误: {e}]"

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False


class AudioProcessor:
    """Process audio through Whisper-compatible models."""

    def __init__(self):
        self._model = None
        self._loaded = False

    def load_model(self, model_size: str = "base"):
        """Load a Whisper model for audio transcription."""
        try:
            import whisper
            self._model = whisper.load_model(model_size)
            self._loaded = True
            logger.info(f"音频模型已加载: whisper-{model_size}")
        except ImportError:
            try:
                from pywhispercpp.model import Model
                self._model = Model(model_size)
                self._loaded = True
                logger.info(f"音频模型已加载: pywhispercpp-{model_size}")
            except ImportError:
                logger.warning("whisper/pywhispercpp 未安装, 音频处理不可用")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        if not self._loaded or self._model is None:
            return "[音频模型未加载]"

        if not os.path.exists(audio_path):
            return f"[音频文件不存在: {audio_path}]"

        try:
            if hasattr(self._model, "transcribe"):
                result = self._model.transcribe(audio_path)
                if isinstance(result, dict):
                    return result.get("text", "")
                return str(result)
            else:
                segments = self._model.transcribe(audio_path)
                return " ".join(s.text for s in segments)
        except Exception as e:
            logger.error(f"音频转写失败: {e}")
            return f"[音频转写错误: {e}]"

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False


class MultimodalCache:
    """Cache for multimodal processing results to avoid redundant computation."""

    def __init__(self, cache_dir: str = "~/.cache/ai_engine/multimodal_cache"):
        self._cache_dir = os.path.expanduser(cache_dir)
        self._cache: Dict[str, str] = {}
        ensure_dir(self._cache_dir)

    def _file_hash(self, filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:32]

    def get(self, filepath: str) -> Optional[str]:
        file_hash = self._file_hash(filepath)
        if file_hash in self._cache:
            return self._cache[file_hash]

        cache_file = os.path.join(self._cache_dir, f"{file_hash}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                text = f.read()
            self._cache[file_hash] = text
            return text
        return None

    def put(self, filepath: str, result: str):
        file_hash = self._file_hash(filepath)
        self._cache[file_hash] = result
        cache_file = os.path.join(self._cache_dir, f"{file_hash}.txt")
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(result)

    def clear(self):
        self._cache.clear()
        import shutil
        if os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir, ignore_errors=True)
        ensure_dir(self._cache_dir)


class MultimodalManager:
    """Unified multimodal processing manager."""

    def __init__(self, config: "EngineConfig"):
        self._config = config
        self._image_processor = ImageProcessor(config)
        self._audio_processor = AudioProcessor()
        self._cache = MultimodalCache()

    def process_file(
        self,
        filepath: str,
        prompt: str = "",
    ) -> str:
        """Process any multimodal file, with caching."""
        cached = self._cache.get(filepath)
        if cached is not None:
            logger.info(f"多模态缓存命中: {filepath}")
            return cached

        suffix = Path(filepath).suffix.lower()

        if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            result = self._image_processor.process_image(filepath, prompt or "描述这张图片")
        elif suffix in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            result = self._audio_processor.transcribe(filepath)
        else:
            result = f"[不支持的文件类型: {suffix}]"

        if not result.startswith("["):
            self._cache.put(filepath, result)

        return result

    def load_vision_model(self, model_path: str, clip_path: str = ""):
        self._image_processor.load_model(model_path, clip_path)

    def load_audio_model(self, model_size: str = "base"):
        self._audio_processor.load_model(model_size)

    def unload_all(self):
        self._image_processor.unload()
        self._audio_processor.unload()

    @property
    def stats(self) -> Dict:
        return {
            "vision_loaded": self._image_processor._loaded,
            "audio_loaded": self._audio_processor._loaded,
            "cache_entries": len(self._cache._cache),
        }
