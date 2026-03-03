"""Configuration management with hardware auto-detection and validation."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from engine.utils.hardware import HardwareInfo, detect_hardware


@dataclass
class ModelConfig:
    path: str = ""
    allowed_precisions: list = field(default_factory=lambda: ["f16", "bf16"])
    n_ctx: int = 4096
    n_batch: int = 512
    n_threads: int = 0
    n_gpu_layers: int = -1  # -1 = auto (offload all to GPU on Apple Silicon/CUDA)
    use_mmap: bool = True
    seed: int = -1
    chat_template: str = ""  # auto-detect from model metadata


@dataclass
class L1CacheConfig:
    enabled: bool = True
    max_memory_mb: int = 100
    eviction: str = "lru"


@dataclass
class L2CacheConfig:
    enabled: bool = True
    max_memory_mb: int = 200
    eviction: str = "lfu"
    top_k_logits: int = 100


@dataclass
class L3CacheConfig:
    enabled: bool = False
    max_disk_gb: int = 50
    similarity_threshold: float = 0.98
    eviction: str = "lru_freq"


@dataclass
class L4CacheConfig:
    enabled: bool = True
    max_disk_gb: int = 100
    ttl_days: int = 90
    eviction: str = "ttl_lru"


@dataclass
class CacheConfig:
    base_dir: str = "~/.cache/ai_engine"
    l1: L1CacheConfig = field(default_factory=L1CacheConfig)
    l2: L2CacheConfig = field(default_factory=L2CacheConfig)
    l3: L3CacheConfig = field(default_factory=L3CacheConfig)
    l4: L4CacheConfig = field(default_factory=L4CacheConfig)

    # 三大缓存原则 (启用后覆盖多策略/归一化/语义)
    instruction_only: bool = True   # 不存整个会话，只认单次指令的完整内容
    token_level: bool = True        # 不存单次指令的整体，只存逐 token 计算过程 (KV/logits)
    strict_exact: bool = True       # 指令一字不差才命中；变一字则重新计算+存新缓存

    # 前缀缓存：长对话前面不变则前面全复用，减少重复计算
    prefix_cache: bool = True       # 启用 KV 前缀缓存（llama-cpp 内置）
    prefix_cache_capacity_mb: int = 512  # 前缀缓存容量 (MB)


@dataclass
class ResourceLimits:
    max_memory_mb: int = 2048
    max_cpu_percent: int = 20
    idle_cpu_percent: int = 3


@dataclass
class SafetyConfig:
    enabled: bool = True
    content_filter: bool = True
    privacy_encryption: bool = False
    encryption_key_file: str = ""


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "~/.cache/ai_engine/logs"
    max_days: int = 30
    audit_max_days: int = 90


@dataclass
class MonitoringConfig:
    enabled: bool = True
    stats_interval_seconds: int = 60


@dataclass
class EngineConfig:
    version: str = "0.1.0"
    name: str = "原生全精度大模型混合缓存加速引擎"
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    hardware: Optional[HardwareInfo] = field(default=None, repr=False)

    def auto_tune(self, hw: HardwareInfo):
        """Auto-tune configuration based on detected hardware."""
        self.hardware = hw
        if self.model.n_threads == 0:
            self.model.n_threads = hw.optimal_thread_count()
        ctx = hw.optimal_ctx_size(self.resource_limits.max_memory_mb)
        if self.model.n_ctx > ctx:
            self.model.n_ctx = ctx
        if hw.total_memory_mb < 16384:
            self.cache.l1.max_memory_mb = min(self.cache.l1.max_memory_mb, 50)
            self.cache.l2.max_memory_mb = min(self.cache.l2.max_memory_mb, 100)
        if hw.disk_free_gb < 50:
            self.cache.l4.max_disk_gb = min(self.cache.l4.max_disk_gb, int(hw.disk_free_gb * 0.3))
            self.cache.l3.max_disk_gb = min(self.cache.l3.max_disk_gb, int(hw.disk_free_gb * 0.2))


def _apply_dict(target, data: dict):
    """Recursively apply dict values to a dataclass instance."""
    for key, value in data.items():
        if hasattr(target, key):
            current = getattr(target, key)
            if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
                _apply_dict(current, value)
            else:
                setattr(target, key, value)


def load_config(config_path: Optional[str] = None) -> EngineConfig:
    """Load configuration from YAML file with hardware auto-detection."""
    config = EngineConfig()

    if config_path:
        path = Path(os.path.expanduser(config_path))
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if "engine" in raw:
                for k in ("version", "name"):
                    if k in raw["engine"]:
                        setattr(config, k, raw["engine"][k])
            if "model" in raw:
                _apply_dict(config.model, raw["model"])
            if "cache" in raw:
                _apply_dict(config.cache, raw["cache"])
            if "resource_limits" in raw:
                _apply_dict(config.resource_limits, raw["resource_limits"])
            if "safety" in raw:
                _apply_dict(config.safety, raw["safety"])
            if "logging" in raw:
                _apply_dict(config.logging, raw["logging"])
            if "monitoring" in raw:
                _apply_dict(config.monitoring, raw["monitoring"])

    hw = detect_hardware(config.cache.base_dir)
    config.auto_tune(hw)
    return config
