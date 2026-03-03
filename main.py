"""Entry point for the Native Full-Precision LLM Hybrid Cache Acceleration Engine.

Core philosophy: Cache is the product. Model is the factory.
- Startup: ZERO resource usage. Only validate model, open disk cache.
- Cache hit: Read from disk. No model, no GPU, no CPU load.
- Cache miss: Load model on-demand, infer, cache result, auto-unload.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["GGML_METAL_LOG_LEVEL"] = "0"

from engine.config import load_config
from engine.monitoring.logger import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="原生全精度大模型混合缓存加速引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
核心理念: 缓存即推理，硬盘即算力
  - 启动零资源: 不加载模型，仅打开磁盘缓存
  - 缓存命中: 纯硬盘读取，不消耗CPU/GPU/内存
  - 缓存未命中: 按需加载模型推理，完成后自动释放

缓存时机: 每句话发完就存内存，会话结束只存硬盘
  - 符合「逐 token 存/取」「只认单次指令」的核心设计
  - 缓存即时生效，不依赖会话是否结束

示例:
  python main.py --model /path/to/model.gguf
  python main.py --model /path/to/model.gguf --ctx-size 2048
  python main.py --model /path/to/model.gguf --gpu-layers 0  # 纯CPU
        """,
    )
    parser.add_argument("--model", "-m", type=str, default="",
                        help="GGUF模型文件路径 (仅校验不加载，按需时才加载)")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--ctx-size", "-n", type=int, default=0,
                        help="上下文窗口大小")
    parser.add_argument("--threads", "-t", type=int, default=0,
                        help="推理线程数 (0=自动)")
    parser.add_argument("--gpu-layers", "-ngl", type=int, default=-1,
                        help="GPU offload层数 (-1=全部, 0=纯CPU)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config.model.path = args.model
    if args.ctx_size > 0:
        config.model.n_ctx = args.ctx_size
    if args.threads > 0:
        config.model.n_threads = args.threads
    if args.gpu_layers != -1:
        config.model.n_gpu_layers = args.gpu_layers

    setup_logging(log_dir=config.logging.dir, level=config.logging.level)

    from engine.core.inference import InferenceEngine
    from engine.cache.scheduler import CacheScheduler
    from engine.cli.interface import CLIInterface

    engine = InferenceEngine(config)

    # Only VALIDATE model (check red lines), do NOT load into memory
    if config.model.path:
        try:
            engine.validate_only(config.model.path)
        except Exception as e:
            from rich.console import Console
            Console().print(f"[yellow]模型校验失败: {e}[/yellow]")

    scheduler = CacheScheduler(config, engine)
    cli = CLIInterface(config, engine, scheduler)
    cli.run()


if __name__ == "__main__":
    main()
