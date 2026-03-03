"""Display utilities for CLI output formatting."""

import sys
import time
from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

BANNER = """
====================================================
  原生全精度大模型混合缓存加速引擎 v{version}
====================================================
  三大红线承诺:
    [1] 不量化 - 全程原生BF16/FP16全精度
    [2] 不蒸馏 - 完整保留原生模型结构
    [3] 不减性能 - 输出与原生模型100%一致
====================================================
"""


def show_banner(version: str = "0.1.0"):
    console.print(BANNER.format(version=version), style="bold cyan")


def show_model_info(metadata: dict, model_hash: str):
    table = Table(title="模型信息", show_header=False)
    table.add_column("属性", style="bold")
    table.add_column("值")

    arch = metadata.get("general.architecture", "unknown")
    table.add_row("架构", arch)
    table.add_row("层数", str(metadata.get(f"{arch}.block_count", "N/A")))
    table.add_row("注意力头", str(metadata.get(f"{arch}.attention.head_count", "N/A")))
    table.add_row("嵌入维度", str(metadata.get(f"{arch}.embedding_length", "N/A")))
    table.add_row("模型哈希", model_hash[:16] + "...")

    ftype = metadata.get("general.file_type", "N/A")
    ftype_map = {0: "F32", 1: "F16", 26: "BF16"}
    table.add_row("精度类型", ftype_map.get(ftype, str(ftype)))

    console.print(table)


def show_hardware_info(hw_info):
    table = Table(title="硬件信息", show_header=False)
    table.add_column("属性", style="bold")
    table.add_column("值")
    table.add_row("CPU", f"{hw_info.cpu_count_physical}核 ({hw_info.cpu_count_logical}线程)")
    table.add_row("内存", f"{hw_info.total_memory_mb}MB (可用 {hw_info.available_memory_mb}MB)")
    table.add_row("磁盘", f"总计 {hw_info.disk_total_gb:.0f}GB (空闲 {hw_info.disk_free_gb:.0f}GB)")
    table.add_row("系统", f"{hw_info.os_name} {hw_info.arch}")
    console.print(table)


def show_status(stats: dict, resource_info: dict):
    table = Table(title="引擎状态", show_header=False)
    table.add_column("指标", style="bold")
    table.add_column("值")
    table.add_row("总请求数", str(stats.get("total_requests", 0)))
    table.add_row("缓存命中率", f"{stats.get('hit_rate', 0) * 100:.1f}%")
    table.add_row("L1缓存条目", str(stats.get("l1", {}).get("entries", 0)))
    table.add_row("L4缓存条目", str(stats.get("l4", {}).get("entries", 0)))
    table.add_row("内存占用", f"{resource_info.get('memory_rss_mb', 0):.1f}MB")
    table.add_row("CPU使用率", f"{resource_info.get('cpu_percent', 0):.1f}%")
    console.print(table)


def show_help():
    table = Table(title="常用指令", show_header=True)
    table.add_column("指令", style="bold cyan")
    table.add_column("说明")
    table.add_row("/help", "显示完整指令列表")
    table.add_row("/exit", "安全退出引擎")
    table.add_row("/clear", "清空当前对话上下文")
    table.add_row("/status", "查看引擎运行状态")
    table.add_row("/model [路径]", "加载/切换模型")
    table.add_row("/cache_hit", "查看缓存命中率")
    table.add_row("/cache_clear", "清理所有缓存")
    table.add_row("/semantic_on/off", "开启/关闭语义缓存")
    table.add_row("/temp [值]", "调整temperature")
    console.print(table)


def show_full_help(registry):
    """Show all commands grouped by category."""
    cats = registry.get_by_category()
    for cat_name, cmds in cats.items():
        table = Table(title=cat_name, show_header=True)
        table.add_column("指令", style="bold cyan", min_width=20)
        table.add_column("说明")
        for cmd in sorted(cmds, key=lambda c: c.name):
            table.add_row(cmd.usage, cmd.description)
        console.print(table)
    console.print()


def show_faq():
    """Display frequently asked questions."""
    faqs = [
        ("Q: 如何加载模型?", "A: /model /path/to/model.gguf (仅支持原生F16/BF16 GGUF)"),
        ("Q: 为什么模型加载失败?", "A: 请确认使用原生全精度GGUF, 量化模型会被拒绝"),
        ("Q: 如何提高命中率?", "A: 开启语义缓存 /semantic_on, 多使用相似问法"),
        ("Q: 语义缓存有误差吗?", "A: <=0.1%, 可 /semantic_off 切换0误差模式"),
        ("Q: 缓存占用太大?", "A: /cache_clear 清理, 或 /cache_limit 调整上限"),
        ("Q: 如何备份数据?", "A: /cache_backup [路径], /cache_restore <路径> 恢复"),
    ]
    table = Table(title="常见问题", show_header=False)
    table.add_column("", style="bold")
    table.add_column("")
    for q, a in faqs:
        table.add_row(q, a)
    console.print(table)


def stream_print(text: str, delay: float = 0):
    """Print text with optional character-by-character streaming effect."""
    sys.stdout.write(text)
    sys.stdout.flush()
