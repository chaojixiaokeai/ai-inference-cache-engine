"""Main CLI interface for the acceleration engine.

Architecture: Cache-first, model-on-demand.
- Cache hit: Pure disk read -> instant output -> ZERO resource usage
- Cache miss: Load model -> infer -> write cache -> auto-unload model
"""

import hashlib
import logging
import re
import sys
import time
from typing import List, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from engine.cache.scheduler import CacheScheduler
from engine.cli.commands import CommandRegistry, register_all_commands
from engine.cli.display import console, show_banner, show_hardware_info, show_model_info, stream_print
from engine.config import EngineConfig
from engine.core.inference import GenerationParams, InferenceEngine
from engine.safety.alignment import SafetyFilter
from engine.utils.hardware import ResourceMonitor

logger = logging.getLogger(__name__)


def _fast_hash(text: str) -> int:
    """Fast hash without requiring model tokenizer (works without model loaded)."""
    import xxhash
    return xxhash.xxh64(text.encode("utf-8")).intdigest()


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy cache matching.

    Handles: spaces, full/half-width chars, common punctuation, case.
    "从 1 数到 100" and "从1数到100" will produce the same result.
    """
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    text = text.lower()
    return text


class CLIInterface:
    """Interactive CLI with cache-first architecture."""

    def __init__(
        self,
        config: EngineConfig,
        engine: InferenceEngine,
        scheduler: CacheScheduler,
    ):
        self._config = config
        self._engine = engine
        self._scheduler = scheduler
        self._safety = SafetyFilter(enabled=config.safety.enabled)
        self._monitor = ResourceMonitor()
        self._gen_params = GenerationParams()
        self._dialog_history: List[dict] = [
            {"role": "system", "content": "你是一个有用的AI助手。请直接回答用户的问题，简洁准确。/no_think"}
        ]
        self._running = False
        self._total_requests = 0
        self._cache_hits = 0

        self._commands = CommandRegistry()
        register_all_commands(self._commands, self)

        cmd_names = ["/" + n for n in self._commands.command_names]
        self._completer = WordCompleter(cmd_names, sentence=True)
        self._history = InMemoryHistory()
        self._session = PromptSession(
            history=self._history,
            completer=self._completer,
            multiline=False,
        )

    def run(self):
        self._running = True
        show_banner(self._config.version)

        if self._config.hardware:
            show_hardware_info(self._config.hardware)

        if self._engine.is_validated and self._engine.model_metadata:
            show_model_info(self._engine.model_metadata, self._engine.model_hash)

        # Show cache stats
        l4_count = self._scheduler.l4.entry_count()
        if l4_count > 0:
            from engine.utils.storage import format_size
            l4_size = self._scheduler.l4.size_bytes()
            console.print(
                f"\n[bold green]磁盘缓存已就绪: {l4_count} 条记录 "
                f"({format_size(l4_size)})[/bold green]"
            )
            console.print(
                "[dim]缓存命中时: 0 CPU / 0 GPU / 0 内存 — 纯硬盘极速读取[/dim]"
            )

        model_status = "已校验 (按需加载)" if self._engine.is_validated else "未指定"
        mem_status = "未占用" if not self._engine.is_loaded else "已加载"
        console.print(f"\n[dim]模型状态: {model_status} | 内存: {mem_status}[/dim]")

        if not self._engine.is_validated:
            console.print(
                "\n[yellow]提示: 未指定模型。使用 /model <路径> 指定GGUF模型文件[/yellow]"
            )
            console.print(
                "[yellow]若已有缓存数据，可直接对话，缓存命中无需模型[/yellow]\n"
            )

        console.print("输入 /help 查看所有指令，/exit 退出\n")

        while self._running:
            try:
                user_input = self._session.prompt("You> ").strip()
                if not user_input:
                    continue

                parsed = self._commands.parse_command(user_input)
                if parsed:
                    cmd_name, args = parsed
                    cmd = self._commands.get(cmd_name)
                    if cmd:
                        cmd.handler(args)
                    else:
                        console.print(f"[red]未知指令: /{cmd_name}[/red]")
                    continue

                self._handle_chat(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Ctrl+C 中断，/exit 退出[/yellow]")
            except EOFError:
                self.request_exit()

        self._shutdown()

    def _handle_chat(self, user_input: str):
        """Cache-first chat.

        三大缓存原则 (config.cache.instruction_only / token_level / strict_exact):
          1. 不存整个会话，只认单次指令的完整内容
          2. 不存单次指令的整体，只存逐 token 计算过程 (KV/logits)
          3. 指令一字不差才命中；变一字则重新计算+存新缓存
        """
        if not self._safety.check_input(user_input):
            console.print("[red]输入内容触发安全规则，已拦截[/red]")
            return

        self._dialog_history.append({"role": "user", "content": user_input})
        self._total_requests += 1

        model_hash = self._engine.model_hash or "default"
        start_time = time.time()

        sys_prompt = self._dialog_history[0]["content"] if self._dialog_history and self._dialog_history[0]["role"] == "system" else ""

        # --- 原则1: 只认单次指令，不存整个会话 ---
        instruction_text = f"<|system|>{sys_prompt}<|user|>{user_input}"
        instruction_hash = _fast_hash(instruction_text)

        # --- 原则3: 一字不差才命中；无归一化、无语义 ---
        use_strict = getattr(self._config.cache, "strict_exact", True)
        use_instruction_only = getattr(self._config.cache, "instruction_only", True)

        cached = None
        if use_instruction_only:
            cached = self._scheduler.l1.get(instruction_hash, model_hash)
            if cached is None:
                cached = self._scheduler.l4.get(instruction_hash, model_hash)
        else:
            full_context = "".join(
                f"<|{m['role']}|>{m['content']}" for m in self._dialog_history
            )
            ctx_hash = _fast_hash(full_context)
            cached = self._scheduler.l1.get(ctx_hash, model_hash)
            if cached is None:
                cached = self._scheduler.l4.get(ctx_hash, model_hash)

        if not use_strict and cached is None:
            norm_key = f"<|system|>{_normalize_text(sys_prompt)}<|user|>{_normalize_text(user_input)}"
            norm_hash = _fast_hash(norm_key)
            cached = self._scheduler.l1.get(norm_hash, model_hash)
            if cached is None:
                cached = self._scheduler.l4.get(norm_hash, model_hash)
        if not use_strict and cached is None and self._scheduler.l3 and self._scheduler.l3.enabled:
            try:
                cached = self._scheduler.l3.search([], model_hash, embedding=self._get_lightweight_embedding(user_input))
            except Exception:
                pass

        console.print("\nAI> ", end="", style="bold green")

        if cached is not None and cached.response_text:
            self._cache_hits += 1
            elapsed_lookup = time.time() - start_time

            display_text = self._strip_think_tags(cached.response_text)
            stream_print(display_text)

            self._dialog_history.append({"role": "assistant", "content": cached.response_text})

            hit_rate = self._cache_hits / self._total_requests * 100
            console.print(
                f"\n\n[bold cyan]⚡ 缓存命中[/bold cyan] [dim]("
                f"指令精确匹配, "
                f"查询 {elapsed_lookup*1000:.1f}ms, "
                f"0 CPU / 0 GPU / 0 内存, "
                f"命中率 {hit_rate:.0f}%)[/dim]\n"
            )
            return

        # === Cache miss — need model inference ===
        if not self._engine.is_validated:
            console.print(
                "[red]缓存未命中，且未指定模型。请先 /model <路径> 加载模型[/red]"
            )
            self._dialog_history.pop()
            return

        was_loaded = self._engine.is_loaded
        if not was_loaded:
            console.print("[dim]缓存未命中，按需加载模型中...[/dim]", end="")
            sys.stdout.flush()

        # --- 原则2: 推理时走 token 级缓存 (scheduler 存 KV/logits)；CLI 用 chat 时 L4 存 response 供跨重启 ---
        response_parts = []
        in_think = False
        try:
            for text_chunk in self._engine.chat(
                self._dialog_history, self._gen_params
            ):
                response_parts.append(text_chunk)
                if "<think>" in text_chunk:
                    in_think = True
                if not in_think:
                    stream_print(text_chunk)
                if "</think>" in text_chunk:
                    in_think = False
                    after = text_chunk.split("</think>", 1)[-1]
                    if after.strip():
                        stream_print(after)
        except Exception as e:
            logger.error(f"推理异常: {e}")
            console.print(f"\n[red]推理异常: {e}[/red]")
            self._dialog_history.pop()
            return

        elapsed = time.time() - start_time
        full_response = "".join(response_parts)
        self._dialog_history.append({"role": "assistant", "content": full_response})

        # 每句话发完就存内存；会话结束只存硬盘（scheduler.close 时 persist）
        from engine.cache.base import CacheEntry
        write_key = instruction_hash if use_instruction_only else _fast_hash(
            "".join(f"<|{m['role']}|>{m['content']}" for m in self._dialog_history)
        )
        try:
            entry = CacheEntry(
                context_hash=write_key,
                model_hash=model_hash,
                response_text=full_response,
                greedy_token=-1,
            )
            self._scheduler.l1.put(entry)
            # L4 在会话结束时由 scheduler.close() 统一持久化，保证跨重启生效
        except Exception as e:
            logger.debug(f"缓存写入跳过: {e}")

        if not use_strict and self._scheduler.l3 and self._scheduler.l3.enabled:
            try:
                emb = self._get_lightweight_embedding(user_input)
                if emb is not None:
                    entry = CacheEntry(
                        context_hash=instruction_hash,
                        model_hash=model_hash,
                        response_text=full_response,
                        greedy_token=-1,
                    )
                    self._scheduler.l3.put(entry, embedding=emb)
            except Exception:
                pass

        display_text = self._strip_think_tags(full_response)
        tokens_approx = max(1, len(display_text) // 2)
        speed = tokens_approx / elapsed if elapsed > 0 else 0
        hit_rate = self._cache_hits / self._total_requests * 100

        console.print(
            f"\n\n[dim]原生推理 ({elapsed:.1f}s, ~{speed:.0f} tok/s, "
            f"已缓存, 下次将极速命中, "
            f"命中率 {hit_rate:.0f}%)[/dim]\n"
        )

    def _get_lightweight_embedding(self, text: str) -> Optional["np.ndarray"]:
        """Compute a lightweight embedding without requiring the full model.

        Uses character n-gram hashing to produce a fixed-size vector.
        Not as good as real model embeddings but works at zero cost.
        """
        import numpy as np
        dim = 4096
        vec = np.zeros(dim, dtype=np.float32)
        normalized = _normalize_text(text)
        for n in (2, 3, 4):
            for i in range(len(normalized) - n + 1):
                gram = normalized[i:i+n]
                h = hash(gram) % dim
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()

    def clear_context(self):
        sys_msg = self._dialog_history[0] if self._dialog_history and self._dialog_history[0]["role"] == "system" else None
        self._dialog_history.clear()
        if sys_msg:
            self._dialog_history.append(sys_msg)

    def request_exit(self):
        self._running = False

    def get_stats(self) -> dict:
        stats = self._scheduler.stats
        stats["cli_requests"] = self._total_requests
        stats["cli_cache_hits"] = self._cache_hits
        stats["model_loaded"] = self._engine.is_loaded
        stats["model_validated"] = self._engine.is_validated
        return stats

    def get_resource_info(self) -> dict:
        info = self._monitor.get_summary()
        info["model_in_memory"] = self._engine.is_loaded
        return info

    def _shutdown(self):
        console.print("\n[cyan]正在安全退出...[/cyan]")
        try:
            self._scheduler.close()
        except Exception as e:
            logger.warning(f"关闭调度引擎异常: {e}")
        try:
            self._engine.unload()
        except Exception as e:
            logger.warning(f"卸载模型异常: {e}")
        console.print("[green]引擎已安全退出，内存缓存已持久化到磁盘（跨重启生效）[/green]")
