"""Full command system for the CLI interface (7 categories, 40+ commands)."""

import logging
import sys
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from engine.cli.interface import CLIInterface

logger = logging.getLogger(__name__)


class Command:
    def __init__(self, name: str, description: str, handler: Callable,
                 category: str = "基础", usage: str = ""):
        self.name = name
        self.description = description
        self.handler = handler
        self.category = category
        self.usage = usage or f"/{name}"


class CommandRegistry:
    """Registry and dispatcher for CLI commands."""

    def __init__(self):
        self._commands: Dict[str, Command] = {}

    def register(self, name: str, description: str, handler: Callable,
                 category: str = "基础", usage: str = ""):
        self._commands[name] = Command(name, description, handler, category, usage)

    def get(self, name: str) -> Optional[Command]:
        return self._commands.get(name)

    def get_all(self) -> Dict[str, Command]:
        return self._commands.copy()

    def get_by_category(self) -> Dict[str, list]:
        cats: Dict[str, list] = {}
        for cmd in self._commands.values():
            cats.setdefault(cmd.category, []).append(cmd)
        return cats

    def parse_command(self, text: str) -> Optional[Tuple[str, str]]:
        text = text.strip()
        if not text.startswith("/"):
            return None
        parts = text[1:].split(maxsplit=1)
        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        return cmd_name, args

    @property
    def command_names(self):
        return list(self._commands.keys())


def register_all_commands(registry: CommandRegistry, cli: "CLIInterface"):
    """Register all commands across 6 categories."""
    from engine.cli.display import console

    # ===== Category 1: Basic Operations =====
    def cmd_help(args: str):
        from engine.cli.display import show_full_help
        show_full_help(registry)

    def cmd_exit(args: str):
        cli.request_exit()

    def cmd_clear(args: str):
        cli.clear_context()
        console.print("[green]对话上下文已清空[/green]")

    def cmd_history(args: str):
        if not cli._dialog_history:
            console.print("[dim]暂无对话历史[/dim]")
            return
        for i, msg in enumerate(cli._dialog_history[-20:]):
            role = "You" if msg["role"] == "user" else "AI"
            text = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
            console.print(f"  [{i+1}] {role}: {text}")

    def cmd_seed(args: str):
        if args.strip():
            try:
                cli._gen_params.seed = int(args.strip())
                console.print(f"[green]随机种子: {cli._gen_params.seed}[/green]")
            except ValueError:
                console.print("[red]请输入整数[/red]")
        else:
            console.print(f"当前随机种子: {cli._gen_params.seed}")

    registry.register("help", "显示所有指令", cmd_help, "基础操作")
    registry.register("exit", "安全退出", cmd_exit, "基础操作")
    registry.register("quit", "安全退出", cmd_exit, "基础操作")
    registry.register("clear", "清空上下文", cmd_clear, "基础操作")
    registry.register("history", "查看对话历史", cmd_history, "基础操作")
    registry.register("seed", "设置随机种子", cmd_seed, "基础操作", "/seed [整数]")

    # ===== Category 2: Config Adjustment =====
    def cmd_temp(args: str):
        if args.strip():
            try:
                val = float(args)
                cli._gen_params.temperature = max(0, min(2.0, val))
                console.print(f"[green]temperature={cli._gen_params.temperature}[/green]")
            except ValueError:
                console.print("[red]请输入0-2的浮点数[/red]")
        else:
            console.print(f"temperature={cli._gen_params.temperature}")

    def cmd_topk(args: str):
        if args.strip():
            try:
                cli._gen_params.top_k = max(0, int(args))
                console.print(f"[green]top_k={cli._gen_params.top_k}[/green]")
            except ValueError:
                console.print("[red]请输入正整数[/red]")
        else:
            console.print(f"top_k={cli._gen_params.top_k}")

    def cmd_topp(args: str):
        if args.strip():
            try:
                val = float(args)
                cli._gen_params.top_p = max(0, min(1.0, val))
                console.print(f"[green]top_p={cli._gen_params.top_p}[/green]")
            except ValueError:
                console.print("[red]请输入0-1的浮点数[/red]")
        else:
            console.print(f"top_p={cli._gen_params.top_p}")

    def cmd_max_tokens(args: str):
        if args.strip():
            try:
                cli._gen_params.max_tokens = max(1, int(args))
                console.print(f"[green]max_tokens={cli._gen_params.max_tokens}[/green]")
            except ValueError:
                console.print("[red]请输入正整数[/red]")
        else:
            console.print(f"max_tokens={cli._gen_params.max_tokens}")

    def cmd_model(args: str):
        if not args.strip():
            if cli._engine.is_loaded:
                from engine.cli.display import show_model_info
                show_model_info(cli._engine.model_metadata, cli._engine.model_hash)
            else:
                console.print("[yellow]未加载模型[/yellow]")
            return
        try:
            cli._engine.load(args.strip())
            from engine.cli.display import show_model_info
            show_model_info(cli._engine.model_metadata, cli._engine.model_hash)
        except Exception as e:
            console.print(f"[red]模型加载失败: {e}[/red]")

    def cmd_cache_limit(args: str):
        parts = args.strip().split()
        if len(parts) == 2:
            level, value = parts[0].upper(), parts[1]
            try:
                val = int(value)
                if level == "L1":
                    cli._scheduler.l1._max_bytes = val * 1024 * 1024
                    console.print(f"[green]L1 上限={val}MB[/green]")
                elif level == "L4":
                    console.print(f"[green]L4 上限设置需重启生效[/green]")
                else:
                    console.print("[red]可选: L1, L4[/red]")
            except ValueError:
                console.print("[red]请输入整数(MB)[/red]")
        else:
            console.print("用法: /cache_limit <L1|L4> <MB>")

    registry.register("temp", "调整temperature", cmd_temp, "配置调整", "/temp [0-2]")
    registry.register("topk", "调整top-k", cmd_topk, "配置调整", "/topk [整数]")
    registry.register("topp", "调整top-p", cmd_topp, "配置调整", "/topp [0-1]")
    registry.register("max_tokens", "设置最大生成长度", cmd_max_tokens, "配置调整", "/max_tokens [整数]")
    registry.register("model", "加载/查看模型", cmd_model, "配置调整", "/model [路径]")
    registry.register("cache_limit", "调整缓存上限", cmd_cache_limit, "配置调整", "/cache_limit <L1|L4> <MB>")

    # ===== Category 3: Cache Management =====
    def cmd_cache_hit(args: str):
        stats = cli.get_stats()
        console.print(f"全局命中率: {stats.get('hit_rate', 0)*100:.1f}%")
        console.print(f"L1 命中率: {stats.get('l1',{}).get('hit_rate',0)*100:.1f}% ({stats.get('l1',{}).get('entries',0)} 条)")
        if "l2" in stats:
            console.print(f"L2 命中率: {stats['l2'].get('hit_rate',0)*100:.1f}% ({stats['l2'].get('entries',0)} 条)")
        console.print(f"L4 命中率: {stats.get('l4',{}).get('hit_rate',0)*100:.1f}% ({stats.get('l4',{}).get('entries',0)} 条)")
        if "l3" in stats:
            console.print(f"L3 命中率: {stats['l3'].get('hit_rate',0)*100:.1f}% ({stats['l3'].get('entries',0)} 条)")

    def cmd_cache_clear(args: str):
        cli._scheduler.clear_all()
        console.print("[green]所有缓存已清空[/green]")

    def cmd_cache_backup(args: str):
        try:
            from engine.cache.lifecycle import LifecycleManager
            lm = LifecycleManager(cli._config, cli._scheduler)
            path = lm.backup(args.strip() if args.strip() else None)
            console.print(f"[green]备份完成: {path}[/green]")
        except Exception as e:
            console.print(f"[red]备份失败: {e}[/red]")

    def cmd_cache_restore(args: str):
        if not args.strip():
            console.print("用法: /cache_restore <备份路径>")
            return
        try:
            from engine.cache.lifecycle import LifecycleManager
            lm = LifecycleManager(cli._config, cli._scheduler)
            if lm.restore(args.strip()):
                console.print("[green]恢复完成[/green]")
            else:
                console.print("[red]恢复失败[/red]")
        except Exception as e:
            console.print(f"[red]恢复失败: {e}[/red]")

    def cmd_semantic_on(args: str):
        if cli._scheduler.l3 is not None:
            cli._scheduler.l3.enabled = True
            console.print("[green]语义缓存已开启[/green]")
        else:
            console.print("[yellow]语义缓存模块未初始化[/yellow]")

    def cmd_semantic_off(args: str):
        if cli._scheduler.l3 is not None:
            cli._scheduler.l3.enabled = False
        console.print("[green]语义缓存已关闭 (严格0误差模式)[/green]")

    registry.register("cache_hit", "查看命中率", cmd_cache_hit, "缓存管理")
    registry.register("cache_clear", "清理所有缓存", cmd_cache_clear, "缓存管理")
    registry.register("cache_backup", "手动备份缓存", cmd_cache_backup, "缓存管理", "/cache_backup [路径]")
    registry.register("cache_restore", "从备份恢复", cmd_cache_restore, "缓存管理", "/cache_restore <路径>")
    registry.register("semantic_on", "开启语义缓存", cmd_semantic_on, "缓存管理")
    registry.register("semantic_off", "关闭语义缓存", cmd_semantic_off, "缓存管理")

    # ===== Category 4: Safety Management =====
    def cmd_safety_level(args: str):
        if args.strip().lower() in ("on", "true", "1"):
            cli._safety.enabled = True
            console.print("[green]安全过滤: 已开启[/green]")
        elif args.strip().lower() in ("off", "false", "0"):
            cli._safety.enabled = False
            console.print("[yellow]安全过滤: 已关闭[/yellow]")
        else:
            console.print(f"安全过滤: {'开启' if cli._safety.enabled else '关闭'}")

    def cmd_safety_log(args: str):
        stats = cli._safety.stats
        console.print(f"安全拦截次数: {stats['blocked_count']}")

    def cmd_data_clear(args: str):
        if args.strip() == "confirm":
            from engine.safety.privacy import wipe_all_data
            wipe_all_data(cli._config.cache.base_dir)
            cli._scheduler.clear_all()
            console.print("[green]所有用户数据已清除[/green]")
        else:
            console.print("[yellow]此操作不可恢复! 请输入: /data_clear confirm[/yellow]")

    registry.register("safety_level", "调整安全等级", cmd_safety_level, "安全管理", "/safety_level [on|off]")
    registry.register("safety_log", "查看安全拦截日志", cmd_safety_log, "安全管理")
    registry.register("data_clear", "一键清除所有数据", cmd_data_clear, "安全管理", "/data_clear confirm")

    # ===== Category 5: Operations & Monitoring =====
    def cmd_status(args: str):
        from engine.cli.display import show_status
        show_status(cli.get_stats(), cli.get_resource_info())

    def cmd_check(args: str):
        from engine.monitoring.health import run_health_check
        report = run_health_check(cli._config, cli._engine, cli._scheduler)
        console.print(f"\n状态: [{'green' if report['status']=='healthy' else 'yellow'}]{report['status']}[/]")
        for check in report["checks"]:
            icon = "✓" if check["status"] == "pass" else "!"
            console.print(f"  {icon} {check['name']}: {check['detail']}")
        for warn in report.get("warnings", []):
            console.print(f"  [yellow]⚠ {warn}[/yellow]")

    def cmd_compact(args: str):
        try:
            cli._scheduler.l4.compact()
            console.print("[green]数据库碎片整理完成[/green]")
        except Exception as e:
            console.print(f"[red]碎片整理失败: {e}[/red]")

    def cmd_maintenance(args: str):
        from engine.cache.lifecycle import LifecycleManager
        lm = LifecycleManager(cli._config, cli._scheduler)
        report = lm.run_full_maintenance()
        for action in report["actions"]:
            console.print(f"  [green]✓[/green] {action}")
        for err in report["errors"]:
            console.print(f"  [red]✗[/red] {err}")
        if "storage" in report:
            console.print(f"  缓存总大小: {report['storage']['total_cache_size']}")
            console.print(f"  磁盘剩余: {report['storage']['disk_free']}")

    def cmd_log(args: str):
        import os
        log_dir = os.path.expanduser(cli._config.logging.dir)
        log_file = os.path.join(log_dir, "runtime.log")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                n = min(20, len(lines))
                console.print(f"[dim]最近 {n} 条日志:[/dim]")
                for line in lines[-n:]:
                    console.print(f"  {line.rstrip()}")
            except Exception as e:
                console.print(f"[red]读取日志失败: {e}[/red]")
        else:
            console.print("[dim]日志文件不存在[/dim]")

    def cmd_storage(args: str):
        from engine.cache.lifecycle import LifecycleManager
        from engine.utils.storage import format_size
        lm = LifecycleManager(cli._config, cli._scheduler)
        report = lm.get_storage_report()
        console.print("存储使用情况:")
        console.print(f"  L1 内存: {format_size(report.get('l1_memory_bytes', 0))}")
        if "l2_memory_bytes" in report:
            console.print(f"  L2 内存: {format_size(report['l2_memory_bytes'])}")
        console.print(f"  L4 磁盘: {format_size(report.get('l4_disk_bytes', 0))}")
        if "l3_disk_bytes" in report:
            console.print(f"  L3 磁盘: {format_size(report['l3_disk_bytes'])}")
        console.print(f"  总计: {format_size(report.get('total_disk_bytes', 0))}")
        console.print(f"  磁盘剩余: {report.get('disk_free_gb', 0):.1f}GB")

    registry.register("status", "查看引擎状态", cmd_status, "运维工具")
    registry.register("check", "校验缓存完整性", cmd_check, "运维工具")
    registry.register("compact", "整理数据库碎片", cmd_compact, "运维工具")
    registry.register("maintenance", "执行全量维护", cmd_maintenance, "运维工具")
    registry.register("log", "查看运行日志", cmd_log, "运维工具", "/log")
    registry.register("storage", "查看存储使用", cmd_storage, "运维工具")

    # ===== Category 6: Help =====
    def cmd_faq(args: str):
        from engine.cli.display import show_faq
        show_faq()

    def cmd_redlines(args: str):
        console.print("\n[bold]三大不可突破红线:[/bold]")
        console.print("  [red]1. 不量化[/red] - 全程原生BF16/FP16全精度权重")
        console.print("  [red]2. 不蒸馏[/red] - 完整保留原生模型全部结构")
        console.print("  [red]3. 不减性能[/red] - 输出与原生模型100%一致\n")

    def cmd_cache_principles(args: str):
        """显示三大缓存原则及当前配置"""
        io = getattr(cli._config.cache, "instruction_only", True)
        tl = getattr(cli._config.cache, "token_level", True)
        se = getattr(cli._config.cache, "strict_exact", True)
        console.print("\n[bold]三大缓存原则:[/bold]")
        console.print("  1. [cyan]不存整个会话，只认单次指令的完整内容[/cyan]")
        console.print(f"     → instruction_only = {io}")
        console.print("  2. [cyan]不存单次指令的整体，只存逐 token 计算过程 (KV/logits)[/cyan]")
        console.print(f"     → token_level = {tl}")
        console.print("  3. [cyan]指令一字不差才命中；变一字则重新计算+存新缓存[/cyan]")
        console.print(f"     → strict_exact = {se}")
        console.print("\n[bold]缓存时机:[/bold]")
        console.print("  [cyan]每句话发完就存内存，会话结束只存硬盘[/cyan]")
        console.print("  → 缓存即时生效，不依赖会话是否结束")
        pc = getattr(cli._config.cache, "prefix_cache", True)
        console.print("\n[bold]前缀缓存:[/bold]")
        console.print(f"  [cyan]长对话前面不变则前面全复用 (KV 级)[/cyan] → prefix_cache = {pc}")
        console.print()

    # ===== Category 7: Resource Management =====
    def cmd_unload(args: str):
        if cli._engine.is_loaded:
            cli._engine.unload()
            console.print("[green]模型已从内存卸载，释放全部CPU/GPU/内存资源[/green]")
            console.print("[dim]缓存数据保留在磁盘，缓存命中仍可极速输出[/dim]")
        else:
            console.print("[dim]模型当前未加载在内存中[/dim]")

    def cmd_precompute(args: str):
        """Batch pre-compute responses to populate cache."""
        if not args.strip():
            console.print("用法: /precompute <文件路径>")
            console.print("文件格式: 每行一个问题，引擎会逐个推理并缓存结果")
            console.print("预计算完成后可卸载模型，纯从缓存极速响应")
            return

        import os
        filepath = os.path.expanduser(args.strip())
        if not os.path.exists(filepath):
            console.print(f"[red]文件不存在: {filepath}[/red]")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if not questions:
            console.print("[yellow]文件中没有有效的问题[/yellow]")
            return

        console.print(f"\n[bold]开始批量预计算 {len(questions)} 个问题...[/bold]")
        console.print("[dim]完成后可 /unload 卸载模型，纯缓存极速响应[/dim]\n")

        from engine.cache.base import CacheEntry
        from engine.cli.interface import _fast_hash

        use_instruction_only = getattr(cli._config.cache, "instruction_only", True)
        use_strict = getattr(cli._config.cache, "strict_exact", True)

        success = 0
        for i, question in enumerate(questions, 1):
            console.print(f"  [{i}/{len(questions)}] {question[:50]}...", end=" ")
            try:
                sys_prompt = "你是一个有用的AI助手。请直接回答用户的问题，简洁准确。/no_think"
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ]
                model_hash = cli._engine.model_hash or "default"

                instruction_text = f"<|system|>{sys_prompt}<|user|>{question}"
                instruction_hash = _fast_hash(instruction_text)

                existing = cli._scheduler.l1.get(instruction_hash, model_hash) or cli._scheduler.l4.get(instruction_hash, model_hash)
                if existing and existing.response_text:
                    console.print("[cyan]已有缓存,跳过[/cyan]")
                    success += 1
                    continue

                parts = []
                for chunk in cli._engine.chat(messages, cli._gen_params):
                    parts.append(chunk)
                response = "".join(parts)

                keys = [instruction_hash]
                if not use_strict:
                    from engine.cli.interface import _normalize_text
                    norm_hash = _fast_hash(f"<|system|>{_normalize_text(sys_prompt)}<|user|>{_normalize_text(question)}")
                    if norm_hash != instruction_hash:
                        keys.append(norm_hash)

                for key in keys:
                    entry = CacheEntry(
                        context_hash=key,
                        model_hash=model_hash,
                        response_text=response,
                        greedy_token=-1,
                    )
                    cli._scheduler.l1.put(entry)
                    # L4 在会话结束时由 scheduler.close() 统一持久化
                success += 1
                console.print(f"[green]已缓存 ({len(response)}字)[/green]")
            except Exception as e:
                console.print(f"[red]失败: {e}[/red]")

        console.print(f"\n[bold green]预计算完成: {success}/{len(questions)} 成功[/bold green]")
        console.print("[dim]输入 /unload 卸载模型，之后这些问题将纯从磁盘极速响应[/dim]\n")

    def cmd_resource(args: str):
        info = cli.get_resource_info()
        console.print(f"  进程内存: {info['memory_rss_mb']:.1f}MB")
        console.print(f"  模型状态: {'已加载(占用内存)' if info.get('model_in_memory') else '未加载(零占用)'}")
        console.print(f"  缓存L1内存: {cli._scheduler.l1.size_bytes() / 1024 / 1024:.1f}MB")
        console.print(f"  缓存L4磁盘: {cli._scheduler.l4.size_bytes() / 1024 / 1024:.1f}MB")

    def cmd_tune_low_compute(args: str):
        """一键调优：提高命中率（少用算力）+ 推理时 token 更快。"""
        # 1) 开启 L3 语义缓存 → 相似问题命中缓存，减少推理次数
        if cli._scheduler.l3 is not None:
            cli._scheduler.l3.enabled = True
            console.print("[green]✓ 已开启 L3 语义缓存[/green] — 相似问题可命中，减少算力")
        # 2) 推理参数：按电源状态推荐 n_batch / n_threads，下次加载模型生效
        try:
            from engine.extensions.optimizer import PowerManager
            pm = PowerManager(cli._config)
            profile = pm.get_power_profile()
            n_threads = profile.get("n_threads", 4)
            n_batch = profile.get("n_batch", 512)
            cli._config.model.n_threads = n_threads
            cli._config.model.n_batch = n_batch
            console.print(f"[green]✓ 推理参数已设为[/green] n_threads={n_threads}, n_batch={n_batch} — 下次加载模型时生效，token 更快")
        except Exception:
            cli._config.model.n_batch = 512
            console.print("[green]✓ 推理参数已设为[/green] n_batch=512 — 下次加载模型时生效")
        # 3) 提示：预计算 + 卸载 = 日常少算力
        console.print("[dim]建议: 用 /precompute <问题文件> 批量填缓存，再用 /unload 卸载模型，日常多数请求走缓存、基本不占算力[/dim]")

    registry.register("unload", "卸载模型释放资源", cmd_unload, "资源管理")
    registry.register("tune_low_compute", "一键调优：少用算力+token加速", cmd_tune_low_compute, "资源管理", "/tune_low_compute")
    registry.register("precompute", "批量预计算填充缓存", cmd_precompute, "资源管理", "/precompute <问题文件>")
    registry.register("resource", "查看资源占用", cmd_resource, "资源管理")

    registry.register("faq", "常见问题", cmd_faq, "帮助")
    registry.register("redlines", "查看三大红线", cmd_redlines, "帮助")
    registry.register("cache_principles", "查看三大缓存原则", cmd_cache_principles, "帮助", "/cache_principles")
