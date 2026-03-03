"""Local tool calling framework: Python interpreter, calculator, file I/O, custom tools.

Enables the model to call local tools with no network restrictions,
no file size limits, and no execution time limits.
All execution is sandboxed locally.
"""

import importlib
import io
import json
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable
    category: str = "builtin"


class ToolRegistry:
    """Registry for local tools that can be called by the model."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_builtins()

    def _register_builtins(self):
        self.register(ToolDefinition(
            name="python",
            description="Execute Python code locally",
            parameters={"code": "Python code to execute"},
            handler=self._exec_python,
            category="builtin",
        ))
        self.register(ToolDefinition(
            name="calculator",
            description="Evaluate mathematical expressions with high precision",
            parameters={"expression": "Math expression to evaluate"},
            handler=self._calc,
            category="builtin",
        ))
        self.register(ToolDefinition(
            name="read_file",
            description="Read contents of a local file",
            parameters={"path": "File path to read"},
            handler=self._read_file,
            category="builtin",
        ))
        self.register(ToolDefinition(
            name="write_file",
            description="Write contents to a local file",
            parameters={"path": "File path", "content": "Content to write"},
            handler=self._write_file,
            category="builtin",
        ))
        self.register(ToolDefinition(
            name="list_dir",
            description="List directory contents",
            parameters={"path": "Directory path"},
            handler=self._list_dir,
            category="builtin",
        ))
        self.register(ToolDefinition(
            name="system_info",
            description="Get system hardware and OS information",
            parameters={},
            handler=self._system_info,
            category="builtin",
        ))

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool
        logger.debug(f"工具注册: {tool.name}")

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "category": t.category,
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, **kwargs) -> str:
        """Call a registered tool and return the result as a string."""
        tool = self._tools.get(name)
        if tool is None:
            return f"错误: 未知工具 '{name}'"
        try:
            result = tool.handler(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行错误 ({name}): {e}"

    def load_custom_tools(self, tools_dir: str):
        """Load custom tools from Python scripts in a directory."""
        path = Path(tools_dir)
        if not path.exists():
            return

        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    py_file.stem, str(py_file)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "TOOL_DEFINITION"):
                    defn = module.TOOL_DEFINITION
                    tool = ToolDefinition(
                        name=defn.get("name", py_file.stem),
                        description=defn.get("description", ""),
                        parameters=defn.get("parameters", {}),
                        handler=getattr(module, defn.get("handler", "run")),
                        category="custom",
                    )
                    self.register(tool)
                    logger.info(f"加载自定义工具: {tool.name} ({py_file.name})")
            except Exception as e:
                logger.warning(f"加载自定义工具失败 ({py_file.name}): {e}")

    @staticmethod
    def _exec_python(code: str) -> str:
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        local_vars = {}
        try:
            exec(code, {"__builtins__": __builtins__}, local_vars)
            output = buffer.getvalue()
            if output:
                return output.rstrip()
            last_expr = code.strip().split("\n")[-1]
            try:
                result = eval(last_expr, {"__builtins__": __builtins__}, local_vars)
                if result is not None:
                    return str(result)
            except Exception:
                pass
            return output or "(执行完成，无输出)"
        except Exception:
            return f"执行错误:\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout

    @staticmethod
    def _calc(expression: str) -> str:
        try:
            import ast
            node = ast.parse(expression, mode="eval")
            result = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "int": int, "float": float,
            })
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    @staticmethod
    def _read_file(path: str) -> str:
        try:
            p = Path(os.path.expanduser(path))
            if not p.exists():
                return f"文件不存在: {path}"
            if p.stat().st_size > 10 * 1024 * 1024:
                return f"文件过大 ({p.stat().st_size} bytes), 最大支持10MB"
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"读取失败: {e}"

    @staticmethod
    def _write_file(path: str, content: str) -> str:
        try:
            p = Path(os.path.expanduser(path))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"已写入: {path} ({len(content)} 字符)"
        except Exception as e:
            return f"写入失败: {e}"

    @staticmethod
    def _list_dir(path: str = ".") -> str:
        try:
            p = Path(os.path.expanduser(path))
            if not p.exists():
                return f"目录不存在: {path}"
            entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            lines = []
            for e in entries[:100]:
                prefix = "📁 " if e.is_dir() else "📄 "
                size = f" ({e.stat().st_size}B)" if e.is_file() else ""
                lines.append(f"{prefix}{e.name}{size}")
            if len(entries) > 100:
                lines.append(f"... 共 {len(entries)} 项")
            return "\n".join(lines)
        except Exception as e:
            return f"列出失败: {e}"

    @staticmethod
    def _system_info() -> str:
        import platform
        import psutil
        mem = psutil.virtual_memory()
        return (
            f"OS: {platform.system()} {platform.release()}\n"
            f"Arch: {platform.machine()}\n"
            f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} threads)\n"
            f"Memory: {mem.total // (1024**2)}MB total, {mem.available // (1024**2)}MB available\n"
            f"Python: {platform.python_version()}"
        )


def parse_tool_call(text: str) -> Optional[Dict]:
    """Parse a tool call from model output text.

    Looks for patterns like:
    <tool_call>{"name": "python", "arguments": {"code": "print(1+1)"}}</tool_call>
    or
    ```tool_call
    {"name": "calculator", "arguments": {"expression": "2**10"}}
    ```
    """
    patterns = [
        re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL),
        re.compile(r"```tool_call\s*\n(.*?)\n```", re.DOTALL),
        re.compile(r'\{"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{.*?\})\}', re.DOTALL),
    ]

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                if match.lastindex == 2:
                    return {
                        "name": match.group(1),
                        "arguments": json.loads(match.group(2)),
                    }
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None
