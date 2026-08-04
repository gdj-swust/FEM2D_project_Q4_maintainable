"""领域异常 — 库代码不直接 sys.exit.

CLI 交互/输入层抛 :class:`CliError`, 由 runner.main / 示例脚本捕获
转换为进程退出码 (0 正常 / 1 用户错误 / 2 内部错误); 库嵌入方
(Jupyter/单元测试) 得到异常而非进程自杀。
"""
import sys


def reconfigure_streams():
    """编码安全网 — 非中文 Windows (cp1252 等) 下中文输出不崩溃.

    run_demo.py 与 runner.main 共用 (曾 7 行逐字复制两份):
    输出流不支持 reconfigure 或重配失败时保持原流不变。
    """
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


class UnderconstrainedError(RuntimeError):
    """模型欠约束 (刚体模态未锁死) — 用户漏加 BC, CLI 应映射退出码 1."""


class CliError(Exception):
    """CLI 输入/交互错误 — exit_code 由 CLI 层转换 (默认 1)."""

    def __init__(self, message, *, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code


class GeoScriptRejected(ValueError):
    """.geo 含被禁止的危险指令 (SystemCall) — 拒绝执行该脚本.

    Gmsh 脚本语言支持 SystemCall 执行任意系统命令, .geo 属"可信、可执行
    式输入"。黑名单拦截在 scripts.gmsh_runner.sanitize_geo_source;
    独立类型使调用链能区分"脚本被拒" (用户错误 → 退出码 1) 与一般失败。
    """
