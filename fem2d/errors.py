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


class CliError(Exception):
    """CLI 输入/交互错误 — exit_code 由 CLI 层转换 (默认 1)."""

    def __init__(self, message, *, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code
