"""领域异常 — 库代码不直接 sys.exit (审查: 27 处 CLI 耦合).

CLI 交互/输入层抛 :class:`CliError`, 由 runner.main / 示例脚本捕获
转换为进程退出码; 库嵌入方 (Jupyter/单元测试) 得到异常而非进程自杀。
"""


class CliError(Exception):
    """CLI 输入/交互错误 — exit_code 由 CLI 层转换 (默认 1)."""

    def __init__(self, message, *, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code
