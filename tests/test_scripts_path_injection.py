"""组 E — 审计脚本直接运行 (sys.path 根目录注入) 判别性测试.

`python scripts/xxx.py` 直接运行时 sys.path[0] = scripts/ — 若无根目录
注入, `import fem2d` 落到 editable install 指向的外部副本 (本机曾静默
测到主分支旧实现) 或 ModuleNotFoundError (干净环境)。本测试在子进程
中用 -S 跳过 site 处理 (等价无 pip install 的干净环境, 也绕开本机
wrapt 对 sys.meta_path 的劫持) 执行脚本模块级代码, 断言:
1) 无 ModuleNotFoundError; 2) fem2d 解析到本 worktree。
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

_SCRIPT_MODULES = [
    "audit_contract_probe",
    "fuzz_api",
    "combo_fuzz",
    "regression_compare",
]


def _run_script_clean_env(mod_name, cwd):
    """子进程 (-S: 无 site 处理 → 无可编辑 finder/wrapt 劫持, 等价干净
    环境) + scripts/ 前置 (直接运行语义), 执行脚本模块级代码
    (__name__ 非 __main__ — 不跑 main)."""
    import sysconfig

    script = SCRIPTS_DIR / f"{mod_name}.py"
    code = (
        "import sys\n"
        f"sys.path.append({sysconfig.get_paths()['purelib']!r})\n"
        "sys.path.insert(0, __import__('os').path.dirname"
        f"({str(script)!r}))\n"
        f"src = open({str(script)!r}, encoding='utf-8').read()\n"
        "g = {'__name__': 'script_under_test',"
        f" '__file__': {str(script)!r}, '__builtins__': __builtins__}}\n"
        "exec(compile(src, 'script_under_test', 'exec'), g)\n"
        "import fem2d\n"
        "print('RESOLVED:', fem2d.__file__)\n"
    )
    return subprocess.run(
        [sys.executable, "-S", "-c", code],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=cwd,
    )


@pytest.mark.parametrize("mod_name", _SCRIPT_MODULES)
def test_script_direct_run_resolves_this_worktree_fem2d(mod_name, tmp_path):
    """判别性: 干净环境下直接运行脚本 — 根目录注入必须生效, 解析到
    本 worktree 的 fem2d (旧实现: ModuleNotFoundError → 退出码非零)."""
    proc = _run_script_clean_env(mod_name, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESOLVED:" in proc.stdout, proc.stdout + proc.stderr
    resolved = Path(proc.stdout.split("RESOLVED:")[1].strip()).resolve()
    assert resolved.parent.parent == REPO_ROOT, resolved


def test_fuzz_api_classify_catch_semantics():
    """判别性: fuzz 捕获只接受预期异常 (ValueError/TypeError/CliError) 且
    消息非空; RuntimeError/OverflowError 等计入 unexpected — 曾全部当成功
    忽略 (捕获过宽, RuntimeError/OverflowError 静默放过)."""
    import importlib

    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        fuzz_api = importlib.import_module("fuzz_api")
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        sys.modules.pop("fuzz_api", None)

    from fem2d.errors import CliError

    # 预期异常带非空消息 → 正确行为 (不报)
    assert fuzz_api._classify(ValueError("bad input")) is None
    assert fuzz_api._classify(TypeError("bad type")) is None
    assert fuzz_api._classify(CliError("bad cli")) is None
    # 预期异常空消息 → bug (无诊断)
    empty = fuzz_api._classify(ValueError(""))
    assert empty is not None and "空消息" in empty
    # 非预期异常 → unexpected
    assert fuzz_api._classify(
        OverflowError("cannot convert float infinity to integer")
    ).startswith("unexpected OverflowError")
    assert fuzz_api._classify(
        IndexError("list index out of range")
    ).startswith("unexpected IndexError")
    assert fuzz_api._classify(RuntimeError("boom")).startswith(
        "unexpected RuntimeError"
    )
