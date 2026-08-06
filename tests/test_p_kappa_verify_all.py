# -*- coding: utf-8 -*-
"""P-κ verify_all.py 统一验收入口 — CLI 契约 / 单阶段 / 红侧自证 / 无 gmsh 构造.

判别性:
  - --help 列出全部模式; 未知 flag / 未知 --stage → 退出码 1 + 错误消息
    (argparse 默认用法错误码 2 已被覆写为 1, 与 "test 组 FAIL = 2" 区分);
  - --list-stages 输出全部阶段与分组, 不执行任何阶段;
  - --drift 从任意子目录真实跑漂移门 → 退出码 0 (本仓库当前全绿; 无 gmsh
    API 时跳过 — 漂移门需 gmsh API);
  - --fake-fail (构造性红侧, 不引入任何坏文件): 汇总表标注 FAIL 且退出码 2
    (test 组), 与 lint 组退出码 3 语义区分;
  - 无 gmsh 构造单测: 假 gmsh.py 内容为 raise ImportError (测试 skip 守卫
    只捕 ImportError, OSError 会导致 1 个测试失败 — 历史教训), PYTHONPATH
    前置路径存在, 上下文退出后临时目录清理。

约束: 本文件测试不得调用真全量 pytest (CI 会递归) — 只用 --list-stages /
--stage / --fake-fail / 构造函数直测覆盖。
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
VERIFY_ALL = SCRIPTS_DIR / "verify_all.py"

# --list-stages 的阶段契约 (verify_all.STAGES 键, 硬编码锁定防改名)
ALL_STAGE_NAMES = [
    "pytest", "probe", "fuzz", "drift", "no-gmsh", "fake-fail",
    "compileall", "ruff", "mypy", "vulture", "self-test",
]


def _run_cli(*args, cwd=None):
    """子进程调 verify_all.py (UTF-8 捕获, Windows cp936 兼容)."""
    return subprocess.run(
        [sys.executable, str(VERIFY_ALL), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=cwd or REPO_ROOT)


def _import_verify_all():
    """进程内 import verify_all (仅构造单测用, 不触发任何阶段)."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        return importlib.import_module("verify_all")
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        sys.modules.pop("verify_all", None)


def test_help_lists_all_modes(tmp_path):
    """CLI 契约: --help 从任意目录列出全部模式与 flag."""
    proc = _run_cli("--help", cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for flag in ("--fast", "--full", "--no-gmsh", "--drift", "--list-stages",
                 "--stage", "--fake-fail"):
        assert flag in proc.stdout, proc.stdout


def test_unknown_flag_exit_1():
    """未知 flag → 退出码 1 + 错误消息 (argparse 默认 2 已覆写)."""
    proc = _run_cli("--bogus")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "error" in proc.stderr


def test_unknown_stage_exit_1():
    """未知 --stage → 退出码 1 + 错误消息."""
    proc = _run_cli("--stage", "nope")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "nope" in proc.stderr


def test_list_stages_contract_from_subdir(tmp_path):
    """阶段划分: --list-stages 输出全部阶段与分组 (不执行, 从子目录可调)."""
    proc = _run_cli("--list-stages", cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in ALL_STAGE_NAMES:
        assert name in proc.stdout, proc.stdout
    for group in ("test", "lint"):
        assert group in proc.stdout, proc.stdout


def test_drift_real_green_from_subdir():
    """单阶段真实执行: --drift 从 tests/ 子目录调用 → 漂移门 6 值
    rel<1e-9, 退出码 0 (本仓库当前全绿)."""
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("Gmsh Python API 不可用 — 漂移门真实运行测试跳过")
    proc = _run_cli("--drift", cwd=REPO_ROOT / "tests")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "drift | PASS" in proc.stdout, proc.stdout
    assert "max rel" in proc.stdout, proc.stdout
    assert not (REPO_ROOT / "_ci_drift.msh").exists(), \
        "漂移门临时 _ci_drift.msh 必须清理"


def test_fake_fail_red_side():
    """构造性红侧: --fake-fail (必然失败假命令, 不引入坏文件) →
    汇总表 FAIL + 退出码 2 (test 组)."""
    proc = _run_cli("--fake-fail")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "fake-fail | FAIL" in proc.stdout, proc.stdout
    assert "退出码 2" in proc.stdout, proc.stdout


def test_no_gmsh_env_construction():
    """无 gmsh 子命令构造单测 (不真跑全量 pytest): 假 gmsh.py 内容 /
    PYTHONPATH 前置 (os.pathsep) / 退出后清理."""
    verify_all = _import_verify_all()
    with verify_all.no_gmsh_pytest_env() as (tmp, env):
        assert tmp.parent == verify_all.REPO_ROOT, \
            "假 gmsh 临时目录必须在当前 worktree 内"
        gmsh_py = tmp / "gmsh.py"
        assert gmsh_py.is_file()
        assert gmsh_py.read_text(encoding="utf-8") == \
            'raise ImportError("gmsh unavailable")\n'
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp)
    assert not tmp.exists(), "退出后假 gmsh 临时目录必须清理"
