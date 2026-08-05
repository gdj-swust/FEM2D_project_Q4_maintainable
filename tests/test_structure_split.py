"""B 结构轮判别性测试 — 7 长函数拆分 + Include 递归闭环 (S8).

判别性 (回滚必须红):
- S1-S7: 任何目标函数回到 >60 行 (拆分回滚) → AST 行数断言失败;
- S8: gmsh_adapter._safe_geo_source 不再传 geo_path (递归扫描回滚)
  → 深层 Include 的 SystemCall/循环不再被拦截, pytest.raises 失败。
"""
import ast
import os
from pathlib import Path

import pytest

from fem2d.bc import apply_elimination
from fem2d.boundary.naming import segments_from_physical_curves
from fem2d.convergence import run_cantilever_convergence
from fem2d.error_est import element_refinement_indicator
from fem2d.errors import GeoScriptRejected
from fem2d.gmsh_adapter import _safe_geo_source
from fem2d.runner import main as runner_main
from fem2d.verification import run_plane_verification

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (文件, 函数) — 结构轮 S1-S7 目标. apply_elimination 实际位于
# fem2d/bc.py (任务书表锚点写 solver.py 属笔误; 交接文档与验收均按函数).
SPLIT_TARGETS = [
    ("fem2d/convergence.py", "run_cantilever_convergence"),
    ("fem2d/gmsh_adapter.py", "_extract_regions"),
    ("fem2d/error_est.py", "element_refinement_indicator"),
    ("fem2d/bc.py", "apply_elimination"),
    ("fem2d/verification.py", "run_plane_verification"),
    ("fem2d/boundary/naming.py", "segments_from_physical_curves"),
    ("fem2d/runner.py", "main"),
]

MAX_FUNCTION_LINES = 60


def _function_line_count(module_name, function_name):
    tree = ast.parse(
        (PROJECT_ROOT / module_name).read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name):
            return (node.end_lineno or node.lineno) - node.lineno + 1
    raise AssertionError(
        f"{module_name} 中找不到函数 {function_name}")


@pytest.mark.parametrize(
    "module_name,function_name", SPLIT_TARGETS,
    ids=[name for _, name in SPLIT_TARGETS])
def test_split_function_within_60_lines(module_name, function_name):
    """S1-S7: 拆分后每个目标函数 ≤ 60 行 — 回滚拆分 (超长) 此用例红."""
    assert _function_line_count(
        module_name, function_name) <= MAX_FUNCTION_LINES


def test_split_targets_remain_importable():
    """S1-S7: 拆分后公开入口仍可导入 (签名/模块路径未破坏)."""
    assert callable(apply_elimination)
    assert callable(segments_from_physical_curves)
    assert callable(run_cantilever_convergence)
    assert callable(element_refinement_indicator)
    assert callable(run_plane_verification)
    assert callable(runner_main)


def test_plane_verification_still_passes():
    """S5 行为烟测: 拆分后验证计数不变 (6 PASS: Test1×4 + Test2×2)."""
    assert run_plane_verification() == (6, 0)


# ─────────────────────────────────────────────────────────────────────
# S8 — _safe_geo_source Include 递归闭环 (API 路径)
#
# 与 scripts.gmsh_runner 的子进程路径共用同一扫描器; 此处锁定 API 路径
# 也传入 geo_path. 回滚 (只传 source) → 深层 Include 不再扫描 → 红.
# ─────────────────────────────────────────────────────────────────────

def _write_geo(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_api_path_rejects_deep_include_systemcall(tmp_path):
    """深层 Include (a → b → c) 中恶意命令被拦截, 错误含完整引用链."""
    _write_geo(tmp_path, "c.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "b.geo", 'Include "c.geo";\n')
    _write_geo(tmp_path, "a.geo", 'Include "b.geo";\n')
    with pytest.raises(GeoScriptRejected,
                       match=r"a\.geo → b\.geo → c\.geo.*SystemCall"):
        _safe_geo_source(str(tmp_path / "a.geo"))


def test_api_path_include_cycle_rejected(tmp_path):
    """A→B→A 循环 Include → 拒绝并报引用链 (active 集合防死循环)."""
    _write_geo(tmp_path, "b.geo", 'Include "a.geo";\n')
    _write_geo(tmp_path, "a.geo", 'Include "b.geo";\n')
    with pytest.raises(GeoScriptRejected,
                       match=r"循环引用.*a\.geo → b\.geo → a\.geo"):
        _safe_geo_source(str(tmp_path / "a.geo"))


def test_api_path_include_diamond_not_false_cycle(tmp_path):
    """A→B→C 钻石共享 (done 集合) — 不误判循环, 正常放行."""
    _write_geo(tmp_path, "d.geo", "lc = 1;\n")
    _write_geo(tmp_path, "b.geo", 'Include "d.geo";\n')
    _write_geo(tmp_path, "c.geo", 'Include "d.geo";\n')
    _write_geo(tmp_path, "a.geo",
               'Include "b.geo";\nInclude "c.geo";\n')
    path, temp = _safe_geo_source(str(tmp_path / "a.geo"))
    # 内容未变 → 不生成临时副本, 直接返回原文件绝对路径
    assert path == os.path.abspath(str(tmp_path / "a.geo"))
    assert temp is None


def test_api_path_missing_include_skipped(tmp_path):
    """Include 目标缺失 → 跳过不报错 (Gmsh 解析时自行报错)."""
    _write_geo(tmp_path, "a.geo", 'Include "nope.geo";\n')
    assert _safe_geo_source(str(tmp_path / "a.geo"))[1] is None
