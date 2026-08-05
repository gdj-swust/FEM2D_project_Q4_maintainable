"""A8 (P3, R-α): fuzz 分支 30/31 重复修复 + 值池补 complex 数组.

审查现象 (2026-08-05): 分支 30 与 31 同一入口同一 lambda (完全重复);
头部注释称 11 新入口实际含 1 对重复; 值池无 complex 数组值 (P2-A1
complex 静默错值的盲区来源)。

修复 (scripts/fuzz_api.py):
  * 分支 31 → estimate_error 的 method 参数面 (池值恒非法 → 全部
    必须报 ValueError; 此前该参数面无入口覆盖)
  * 值池补 complex 数组 (kind 14: (3,) 单向量; kind 15: (2,3) 批量)
  * _is_stress_vec 对 complex 返回 False — 静默成功必须查得出
  * 注释表述修正

值池扩展的必然结果 (numpy≥2 astype(float) 对 complex 静默丢虚部):
compute_traction_jumps / nodal_L2_projection 补 complex 拒绝守卫
(A1 同族, 见本文件与 fuzz 500 验收)。

判别性: 回滚本提交 → 分支 30/31 结构断言红 (分支计数差)。
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from fem2d.error_est import compute_traction_jumps
from fem2d.mesh import Mesh
from fem2d.stress import nodal_L2_projection

FUZZ_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fuzz_api.py"

NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def _mesh():
    return Mesh(NODES, ELEMS, E=1e6, nu=0.3, thickness=1.0)


def _fuzz_branch_bodies():
    """提取 main() 中 elif i == 30 / i == 31 的 if 体源码 (ast 还原)."""
    tree = ast.parse(FUZZ_PATH.read_text(encoding="utf-8"))
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    bodies = {}
    for node in ast.walk(main):
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "i"
                    and isinstance(test.comparators[0], ast.Constant)):
                key = f"i == {test.comparators[0].value}"
                if key in ("i == 30", "i == 31"):
                    bodies[key] = ast.unparse(node.body)
    return bodies


def test_fuzz_branch_30_and_31_distinct():
    # 回滚 → 分支 30/31 完全重复 (同一入口同一 lambda)
    bodies = _fuzz_branch_bodies()
    assert set(bodies) == {"i == 30", "i == 31"}
    assert bodies["i == 30"] != bodies["i == 31"]
    # 分支 31 覆盖 estimate_error 的 method 参数面
    assert "estimate_error" in bodies["i == 31"]


def test_fuzz_value_pool_has_complex_arrays():
    import sys
    import types

    spec = {"os": __import__("os"), "sys": sys, "random": __import__("random"),
            "np": np}
    import importlib.util
    module_spec = importlib.util.spec_from_file_location(
        "fuzz_api_under_test", FUZZ_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    import random
    rng = random.Random(12345)
    kinds = set()
    for _ in range(400):
        v = module._rand_value(rng)
        if isinstance(v, np.ndarray) and v.dtype.kind == "c":
            kinds.add(v.shape)
    assert (3,) in kinds        # complex 单向量
    assert (2, 3) in kinds      # complex 批量
    # silent_ok 契约: complex 应力向量不算"合法" (静默成功必须查得出)
    assert module._is_stress_vec(np.array([1 + 1j, 2.0, 3.0])) is False
    assert module._is_stress_vec(np.array([1.0, 2.0, 3.0])) is True


def test_compute_traction_jumps_complex_rejected():
    # numpy≥2 astype(float) 静默丢虚部 — 值池补 complex 后必须拒绝
    with pytest.raises(ValueError, match="compute_traction_jumps"):
        compute_traction_jumps(
            _mesh(), np.array([[1 + 1j, 1.0, 0.0], [1.0, 2.0, 0.5]]))


def test_nodal_L2_complex_rejected():
    with pytest.raises(ValueError, match="nodal_L2_projection"):
        nodal_L2_projection(
            _mesh(), np.array([[1 + 1j, 1.0, 0.0], [1.0, 2.0, 0.5]]))


def test_finite_inputs_unchanged():
    # 冻结区: 有限实数输入行为不变
    m = _mesh()
    out = nodal_L2_projection(m, np.ones((2, 3)))
    assert out.shape == (4, 3)
    j = compute_traction_jumps(m, np.ones((2, 3)))
    assert isinstance(j, list) and len(j) >= 1
