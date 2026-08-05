"""A10 (P3, R-α, 冻结区): ElementLocator.candidates 非有限输入返回空.

审查现象 (2026-08-05): 公开 kernel API find_containing_element 对 inf
坐标冒裸 OverflowError (经 locator.candidates 的 int() 转换);
文档化入口 point_in_element 已防护 (AABB + 有限性前置), kernel 层
无防护。

修复 (冻结区特殊程序): ElementLocator.candidates 对非有限/非数值输入
先返回空数组 (域外语义: 空候选集 → find_containing_element 返回 -1)。
**有限输入行为逐位不变** — 本文件金标准 = 修复前基线录制。

判别性: 回滚本提交 → inf 坐标裸 OverflowError 回归。
"""
import numpy as np
import pytest

from fem2d.mesh import Mesh

NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def _mesh():
    m = Mesh(NODES, ELEMS, E=2.1e11, nu=0.3, thickness=0.01)
    m.build_connectivity()
    return m


# ── 判别性红侧: inf 必须被拒绝而非裸 OverflowError ──

def test_candidates_inf_returns_empty():
    # 回滚 → 裸 OverflowError "cannot convert float infinity to integer"
    m = _mesh()
    out = m.locator.candidates(float("inf"), 0.5)
    assert isinstance(out, np.ndarray) and out.size == 0


def test_candidates_neg_inf_returns_empty():
    m = _mesh()
    out = m.locator.candidates(float("-inf"), 0.5)
    assert out.size == 0


def test_candidates_nan_returns_empty():
    m = _mesh()
    assert m.locator.candidates(float("nan"), 0.5).size == 0


def test_find_containing_element_inf_returns_minus_one():
    # 回滚 → 裸 OverflowError
    m = _mesh()
    assert m.element_kernel.find_containing_element(m, float("inf"), 0.5) == -1


def test_point_in_element_inf_still_valueerror():
    # 文档化入口保持既有契约 (前置有限性检查 → 带上下文 ValueError)
    from fem2d.stress import point_in_element
    with pytest.raises(ValueError, match="有限数值"):
        point_in_element(_mesh(), float("inf"), 0.5)


# ── 冻结区: 有限输入逐位不变 (金标准 = 修复前基线录制) ──

def test_candidates_finite_bit_exact_gold():
    m = _mesh()
    np.testing.assert_array_equal(
        m.locator.candidates(0.5, 0.5), np.array([0, 1]))
    np.testing.assert_array_equal(
        m.locator.candidates(1.0, 0.0), np.array([0, 1]))
    assert m.locator.candidates(-5.0, 0.5).size == 0


def test_find_containing_element_finite_bit_exact_gold():
    m = _mesh()
    assert m.element_kernel.find_containing_element(m, 0.5, 0.5) == 0
    assert m.element_kernel.find_containing_element(m, 1.0, 1.0) == 0
    assert m.element_kernel.find_containing_element(m, 1.0, 0.0) == 0
    assert m.element_kernel.find_containing_element(m, 0.2, 0.2) == 0
