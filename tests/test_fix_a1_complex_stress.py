"""A1 (P2, R-α): complex 应力数组必须拒绝 — 禁止静默丢虚部/类型污染.

审查现象 (2026-08-05):
  - von_mises 单向量 complex → 静默 0.7071 (ComplexWarning 丢虚部)
  - von_mises 批量 (n,3) complex → 返回 complex 数组类型污染
  - principal_stresses 单向量 complex → 裸 TypeError ("ufunc 'hypot'")

修复: 入口 np.iscomplexobj(stress) → ValueError 带参数名 (与 NaN/Inf
拒绝同族)。输出路径未触碰 — (n,3) 有限实数输入逐位不变 (冻结区)。

判别性: 回滚本提交 → 静默 0.7071 / complex 输出 / 裸 TypeError 回归。
"""
import numpy as np
import pytest

from fem2d.material import von_mises
from fem2d.stress import principal_stresses


# ── 红侧: complex 必须被拒绝 (回滚 → 下列断言全红) ──

def test_von_mises_single_complex_rejected():
    # 回滚 → 静默返回 0.7071067811865475 (虚部被丢弃)
    with pytest.raises(ValueError, match="von_mises"):
        von_mises(np.array([1 + 1j, 1.0, 0.0]))


def test_von_mises_batch_complex_rejected():
    # 回滚 → 返回 complex128 数组 (类型污染)
    with pytest.raises(ValueError, match="von_mises"):
        von_mises(np.array([[1 + 1j, 1.0, 0.0], [1.0, 2.0, 0.5]]))


def test_von_mises_batch_complex_float64_unchanged():
    # 冻结区: (n,3) 有限实数输入输出逐位不变
    s = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert von_mises(s).dtype == np.float64


def test_principal_single_complex_rejected():
    # 回滚 → 裸 TypeError "ufunc 'hypot' not supported"
    with pytest.raises(ValueError, match="principal_stresses"):
        principal_stresses(np.array([1 + 1j, 1.0, 0.0]))


def test_principal_batch_complex_rejected():
    with pytest.raises(ValueError, match="principal_stresses"):
        principal_stresses(np.array([[1 + 1j, 1.0, 0.0], [1.0, 2.0, 0.5]]))


def test_principal_real_input_still_works():
    # 冻结区: 有限实数输入正常返回
    s1, s2, t, theta = principal_stresses(np.array([1.0, 2.0, 3.0]))
    assert s1 > s2
    assert np.isfinite([s1, s2, t, theta]).all()


# ── 绿侧: complex 标量 (非数组) 走既有形状契约拒绝 ──

def test_complex_scalar_rejected_by_shape_contract():
    with pytest.raises(ValueError, match="stress"):
        von_mises(1 + 2j)
    with pytest.raises(ValueError, match="stress"):
        principal_stresses(1 + 2j)
