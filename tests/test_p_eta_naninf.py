"""P-η: error_est 统一 NaN/Inf 入口防护 + 批量归约/向量化重构等价性.

任务1 (外部审查 2026-08-06): _traction_jump_arrays 只查 complex 与形状,
不查有限性 — NaN 应力被 L366 np.where 静默归零成 jump_rel=0.0 (当作
"无跳跃"), 单单元空数据路径直接返回空列表吞掉非法数据. 本文件锁定
修复后行为: 任何 NaN/Inf 应力入口 → ValueError, 非法数据不得静默.

任务2/3: logaddexp 归约 (排序+reduceat) 与残差循环向量化重构 —
与基线逐单元/逐边实现逐位一致 (随机网格自对比).
"""
import math

import numpy as np
import pytest

from fem2d.element import evaluate_vector_field
from fem2d.error_est import (
    _logaddexp_scatter,
    compute_traction_jumps,
    element_refinement_indicator,
    estimate,
)
from fem2d.loads_core import LINE_GAUSS
from fem2d.mesh import Mesh


def _two_tri(body_force=None, surface_tractions=None):
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        elem_type="CPS3",
        body_force=body_force,
        surface_tractions=surface_tractions)


def _single_tri():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
        elements=np.array([[0, 1, 2]], dtype=int),
        elem_type="CPS3")


# ═══════════════════════════════════════════════════════════════
# 任务1: NaN/Inf 入口防护 (红侧: 修复前静默接受/静默忽略)
# ═══════════════════════════════════════════════════════════════

def test_traction_jumps_nan_stress_rejected():
    """单元应力含 NaN → ValueError (红侧: 返回 jump_abs=nan, jump_rel=0.0)."""
    mesh = _two_tri()
    stress = np.array([[1e6, 0.0, np.nan], [2e6, 0.0, 0.0]])
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_traction_jumps(mesh, stress)


def test_traction_jumps_inf_stress_rejected():
    """单元应力含 Inf → ValueError."""
    mesh = _two_tri()
    stress = np.array([[1e6, np.inf, 0.0], [2e6, 0.0, 0.0]])
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_traction_jumps(mesh, stress)


def test_traction_jumps_single_element_nan_rejected():
    """单单元网格 (无内部边, 空数据路径) + NaN → ValueError —
    有限性校验必须先于空数据提前返回 (红侧: 静默返回 [])."""
    mesh = _single_tri()
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_traction_jumps(mesh, np.array([[1e6, np.nan, 0.0]]))


def test_refinement_indicator_nan_stress_rejected():
    """两单元网格 + NaN 单元应力 → ValueError (红侧: 返回 [nan nan])."""
    mesh = _two_tri()
    result = {"stress": np.array([[1e6, np.nan, 0.0], [1e6, 0.0, 0.0]])}
    with pytest.raises(ValueError, match="NaN/Inf"):
        element_refinement_indicator(mesh, result)


def test_refinement_indicator_nan_stress_qp_rejected():
    """result['stress_qp'] 含 NaN → ValueError (红侧: 被静默忽略)."""
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3)),
              "stress_qp": np.full((2, 2, 3), np.nan)}
    with pytest.raises(ValueError, match="stress_qp"):
        element_refinement_indicator(mesh, result)


def test_refinement_indicator_finite_stress_qp_ok():
    """有限 stress_qp 不受新入口防护影响 (solve() 正常结果不被误拒)."""
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3)),
              "stress_qp": np.ones((2, 2, 3))}
    eta = element_refinement_indicator(mesh, result)
    assert eta.shape == (2,) and np.all(np.isfinite(eta))


def test_estimate_nan_stress_qp_rejected():
    """estimate 的 stress_qp 入口含 NaN → ValueError (统一防护覆盖)."""
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3)),
              "stress_qp": np.full((2, 2, 3), np.nan)}
    with pytest.raises(ValueError, match="NaN/Inf"):
        estimate(mesh, result, verbose=False)


# ═══════════════════════════════════════════════════════════════
# 任务2: logaddexp 归约等价性 (排序+reduceat vs ufunc.at)
# ═══════════════════════════════════════════════════════════════

def test_logaddexp_scatter_matches_ufunc_at():
    """排序+reduceat 归约与 np.logaddexp.at 逐位一致 (随机含重复 eid/-inf).

    内部调用场景: eta_log 基恒为 -inf (element_refinement_indicator
    用 np.full(n_elem, -inf) 初始化), logaddexp(-inf, x) ≡ x, 归约
    结合序重排不引入任何 ulp 差异.
    """
    rng = np.random.default_rng(20260806)
    for _ in range(50):
        n = 200
        eids = rng.integers(0, 40, size=rng.integers(1, 600))
        terms = rng.normal(0.0, 10.0, size=len(eids))
        terms[rng.random(len(eids)) < 0.3] = -np.inf   # 零跳跃边
        got = np.full(n, -np.inf)
        _logaddexp_scatter(got, eids, terms)
        ref = np.full(n, -np.inf)
        np.logaddexp.at(ref, eids, terms)
        assert np.array_equal(got, ref)
