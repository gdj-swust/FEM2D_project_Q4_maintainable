"""solver.py 分支补测 — 包 2 覆盖率任务.

未覆盖行集中: estimate_condition 稀疏特征值路径/状态分级/异常归因、
奇异性守卫 (MatrixRankWarning → 响亮错误)、penalty 求解器白名单、
残差溢出回退、平衡检查空反力、Q4R 长宽比/沙漏能分级告警、结果有限性。

判别性: 断言具体状态字符串/异常消息/警告文本/日志内容。
"""
import warnings

import numpy as np
import pytest
from scipy import sparse

import fem2d.solver as solver_mod
from fem2d.mesh import Mesh
from fem2d.solver import (
    _check_result_finite,
    _check_solution_finite,
    _compute_element_response,
    _compute_residual,
    _condition_report,
    _global_balance_check,
    _hourglass_monitor,
    _q4r_aspect_ratio_warning,
    _solve_linear_system,
    _solve_with_singular_guard,
    estimate_condition,
)


def _quad(elem_type="CPS4"):
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type=elem_type)


def _spdiag(vals):
    return sparse.diags(np.asarray(vals, dtype=float)).tocsr()


# ═══════════════════════════════════════════════════════════════
# estimate_condition
# ═══════════════════════════════════════════════════════════════

def test_estimate_condition_sparse_path():
    """大矩阵 → 稀疏特征值路径 (eigsh LM + shift-invert λ_min)."""
    K = _spdiag(np.linspace(1.0, 10.0, 500))
    info = estimate_condition(K)          # auto → n≥500 → sparse
    assert info["status"] == "GOOD"
    assert info["condition_number"] == pytest.approx(10.0, rel=1e-6)


def test_estimate_condition_warn_status():
    """κ ∈ (1e12, 1e14] → WARN."""
    K = _spdiag(np.linspace(1.0, 1e13, 64))
    info = estimate_condition(K, method="dense")
    assert info["status"] == "WARN"
    assert info["condition_number"] == pytest.approx(1e13, rel=1e-6)


def test_estimate_condition_ok_status():
    """κ ∈ (1e8, 1e12] → OK."""
    K = _spdiag(np.linspace(1.0, 1e10, 64))
    info = estimate_condition(K, method="dense")
    assert info["status"] == "OK"


def test_estimate_condition_singular_status():
    """特征值求解失败 (NaN 矩阵) → SINGULAR? 带错误信息."""
    K = sparse.csr_matrix(np.full((8, 8), np.nan))
    info = estimate_condition(K, method="dense")
    assert info["status"] == "SINGULAR?"
    assert info["condition_number"] is None
    assert info["error"]


# ═══════════════════════════════════════════════════════════════
# estimate_condition 返回形状统一 (pkg11 A13)
# ═══════════════════════════════════════════════════════════════

_EC_KEYS = {"condition_number", "lambda_min", "lambda_max",
            "digits_lost", "status", "error"}


def test_estimate_condition_failure_shape_matches_success():
    """失败路径补齐全部键 (None 填充) — 曾缺 4 键, 逐键访问崩溃.

    判别性: 对成功/失败各取一次返回, 键集必须完全一致。
    """
    ok = estimate_condition(_spdiag([1.0, 2.0, 3.0]), method="dense")
    assert set(ok) == _EC_KEYS
    assert ok["error"] is None
    fail = estimate_condition(
        sparse.csr_matrix(np.full((4, 4), np.nan)), method="dense")
    assert set(fail) == _EC_KEYS
    assert fail["condition_number"] is None
    assert fail["lambda_min"] is None
    assert fail["lambda_max"] is None
    assert fail["digits_lost"] is None
    assert fail["status"] == "SINGULAR?"
    # 稀疏路径的奇异矩阵 (零矩阵 shift-invert 必失败) 同样补齐
    fail_sparse = estimate_condition(
        sparse.csr_matrix(np.zeros((4, 4))), method="sparse")
    assert set(fail_sparse) == _EC_KEYS
    assert fail_sparse["status"] == "SINGULAR?"


def test_estimate_condition_skip_status():
    """非特征值类异常 (空矩阵 IndexError) → SKIP (与 SINGULAR? 区分)."""
    K = sparse.csr_matrix((0, 0))
    info = estimate_condition(K, method="dense")
    assert info["status"] == "SKIP"
    assert "IndexError" in info["error"]


# ═══════════════════════════════════════════════════════════════
# _solve_with_singular_guard
# ═══════════════════════════════════════════════════════════════

def test_singular_guard_converts_matrix_rank_warning():
    """spsolve 发 MatrixRankWarning → 转为响亮 RuntimeError (曾返回 NaN)."""
    def _fn():
        warnings.warn("Matrix is exactly singular",
                      sparse.linalg.MatrixRankWarning)
        return np.zeros(3)
    with pytest.raises(RuntimeError, match="singular or ill-conditioned"):
        _solve_with_singular_guard(_fn)


def test_singular_guard_passes_clean_result():
    """无警告 → 原样返回."""
    assert _solve_with_singular_guard(lambda: 42) == 42


# ═══════════════════════════════════════════════════════════════
# _solve_linear_system 分支
# ═══════════════════════════════════════════════════════════════

def _solve_args():
    K = _spdiag([2.0, 2.0, 2.0, 2.0])
    F = np.zeros(4)
    return dict(K=K, F=F, free_dofs=np.array([2, 3]),
                fixed_dofs=np.array([0, 1]), prescribed=np.zeros(2),
                method="elimination", linear_solver="direct",
                n_dof=4, log=lambda *a, **k: None)


def test_solve_linear_system_penalty_rejects_cg():
    """penalty 只允许 direct → cg 显式拒绝 (曾静默降级)."""
    args = _solve_args()
    args["method"] = "penalty"
    args["linear_solver"] = "cg"
    with pytest.raises(ValueError, match="require the direct solver"):
        _solve_linear_system(**args)


def test_solve_linear_system_unknown_method():
    """未知 method → ValueError."""
    args = _solve_args()
    args["method"] = "bogus"
    with pytest.raises(ValueError, match="Unknown method"):
        _solve_linear_system(**args)


def test_solve_linear_system_unknown_solver():
    """未知 linear_solver → ValueError (纯 Dirichlet 前校验, 曾静默成功)."""
    args = _solve_args()
    args["linear_solver"] = "bogus"
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        _solve_linear_system(**args)


def test_solve_linear_system_solver_validated_before_all_branches():
    """非法 solver 名在任何分支入口统一拒绝 (pkg11 A1).

    判别性: 纯 Dirichlet (全约束, 无需求解) 与 penalty 分支同样必须
    在入口拒绝 "spqr" — 曾 elimination 分支二次校验与入口校验逐字
    重复, 任一分支漏校验都会静默放行.
    """
    args = _solve_args()
    args["linear_solver"] = "spqr"
    args["free_dofs"] = np.array([], dtype=int)   # 纯 Dirichlet
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        _solve_linear_system(**args)
    args["free_dofs"] = np.array([2, 3])
    args["method"] = "penalty"
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        _solve_linear_system(**args)


def test_solve_linear_system_cg_block_alias_normalized():
    """cg-block 兼容别名在 elimination 分支归一为 cg (pkg11 A1).

    判别性: 别名映射只发生一次 — 归一后的 solver_key 直接用于
    apply_elimination, 迭代信息须按 cg 路径返回.
    """
    args = _solve_args()
    args["linear_solver"] = "cg-block"
    u, reactions, _kmod, _fmod, info, _dirichlet = _solve_linear_system(**args)
    assert info["name"] == "cg"
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(reactions))


# ═══════════════════════════════════════════════════════════════
# 有限性守卫
# ═══════════════════════════════════════════════════════════════

def test_check_solution_finite_nan_displacement():
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        _check_solution_finite(np.array([1.0, np.nan]), np.zeros(2),
                               "elimination")


def test_check_solution_finite_nan_reactions():
    with pytest.raises(RuntimeError, match="Reaction forces contain NaN"):
        _check_solution_finite(np.zeros(2), np.array([np.nan, 0.0]),
                               "elimination")


def test_check_result_finite_raises():
    """应力含 NaN → RuntimeError 指名失败数组."""
    with pytest.raises(RuntimeError, match="Result stress contains NaN/Inf"):
        _check_result_finite(np.array([[np.nan, 0, 0]]), np.zeros((1, 3)),
                             np.zeros(1))


# ═══════════════════════════════════════════════════════════════
# _compute_residual
# ═══════════════════════════════════════════════════════════════

def test_compute_residual_overflow_forces_residual_one():
    """中间量溢出 → 相对残差置 1.0 → 必被拒绝 (曾显示'残差 0'假象)."""
    K = sparse.csr_matrix(np.array([[1e200]]))
    F = np.array([1e200])
    logs = []
    with pytest.raises(RuntimeError, match="too large"):
        _compute_residual(K, F, None, None, np.array([1e200]),
                          np.array([0]), "elimination", False, logs.append)
    assert any("Residual" in line for line in logs)


def test_compute_residual_trivial_solution_ok():
    """零载荷零位移 → 平凡解 [OK] 标签."""
    K = sparse.csr_matrix(np.array([[2.0]]))
    logs = []
    residual, residual_abs = _compute_residual(
        K, np.zeros(1), None, None, np.zeros(1), np.array([0]),
        "elimination", False, logs.append)
    assert residual == 0.0 and residual_abs == 0.0
    assert any("trivial solution" in line for line in logs)


def test_compute_residual_zero_u_nonzero_f_warns():
    """零位移但载荷非零 → WARN 日志先打, 随后残差检查必拒 (u≡0 时
    residual = |F|/|F| = 1.0 > 1e-3)."""
    K = sparse.csr_matrix(np.array([[2.0]]))
    logs = []
    with pytest.raises(RuntimeError, match="too large"):
        _compute_residual(
            K, np.array([5.0]), None, None, np.zeros(1), np.array([0]),
            "elimination", False, logs.append)
    assert any("||K·u - F||" in line for line in logs)


# ═══════════════════════════════════════════════════════════════
# _global_balance_check / _compute_element_response
# ═══════════════════════════════════════════════════════════════

def test_global_balance_check_no_reactions_none():
    """无反力 (penalty 后无固定 DOF) → 返回 None."""
    mesh = _quad()
    assert _global_balance_check(
        mesh, sparse.eye(8), np.zeros(8), np.zeros(8), None,
        np.array([], dtype=int), False, lambda *a, **k: None) is None


def test_compute_element_response_nonpositive_area(monkeypatch):
    """积分点面积权重非正 → RuntimeError (曾静默除零)."""
    mesh = _quad()
    monkeypatch.setattr(
        mesh.element_kernel, "response_at_quadrature",
        lambda m, u: (np.zeros((1, 4, 3)), np.zeros((1, 4, 3)),
                      np.array([[-2.0, 0.5, 0.5, 0.5]])))
    with pytest.raises(RuntimeError, match="non-positive area"):
        _compute_element_response(mesh, np.zeros(8))


# ═══════════════════════════════════════════════════════════════
# Q4R 告警分级
# ═══════════════════════════════════════════════════════════════

def test_q4r_aspect_ratio_warning_documented_failure_zone():
    """长宽比 ≥50 → 文档失效区强警告."""
    mesh = _quad(elem_type="CPS4R")
    mesh._nodes = np.array([[0., 0.], [50., 0.], [50., 1.], [0., 1.]])
    with pytest.warns(RuntimeWarning, match="长宽比最大 50"):
        _q4r_aspect_ratio_warning(mesh)


def test_hourglass_ratio_dominant_warns():
    """沙漏能占比 >90% → 结果不可靠强警告."""
    mesh = _quad(elem_type="CPS4R")
    K = sparse.eye(8)
    u = np.full(8, 0.01)                 # 内能 ~ 4e-4
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mesh.element_kernel, "hourglass_energy",
                        lambda m, u_e: np.array([100.0]))
    with pytest.warns(RuntimeWarning, match="hourglass modes dominate"):
        _hourglass_monitor(mesh, K, u, np.zeros(8), lambda *a, **k: None)
    monkeypatch.undo()


def test_hourglass_ratio_high_warns():
    """沙漏能占比 30-90% → 可靠性警告."""
    mesh = _quad(elem_type="CPS4R")
    K = sparse.eye(8)
    u = np.full(8, 0.01)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mesh.element_kernel, "hourglass_energy",
                        lambda m, u_e: np.array([2e-4]))
    with pytest.warns(RuntimeWarning, match="> 30%"):
        _hourglass_monitor(mesh, K, u, np.zeros(8), lambda *a, **k: None)
    monkeypatch.undo()


def test_hourglass_internal_energy_nonfinite_raises():
    """内能溢出 → RuntimeError (曾返回'成功'的 inf 结果)."""
    mesh = _quad(elem_type="CPS4R")
    K = sparse.diags(np.full(8, 1e300)).tocsr()
    u = np.full(8, 1e300)
    with pytest.raises(RuntimeError, match="Internal energy"):
        _hourglass_monitor(mesh, K, u, np.zeros(8), lambda *a, **k: None)


# ═══════════════════════════════════════════════════════════════
# _condition_report
# ═══════════════════════════════════════════════════════════════

def test_condition_report_disabled_returns_none():
    """check_condition=False → None (不付特征值成本)."""
    assert _condition_report(sparse.eye(8), np.array([0, 1]), False,
                             lambda *a, **k: None) is None


def test_condition_report_large_system_warns(monkeypatch):
    """K_aa > 20000 DOF → 提示稀疏估计可能较慢."""
    logs = []
    K = sparse.eye(25001).tocsr()
    monkeypatch.setattr(
        solver_mod, "estimate_condition",
        lambda K: {"status": "GOOD", "condition_number": 1.0,
                    "digits_lost": 0.0})
    info = _condition_report(K, np.arange(25000), True, logs.append)
    assert any("请耐心等待" in line for line in logs)
    assert info["condition_number"] == 1.0
