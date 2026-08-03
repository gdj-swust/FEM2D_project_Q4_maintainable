"""error_est.py 分支补测 — 包 2 覆盖率任务.

未覆盖行集中: 恢复形状/权重契约守卫、恢复积分点形状冲突、外部内核
兼容路径 (shape/weights 均 None)、未知方法拒绝、退化内部边过滤、
sigma_ref 显式参考、残差估计器 (Verfürth) 的体力/加载边分支。

判别性: 断言具体异常消息/数值结果/数组形状。
"""
import types

import numpy as np
import pytest

from fem2d.error_est import (
    _element_energy_errors,
    _traction_jump_arrays,
    compute_traction_jumps,
    element_refinement_indicator,
    estimate,
)
from fem2d.mesh import Mesh


def _two_tri(body_force=None, surface_tractions=None):
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        elem_type="CPS3",
        body_force=body_force,
        surface_tractions=surface_tractions)


class _CompatKernel:
    """外部内核: 无批量恢复 (shape/weights 均 None) → 逐单元路径."""

    name = "compat"

    @staticmethod
    def recovery_shape_matrix(mesh):
        return None

    @staticmethod
    def recovery_weights(mesh):
        return None

    @staticmethod
    def recovery_quadrature(mesh, eid):
        # 分片单位阵: 行和=1 (真实内核的分片单位性质), 否则 σ* 被放大
        return (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                np.array([0.5, 0.5]))


class _CompatKernelZeroW:
    name = "compat0"

    @staticmethod
    def recovery_shape_matrix(mesh):
        return None

    @staticmethod
    def recovery_weights(mesh):
        return None

    @staticmethod
    def recovery_quadrature(mesh, eid):
        return (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                np.zeros(2))


def _fake_mesh(kernel, n_elem=2, npe=3):
    return types.SimpleNamespace(
        element_kernel=kernel,
        elements=np.zeros((n_elem, npe), dtype=int),
        n_elements=n_elem,
        thickness=1.0)


# ═══════════════════════════════════════════════════════════════
# _element_energy_errors 契约守卫
# ═══════════════════════════════════════════════════════════════

def test_energy_errors_shape_matrix_invalid():
    """恢复形状矩阵列数 ≠ 单元节点数 → ValueError."""
    kernel = types.SimpleNamespace(
        name="k", nodes_per_element=3,
        recovery_shape_matrix=lambda m: np.ones((4, 2)),
        recovery_weights=lambda m: np.ones((2, 2)))
    mesh = _fake_mesh(kernel)
    with pytest.raises(ValueError, match="shape matrix has invalid shape"):
        _element_energy_errors(
            mesh, "SPR", np.zeros((2, 3)), None, np.zeros((6, 3)),
            np.eye(3))


def test_energy_errors_weights_shape_invalid():
    """恢复权重形状 ≠ (n_elem, n_sample) → ValueError."""
    kernel = types.SimpleNamespace(
        name="k", nodes_per_element=3,
        recovery_shape_matrix=lambda m: np.ones((2, 3)),
        recovery_weights=lambda m: np.ones((5, 2)))
    mesh = _fake_mesh(kernel)
    with pytest.raises(ValueError, match="recovery weights have shape"):
        _element_energy_errors(
            mesh, "SPR", np.zeros((2, 3)), None, np.zeros((6, 3)),
            np.eye(3))


def test_energy_errors_stress_qp_shape_mismatch():
    """恢复积分点形状 ≠ stress_qp → ValueError (曾静默广播错位)."""
    kernel = types.SimpleNamespace(
        name="k", nodes_per_element=3,
        recovery_shape_matrix=lambda m: np.ones((2, 3)),
        recovery_weights=lambda m: np.ones((2, 2)))
    mesh = _fake_mesh(kernel)
    stress_qp = np.zeros((2, 2, 2))       # (n_elem, 2, 3) 期望
    with pytest.raises(ValueError, match="Recovered quadrature has shape"):
        _element_energy_errors(
            mesh, "SPR", np.zeros((2, 3)), stress_qp, np.zeros((6, 3)),
            np.eye(3))


def test_energy_errors_compat_kernel_values():
    """外部内核 (无批量恢复) → 逐单元路径数值正确."""
    mesh = _fake_mesh(_CompatKernel())
    stress = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    s_star = np.tile([1.0, 0.0, 0.0], (6, 1))
    err, rec, s_scale, use_scale, vol_sqrt = _element_energy_errors(
        mesh, "SPR", stress, None, s_star, np.eye(3))
    assert use_scale is True
    assert err.shape == (2,) and rec.shape == (2,)
    assert np.allclose(err, 0.0)          # σ* == σ → 零误差
    assert np.all(rec > 0.0)
    assert vol_sqrt > 0.0


def test_energy_errors_compat_kernel_zero_weights():
    """全零积分权 (maxw=0) → 能量全零, 不除零不崩溃."""
    mesh = _fake_mesh(_CompatKernelZeroW())
    stress = np.ones((2, 3))
    err, rec, _s_scale, use_scale, vol_sqrt = _element_energy_errors(
        mesh, "SPR", stress, None, np.ones((6, 3)), np.eye(3))
    assert use_scale is True
    assert np.allclose(err, 0.0) and np.allclose(rec, 0.0)
    assert vol_sqrt == 0.0


# ═══════════════════════════════════════════════════════════════
# estimate 方法契约
# ═══════════════════════════════════════════════════════════════

def test_estimate_unknown_method_rejected():
    """未知误差方法 → ValueError (曾静默当 SPR 算)."""
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3))}
    with pytest.raises(ValueError, match="Unknown error estimation method"):
        estimate(mesh, result, method="bogus", verbose=False)


# ═══════════════════════════════════════════════════════════════
# 牵引跳跃数组: 退化边过滤 / sigma_ref
# ═══════════════════════════════════════════════════════════════

def test_traction_jump_arrays_filters_zero_length_edges():
    """零长内部边 → 过滤 (曾 NaN 跳跃污染整体统计)."""
    mesh = _two_tri()
    mesh.build_connectivity()
    # 注入一条节点重合的内部边 (正常网格构造不会产生, 直接构造防御场景)
    mesh.internal_edge_data = np.array([[1, 2, 0, 1]])
    mesh._nodes = np.array(mesh._nodes)    # 可写副本
    mesh._nodes[2] = mesh._nodes[1]        # 节点 2 与 1 重合 → 零长边
    edge_data, lengths, jump_abs, jump_rel = _traction_jump_arrays(
        mesh, np.ones((2, 3)))
    assert len(edge_data) == 0             # 全被过滤
    assert jump_rel.shape == (0,)


def test_traction_jumps_sigma_ref_fixed_denominator():
    """sigma_ref → jump_rel = jump_abs / sigma_ref (跨网格可比)."""
    mesh = _two_tri()
    stress = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    jumps = compute_traction_jumps(mesh, stress, sigma_ref=1e6)
    assert len(jumps) == 1
    assert jumps[0]["jump_abs"] == pytest.approx(
        jumps[0]["jump_rel"] * 1e6, rel=1e-12)


def test_traction_jumps_sigma_ref_invalid_rejected():
    """sigma_ref 非正/非有限 → ValueError (曾 max(...,1e-30) 静默覆盖)."""
    mesh = _two_tri()
    stress = np.ones((2, 3))
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            compute_traction_jumps(mesh, stress, sigma_ref=bad)


# ═══════════════════════════════════════════════════════════════
# element_refinement_indicator (Verfürth)
# ═══════════════════════════════════════════════════════════════

def test_refinement_indicator_body_force_contribution():
    """体力残差项: h_K²·A_K·|f|² 进入 η_K (非零体力 → 指示值上升)."""
    mesh = _two_tri(body_force=(1e6, 0.0))
    mesh.fix_node(0, "both", 0.0)
    mesh.fix_node(3, "both", 0.0)
    result = {"stress": np.ones((2, 3))}
    eta = element_refinement_indicator(mesh, result)
    assert eta.shape == (2,)
    assert np.all(np.isfinite(eta))


def test_refinement_indicator_zero_body_force_skipped():
    """零体力 → 体力项跳过 (f_norm2==0 continue), 结果仍有限."""
    mesh = _two_tri(body_force=(0.0, 0.0))
    result = {"stress": np.zeros((2, 3))}
    eta = element_refinement_indicator(mesh, result)
    assert eta.shape == (2,) and np.all(np.isfinite(eta))
    assert np.all(eta == 0.0)             # 零应力零载荷 → 零指示


def test_refinement_indicator_nonmesh_edge_skipped():
    """面力指定到非网格边 (对角线) → 跳过不崩溃."""
    mesh = _two_tri(surface_tractions=[
        {"nodes": (0, 3), "traction": (1e6, 0.0)}])   # 对角线, 非网格边
    result = {"stress": np.ones((2, 3))}
    eta = element_refinement_indicator(mesh, result)
    assert eta.shape == (2,) and np.all(np.isfinite(eta))


def test_refinement_indicator_pressure_callable():
    """加载边压力表达式 (callable) → 残差积分正确合并."""
    def _p(x, y):
        return 1e6 * (1.0 - x)
    mesh = _two_tri(surface_tractions=[
        {"nodes": (0, 1), "traction": (_p,), "is_pressure": True}])
    result = {"stress": np.ones((2, 3))}
    eta = element_refinement_indicator(mesh, result)
    assert eta.shape == (2,)
    assert eta[0] >= 0.0 and np.all(np.isfinite(eta))


def test_refinement_indicator_result_contract():
    """result 非 dict/缺 stress → ValueError (曾裸 KeyError)."""
    mesh = _two_tri()
    with pytest.raises(ValueError, match="必须"):
        element_refinement_indicator(mesh, {"u": np.zeros(8)})
