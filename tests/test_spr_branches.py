"""spr.py 分支补测 — 包 2 覆盖率任务.

未覆盖行集中: 恢复采样位置契约 (形状矩阵/逐单元回退)、输入维度守卫、
手工构造 node_to_elems 的兼容路径、精确路径的少量样本回退。

判别性: 断言具体异常消息/数组形状/数值结果。
"""
import numpy as np
import pytest

from fem2d.mesh import Mesh
from fem2d.spr import (
    _node_patch_csr,
    _prepare_samples,
    recovery_sample_positions,
    spr_recovery,
)


def _quad():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type="CPS4")


def _tri():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
        elements=np.array([[0, 1, 2]], dtype=int),
        elem_type="CPS3")


# ═══════════════════════════════════════════════════════════════
# recovery_sample_positions
# ═══════════════════════════════════════════════════════════════

def test_recovery_positions_shape_matrix_mismatch(monkeypatch):
    """形状矩阵采样点数 ≠ 给定样本数 → ValueError (曾静默错位)."""
    mesh = _quad()
    monkeypatch.setattr(mesh.element_kernel, "recovery_shape_matrix",
                        lambda m: np.ones((4, 2)))
    with pytest.raises(ValueError, match="recovery rule provides"):
        recovery_sample_positions(mesh, 3)


def test_recovery_positions_quadrature_fallback(monkeypatch):
    """无形状矩阵 → 逐单元积分规则路径, 位置 = shape @ nodes."""
    mesh = _quad()
    monkeypatch.setattr(mesh.element_kernel, "recovery_shape_matrix",
                        lambda m: None)
    monkeypatch.setattr(
        mesh.element_kernel, "recovery_quadrature",
        lambda m, eid: (np.eye(4), np.ones(4)))
    positions = recovery_sample_positions(mesh, 4)
    assert positions.shape == (1, 4, 2)
    assert np.allclose(positions[0], mesh.nodes)   # 单位阵 → 节点坐标


def test_recovery_positions_quadrature_mismatch(monkeypatch):
    """回退路径采样点数不符 → ValueError."""
    mesh = _quad()
    monkeypatch.setattr(mesh.element_kernel, "recovery_shape_matrix",
                        lambda m: None)
    monkeypatch.setattr(
        mesh.element_kernel, "recovery_quadrature",
        lambda m, eid: (np.ones((2, 4)), np.ones(2)))
    with pytest.raises(ValueError, match="kernel returned"):
        recovery_sample_positions(mesh, 3)


# ═══════════════════════════════════════════════════════════════
# _prepare_samples
# ═══════════════════════════════════════════════════════════════

def test_prepare_samples_rejects_wrong_ndim():
    """维度不在 (ne,ncomp)/(ne,nqp,ncomp) → ValueError."""
    mesh = _quad()
    with pytest.raises(ValueError, match="must have shape"):
        _prepare_samples(mesh, np.zeros((1, 2, 3, 4)))


def test_prepare_samples_element_count_mismatch():
    """样本数 ≠ 单元数 → ValueError."""
    mesh = _quad()
    with pytest.raises(ValueError, match="first dimension must be"):
        _prepare_samples(mesh, np.zeros((2, 3)))


def test_prepare_samples_nan_rejected():
    """样本含 NaN/Inf → ValueError (曾静默恢复出 NaN 云图)."""
    mesh = _quad()
    with pytest.raises(ValueError, match="NaN/Inf"):
        _prepare_samples(mesh, np.array([[1.0, 0.0, np.nan]]))


def test_prepare_samples_representative_positions():
    """(ne,ncomp) → 单元代表位置 = 节点坐标均值 (CST 形心约定)."""
    mesh = _tri()
    positions, values = _prepare_samples(mesh, np.ones((1, 3)))
    assert positions.shape == (1, 1, 2)
    assert np.allclose(positions[0, 0], mesh.nodes.mean(axis=0))


# ═══════════════════════════════════════════════════════════════
# _node_patch_csr 兼容路径 (手工构造网格)
# ═══════════════════════════════════════════════════════════════

def test_node_patch_csr_list_fallback(monkeypatch):
    """node_to_elems 为 list[list] (手工构造) → 兼容展开."""
    mesh = _quad()
    monkeypatch.setattr(
        mesh, "node_to_elems",
        [[0], [0], [0], [0]])
    ptr, flat = _node_patch_csr(mesh)
    assert flat.tolist() == [0, 0, 0, 0]
    assert ptr.tolist() == [0, 1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════════
# spr_recovery: 少量样本回退路径
# ═══════════════════════════════════════════════════════════════

def test_spr_single_element_constant_field_exact():
    """单单元常应力场 → 节点恢复精确 (零误差, 非恒真)."""
    mesh = _tri()
    recovered = spr_recovery(mesh, np.array([[2.5, 1.0, -0.5]]))
    assert recovered.shape == (3, 3)
    assert np.allclose(recovered, [2.5, 1.0, -0.5], rtol=1e-12)


def test_spr_few_sample_patch_falls_back_to_mean():
    """节点 patch 样本不足 (<3) → 回退算术平均 (原逐点路径)."""
    mesh = Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        elem_type="CPS3")
    # 单分量: 每单元 1 个代表样本 → 每节点 patch 最多 2 样本 → 平均回退
    recovered = spr_recovery(mesh, np.array([[2.0], [4.0]]))
    assert recovered.shape == (4, 1)
    assert np.all(np.isfinite(recovered))
    # 内部节点 (1,2) 同时邻接两单元 → 均值 3.0 (判别性)
    assert recovered[1, 0] == pytest.approx(3.0)
    assert recovered[2, 0] == pytest.approx(3.0)
