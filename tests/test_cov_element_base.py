"""覆盖轮 C1 — element/base.py 缺口行 (19 行, 行为冻结区: 只测不改).

策略: 默认实现 (kernel 协议兜底) 与防御分支全部用最小测试内核触发 —
不触碰任何真实内核 (CST/Q4/Q4R/Q4I 冻结) 的行为.
"""
import numpy as np
import pytest

import fem2d.element.base as EB
from fem2d.element.base import (
    ElementKernel, evaluate_vector_field, register_element,
    get_element_kernel, registered_element_types,
)
from fem2d.mesh import Mesh


def _mesh():
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
             elements=np.array([[0, 1, 2], [0, 3, 2]]),
             E=1e6, nu=0.3, thickness=1.0)
    m.build_connectivity()
    return m


class _MinKernel(ElementKernel):
    """最小协议实现 — 全部默认实现 (兜底) 均保留原样."""

    name = "TESTK"
    aliases = ("TSTK",)
    nodes_per_element = 3
    local_edges = ((0, 1), (1, 2), (2, 0))

    def build_geometry(self, nodes, elements):
        return {}

    def stiffness_batch(self, mesh, element_slice=None):
        n = mesh.n_elements if element_slice is None else len(element_slice)
        return np.zeros((n, 6, 6))

    def jacobian_determinants(self, mesh):
        return np.ones((mesh.n_elements, 1))

    def body_force_vector(self, mesh, eid, body_force):
        return np.zeros(6)

    def shape_values_at(self, coords, x, y, tol=1e-12):
        return None

    def verify_mesh(self, mesh, verbose=True):
        return True


class _RespKernel(_MinKernel):
    """覆盖 compute_response → response_at_quadrature 默认实现可达."""

    name = "RESPK"

    def compute_response(self, mesh, u_e):
        n = mesh.n_elements
        return (np.ones((n, 3)), np.zeros((n, 3)), np.ones(n))


# ── 默认实现兜底 (协议级, 与任何真实内核无关) ──────────────────────────────

def test_stiffness_default_delegates_to_batch():
    """stiffness(eid) 默认实现 → stiffness_batch[m_eid]."""
    k = _MinKernel()
    assert k.stiffness(_mesh(), 1).shape == (6, 6)


def test_degeneracy_measure_default_none():
    """默认退化度量 None → jacobian_report 只做 detJ 尺度检查."""
    k = _MinKernel()
    assert k.degeneracy_measure(_mesh()) is None


def test_body_force_batch_default_none():
    """默认批量体力 None → 调用方回退逐单元路径."""
    k = _MinKernel()
    assert k.body_force_batch(_mesh(), (0.0, -1.0)) is None


def test_recovery_quadrature_default_centroid():
    """默认恢复积分: 形心单样本 + 面积权重 (1/n 等分)."""
    k = _MinKernel()
    mesh = _mesh()
    N, dA = k.recovery_quadrature(mesh, 0)
    assert N.shape == (1, 3) and np.allclose(N, 1.0 / 3)
    assert dA == pytest.approx(mesh.areas[0])


def test_recovery_shape_matrix_default_none():
    """默认恢复形状矩阵 None → 逐元素 recovery_quadrature 路径."""
    k = _MinKernel()
    assert k.recovery_shape_matrix(_mesh()) is None


def test_recovery_weights_default_none():
    k = _MinKernel()
    assert k.recovery_weights(_mesh()) is None


def test_response_at_quadrature_default_calls_compute():
    """response_at_quadrature 默认实现 → 单样本展开 compute_response."""
    k = _RespKernel()
    mesh = _mesh()
    stress_qp, strain_qp, dA = k.response_at_quadrature(mesh, np.zeros(6))
    assert stress_qp.shape == (2, 1, 3)
    assert strain_qp.shape == (2, 1, 3)
    assert dA.shape == (2, 1) and np.allclose(dA[:, 0], mesh.areas)


def test_compute_response_mutual_recursion_guard():
    """两默认实现互调用 → 显式 NotImplementedError (非 RecursionError)."""
    k = _MinKernel()
    with pytest.raises(NotImplementedError, match="response_at_quadrature"):
        k.compute_response(_mesh(), np.zeros(6))


# ── jacobian_report 防御分支 ────────────────────────────────────────────────

def test_jacobian_report_nonfinite_raises():
    """detJ 含 NaN/Inf (超大坐标溢出) → RuntimeError 而非误导性 ok=True."""
    class _NaN(_MinKernel):
        def jacobian_determinants(self, mesh):
            return np.array([[np.inf], [np.nan]])
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _NaN().jacobian_report(_mesh())


def test_jacobian_report_shape_mismatch_raises():
    """detJ 第一维 ≠ 单元数 → RuntimeError (内核契约违规)."""
    class _BadShape(_MinKernel):
        def jacobian_determinants(self, mesh):
            return np.ones((3, 1))
    with pytest.raises(RuntimeError, match="shape"):
        _BadShape().jacobian_report(_mesh())


def test_jacobian_report_measure_shape_mismatch_raises():
    """degeneracy_measure 形状错误 → RuntimeError."""
    class _BadMeasure(_MinKernel):
        def degeneracy_measure(self, mesh):
            return np.ones(3)
    with pytest.raises(RuntimeError, match="degeneracy_measure"):
        _BadMeasure().jacobian_report(_mesh())


# ── 注册表防御分支 ──────────────────────────────────────────────────────────

def test_register_element_empty_name():
    """空类型名 (纯空白) → ValueError."""
    class _Empty(_MinKernel):
        name = "   "
    with pytest.raises(ValueError, match="cannot be empty"):
        register_element(_Empty())


def test_register_element_duplicate_rejected(monkeypatch):
    """同名不同内核实例 → ValueError (防 import 顺序静默换内核)."""
    k1, k2 = _MinKernel(), _MinKernel()
    register_element(k1)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_element(k2)
    finally:
        for key in (k1.name, *k1.aliases):
            EB._REGISTRY.pop(str(key).strip().upper(), None)


def test_register_element_same_instance_idempotent(monkeypatch):
    """同一实例重复注册 → 幂等 (不抛)."""
    k = _MinKernel()
    register_element(k)
    try:
        register_element(k)
        assert get_element_kernel("TESTK") is k
        assert "TESTK" in registered_element_types()
    finally:
        for key in (k.name, *k.aliases):
            EB._REGISTRY.pop(str(key).strip().upper(), None)


# ── evaluate_vector_field 分支 ──────────────────────────────────────────────

def test_evaluate_vector_field_callable_tuple():
    """callable 返回二元组 → 解包 (line 329)."""
    assert evaluate_vector_field(lambda x, y: (2.0, -3.0), 0.5, 0.5) == \
        (2.0, -3.0)


def test_evaluate_vector_field_ndarray_direct():
    """ndarray 常量场 → else 分支直接索引 (line 334)."""
    assert evaluate_vector_field(np.array([1.5, 2.5]), 0.5, 0.5) == \
        (1.5, 2.5)


def test_evaluate_vector_field_callable_bad_shape_raises():
    """callable 返回标量/三维 → ValueError 带求值点."""
    with pytest.raises(ValueError, match="二元组"):
        evaluate_vector_field(lambda x, y: 5.0, 0.5, 0.5)


def test_evaluate_vector_field_nonfinite_raises():
    """分量 Inf → ValueError (line 341)."""
    with pytest.raises(ValueError, match="NaN/Inf"):
        evaluate_vector_field(lambda x, y: (np.inf, 0.0), 0.5, 0.5)
