"""覆盖轮 C1 — BC 施加/线性求解/组装/拓扑核心 缺口行.

bc 与 assembly 直接喂合法/非法 K,F 系统; solver 走真实求解路径触发
平衡检查与特征值兜底; topology_core 直调容器与建表原语.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import fem2d.bc as bc_mod
import fem2d.solver as solver_mod
from fem2d import Mesh
from fem2d.bc import apply_elimination, apply_penalty
from fem2d.loads_schema import _check_load_pair, _check_load_scalar
from fem2d.topology_core import (
    CSRLists,
    EdgeIncidence,
    ElementLocator,
    build_edge_table,
)


def _mesh():
    return Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                elements=np.array([[0, 1, 2], [0, 3, 2]]),
                E=1e6, nu=0.3, thickness=1.0)


# ── bc.py ──────────────────────────────────────────────────────────────────

def test_apply_elimination_cg_block_alias():
    """'cg-block' 别名归一化为 'cg' 并成功求解."""
    K = csr_matrix(np.eye(4))
    F = np.zeros(4)
    u, _ = apply_elimination(
        K, F, np.array([0, 1]), np.array([2, 3]), np.zeros(2),
        linear_solver="cg-block", cg_maxiter=100)
    assert u.shape == (4,)


def test_apply_elimination_pure_dirichlet():
    """自由 DOF 为空 → 全约束直接解 (不调求解器)."""
    K = csr_matrix(np.eye(2))
    u, reactions, info = apply_elimination(
        K, np.zeros(2), np.array([], dtype=int), np.array([0, 1]),
        np.array([0.5, 0.7]), return_info=True)
    assert u.tolist() == [0.5, 0.7]
    assert info["name"] == "direct"


def test_apply_elimination_cg_bad_diagonal():
    """CG 对角非正 (奇异子块) → RuntimeError."""
    K = csr_matrix(np.array([[0., 0.], [0., 1.]]))
    with pytest.raises(RuntimeError, match="positive finite"):
        apply_elimination(K, np.zeros(2), np.array([0]), np.array([1]),
                          np.zeros(1), linear_solver="cg")


def test_apply_elimination_ilu_failure(monkeypatch):
    """ILU 分解失败 → 明确归因的 RuntimeError."""
    def boom(*args, **kwargs):
        raise RuntimeError("ilu failed")
    monkeypatch.setattr(bc_mod, "spilu", boom)
    K = csr_matrix(np.eye(4))
    with pytest.raises(RuntimeError, match="ILU preconditioner"):
        apply_elimination(K, np.zeros(4), np.array([0, 1]),
                          np.array([2, 3]), np.zeros(2),
                          linear_solver="ilu", cg_maxiter=10)


def test_apply_elimination_cg_no_convergence(monkeypatch):
    """CG 返回 info≠0 (不收敛) → RuntimeError 带详情."""
    def fake_cg(*args, **kwargs):
        return np.zeros(1), 42
    monkeypatch.setattr(bc_mod, "cg", fake_cg)
    K = csr_matrix(np.eye(4))
    with pytest.raises(RuntimeError, match="did not converge within 42"):
        apply_elimination(K, np.zeros(4), np.array([0, 1]),
                          np.array([2, 3]), np.zeros(2),
                          linear_solver="cg", cg_maxiter=100)


def test_apply_penalty_nan_prescribed_rejected():
    """给定位移含 NaN → 拒绝 (防静默 NaN 位移)."""
    K = csr_matrix(np.eye(2))
    with pytest.raises(ValueError, match="NaN/Inf"):
        apply_penalty(K, np.zeros(2), np.array([0]), np.array([np.nan]))


# ── solver.py ──────────────────────────────────────────────────────────────

def test_estimate_condition_arpack_fallback(monkeypatch):
    """老 scipy 无 ArpackError → RuntimeError 别名 (特征值失败仍归 SINGULAR)."""
    def boom(*args, **kwargs):
        raise RuntimeError("eigsh failed")
    monkeypatch.setattr(solver_mod, "eigsh", boom)
    try:
        delattr(solver_mod.spsa, "ArpackError")
    except AttributeError:
        pass
    result = solver_mod.estimate_condition(np.eye(4), method="sparse")
    assert result["status"] == "SINGULAR?"


def test_global_balance_check_unbalanced_reactions():
    """支反力与外载不构成自平衡体系 (ΣF≠0) → RuntimeError."""
    m = _mesh()
    from fem2d.solver import _global_balance_check
    K = csr_matrix(np.eye(8))
    with pytest.raises(RuntimeError, match="ΣF"):
        _global_balance_check(
            m, K, np.zeros(8), np.zeros(8),
            reactions=np.array([1e6, 0.0]),
            fixed_dofs=np.array([0, 1]), dirichlet_only=False,
            log=lambda *a, **k: None)


# ── assembly.py ────────────────────────────────────────────────────────────

def test_assemble_nan_stiffness_rejected(monkeypatch):
    """内核返回 NaN 刚度 → 全局对称检查拒绝 (NaN 恒过局部检查)."""
    m = _mesh()
    nldof = 6
    monkeypatch.setattr(
        m.element_kernel, "stiffness_batch",
        lambda mesh, es=None: np.full(
            (es.stop - es.start, nldof, nldof), np.nan))
    from fem2d.assembly import assemble_sparse_vectorized
    with pytest.raises(ValueError, match="NaN/Inf"):
        assemble_sparse_vectorized(m, batch_elements=1)


def test_assemble_batch_none_uses_ne(monkeypatch):
    """batch_elements=None/<=0 → 全量一批 (兼容第三方单调用内核)."""
    m = _mesh()
    nldof = 6
    real_ke = np.zeros((2, nldof, nldof))
    monkeypatch.setattr(
        m.element_kernel, "stiffness_batch",
        lambda mesh, es=None: real_ke)
    from fem2d.assembly import assemble_sparse_vectorized
    K = assemble_sparse_vectorized(m, batch_elements=None)
    assert K.shape == (8, 8)


def test_assemble_signature_probe_failure(monkeypatch):
    """inspect.signature 探测失败 → 按单调用协议处理 (accepts_slice=True)."""
    m = _mesh()
    nldof = 6
    monkeypatch.setattr(
        m.element_kernel, "stiffness_batch",
        lambda mesh, es=None: np.zeros((2, nldof, nldof)))
    monkeypatch.setattr(
        "fem2d.assembly.inspect.signature",
        lambda fn: (_ for _ in ()).throw(ValueError("no signature")))
    from fem2d.assembly import assemble_sparse_vectorized
    K = assemble_sparse_vectorized(m)
    assert K.shape == (8, 8)


def test_assemble_wrong_local_shape_rejected(monkeypatch):
    """内核返回与 (count, nldof, nldof) 不符的刚度形状 → RuntimeError."""
    m = _mesh()
    monkeypatch.setattr(
        m.element_kernel, "stiffness_batch",
        lambda mesh, es=None: np.zeros((2, 2, 2)))
    from fem2d.assembly import assemble_sparse_vectorized
    with pytest.raises(RuntimeError, match="expected"):
        assemble_sparse_vectorized(m)


# ── loads_schema.py ────────────────────────────────────────────────────────

def test_check_load_scalar_1d_array():
    """压力幅值 1 元 ndarray → 取 [0] 分量."""
    assert _check_load_scalar(np.array([1.0]), "p") == (1.0,)


def test_check_load_scalar_1d_array_nan_rejected():
    with pytest.raises(ValueError, match="finite number"):
        _check_load_scalar(np.array([np.nan]), "p")


# ── quality.py ─────────────────────────────────────────────────────────────

def test_quality_compute_1d_determinants(monkeypatch):
    """内核返回 1-D Jacobian 数组 → reshape 为列向量再统计."""
    m = _mesh()
    monkeypatch.setattr(
        m.element_kernel, "jacobian_determinants",
        lambda mesh: np.array([1.0, 1.0]))
    from fem2d.quality import _compute
    result = _compute(m)
    assert result["jacobian_neg"] == 0


# ── topology_core.py ───────────────────────────────────────────────────────

def test_csr_lists_slice_and_repr():
    csr = CSRLists(np.array([0, 2, 3]), np.array([5, 6, 7]))
    assert csr[0] == [5, 6]
    assert csr[0:1] == [[5, 6]]
    assert "CSRLists" in repr(csr)


def test_csr_lists_equality():
    a = CSRLists(np.array([0, 2]), np.array([1, 2]))
    b = CSRLists(np.array([0, 2]), np.array([1, 2]))
    c = CSRLists(np.array([0, 2]), np.array([9, 9]))
    assert a == b
    assert a != c
    assert a != [1, 2]        # NotImplemented 路径
    assert (a == [1, 2]) is False


def test_edge_incidence_getitem_bad_key():
    table = build_edge_table(np.array([[0, 1, 2]]), np.array([[0, 1]]), 4)
    inc = EdgeIncidence(table)
    with pytest.raises(KeyError):
        inc[5]  # 标量键不可解包 → KeyError


def test_edge_incidence_repr_and_table():
    table = build_edge_table(np.array([[0, 1, 2]]), np.array([[0, 1]]), 4)
    inc = EdgeIncidence(table)
    assert inc.table is table
    assert "edges" in repr(inc)


def test_build_edge_table_bad_local_edges_shape():
    with pytest.raises(ValueError, match="local_edges"):
        build_edge_table(np.array([[0, 1, 2]]), np.zeros(3), 4)


def test_build_edge_table_large_node_count():
    """n_nodes > 3e9 → lexsort 排序路径 (防 int64 key 溢出)."""
    table = build_edge_table(
        np.array([[0, 1, 2]]), np.array([[0, 1]]), 4_000_000_000)
    assert table.n_edges >= 1


def test_element_locator_cell_coarsening():
    """单元素网格 + 极小 max_cells → 网格单元尺寸放大循环."""
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    loc = ElementLocator(nodes, np.array([[0, 1, 2]]), max_cells=2)
    assert loc.shape[0] * loc.shape[1] <= 2
