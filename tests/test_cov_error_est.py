"""覆盖轮 C1 — error_est / spr / stress 缺口行 (任务书点名 API 面).

Z2 估计器的兼容路径、显式残差指示器的退化/大坐标防御分支,
SPR 拟合的数值失败兜底, 以及 L2 投影的输入形状契约.
"""
import numpy as np
import pytest

import fem2d.error_est as EE
import fem2d.spr as SPR
import fem2d.stress as STRESS
from fem2d import Mesh


def _mesh():
    return Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                elements=np.array([[0, 1, 2], [0, 3, 2]]),
                E=1e6, nu=0.3, thickness=1.0)


# ── error_est._element_energy_errors 兼容路径 ───────────────────────────────

class _FakeKernel:
    """外部内核: 无批量恢复规则, 提供逐单元 recovery_quadrature."""

    name = "fake"

    @staticmethod
    def recovery_shape_matrix(mesh):
        return None

    @staticmethod
    def recovery_weights(mesh):
        return None

    @staticmethod
    def recovery_quadrature(mesh, eid):
        return np.eye(mesh.elements.shape[1]), np.ones(3)


def test_element_energy_errors_qp_shape_mismatch():
    """兼容路径: stress_qp 采样数 ≠ 恢复规则采样数 → ValueError."""
    mesh = _mesh()
    mesh.element_kernel = _FakeKernel()
    D_inv = np.eye(3)
    stress = np.ones((2, 3, 3))
    stress_qp = np.ones((2, 3, 2))   # 每单元 3 采样 vs 恢复规则 3 列 → 错形状
    s_star = np.ones((4, 3))         # 节点恢复场 (n_nodes, 3)
    with pytest.raises(ValueError, match="shape"):
        EE._element_energy_errors(
            mesh, "SPR", stress, stress_qp, s_star, D_inv)


# ── estimate 报告分支 (eta ∈ (10, 20)) ──────────────────────────────────────

def test_estimate_eta_note_branch(monkeypatch, capsys):
    """eta 落在 (10,20) → '[NOTE] 建议局部加密' 打印分支."""
    mesh = _mesh()
    result = {"stress": np.ones((2, 3))}
    monkeypatch.setattr(
        EE, "_element_energy_errors",
        lambda *a, **k: (np.array([1.0, 0.0]), np.array([44.4, 0.0]),
                         1.0, True, 1.0))
    out = EE.estimate(mesh, result, verbose=True)
    assert 10 < out["eta"] < 20
    assert "NOTE" in capsys.readouterr().out


# ── element_refinement_indicator 防御分支 ──────────────────────────────────

def test_indicator_degenerate_element_body_force():
    """退化单元 (面积 0) + 体力 → h_K/A_K 为 0 → 跳过该项."""
    mesh = Mesh(nodes=np.array([[0., 0.], [1., 0.], [2., 0.]]),
                elements=np.array([[0, 1, 2]]), E=1e6, nu=0.3)
    mesh.body_force = (1.0, 1.0)
    eta = EE.element_refinement_indicator(mesh, {"stress": np.zeros((1, 3))})
    assert eta.shape == (1,)


def test_indicator_large_coord_degenerate_edges():
    """加载边在原点而其余节点在 1e100: 全局 ULP 判据触发 → 边项跳过.

    局部 ULP (按边坐标) 允许法向计算成功, 全局 ULP (按全网格坐标)
    把短边判为退化 → loaded 与自由边两处判据各命中一次.
    """
    nodes = np.array([[0., 0.], [1., 0.], [1e100, 1e100]])
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 2]]),
                E=1e6, nu=0.3)
    mesh.add_traction(0, 1, 1e6, 0.0)   # 加载边 (0 附近, 法向可算)
    mesh.fix_node(0, "x")               # 部分约束 → 自由边残差路径
    eta = EE.element_refinement_indicator(mesh, {"stress": np.zeros((1, 3))})
    assert np.all(np.isfinite(eta))


def test_indicator_boundary_edge_not_in_table():
    """boundary_edges 含 edge_to_elems 缺失的边 → 跳过 (防御)."""
    mesh = _mesh()
    mesh.boundary_edges = [(99, 100)]
    eta = EE.element_refinement_indicator(mesh, {"stress": np.zeros((2, 3))})
    assert np.all(np.isfinite(eta))


def test_indicator_boundary_edge_empty_owners():
    """boundary 边在表中但无 owner 单元 → 跳过 (防御)."""
    mesh = _mesh()
    mesh.boundary_edges = [(99, 100)]
    mesh.edge_to_elems = {(99, 100): []}
    eta = EE.element_refinement_indicator(mesh, {"stress": np.zeros((2, 3))})
    assert np.all(np.isfinite(eta))


def test_indicator_zero_length_boundary_edge():
    """零长边界边 → boundary_outward_normal 抛异常被吞 (防御)."""
    mesh = Mesh(nodes=np.array([[0., 0.], [0., 0.], [1., 0.]]),
                elements=np.array([[0, 1, 2]]), E=1e6, nu=0.3)
    eta = EE.element_refinement_indicator(mesh, {"stress": np.zeros((1, 3))})
    assert np.all(np.isfinite(eta))


# ── spr.py ─────────────────────────────────────────────────────────────────

def test_fit_node_block_all_empty():
    """节点块内无采样点 → 全部返回为 empty (不崩溃)."""
    mesh = _mesh()
    recovered = np.zeros((3, 3))
    empty = SPR._fit_node_block(
        mesh, np.zeros((2, 1, 2)), np.zeros((2, 1, 3)),
        np.zeros(4, dtype=np.int64), np.zeros(0, dtype=np.int64),
        0, 3, recovered)
    assert empty.tolist() == [0, 1, 2]


def test_spr_lstsq_failure_falls_back(monkeypatch):
    """patch 拟合正规方程奇异 → 未解析节点归集."""
    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")
    monkeypatch.setattr(np.linalg, "solve", boom)
    mesh = _mesh()
    recovered = SPR.spr_recovery(mesh, np.ones((2, 3)))
    assert recovered.shape == (4, 3)


def test_spr_nonfinite_fit_marked_unresolved(monkeypatch):
    """拟合系数含 NaN → 标记未解析并清零."""
    def nan_coeffs(normal, rhs):
        return np.full(rhs.shape, np.nan)
    monkeypatch.setattr(np.linalg, "solve", nan_coeffs)
    mesh = _mesh()
    recovered = SPR.spr_recovery(mesh, np.ones((2, 3)))
    assert recovered.shape == (4, 3)


def test_fit_nodes_exact_isolated_node_zeroed():
    """孤立节点 (无单元 patch) → 恢复值置 0."""
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [5., 5.]])
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 3, 2]]),
                E=1e6, nu=0.3)
    mesh.build_connectivity()
    recovered = np.zeros((5, 3))
    SPR._fit_nodes_exact(
        mesh, np.zeros((2, 1, 2)), np.ones((2, 1, 3)),
        np.array([4]), recovered)
    assert recovered[4].tolist() == [0.0, 0.0, 0.0]


def test_fit_nodes_exact_lstsq_fallback_mean(monkeypatch):
    """lstsq 秩不足 → 平均; 平均解算异常 → 均值 (数值兜底)."""
    def bad_rank(design, values, rcond=None):
        return np.zeros(3), None, 1, np.ones(3)
    monkeypatch.setattr(np.linalg, "lstsq", bad_rank)

    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("avg failed")
    monkeypatch.setattr("numpy.average", boom)
    mesh = _mesh()
    mesh.build_connectivity()
    recovered = np.zeros((4, 3))
    SPR._fit_nodes_exact(
        mesh, np.zeros((2, 1, 2)), np.ones((2, 1, 3)),
        np.array([0]), recovered)
    assert np.all(np.isfinite(recovered[0]))


# ── stress.py ──────────────────────────────────────────────────────────────

def test_l2_projection_wrong_sample_count():
    """stress_qp 采样数 ≠ 内核积分规则 → ValueError (契约)."""
    mesh = _mesh()
    with pytest.raises(ValueError, match="stress samples"):
        STRESS.nodal_L2_projection(mesh, np.zeros((2, 5, 3)))


def test_l2_projection_wrong_element_count():
    mesh = _mesh()
    with pytest.raises(ValueError, match="first dimension"):
        STRESS.nodal_L2_projection(mesh, np.zeros((5, 3)))


def test_stress_at_point_coincident_nodes_skipped():
    """浮点重合节点边 → 退化边跳过, 仍返回有效值."""
    nodes = np.array([[0., 0.], [1e-200, 0.], [1., 0.], [0., 1.]])
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 3], [1, 2, 3]]),
                E=1e6, nu=0.3)
    result = {"stress": np.ones((2, 3))}
    s = STRESS.stress_at_point(mesh, result, 0.5, 0.0, mode="sides")
    assert s is not None


def test_stress_at_point_interior_no_neighbor():
    """查询点在单元内部 (非边) → 无邻居 → 返回单元代表应力."""
    mesh = _mesh()
    result = {"stress": np.ones((2, 3))}
    s = STRESS.stress_at_point(mesh, result, 0.5, 0.25, mode="sides")
    assert s.shape == (3,)
