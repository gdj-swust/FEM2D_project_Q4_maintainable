"""性能包: nodal_L2_projection 批量堆叠路径的判别性测试.

背景: CST/Q4R 的 L2 质量阵积分规则逐单元相同 (仅 dA 随单元面积变化),
recovery_shape_matrix 只给 SPR 单点采样, 曾走逐单元 Python 三层循环
(300k 单元 12.8 s)。优化后在恢复规则逐单元一致时堆叠批量计算
(逐位一致), 逐单元变化时回退旧循环。本文件锁定两种行为:
  1. 一致规则 → 批量堆叠结果与手写逐单元参考逐位一致;
  2. 非均匀规则 (第三方内核) → 回退分支与参考逐位一致。
放回旧实现 (无批量分支) 本测试仍然通过 — 它锁定的是路径正确性,
性能加速由 scripts/perf_benchmark.py 的 JSON 对比量化。
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from fem2d.mesh import Mesh
from fem2d.solver import solve
from fem2d.stress import nodal_L2_projection


def _reference_l2(mesh, elem_stress):
    """手写逐单元参考 (旧实现语义)."""
    elem_stress = np.asarray(elem_stress, dtype=float)
    n_nodes, n_comp = mesh.n_nodes, elem_stress.shape[-1]
    kernel = mesh.element_kernel
    conn = mesh.elements
    rows, cols, values = [], [], []
    rhs = np.zeros((n_nodes, n_comp))
    for eid in range(mesh.n_elements):
        conn_e = conn[eid]
        N, dA_e = kernel.recovery_quadrature(mesh, eid)
        local_mass = np.einsum("qi,qj,q->ij", N, N, dA_e)
        if elem_stress.ndim == 2:
            stress_qp = np.broadcast_to(
                elem_stress[eid], (len(dA_e), n_comp))
        else:
            stress_qp = elem_stress[eid]
            if stress_qp.shape[0] == 1 and len(dA_e) != 1:
                stress_qp = np.broadcast_to(
                    stress_qp[0], (len(dA_e), n_comp))
        local_rhs = np.einsum("qi,qc,q->ic", N, stress_qp, dA_e)
        for p, ni in enumerate(conn_e):
            for q, nj in enumerate(conn_e):
                rows.append(int(ni))
                cols.append(int(nj))
                values.append(local_mass[p, q])
            rhs[ni] += local_rhs[p]
    mass = coo_matrix(
        (values, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    factor = splu(mass.tocsc())
    recovered = np.zeros((n_nodes, n_comp))
    for component in range(n_comp):
        recovered[:, component] = factor.solve(rhs[:, component])
    return recovered


def _distorted_cst_mesh(nx=5, ny=4, seed=11):
    rng = np.random.default_rng(seed)
    gx = np.linspace(0.0, 1.0, nx + 1)
    gy = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(gx, gy)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    nodes += rng.normal(0.0, 0.02, nodes.shape)
    tri = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            tri.append([n0, n0 + 1, n0 + nx + 2])
            tri.append([n0, n0 + nx + 2, n0 + nx + 1])
    # 4 角节点全约束
    ncol = nx + 1
    corners = [0, nx, ny * ncol, ny * ncol + nx]
    dofs = [d for n in corners for d in (2 * n, 2 * n + 1)]
    mesh = Mesh(nodes, np.array(tri, dtype=np.int64), E=2.1e11, nu=0.3,
                plane_type="stress", fixed_dofs=dofs, elem_type="CPS3")
    mesh.build_connectivity()
    return mesh


@pytest.mark.parametrize("code", ["CPS3", "CPS4R"])
def test_l2_batched_fallback_bitwise_matches_reference(code):
    """一致恢复规则 → 批量堆叠路径与手写逐单元参考逐位一致.

    CST/Q4R 是单点 SPR 采样 + 多点 L2 质量阵的内核, 走 fallback;
    Q4R 网格复用 Q4 连接, 用同款抖动网格。
    """
    if code == "CPS3":
        mesh = _distorted_cst_mesh()
    else:
        rng = np.random.default_rng(3)
        nx, ny = 5, 4
        gx = np.linspace(0.0, 1.0, nx + 1)
        gy = np.linspace(0.0, 1.0, ny + 1)
        xx, yy = np.meshgrid(gx, gy)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        coords += rng.normal(0.0, 0.02, coords.shape)
        elems = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                elems.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
        ncol = nx + 1
        corners = [0, nx, ny * ncol, ny * ncol + nx]
        dofs = [d for n in corners for d in (2 * n, 2 * n + 1)]
        mesh = Mesh(coords, np.array(elems, dtype=np.int64), E=2.1e11,
                    nu=0.3, plane_type="stress", fixed_dofs=dofs,
                    elem_type=code)
        mesh.build_connectivity()
    res = solve(mesh, verbose=False)
    for src in (res["stress_qp"], res["stress"]):
        fast = nodal_L2_projection(mesh, src)
        ref = _reference_l2(mesh, src)
        assert np.array_equal(fast, ref), (
            f"{code} L2 批量堆叠与逐单元参考不一致: "
            f"max|Δ|={np.max(np.abs(fast - ref)):.3e}")


class _VarNqpKernel:
    """非均匀 nqp 的第三方内核 — 偶数单元截断一半积分点."""

    def __init__(self, inner):
        self._inner = inner
        self.name = "var-nqp"

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def recovery_quadrature(self, mesh, eid):
        N, dA = self._inner.recovery_quadrature(mesh, eid)
        if eid % 2 == 0 and len(dA) > 1:
            return N[:2], dA[:2]
        return N, dA


def test_l2_nonuniform_nqp_falls_back_bitwise():
    """nqp 逐单元变化 → 回退逐单元分支, 与参考逐位一致."""
    mesh = _distorted_cst_mesh(nx=3, ny=3, seed=5)
    mesh.element_kernel = _VarNqpKernel(mesh.element_kernel)
    res = solve(mesh, verbose=False)
    fast = nodal_L2_projection(mesh, res["stress"])
    ref = _reference_l2(mesh, res["stress"])
    assert np.array_equal(fast, ref)


def test_l2_batched_fallback_keeps_orphan_error():
    """批量堆叠路径保留孤立节点错误契约 (曾只在逐单元路径检查)."""
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [5, 5]], dtype=float)
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 2]]), E=210e9,
                nu=0.3, thickness=1.0, elem_type="CPS3")
    mesh.build_connectivity()
    with pytest.raises(ValueError, match="孤立节点"):
        nodal_L2_projection(mesh, np.ones((1, 3)))
