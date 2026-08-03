"""Functional-form BCs: fix_nodes_func, callable traction/pressure/body.

Every check compares against an analytic expectation at machine precision.
Reactions are summed by global DOF index (fixed_dofs is sorted, so
positional odd/even indexing does NOT select x/y components reliably).
"""
import numpy as np

from fem2d import Mesh, solve

E, NU, T = 210e9, 0.3, 0.01


def rect_mesh(nx=8, ny=4, L=2.0, H=1.0):
    x = np.linspace(0.0, L, nx + 1)
    y = np.linspace(0.0, H, ny + 1)
    nodes = np.array([[xi, yi] for yi in y for xi in x])
    elems = []
    for i in range(nx):
        for j in range(ny):
            a = j * (nx + 1) + i
            elems.append([a, a + 1, a + nx + 2, a + nx + 1])
    return nodes, np.array(elems, dtype=int)


def _mesh():
    nodes, elems = rect_mesh()
    return Mesh(nodes=nodes, elements=elems, E=E, nu=NU, thickness=T,
                plane_type="stress", elem_type="CPS4")


def _reaction_vector(result, mesh):
    full = np.zeros(mesh.n_dof)
    full[mesh.fixed_dofs] = result["reactions"]
    return full


def test_fix_nodes_func_recovers_linear_field():
    """函数位移 BC: 线性场施加于全部边界 → 域内精确恢复 (patch 等价)."""
    m = _mesh()
    nodes = m.nodes
    bdy = [nid for nid, (xi, yi) in enumerate(nodes)
           if (abs(xi - 0.0) < 1e-12 or abs(xi - 2.0) < 1e-12
               or abs(yi - 0.0) < 1e-12 or abs(yi - 1.0) < 1e-12)]
    ex, ey, gxy = 1e-4, -5e-5, 2e-5
    m.fix_nodes_func(bdy, lambda x, y: (ex * x + 0.5 * gxy * y,
                                        0.5 * gxy * x + ey * y))
    r = solve(m, verbose=False)
    u2 = r["u"].reshape(-1, 2)
    err = max(
        abs(u2[nid, 0] - (ex * nodes[nid, 0] + 0.5 * gxy * nodes[nid, 1]))
        + abs(u2[nid, 1] - (0.5 * gxy * nodes[nid, 0] + ey * nodes[nid, 1]))
        for nid in range(len(nodes)))
    assert err < 1e-14


def test_fix_nodes_func_scalar_matches_manual():
    """标量常数函数 == 逐节点 fix_node."""
    m1 = _mesh()
    m2 = _mesh()
    bdy = [nid for nid, (xi, yi) in enumerate(m1.nodes)
           if (abs(xi - 0.0) < 1e-12 or abs(xi - 2.0) < 1e-12
               or abs(yi - 0.0) < 1e-12 or abs(yi - 1.0) < 1e-12)]
    for nid in bdy:
        m1.fix_node(nid, "both", 1e-4)
    m2.fix_nodes_func(bdy, 1e-4)
    r1 = solve(m1, verbose=False)
    r2 = solve(m2, verbose=False)
    assert np.max(np.abs(r1["u"] - r2["u"])) < 1e-14


def _fix_left_bottom(m):
    for nid, (xi, _) in enumerate(m.nodes):
        if abs(xi - 0.0) < 1e-12:
            m.fix_node(nid, "both", 0.0)
    for nid, (_, yi) in enumerate(m.nodes):
        if abs(yi - 0.0) < 1e-12:
            m.fix_node(nid, "y", 0.0)


def _right_edge(m):
    return [nid for nid, (xi, _) in enumerate(m.nodes)
            if abs(xi - 2.0) < 1e-12]


def test_callable_traction_linear_distribution():
    """callable 面力 (线性分布): 合力 = t·∫t(y)dy 机器精度."""
    m = _mesh()
    right = _right_edge(m)
    for k in range(len(right) - 1):
        m.add_traction(right[k], right[k + 1], lambda x, y: 1e6 * y, 0.0)
    _fix_left_bottom(m)
    r = solve(m, verbose=False)
    R = _reaction_vector(r, m)
    theory = T * 1e6 * 1.0 / 2.0
    assert abs(R[0::2].sum() + theory) / theory < 1e-12


def test_callable_pressure_hydrostatic():
    """callable 压力 (线性水压): 水平合力 = t·∫p(y)dy 机器精度."""
    m = _mesh()
    right = _right_edge(m)
    p0 = 2e6
    for k in range(len(right) - 1):
        m.add_pressure(right[k], right[k + 1], lambda x, y: p0 * y / 1.0)
    _fix_left_bottom(m)
    r = solve(m, verbose=False)
    R = _reaction_vector(r, m)
    theory = T * p0 * 1.0 / 2.0
    # 正压 = 压缩 (指向域内), 右边界外法向 +x → 外力 -x, 反力 +x
    assert abs(R[0::2].sum() - theory) / theory < 1e-12


def test_callable_body_force():
    """callable 体力 (线性): 总反力 = t·∫∫b dA 机器精度."""
    m = _mesh()
    m.body_force = (0.0, lambda x, y: -78000.0 * (1.0 + x / 2.0))
    _fix_left_bottom(m)
    r = solve(m, verbose=False)
    R = _reaction_vector(r, m)
    theory = T * 78000.0 * 3.0       # ∫∫(1+x/2) dA over [0,2]×[0,1] = 3
    assert abs(R[1::2].sum() - theory) / theory < 1e-12


def test_mixed_tuple_body_force():
    """混合元组 (callable, float) 体力."""
    m = _mesh()
    m.body_force = (lambda x, y: 1000.0 * x, -500.0)
    _fix_left_bottom(m)
    r = solve(m, verbose=False)
    R = _reaction_vector(r, m)
    assert abs(R[0::2].sum() + T * 1000.0 * 2.0) < 1e-10   # ∫∫x dA = 2
    assert abs(R[1::2].sum() - T * 500.0 * 2.0) < 1e-10    # ∫∫1 dA = 2
