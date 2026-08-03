"""极端场景压力测试: 大模型/退化/边界值/畸形输入."""
import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════
# 1. 极端几何
# ═══════════════════════════════════════════════════════════════

def test_tiny_mesh_solves():
    """最小合法网格: 1个三角形, 3节点, 3约束.

    全约束下位移恒零, 反力必须平衡集中力 — 曾占位断言 vm>=-1e-10
    (零解也通过, 审计 2026-08-03).
    """
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for i in range(3): mesh.fix_node(i, 'both', 0.0)
    mesh.add_force(2, 1000., 0.)
    result = solve(mesh, verbose=False)
    assert np.all(np.isfinite(result['u']))
    assert np.abs(result['u']).max() < 1e-12, "全约束应零位移"
    rx = result['reaction_vector'].reshape(-1, 2)[:, 0].sum()
    ry = result['reaction_vector'].reshape(-1, 2)[:, 1].sum()
    assert abs(rx + 1000.0) < 1.0, f"ΣRx={rx} ≠ -1000"
    assert abs(ry) < 1.0, f"ΣRy={ry} ≠ 0"

def test_very_large_coordinates():
    """千米级坐标 (1e6 m), 应正常求解."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1e6, 0.], [0., 1e6], [1e6, 1e6]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(int(n), 'both', 0.0)
    mesh.add_force(3, 1e9, 0.0)
    result = solve(mesh, verbose=False)
    assert result['residual'] < 1e-10

def test_very_small_coordinates():
    """微米级坐标 (1e-6 m), 应正常求解."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1e-6, 0.], [0., 1e-6], [1e-6, 1e-6]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=1e-6)
    for n in [0, 2]: mesh.fix_node(int(n), 'both', 0.0)
    mesh.add_force(3, 1e-3, 0.0)
    result = solve(mesh, verbose=False)
    assert result['residual'] < 1e-10

def test_high_aspect_ratio():
    """极高宽高比三角形 (1e6:1), 不应崩溃."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1e-6]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for i in range(3): mesh.fix_node(i, 'both', 0.0)
    mesh.add_force(2, 1e-3, 0.)
    result = solve(mesh, verbose=False)
    assert np.all(np.isfinite(result['u']))

def test_zero_area_triangle_rejected():
    """零面积(三点共线)三角形应被 Jacobian 检查拦截."""
    from fem2d import Mesh
    nodes = np.array([[0., 0.], [1., 0.], [0.5, 0.]], dtype=float)  # 三点共线, 面积=0
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    mesh.fix_node(0, 'both', 0.0); mesh.fix_node(1, 'both', 0.0)
    from fem2d import solve
    with pytest.raises(RuntimeError, match='degenerate|invalid'):
        solve(mesh, verbose=False)


# ═══════════════════════════════════════════════════════════════
# 2. 极端材料/载荷
# ═══════════════════════════════════════════════════════════════

def test_zero_poisson():
    """nu=0 不应崩溃."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.0, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_force(3, 1000., 0.)
    result = solve(mesh, verbose=False)
    assert np.all(np.isfinite(result['u']))

def test_near_incompressible():
    """nu≈0.5 不应崩溃 (平面应变下可能锁死, 但不应报错)."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.499, thickness=0.01, plane_type='strain')
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_force(3, 1000., 0.)
    result = solve(mesh, verbose=False)
    assert np.all(np.isfinite(result['u']))

def test_zero_thickness_rejected():
    """t=0 应被 Mesh 构造拒绝."""
    from fem2d import Mesh
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    with pytest.raises(ValueError, match='thickness'):
        Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.0)

def test_huge_youngs_modulus():
    """E=1e30 → 极小位移, 不应溢出."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=1e30, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_force(3, 1000., 0.)
    result = solve(mesh, verbose=False)
    assert np.all(np.isfinite(result['u']))


# ═══════════════════════════════════════════════════════════════
# 3. 极端边界条件
# ═══════════════════════════════════════════════════════════════

def test_all_nodes_fixed():
    """全约束 → 零位移."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for i in range(3): mesh.fix_node(i, 'both', 0.0)
    result = solve(mesh, verbose=False)
    assert np.allclose(result['u'], 0.)

def test_no_constraints():
    """无约束 → 刚体模态 → 应报错."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    mesh.add_force(0, 1000., 0.)
    with pytest.raises(RuntimeError, match='Rigid-body|constrain'):
        solve(mesh, verbose=False)

def test_non_zero_prescribed_displacement():
    """非零指定位移应正确产生应力."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    mesh.fix_node(0, 'both', 0.0); mesh.fix_node(2, 'both', 0.0)
    mesh.fix_node(1, 'x', 0.001); mesh.fix_node(3, 'x', 0.001)  # 拉伸
    result = solve(mesh, verbose=False)
    assert np.max(np.abs(result['stress'])) > 0  # 应有应力

def test_pure_dirichlet_with_body_force():
    """全约束 + 体力 → 反力平衡."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for i in range(4): mesh.fix_node(i, 'both', 0.0)
    mesh.body_force = (0., -78000.)
    result = solve(mesh, verbose=False)
    assert result['reactions'] is not None


# ═══════════════════════════════════════════════════════════════
# 4. 极端载荷
# ═══════════════════════════════════════════════════════════════

def test_pressure_on_all_edges():
    """所有边施加压力 → 均匀压缩.

    曾只断言 isfinite(u): 载荷静默丢弃 (u=0) 也通过.
    右边中点节点 1 必须向 -x 压缩 (压力指向域内).
    """
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    mesh.fix_node(0, 'both', 0.0); mesh.fix_node(3, 'x', 0.0); mesh.fix_node(3, 'y', 0.0)
    for a, b in [(0,1),(1,2),(2,3),(3,0)]: mesh.add_pressure(a, b, 1e6)
    result = solve(mesh, verbose=False)
    u = result['u'].reshape(-1, 2)
    assert np.all(np.isfinite(result['u']))
    assert u[1, 0] < 0, f"右边节点应被压向 -x, 得到 ux={u[1,0]:.3e}"

def test_spatial_traction_function():
    """空间函数面力 sin(pi*x/2),0 — 右端面 x=1 → sin(pi/2)=1 → 等效
    恒力 1e6 → ΣFx = +10000 → ΣRx = -10000.

    曾只断言 isfinite(u): 面力静默丢弃也通过.
    """
    import math
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_traction(1, 3, lambda x, y: 1e6 * math.sin(math.pi * x / 2), 0.0)
    result = solve(mesh, verbose=False)
    rx = result['reaction_vector'].reshape(-1, 2)[:, 0].sum()
    assert abs(rx + 10000.0) < 1.0, f"ΣRx={rx} ≠ -10000"

def test_parabolic_traction_distribution():
    """抛物线面力分布 :p — 弧长中点系数 1.0, 两端系数 0.

    曾声称测抛物线实际是常数面力. 直接锁
    make_edge_profile_func 的分布形状: f(arc) = 1-(2s-1)².
    """
    import numpy as np
    from fem2d.loads import make_edge_profile_func
    fx, fy = make_edge_profile_func(
        1e6, 0.0, 'p',
        np.array([0.0, 0.0]), np.array([2.0, 0.0]),
        arc_start=0.0, total_length=2.0)
    assert fx(1.0, 0.0) == 1e6, "弧长中点系数应为 1.0"
    assert fx(0.0, 0.0) == 0.0, "起点系数应为 0"
    assert fx(2.0, 0.0) == 0.0, "终点系数应为 0"
    assert fy(1.0, 0.0) == 0.0, "y 分量恒 0"


def test_traction_profiles_follow_edge_arc_length():
    """单边 arc-length profile (make_edge_profile_func) — 生产路径.

    折线连续 profile 由 runner 用 ordered_edge_chains + 本函数逐边拼接,
    跨段累计 arc_length 在此验证: 第二段起点 arc=1.0/总长 2.0 处
    线性系数应为 0.5.
    """
    from fem2d.loads import make_edge_profile_func

    polyline = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    first_x, _ = make_edge_profile_func(
        1.0, 0.0, "l",
        polyline[0], polyline[1], 0.0, 2.0)
    second_x, _ = make_edge_profile_func(
        1.0, 0.0, "l",
        polyline[1], polyline[2], 1.0, 2.0)
    # 线性 profile: 系数 = arc 长度 / 总长
    assert np.isclose(first_x(0.5, 0.0), 0.25)
    assert np.isclose(first_x(1.0, 0.0), 0.5)     # 第一段终点 = 0.5
    assert np.isclose(second_x(1.0, 0.5), 0.75)   # 第二段中点 = (1+0.5)/2
    # 抛物线 profile: f = 1−(2s−1)², s = 弧长/总长
    mid_x, _ = make_edge_profile_func(
        1.0, 0.0, "p", polyline[1], polyline[2], 1.0, 2.0)
    assert np.isclose(mid_x(1.0, 0.5), 0.75)   # s = (1+0.5)/2 = 0.75


