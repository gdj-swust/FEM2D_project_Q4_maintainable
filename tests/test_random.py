"""随机网格一致性 + SPR边界情况 + 大网格."""
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 1. 随机网格: 求解→应力→误差估计 全链路一致性
# ═══════════════════════════════════════════════════════════════

def test_random_mesh_solves():
    """随机生成 N 个三角形, 应能正常求解."""
    from fem2d import Mesh, solve
    rng = np.random.RandomState(42)
    nodes = rng.rand(30, 2) * 10.0
    from scipy.spatial import Delaunay
    elems = Delaunay(nodes).simplices
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    # 固定左边界 + 右边界 y 向约束 (防刚体转动)
    for n in mesh.nodes_on_edge('x', 'min'): mesh.fix_node(int(n), 'both', 0.0)
    for n in mesh.nodes_on_edge('x', 'max'): mesh.fix_node(int(n), 'y', 0.0)
    right = mesh.nodes_on_edge('x', 'max')
    if len(right) > 0: mesh.add_force(int(right[len(right)//2]), 0., -1000.)
    result = solve(mesh, verbose=False)
    assert result['residual'] < 1e-10
    assert np.all(np.isfinite(result['u']))

def test_random_mesh_stress_recovery():
    """随机网格 SPR 恢复应全为有限值."""
    from fem2d import Mesh, solve, spr_recovery
    rng = np.random.RandomState(123)
    nodes = rng.rand(20, 2) * 5.0
    from scipy.spatial import Delaunay
    elems = Delaunay(nodes).simplices
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in mesh.nodes_on_edge('x', 'min'): mesh.fix_node(int(n), 'both', 0.0)
    for n in mesh.nodes_on_edge('x', 'max'): mesh.fix_node(int(n), 'y', 0.0)
    right = mesh.nodes_on_edge('x', 'max')
    if len(right) > 0: mesh.add_force(int(right[0]), 0., -1000.)
    result = solve(mesh, verbose=False)
    recovered = spr_recovery(mesh, result['stress'])
    assert np.all(np.isfinite(recovered))

def test_random_mesh_z2_estimate():
    """随机网格 Z2 误差估计应合理 (0 < eta < 200)."""
    from fem2d import Mesh, estimate_error, solve
    rng = np.random.RandomState(99)
    nodes = rng.rand(20, 2) * 3.0
    from scipy.spatial import Delaunay
    elems = Delaunay(nodes).simplices
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in mesh.nodes_on_edge('x', 'min'): mesh.fix_node(int(n), 'both', 0.0)
    for n in mesh.nodes_on_edge('x', 'max'): mesh.fix_node(int(n), 'y', 0.0)
    # 多个集中力确保非零应力 — x 向力 (x-max 边只固定了 y, x 自由).
    # 曾加 y 向力到 y 已固定节点: 应力≈0, eta=0 恒真通过
    for n in mesh.nodes_on_edge('x', 'max'): mesh.add_force(int(n), -500., 0.)
    result = solve(mesh, verbose=False)
    z2 = estimate_error(mesh, result)
    assert 0 < z2['eta'] < 200, f"eta 应 >0 (载荷生效), 得到 {z2['eta']:.4f}"

def test_random_mesh_quality_report():
    """随机网格质量报告不应崩溃 — 得分必须反映真实质量差异.

    曾断言 0<=score<=100 (定义域恒真, 评估器恒返回常数也通过, 审计
    2026-08-03). 规则网格得分必须显著高于随机 Delaunay 网格.
    """
    from fem2d import Mesh, evaluate_mesh_quality
    rng = np.random.RandomState(7)
    nodes = rng.rand(100, 2) * 10.0
    from scipy.spatial import Delaunay
    elems = Delaunay(nodes).simplices
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    q_random = evaluate_mesh_quality(mesh)
    regular = Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]]),
        E=210e9, nu=0.3, thickness=0.01)
    q_regular = evaluate_mesh_quality(regular)
    assert q_random['n'] == len(elems)
    assert q_regular['score'] > 90, \
        f"规则网格得分 {q_regular['score']:.1f} 异常 (评估器可能失效)"
    assert q_random['score'] < q_regular['score'], \
        f"随机网格 {q_random['score']:.1f} 不应高于规则网格 {q_regular['score']:.1f}"


# ═══════════════════════════════════════════════════════════════
# 2. SPR 边界情况
# ═══════════════════════════════════════════════════════════════

def test_spr_single_element():
    """单元素 SPR 应退化为自身应力."""
    from fem2d import Mesh, solve, spr_recovery
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for i in range(3): mesh.fix_node(i, 'both', 0.0)
    mesh.add_force(2, 1000., 0.)
    result = solve(mesh, verbose=False)
    recovered = spr_recovery(mesh, result['stress'])
    # 单元素: 恢复值应等于原始应力 — rtol=0.1 曾使任何大偏差都通过
    #
    assert np.allclose(recovered, result['stress'][0], rtol=1e-12, atol=0.0)

def test_spr_boundary_node():
    """边界节点 SPR 应不产生 NaN."""
    from fem2d import Mesh, solve, spr_recovery
    nodes = np.array([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 3]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_force(2, 0., -1000.)
    result = solve(mesh, verbose=False)
    recovered = spr_recovery(mesh, result['stress'])
    assert np.all(np.isfinite(recovered))


def test_negative_coordinates():
    """负坐标应正常处理."""
    from fem2d import Mesh, solve
    nodes = np.array([[-1., -1.], [0., -1.], [-1., 0.], [0., 0.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_force(3, 1000., 0.)
    result = solve(mesh, verbose=False)
    assert np.all(np.isfinite(result['u']))


# ═══════════════════════════════════════════════════════════════
# 4. 载荷积分精确性
# ═══════════════════════════════════════════════════════════════

def test_body_force_total_matches_analytic():
    """体力总合力 = ρg × 面积 × 厚度."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [2., 0.], [0., 3.], [2., 3.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    t = 0.01; total_area = 6.0
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=t)
    for i in range(4): mesh.fix_node(i, 'both', 0.0)
    mesh.body_force = (0., -78000.)
    result = solve(mesh, verbose=False)
    # 总反力应平衡体力: ΣR_y = -ΣF_y = +78000 * total_area * t
    expected_ry = 78000. * total_area * t  # = 4680 N 向上
    ry = result['reaction_vector'].reshape(-1, 2)[:, 1].sum()
    assert abs(ry - expected_ry) < 1.0

def test_pressure_resultant():
    """压力合力 = p × 边长 × 厚度."""
    from fem2d import Mesh, solve
    # 边 (1,3): (2,0)→(2,2), 长度=2m
    nodes = np.array([[0., 0.], [2., 0.], [0., 2.], [2., 2.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_pressure(1, 3, 1e6)  # 右端面 2m, 1MPa
    result = solve(mesh, verbose=False)
    # 压力合力 = 1e6 * 2m * 0.01m = 20000 N ← 反力 ≈ +20000 N →
    expected_rx = 20000.
    rx = result['reaction_vector'].reshape(-1, 2)[:, 0].sum()
    assert abs(rx - expected_rx) < 1.0